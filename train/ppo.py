"""Masked PPO with league opponents, starting from models/bc_v1.pt (handoff B12, B13, W6).

    python -m train.ppo --device cpu --minutes 20 --run cpu20
    python -m train.ppo --device cuda --hours 9 --run thu --workers 14
    python -m train.ppo --run thu --resume runs/thu/last.pt --hours 2

256 envs x 128 steps per update, discount 0.997, GAE 0.95, clip 0.2, entropy 0.01,
lr 2.5e-4 (linear to 0), 4 epochs, minibatch 4,096, value loss 0.5 with clipping,
grad-norm clip 0.5, approx-KL early stop at 0.03. The first 20 updates train only the value
head (BC never trained it). Reward shaping falls linearly from 1.0 to 0.1 over the planned
updates (planned = time budget / measured update time, fixed after 5 updates).

Writes runs/<run>/log.csv (one row per update; win rates per opponent type every
--log-every updates), runs/<run>/last.pt every --log-every updates (for --resume) and
runs/<run>/snap_<update>.pt every --snap-every updates (bot.model format: LearnedPolicy and
the gauntlet load it; also added to the league). Ctrl+C or the time limit saves last.pt and a
final snapshot.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.distributions import Categorical

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from bot import encode as enc  # noqa: E402
from bot import model  # noqa: E402
from bot.env import VecEnv  # noqa: E402
from train.league import KINDS, League  # noqa: E402

LOG_FIELDS = [
    "update", "wall_s", "decisions_per_s", "policy_loss", "value_loss", "entropy", "approx_kl",
    "kl_stop", "clip_frac", "lr", "shape", "mean_ep_reward", "episodes", "warmup", "planned_updates",
] + [f"{p}{k}" for k in KINDS for p in ("wr_", "n_")] + [f"{p}{k}" for k in KINDS for p in ("wrd_", "nd_")]


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--run", default="ppo")
    ap.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    ap.add_argument("--hours", type=float, default=1.0)
    ap.add_argument("--minutes", type=float, default=None, help="overrides --hours")
    ap.add_argument("--init", default="models/bc_v1.pt")
    ap.add_argument("--resume", default=None, help="runs/<run>/last.pt")
    ap.add_argument("--deck", default=None, help="learner deck JSON (default bot/decks/default.json)")
    ap.add_argument("--deck-mix", type=float, default=0.4, help="share of episodes the learner plays --deck")
    ap.add_argument("--envs", type=int, default=256)
    ap.add_argument("--steps", type=int, default=128)
    ap.add_argument("--workers", type=int, default=None, help="env worker processes (default cpu_count - 2)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--gamma", type=float, default=0.997)
    ap.add_argument("--gae", type=float, default=0.95)
    ap.add_argument("--clip", type=float, default=0.2)
    ap.add_argument("--ent", type=float, default=0.01)
    ap.add_argument("--vf", type=float, default=0.5)
    ap.add_argument("--lr", type=float, default=2.5e-4)
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--minibatch", type=int, default=4096)
    ap.add_argument("--max-grad-norm", type=float, default=0.5)
    ap.add_argument("--target-kl", type=float, default=0.03)
    ap.add_argument("--warmup", type=int, default=20, help="value-head-only updates")
    ap.add_argument("--shape-start", type=float, default=1.0)
    ap.add_argument("--shape-end", type=float, default=0.1)
    ap.add_argument("--snap-every", type=int, default=50)
    ap.add_argument("--log-every", type=int, default=10)
    ap.add_argument("--threads", type=int, default=4, help="torch CPU threads in the main process")
    return ap.parse_args(argv)


def _atomic_torch_save(obj, path: Path) -> None:
    tmp = path.with_name(path.name + ".tmp")
    torch.save(obj, tmp)
    os.replace(tmp, path)


def save_snapshot(net, path: Path) -> None:
    tmp = path.with_name(path.name + ".tmp")
    net.save(tmp)
    os.replace(tmp, path)


def _resolve(p: str) -> Path:
    q = Path(p)
    return q if q.is_absolute() else (Path.cwd() / q)


class Logger:
    def __init__(self, path: Path):
        self.path = path
        new = not path.exists() or path.stat().st_size == 0
        self.f = path.open("a", newline="", encoding="utf-8")
        self.w = csv.DictWriter(self.f, fieldnames=LOG_FIELDS, extrasaction="ignore")
        if new:
            self.w.writeheader()
            self.f.flush()

    def row(self, d: dict) -> None:
        self.w.writerow({k: (f"{v:.6g}" if isinstance(v, float) else v) for k, v in d.items()})
        self.f.flush()

    def close(self):
        self.f.close()


def win_rates(episodes: list[dict], league: League) -> dict:
    out = {}
    for prefix, eps in (("", episodes), ("d", [e for e in episodes if e["deck_kind"] == "default"])):
        by = defaultdict(lambda: [0, 0])
        for e in eps:
            k = league.kind_of(e["opponent"])
            by[k][0] += e["result"] == "win"
            by[k][1] += 1
        for k in KINDS:
            w, n = by.get(k, (0, 0))
            out[f"wr{prefix}_{k}"] = w / n if n else ""
            out[f"n{prefix}_{k}"] = n
    return out


def compute_gae(rew, done, val, last_val, gamma, lam):
    T = rew.shape[0]
    adv = np.zeros_like(rew)
    last = np.zeros_like(last_val)
    for t in reversed(range(T)):
        nxt = last_val if t == T - 1 else val[t + 1]
        nonterm = 1.0 - done[t]
        delta = rew[t] + gamma * nxt * nonterm - val[t]
        last = delta + gamma * lam * nonterm * last
        adv[t] = last
    return adv, adv + val


def main(argv=None) -> dict:
    args = parse_args(argv)
    budget = args.minutes * 60 if args.minutes is not None else args.hours * 3600
    device = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    if device == "cpu":
        torch.set_num_threads(max(1, args.threads))
    torch.manual_seed(args.seed)
    np_rng = np.random.default_rng(args.seed)

    run_dir = REPO / "runs" / args.run
    run_dir.mkdir(parents=True, exist_ok=True)
    deck = json.loads(_resolve(args.deck).read_text(encoding="utf-8")) if args.deck else None

    ckpt = None
    if args.resume:
        ckpt = torch.load(_resolve(args.resume), map_location="cpu", weights_only=False)
        net = model.Net(ckpt.get("config"))
        net.load_state_dict(ckpt["model"])
        league = League.from_state(ckpt["league"])
        start_update = int(ckpt["update"])
    else:
        net = model.load(_resolve(args.init))
        league = League(run_dir, bc_path=REPO / "models" / "bc_v1.pt")
        start_update = 0
    net.to(device)
    opt = torch.optim.Adam(net.parameters(), lr=args.lr, eps=1e-5)
    vopt = torch.optim.Adam(net.value.parameters(), lr=args.lr, eps=1e-5)
    if ckpt is not None:
        opt.load_state_dict(ckpt["optimizer"])
        vopt.load_state_dict(ckpt["value_optimizer"])

    T, N = args.steps, args.envs
    env = VecEnv(n=N, workers=args.workers, seed=args.seed + start_update * 7919, opponent_sampler=league,
                 learner_deck=deck, deck_mix=args.deck_mix, shape=args.shape_start)
    if league.snapshots:
        env.set_snapshots(league.snapshots)
    log = Logger(run_dir / "log.csv")
    print(f"run {run_dir} | device {device} | envs {N} x steps {T} | workers {env.workers} | "
          f"budget {budget / 60:.1f} min | start update {start_update} | snapshots {len(league.snapshots)}", flush=True)

    buf = {k: np.zeros((T, N) + enc.SHAPES[k], dtype=np.float32) for k in enc.KEYS}
    act_b = np.zeros((T, N), dtype=np.int64)
    logp_b = np.zeros((T, N), dtype=np.float32)
    val_b = np.zeros((T, N), dtype=np.float32)
    rew_b = np.zeros((T, N), dtype=np.float32)
    done_b = np.zeros((T, N), dtype=np.float32)
    ep_ret = np.zeros(N, dtype=np.float64)

    t_start = time.perf_counter()
    update = start_update
    planned = None
    window: list[dict] = []
    last_snap = None
    status = "time limit"
    obs = env.reset()
    t_loop = time.perf_counter()  # update-time estimate excludes worker start-up and the first reset

    def save_last():
        _atomic_torch_save({
            "model": net.state_dict(), "config": net.config, "FEAT_DIM": model.FEAT_DIM, "N_ACTIONS": model.N_ACTIONS,
            "optimizer": opt.state_dict(), "value_optimizer": vopt.state_dict(), "update": update,
            "league": league.state(), "args": vars(args),
        }, run_dir / "last.pt")

    def snapshot():
        nonlocal last_snap
        p = run_dir / f"snap_{update}.pt"
        save_snapshot(net, p)
        league.add_snapshot(p)
        env.set_snapshots(league.snapshots)
        last_snap = p

    try:
        while time.perf_counter() - t_start < budget:
            tu = time.perf_counter()
            session = update - start_update
            frac = min(1.0, update / planned) if planned else 0.0
            lr = args.lr * max(0.0, 1.0 - frac)
            shape = args.shape_start + (args.shape_end - args.shape_start) * frac
            env.set_shape(shape)
            for g in opt.param_groups + vopt.param_groups:
                g["lr"] = lr
            warm = update < args.warmup

            # --- rollout ---
            net.eval()
            finished = []
            with torch.no_grad():
                for t in range(T):
                    for k in enc.KEYS:
                        buf[k][t] = obs[k]
                    logits, v = net(model.to_torch(obs, device))
                    dist = Categorical(logits=logits)
                    a = dist.sample()
                    act_b[t] = a.cpu().numpy()
                    logp_b[t] = dist.log_prob(a).cpu().numpy()
                    val_b[t] = v.cpu().numpy()
                    obs, r, d, infos = env.step(act_b[t])
                    rew_b[t] = r
                    done_b[t] = d
                    ep_ret += r
                    for info in infos:
                        info["return"] = float(ep_ret[info["env"]])
                        ep_ret[info["env"]] = 0.0
                        league.counts[league.kind_of(info["opponent"])] += 1
                        finished.append(info)
                last_val = net(model.to_torch(obs, device))[1].cpu().numpy()
            t_roll = time.perf_counter() - tu
            adv, ret = compute_gae(rew_b, done_b, val_b, last_val, args.gamma, args.gae)

            # --- update ---
            B = T * N
            flat = {k: buf[k].reshape((B,) + enc.SHAPES[k]) for k in enc.KEYS}
            f_act, f_logp, f_val = act_b.reshape(B), logp_b.reshape(B), val_b.reshape(B)
            f_adv, f_ret = adv.reshape(B), ret.reshape(B)
            net.train()
            stats = defaultdict(list)
            kl_stop = 0
            mb = min(args.minibatch, B)
            for epoch in range(args.epochs):
                perm = np_rng.permutation(B)
                for s in range(0, B - mb + 1, mb):
                    i = perm[s:s + mb]
                    x = {k: torch.from_numpy(flat[k][i]).to(device) for k in enc.KEYS}
                    a = torch.from_numpy(f_act[i]).to(device)
                    old_logp = torch.from_numpy(f_logp[i]).to(device)
                    old_v = torch.from_numpy(f_val[i]).to(device)
                    ret_t = torch.from_numpy(f_ret[i]).to(device)
                    adv_t = torch.from_numpy(f_adv[i]).to(device)
                    adv_t = (adv_t - adv_t.mean()) / (adv_t.std() + 1e-8)

                    logits, v = net(x)
                    dist = Categorical(logits=logits)
                    logp = dist.log_prob(a)
                    log_ratio = logp - old_logp
                    ratio = log_ratio.exp()
                    pg = torch.max(-adv_t * ratio, -adv_t * ratio.clamp(1 - args.clip, 1 + args.clip)).mean()
                    v_clip = old_v + (v - old_v).clamp(-args.clip, args.clip)
                    vloss = 0.5 * torch.max((v - ret_t) ** 2, (v_clip - ret_t) ** 2).mean()
                    ent = dist.entropy().mean()
                    with torch.no_grad():
                        approx_kl = float(((ratio - 1) - log_ratio).mean())
                        clipf = float(((ratio - 1).abs() > args.clip).float().mean())

                    if warm:
                        loss = args.vf * vloss
                        vopt.zero_grad(set_to_none=True)
                        opt.zero_grad(set_to_none=True)
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(net.value.parameters(), args.max_grad_norm)
                        vopt.step()
                    else:
                        loss = pg + args.vf * vloss - args.ent * ent
                        opt.zero_grad(set_to_none=True)
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(net.parameters(), args.max_grad_norm)
                        opt.step()
                    for k, val in (("pg", float(pg.detach())), ("vl", float(vloss.detach())), ("ent", float(ent.detach())),
                                   ("kl", approx_kl), ("clip", clipf)):
                        stats[k].append(val)
                    if not warm and approx_kl > args.target_kl:
                        kl_stop = 1
                        break
                if kl_stop:
                    break

            update += 1
            session += 1
            wall = time.perf_counter() - t_start
            if session in (1, 5):
                per = (time.perf_counter() - t_loop) / session
                planned = max(update, start_update + int((budget - (t_loop - t_start)) / per))
            window.extend(finished)
            row = {
                "update": update, "wall_s": round(wall, 1), "decisions_per_s": round(B / (time.perf_counter() - tu), 1),
                "policy_loss": float(np.mean(stats["pg"])), "value_loss": float(np.mean(stats["vl"])),
                "entropy": float(np.mean(stats["ent"])), "approx_kl": float(np.mean(stats["kl"])),
                "kl_stop": kl_stop, "clip_frac": float(np.mean(stats["clip"])), "lr": lr, "shape": shape,
                "mean_ep_reward": float(np.mean([e["return"] for e in finished])) if finished else "",
                "episodes": len(finished), "warmup": int(warm), "planned_updates": planned or "",
            }
            line = (f"u{update:4d} {wall / 60:6.1f}m {row['decisions_per_s']:7.0f}/s roll {t_roll:4.1f}s "
                    f"pg {row['policy_loss']:+.4f} vl {row['value_loss']:.4f} ent {row['entropy']:.3f} "
                    f"kl {row['approx_kl']:.4f}{'!' if kl_stop else ''} lr {lr:.2e} shape {shape:.2f} "
                    f"eps {len(finished)}{' warm' if warm else ''}")
            if update % args.log_every == 0:
                wr = win_rates(window, league)
                row.update(wr)
                line += " | win " + " ".join(
                    f"{k} {wr['wr_' + k]:.0%}({wr['n_' + k]})" if wr["n_" + k] else f"{k} -" for k in KINDS)
                line += " | default deck " + " ".join(
                    f"{k} {wr['wrd_' + k]:.0%}({wr['nd_' + k]})" for k in KINDS if wr["nd_" + k])
                window = []
                save_last()
            log.row(row)
            print(line, flush=True)
            if update % args.snap_every == 0:
                snapshot()
    except KeyboardInterrupt:
        status = "interrupted"
    finally:
        try:
            save_last()
            if update > start_update and (last_snap is None or last_snap.name != f"snap_{update}.pt"):
                snapshot()
        finally:
            env.close()
            log.close()
    summary = {"status": status, "updates": update, "session_updates": update - start_update,
               "minutes": round((time.perf_counter() - t_start) / 60, 2), "snapshots": list(league.snapshots),
               "league_counts": dict(league.counts), "run_dir": str(run_dir)}
    print(f"done ({status}): {summary['session_updates']} updates this session (total {update}) in "
          f"{summary['minutes']} min | snapshots {len(league.snapshots)} | episodes by opponent {dict(league.counts)}",
          flush=True)
    return summary


if __name__ == "__main__":
    main()

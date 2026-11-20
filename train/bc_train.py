"""Behaviour cloning on recorded scripted decisions (handoff B9).

    python -m train.bc_train                                  # data/bc -> models/bc_v1.pt
    python -m train.bc_train --epochs 10 --wait-weight 0.5    # the fallbacks if non-wait acc < 60%

Cross-entropy on Net's masked logits, AdamW lr 3e-4, batch 1,024, 5 epochs; 5% of the shard
FILES held out (at least one). Shards stay float16 in RAM and are converted per batch. The
best epoch by held-out non-wait accuracy is saved with FEAT_DIM, N_ACTIONS, config and a
"trained_on" dict (data meta.json + final metrics).
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from bot import encode as enc  # noqa: E402
from bot import model  # noqa: E402


def load_shards(files: list[Path]) -> dict[str, np.ndarray]:
    parts: dict[str, list] = {k: [] for k in enc.KEYS + ("label",)}
    for f in files:
        with np.load(f) as z:
            for k in parts:
                parts[k].append(z[k])
    return {k: np.concatenate(v) for k, v in parts.items()}


def drop_illegal(d: dict[str, np.ndarray]) -> tuple[dict[str, np.ndarray], int]:
    lab = d["label"].astype(np.int64)
    ok = d["mask"][np.arange(len(lab)), lab] > 0
    bad = int((~ok).sum())
    if bad:
        d = {k: v[ok] for k, v in d.items()}
    return d, bad


def batch(d: dict[str, np.ndarray], idx: np.ndarray, device) -> tuple[dict, torch.Tensor]:
    x = {k: torch.from_numpy(d[k][idx].astype(np.float32)).to(device) for k in enc.KEYS}
    y = torch.from_numpy(d["label"][idx].astype(np.int64)).to(device)
    return x, y


def _weights(y: torch.Tensor, wait_weight: float) -> torch.Tensor:
    w = torch.ones_like(y, dtype=torch.float32)
    w[y == 0] = wait_weight
    return w


@torch.no_grad()
def evaluate(net, d, device, bs=4096, wait_weight=1.0) -> dict:
    net.eval()
    n = len(d["label"])
    loss_sum = wsum = 0.0
    correct = wait_c = wait_n = nw_c = nw_n = top3_c = 0
    for s in range(0, n, bs):
        idx = np.arange(s, min(s + bs, n))
        x, y = batch(d, idx, device)
        logits, _ = net(x)
        w = _weights(y, wait_weight)
        loss_sum += float((F.cross_entropy(logits, y, reduction="none") * w).sum())
        wsum += float(w.sum())
        pred = logits.argmax(-1)
        hit = pred == y
        wait = y == 0
        correct += int(hit.sum())
        wait_c += int(hit[wait].sum())
        wait_n += int(wait.sum())
        nw_c += int(hit[~wait].sum())
        nw_n += int((~wait).sum())
        top3 = (logits.topk(3, dim=-1).indices == y[:, None]).any(-1)
        top3_c += int(top3[~wait].sum())
    return {
        "loss": loss_sum / max(wsum, 1e-9),
        "acc": correct / max(n, 1),
        "acc_wait": wait_c / max(wait_n, 1),
        "acc_nonwait": nw_c / max(nw_n, 1),
        "top3_nonwait": top3_c / max(nw_n, 1),
        "n": n,
        "n_nonwait": nw_n,
    }


def save_checkpoint(net, path: Path, trained_on: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "state_dict": net.state_dict(),
        "FEAT_DIM": model.FEAT_DIM,
        "N_ACTIONS": model.N_ACTIONS,
        "config": net.config,
        "trained_on": trained_on,
    }, path)


def main(argv=None) -> dict:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--data", default=str(REPO / "data" / "bc"))
    ap.add_argument("--out", default=str(REPO / "models" / "bc_v1.pt"))
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch", type=int, default=1024)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--holdout", type=float, default=0.05, help="share of shard files held out")
    ap.add_argument("--wait-weight", type=float, default=1.0, help="loss weight on wait labels")
    ap.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--threads", type=int, default=None, help="torch CPU threads (default: torch's choice)")
    args = ap.parse_args(argv)

    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else ("cpu" if args.device == "auto" else args.device)
    if args.threads:
        torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)

    data_dir = Path(args.data)
    files = sorted(data_dir.glob("shard_*.npz"))
    if len(files) < 2:
        raise SystemExit(f"need at least 2 shard files in {data_dir}, found {len(files)}")
    shuffled = files[:]
    rng.shuffle(shuffled)
    n_hold = max(1, round(len(files) * args.holdout))
    hold_files, train_files = sorted(shuffled[:n_hold]), sorted(shuffled[n_hold:])
    meta_path = data_dir / "meta.json"
    data_meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}

    t0 = time.perf_counter()
    train, bad_t = drop_illegal(load_shards(train_files))
    hold, bad_h = drop_illegal(load_shards(hold_files))
    n = len(train["label"])
    print(f"device {device} | train {n:,} decisions ({len(train_files)} files), held out "
          f"{len(hold['label']):,} ({len(hold_files)} files) | illegal labels dropped {bad_t + bad_h} | "
          f"wait share {np.mean(train['label'] == 0):.1%} | loaded in {time.perf_counter() - t0:.1f}s", flush=True)

    net = model.Net().to(device)
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr)
    best = None
    history = []
    np_rng = np.random.default_rng(args.seed)
    for epoch in range(1, args.epochs + 1):
        net.train()
        te = time.perf_counter()
        perm = np_rng.permutation(n)
        loss_sum = 0.0
        steps = 0
        for s in range(0, n, args.batch):
            x, y = batch(train, perm[s:s + args.batch], device)
            logits, _ = net(x)
            loss = (F.cross_entropy(logits, y, reduction="none") * _weights(y, args.wait_weight)).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            loss_sum += float(loss.detach())
            steps += 1
        m = evaluate(net, hold, device, wait_weight=args.wait_weight)
        m.update(epoch=epoch, train_loss=loss_sum / max(steps, 1), seconds=round(time.perf_counter() - te, 1))
        history.append(m)
        print(f"epoch {epoch}: train loss {m['train_loss']:.4f} | held-out loss {m['loss']:.4f} | acc {m['acc']:.1%} | "
              f"wait {m['acc_wait']:.1%} | non-wait {m['acc_nonwait']:.1%} | top-3 non-wait {m['top3_nonwait']:.1%} | "
              f"{m['seconds']}s", flush=True)
        if best is None or m["acc_nonwait"] > best["acc_nonwait"]:
            best = dict(m)
            best_state = {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}

    net = net.cpu()
    net.load_state_dict(best_state)
    trained_on = {
        "data": data_meta,
        "train_files": [f.name for f in train_files],
        "holdout_files": [f.name for f in hold_files],
        "args": vars(args),
        "best_epoch": best["epoch"],
        "metrics": best,
        "history": history,
    }
    out = Path(args.out)
    save_checkpoint(net, out, trained_on)
    print(f"saved epoch {best['epoch']} -> {out} ({out.stat().st_size / 1e6:.2f} MB): held-out non-wait "
          f"{best['acc_nonwait']:.1%}, top-3 {best['top3_nonwait']:.1%}, wait {best['acc_wait']:.1%}, "
          f"overall {best['acc']:.1%} | total {time.perf_counter() - t0:.0f}s")
    return trained_on


if __name__ == "__main__":
    main()

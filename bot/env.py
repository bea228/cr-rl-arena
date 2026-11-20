"""Batched training environment (handoff B11, W11).

    env = VecEnv(n=256, workers=14, seed=0, opponent_sampler=League(...))
    obs = env.reset()                          # dict of [n, ...] float32 arrays (bot.encode keys)
    obs, reward, done, infos = env.step(actions)   # actions: int array [n] of learner indices
    env.set_shape(0.5); env.close()

The learner is seat 0 in every battle. One step applies the learner's action and the opponent
policy's action on the first tick, then advances 9 more ticks with None (10 ticks = one
decision). Reward: +1 win / -1 loss / 0 draw at the end, plus each step
shape * (enemy tower hp lost - our tower hp lost) / 10,000 (princess + king, absolute hp).
A finished battle is reset at once with a new seed and decks; the returned obs for that env
is the first obs of the new episode and done[i] marks the boundary; infos holds one dict per
finished episode.

The Python work per env (observe, trackers, encode, opponent policy) is the bottleneck, so
the battles are split across W worker processes (spawn context, daemon, Pipes); the main
process only holds the learner network. workers=0 runs everything in-process (tests, debug).

Seeds: every env index i has its own random.Random(seed + i * 1_000_003) for episode seeds,
decks and opponent draws, so results are identical for any worker count.

Decks (never held-out cards): learner = `learner_deck` (default.json) with probability
`deck_mix` (0.4), else a deck from `deck_list` with probability `deck_list_mix` (0; the hook
for the handoff's "chosen deck list"), else random_deck. Opponent = random_deck always.

    python -m bot.env --bench            # decisions/s, random and scripted opponents, 2 worker counts
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import random
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
DEFAULT_DECK_FILE = REPO / "bot" / "decks" / "default.json"
STEP_TICKS = 10
SHAPE_SCALE = 10_000.0
SEED_STRIDE = 1_000_003

# entity fields (see bot.state)
_OWNER, _CARD, _HP, _IS_TOWER = 0, 1, 4, 7


class FixedOpponent:
    """Picklable sampler that always returns the same spec."""

    def __init__(self, spec: str = "random"):
        self.spec = spec

    def __call__(self, rng: random.Random) -> str:
        return self.spec

    def set_snapshots(self, paths) -> None:
        pass


def tower_hp(obs: dict) -> tuple[float, float]:
    """(our princess+king hp, enemy princess+king hp); destroyed towers are absent -> 0."""
    ours = theirs = 0.0
    for e in obs.get("entities", []):
        if e[_IS_TOWER]:
            if int(e[_OWNER]) == 0:
                ours += float(e[_HP])
            else:
                theirs += float(e[_HP])
    return ours, theirs


def opponent_kind(spec: str, bc_path: str | None = None) -> str:
    """'random' | 'scripted' | 'bc' | 'snapshot' | spec."""
    if spec in ("random", "scripted"):
        return spec
    if spec.startswith("learned:"):
        p = spec[len("learned:"):]
        if Path(p).name.startswith("snap_"):
            return "snapshot"
        if bc_path is None or Path(p).resolve() == Path(bc_path).resolve() or Path(p).name == "bc_v1.pt":
            return "bc"
        return "learned"
    return spec


class _Shard:
    """Owns a slice of battles (global env indices `idx`); used in-process or inside a worker."""

    def __init__(self, idx, seed, sampler, learner_deck, deck_mix, deck_list, deck_list_mix, shape):
        import cr_sim

        from evals.decks import load_heldout_cards

        self.idx = list(idx)
        self.sampler = sampler or FixedOpponent("random")
        self.learner_deck = list(learner_deck)
        self.deck_mix = float(deck_mix)
        self.deck_list = [list(d) for d in (deck_list or [])]
        self.deck_list_mix = float(deck_list_mix) if self.deck_list else 0.0
        self.shape = float(shape)
        self.heldout = set(load_heldout_cards())
        self.rngs = [random.Random(seed + i * SEED_STRIDE) for i in self.idx]
        self.battles = [cr_sim.Battle() for _ in self.idx]
        k = len(self.idx)
        self.views = [None] * k
        self.trackers = [None] * k
        self.opp = [None] * k
        self.ep = [None] * k  # per-episode info
        self.obs = [None] * k  # learner's current raw obs
        self.hp = [(0.0, 0.0)] * k

    # --- episode setup ---
    def pick_decks(self, rng: random.Random) -> tuple[list[str], str, list[str]]:
        from evals.decks import random_deck

        r = rng.random()
        if r < self.deck_mix:
            d0, kind = list(self.learner_deck), "default"
        elif r < self.deck_mix + self.deck_list_mix:
            d0, kind = list(rng.choice(self.deck_list)), "list"
        else:
            d0, kind = random_deck(rng, exclude=self.heldout), "random"
        return d0, kind, random_deck(rng, exclude=self.heldout)

    def _new_episode(self, j: int) -> None:
        from bot.trackers import Trackers
        from evals.policies import make_policy

        rng = self.rngs[j]
        gs = rng.randrange(2**31)
        d0, kind, d1 = self.pick_decks(rng)
        spec = self.sampler(rng)
        b = self.battles[j]
        b.reset(gs, d0, d1)
        self.views[j] = (b.view(0), b.view(1))
        opp = make_policy(spec, seed=gs)
        opp.reset(d1)
        self.opp[j] = opp
        self.trackers[j] = Trackers()
        self.ep[j] = {"opponent": spec, "deck_kind": kind, "seed": gs, "deck0": d0, "deck1": d1}
        self._observe(j)
        self.hp[j] = tower_hp(self.obs[j])

    def _observe(self, j: int) -> None:
        o = self.views[j][0].observe()
        self.trackers[j].update(o)
        self.obs[j] = o

    def _encoded(self) -> dict:
        from bot.encode import batch_encode, encode

        return batch_encode([encode(self.obs[j], self.trackers[j]) for j in range(len(self.idx))])

    # --- protocol ---
    def reset(self) -> dict:
        for j in range(len(self.idx)):
            self._new_episode(j)
        return self._encoded()

    def step(self, actions) -> tuple:
        from bot import placements

        k = len(self.idx)
        rew = np.zeros(k, dtype=np.float32)
        done = np.zeros(k, dtype=bool)
        infos = []
        for j in range(k):
            b = self.battles[j]
            a0 = placements.decode(int(actions[j]), self.obs[j])
            a1 = self.opp[j].act(self.views[j][1].observe())
            s0 = None if a0 is None else (int(a0[0]), float(a0[1]), float(a0[2]))
            s1 = None if a1 is None else (int(a1[0]), float(a1[1]), float(a1[2]))
            status = b.step(s0, s1)
            for _ in range(STEP_TICKS - 1):
                if status:
                    break
                status = b.step(None, None)
            self._observe(j)
            ours, theirs = tower_hp(self.obs[j])
            p_ours, p_theirs = self.hp[j]
            r = self.shape * ((p_theirs - theirs) - (p_ours - ours)) / SHAPE_SCALE
            self.hp[j] = (ours, theirs)
            if status:
                result = {1: "win", 2: "loss"}.get(int(status), "draw")
                r += {"win": 1.0, "loss": -1.0, "draw": 0.0}[result]
                info = dict(self.ep[j], env=self.idx[j], result=result,
                            crowns=list(b.crowns()), ticks=int(b.tick))
                infos.append(info)
                done[j] = True
                self._new_episode(j)
            rew[j] = r
        return self._encoded(), rew, done, infos

    def set_shape(self, x: float) -> None:
        self.shape = float(x)

    def set_snapshots(self, paths) -> None:
        if hasattr(self.sampler, "set_snapshots"):
            self.sampler.set_snapshots(list(paths))


def _worker(conn, args) -> None:
    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[var] = "1"
    sys.path.insert(0, str(REPO))
    try:
        import torch

        torch.set_num_threads(1)
    except ImportError:
        pass
    shard = _Shard(*args)
    try:
        while True:
            cmd, data = conn.recv()
            if cmd == "reset":
                conn.send(shard.reset())
            elif cmd == "step":
                conn.send(shard.step(data))
            elif cmd == "shape":
                shard.set_shape(data)
            elif cmd == "snapshots":
                shard.set_snapshots(data)
            elif cmd == "close":
                break
    except (EOFError, KeyboardInterrupt, BrokenPipeError):
        pass
    finally:
        conn.close()


def _concat(parts: list[dict]) -> dict:
    return {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}


class VecEnv:
    def __init__(self, n=256, workers=None, seed=0, opponent_sampler=None, learner_deck=None,
                 deck_mix=0.4, deck_list=None, deck_list_mix=0.0, shape=1.0):
        if learner_deck is None:
            learner_deck = json.loads(DEFAULT_DECK_FILE.read_text(encoding="utf-8"))
        if workers is None:
            workers = max(1, (os.cpu_count() or 2) - 2)
        self.n = int(n)
        self.workers = max(0, min(int(workers), self.n))
        sampler = opponent_sampler or FixedOpponent("random")
        common = (seed, sampler, learner_deck, deck_mix, deck_list, deck_list_mix, shape)
        self._procs, self._conns, self._local = [], [], None
        self.closed = False
        if self.workers == 0:
            self._local = _Shard(range(self.n), *common)
            return
        ctx = mp.get_context("spawn")
        for w, part in enumerate(np.array_split(np.arange(self.n), self.workers)):
            parent, child = ctx.Pipe()
            p = ctx.Process(target=_worker, args=(child, (part.tolist(), *common)), daemon=True,
                            name=f"vecenv-{w}")
            p.start()
            child.close()
            self._procs.append(p)
            self._conns.append(parent)
        self._splits = np.cumsum([len(p) for p in np.array_split(np.arange(self.n), self.workers)])[:-1]

    def reset(self) -> dict:
        if self._local is not None:
            return self._local.reset()
        for c in self._conns:
            c.send(("reset", None))
        return _concat([c.recv() for c in self._conns])

    def step(self, actions) -> tuple:
        actions = np.asarray(actions, dtype=np.int64)
        if self._local is not None:
            return self._local.step(actions)
        for c, part in zip(self._conns, np.split(actions, self._splits)):
            c.send(("step", part))
        res = [c.recv() for c in self._conns]
        infos = [i for r in res for i in r[3]]
        return (_concat([r[0] for r in res]), np.concatenate([r[1] for r in res]),
                np.concatenate([r[2] for r in res]), infos)

    def _broadcast(self, cmd, data) -> None:
        if self._local is not None:
            getattr(self._local, {"shape": "set_shape", "snapshots": "set_snapshots"}[cmd])(data)
            return
        for c in self._conns:
            c.send((cmd, data))

    def set_shape(self, x: float) -> None:
        self._broadcast("shape", float(x))

    def set_snapshots(self, paths) -> None:
        self._broadcast("snapshots", [str(p) for p in paths])

    def close(self) -> None:
        if self.closed:
            return
        self.closed = True
        for c in self._conns:
            try:
                c.send(("close", None))
            except (BrokenPipeError, OSError):
                pass
        for p in self._procs:
            p.join(timeout=5)
            if p.is_alive():
                p.terminate()
                p.join(timeout=2)
        for c in self._conns:
            c.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass


def random_actions(obs: dict, rng: np.random.Generator) -> np.ndarray:
    """Uniform over the legal mask entries of each env."""
    m = obs["mask"] > 0
    u = rng.random(m.shape) * m
    return u.argmax(1)


def measure(n=256, workers=None, opponent="random", steps=40, seed=0) -> float:
    env = VecEnv(n=n, workers=workers, seed=seed, opponent_sampler=FixedOpponent(opponent))
    try:
        rng = np.random.default_rng(seed)
        obs = env.reset()
        obs, *_ = env.step(random_actions(obs, rng))  # warm-up
        t0 = time.perf_counter()
        for _ in range(steps):
            obs, *_ = env.step(random_actions(obs, rng))
        return n * steps / (time.perf_counter() - t0)
    finally:
        env.close()


def benchmark(worker_counts=None, n=256, steps=40) -> dict:
    cpus = os.cpu_count() or 2
    if worker_counts is None:
        worker_counts = sorted({max(1, cpus // 2 - 1), max(1, cpus - 2)})
    out = {}
    for w in worker_counts:
        for opp in ("random", "scripted"):
            dps = measure(n=n, workers=w, opponent=opp, steps=steps)
            out[(w, opp)] = dps
            print(f"VecEnv n={n} workers={w:2d} opponent={opp:8s}: {dps:,.0f} decisions/s", flush=True)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", action="store_true")
    ap.add_argument("--workers", type=int, nargs="*", default=None)
    ap.add_argument("--envs", type=int, default=256)
    ap.add_argument("--steps", type=int, default=40)
    a = ap.parse_args()
    if a.bench:
        benchmark(a.workers, n=a.envs, steps=a.steps)

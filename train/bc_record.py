"""Record scripted-bot decisions for behaviour cloning (handoff B8).

    python -m train.bc_record                      # defaults: 8000 games into data/bc/
    python -m train.bc_record --games 200 --workers 4 --out data/bc_small

Game mix: 50% scripted vs scripted (both seats recorded), 50% scripted vs RandomPolicy
(only the scripted seat recorded, seat chosen at random). The recorded side's deck is
bot/decks/default.json 30% of the time, else random_deck(exclude=heldout cards); opponents
always get random_deck(exclude=heldout). Game i uses seed --seed + i, so a run is reproducible
for a given --games-per-shard.

At each decision tick, for each scripted seat: idx = act_index(obs) (updates its trackers);
keep it if idx != 0, or with probability --keep-wait if it is a wait; kept decisions are
stored as encode(obs, trackers) + idx. Each worker job (a chunk of games) writes its own
shard data/bc/shard_<chunk>.npz: float16 features, uint8 masks, int16 labels, compressed.
data/bc/meta.json records the commit, args and totals.
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import random
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
SCRIPTED_VERSION = "scripted v1 after fix pass, royal_ghost rule"
DEFAULT_DECK_FILE = REPO / "bot" / "decks" / "default.json"

FLOAT_KEYS = ("globals", "hand", "units", "enemy_deck")
MASK_KEYS = ("units_mask", "enemy_deck_mask", "mask")


def _git_commit() -> str:
    try:
        out = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=REPO,
                             capture_output=True, text=True, timeout=10)
        return out.stdout.strip() or "unknown"
    except Exception:
        return "unknown"


def game_setup(i: int, seed: int, heldout: list[str], default_deck: list[str]):
    """(game_seed, mode, deck0, deck1, scripted_seats) for game i; deterministic in (seed, i)."""
    from evals.decks import random_deck

    gs = seed + i
    rng = random.Random(gs)
    mode = "scripted" if rng.random() < 0.5 else "random"
    rec_deck = list(default_deck) if rng.random() < 0.3 else random_deck(rng, exclude=heldout)
    opp_deck = random_deck(rng, exclude=heldout)
    if mode == "scripted":
        return gs, mode, rec_deck, opp_deck, (0, 1)
    seat = rng.randrange(2)
    decks = (rec_deck, opp_deck) if seat == 0 else (opp_deck, rec_deck)
    return gs, mode, decks[0], decks[1], (seat,)


def record_game(i: int, seed: int, keep_wait: float, heldout, default_deck):
    """Play game i; return (list of (encoded, idx), n_decisions_seen)."""
    import cr_sim

    from bot import placements
    from bot.encode import encode
    from bot.policy import DECIDE_EVERY, RandomPolicy
    from bot.scripted import ScriptedPolicy

    gs, mode, d0, d1, scripted = game_setup(i, seed, heldout, default_deck)
    b = cr_sim.Battle()
    b.reset(gs, d0, d1)
    views = (b.view(0), b.view(1))
    pols = [ScriptedPolicy() if s in scripted else RandomPolicy(gs * 2 + s) for s in (0, 1)]
    pols[0].reset(d0)
    pols[1].reset(d1)
    keep_rng = random.Random(gs * 7919 + 1)
    out, seen = [], 0
    while not b.done():
        step = [None, None]
        if b.tick % DECIDE_EVERY == 0:
            for s in (0, 1):
                obs = views[s].observe()
                if s in scripted:
                    idx = pols[s].act_index(obs)
                    seen += 1
                    if idx != 0 or keep_rng.random() < keep_wait:
                        out.append((encode(obs, pols[s].trackers), idx))
                    a = placements.decode(idx, obs)
                else:
                    a = pols[s].act(obs)
                if a is not None:
                    step[s] = (int(a[0]), float(a[1]), float(a[2]))
        b.step(step[0], step[1])
    return out, seen


def write_shard(path: Path, recs: list) -> None:
    from bot.encode import KEYS

    arrays = {}
    for k in KEYS:
        stacked = np.stack([x[k] for x, _ in recs])
        arrays[k] = stacked.astype(np.uint8 if k in MASK_KEYS else np.float16)
    arrays["label"] = np.asarray([idx for _, idx in recs], dtype=np.int16)
    np.savez_compressed(path, **arrays)


def run_chunk(job: tuple) -> dict:
    """Worker: play a chunk of games, write one shard. job = (chunk, game indices, seed, keep_wait, out, heldout, deck)."""
    chunk, games, seed, keep_wait, out_dir, heldout, default_deck = job
    recs, seen = [], 0
    for i in games:
        r, n = record_game(i, seed, keep_wait, heldout, default_deck)
        recs.extend(r)
        seen += n
    waits = sum(1 for _, idx in recs if idx == 0)
    path = Path(out_dir) / f"shard_{chunk:04d}.npz"
    if recs:
        write_shard(path, recs)
    return {"chunk": chunk, "games": len(games), "stored": len(recs), "waits": waits, "seen": seen,
            "bytes": path.stat().st_size if recs else 0}


def _init_worker():
    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[var] = "1"


def main(argv=None) -> dict:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--games", type=int, default=8000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--keep-wait", type=float, default=0.1, help="probability of keeping a wait decision")
    ap.add_argument("--games-per-shard", type=int, default=250)
    ap.add_argument("--max-decisions", type=int, default=None, help="stop once this many are stored")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--out", default=str(REPO / "data" / "bc"))
    args = ap.parse_args(argv)

    sys.path.insert(0, str(REPO))
    from evals.decks import load_heldout_cards

    heldout = load_heldout_cards()
    default_deck = json.loads(DEFAULT_DECK_FILE.read_text(encoding="utf-8"))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for old in list(out.glob("shard_*.npz")) + [out / "meta.json"]:
        old.unlink(missing_ok=True)

    gps = max(1, args.games_per_shard)
    jobs = [(c, list(range(s, min(s + gps, args.games))), args.seed, args.keep_wait, str(out), heldout, default_deck)
            for c, s in enumerate(range(0, args.games, gps))]
    t0 = time.perf_counter()
    results = []
    stopped_early = False

    def _progress(r):
        results.append(r)
        stored = sum(x["stored"] for x in results)
        games = sum(x["games"] for x in results)
        print(f"  shard {r['chunk']:4d}: {r['stored']:6d} stored | total {stored:,} decisions, {games} games, "
              f"{time.perf_counter() - t0:.0f}s", flush=True)
        return stored

    if args.workers <= 1:
        for j in jobs:
            if _progress(run_chunk(j)) >= (args.max_decisions or float("inf")):
                stopped_early = True
                break
    else:
        with mp.Pool(args.workers, initializer=_init_worker) as pool:
            for r in pool.imap_unordered(run_chunk, jobs):
                if _progress(r) >= (args.max_decisions or float("inf")):
                    stopped_early = True
                    pool.terminate()
                    break

    wall = time.perf_counter() - t0
    stored = sum(r["stored"] for r in results)
    waits = sum(r["waits"] for r in results)
    seen = sum(r["seen"] for r in results)
    games = sum(r["games"] for r in results)
    disk = sum(p.stat().st_size for p in out.glob("shard_*.npz"))
    meta = {
        "commit": _git_commit(),
        "scripted_version": SCRIPTED_VERSION,
        "args": vars(args),
        "games": games,
        "shards": len(list(out.glob("shard_*.npz"))),
        "decisions_seen": seen,
        "decisions_stored": stored,
        "waits_stored": waits,
        "wait_share": waits / stored if stored else 0.0,
        "stopped_early": stopped_early,
        "wall_seconds": round(wall, 1),
        "decisions_per_second": round(seen / wall, 1) if wall else 0.0,
        "disk_bytes": disk,
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
    print(f"games {games} | decisions seen {seen:,} | stored {stored:,} | wait share {100 * meta['wait_share']:.1f}%")
    print(f"wall {wall:.1f}s | {meta['decisions_per_second']:,.0f} decisions/s seen | "
          f"{stored / wall:,.0f} stored/s | disk {disk / 1e6:.1f} MB in {meta['shards']} shards -> {out}")
    return meta


if __name__ == "__main__":
    main()

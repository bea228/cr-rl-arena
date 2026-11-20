"""Gauntlet every PPO snapshot in a run that hasn't been evaluated yet.

    python -m train.eval_snaps runs/<run> [--games 200] [--workers N]

For each runs/<run>/snap_*.pt not already in runs/<run>/evals.csv: learned:<snap> vs random
and vs scripted, both with --a-deck bot/decks/default.json (seed 0, rotating opponent decks).
Appends one row per snapshot to evals.csv and prints a table with bc_v1 on the same settings
as the first row (bc_v1's numbers are cached in evals.csv too, per game count).
"""
from __future__ import annotations

import argparse
import csv
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

FIELDS = ["model", "update", "games", "vs_random", "vs_scripted", "crown_diff_random", "crown_diff_scripted"]
DEFAULT_DECK = REPO / "bot" / "decks" / "default.json"
BC_PATH = REPO / "models" / "bc_v1.pt"


def _update_of(p: Path) -> int:
    m = re.match(r"snap_(\d+)\.pt$", p.name)
    return int(m.group(1)) if m else -1


def evaluate(model_path: Path, games: int, workers, seed: int = 0) -> dict:
    from evals.gauntlet import run_gauntlet

    spec = f"learned:{model_path}"
    r = run_gauntlet(spec, "random", games=games, seed=seed, workers=workers, a_deck=str(DEFAULT_DECK), save=False)
    s = run_gauntlet(spec, "scripted", games=games, seed=seed, workers=workers, a_deck=str(DEFAULT_DECK), save=False)
    return {"vs_random": r["win_rate"], "vs_scripted": s["win_rate"],
            "crown_diff_random": r["mean_crown_diff"], "crown_diff_scripted": s["mean_crown_diff"]}


def read_rows(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def append_row(path: Path, row: dict) -> None:
    new = not path.exists() or path.stat().st_size == 0
    with path.open("a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS, extrasaction="ignore")
        if new:
            w.writeheader()
        w.writerow({k: (f"{v:.4f}" if isinstance(v, float) else v) for k, v in row.items()})


def main(argv=None) -> list[dict]:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("run_dir")
    ap.add_argument("--games", type=int, default=200)
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--bc", default=str(BC_PATH), help="baseline model for the first table row")
    args = ap.parse_args(argv)

    run = Path(args.run_dir)
    if not run.is_absolute():
        run = Path.cwd() / run
    out = run / "evals.csv"
    rows = read_rows(out)
    done = {(r["model"], int(r["games"])) for r in rows}

    bc_key = ("bc_v1", args.games)
    if bc_key not in done:
        row = {"model": "bc_v1", "update": 0, "games": args.games, **evaluate(Path(args.bc), args.games, args.workers)}
        append_row(out, row)
        rows.append({k: str(v) for k, v in row.items()})
    for snap in sorted(run.glob("snap_*.pt"), key=_update_of):
        if (snap.name, args.games) in done:
            continue
        row = {"model": snap.name, "update": _update_of(snap), "games": args.games,
               **evaluate(snap, args.games, args.workers)}
        append_row(out, row)
        rows.append({k: str(v) for k, v in row.items()})
        print(f"  {snap.name}: vs random {row['vs_random']:.1%}, vs scripted {row['vs_scripted']:.1%}", flush=True)

    table = [r for r in rows if int(r["games"]) == args.games]
    table.sort(key=lambda r: (r["model"] != "bc_v1", int(r["update"])))
    print(f"\n{run.name}: {args.games} games each, a-deck default.json, seed 0")
    print(f"{'model':16} {'vs random':>10} {'vs scripted':>12}")
    for r in table:
        print(f"{r['model']:16} {float(r['vs_random']):>10.1%} {float(r['vs_scripted']):>12.1%}")
    return table


if __name__ == "__main__":
    main()

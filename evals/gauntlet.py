"""Gauntlet: policy a vs policy b over a fixed deck list (handoff B5 + W11).

    ./.venv/Scripts/python -m evals.gauntlet --a random --b random --games 400
    ./.venv/Scripts/python -m evals.gauntlet --a scripted --b random --a-deck bot/decks/default.json
    ./.venv/Scripts/python -m evals.gauntlet --a scripted --b random --decks heldout

Game i uses seed seed + i; a plays decks[i % n] (or the --a-deck), b plays decks[(i + 1) % n];
a is seat 0 on even i and seat 1 on odd i. Every number is from a's point of view.
Results are identical for any --workers (fresh policies per game, seeded from the game seed).
"""
from __future__ import annotations

import argparse
import json
import multiprocessing
import os
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

from evals.policies import make_policy, spec_slug

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
RESULTS_DIR = HERE / "results"
FLIP = {"win": "loss", "loss": "win", "draw": "draw"}
N_WORST = 10


def _resolve_decks(decks) -> list:
    from evals.decks import load_gauntlet, load_heldout_decks

    if isinstance(decks, list):
        return decks
    if decks == "gauntlet":
        return load_gauntlet()
    if decks == "heldout":
        return load_heldout_decks()
    with open(decks, encoding="utf-8") as f:
        out = json.load(f)
    if not isinstance(out, list) or not out or not all(isinstance(d, list) for d in out):
        raise ValueError(f"{decks}: expected a JSON list of decks")
    return out


def _load_deck(path) -> list:
    with open(path, encoding="utf-8") as f:
        deck = json.load(f)
    if not isinstance(deck, list) or not all(isinstance(c, str) for c in deck):
        raise ValueError(f"{path}: expected a JSON list of card names")
    return deck


def _limit_torch_threads() -> None:
    """torch.set_num_threads(1), but only if torch is already loaded in this process."""
    torch = sys.modules.get("torch")
    if torch is not None and torch.get_num_threads() != 1:
        torch.set_num_threads(1)


def _init_worker() -> None:
    # W11: one thread per worker. Importing torch eagerly here costs ~11 s per worker on Windows
    # (15 workers importing at once made the Pool slower than a plain loop), so instead pin the
    # thread env vars, which torch reads when a learned policy imports it later, and set the
    # count directly if torch is already loaded.
    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[var] = "1"
    _limit_torch_threads()


def _play_one(job: tuple) -> dict:
    """Worker: one game. job = (i, a_spec, b_spec, game_seed, a_deck, b_deck)."""
    from evals.match import play

    i, a_spec, b_spec, game_seed, a_deck, b_deck = job
    a_seat = i % 2
    b_seat = 1 - a_seat
    pa = make_policy(a_spec, seed=game_seed * 2 + a_seat)
    pb = make_policy(b_spec, seed=game_seed * 2 + b_seat)
    _limit_torch_threads()  # a learned policy may have just imported torch
    if a_seat == 0:
        m = play(pa, pb, game_seed, a_deck, b_deck)
    else:
        m = play(pb, pa, game_seed, b_deck, a_deck)
    result = m["result"] if a_seat == 0 else FLIP[m["result"]]
    return {
        "i": i,
        "a_seat": a_seat,
        "result": result,
        "crown_diff": m["crowns"][a_seat] - m["crowns"][b_seat],
        "ticks": m["ticks"],
        "ignored_a": m["ignored"][a_seat],
        "ignored_b": m["ignored"][b_seat],
        "decisions_a": m["decisions"][a_seat],
        "decisions_b": m["decisions"][b_seat],
        "reasons_a": m["ignored_by_reason"][a_seat],
        "reasons_b": m["ignored_by_reason"][b_seat],
        "match": m if result == "loss" else None,  # only losses can become replays; keeps IPC small
    }


def _git_commit() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], cwd=REPO, capture_output=True, text=True, timeout=10
        )
        return out.stdout.strip() if out.returncode == 0 and out.stdout.strip() else "unknown"
    except Exception:
        return "unknown"


def _stem(a, b, decks, a_deck) -> str:
    stem = f"{spec_slug(a)}_vs_{spec_slug(b)}"
    if a_deck is not None:
        stem += f"_on_{Path(a_deck).stem}"
    if decks == "heldout":
        stem += "_heldout"
    elif isinstance(decks, str) and decks != "gauntlet":
        stem += f"_{Path(decks).stem}"
    return stem


def _rate(num, den) -> float:
    return num / den if den else 0.0


def run_gauntlet(a, b, games=400, seed=0, decks="gauntlet", workers=None, a_deck=None, save=True) -> dict:
    t0 = time.perf_counter()
    deck_list = _resolve_decks(decks)
    n = len(deck_list)
    fixed = _load_deck(a_deck) if a_deck is not None else None
    if workers is None:
        workers = max(1, (os.cpu_count() or 2) - 1)
    workers = max(1, int(workers))

    jobs = [
        (i, a, b, seed + i, fixed if fixed is not None else deck_list[i % n], deck_list[(i + 1) % n])
        for i in range(games)
    ]
    if workers == 1 or games <= 1:
        rows = [_play_one(j) for j in jobs]
    else:
        chunksize = max(1, games // (workers * 4))
        with multiprocessing.Pool(min(workers, games), initializer=_init_worker) as pool:
            rows = list(pool.imap(_play_one, jobs, chunksize=chunksize))

    res = Counter(r["result"] for r in rows)
    reasons_a: Counter = Counter()
    reasons_b: Counter = Counter()
    for r in rows:
        reasons_a.update(r["reasons_a"])
        reasons_b.update(r["reasons_b"])
    ign_a = sum(r["ignored_a"] for r in rows)
    ign_b = sum(r["ignored_b"] for r in rows)
    dec_a = sum(r["decisions_a"] for r in rows)
    dec_b = sum(r["decisions_b"] for r in rows)

    group_by = "b_deck" if fixed is not None else "a_deck"
    per: dict[int, list[int]] = {}
    for r in rows:
        k = (r["i"] + 1) % n if fixed is not None else r["i"] % n
        wins_games = per.setdefault(k, [0, 0])
        wins_games[0] += r["result"] == "win"
        wins_games[1] += 1
    per_deck = {str(k): {"win_rate": w / g, "games": g} for k, (w, g) in sorted(per.items())}
    worst_idx = min(per, key=lambda k: (per[k][0] / per[k][1], k)) if per else None

    losses = sorted((r for r in rows if r["result"] == "loss"), key=lambda r: (r["crown_diff"], r["ticks"], r["i"]))
    worst = losses[:N_WORST]
    stem = _stem(a, b, decks, a_deck)

    summary = {
        "a": a,
        "b": b,
        "args": {
            "a": a, "b": b, "games": games, "seed": seed,
            "decks": decks if isinstance(decks, str) else "<list>",
            "workers": workers, "a_deck": str(a_deck) if a_deck is not None else None,
        },
        "games": games,
        "a_seat0_games": sum(1 for r in rows if r["a_seat"] == 0),
        "win_rate": _rate(res["win"], games),
        "draw_rate": _rate(res["draw"], games),
        "loss_rate": _rate(res["loss"], games),
        "mean_crown_diff": _rate(sum(r["crown_diff"] for r in rows), games),
        "ignored_rate_a": _rate(ign_a, dec_a),
        "ignored_rate_b": _rate(ign_b, dec_b),
        "ignored_a": ign_a, "decisions_a": dec_a,
        "ignored_b": ign_b, "decisions_b": dec_b,
        "ignored_reasons_a": dict(sorted(reasons_a.items())),
        "ignored_reasons_b": dict(sorted(reasons_b.items())),
        "per_deck_group_by": group_by,
        "per_deck": per_deck,
        "worst_deck_index": worst_idx,
        "worst_deck_win_rate": per[worst_idx][0] / per[worst_idx][1] if worst_idx is not None else None,
        "worst_losses": [
            {
                "game": r["i"], "seed": r["match"]["seed"], "a_seat": r["a_seat"],
                "crown_diff": r["crown_diff"], "ticks": r["ticks"],
                "seat0_result": r["match"]["result"], "crowns": r["match"]["crowns"],
            }
            for r in worst
        ],
        "stem": stem,
        "git_commit": _git_commit(),
    }

    if save:
        from evals.replay import save_match

        results_dir = Path(RESULTS_DIR)
        replay_dir = results_dir / "replays" / stem
        results_dir.mkdir(parents=True, exist_ok=True)
        if replay_dir.exists():
            for old in replay_dir.glob("loss_*.json"):
                old.unlink()
        for rank, (r, entry) in enumerate(zip(worst, summary["worst_losses"]), start=1):
            p = replay_dir / f"loss_{rank}_g{r['i']}.json"
            save_match(p, r["match"])
            entry["path"] = str(p)
        summary["path"] = str(results_dir / f"{stem}.json")

    summary["wall_time_s"] = time.perf_counter() - t0
    if save:
        with open(summary["path"], "w", encoding="utf-8", newline="\n") as f:
            json.dump(summary, f, indent=1)
            f.write("\n")
    return summary


def print_summary(s: dict) -> None:
    print(f"{s['a']} vs {s['b']}  ({s['games']} games, decks {s['args']['decks']}"
          f"{', a-deck ' + s['args']['a_deck'] if s['args']['a_deck'] else ''}, seed {s['args']['seed']})")
    print(f"win rate (a)       {s['win_rate']:.1%}")
    print(f"draw rate          {s['draw_rate']:.1%}")
    print(f"mean crown diff    {s['mean_crown_diff']:+.2f}")
    print(f"ignored rate a     {s['ignored_rate_a']:.2%}  ({s['ignored_a']}/{s['decisions_a']})")
    print(f"ignored rate b     {s['ignored_rate_b']:.2%}  ({s['ignored_b']}/{s['decisions_b']})")
    print(f"ignored reasons a  {s['ignored_reasons_a']}")
    if s["worst_deck_index"] is not None:
        print(f"worst-deck win     {s['worst_deck_win_rate']:.1%}  "
              f"({s['per_deck_group_by']} #{s['worst_deck_index']})")
    print(f"games              {s['games']}  (a seat 0 in {s['a_seat0_games']})")
    print(f"wall time          {s['wall_time_s']:.1f} s  (commit {s['git_commit']})")
    if "path" in s:
        print(f"saved              {s['path']}  (+{len(s['worst_losses'])} loss replays)")


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--a", required=True, help="policy spec: random, scripted, learned:<path>, class:<mod>.<Cls>")
    ap.add_argument("--b", required=True, help="policy spec (same forms as --a)")
    ap.add_argument("--games", type=int, default=400)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--decks", default="gauntlet", help="gauntlet | heldout | <json file with a list of decks>")
    ap.add_argument("--workers", type=int, default=None, help="default: cpu_count - 1; 1 = no Pool")
    ap.add_argument("--a-deck", default=None, help="a always plays this deck JSON; b rotates through --decks")
    args = ap.parse_args(argv)
    s = run_gauntlet(args.a, args.b, games=args.games, seed=args.seed, decks=args.decks,
                     workers=args.workers, a_deck=args.a_deck, save=True)
    print_summary(s)


if __name__ == "__main__":
    main()

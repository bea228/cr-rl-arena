"""Deck round robin: which candidate deck should the scripted bot take to the tournament (B16 + C14).

    ./.venv/Scripts/python -m evals.deck_rr
    ./.venv/Scripts/python -m evals.deck_rr --games-screen 300 --games-final 1000 --top 3

Our bot (scripted) always plays the candidate deck; each opponent rotates through the 40 gauntlet
decks (evals.gauntlet with --a-deck). Two phases:
  1. screen: every deck in candidate_decks.json, --games-screen games per opponent, seed 0
  2. final:  the --top best non-baseline decks by screen score, plus the baseline,
             --games-final games per opponent, seed 50000
Per deck: win rate vs each opponent, mean, lowest, score = mean - std across opponents
(population std), and our ignored-move rate. Saved to evals/results/deck_rr.json.
"""
from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import tempfile
import time
from pathlib import Path

from evals.gauntlet import RESULTS_DIR, run_gauntlet

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
CANDIDATES = HERE / "candidate_decks.json"
OUT_NAME = "deck_rr.json"

OUR_BOT = "scripted"
OPPONENTS = {
    "random": "random",
    "scripted": "scripted",
    "v0": "class:bot.scripted_v0.ScriptedPolicyV0",
}
SCREEN_SEED = 0
FINAL_SEED = 50000
IGNORED_MAX = 0.001  # "ignored-move rate ~0" for the choice rule


def load_candidates(path=CANDIDATES) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        out = json.load(f)
    if not isinstance(out, list) or not all(isinstance(e, dict) and {"name", "deck"} <= set(e) for e in out):
        raise ValueError(f"{path}: expected a list of {{name, deck}} objects")
    return out


def avg_elixir(deck: list[str]) -> float:
    from bot.cards import elixir

    return sum(elixir(c) for c in deck) / len(deck)


def c13_violations(deck: list[str]) -> list[str]:
    """Handoff C13 rules. Air and splash count only troops/buildings, not spells. [] = passes."""
    from bot.cards import DEPLOYABLE, role

    probs = []
    if len(deck) != 8 or len(set(deck)) != 8:
        probs.append("needs 8 distinct cards")
    unknown = [c for c in deck if c not in DEPLOYABLE]
    if unknown:
        return probs + [f"not deployable: {unknown}"]
    r = {c: role(c) for c in deck}
    units = [c for c in deck if not r[c]["spell"]]
    n_wc = sum(r[c]["win_condition"] for c in deck)
    n_air = sum(r[c]["hits_air"] for c in units)
    n_splash = sum(r[c]["splash"] for c in units)
    n_spell = sum(r[c]["spell"] for c in deck)
    avg = avg_elixir(deck)
    if n_wc != 1:
        probs.append(f"{n_wc} win conditions (need 1)")
    if n_air < 2:
        probs.append(f"{n_air} non-spell cards hit air (need 2+)")
    if n_splash < 1:
        probs.append("no non-spell splash card")
    if not 1 <= n_spell <= 2:
        probs.append(f"{n_spell} spells (need 1-2)")
    if not 3.0 <= avg <= 3.8:
        probs.append(f"avg elixir {avg:.3f} (need 3.0-3.8)")
    return probs


def evaluate_deck(name: str, deck: list[str], games: int, seed: int, workers=None, opponents=None) -> dict:
    """One table row: our bot on `deck` vs each opponent, `games` games each."""
    opponents = OPPONENTS if opponents is None else opponents
    win, ign, dec = {}, 0, 0
    with tempfile.TemporaryDirectory() as tmp:
        deck_path = Path(tmp) / "deck.json"
        with open(deck_path, "w", encoding="utf-8", newline="\n") as f:
            json.dump(list(deck), f)
        for label, spec in opponents.items():
            s = run_gauntlet(OUR_BOT, spec, games=games, seed=seed, decks="gauntlet",
                             workers=workers, a_deck=str(deck_path), save=False)
            win[label] = s["win_rate"]
            ign += s["ignored_a"]
            dec += s["decisions_a"]
    rates = list(win.values())
    mean = statistics.fmean(rates)
    std = statistics.pstdev(rates)
    return {
        "name": name,
        "deck": list(deck),
        "avg_elixir": avg_elixir(deck),
        "games_per_opponent": games,
        "seed": seed,
        "win_rate": win,
        "mean": mean,
        "lowest": min(rates),
        "std": std,
        "score": mean - std,
        "ignored_rate": ign / dec if dec else 0.0,
    }


def run_phase(cands: list[dict], games: int, seed: int, workers=None) -> list[dict]:
    rows = [evaluate_deck(c["name"], c["deck"], games, seed, workers) for c in cands]
    return sorted(rows, key=lambda r: -r["score"])


def choose(final_rows: list[dict], baseline: str) -> tuple[str, str]:
    """C14 rule: best final score, only if vs-random >= baseline's and ignored ~0; else the baseline."""
    base = next(r for r in final_rows if r["name"] == baseline)
    best = max(final_rows, key=lambda r: r["score"])
    if best["name"] == baseline:
        return baseline, "baseline has the highest final score"
    if best["win_rate"]["random"] < base["win_rate"]["random"]:
        return baseline, (f"{best['name']} scores higher but loses to random more often "
                          f"({best['win_rate']['random']:.1%} < {base['win_rate']['random']:.1%})")
    if best["ignored_rate"] > IGNORED_MAX:
        return baseline, f"{best['name']} has ignored-move rate {best['ignored_rate']:.2%}"
    return best["name"], (f"highest final score ({best['score']:.3f} vs baseline {base['score']:.3f}), "
                          f"vs random {best['win_rate']['random']:.1%} >= baseline {base['win_rate']['random']:.1%}, "
                          f"ignored {best['ignored_rate']:.2%}")


def format_table(rows: list[dict]) -> str:
    opps = list(rows[0]["win_rate"]) if rows else []
    head = ["deck", "avg elx"] + [f"vs {o}" for o in opps] + ["mean", "lowest", "score", "ignored"]
    lines = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for r in rows:
        cells = [r["name"], f"{r['avg_elixir']:.2f}"] + [f"{r['win_rate'][o]:.1%}" for o in opps]
        cells += [f"{r['mean']:.1%}", f"{r['lowest']:.1%}", f"{r['score']:.3f}", f"{r['ignored_rate']:.2%}"]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def _git_commit() -> str:
    try:
        out = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=REPO,
                             capture_output=True, text=True, timeout=10)
        return out.stdout.strip() or "unknown"
    except Exception:
        return "unknown"


def main(argv=None) -> dict:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--games-screen", type=int, default=300)
    ap.add_argument("--games-final", type=int, default=1000)
    ap.add_argument("--top", type=int, default=3)
    ap.add_argument("--workers", type=int, default=None, help="default: cpu_count - 1")
    args = ap.parse_args(argv)

    t0 = time.perf_counter()
    cands = load_candidates()
    baseline = cands[0]["name"]

    screen = run_phase(cands, args.games_screen, SCREEN_SEED, args.workers)
    t_screen = time.perf_counter() - t0
    print(f"\nPhase 1 screen: {args.games_screen} games per opponent, seed {SCREEN_SEED} ({t_screen:.0f} s)")
    print(format_table(screen))

    top = [r["name"] for r in screen if r["name"] != baseline][: args.top]
    finalists = [c for c in cands if c["name"] in top or c["name"] == baseline]
    final = run_phase(finalists, args.games_final, FINAL_SEED, args.workers)
    wall = time.perf_counter() - t0
    print(f"\nPhase 2 final: {args.games_final} games per opponent, seed {FINAL_SEED} ({wall - t_screen:.0f} s)")
    print(format_table(final))

    choice, why = choose(final, baseline)
    chosen = next(c["deck"] for c in cands if c["name"] == choice)
    print(f"\nchoice: {choice}  {chosen}\n  {why}\nwall time {wall:.0f} s")

    out = {
        "args": vars(args), "our_bot": OUR_BOT, "opponents": OPPONENTS, "baseline": baseline,
        "screen": screen, "final": final,
        "choice": {"name": choice, "deck": chosen, "why": why},
        "wall_time_s": wall, "git_commit": _git_commit(),
    }
    results = Path(RESULTS_DIR)
    results.mkdir(parents=True, exist_ok=True)
    with open(results / OUT_NAME, "w", encoding="utf-8", newline="\n") as f:
        json.dump(out, f, indent=1)
        f.write("\n")
    print(f"saved {results / OUT_NAME}")
    return out


if __name__ == "__main__":
    main()

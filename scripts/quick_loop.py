"""Quick scoreboard: ScriptedPolicy vs RandomPolicy on the gauntlet decks.

    ./.venv/Scripts/python -m scripts.quick_loop --games 40
    ./.venv/Scripts/python scripts/quick_loop.py --games 40 --deck random
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse  # noqa: E402
import json  # noqa: E402
import random  # noqa: E402
import time  # noqa: E402
from collections import Counter  # noqa: E402

from bot.policy import RandomPolicy  # noqa: E402
from bot.scripted import ScriptedPolicy  # noqa: E402
from evals.decks import load_gauntlet, random_deck  # noqa: E402
from evals.match import play  # noqa: E402

DEFAULT_DECK_FILE = os.path.join("bot", "decks", "default.json")
FLIP = {"win": "loss", "loss": "win", "draw": "draw"}


def run(games: int, seed: int, deck_arg: str, verbose: bool = False) -> dict:
    gauntlet = load_gauntlet()
    fixed = None
    if deck_arg != "random":
        with open(deck_arg, encoding="utf-8") as f:
            fixed = json.load(f)
    bot = ScriptedPolicy()
    results = Counter()
    crown_diff = 0
    ignored = Counter()
    losses = []
    t0 = time.perf_counter()
    for i in range(games):
        g = seed + i
        deck = fixed if fixed is not None else random_deck(random.Random(g))
        opp_deck = gauntlet[i % len(gauntlet)]
        opp = RandomPolicy(g)
        seat = i % 2
        if seat == 0:
            r = play(bot, opp, g, deck, opp_deck)
            res = r["result"]
        else:
            r = play(opp, bot, g, opp_deck, deck)
            res = FLIP[r["result"]]
        results[res] += 1
        crown_diff += r["crowns"][seat] - r["crowns"][1 - seat]
        ignored.update(r["ignored_by_reason"][seat])
        if res != "win":
            losses.append((g, seat, res, r["crowns"], opp_deck))
    ms = (time.perf_counter() - t0) * 1000 / max(games, 1)
    out = {
        "games": games,
        "win_rate": results["win"] / games,
        "draw_rate": results["draw"] / games,
        "mean_crown_diff": crown_diff / games,
        "ignored": dict(ignored),
        "ms_per_game": ms,
        "fires": dict(bot.fires),
        "mask_fallbacks": bot.mask_fallbacks,
        "non_wins": losses,
    }
    return out


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--games", type=int, default=40)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--deck", default=DEFAULT_DECK_FILE, help='deck JSON file, or "random"')
    ap.add_argument("--show-losses", action="store_true")
    a = ap.parse_args(argv)
    r = run(a.games, a.seed, a.deck)
    ignored = {"not_your_side": 0, **r["ignored"]}
    print(f"games            {r['games']}  (deck: {a.deck}, seed {a.seed})")
    print(f"scripted win     {r['win_rate']:.1%}")
    print(f"draw             {r['draw_rate']:.1%}")
    print(f"mean crown diff  {r['mean_crown_diff']:+.2f}")
    print(f"ignored moves    {ignored}")
    print(f"ms per game      {r['ms_per_game']:.0f}")
    print(f"rule fires       {r['fires']}")
    print(f"mask fallbacks   {r['mask_fallbacks']}")
    if a.show_losses:
        for g, seat, res, crowns, opp_deck in r["non_wins"]:
            print(f"  {res:4} seed {g} seat {seat} crowns {crowns} opp {opp_deck}")


if __name__ == "__main__":
    main()

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cr_sim  # noqa: E402

from bot.cards import DEPLOYABLE  # noqa: E402
from evals import deck_rr  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
_BATTLE = cr_sim.Battle()


def test_candidates_load_and_names_unique():
    cands = deck_rr.load_candidates()
    assert len(cands) == 10
    names = [c["name"] for c in cands]
    assert len(set(names)) == len(names)


def test_every_candidate_is_legal():
    for c in deck_rr.load_candidates():
        deck = c["deck"]
        assert len(deck) == 8 and len(set(deck)) == 8, c["name"]
        assert all(card in DEPLOYABLE for card in deck), c["name"]
        _BATTLE.check_deck(list(deck))  # raises ValueError if illegal


def test_non_baseline_candidates_meet_c13():
    cands = deck_rr.load_candidates()
    assert deck_rr.c13_violations(cands[0]["deck"])  # the baseline has no spell, so it breaks C13
    for c in cands[1:]:
        assert deck_rr.c13_violations(c["deck"]) == [], c["name"]


def test_tournament_deck_is_a_legal_candidate():
    with open(REPO / "bot" / "decks" / "tournament.json", encoding="utf-8") as f:
        deck = json.load(f)
    _BATTLE.check_deck(list(deck))
    assert deck in [c["deck"] for c in deck_rr.load_candidates()]


def test_tiny_run_returns_table_row():
    c = deck_rr.load_candidates()[1]
    rows = deck_rr.run_phase([c], games=4, seed=0, workers=1)
    assert len(rows) == 1
    r = rows[0]
    for key in ("name", "deck", "avg_elixir", "win_rate", "mean", "lowest", "std", "score", "ignored_rate"):
        assert key in r
    assert set(r["win_rate"]) == set(deck_rr.OPPONENTS)
    assert abs(r["score"] - (r["mean"] - r["std"])) < 1e-12
    assert r["lowest"] == min(r["win_rate"].values())
    assert "deck" in deck_rr.format_table(rows)


def test_choose_keeps_baseline_unless_better():
    def row(name, rnd, score):
        return {"name": name, "win_rate": {"random": rnd}, "score": score, "ignored_rate": 0.0}

    base = row("default", 0.94, 0.70)
    assert deck_rr.choose([base, row("X", 0.95, 0.75)], "default")[0] == "X"
    assert deck_rr.choose([base, row("X", 0.90, 0.75)], "default")[0] == "default"
    assert deck_rr.choose([base, row("X", 0.95, 0.65)], "default")[0] == "default"

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bot.policy import DECIDE_EVERY, RandomPolicy  # noqa: E402
from evals.decks import load_gauntlet  # noqa: E402
from evals.match import play  # noqa: E402


def _check(r):
    assert r["result"] in ("win", "loss", "draw")
    assert r["ticks"] > 0
    assert len(r["crowns"]) == 2
    for tick, seat, a in r["actions"]:
        assert tick % DECIDE_EVERY == 0
        assert seat in (0, 1)
        assert type(a[0]) is int and type(a[1]) is float and type(a[2]) is float
    assert r["decisions"] == [sum(1 for _, s, _ in r["actions"] if s == seat) for seat in (0, 1)]
    for seat in (0, 1):
        assert r["ignored"][seat] == sum(r["ignored_by_reason"][seat].values())
        assert r["ignored"][seat] <= r["decisions"][seat]
    json.dumps(r)  # JSON-safe


def test_play_deterministic():
    d = load_gauntlet()
    r1 = play(RandomPolicy(1), RandomPolicy(2), 5, d[0], d[1])
    r2 = play(RandomPolicy(1), RandomPolicy(2), 5, d[0], d[1])
    assert r1 == r2
    _check(r1)
    assert sum(r1["decisions"]) > 0


def test_play_default_deck():
    d = load_gauntlet()
    r = play(RandomPolicy(3), RandomPolicy(4), 7, "default", d[2])
    _check(r)
    assert r["deck0"] == "default"

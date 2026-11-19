import importlib
import importlib.util
import os
import sys

import pytest

# tests/ has no conftest (shared folder rule), so put the repo root on the path here.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cr_sim  # noqa: E402

from bot.policy import DECIDE_EVERY, Policy, RandomPolicy  # noqa: E402

DECK = ["knight", "archers", "giant", "musketeer", "valkyrie", "minions", "wizard", "minipekka"]


def val(x):
    return x() if callable(x) else x


def collect_observations(n=100, max_games=10):
    """Decision-tick observations of seat 0 from real local games.

    Seat 0 is driven by RandomPolicy so the game actually progresses."""
    out = []
    pol = RandomPolicy(seed=0)
    for game in range(max_games):
        b = cr_sim.Battle()
        b.reset(game + 1, DECK, "default")
        pol.reset(DECK)
        v0 = b.view(0)
        while not val(b.done):
            obs = v0.observe()
            a = None
            if obs["tick"] % DECIDE_EVERY == 0:
                out.append(obs)
                a = pol.act(obs)
            # local Battle.step wants a tuple; the server takes the list as JSON
            b.step(tuple(a) if a is not None else None, None)
            if len(out) >= n:
                return out
    return out


@pytest.fixture(scope="module")
def observations():
    obs = collect_observations(150)
    assert len(obs) >= 100
    return obs


def test_imports_and_constant():
    assert DECIDE_EVERY == 10


def test_base_policy_contract():
    p = Policy()
    assert p.reset(["knight"] * 8) is None
    with pytest.raises(NotImplementedError):
        p.act_index({})


def test_random_policy_actions_are_valid(observations):
    from bot import placements

    pol = RandomPolicy(seed=7)
    pol.reset(DECK)
    indices = []
    for obs in observations:
        i = pol.act_index(obs)
        indices.append(i)
        assert 0 <= i < placements.N_ACTIONS
        assert placements.mask(obs)[i], placements.mask_reason(obs, i)
        a = placements.decode(i, obs)
        if i == 0:
            assert a is None
            continue
        assert isinstance(a, list) and len(a) == 3
        slot, x, y = a
        assert type(slot) is int and 0 <= slot <= 3
        assert obs["legal"][slot]
        assert isinstance(x, float) and isinstance(y, float)
    waits = indices.count(0)
    assert 0 < waits < len(indices)
    assert waits >= 0.4 * len(indices)  # 60% wait plus "no legal move" waits


def test_random_policy_act_matches_act_index(observations):
    a, b = RandomPolicy(seed=5), RandomPolicy(seed=5)
    from bot import placements

    for o in observations:
        assert a.act(o) == placements.decode(b.act_index(o), o)


def test_random_policy_waits_when_nothing_legal(observations):
    pol = RandomPolicy(seed=9)
    o = {**observations[0], "legal": [False] * 4}
    assert all(pol.act_index(o) == 0 for _ in range(50))


def test_random_policy_deterministic(observations):
    a, b = RandomPolicy(seed=123), RandomPolicy(seed=123)
    a.reset(DECK)
    b.reset(DECK)
    assert [a.act(o) for o in observations] == [b.act(o) for o in observations]


def test_policy_module_does_not_import_placements_at_top():
    """W1: bot.policy imports placements lazily, inside methods."""
    import ast

    path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "bot", "policy.py")
    with open(path, encoding="utf-8") as f:
        tree = ast.parse(f.read())
    top = [n for n in tree.body if isinstance(n, (ast.Import, ast.ImportFrom))]
    names = [a.name for n in top for a in n.names] + [getattr(n, "module", "") or "" for n in top]
    assert not any("placements" in n for n in names)
    assert hasattr(importlib.import_module("bot.policy"), "Policy")

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
    pol = RandomPolicy(seed=7)
    pol.reset(DECK)
    results = [pol.act(obs) for obs in observations]
    for obs, a in zip(observations, results):
        if a is None:
            continue
        assert isinstance(a, list) and len(a) == 3
        slot, x, y = a
        assert type(slot) is int and 0 <= slot <= 3
        assert obs["legal"][slot]
        assert isinstance(x, float) and 2.0 <= x <= 16.0
        assert isinstance(y, float) and 9.0 <= y <= 14.0
    assert any(a is None for a in results)
    assert any(a is not None for a in results)


def test_random_policy_deterministic(observations):
    a, b = RandomPolicy(seed=123), RandomPolicy(seed=123)
    a.reset(DECK)
    b.reset(DECK)
    assert [a.act(o) for o in observations] == [b.act(o) for o in observations]


def test_random_policy_act_index_not_yet():
    with pytest.raises(NotImplementedError):
        RandomPolicy(seed=1).act_index({})


def test_policy_imports_without_placements():
    mod = importlib.import_module("bot.policy")
    assert hasattr(mod, "Policy")
    if os.path.exists(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "bot", "placements.py")):
        pytest.skip("bot/placements.py has landed; absence check no longer applies")
    assert importlib.util.find_spec("bot.placements") is None

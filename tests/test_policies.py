import os
import sys

import pytest

# tests/ has no conftest (shared folder rule), so put the repo root on the path here.
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

import cr_sim  # noqa: E402

from bot.policy import DECIDE_EVERY, RandomPolicy  # noqa: E402
from bot.scripted import ScriptedPolicy  # noqa: E402
from evals.decks import load_gauntlet  # noqa: E402
from evals.policies import make_policy, spec_slug  # noqa: E402


@pytest.fixture(scope="module")
def observations():
    """Seat-0 decision observations from one real local game driven by RandomPolicy."""
    deck0, deck1 = load_gauntlet()[:2]
    b = cr_sim.Battle()
    b.reset(11, deck0, deck1)
    pol = RandomPolicy(0)
    pol.reset(deck0)
    v0 = b.view(0)
    out = []
    while not b.done() and len(out) < 60:
        obs = v0.observe()
        a = None
        if obs["tick"] % DECIDE_EVERY == 0:
            out.append(obs)
            a = pol.act(obs)
        b.step(tuple(a) if a is not None else None, None)
    assert len(out) >= 50
    return deck0, out


def test_builds_expected_classes():
    assert type(make_policy("random", seed=1)) is RandomPolicy
    assert type(make_policy("class:bot.policy.RandomPolicy", seed=1)) is RandomPolicy
    assert type(make_policy("scripted")) is ScriptedPolicy
    assert type(make_policy("class:bot.scripted.ScriptedPolicy", seed=3)) is ScriptedPolicy  # no seed arg


@pytest.mark.parametrize("spec", ["random", "class:bot.policy.RandomPolicy", "scripted"])
def test_same_seed_same_actions(observations, spec):
    deck, obs = observations
    p, q = make_policy(spec, seed=42), make_policy(spec, seed=42)
    p.reset(deck)
    q.reset(deck)
    assert [p.act(o) for o in obs] == [q.act(o) for o in obs]


def test_seed_matters_for_random(observations):
    deck, obs = observations
    p, q = make_policy("random", seed=1), make_policy("random", seed=2)
    p.reset(deck)
    q.reset(deck)
    assert [p.act(o) for o in obs] != [q.act(o) for o in obs]


@pytest.mark.parametrize("spec", ["foo", "", "learned:", "class:", "class:NoDot", "class:nope_mod.Nope"])
def test_unknown_spec(spec):
    with pytest.raises(ValueError) as e:
        make_policy(spec)
    if spec in ("foo", ""):
        assert "random" in str(e.value) and "class:" in str(e.value)


@pytest.mark.skipif(os.path.exists(os.path.join(REPO, "bot", "learned.py")), reason="bot/learned.py exists")
def test_learned_missing_is_clear():
    with pytest.raises(ValueError, match=r"bot/learned\.py.*doesn't exist yet"):
        make_policy("learned:x.pt")


def test_spec_slug():
    assert spec_slug("learned:models/bc_v1.pt") == "learned-models-bc_v1"
    assert spec_slug("class:bot.scripted_v0.ScriptedPolicyV0") == "class-bot.scripted_v0.ScriptedPolicyV0"
    assert spec_slug("random") == "random"
    assert spec_slug("learned:models\\a b.pt") == "learned-models-a-b"
    for s in ("learned:C:/x/y.pt", 'class:a.B<>|?*"'):
        assert not any(ch in spec_slug(s) for ch in ':/\\<>"|?* ')

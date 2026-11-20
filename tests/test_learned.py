import json
import os
import sys
import time

import pytest

# tests/ has no conftest (shared folder rule), so put the repo root on the path here.
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

torch = pytest.importorskip("torch")

import cr_sim  # noqa: E402

from bot import learned, model  # noqa: E402
from bot.learned import LearnedPolicy  # noqa: E402
from bot.policy import RandomPolicy  # noqa: E402
from evals.decks import load_gauntlet  # noqa: E402
from evals.match import play  # noqa: E402
from evals.policies import make_policy  # noqa: E402

BC_V1 = os.path.join(ROOT, "models", "bc_v1.pt")
with open(os.path.join(ROOT, "bot", "decks", "default.json"), encoding="utf-8") as _f:
    DECK = json.load(_f)


@pytest.fixture(scope="module")
def fresh_path(tmp_path_factory):
    torch.manual_seed(0)
    p = tmp_path_factory.mktemp("learned") / "fresh.pt"
    model.Net().save(p)
    return str(p)


def test_plays_two_games(fresh_path):
    d = load_gauntlet()
    pol = LearnedPolicy(fresh_path)
    for g in range(2):
        r = play(pol, RandomPolicy(g), g, d[g], d[g + 1])
        assert r["result"] in ("win", "loss", "draw")
    assert pol.mask_fallbacks == 0
    assert pol.decisions > 100


def test_cache_hit(fresh_path):
    LearnedPolicy(fresh_path)
    loads = learned.LOAD_STATS["loads"]
    hits = learned.LOAD_STATS["hits"]
    p2 = LearnedPolicy(fresh_path)
    assert learned.LOAD_STATS["loads"] == loads
    assert learned.LOAD_STATS["hits"] == hits + 1
    assert p2.net is learned._NETS[os.path.abspath(fresh_path)]


def test_refuses_wrong_feat_dim(fresh_path, tmp_path):
    bad = tmp_path / "bad.pt"
    ck = torch.load(fresh_path, weights_only=True)
    ck["FEAT_DIM"] = model.FEAT_DIM + 1
    torch.save(ck, bad)
    with pytest.raises(ValueError):
        LearnedPolicy(str(bad))


def test_act_returns_none_or_action(fresh_path):
    d = load_gauntlet()
    b = cr_sim.Battle()
    b.reset(1, d[0], d[1])
    pol = LearnedPolicy(fresh_path, temperature=1.0, seed=3)
    pol.reset(d[0])
    v = b.view(0)
    for t in range(600):
        if t % 10 == 0:
            a = pol.act(v.observe())
            assert a is None or (len(a) == 3 and type(a[0]) is int
                                 and type(a[1]) is float and type(a[2]) is float)
        b.step(None, None)


def test_guardrails_flag_without_module(fresh_path):
    pol = LearnedPolicy(fresh_path)
    assert pol.guardrails_active == os.path.exists(os.path.join(ROOT, "bot", "guardrails.py"))


@pytest.mark.skipif(not os.path.exists(BC_V1), reason="models/bc_v1.pt not trained yet")
def test_bc_v1_loads_fast_and_plays():
    learned._NETS.pop(os.path.abspath(BC_V1), None)
    t0 = time.perf_counter()
    pol = make_policy(f"learned:{BC_V1}")
    dt = time.perf_counter() - t0
    print(f"\nbc_v1 first load {dt:.2f}s")
    assert dt < 2.0
    d = load_gauntlet()
    r = play(pol, RandomPolicy(0), 0, DECK, d[1])
    assert r["result"] in ("win", "loss", "draw")
    assert pol.mask_fallbacks == 0
    assert os.path.getsize(BC_V1) < 10_000_000

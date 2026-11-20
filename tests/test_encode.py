import math
import os
import sys

import numpy as np

# tests/ has no conftest (shared folder rule), so put the repo root on the path here.
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import cr_sim  # noqa: E402

from bot import cards, encode as enc, placements  # noqa: E402
from bot.policy import DECIDE_EVERY, RandomPolicy  # noqa: E402
from bot.trackers import Trackers  # noqa: E402
from evals.decks import load_gauntlet  # noqa: E402
from evals.match import to_step  # noqa: E402

N_OBS = 200


class StubTrackers:
    """Only the two calls the encoder may use (W9)."""

    def enemy_elixir(self):
        return 7.5

    def enemy_cards(self):
        return [("knight", True), ("fireball", False), ("not_a_card", True)]


def _random_obs(n=N_OBS, every=3):
    """n (obs, trackers-after-update) pairs from RandomPolicy games, sampled every `every` decisions."""
    decks = load_gauntlet()
    out = []
    g = 0
    while len(out) < n:
        b = cr_sim.Battle()
        b.reset(g, decks[g % len(decks)], decks[(g + 1) % len(decks)])
        views = (b.view(0), b.view(1))
        pols = (RandomPolicy(2 * g), RandomPolicy(2 * g + 1))
        tr = Trackers()
        k = 0
        while not b.done() and len(out) < n:
            step = [None, None]
            if b.tick % DECIDE_EVERY == 0:
                o0 = views[0].observe()
                tr.update(o0)
                if k % every == 0:
                    out.append((o0, enc.encode(o0, tr)))
                k += 1
                step = [to_step(pols[0].act(o0)), to_step(pols[1].act(views[1].observe()))]
            b.step(step[0], step[1])
        g += 1
    return out


SAMPLES = _random_obs()


def test_constants():
    assert enc.FEAT_DIM == cards.FEAT_DIM
    assert enc.N_ACTIONS == placements.N_ACTIONS
    assert enc.HAND_DIM == cards.FEAT_DIM + 2
    assert enc.UNIT_DIM == cards.FEAT_DIM + 7
    assert enc.ENEMY_DIM == cards.FEAT_DIM + 1
    print(f"\nFEAT_DIM {enc.FEAT_DIM}; shapes {enc.SHAPES}")


def test_shapes_fixed_finite_bounded():
    for _, x in SAMPLES:
        assert set(x) == set(enc.KEYS)
        for k, v in x.items():
            assert v.dtype == np.float32, k
            assert v.shape == enc.SHAPES[k], (k, v.shape)
            assert np.isfinite(v).all(), k
            assert v.min() >= -1.0 and v.max() <= 2.0, (k, v.min(), v.max())


def test_units_sorted_masked_and_mask_matches():
    saw_units = False
    for o, x in SAMPLES:
        n = int(x["units_mask"].sum())
        assert n == min(len(o["entities"]), enc.N_UNITS)
        assert (x["units_mask"][:n] == 1).all() and (x["units_mask"][n:] == 0).all()
        assert (x["units"][n:] == 0).all()
        xs = x["units"][:n, enc.FEAT_DIM + 1] * 18
        ys = x["units"][:n, enc.FEAT_DIM + 2] * 32
        d = np.hypot(xs - 9.0, ys - 3.0)
        assert (np.diff(d) >= -1e-4).all()
        saw_units |= n > 6
        assert (x["mask"] == np.asarray(placements.mask(o), dtype=np.float32)).all()
    assert saw_units


def test_globals_content():
    o, x = SAMPLES[0]
    g = x["globals"]
    assert g[0] == np.float32(o["elixir"] / 10)
    assert g[3:6].sum() == 1.0
    assert 0.0 <= g[6] <= 1.0
    assert (g[9:15] == 1.0).all()  # tick 0: all towers full


def test_stub_trackers_and_list_entities():
    o, _ = SAMPLES[-1]
    o = dict(o)
    o["entities"] = [list(e) for e in o["entities"]]
    x = enc.encode(o, StubTrackers())
    for k, v in x.items():
        assert v.shape == enc.SHAPES[k] and np.isfinite(v).all()
    assert x["globals"][1] == np.float32(0.75)
    assert list(x["enemy_deck_mask"]) == [1, 1, 1, 0, 0, 0, 0, 0]
    assert (x["enemy_deck"][0, : enc.FEAT_DIM] == cards.feat("knight")).all()
    assert x["enemy_deck"][0, enc.FEAT_DIM] == 1.0 and x["enemy_deck"][1, enc.FEAT_DIM] == 0.0
    assert (x["enemy_deck"][2, : enc.FEAT_DIM] == 0).all()  # unknown name -> zeros
    assert x["globals"][15] == np.float32(3 / 8)
    tup = enc.encode(SAMPLES[-1][0], StubTrackers())
    for k in enc.KEYS:
        assert (tup[k] == x[k]).all()


def test_next_card_row():
    o, x = SAMPLES[5]
    row = x["hand"][4]
    assert (row[: enc.FEAT_DIM] == cards.feat(o["next_card"])).all()
    assert row[enc.FEAT_DIM] == 0.0
    assert math.isclose(row[enc.FEAT_DIM + 1], cards.elixir(o["next_card"]) / 10, rel_tol=1e-6)


def test_batch_encode():
    b = enc.batch_encode([x for _, x in SAMPLES[:7]])
    for k in enc.KEYS:
        assert b[k].shape == (7,) + enc.SHAPES[k] and b[k].dtype == np.float32


def test_encode_does_not_import_torch():
    import subprocess

    code = "import sys; import bot.encode; assert 'torch' not in sys.modules"
    subprocess.run([sys.executable, "-c", code], cwd=ROOT, check=True)

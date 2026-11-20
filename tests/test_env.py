import os
import random
import sys

import numpy as np
import pytest

# tests/ has no conftest (shared folder rule), so put the repo root on the path here.
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from bot import encode as enc  # noqa: E402
from bot.env import FixedOpponent, VecEnv, _Shard, random_actions, tower_hp  # noqa: E402
from evals.decks import load_heldout_cards  # noqa: E402

DEFAULT = ["knight", "archers", "giant", "musketeer", "valkyrie", "minions", "wizard", "minipekka"]


def _run(workers, steps=30, n=4, seed=5, opponent="scripted"):
    env = VecEnv(n=n, workers=workers, seed=seed, opponent_sampler=FixedOpponent(opponent))
    try:
        rng = np.random.default_rng(1)
        obs = env.reset()
        trace = [obs]
        for _ in range(steps):
            obs, r, d, infos = env.step(random_actions(obs, rng))
            trace.append((obs, r, d))
        return trace
    finally:
        env.close()


def test_same_results_in_process_and_two_workers():
    a = _run(0)
    b = _run(2)
    for k in enc.KEYS:
        assert np.array_equal(a[0][k], b[0][k])
    for (oa, ra, da), (ob, rb, db) in zip(a[1:], b[1:]):
        assert np.array_equal(ra, rb) and np.array_equal(da, db)
        for k in enc.KEYS:
            assert np.array_equal(oa[k], ob[k]), k


def test_obs_shapes_match_encode():
    env = VecEnv(n=3, workers=0, seed=0)
    obs = env.reset()
    for k in enc.KEYS:
        assert obs[k].shape == (3,) + enc.SHAPES[k] and obs[k].dtype == np.float32
    obs, r, d, infos = env.step(np.zeros(3, dtype=np.int64))
    assert r.shape == (3,) and d.shape == (3,) and r.dtype == np.float32
    env.close()


def test_episode_end_reward_and_auto_reset():
    """Play until every env has finished an episode; check terminal reward and the reset."""
    env = VecEnv(n=3, workers=0, seed=11, opponent_sampler=FixedOpponent("scripted"), shape=0.0)
    obs = env.reset()
    sh = env._local
    first_seeds = [ep["seed"] for ep in sh.ep]
    seen = set()
    rng = np.random.default_rng(0)
    for _ in range(700):  # 6,000 ticks max per game = 600 steps
        obs, r, d, infos = env.step(random_actions(obs, rng))
        for info in infos:
            j = info["env"]
            assert d[j]
            assert r[j] == {"win": 1.0, "loss": -1.0, "draw": 0.0}[info["result"]]  # shape 0: pure outcome
            assert info["opponent"] == "scripted" and info["deck_kind"] in ("default", "random")
            assert sh.ep[j]["seed"] != first_seeds[j] or j in seen  # a new episode started
            assert sh.battles[j].tick <= 1 and not sh.battles[j].done()
            seen.add(j)
        assert (d == np.isin(np.arange(3), [i["env"] for i in infos])).all()
        if len(seen) == 3:
            break
    assert seen == {0, 1, 2}
    env.close()


def test_shaping_is_tower_hp_difference():
    sh = _Shard([0], 0, FixedOpponent("random"), DEFAULT, 1.0, None, 0.0, 2.0)
    sh.reset()
    obs = sh.obs[0]
    ours, theirs = tower_hp(obs)
    assert ours > 0 and theirs > 0
    sh.hp[0] = (ours, theirs + 1000.0)  # pretend the enemy towers had 1,000 more hp
    _, r, d, _ = sh.step([0])
    o2, t2 = tower_hp(sh.obs[0])
    expect = 2.0 * ((theirs + 1000.0 - t2) - (ours - o2)) / 10_000
    assert not d[0] and r[0] == pytest.approx(expect, abs=1e-6)


def test_no_heldout_cards_in_200_resets():
    held = set(load_heldout_cards())
    sh = _Shard([0], 3, FixedOpponent("random"), DEFAULT, 0.4, [DEFAULT[::-1]], 0.3, 1.0)
    rng = random.Random(0)
    kinds = set()
    for _ in range(200):
        d0, kind, d1 = sh.pick_decks(rng)
        kinds.add(kind)
        assert not (set(d0) | set(d1)) & held
        assert len(d0) == 8 and len(d1) == 8
    assert kinds == {"default", "list", "random"}
    env = VecEnv(n=2, workers=0, seed=4)
    env.reset()
    for _ in range(20):
        env._local._new_episode(0)
        ep = env._local.ep[0]
        assert not (set(ep["deck0"]) | set(ep["deck1"])) & held
    env.close()


def test_close_leaves_no_children():
    import multiprocessing as mp

    env = VecEnv(n=4, workers=2, seed=0)
    env.reset()
    procs = list(env._procs)
    assert all(p.is_alive() for p in procs)
    env.close()
    assert not any(p.is_alive() for p in procs)
    assert not [c for c in mp.active_children() if c.name.startswith("vecenv-")]


def test_set_shape_and_snapshots_reach_workers():
    env = VecEnv(n=2, workers=0, seed=0)
    env.set_shape(0.3)
    assert env._local.shape == 0.3
    env.set_snapshots(["a.pt"])  # FixedOpponent ignores it; no error
    env.close()

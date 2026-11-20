import csv
import os
import random
import shutil
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pytest

# tests/ has no conftest (shared folder rule), so put the repo root on the path here.
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

torch = pytest.importorskip("torch")

from bot import model  # noqa: E402
from bot.learned import LearnedPolicy  # noqa: E402
from train import ppo  # noqa: E402
from train.league import KINDS, League  # noqa: E402

RUN = "pytest_tiny"
RUN_DIR = Path(ROOT) / "runs" / RUN


@pytest.fixture(scope="module")
def tiny_run():
    shutil.rmtree(RUN_DIR, ignore_errors=True)
    common = ["--run", RUN, "--envs", "8", "--steps", "16", "--device", "cpu", "--workers", "2",
              "--snap-every", "3", "--log-every", "2", "--warmup", "2", "--threads", "2"]
    s1 = ppo.main(common + ["--minutes", "0.15"])
    rows1 = list(csv.DictReader((RUN_DIR / "log.csv").open()))
    s2 = ppo.main(common + ["--minutes", "0.1", "--resume", str(RUN_DIR / "last.pt")])
    yield s1, s2, rows1
    shutil.rmtree(RUN_DIR, ignore_errors=True)


def test_tiny_run_writes_log_last_and_snapshots(tiny_run):
    s1, _, rows1 = tiny_run
    assert s1["updates"] >= 3
    assert len(rows1) == s1["updates"]
    assert set(ppo.LOG_FIELDS) == set(rows1[0])
    for r in rows1:  # win-rate columns are filled exactly on --log-every updates
        assert (r["n_scripted"] != "") == (int(r["update"]) % 2 == 0)
    assert rows1[0]["warmup"] == "1"
    snaps = sorted(RUN_DIR.glob("snap_*.pt"))
    assert snaps
    pol = LearnedPolicy(str(snaps[0]))  # snapshots load through LearnedPolicy
    assert pol.net is not None


def test_resume_continues(tiny_run):
    s1, s2, _ = tiny_run
    assert s2["updates"] > s1["updates"]
    assert s2["session_updates"] == s2["updates"] - s1["updates"]
    rows = list(csv.DictReader((RUN_DIR / "log.csv").open()))
    ups = [int(r["update"]) for r in rows]
    assert ups == list(range(1, s2["updates"] + 1))  # appended, header once
    ck = torch.load(RUN_DIR / "last.pt", weights_only=False)
    assert ck["update"] == s2["updates"]
    assert len(ck["league"]["snapshots"]) >= len(s1["snapshots"])


def test_league_samples_all_kinds(tmp_path):
    lg = League(tmp_path)
    rng = random.Random(0)
    before = Counter(lg.kind_of(lg.sample(rng)) for _ in range(2000))
    assert set(before) == {"scripted", "random", "bc"}  # no snapshot yet -> its share goes to bc
    assert 0.55 < before["bc"] / 2000 < 0.65
    lg.add_snapshot(tmp_path / "snap_50.pt")
    lg.add_snapshot(tmp_path / "snap_100.pt")
    specs = [lg.sample(rng) for _ in range(4000)]
    kinds = Counter(lg.kind_of(s) for s in specs)
    assert set(kinds) == set(KINDS)
    for k, share in (("snapshot", 0.5), ("scripted", 0.3), ("random", 0.1), ("bc", 0.1)):
        assert abs(kinds[k] / 4000 - share) < 0.03, (k, kinds[k])
    snaps = Counter(s for s in specs if "snap_" in s)
    assert len(snaps) == 2  # uniform over all saved, not only the latest
    import pickle

    lg2 = pickle.loads(pickle.dumps(lg))
    assert lg2.snapshots == lg.snapshots
    lg3 = League.from_state(lg.state())
    assert lg3.snapshots == lg.snapshots


def test_gae_handles_done():
    rew = np.array([[0.0], [1.0], [0.0]], dtype=np.float32)
    done = np.array([[0.0], [1.0], [0.0]], dtype=np.float32)
    val = np.array([[0.5], [0.5], [0.5]], dtype=np.float32)
    adv, ret = ppo.compute_gae(rew, done, val, np.array([2.0], dtype=np.float32), 0.9, 1.0)
    assert ret[1, 0] == pytest.approx(1.0)  # terminal: no bootstrap
    assert ret[2, 0] == pytest.approx(0.0 + 0.9 * 2.0)
    assert ret[0, 0] == pytest.approx(0.0 + 0.9 * 1.0)


def test_snapshot_is_model_format(tmp_path):
    net = model.Net()
    p = tmp_path / "snap_1.pt"
    ppo.save_snapshot(net, p)
    assert isinstance(model.load(p), model.Net)
    assert not list(tmp_path.glob("*.tmp"))

import json
import os
import sys

import numpy as np
import pytest

# tests/ has no conftest (shared folder rule), so put the repo root on the path here.
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

pytest.importorskip("torch")

from bot import encode as enc, model  # noqa: E402
from train import bc_record, bc_train  # noqa: E402


@pytest.fixture(scope="module")
def recorded(tmp_path_factory):
    out = tmp_path_factory.mktemp("bc")
    meta = bc_record.main(["--games", "4", "--games-per-shard", "2", "--workers", "1", "--out", str(out)])
    return out, meta


def test_record_shards(recorded):
    out, meta = recorded
    shards = sorted(out.glob("shard_*.npz"))
    assert len(shards) == 2
    total = waits = 0
    for f in shards:
        with np.load(f) as z:
            assert set(z.files) == set(enc.KEYS) | {"label"}
            n = len(z["label"])
            assert n > 0 and z["label"].dtype == np.int16
            for k in enc.KEYS:
                assert z[k].shape == (n,) + enc.SHAPES[k], k
                assert z[k].dtype == (np.uint8 if k in bc_record.MASK_KEYS else np.float16), k
            lab = z["label"].astype(np.int64)
            assert (z["mask"][np.arange(n), lab] == 1).all()  # every label legal under its mask
            assert ((lab >= 0) & (lab < enc.N_ACTIONS)).all()
            total += n
            waits += int((lab == 0).sum())
    print(f"\ntiny recording: {total} decisions, wait share {waits / total:.1%}")
    assert meta["decisions_stored"] == total and meta["waits_stored"] == waits
    m = json.loads((out / "meta.json").read_text())
    assert m["games"] == 4 and m["scripted_version"] == bc_record.SCRIPTED_VERSION


def test_game_mix_deterministic():
    from evals.decks import load_heldout_cards

    held = load_heldout_cards()
    deck = ["knight", "archers", "giant", "musketeer", "valkyrie", "minions", "wizard", "minipekka"]
    a = [bc_record.game_setup(i, 0, held, deck) for i in range(200)]
    assert a == [bc_record.game_setup(i, 0, held, deck) for i in range(200)]
    modes = [m for _, m, *_ in a]
    assert 60 < modes.count("scripted") < 140
    for _, _, d0, d1, _ in a:
        assert not (set(d0) | set(d1)) & set(held) or d0 == deck or d1 == deck


def test_tiny_training_saves_loadable_model(recorded, tmp_path):
    out, _ = recorded
    path = tmp_path / "bc_tiny.pt"
    info = bc_train.main(["--data", str(out), "--out", str(path), "--epochs", "1", "--device", "cpu"])
    net = model.load(path)
    assert isinstance(net, model.Net)
    assert info["best_epoch"] == 1 and len(info["holdout_files"]) == 1
    import torch

    assert "trained_on" in torch.load(path, weights_only=False)

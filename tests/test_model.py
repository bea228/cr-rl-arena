import os
import statistics
import sys
import time

import numpy as np
import pytest

# tests/ has no conftest (shared folder rule), so put the repo root on the path here.
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

torch = pytest.importorskip("torch")

import cr_sim  # noqa: E402

from bot import encode as enc, model  # noqa: E402
from bot.trackers import Trackers  # noqa: E402
from evals.decks import load_gauntlet  # noqa: E402


def _encoded(n=64):
    """n real encodings from one game (random-ish board via fixed actions)."""
    d = load_gauntlet()
    b = cr_sim.Battle()
    b.reset(3, d[0], d[1])
    v = b.view(0)
    tr = Trackers()
    out = []
    while len(out) < n and not b.done():
        if b.tick % 10 == 0:
            o = v.observe()
            tr.update(o)
            out.append(enc.encode(o, tr))
            act = (0, 3.5, 10.5) if o["legal"][0] else None
            b.step(act, (1, 14.5, 10.5))
        else:
            b.step(None, None)
    return out


ENC = _encoded()


@pytest.fixture(scope="module")
def net():
    torch.manual_seed(0)
    return model.Net().eval()


def test_param_count(net):
    n = net.n_params()
    print(f"\nparams {n:,}")
    assert n < 1_000_000


def test_cpu_forward_under_5ms(net):
    torch.set_num_threads(1)
    x = model.to_torch(ENC[10])
    times = []
    with torch.no_grad():
        for _ in range(10):
            net(x)
        for _ in range(50):
            t0 = time.perf_counter()
            net(x)
            times.append(time.perf_counter() - t0)
    med = statistics.median(times) * 1000
    print(f"\nCPU forward (batch 1, 1 thread) median {med:.2f} ms")
    assert med < 5.0


def test_shapes_and_masking(net):
    x = model.to_torch(enc.batch_encode(ENC))
    with torch.no_grad():
        logits, value = net(x)
    assert logits.shape == (64, enc.N_ACTIONS) and value.shape == (64,)
    masked = x["mask"] == 0
    assert masked.any()
    assert (logits[masked] == model.MASKED_LOGIT).all()
    assert (logits[~masked] > -1e8).all()
    assert torch.isfinite(value).all()


def test_single_obs_and_empty_sides(net):
    x = model.to_torch(ENC[0])
    x["units_mask"] = torch.zeros_like(x["units_mask"])  # no units at all
    x["enemy_deck_mask"] = torch.zeros_like(x["enemy_deck_mask"])
    with torch.no_grad():
        logits, value = net(x)
    assert logits.shape == (1, enc.N_ACTIONS)
    assert torch.isfinite(value).all() and not torch.isnan(logits).any()


def test_save_load_roundtrip(net, tmp_path):
    p = tmp_path / "net.pt"
    net.save(p)
    net2 = model.load(p).eval()
    x = model.to_torch(enc.batch_encode(ENC[:8]))
    with torch.no_grad():
        a, va = net(x)
        b, vb = net2(x)
    assert torch.equal(a, b) and torch.equal(va, vb)


def test_load_rejects_wrong_feat_dim(net, tmp_path):
    p = tmp_path / "bad.pt"
    torch.save({"state_dict": net.state_dict(), "FEAT_DIM": model.FEAT_DIM + 1,
                "N_ACTIONS": model.N_ACTIONS, "config": net.config}, p)
    with pytest.raises(ValueError):
        model.load(p)

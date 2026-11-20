import csv
import os
import sys

import pytest

# tests/ has no conftest (shared folder rule), so put the repo root on the path here.
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

torch = pytest.importorskip("torch")

from bot import model  # noqa: E402
from train import eval_snaps  # noqa: E402


def test_eval_snaps_appends_only_new(tmp_path, capsys):
    torch.manual_seed(0)
    bc = tmp_path / "bc.pt"
    model.Net().save(bc)
    for u in (10, 2):
        model.Net().save(tmp_path / f"snap_{u}.pt")
    args = [str(tmp_path), "--games", "2", "--workers", "1", "--bc", str(bc)]
    table = eval_snaps.main(args)
    assert [r["model"] for r in table] == ["bc_v1", "snap_2.pt", "snap_10.pt"]  # bc first, then by update
    for r in table:
        assert 0.0 <= float(r["vs_random"]) <= 1.0 and 0.0 <= float(r["vs_scripted"]) <= 1.0
    assert "vs scripted" in capsys.readouterr().out

    model.Net().save(tmp_path / "snap_20.pt")
    eval_snaps.main(args)
    rows = list(csv.DictReader((tmp_path / "evals.csv").open()))
    assert [r["model"] for r in rows] == ["bc_v1", "snap_2.pt", "snap_10.pt", "snap_20.pt"]  # nothing re-run

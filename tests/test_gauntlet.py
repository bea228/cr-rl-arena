import copy
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evals import gauntlet  # noqa: E402
from evals.replay import rerun  # noqa: E402


def _comparable(s):
    s = copy.deepcopy(s)
    s.pop("wall_time_s")
    s["args"].pop("workers")  # an input, not a result
    return s


def test_workers_do_not_change_results():
    s1 = gauntlet.run_gauntlet("random", "random", games=20, workers=1, save=False)
    s2 = gauntlet.run_gauntlet("random", "random", games=20, workers=2, save=False)
    assert _comparable(s1) == _comparable(s2)
    assert s1["games"] == 20
    assert s1["a_seat0_games"] == 10
    assert abs(s1["win_rate"] + s1["draw_rate"] + s1["loss_rate"] - 1) < 1e-9
    assert sum(d["games"] for d in s1["per_deck"].values()) == 20
    assert 0 <= s1["ignored_rate_a"] <= 1 and 0 <= s1["ignored_rate_b"] <= 1


def test_save_and_replays_rerun(tmp_path, monkeypatch):
    monkeypatch.setattr(gauntlet, "RESULTS_DIR", tmp_path)
    stale = tmp_path / "replays" / "random_vs_random" / "loss_99_g0.json"
    stale.parent.mkdir(parents=True)
    stale.write_text("{}")

    s = gauntlet.run_gauntlet("random", "random", games=20, workers=1, save=True)
    out = tmp_path / "random_vs_random.json"
    assert out.exists() and Path(s["path"]) == out
    with open(out, encoding="utf-8") as f:
        on_disk = json.load(f)
    assert on_disk["win_rate"] == s["win_rate"]
    assert on_disk["git_commit"]
    assert not stale.exists()

    worst = s["worst_losses"]
    assert worst, "random vs random over 20 games should produce at least one loss for a"
    assert len(worst) <= 10
    assert [w["crown_diff"] for w in worst] == sorted(w["crown_diff"] for w in worst)
    assert sorted(p.name for p in stale.parent.glob("loss_*.json")) == sorted(Path(w["path"]).name for w in worst)
    for w in worst:
        r = rerun(w["path"])
        assert r["result"] == w["seat0_result"]
        assert r["crowns"] == w["crowns"]
        a, b = w["a_seat"], 1 - w["a_seat"]
        assert r["crowns"][a] - r["crowns"][b] == w["crown_diff"] <= 0


def test_a_deck_and_stem(tmp_path, monkeypatch):
    monkeypatch.setattr(gauntlet, "RESULTS_DIR", tmp_path)
    deck_file = Path(gauntlet.REPO) / "bot" / "decks" / "default.json"
    s = gauntlet.run_gauntlet("random", "random", games=4, workers=1, a_deck=str(deck_file))
    assert s["stem"] == "random_vs_random_on_default"
    assert s["per_deck_group_by"] == "b_deck"
    assert set(s["per_deck"]) == {"1", "2", "3", "4"}
    assert (tmp_path / "random_vs_random_on_default.json").exists()

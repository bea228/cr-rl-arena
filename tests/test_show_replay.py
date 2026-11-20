import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from bot.policy import RandomPolicy  # noqa: E402
from evals.decks import load_gauntlet  # noqa: E402
from evals.match import play  # noqa: E402
from evals.replay import save_match  # noqa: E402


def test_show_replay_runs(tmp_path):
    decks = load_gauntlet()
    m = play(RandomPolicy(1), RandomPolicy(2), 7, decks[0], decks[1])
    path = tmp_path / "loss_1_g0.json"
    save_match(path, m)
    script = os.path.join(ROOT, "scripts", "show_replay.py")
    r = subprocess.run([sys.executable, script, str(path), "--seat", "0"],
                       capture_output=True, text=True, encoding="utf-8", timeout=120)
    assert r.returncode == 0, r.stderr
    assert f"result {m['result']}" in r.stdout
    assert " us " in r.stdout or " them " in r.stdout
    s = subprocess.run([sys.executable, script, "--summary", str(tmp_path), "--seat", "0"],
                       capture_output=True, text=True, encoding="utf-8", timeout=120)
    assert s.returncode == 0, s.stderr
    assert m["result"] in s.stdout

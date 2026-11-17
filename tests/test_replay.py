import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bot.policy import RandomPolicy  # noqa: E402
from evals.decks import load_gauntlet  # noqa: E402
from evals.match import play  # noqa: E402
from evals.replay import rerun, save_match  # noqa: E402


@pytest.mark.parametrize("seed", range(20))
def test_replay_matches(seed, tmp_path):
    d = load_gauntlet()
    m = play(RandomPolicy(seed), RandomPolicy(seed + 1000), seed, d[seed], d[seed + 1])
    path = tmp_path / "sub" / f"m{seed}.json"
    save_match(path, m)
    r = rerun(path)
    assert r == {"result": m["result"], "crowns": m["crowns"], "ticks": m["ticks"]}

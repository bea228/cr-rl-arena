import os
import sys

# tests/ has no conftest (shared folder rule), so put the repo root on the path here.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cr_sim  # noqa: E402

from bot import state  # noqa: E402
from bot.policy import DECIDE_EVERY, RandomPolicy  # noqa: E402

DECK = ["knight", "archers", "giant", "musketeer", "valkyrie", "minions", "wizard", "minipekka"]


def fresh_obs(seed=1, seat=0):
    b = cr_sim.Battle()
    b.reset(seed, DECK, "default")
    return b.view(seat).observe()


def test_fresh_battle_towers_and_threat():
    for seat in (0, 1):
        s = state.parse(fresh_obs(seat=seat))
        for d in (s.my_tower_hp, s.enemy_tower_hp, s.my_tower_max, s.enemy_tower_max):
            assert set(d) == {"L", "R", "K"}
            assert all(v > 0 for v in d.values())
        assert s.my_tower_hp == s.my_tower_max
        assert s.enemy_tower_hp == s.enemy_tower_max
        assert s.threat == {"L": 0, "R": 0}
        assert s.defense == {"L": 0, "R": 0}
        assert s.mine == [] and s.enemy == []


def test_scalar_fields():
    obs = fresh_obs()
    s = state.parse(obs)
    assert s.tick == 0 and s.phase == "single"
    assert s.elixir == obs["elixir"]
    assert s.hand == list(obs["hand"]) and len(s.hand) == 4
    assert s.cost == list(obs["hand_cost"])
    assert s.legal == list(obs["legal"])


def test_lane_of():
    assert state.lane_of(0.5) == "L"
    assert state.lane_of(8.99) == "L"
    assert state.lane_of(9.0) == "R"
    assert state.lane_of(17.5) == "R"


def test_list_entities_and_destroyed_tower():
    """Server obs carries lists, and a destroyed tower is simply missing."""
    obs = fresh_obs()
    ents = [list(e) for e in obs["entities"]]
    # drop the enemy left princess tower (x 3.5, y 25.5)
    ents = [e for e in ents if not (e[0] == 1 and e[1] == "princess_tower" and e[2] < 9)]
    s = state.parse({**obs, "entities": ents})
    assert s.enemy_tower_hp["L"] == 0 and s.enemy_tower_max["L"] == 0
    assert s.enemy_tower_hp["R"] > 0 and s.enemy_tower_hp["K"] > 0


def test_threat_and_defense_from_units():
    obs = fresh_obs()
    base = [list(e) for e in obs["entities"]]
    #          owner card       x    y    hp   max  fly    tower  dep    id  hw   hh
    extra = [
        [1, "knight", 4.0, 12.0, 1000, 1400, False, False, False, 90, 0.5, 0.5],  # threat L
        [1, "giant", 12.0, 16.5, 3000, 3000, False, False, True, 91, 0.5, 0.5],   # threat R (deploying)
        [1, "archers", 12.0, 20.0, 300, 300, False, False, False, 92, 0.5, 0.5],  # y >= 17: no threat
        [0, "minions", 15.0, 10.0, 200, 200, True, False, False, 93, 0.5, 0.5],   # defense R
        [0, "knight", 3.0, 18.0, 1400, 1400, False, False, False, 94, 0.5, 0.5],  # y >= 17: no defense
    ]
    s = state.parse({**obs, "entities": base + extra})
    assert s.threat == {"L": 1000, "R": 3000}
    assert s.defense == {"L": 0, "R": 200}
    assert [u.id for u in s.enemy] == [90, 91, 92]
    assert [u.id for u in s.mine] == [93, 94]
    g = s.enemy[1]
    assert (g.name, g.lane, g.deploying, g.flying, g.max_hp) == ("giant", "R", True, False, 3000)
    assert s.mine[0].flying is True


def test_parse_never_raises_in_real_games():
    pol = RandomPolicy(seed=0)
    for game in range(3):
        b = cr_sim.Battle()
        b.reset(game + 1, DECK, "default")
        pol.reset(DECK)
        v = b.view(0)
        while not b.done():
            obs = v.observe()
            s = state.parse(obs)
            assert all(u.lane in ("L", "R") for u in s.mine + s.enemy)
            a = pol.act(obs) if obs["tick"] % DECIDE_EVERY == 0 else None
            b.step(tuple(a) if a is not None else None, None)
        state.parse(v.observe())  # final obs, towers may be gone

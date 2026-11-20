import json
import os
import sys

import pytest

# tests/ has no conftest (shared folder rule), so put the repo root on the path here.
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import cr_sim  # noqa: E402

from bot import placements, state  # noqa: E402
from bot.policy import DECIDE_EVERY, RandomPolicy  # noqa: E402
from bot.scripted import (  # noqa: E402
    ScriptedPolicy,
    attack_index,
    defend_index,
    noleak_index,
    opening_index,
    spell_index,
)
from evals.decks import load_gauntlet  # noqa: E402

with open(os.path.join(ROOT, "bot", "decks", "default.json"), encoding="utf-8") as _f:
    DECK = json.load(_f)

P = placements.NAMES
DEF_OR_KING = {P["king_pull"], P["def_L"], P["def_R"]}


def test_default_deck_file():
    assert len(DECK) == 8 and all(isinstance(c, str) for c in DECK)
    cr_sim.Battle().check_deck(list(DECK))  # raises ValueError if illegal


def test_rule_functions_importable():
    for fn in (defend_index, opening_index, spell_index, attack_index, noleak_index):
        assert callable(fn)


def test_every_index_is_masked_legal_over_10_games():
    bot = ScriptedPolicy()
    gauntlet = load_gauntlet()
    n_moves = 0
    for game in range(10):
        b = cr_sim.Battle()
        b.reset(game + 1, DECK, gauntlet[game])
        bot.reset(DECK)
        opp = RandomPolicy(game)
        v0, v1 = b.view(0), b.view(1)
        while not b.done():
            a0 = a1 = None
            if b.tick % DECIDE_EVERY == 0:
                obs = v0.observe()
                i = bot.act_index(obs)
                assert placements.mask(obs)[i], placements.mask_reason(obs, i)
                a0 = placements.decode(i, obs)
                n_moves += a0 is not None
                a1 = opp.act(v1.observe())
            b.step(tuple(a0) if a0 else None, tuple(a1) if a1 else None)
    assert bot.mask_fallbacks == 0
    assert n_moves > 0
    assert sum(bot.fires.values()) > 0


def test_opening_waits_at_tick_0():
    b = cr_sim.Battle()
    b.reset(1, DECK, "default")
    obs = b.view(0).observe()
    assert obs["tick"] == 0
    assert opening_index(obs, state.parse(obs)) == 0
    assert ScriptedPolicy().act_index(obs) == 0


def test_defend_against_real_giant():
    for seed in range(1, 300):
        b = cr_sim.Battle()
        b.reset(seed, DECK, DECK)
        hand1 = b.view(1).observe()["hand"]
        if "giant" in hand1:
            break
    else:
        pytest.fail("no seed puts a giant in seat 1's opening hand")
    # seat 1 plays the giant at bridge_L of its own frame; it walks down toward us
    b.step(None, (hand1.index("giant"), 3.5, 14.5))
    assert b.view(1).observe()["last_action"] == "ok"
    v0 = b.view(0)
    giant = None
    for _ in range(600):
        b.step(None, None)
        obs = v0.observe()
        giants = [e for e in obs["entities"] if e[0] == 1 and e[1] == "giant"]
        if giants and giants[0][3] < 17:
            giant = giants[0]
            break
    assert giant is not None, "enemy giant never crossed y < 17"
    st = state.parse(obs)
    lane = state.lane_of(giant[2])
    assert st.threat[lane] > 0
    i = defend_index(obs, st)
    assert i is not None
    slot, p = placements.split(i)
    assert p in DEF_OR_KING
    if p != P["king_pull"]:
        assert p == (P["def_L"] if lane == "L" else P["def_R"])
    assert obs["legal"][slot] and not placements.is_spell(obs["hand"][slot])
    assert placements.mask(obs)[i]


def test_noleak_at_full_elixir():
    b = cr_sim.Battle()
    b.reset(1, DECK, "default")
    v0 = b.view(0)
    obs = v0.observe()
    while obs["elixir"] < 10:
        b.step(None, None)
        obs = v0.observe()
    assert obs["tick"] > 200
    st = state.parse(obs)
    assert not st.enemy
    i = noleak_index(obs, st)
    assert i is not None
    slot, p = placements.split(i)
    # C4's back_ tiles strand troops behind our own princess tower in the sim, so the
    # "from behind" tile is def_ of the attack lane (see bot/scripted.py BEHIND).
    assert p in (P["def_L"], P["def_R"])
    assert not placements.is_spell(obs["hand"][slot])
    # attack (elixir >= 8, no threat) fires first in the full policy; either way a real move
    j = ScriptedPolicy().act_index(obs)
    assert j != 0 and placements.mask(obs)[j]


def _with(obs, extra, hand):
    return {**obs, "entities": [list(e) for e in obs["entities"]] + extra,
            "hand": hand, "hand_cost": [4, 3, 3, 3], "legal": [True] * 4}


def _unit(x, y, uid):
    return [1, "goblins", x, y, 200, 200, False, False, False, uid, 0.5, 0.5]


def test_spell_on_cluster_only_with_two_units():
    b = cr_sim.Battle()
    b.reset(1, DECK, "default")
    obs = b.view(0).observe()
    hand = ["fireball", "knight", "archers", "minions"]
    two = _with(obs, [_unit(9.0, 20.0, 90), _unit(10.0, 20.0, 91)], hand)
    i = spell_index(two, state.parse(two))
    assert i is not None and placements.split(i) == (0, P["cluster"])
    one = _with(obs, [_unit(9.0, 20.0, 90), _unit(16.0, 28.0, 91)], hand)
    assert spell_index(one, state.parse(one)) is None


GHOST_OPP = ["royal_ghost", "knight", "archers", "giant", "musketeer", "valkyrie", "minions", "wizard"]


def test_defend_ranged_vs_royal_ghost_despite_melee_cover():
    """J2 fix: a Royal Ghost is invisible except right after each hit, so melee units in the lane
    walk past it and it takes the tower alone. v0 counted our knight's hp as cover and stayed
    silent; now only ranged units count against it and the defender must be ranged."""
    from bot import scripted_v0
    from bot.scripted import ranged

    for seed in range(1, 400):
        b = cr_sim.Battle()
        b.reset(seed, DECK, GHOST_OPP)
        h0, h1 = b.view(0).observe()["hand"], b.view(1).observe()["hand"]
        if "knight" in h0 and "royal_ghost" in h1 and any(ranged(n) for n in h0 if n != "knight"):
            break
    else:
        pytest.fail("no seed with knight + a ranged card for us and royal_ghost for them")
    for _ in range(60):
        b.step(None, None)
    b.step((h0.index("knight"), 3.5, 9.5), (h1.index("royal_ghost"), 3.5, 14.5))
    v0 = b.view(0)
    for _ in range(300):
        b.step(None, None)
        obs = v0.observe()
        st = state.parse(obs)
        ghosts = [u for u in st.enemy if u.name == "royal_ghost"]
        if ghosts and ghosts[0].y < 17 and ghosts[0].lane == "L":
            break
    else:
        pytest.fail("enemy royal_ghost never entered our L lane")
    assert any(u.name == "knight" and u.lane == "L" for u in st.mine)
    assert scripted_v0.defend_index(obs, st) is None  # v0: knight hp "covers" the ghost
    i = defend_index(obs, st)
    assert i is not None and placements.mask(obs)[i]
    slot, p = placements.split(i)
    assert ranged(obs["hand"][slot])
    assert p == P["def_L"]

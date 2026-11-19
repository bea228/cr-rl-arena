import os
import sys

import pytest

# tests/ has no conftest (shared folder rule), so put the repo root on the path here.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cr_sim  # noqa: E402

from bot import placements as P  # noqa: E402
from bot.policy import DECIDE_EVERY, RandomPolicy  # noqa: E402

DECK = ["knight", "archers", "giant", "musketeer", "valkyrie", "minions", "wizard", "minipekka"]
SPELL_DECK = ["fireball", "arrows", "zap", "mirror", "knight", "archers", "giant", "musketeer"]


def battle_with_knight_in_hand():
    """A fresh Battle (seat 0 view) whose opening hand holds a knight."""
    for seed in range(1, 200):
        b = cr_sim.Battle()
        b.reset(seed, DECK, "default")
        obs = b.view(0).observe()
        if "knight" in obs["hand"]:
            return b, obs
    raise AssertionError("no seed gives a knight in the opening hand")


def unit(owner, card, x, y, uid):
    return [owner, card, x, y, 500, 500, False, False, False, uid, 0.5, 0.5]


def with_entities(obs, *extra, hand=None, legal=None):
    out = {**obs, "entities": [list(e) for e in obs["entities"]] + [list(e) for e in extra]}
    if hand is not None:
        out["hand"] = list(hand)
    if legal is not None:
        out["legal"] = list(legal)
    return out


# --- table and indices ---------------------------------------------------

def test_table_shape():
    assert P.N_ACTIONS == 53 == 1 + 4 * 13
    assert len(P.PLACEMENTS) == 13
    assert P.NAMES["bridge_L"] == 0 and P.NAMES["on_threat"] == 12
    assert P.PLACEMENTS[11][1:] == (None, None) and P.PLACEMENTS[12][1:] == (None, None)
    assert P.PLACEMENTS[4] == ("king_pull", 8.5, 9.5)


def test_encode_split_roundtrip():
    seen = set()
    for slot in range(4):
        for p in range(13):
            i = P.encode(slot, p)
            assert i == 1 + slot * 13 + p
            assert P.split(i) == (slot, p)
            seen.add(i)
    assert seen == set(range(1, 53))
    with pytest.raises(ValueError):
        P.split(0)
    with pytest.raises(ValueError):
        P.encode(4, 0)


def test_decode_wait_is_none():
    _, obs = battle_with_knight_in_hand()
    assert P.decode(0, obs) is None


def test_decode_fixed_types():
    _, obs = battle_with_knight_in_hand()
    a = P.decode(P.encode(2, 5), obs)
    assert a == [2, 3.5, 9.5]
    assert type(a[0]) is int and type(a[1]) is float and type(a[2]) is float


# --- simulator acceptance (Done-when for C3) -----------------------------

@pytest.mark.parametrize("p", range(9))
def test_knight_ok_on_own_side_placements(p):
    b, obs = battle_with_knight_in_hand()
    slot = obs["hand"].index("knight")
    idx = P.encode(slot, p)
    assert P.mask(obs)[idx], P.mask_reason(obs, idx)
    a = P.decode(idx, obs)
    b.step(tuple(a), None)
    assert b.view(0).observe()["last_action"] == "ok"


# --- mask rules ----------------------------------------------------------

def test_mask_fresh_ground_hand():
    _, obs = battle_with_knight_in_hand()
    m = P.mask(obs)
    assert len(m) == 53 and m[0] is True
    for slot in range(4):
        for p in range(13):
            i = P.encode(slot, p)
            want = obs["legal"][slot] and p <= 8  # no enemy units yet; ground cards
            assert m[i] == want, (slot, p, P.mask_reason(obs, i))
    assert P.mask_reason(obs, P.encode(0, 9)) in ("ground_on_enemy_tower", "slot_not_legal")
    assert P.mask_reason(obs, P.encode(0, 11)) in ("no_enemy_unit", "slot_not_legal")
    assert P.legal_indices(obs) == [i for i in range(1, 53) if m[i]]


def test_mask_illegal_slot_and_wait_always_true():
    _, obs = battle_with_knight_in_hand()
    o = {**obs, "legal": [False, True, False, True]}
    m = P.mask(o)
    assert m[0]
    assert not any(m[P.encode(0, p)] for p in range(13))
    assert not any(m[P.encode(2, p)] for p in range(13))
    assert P.mask_reason(o, P.encode(0, 0)) == "slot_not_legal"
    none_legal = {**obs, "legal": [False] * 4}
    assert P.mask(none_legal) == [True] + [False] * 52
    assert P.legal_indices(none_legal) == []


def test_mask_spells_mirror_and_enemies():
    _, obs = battle_with_knight_in_hand()
    o = with_entities(obs, unit(1, "knight", 4.0, 20.0, 99),
                      hand=["fireball", "mirror", "knight", "arrows"], legal=[True] * 4)
    m = P.mask(o)
    for slot, spell in ((0, True), (1, False), (2, False), (3, True)):
        for p in range(13):
            i = P.encode(slot, p)
            want = (p >= 9) if spell else (p not in (9, 10))
            assert m[i] == want, (slot, p, P.mask_reason(o, i))
    assert P.mask_reason(o, P.encode(0, 0)) == "spell_on_own_side"
    assert P.mask_reason(o, P.encode(1, 9)) == "ground_on_enemy_tower"
    # enemy towers do not count as enemy units
    o2 = with_entities(obs, hand=["fireball"] * 4, legal=[True] * 4)
    assert not P.mask(o2)[P.encode(0, 11)] and not P.mask(o2)[P.encode(0, 12)]
    assert P.mask(o2)[P.encode(0, 9)]


# --- computed placements -------------------------------------------------

def test_cluster_picks_densest_group():
    _, obs = battle_with_knight_in_hand()
    o = with_entities(
        obs,
        unit(1, "goblins", 12.0, 20.0, 50),
        unit(1, "goblins", 13.0, 20.0, 51),
        unit(1, "goblins", 12.0, 21.0, 52),
        unit(1, "giant", 4.0, 12.0, 53),  # alone
        unit(0, "knight", 12.5, 20.5, 54),  # ours, ignored
        hand=["fireball", "knight", "arrows", "zap"], legal=[True] * 4,
    )
    x, y = P.decode(P.encode(0, P.NAMES["cluster"]), o)[1:]
    assert (x, y) == pytest.approx(((12 + 13 + 12) / 3, (20 + 20 + 21) / 3))
    # ground card: same x, y clamped to 14.5
    a = P.decode(P.encode(1, P.NAMES["cluster"]), o)
    assert a[1] == pytest.approx(37 / 3) and a[2] == 14.5


def test_cluster_tie_lowest_y_then_id():
    _, obs = battle_with_knight_in_hand()
    o = with_entities(obs, unit(1, "a", 3.0, 20.0, 61), unit(1, "b", 15.0, 18.0, 62),
                      unit(1, "c", 9.0, 18.0, 60), hand=["zap"] * 4, legal=[True] * 4)
    # all singletons; lowest y is 18 (ids 62, 60) -> id 60
    assert P.decode(P.encode(0, 11), o)[1:] == [9.0, 18.0]


def test_on_threat_nearest_to_our_towers():
    _, obs = battle_with_knight_in_hand()
    o = with_entities(obs, unit(1, "hog_rider", 14.0, 11.0, 70), unit(1, "giant", 4.0, 16.0, 71),
                      unit(1, "balloon", 9.0, 28.0, 72), hand=["fireball", "knight", "zap", "zap"],
                      legal=[True] * 4)
    assert P.decode(P.encode(0, 12), o)[1:] == [14.0, 11.0]
    # all far up the board: ground card clamps y
    o2 = with_entities(obs, unit(1, "x", 3.5, 22.0, 80), hand=["fireball", "knight", "zap", "zap"],
                       legal=[True] * 4)
    assert P.decode(P.encode(0, 12), o2)[1:] == [3.5, 22.0]
    assert P.decode(P.encode(1, 12), o2)[1:] == [3.5, 14.5]


def test_computed_positions_clamped_into_board():
    _, obs = battle_with_knight_in_hand()
    o = with_entities(obs, unit(1, "x", -1.0, 33.0, 90), hand=["zap"] * 4, legal=[True] * 4)
    assert P.decode(P.encode(0, 12), o)[1:] == [0.5, 31.5]


# --- real-game sweeps ----------------------------------------------------

def collect_observations(n=200, deck=DECK, opp="default"):
    """Seat-0 decision observations; both seats play random masked moves."""
    out = []
    pol, opp_pol = RandomPolicy(seed=0), RandomPolicy(seed=1)
    game = 0
    while len(out) < n:
        b = cr_sim.Battle()
        b.reset(game + 1, deck, opp)
        v0, v1 = b.view(0), b.view(1)
        while not b.done() and len(out) < n:
            obs = v0.observe()
            a0 = a1 = None
            if obs["tick"] % DECIDE_EVERY == 0:
                out.append(obs)
                a0 = pol.act(obs)
                a1 = opp_pol.act(v1.observe())
            b.step(tuple(a0) if a0 else None, tuple(a1) if a1 else None)
        game += 1
    return out


@pytest.mark.parametrize("deck", [DECK, SPELL_DECK])
def test_every_allowed_index_decodes(deck):
    obs_list = collect_observations(200, deck)
    saw_enemy = False
    for obs in obs_list:
        m = P.mask(obs)
        saw_enemy |= bool(P.enemy_units(obs))
        for i in range(53):
            if not m[i]:
                continue
            a = P.decode(i, obs)
            if i == 0:
                assert a is None
                continue
            slot, x, y = a
            assert type(slot) is int and obs["legal"][slot]
            assert 0.5 <= x <= 17.5 and 0.5 <= y <= 31.5
            if not P.is_spell(obs["hand"][slot]):
                assert y <= 14.5
    assert saw_enemy


def test_masked_random_play_never_wrong_side():
    """Masked random actions are never rejected for side, and are mostly accepted."""
    for deck in (DECK, SPELL_DECK):
        pol0, pol1 = RandomPolicy(seed=3), RandomPolicy(seed=4)
        b = cr_sim.Battle()
        b.reset(11, deck, SPELL_DECK)
        pol0.reset(deck)
        pol1.reset(SPELL_DECK)
        views = (b.view(0), b.view(1))
        reasons = {}
        while not b.done():
            step = [None, None]
            if b.tick % DECIDE_EVERY == 0:
                for seat, pol in ((0, pol0), (1, pol1)):
                    a = pol.act(views[seat].observe())
                    step[seat] = tuple(a) if a is not None else None
            b.step(step[0], step[1])
            for seat in (0, 1):
                if step[seat] is not None:
                    r = b.last_action(seat)
                    reasons[r] = reasons.get(r, 0) + 1
        assert "not_your_side" not in reasons, reasons
        assert reasons.get("ok", 0) > 0

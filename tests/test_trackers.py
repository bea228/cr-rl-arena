import os
import sys
import time

import pytest

# tests/ has no conftest (shared folder rule), so put the repo root on the path here.
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import cr_sim  # noqa: E402

from bot import cards  # noqa: E402
from bot.policy import DECIDE_EVERY  # noqa: E402
from bot.scripted import ScriptedPolicy  # noqa: E402
from bot.trackers import MAX_ELIXIR, START_ELIXIR, Trackers, regen  # noqa: E402
from evals.decks import load_gauntlet  # noqa: E402
from evals.match import to_step  # noqa: E402

N_GAMES = 50

# our towers + enemy towers, as tuples (owner, card, x, y, hp, max_hp, flying, is_tower, deploying, id, hw, hh)
TOWERS = [
    (0, "princess_tower", 3.5, 6.5, 2534, 2534, False, True, False, 1, 1.5, 1.5),
    (0, "princess_tower", 14.5, 6.5, 2534, 2534, False, True, False, 2, 1.5, 1.5),
    (0, "king_tower", 9.0, 3.0, 4008, 4008, False, True, False, 3, 2.0, 2.0),
    (1, "princess_tower", 3.5, 25.5, 2534, 2534, False, True, False, 4, 1.5, 1.5),
    (1, "princess_tower", 14.5, 25.5, 2534, 2534, False, True, False, 5, 1.5, 1.5),
    (1, "king_tower", 9.0, 29.0, 4008, 4008, False, True, False, 6, 2.0, 2.0),
]


def unit(card, eid, owner=1, x=9.0, y=20.0, as_list=False):
    e = (owner, card, x, y, 100, 100, False, False, True, eid, 0.5, 0.5)
    return list(e) if as_list else e


def obs(tick, units=(), towers=True):
    ents = (list(TOWERS) if towers else []) + list(units)
    phase = "single" if tick < 2400 else "double" if tick < 3600 else "overtime"
    return {"tick": tick, "phase": phase, "entities": ents}


# --- C8 -------------------------------------------------------------------

def test_cycle_abcde():
    t = Trackers()
    t.reset()
    played = ["knight", "archers", "giant", "musketeer", "valkyrie"]  # A..E
    t.update(obs(0))
    for i, c in enumerate(played):
        t.update(obs(100 * (i + 1), [unit(c, 100 + i)]))
    got = dict(t.enemy_cards())
    assert [c for c, _ in t.enemy_cards()] == played
    assert got["knight"] is True  # A possible
    for c in played[1:]:
        assert got[c] is False  # B..E not in hand
    assert t.n_known() == 5


def test_cycle_max_eight_and_repeats():
    t = Trackers()
    deck = ["knight", "archers", "giant", "musketeer", "valkyrie", "wizard", "minipekka", "bomber", "prince"]
    t.update(obs(0))
    for i, c in enumerate(deck + ["knight"]):
        t.update(obs(10 * (i + 1), [unit(c, 200 + i)]))
    assert t.n_known() == 8
    assert "prince" not in dict(t.enemy_cards())
    assert t.cycle.recent() == ["knight", "prince", "bomber", "minipekka"]
    assert dict(t.enemy_cards())["knight"] is False


# --- regen ----------------------------------------------------------------

@pytest.mark.parametrize("t0,t1,expected", [
    (0, 100, 100 / 56),                          # single
    (2400, 2500, 100 / 28),                      # double
    (3600, 3650, 50 * 3 / 56),                   # overtime
    (2350, 2450, 50 / 56 + 50 / 28),             # crossing 2400
    (3590, 3610, 10 / 28 + 10 * 3 / 56),         # crossing 3600
])
def test_regen_rates(t0, t1, expected):
    assert regen(t0, t1) == pytest.approx(expected)


@pytest.mark.parametrize("ticks", [[0, 100], [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100], [0, 37, 100]])
def test_regen_single_any_cadence(ticks):
    t = Trackers()
    for k in ticks:
        t.update(obs(k))
    assert t.enemy_elixir() == pytest.approx(min(MAX_ELIXIR, START_ELIXIR + 100 / 56))


def _drain(t, tick, eid):
    """Enemy plays golem + zappies (8 + 4): the estimate drops to 0."""
    t.update(obs(tick, [unit("golem", eid), unit("zappies", eid + 1)]))  # 8 + 4 -> 0


@pytest.mark.parametrize("start,end", [(2400, 2500), (3600, 3650), (2350, 2450)])
def test_regen_phases_via_update(start, end):
    t = Trackers()
    t.update(obs(0))
    _drain(t, start, 900)
    assert t.enemy_elixir() == 0.0
    t.update(obs(end, [unit("golem", 900), unit("zappies", 901)]))
    assert t.enemy_elixir() == pytest.approx(min(MAX_ELIXIR, regen(start, end)))


def test_cap_at_ten():
    t = Trackers()
    t.update(obs(0))
    t.update(obs(5000))
    assert t.enemy_elixir() == MAX_ELIXIR


# --- charging -------------------------------------------------------------

def test_knight_charges_three():
    t = Trackers()
    t.update(obs(0))
    t.update(obs(10, [unit("knight", 50)]))
    assert t.enemy_elixir() == pytest.approx(5 + 10 / 56 - 3)
    t.update(obs(20, [unit("knight", 50)]))  # same id again: no charge
    assert t.enemy_elixir() == pytest.approx(5 + 20 / 56 - 3)


def test_minions_charge_once_per_update():
    t = Trackers()
    t.update(obs(0))
    t.update(obs(10, [unit("minions", 60), unit("minions", 61), unit("minions", 62)]))
    assert t.enemy_elixir() == pytest.approx(5 + 10 / 56 - cards.elixir("minions"))
    assert cards.elixir("minions") == 3


def test_golemites_after_golem_charge_nothing():
    t = Trackers()
    t.update(obs(0))
    t.update(obs(400, [unit("golem", 70)]))
    e = t.enemy_elixir()
    t.update(obs(410, [unit("golemite", 71), unit("golemite", 72)]))  # golem gone already
    assert t.enemy_elixir() == pytest.approx(e + 10 / 56)
    assert [c for c, _ in t.enemy_cards()] == ["golem"]


def test_spawned_by_unit_on_board_not_charged():
    t = Trackers()
    t.update(obs(0))
    t.update(obs(300, [unit("witch", 80)]))
    e = t.enemy_elixir()
    t.update(obs(310, [unit("witch", 80), unit("skeletons", 81), unit("skeletons", 82)]))
    assert t.enemy_elixir() == pytest.approx(e + 10 / 56)


def test_evo_knight_charges_three_as_knight():
    t = Trackers()
    t.update(obs(0))
    t.update(obs(10, [unit("evo_knight", 90)]))
    assert t.enemy_elixir() == pytest.approx(5 + 10 / 56 - 3)
    assert t.enemy_cards() == [("knight", False)]


def test_non_deck_towers_ours_and_unknown_ignored():
    t = Trackers()
    t.update(obs(0))
    t.update(obs(10, [unit("lava_pup", 91), unit("knight", 92, owner=0), unit("not_a_card", 93)]))
    assert t.enemy_elixir() == pytest.approx(5 + 10 / 56)
    assert t.enemy_cards() == []


def test_never_below_zero_and_lists_and_garbage():
    t = Trackers()
    t.update(obs(0))
    t.update(obs(10, [unit("golem", 1, as_list=True), unit("pekka", 2, as_list=True)]))
    assert t.enemy_elixir() == 0.0
    t.update({"tick": 20, "entities": [["bad"]]})  # malformed entity: no raise
    t.update({})
    assert 0.0 <= t.enemy_elixir() <= MAX_ELIXIR


def test_new_match_without_reset():
    t = Trackers()
    t.update(obs(0))
    t.update(obs(500, [unit("knight", 5)]))
    t.update(obs(0))  # tick went backwards: a new match
    assert t.enemy_elixir() == START_ELIXIR and t.enemy_cards() == []


def test_scripted_policy_owns_trackers():
    p = ScriptedPolicy()
    p.reset(["knight"] * 8)
    assert isinstance(p.trackers, Trackers)
    first = p.trackers
    p.reset(["knight"] * 8)
    assert p.trackers is not first


# --- C7 accuracy ----------------------------------------------------------

def _has_spell(deck):
    return any(cards.card_type(c) == "spell" for c in deck)


def _accuracy_game(i, decks):
    """One scripted-vs-scripted game on gauntlet decks i, i+1 (seed i). Returns (hits, total)."""
    d0, d1 = decks[i % len(decks)], decks[(i + 1) % len(decks)]
    b = cr_sim.Battle()
    b.reset(i, d0, d1)
    views = (b.view(0), b.view(1))
    pols = (ScriptedPolicy(), ScriptedPolicy())
    pols[0].reset(d0)
    pols[1].reset(d1)
    tr = Trackers()
    hits = total = 0
    while not b.done():
        step = [None, None]
        if b.tick % DECIDE_EVERY == 0:
            o0 = views[0].observe()
            tr.update(o0)
            total += 1
            hits += abs(tr.enemy_elixir() - b.elixir(1)) <= 1.0  # the only b.elixir(1) read
            for seat, o in ((0, o0), (1, views[1].observe())):
                step[seat] = to_step(pols[seat].act(o))
        b.step(step[0], step[1])
    return hits, total, _has_spell(d1)


def test_c7_elixir_accuracy():
    decks = load_gauntlet()
    t0 = time.time()
    all_h = all_n = ns_h = ns_n = ns_games = 0
    for i in range(N_GAMES):
        h, n, spell = _accuracy_game(i, decks)
        all_h += h
        all_n += n
        if not spell:
            ns_h += h
            ns_n += n
            ns_games += 1
    pct_all = 100 * all_h / all_n
    pct_ns = 100 * ns_h / ns_n if ns_n else float("nan")
    print(f"\nC7 elixir within 1.0: all {N_GAMES} games {pct_all:.1f}% ({all_h}/{all_n}); "
          f"no-spell decks ({ns_games} games) {pct_ns:.1f}% ({ns_h}/{ns_n}); {time.time() - t0:.1f}s")
    assert ns_games > 0, "no spell-free opponent deck in the first 50 games"
    assert pct_ns >= 90.0

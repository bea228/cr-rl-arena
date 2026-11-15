"""Deck legality: 8 different, playable cards, at most one champion."""
import pytest

import cr_sim

GOOD = ["knight", "archers", "giant", "musketeer", "valkyrie", "minions", "wizard", "minipekka"]


def _bad(deck, match):
    b = cr_sim.Battle()
    with pytest.raises(ValueError, match=match):
        b.check_deck(deck)
    with pytest.raises(ValueError, match=match):  # reset enforces the same rules
        b.reset(1, deck, GOOD)


def test_good_deck_is_accepted():
    b = cr_sim.Battle()
    assert b.check_deck(GOOD) is None
    b.reset(1, GOOD, GOOD)
    assert sorted(b.hand(0) + [b.next_card(0)]) != []


def test_wrong_size_rejected():
    _bad(GOOD[:7], "8 cards")


def test_unknown_card_rejected():
    _bad(GOOD[:7] + ["knigth"], "unknown card")


def test_duplicates_rejected():
    _bad(["knight"] * 8, "duplicate")
    _bad(GOOD[:7] + ["knight"], "duplicate")


@pytest.mark.parametrize("name", ["king_tower", "princess_tower", "golemite", "evo_knight"])
def test_non_cards_rejected(name):
    _bad(GOOD[:7] + [name], "not a playable card")

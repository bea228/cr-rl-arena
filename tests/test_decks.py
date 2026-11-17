import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evals import decks  # noqa: E402


def test_gauntlet_decks():
    gauntlet = decks.load_gauntlet()
    heldout = set(decks.load_heldout_cards())
    assert len(gauntlet) == decks.N_GAUNTLET
    for d in gauntlet:
        assert len(d) == 8 and len(set(d)) == 8
        assert decks.is_legal(d)
        assert not heldout & set(d)


def test_heldout_cards():
    heldout = decks.load_heldout_cards()
    assert len(heldout) == 12 and len(set(heldout)) == 12
    assert heldout == sorted(heldout)
    deployable = set(decks.deployable_cards())
    champs = decks.champions()
    for c in heldout:
        assert c in deployable
        assert c not in champs
        assert c not in decks.DEFAULT_DECK


def test_heldout_decks():
    heldout = set(decks.load_heldout_cards())
    hd = decks.load_heldout_decks()
    assert len(hd) == decks.N_HELDOUT_DECKS
    for d in hd:
        assert len(set(d)) == 8
        assert len(heldout & set(d)) >= 3
        assert decks.is_legal(d)


def test_regenerate_is_byte_identical(tmp_path):
    decks.write_all(tmp_path)
    for name in (decks.GAUNTLET_FILE, decks.HELDOUT_CARDS_FILE, decks.HELDOUT_DECKS_FILE):
        fresh = (tmp_path / name).read_bytes()
        assert b"\r" not in fresh and fresh.endswith(b"\n")
        # core.autocrlf=true checkouts turn the committed LF files into CRLF; ignore only that
        committed = (decks.HERE / name).read_bytes().replace(b"\r\n", b"\n")
        assert fresh == committed, name


def test_random_deck_respects_exclude():
    import random

    rng = random.Random(5)
    ex = decks.deployable_cards()[:50]
    for _ in range(20):
        d = decks.random_deck(rng, exclude=ex)
        assert not set(ex) & set(d)
        assert decks.is_legal(d)

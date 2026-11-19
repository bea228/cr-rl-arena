import json
import os
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bot import cards  # noqa: E402

_DATA = json.loads(
    (Path(__file__).resolve().parents[1] / "gym" / "data" / "cards.json").read_text(encoding="utf-8")
)
ALL_NAMES = list(_DATA["cards"]) + list(_DATA["towers"])


def test_feat_every_card_and_tower():
    assert len(ALL_NAMES) == 134
    for n in ALL_NAMES:
        v = cards.feat(n)
        assert v.shape == (cards.FEAT_DIM,), n
        assert v.dtype == np.float32, n
        assert np.all(np.isfinite(v)), n
        assert np.all((v >= 0) & (v <= 2)), (n, v)


def test_feat_dim():
    assert isinstance(cards.FEAT_DIM, int)
    for n in ALL_NAMES:
        assert len(cards.feat(n)) == cards.FEAT_DIM


def test_feat_returns_copy():
    a = cards.feat("knight")
    a[:] = 99.0
    b = cards.feat("knight")
    assert not np.any(b == 99.0)


def test_towers_have_no_type_flags():
    for t in _DATA["towers"]:
        assert cards.card_type(t) == "tower"
        assert cards.feat(t)[1:4].sum() == 0


def test_unknown_name():
    with pytest.raises(KeyError):
        cards.feat("not_a_card")
    with pytest.raises(KeyError):
        cards.role("not_a_card")


def test_roles():
    k = cards.role("knight")
    for flag in ("hits_air", "spell", "building", "flying", "swarm", "tank", "win_condition"):
        assert k[flag] is False, flag

    g = cards.role("giant")
    assert g["tank"] and g["win_condition"] and not g["hits_air"]

    m = cards.role("minions")
    assert m["flying"] and m["hits_air"] and m["swarm"]

    f = cards.role("fireball")
    assert f["spell"] and f["splash"]

    c = cards.role("cannon")
    assert c["building"] and not c["hits_air"]

    s = cards.role("skeleton_army")
    assert s["swarm"]
    # skeleton_army costs 3 in cards.json, so it is not cheap.
    assert cards.elixir("skeleton_army") == 3
    assert s["cheap"] is False


def test_role_keys():
    expected = {"spell", "building", "flying", "hits_air", "splash", "swarm", "tank", "win_condition", "cheap"}
    for n in ALL_NAMES:
        r = cards.role(n)
        assert set(r) == expected
        assert all(isinstance(x, bool) for x in r.values())


def test_card_sets():
    assert len(cards.ALL_CARDS) == 132
    assert cards.ALL_CARDS == sorted(cards.ALL_CARDS)
    assert len(cards.DEPLOYABLE) == 116
    assert len(cards.NON_DECK) == 16
    assert cards.DEPLOYABLE | cards.NON_DECK == set(cards.ALL_CARDS)
    assert not cards.DEPLOYABLE & cards.NON_DECK


def test_spawned_by():
    assert "golemite" in cards.spawned_by("golem")
    assert "evo_knight" in cards.spawned_by("knight")
    assert cards.spawned_by("fireball") == frozenset()
    for n in cards.ALL_CARDS:
        for s in cards.spawned_by(n):
            assert s in _DATA["cards"], (n, s)


def test_raw_is_copy():
    r = cards.raw("knight")
    r["hp"] = -1
    assert cards.raw("knight")["hp"] == _DATA["cards"]["knight"]["hp"]

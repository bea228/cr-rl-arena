"""Card stat table: feature vectors, role flags and lookups from gym/data/cards.json.

Handoff step C1 / section 3.3. Loaded once at import; import-light (numpy + cr_sim only).
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np

import cr_sim

_CARDS_PATH = Path(__file__).resolve().parents[1] / "gym" / "data" / "cards.json"

with _CARDS_PATH.open(encoding="utf-8") as _f:
    _DATA: dict[str, Any] = json.load(_f)

_CARDS: dict[str, dict[str, Any]] = _DATA["cards"]
_TOWERS: dict[str, dict[str, Any]] = _DATA["towers"]

ALL_CARDS: list[str] = sorted(_CARDS)
DEPLOYABLE: frozenset[str] = frozenset(cr_sim.Battle().deployable_cards())
NON_DECK: frozenset[str] = frozenset(_CARDS) - DEPLOYABLE


def _lookup(name: str) -> dict[str, Any]:
    if name in _CARDS:
        return _CARDS[name]
    if name in _TOWERS:
        return _TOWERS[name]
    raise KeyError(f"{name!r} is not in cards.json (cards or towers)")


def raw(name: str) -> dict[str, Any]:
    """A copy of the card's (or tower's) dict from cards.json."""
    return json.loads(json.dumps(_lookup(name)))


def elixir(name: str) -> int:
    return int(_lookup(name).get("elixir", 0))


def card_type(name: str) -> str:
    """'troop' | 'spell' | 'building' | 'tower'."""
    return _lookup(name).get("type", "tower")


def _num(c: dict[str, Any], key: str) -> float:
    return float(c.get(key, 0) or 0)


def _flag(x: bool) -> float:
    return 1.0 if x else 0.0


def _build_feat(c: dict[str, Any]) -> np.ndarray:
    count = float(c.get("count", 1))  # a missing count means one unit
    hp = _num(c, "hp")
    damage = _num(c, "damage")
    hit_speed = _num(c, "hit_speed")
    dps = damage / hit_speed if damage and hit_speed else 0.0
    ctype = c.get("type")
    targets = c.get("targets")
    v = [
        _num(c, "elixir") / 10,
        _flag(ctype == "troop"),
        _flag(ctype == "spell"),
        _flag(ctype == "building"),
        math.log1p(hp * count) / 12,
        math.log1p(damage) / 8,
        math.log1p(dps * count) / 8,
        _num(c, "range") / 12,
        _num(c, "speed") / 2,
        _flag(targets in ("ground", "both")),
        _flag(targets == "both"),
        _flag(bool(c.get("target_buildings_only"))),
        _flag(bool(c.get("flying"))),
        _flag("splash_radius" in c or "radius" in c),
        count / 15,
        _num(c, "deploy_time") / 3,
        _flag(bool(c.get("lifetime"))),
        _flag("spawn" in c or "death_spawn" in c),
        _flag(bool(c.get("champion"))),
        _flag(bool(c.get("shield"))),
        _flag("charge" in c or "dash" in c),
        _flag(bool(c.get("stun") or c.get("on_hit_freeze") or c.get("on_hit_slow"))),
        _num(c, "radius") / 5,
        _num(c, "crown_tower_damage_pct") / 100,
    ]
    return np.asarray(v, dtype=np.float32)


_FEAT_CACHE: dict[str, np.ndarray] = {}


def feat(name: str) -> np.ndarray:
    """float32 vector of length FEAT_DIM for any card or tower. Returns a copy."""
    vec = _FEAT_CACHE.get(name)
    if vec is None:
        vec = _build_feat(_lookup(name))
        _FEAT_CACHE[name] = vec
    return vec.copy()


FEAT_DIM: int = len(feat("knight"))


def role(name: str) -> dict[str, bool]:
    """Flags used by the scripted rules."""
    c = _lookup(name)
    return {
        "spell": c.get("type") == "spell",
        "building": c.get("type") == "building",
        "flying": bool(c.get("flying")),
        "hits_air": c.get("targets") == "both",
        "splash": "splash_radius" in c or "radius" in c,
        "swarm": c.get("count", 1) >= 3,
        "tank": c.get("hp", 0) >= 2500,
        "win_condition": bool(c.get("target_buildings_only")),
        "cheap": c.get("elixir", 0) <= 2,
    }


def _direct_spawns(c: dict[str, Any]) -> set[str]:
    out: set[str] = set()
    for key in ("spawn", "death_spawn", "deploy_spawn"):
        link = c.get(key)
        if isinstance(link, dict) and "card" in link:
            out.add(link["card"])
    for key in ("curse_spawn", "evolved"):
        link = c.get(key)
        if isinstance(link, str):
            out.add(link)
    return out


def spawned_by(name: str) -> frozenset[str]:
    """Every card name this card can put on the board besides itself (recursive)."""
    seen: set[str] = set()
    stack = list(_direct_spawns(_lookup(name)))
    while stack:
        n = stack.pop()
        if n in seen or n == name:
            continue
        seen.add(n)
        if n in _CARDS:
            stack.extend(_direct_spawns(_CARDS[n]))
    return frozenset(seen)

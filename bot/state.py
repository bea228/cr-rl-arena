"""Observation parser (handoff step C2).

``parse(obs) -> State`` turns the raw obs dict (our own frame, we are the
bottom player) into lane-level summaries the scripted rules work from.
Entities are tuples locally and lists from the server, so they are indexed,
never unpacked by type. ``parse`` never raises on a valid obs: a destroyed
tower is simply absent from ``entities`` and reads 0.
"""
from __future__ import annotations

from dataclasses import dataclass, field

# entity field indices: (owner, card, x, y, hp, max_hp, flying, is_tower, deploying, id, half_w, half_h)
OWNER, CARD, X, Y, HP, MAX_HP, FLYING, IS_TOWER, DEPLOYING, ID, HALF_W, HALF_H = range(12)

MID_X = 9.0  # lane split: L if x < 9 else R
THREAT_Y = 17.0  # units below this y count toward threat / defense
LANES = ("L", "R")
TOWER_KEYS = ("L", "R", "K")


def lane_of(x: float) -> str:
    """'L' if x < 9 else 'R'."""
    return "L" if x < MID_X else "R"


@dataclass
class Unit:
    name: str
    x: float
    y: float
    hp: int
    max_hp: int
    lane: str
    flying: bool
    deploying: bool
    id: int


def _zero_towers() -> dict[str, int]:
    return {k: 0 for k in TOWER_KEYS}


def _zero_lanes() -> dict[str, int]:
    return {k: 0 for k in LANES}


@dataclass
class State:
    tick: int = 0
    phase: str = "single"
    elixir: float = 0.0
    hand: list[str] = field(default_factory=list)
    cost: list[int] = field(default_factory=list)
    legal: list[bool] = field(default_factory=list)
    mine: list[Unit] = field(default_factory=list)
    enemy: list[Unit] = field(default_factory=list)
    my_tower_hp: dict[str, int] = field(default_factory=_zero_towers)
    enemy_tower_hp: dict[str, int] = field(default_factory=_zero_towers)
    my_tower_max: dict[str, int] = field(default_factory=_zero_towers)
    enemy_tower_max: dict[str, int] = field(default_factory=_zero_towers)
    threat: dict[str, int] = field(default_factory=_zero_lanes)
    defense: dict[str, int] = field(default_factory=_zero_lanes)


def _tower_key(card: str, x: float) -> str | None:
    if card == "king_tower":
        return "K"
    if card == "princess_tower":
        return lane_of(x)
    return None


def parse(obs: dict) -> State:
    s = State(
        tick=int(obs.get("tick", 0)),
        phase=str(obs.get("phase", "single")),
        elixir=float(obs.get("elixir", 0.0)),
        hand=list(obs.get("hand", [])),
        cost=[int(c) for c in obs.get("hand_cost", [])],
        legal=[bool(v) for v in obs.get("legal", [])],
    )
    for e in obs.get("entities", []):
        owner = int(e[OWNER])
        card = str(e[CARD])
        x, y = float(e[X]), float(e[Y])
        hp, max_hp = int(e[HP]), int(e[MAX_HP])
        if e[IS_TOWER]:
            key = _tower_key(card, x)
            if key is None:
                continue
            hp_d, max_d = (s.my_tower_hp, s.my_tower_max) if owner == 0 else (s.enemy_tower_hp, s.enemy_tower_max)
            hp_d[key] = hp
            max_d[key] = max_hp
            continue
        u = Unit(
            name=card, x=x, y=y, hp=hp, max_hp=max_hp, lane=lane_of(x),
            flying=bool(e[FLYING]), deploying=bool(e[DEPLOYING]), id=int(e[ID]),
        )
        if owner == 0:
            s.mine.append(u)
            if y < THREAT_Y:
                s.defense[u.lane] += hp
        else:
            s.enemy.append(u)
            if y < THREAT_Y:
                s.threat[u.lane] += hp
    return s

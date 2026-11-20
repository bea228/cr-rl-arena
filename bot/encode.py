"""Observation encoder (handoff B6, section 3.4).

``encode(obs, trackers) -> dict`` of float32 numpy arrays with fixed shapes:

    globals          (N_GLOBALS,)               16
    hand             (N_HAND, HAND_DIM)         5 x (FEAT_DIM + 2)
    units            (N_UNITS, UNIT_DIM)        48 x (FEAT_DIM + 7)
    units_mask       (N_UNITS,)
    enemy_deck       (N_ENEMY_DECK, ENEMY_DIM)  8 x (FEAT_DIM + 1)
    enemy_deck_mask  (N_ENEMY_DECK,)
    mask             (N_ACTIONS,)               placements.mask as 1/0

``trackers`` only needs ``enemy_elixir()`` and ``enemy_cards()`` (W9). numpy only: the
live client must not need torch. ``batch_encode`` stacks encodings for training.
"""
from __future__ import annotations

import math

import numpy as np

from bot import cards, placements, state

FEAT_DIM = cards.FEAT_DIM
N_ACTIONS = placements.N_ACTIONS

N_GLOBALS = 16
N_HAND = 5  # four slots + next card
N_UNITS = 48
N_ENEMY_DECK = 8
HAND_EXTRA = 2  # affordable, cost/10
UNIT_EXTRA = 7  # owner, x/18, y/32, hp fraction, is_tower, deploying, flying
ENEMY_EXTRA = 1  # could_be_in_hand
HAND_DIM = FEAT_DIM + HAND_EXTRA
UNIT_DIM = FEAT_DIM + UNIT_EXTRA
ENEMY_DIM = FEAT_DIM + ENEMY_EXTRA

KEYS = ("globals", "hand", "units", "units_mask", "enemy_deck", "enemy_deck_mask", "mask")
SHAPES = {
    "globals": (N_GLOBALS,),
    "hand": (N_HAND, HAND_DIM),
    "units": (N_UNITS, UNIT_DIM),
    "units_mask": (N_UNITS,),
    "enemy_deck": (N_ENEMY_DECK, ENEMY_DIM),
    "enemy_deck_mask": (N_ENEMY_DECK,),
    "mask": (N_ACTIONS,),
}

MAX_TICKS = 6000  # 3,600 regulation + 2,400 overtime
KING_XY = (9.0, 3.0)  # our king tower, in our own frame
ARENA_W, ARENA_H = 18.0, 32.0
PHASES = ("single", "double", "overtime")

# entity field indices (see bot.state)
_OWNER, _CARD, _X, _Y, _HP, _MAX_HP, _FLYING, _IS_TOWER, _DEPLOYING, _ID = range(10)

_ZERO_FEAT = np.zeros(FEAT_DIM, dtype=np.float32)


def card_feat(name) -> np.ndarray:
    """cards.feat(name), or zeros for an unknown name."""
    try:
        return cards.feat(str(name))
    except KeyError:
        return _ZERO_FEAT.copy()


def _elixir_cost(name) -> float:
    try:
        return float(cards.elixir(str(name)))
    except KeyError:
        return 0.0


def _frac(hp, max_hp) -> float:
    return float(hp) / float(max_hp) if max_hp and max_hp > 0 else 0.0


def _finite(v: float) -> float:
    return v if math.isfinite(v) else 0.0


def encode_globals(obs: dict, st: state.State, trackers) -> np.ndarray:
    g = np.zeros(N_GLOBALS, dtype=np.float32)
    my = _finite(float(obs.get("elixir", 0.0)))
    their = _finite(float(trackers.enemy_elixir()))
    crowns = list(obs.get("crowns", (0, 0))) + [0, 0]
    phase = str(obs.get("phase", "single"))
    tick = int(obs.get("tick", 0))
    g[0] = my / 10
    g[1] = their / 10
    g[2] = (my - their) / 10
    for i, p in enumerate(PHASES):
        g[3 + i] = 1.0 if phase == p else 0.0
    g[6] = max(0.0, 1.0 - tick / MAX_TICKS)
    g[7] = float(crowns[0]) / 3
    g[8] = float(crowns[1]) / 3
    for i, k in enumerate(state.TOWER_KEYS):
        g[9 + i] = _frac(st.my_tower_hp[k], st.my_tower_max[k])
        g[12 + i] = _frac(st.enemy_tower_hp[k], st.enemy_tower_max[k])
    g[15] = len(list(trackers.enemy_cards())[:N_ENEMY_DECK]) / N_ENEMY_DECK
    return g


def encode_hand(obs: dict) -> np.ndarray:
    h = np.zeros((N_HAND, HAND_DIM), dtype=np.float32)
    elixir = float(obs.get("elixir", 0.0))
    hand = list(obs.get("hand", []))[:4]
    costs = list(obs.get("hand_cost", []))
    for i, name in enumerate(hand):
        cost = float(costs[i]) if i < len(costs) else _elixir_cost(name)
        h[i, :FEAT_DIM] = card_feat(name)
        h[i, FEAT_DIM] = 1.0 if elixir >= cost else 0.0
        h[i, FEAT_DIM + 1] = cost / 10
    nxt = obs.get("next_card")
    if nxt:
        h[4, :FEAT_DIM] = card_feat(nxt)
        h[4, FEAT_DIM] = 0.0
        h[4, FEAT_DIM + 1] = _elixir_cost(nxt) / 10
    return h


def _dist_key(e) -> tuple[float, int]:
    return (math.hypot(float(e[_X]) - KING_XY[0], float(e[_Y]) - KING_XY[1]), int(e[_ID]))


def encode_units(obs: dict) -> tuple[np.ndarray, np.ndarray]:
    u = np.zeros((N_UNITS, UNIT_DIM), dtype=np.float32)
    m = np.zeros(N_UNITS, dtype=np.float32)
    ents = sorted(obs.get("entities", []), key=_dist_key)[:N_UNITS]
    for i, e in enumerate(ents):
        u[i, :FEAT_DIM] = card_feat(e[_CARD])
        u[i, FEAT_DIM:] = (
            float(e[_OWNER]),
            float(e[_X]) / ARENA_W,
            float(e[_Y]) / ARENA_H,
            _frac(e[_HP], e[_MAX_HP]),
            1.0 if e[_IS_TOWER] else 0.0,
            1.0 if e[_DEPLOYING] else 0.0,
            1.0 if e[_FLYING] else 0.0,
        )
        m[i] = 1.0
    return u, m


def encode_enemy_deck(trackers) -> tuple[np.ndarray, np.ndarray]:
    d = np.zeros((N_ENEMY_DECK, ENEMY_DIM), dtype=np.float32)
    m = np.zeros(N_ENEMY_DECK, dtype=np.float32)
    for i, (name, in_hand) in enumerate(list(trackers.enemy_cards())[:N_ENEMY_DECK]):
        d[i, :FEAT_DIM] = card_feat(name)
        d[i, FEAT_DIM] = 1.0 if in_hand else 0.0
        m[i] = 1.0
    return d, m


def encode(obs: dict, trackers) -> dict[str, np.ndarray]:
    st = state.parse(obs)
    units, units_mask = encode_units(obs)
    deck, deck_mask = encode_enemy_deck(trackers)
    return {
        "globals": encode_globals(obs, st, trackers),
        "hand": encode_hand(obs),
        "units": units,
        "units_mask": units_mask,
        "enemy_deck": deck,
        "enemy_deck_mask": deck_mask,
        "mask": np.asarray(placements.mask(obs), dtype=np.float32),
    }


def batch_encode(items: list[dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
    """List of encode() outputs -> dict of arrays with a leading batch dimension."""
    return {k: np.stack([it[k] for it in items]).astype(np.float32, copy=False) for k in KEYS}

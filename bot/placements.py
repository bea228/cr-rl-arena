"""The 53-action table and its mask (handoff step C3, section 3.2).

N_ACTIONS = 1 + 4 * 13. Index 0 is wait; otherwise index = 1 + slot * 13 + p
for hand slot 0..3 and placement p 0..12.

"Spell" means ``cards.role(name)["spell"]`` and the card is not "mirror".
Mirror and every troop or building are "ground cards": they can't use
tower_L / tower_R, and their computed positions are clamped to y <= 14.5.
Decoded positions are left unsnapped (the simulator snaps to tile centres).
"""
from __future__ import annotations

import math

from bot import cards

# entity field indices (see bot.state)
_OWNER, _CARD, _X, _Y, _IS_TOWER, _ID = 0, 1, 2, 3, 7, 9

PLACEMENTS: list[tuple[str, float | None, float | None]] = [
    ("bridge_L", 3.5, 14.5),
    ("bridge_R", 14.5, 14.5),
    ("back_L", 3.5, 2.5),
    ("back_R", 14.5, 2.5),
    ("king_pull", 8.5, 9.5),
    ("def_L", 3.5, 9.5),
    ("def_R", 14.5, 9.5),
    ("mid_L", 5.5, 11.5),
    ("mid_R", 12.5, 11.5),
    ("tower_L", 3.5, 25.5),
    ("tower_R", 14.5, 25.5),
    ("cluster", None, None),
    ("on_threat", None, None),
]
NAMES: dict[str, int] = {name: p for p, (name, _, _) in enumerate(PLACEMENTS)}
N_PLACEMENTS = len(PLACEMENTS)  # 13
N_SLOTS = 4
N_ACTIONS = 1 + N_SLOTS * N_PLACEMENTS  # 53
WAIT = 0

FIXED_OWN_SIDE = range(0, 9)  # placements 0..8
ENEMY_TOWERS = (NAMES["tower_L"], NAMES["tower_R"])
COMPUTED = (NAMES["cluster"], NAMES["on_threat"])

CLUSTER_RADIUS = 2.5
GROUND_MAX_Y = 14.5
X_MIN, X_MAX, Y_MIN, Y_MAX = 0.5, 17.5, 0.5, 31.5


def encode(slot: int, p: int) -> int:
    if not (0 <= slot < N_SLOTS and 0 <= p < N_PLACEMENTS):
        raise ValueError(f"bad slot/placement {slot}/{p}")
    return 1 + slot * N_PLACEMENTS + p


def split(index: int) -> tuple[int, int]:
    """index >= 1 -> (slot, p)."""
    if not (1 <= index < N_ACTIONS):
        raise ValueError(f"index {index} has no slot/placement")
    return divmod(index - 1, N_PLACEMENTS)


def is_spell(name: str) -> bool:
    if name == "mirror":
        return False
    try:
        return bool(cards.role(name)["spell"])
    except KeyError:
        return False


def enemy_units(obs: dict) -> list:
    """Owner-1 non-tower entities (deploying ones included)."""
    return [e for e in obs.get("entities", []) if int(e[_OWNER]) == 1 and not e[_IS_TOWER]]


def _cluster(units: list) -> tuple[float, float]:
    best_key, best_group = None, None
    for u in units:
        group = [v for v in units if math.dist((u[_X], u[_Y]), (v[_X], v[_Y])) <= CLUSTER_RADIUS]
        key = (-(len(group) - 1), float(u[_Y]), int(u[_ID]))
        if best_key is None or key < best_key:
            best_key, best_group = key, group
    n = len(best_group)
    return sum(float(v[_X]) for v in best_group) / n, sum(float(v[_Y]) for v in best_group) / n


def _on_threat(units: list, obs: dict) -> tuple[float, float]:
    towers = [(float(e[_X]), float(e[_Y])) for e in obs.get("entities", [])
              if int(e[_OWNER]) == 0 and e[_IS_TOWER]]
    if not towers:  # game over; fall back to our king position
        towers = [(9.0, 3.0)]

    def key(u):
        d = min(math.dist((float(u[_X]), float(u[_Y])), t) for t in towers)
        return (d, int(u[_ID]))

    u = min(units, key=key)
    return float(u[_X]), float(u[_Y])


def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def position(p: int, card: str, obs: dict) -> tuple[float, float] | None:
    """Target (x, y) for placement p with this card, or None if p is computed and there's no enemy."""
    name, x, y = PLACEMENTS[p]
    if x is not None:
        return float(x), float(y)
    units = enemy_units(obs)
    if not units:
        return None
    x, y = _cluster(units) if name == "cluster" else _on_threat(units, obs)
    if not is_spell(card):
        y = min(y, GROUND_MAX_Y)
    return _clamp(x, X_MIN, X_MAX), _clamp(y, Y_MIN, Y_MAX)


def decode(index: int, obs: dict) -> list | None:
    """Action index -> [int slot, float x, float y], or None for wait."""
    if index == WAIT:
        return None
    slot, p = split(index)
    hand = obs.get("hand", [])
    card = hand[slot] if slot < len(hand) else ""
    pos = position(p, card, obs)
    if pos is None:  # computed placement with no enemy unit (masked out): wait
        return None
    return [int(slot), float(pos[0]), float(pos[1])]


def mask_reason(obs: dict, index: int) -> str:
    """'ok' if the mask allows ``index``, else which rule blocked it."""
    if index == WAIT:
        return "ok"
    slot, p = split(index)
    legal = obs.get("legal", [])
    hand = obs.get("hand", [])
    if slot >= len(legal) or not legal[slot] or slot >= len(hand):
        return "slot_not_legal"
    spell = is_spell(hand[slot])
    if not spell and p in ENEMY_TOWERS:
        return "ground_on_enemy_tower"
    if spell and p in FIXED_OWN_SIDE:
        return "spell_on_own_side"
    if p in COMPUTED and not enemy_units(obs):
        return "no_enemy_unit"
    return "ok"


def mask(obs: dict) -> list[bool]:
    """53 booleans; index 0 (wait) is always True."""
    return [mask_reason(obs, i) == "ok" for i in range(N_ACTIONS)]


def legal_indices(obs: dict) -> list[int]:
    """Indices the mask allows, excluding 0 (wait)."""
    return [i for i, ok in enumerate(mask(obs)) if ok and i != WAIT]

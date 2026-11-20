"""Scripted bot v0 (handoff step C4).

Five rules, tried in order; the first that returns an index wins, else wait.
Each rule is a module-level function ``(obs, st) -> index | None`` with
``st = state.parse(obs)`` so later guardrails can import them (W3).
"""
from __future__ import annotations

import math

from bot import cards, placements, state
from bot.policy import Policy
from bot.trackers import Trackers

P = placements.NAMES
OPENING_TICKS = 200
DEFEND_RATIO = 1.2
ATTACK_ELIXIR = 8.0
NOLEAK_ELIXIR = 9.5
SPELL_MIN_UNITS = 2


# --- helpers -------------------------------------------------------------

def usable(obs: dict, i: int) -> bool:
    """Slot i is affordable and not mirror (v0 never picks mirror)."""
    legal = obs.get("legal", [])
    hand = obs.get("hand", [])
    return i < len(legal) and i < len(hand) and bool(legal[i]) and hand[i] != "mirror"


def _cost(obs: dict, i: int) -> int:
    return int(obs.get("hand_cost", [0] * 4)[i])


def _non_spells(obs: dict) -> list[int]:
    return [i for i in range(len(obs.get("hand", []))) if usable(obs, i) and not placements.is_spell(obs["hand"][i])]


def _role(name: str) -> dict[str, bool]:
    try:
        return cards.role(name)
    except KeyError:
        return {}


def _type(name: str) -> str:
    try:
        return cards.card_type(name)
    except KeyError:
        return ""


def attack_lane(st: state.State) -> str:
    """Lane whose enemy princess tower has less hp (ties: L)."""
    return "R" if st.enemy_tower_hp["R"] < st.enemy_tower_hp["L"] else "L"


def _lr(lane: str, left: str, right: str) -> int:
    return P[left] if lane == "L" else P[right]


# C4 says tank / win_condition and no-leak cards go to back_L / back_R (y 2.5). In the simulator
# every ground troop deployed there walks into our own princess tower and stays stuck at about
# (5.2, 4.3) / (12.8, 4.3) forever, while still counting as lane defense. Use def_ (y 9.5, just in
# front of the princess tower) as the "from behind" tile until the sim pathing is fixed.
BEHIND = ("def_L", "def_R")


# --- rules ---------------------------------------------------------------

def _raw(name: str) -> dict:
    try:
        return cards.raw(name)
    except KeyError:
        return {}


RANGED_MIN = 5.0  # archers 5, wizard 5.5, musketeer 6


def ranged(name: str) -> bool:
    return float(_raw(name).get("range", 0) or 0) >= RANGED_MIN


def invisible(name: str) -> bool:
    return bool(_raw(name).get("invisible"))


def defending_hp(st: state.State, lane: str) -> int:
    """st.defense[lane] minus our win_condition units (they only target buildings, so they
    walk past threats; counting them kept the defend rule silent while a tower fell)."""
    return sum(
        u.hp for u in st.mine
        if u.lane == lane and u.y < state.THREAT_Y and not _role(u.name).get("win_condition")
    )


def defend_index(obs: dict, st: state.State) -> int | None:
    lane = max(("L", "R"), key=lambda ln: (st.threat[ln], -st.my_tower_hp[ln], ln == "L"))
    threat, defense = st.threat[lane], defending_hp(st, lane)
    threats = [u for u in st.enemy if u.lane == lane and u.y < state.THREAT_Y]
    ghost = any(invisible(u.name) for u in threats)
    if ghost:
        # in practice only ranged units hit an invisible attacker, so melee hp doesn't count as cover
        defense = sum(u.hp for u in st.mine if u.lane == lane and u.y < state.THREAT_Y and ranged(u.name))
    if not (threat > 0 and threat > DEFEND_RATIO * defense):
        return None
    cand = _non_spells(obs)
    if any(u.flying for u in threats):
        cand = [i for i in cand if _role(obs["hand"][i]).get("hits_air")]
    if not cand:
        return None
    if ghost:
        # Royal Ghost: troops can only lock on during the short reveal after each of its hits,
        # so melee defenders walk past it; ranged ones in range of the tower pick it up (J2 notes).
        far = [i for i in cand if ranged(obs["hand"][i])]
        if far:
            cand = far
    if len(threats) >= 3:
        splash = [i for i in cand if _role(obs["hand"][i]).get("splash")]
        if splash:
            cand = splash
    slot = min(cand, key=lambda i: (_cost(obs, i), i))
    if _type(obs["hand"][slot]) == "building":
        p = P["king_pull"]
    else:
        p = _lr(lane, "def_L", "def_R")
    return placements.encode(slot, p)


def opening_index(obs: dict, st: state.State) -> int | None:
    if st.tick < OPENING_TICKS and not st.enemy:
        return 0
    return None


def _damage_spell_radius(name: str) -> float | None:
    if name == "mirror" or not _role(name).get("spell"):
        return None
    c = cards.raw(name)
    r, dmg = c.get("radius"), c.get("damage") or 0
    if r and dmg > 0:
        return float(r)
    return None


def spell_index(obs: dict, st: state.State) -> int | None:
    enemies = placements.enemy_units(obs)
    if len(enemies) < SPELL_MIN_UNITS:
        return None
    best = None
    for i, name in enumerate(obs.get("hand", [])):
        if not usable(obs, i):
            continue
        r = _damage_spell_radius(name)
        if r is None:
            continue
        pos = placements.position(P["cluster"], name, obs)
        if pos is None:
            continue
        n = sum(1 for e in enemies if math.dist(pos, (float(e[2]), float(e[3]))) <= r)
        if n < SPELL_MIN_UNITS:
            continue
        key = (-n, _cost(obs, i), i)
        if best is None or key < best[0]:
            best = (key, i)
    if best is None:
        return None
    return placements.encode(best[1], P["cluster"])


def _troop_strength(name: str) -> float:
    c = cards.raw(name)
    return float(c.get("hp", 0) or 0) * float(c.get("count", 1) or 1)


def attack_index(obs: dict, st: state.State) -> int | None:
    if st.elixir < ATTACK_ELIXIR or st.threat["L"] != 0 or st.threat["R"] != 0:
        return None
    hand = obs.get("hand", [])
    cand = [i for i in range(len(hand)) if usable(obs, i) and _type(hand[i]) == "troop"]
    if not cand:
        return None

    def key(i):
        r = _role(hand[i])
        return (not r.get("win_condition"), not r.get("tank"), -_troop_strength(hand[i]), i)

    slot = min(cand, key=key)
    lane = attack_lane(st)
    r = _role(hand[slot])
    if r.get("tank") or r.get("win_condition"):
        p = _lr(lane, *BEHIND)
    else:
        p = _lr(lane, "bridge_L", "bridge_R")
    return placements.encode(slot, p)


def noleak_index(obs: dict, st: state.State) -> int | None:
    if st.elixir < NOLEAK_ELIXIR:
        return None
    cand = _non_spells(obs)
    if not cand:
        return None
    slot = min(cand, key=lambda i: (_cost(obs, i), i))
    return placements.encode(slot, _lr(attack_lane(st), *BEHIND))


RULES = (
    ("defend", defend_index),
    ("opening", opening_index),
    ("spell", spell_index),
    ("attack", attack_index),
    ("noleak", noleak_index),
)


class ScriptedPolicy(Policy):
    """Rule-based bot v0. Counters add up across matches (reset keeps them)."""

    def __init__(self):
        self.deck = None
        self.fires = {name: 0 for name, _ in RULES} | {"wait": 0}
        self.mask_fallbacks = 0
        self.last_rule = None
        self.trackers = Trackers()

    def reset(self, deck):
        self.deck = list(deck)
        self.trackers = Trackers()

    def act_index(self, obs):
        self.trackers.update(obs)  # C8 hook: no rule reads the trackers yet
        st = state.parse(obs)
        chosen, rule = 0, "wait"
        for name, fn in RULES:
            idx = fn(obs, st)
            if idx is not None:
                chosen, rule = idx, name
                break
        if not placements.mask(obs)[chosen]:
            self.mask_fallbacks += 1
            chosen, rule = 0, "wait"
        self.fires[rule] += 1
        self.last_rule = rule
        return chosen

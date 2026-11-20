"""Enemy elixir and card-cycle trackers (handoff C7 + C8, interface in section 3.4).

What we can see (probe, Nov 19, seat 1 played each card in a local Battle and seat 0's
``observe()["entities"]`` was printed for the next 40 ticks):

- Pure spells NEVER appear: fireball, poison, rocket, arrows, earthquake, zap, tornado,
  freeze, clone, rage, lightning, goblin_curse, barbarian_barrel and royal_delivery left no
  enemy entity at all. Their elixir is spent invisibly, so the estimate drifts high
  whenever the opponent casts one.
- goblin_barrel and graveyard show up only as what they drop: "goblins" 5 ticks after the
  cast, "skeletons" from tick 14 on. Those are deck-card names, so they are charged as
  goblins (2) / skeletons (1), not as the spell (3 / 5). Accepted undercount.
- Troops and buildings appear on the very first observation after the play (the next
  tick), with deploying=True. Multi-unit cards (minions, skeleton_army) appear as several
  entities with the same card name in the same update: charge once per name per update.
- Death spawns (golem -> golemite) appear after the spawner is gone, so spawner names from
  the previous update count too.

Usage::

    t = Trackers(); t.reset()
    t.update(obs)            # every decision tick (or irregularly); never raises
    t.enemy_elixir()         # float 0..10
    t.enemy_cards()          # [(card, could_be_in_hand), ...] <= 8, first-seen order
"""
from __future__ import annotations

from bot import cards

# entity field indices (see bot.state)
_OWNER, _CARD, _IS_TOWER, _ID = 0, 1, 7, 9

START_ELIXIR = 5.0
MAX_ELIXIR = 10.0
DOUBLE_TICK = 2400
OVERTIME_TICK = 3600
# elixir per tick in single / double / overtime
RATES = ((0, 1 / 56), (DOUBLE_TICK, 1 / 28), (OVERTIME_TICK, 3 / 56))

HAND_SIZE = 4  # cards in the opponent's hand; the last 4 distinct plays are not in it
MAX_DECK = 8

# evo form -> base card (an evo costs the base card's elixir)
EVO_BASE: dict[str, str] = {}
for _b in cards.DEPLOYABLE:
    _evo = cards.raw(_b).get("evolved")
    if isinstance(_evo, str):
        EVO_BASE[_evo] = _b

# units that put a deck-card name back on the board without a new play
_REBORN: dict[str, frozenset[str]] = {"phoenix_egg": frozenset({"phoenix"})}


def _spawns(name: str) -> frozenset[str]:
    try:
        return cards.spawned_by(name) | _REBORN.get(name, frozenset())
    except KeyError:
        return frozenset()


def regen(t0: int, t1: int) -> float:
    """Elixir one player gains from tick t0 to t1 (t1 >= t0), split at phase boundaries."""
    if t1 <= t0:
        return 0.0
    total = 0.0
    for i, (start, rate) in enumerate(RATES):
        end = RATES[i + 1][0] if i + 1 < len(RATES) else None
        lo = max(t0, start)
        hi = t1 if end is None else min(t1, end)
        if hi > lo:
            total += (hi - lo) * rate
    return total


class EnemyElixir:
    """Estimated opponent elixir from regen and the cost of each new card we see."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.value = START_ELIXIR
        self.last_tick: int | None = None
        self.seen_ids: set[int] = set()
        self.prev_spawners: frozenset[str] = frozenset()

    def regen_to(self, tick: int) -> None:
        start = 0 if self.last_tick is None else self.last_tick
        self.value = min(MAX_ELIXIR, self.value + regen(start, tick))
        self.last_tick = tick

    def charge(self, amount: float) -> None:
        self.value = max(0.0, self.value - amount)


class EnemyCycle:
    """Distinct enemy deck cards in first-seen order, and the full play order."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.known: list[str] = []
        self.plays: list[str] = []

    def play(self, card: str) -> None:
        self.plays.append(card)
        if card not in self.known and len(self.known) < MAX_DECK:
            self.known.append(card)

    def recent(self) -> list[str]:
        """The last HAND_SIZE distinct cards played, most recent first."""
        out: list[str] = []
        for c in reversed(self.plays):
            if c not in out:
                out.append(c)
                if len(out) == HAND_SIZE:
                    break
        return out

    def enemy_cards(self) -> list[tuple[str, bool]]:
        recent = set(self.recent())
        return [(c, c not in recent) for c in self.known]


class Trackers:
    """Owns EnemyElixir and EnemyCycle; ``update(obs)`` once per decision."""

    def __init__(self):
        self.elixir = EnemyElixir()
        self.cycle = EnemyCycle()

    def reset(self) -> None:
        self.elixir.reset()
        self.cycle.reset()

    def update(self, obs: dict) -> None:
        try:
            self._update(obs)
        except (KeyError, TypeError, ValueError, IndexError):
            pass  # a malformed field never takes the bot down

    def _update(self, obs: dict) -> None:
        tick = int(obs.get("tick", 0))
        if self.elixir.last_tick is not None and tick < self.elixir.last_tick:
            self.reset()  # a new match started without reset()
        self.elixir.regen_to(tick)

        enemy = [e for e in obs.get("entities", []) if int(e[_OWNER]) == 1 and not e[_IS_TOWER]]
        names_now = {str(e[_CARD]) for e in enemy}
        spawners = frozenset().union(*(_spawns(n) for n in names_now)) if names_now else frozenset()
        excluded = spawners | self.elixir.prev_spawners

        new_names: list[str] = []  # first-seen order within this update
        for e in enemy:
            eid = int(e[_ID])
            if eid in self.elixir.seen_ids:
                continue
            self.elixir.seen_ids.add(eid)
            name = str(e[_CARD])
            if name not in new_names:
                new_names.append(name)

        for name in new_names:
            if name in excluded:
                continue  # spawned by an enemy unit on the board now or last update
            card = EVO_BASE.get(name, name)
            if card not in cards.DEPLOYABLE:
                continue  # non-deck spawn-only name, or unknown
            self.elixir.charge(cards.elixir(card))
            self.cycle.play(card)

        self.elixir.prev_spawners = spawners

    def enemy_elixir(self) -> float:
        return float(self.elixir.value)

    def enemy_cards(self) -> list[tuple[str, bool]]:
        return self.cycle.enemy_cards()

    def n_known(self) -> int:
        return len(self.cycle.known)

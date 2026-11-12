"""cr_sim — deterministic Clash-Royale-style battle simulator for RL.

The heavy lifting is a native Rust core (`cr_sim._core`, built with maturin).
This package re-exports its public API and adds optional pure-Python helpers
(trace tools, adapters). The core has zero required third-party dependencies;
`gymnasium`/`numpy` are only needed for the optional adapters.

Hidden information / anti-cheat
-------------------------------
The raw ``Battle`` exposes per-player getters (``elixir(p)``, ``hand(p)``,
``next_card(p)``, ...) for *either* seat, which is convenient for the trusted
harness, debugging, and replay but lets a self-play agent peek at its opponent.
For competitive play, hand each agent a ``PlayerView`` instead::

    b = cr_sim.Battle()
    b.reset(seed, deck0, deck1)
    v0, v1 = b.view(0), b.view(1)          # one read-only window per agent
    action0 = agent0.act(v0)               # v0 can read its own hand/elixir and
    action1 = agent1.act(v1)               # the full board, but NOT v1's hand,
    b.step(action0, action1)               # next card, or elixir.

``PlayerView`` is a live, read-only window: it exposes the full board
(``entities``/``towers``/``projectiles``), its own ``hand``/``hand_cost``/
``next_card``/``elixir``/``crowns``/``champion_status``, the opponent's crown
count and tower state (public), and the fog-of-war ``observe()``/``legal()`` for
its seat. Its private getters take no player argument and there is no accessor
for the opponent's hand, next card, or elixir, so an agent *structurally* cannot
cheat. It has no mutators (``step``/``reset``/``activate_ability``).
"""
from __future__ import annotations

from ._core import (  # noqa: F401
    Battle,
    PlayerView,
    BuildCheck,
    WIDTH,
    HEIGHT,
    TICKS_PER_SECOND,
    REGULATION_TICKS,
    OVERTIME_TICKS,
    RIVER_Y,
    RIVER_HALF,
    BRIDGE_HALF,
    BRIDGES,
    ABILITY_SLOT,
)

__all__ = [
    "Battle",
    "PlayerView",
    "WIDTH",
    "HEIGHT",
    "TICKS_PER_SECOND",
    "REGULATION_TICKS",
    "OVERTIME_TICKS",
    "RIVER_Y",
    "BRIDGES",
    "ABILITY_SLOT",
]
__version__ = "0.1.0"

"""cr_sim — deterministic Clash-Royale-style battle simulator for RL.

The heavy lifting is a native Rust core (`cr_sim._core`, built with maturin).
This package re-exports its public API and adds optional pure-Python helpers
(trace tools, adapters). The core has zero required third-party dependencies;
`gymnasium`/`numpy` are only needed for the optional adapters.
"""
from __future__ import annotations

from ._core import (  # noqa: F401
    Battle,
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
)

__all__ = [
    "Battle",
    "WIDTH",
    "HEIGHT",
    "TICKS_PER_SECOND",
    "REGULATION_TICKS",
    "OVERTIME_TICKS",
    "RIVER_Y",
    "BRIDGES",
]
__version__ = "0.1.0"

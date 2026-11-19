"""Policy contract shared by every bot (handoff section 3.1).

A policy works from the obs dict alone, so the same class runs in local games
and on the server. ``act`` is called only on ticks where
``obs["tick"] % DECIDE_EVERY == 0``; anything that needs memory (trackers)
works from the tick difference since the last call.

Action indices (defined in ``bot.placements``, not here): index 0 = wait,
index = 1 + slot*13 + p for placement p, N_ACTIONS = 53.
"""
from __future__ import annotations

import random

DECIDE_EVERY = 10  # ticks between decisions (0.5 s of game time)


class Policy:
    """Base class. Subclasses implement ``act_index``; ``act`` decodes it."""

    def reset(self, deck):
        """Called at the start of every match with our 8 card names."""
        pass

    def act_index(self, obs):
        """obs dict -> action index 0..N_ACTIONS-1 (0 = wait)."""
        raise NotImplementedError

    def act(self, obs):
        """obs dict -> [slot, x, y] or None; this is what the server receives."""
        from bot import placements  # imported here so bot.policy loads without placements

        return placements.decode(self.act_index(obs), obs)


class RandomPolicy(Policy):
    """Random legal moves, reproducible from ``seed``.

    Waits with probability 0.6; otherwise picks uniformly among the non-wait
    indices that ``placements.mask(obs)`` allows (wait if there are none).
    """

    WAIT_PROB = 0.6

    def __init__(self, seed=None):
        self.rng = random.Random(seed)
        self.deck = None

    def reset(self, deck):
        self.deck = list(deck)

    def act_index(self, obs):
        from bot import placements  # imported here so bot.policy loads without placements

        if self.rng.random() < self.WAIT_PROB:
            return 0
        choices = placements.legal_indices(obs)
        return self.rng.choice(choices) if choices else 0

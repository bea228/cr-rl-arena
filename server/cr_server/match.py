"""Server-authoritative match runner.

The server owns the one true `cr_sim.Battle`. Agents never step their own sim
during a ranked match; they only receive observations and return actions. Because
the engine is deterministic, the full match is reproducible from
`(seed, actions)` — the returned log can be replayed or audited.

An "agent" here is any object with:
    .name : str
    .deck : list[str] | None            (8 card names, or None for default)
    async def act(obs: dict, timeout: float) -> action | None
where action is (slot, x, y) or None (no-op). A timeout / error / malformed
reply is treated as a no-op, so one misbehaving agent cannot stall or crash a match.
"""
from __future__ import annotations

import asyncio
from numbers import Real

# Decide at 5 Hz (every 4 ticks at 20 tps). Deploys are instantaneous, so acting
# between decision points is unnecessary and keeps round-trips bounded (~900/match).
DECISION_EVERY = 4


def _validate(action):
    if action is None:
        return None
    try:
        slot, x, y = action
        if not isinstance(slot, int) or not isinstance(x, Real) or not isinstance(y, Real):
            return None
        return (int(slot), float(x), float(y))
    except (TypeError, ValueError):
        return None


async def run_match(sim, agent0, agent1, *, seed, decision_every=DECISION_EVERY,
                    move_timeout=1.0, max_ticks=None):
    b = sim.Battle()
    b.reset(seed, agent0.deck, agent1.deck)
    if max_ticks is None:
        max_ticks = sim.REGULATION_TICKS + sim.OVERTIME_TICKS + 2

    actions = []
    out = 0
    t = 0
    while out == 0 and t < max_ticks:
        if t % decision_every == 0:
            obs0, obs1 = dict(b.observe(0)), dict(b.observe(1))
            a0, a1 = await asyncio.gather(
                agent0.act(obs0, move_timeout),
                agent1.act(obs1, move_timeout),
            )
            a0, a1 = _validate(a0), _validate(a1)
            if a0 is not None:
                actions.append([t, 0, list(a0)])
            if a1 is not None:
                actions.append([t, 1, list(a1)])
            out = b.step(a0, a1)
        else:
            out = b.step(None, None)
        t += 1

    return {
        "outcome": out,                 # 0 running(shouldn't happen), 1 p0, 2 p1, 3 draw
        "crowns": list(b.crowns()),
        "ticks": t,
        "seed": seed,
        "actions": actions,             # [[tick, player, [slot,x,y]], ...] -> replayable
    }


def replay(sim, seed, actions, deck0=None, deck1=None):
    """Re-simulate a match from its (seed, actions) log and return the outcome.
    Used by the server to audit a result, and by anyone to verify a ladder game."""
    b = sim.Battle()
    b.reset(seed, deck0, deck1)
    by_tick = {}
    for t, p, a in actions:
        by_tick.setdefault(t, [None, None])[p] = tuple(a)
    out, t, maxt = 0, 0, sim.REGULATION_TICKS + sim.OVERTIME_TICKS + 2
    while out == 0 and t < maxt:
        a0, a1 = by_tick.get(t, [None, None])
        out = b.step(a0, a1)
        t += 1
    return {"outcome": out, "crowns": list(b.crowns()), "ticks": t}

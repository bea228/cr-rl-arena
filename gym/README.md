# cr-sim

A deterministic, fixed-point **Clash-Royale-style battle simulator** built for
reinforcement-learning coursework and competition. Native Rust core, Python API.

- **Deterministic**: all gameplay math is Q16.16 fixed-point, so a match is
  bit-identical across machines given the same seed + action log. Replays and
  the competition server agree with every student's local run.
- **Fast**: native Rust engine; trains fine on modest laptops.
- **Accessible**: `pip install` a prebuilt wheel — no Rust toolchain needed.
  Plain Python API with zero required deps; optional `gymnasium`/`pettingzoo`
  adapters for those who want them.

Status: scaffolding + fixed-point core landing. Engine, env API, competition
server, and web replay viewer in progress.

## Stepping many battles at once

`BatchBattle` exists purely for throughput. It decides nothing about rewards,
observations or action encodings, and each battle in it plays out exactly as it
would stepped alone with `Battle.step`.

```python
import cr_sim

bb = cr_sim.BatchBattle(256)            # threads=None: one worker per core
bb.reset(range(256))                    # one seed per battle
outs = bb.step(acts0, acts1)            # one tick; lists of (slot, x, y) | None
for i, o in enumerate(outs):
    if o:                               # finished battles are left untouched
        bb[i].reset(seed=next_seed())
obs = bb[7].view(0).observe()           # bb[i] is an ordinary Battle
```

`step` advances every battle one tick across CPU cores with the GIL released.
`bb.outcomes()`, `bb.done()`, `bb.ticks()`, `bb.crowns()` and `bb.entities()`
return one entry per battle in a single call. Results do not depend on the
thread count.

## Dev build

```bash
cd gym
python3 -m venv .venv && . .venv/bin/activate
pip install maturin
maturin develop --release
python -c "import cr_sim; print(cr_sim.BuildCheck.fixed_selftest())"  # ~1.41421
```

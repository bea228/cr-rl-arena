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

## Dev build

```bash
cd gym
python3 -m venv .venv && . .venv/bin/activate
pip install maturin
maturin develop --release
python -c "import cr_sim; print(cr_sim.BuildCheck.fixed_selftest())"  # ~1.41421
```

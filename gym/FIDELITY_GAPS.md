# Fidelity gaps vs. real Clash Royale

An honest, exhaustive accounting of where the `cr_sim` engine diverges from the
real game. The simulator faithfully models the **macro loop** — elixir tempo,
8-card cycle, two lanes + bridges, tower hitboxes, target commitment, collision,
crowns, and overtime — and is deterministic and self-consistent. It is **not** a
card-complete or frame-exact reimplementation. An agent trained here learns *this*
game well; it would **not** transfer to real Clash Royale as-is.

Tags: `[BUG]` wrong vs real CR (should fix) · `[SIMPLIFIED]` approximation of a
real mechanic · `[MISSING]` real mechanic absent entirely.

Severity (RL impact): **H** changes learned strategy · **M** noticeable ·
**L** minor/edge.

---

## Resolved in the 2026-11-11 mechanics pass

- **King activation bug** — king now wakes only when it is damaged or a Princess
  tower is destroyed (not on Princess chip damage).
- **Deploying units are no longer targetable** until placement completes.
- **Per-card first-hit timing** — `load_time` per card replaces the flat 0.2 s.
- **Mass-weighted push** — collisions split inversely to `mass` (heavier moves less).
- **Shields** — `shield` hp absorbs before body hp (Guards, Dark Prince).
- **Death-spawns** — `death_spawn{card,count}` (Golem→golemites, Tombstone, …).
- **Spawner buildings** — `spawn{card,count,interval}` (Tombstone, Goblin Hut, Witch).
- **Status effects** — stun/freeze, slow, rage now modeled; movement + attack
  cadence honor them.
- **Spell effects** — `effect` = poison (DoT zone) / rage / heal / freeze / slow,
  via a lingering area-effect system.
- **Per-card deploy time** and a summon-only (`deployable:false`) flag.
- **Roster expanded** to ~37 cards with level-11 stats exercising all of the above.

Everything below remains outstanding. Verified still-deterministic after the pass.

---

## 1. Confirmed incorrect rules (bugs, not just simplifications)

- `[BUG]` **H — King-tower activation is too eager.** The king wakes whenever
  *any* friendly tower takes *any* damage (`engine.rs::apply_damage`). Real rule:
  the king activates only when **it** is damaged or when a **Princess tower is
  destroyed** — chip damage to a Princess tower must not wake the king. Current
  behavior lets a single arrow on a Princess tower activate the king early.
- `[BUG]` **M — Deploying units are targetable.** During the 1 s deploy a unit is
  non-colliding but still `alive`, so it can be targeted and attacked. In CR a
  troop is not a valid target until placement completes.
- `[BUG]` **L — Attack wind-up resets on any target change but not on leaving/re-entering range symmetrically.** The generic 0.2 s wind-up (`engaged` flag) is a
  coarse stand-in for per-card load/first-hit timing (see §3).

## 2. Timing & tick model

- `[SIMPLIFIED]` **H — Flat 0.2 s attack wind-up for every card.** Real CR has
  per-card **load time** and **first-hit** values; melee trades (Mini-PEKKA vs
  tank, Prince connect, Inferno lock) depend on them. One global constant is the
  single most impactful *mechanical* inaccuracy after the roster.
- `[SIMPLIFIED]` **M — 20 Hz tick rate.** The real engine advances on a finer
  internal timestep, so hit speeds, deploy times, and elixir quantize to 50 ms
  here and are not all frame-exact.
- `[SIMPLIFIED]` **L — Hit cadence applies damage at cooldown edge**, not at the
  specific damage frame within an attack animation.

## 3. Card roster & per-card mechanics — the largest gap

The gym ships **18 generic cards**; real CR has ~120, most with unique logic that
*is* the strategy. Entire mechanic classes are absent:

- `[MISSING]` **H — Death spawns / splits:** Golem→golemites, Lava Hound→pups,
  Giant Skeleton bomb, Balloon death bomb, Goblin Barrel, Skeleton Barrel,
  Wall Breakers, Elixir Golem blobs, Goblin Giant.
- `[MISSING]` **H — Spawner buildings:** Tombstone, Barbarian/Goblin Hut, Furnace,
  Goblin Drill; and **Elixir Collector** (an entire economic axis).
- `[MISSING]` **H — Charge / dash variants:** only a generic Prince-style charge
  exists. Dark Prince (shield), Battle Ram, Ram Rider (charge+snare), Bandit
  (dash+invuln), Royal Hogs, Elite Barbarians, Lumberjack (death Rage).
- `[MISSING]` **H — Ramp / charge-up attacks:** Inferno Tower & Inferno Dragon
  (ramping damage), Sparky (charged shot, reset on stun), Mortar/X-Bow (siege).
- `[MISSING]` **H — Shields:** Dark Prince, Guards, Royal Recruits (absorb one hit).
- `[MISSING]` **H — "Deploy anywhere" win conditions:** Miner, Goblin Drill,
  Graveyard, Goblin Barrel land on the **enemy** side. The gym restricts troop
  deploys to your own territory (+ opened pocket), so these can't be represented.
- `[MISSING]` **M — Reflect / on-hit effects:** Electro Giant (zaps attackers),
  Fisherman (hook/pull), Mega Knight (jump + spawn/landing damage).
- `[MISSING]` **M — Revive / multi-stage:** Phoenix revive; Golem/Lava/Goblin
  Giant/Elixir Golem staged splits.
- `[MISSING]` **M — Invisibility:** Royal Ghost.
- `[MISSING]` **M — Champions** (one-per-deck + activatable ability): Golden
  Knight, Skeleton King, Archer Queen, Mighty Miner, Monk, Little Prince, etc.
- `[MISSING]` **M — Evolutions** (evolved stats/behavior + evo-cycle counter).
- `[SIMPLIFIED]` **M — Multi-unit spawn formation** uses a simple line offset, not
  CR's per-card spawn formations/positions.

## 4. Spells & status effects

The spell system is **instant area damage + optional stun only**
(`engine.rs::cast_spell`). Missing:

- `[MISSING]` **H — Lingering/area effects:** Poison & Earthquake (DoT over time,
  Earthquake building bonus), Rage (speed/attack buff zone), Heal, Tornado (pull),
  Graveyard (timed skeleton spawns), Clone, Freeze (duration lock), Barbarian
  Barrel / Log (rolling line + knockback/rollthrough), Giant Snowball & Ice spells
  (slow), Lightning (3 highest-HP targets + stun), Fireball/Rocket knockback.
- `[SIMPLIFIED]` **M — Spells are near-instant** (high projectile speed), so there
  is no cast→travel dodge window as in CR.
- `[SIMPLIFIED]` **L — Crown-tower damage reduction** is a flat `crown_pct`
  (~30%); real per-spell crown-tower damage values differ.

## 5. Movement & pathfinding

- `[SIMPLIFIED]` **H — No real pathfinding.** Units "aim at the nearest bridge
  mouth, then the target," with river-blocking plus collision push. They do **not
  route around** placed buildings (Cannon, Tombstone) — they shove against them.
- `[SIMPLIFIED]` **M — Bank-hugging / corner-clipping** near the river differs
  from CR's deterministic tile pathing; the river block simply freezes vertical
  movement off-bridge (`engine.rs::block_river`).
- `[SIMPLIFIED]` **L — Instant turning** (no facing/turn delay).

## 6. Collision & physics

- `[SIMPLIFIED]` **H — Symmetric separation (no mass).** Each body pushes the
  other equally. Real CR is **mass/push-weight** based: a Knight shoves a Skeleton
  far more than the reverse, and some units resist pushing. This changes surrounds,
  pig-pushing, and tank-behind-support positioning.
- `[SIMPLIFIED]` **L — Two relaxation passes/tick**; dense crowds settle
  approximately, not exactly as CR's solver would.

## 7. Targeting & aggro

- `[SIMPLIFIED]` **M — Leash distances are invented** (`sight + 2.5`), not CR
  values; retarget/lock thresholds therefore differ.
- `[SIMPLIFIED]` **M — Acquisition is nearest-valid-in-sight.** CR has specific
  target-priority and aggro-pull rules (pulling a unit off its lane) that are only
  approximated.
- `[SIMPLIFIED]` **L — Tower targeting** locks nearest and holds; real tower
  retarget timing/priority is approximated.

## 8. Deploy mechanics

- `[SIMPLIFIED]` **M — Flat 1.0 s deploy time** for all cards; real deploy/placement
  times vary by card.
- `[MISSING]` **M — Enemy-side deploys** (see Miner/Graveyard/Drill/Barrel in §3).
- `[SIMPLIFIED]` **L — Deployable rows** use `y < 15` (river rows 15–16); close to
  but not pixel-matched with CR's deploy boundary and the exact pocket shape after
  a tower falls.

## 9. Elixir & economy

- `[MISSING]` **M — Elixir Collector / Pump** (generates elixir).
- `[MISSING]` **L — Mirror** (next card at +1 elixir).
- `[SIMPLIFIED]` **L — Overtime elixir multiplier/duration** (2 min, 3×) is a
  reasonable config, not matched to a specific CR mode/patch.
- Correct: start 5, cap 10, 2.8 s regen, 2× starting at 1:00 remaining.

## 10. Towers & win conditions

- `[SIMPLIFIED]` **M — Overtime tiebreak** = lowest remaining tower HP; approximates
  CR's rule but isn't matched to a specific mode.
- `[SIMPLIFIED]` **L — Tower stats** (range 7.5/7.0, hit speeds, HP) are documented
  level-11 values, not verified against authoritative tables.
- Correct: Princess = 1 crown, King = 3 and ends the match; king dormant until
  activated (modulo the activation bug in §1).

## 11. Projectiles

- `[SIMPLIFIED]` **M — Projectiles home to the target entity**; most real CR
  projectiles commit to a position and can be dodged by movement.
- `[MISSING]` **L — Projectile-specific behavior:** pierce (Magic Archer, Dart
  Goblin line), returning axe (Executioner), bounce/pierce (Bowler), recoil.

## 12. Stat accuracy

- `[SIMPLIFIED]` **M — Card numbers are hand-entered level-11 values** believed
  close but **not verified** against the decoded data tables. HP/damage are likely
  near; **hit speeds, ranges, and projectile speeds are the most likely to be off**.
  `gym/data/cards.json` is the single source to correct for exact parity.

## 13. Interface / observability caveats (sim design, not CR mechanics)

- The competition `obs` correctly **hides** opponent hand, next card, and elixir
  (only the full board + your own hand/elixir/crowns are exposed) — this matches
  CR's hidden information. **But** the raw `Battle` API exposes `elixir(opponent)`,
  `hand(opponent)`, etc.; local self-play can "cheat" unless restricted to
  `observe(player)`.
- `[SIMPLIFIED]` **M — Decision cadence.** The server requests actions at 5 Hz
  (every 4 ticks); CR lets you act on any frame. Fine for deploys (instantaneous)
  but caps action granularity.

## 14. Scope not modeled at all

- `[MISSING]` 1v1 only — no 2v2, draft, Touchdown, or event rulesets.
- `[MISSING]` No trophy/ladder matchmaking (server pairs by queue order, not skill
  bands); the Elo ladder is the competition's own, not CR's.
- `[MISSING]` No pre-game (deck-draw order is a seeded shuffle; CR's exact initial
  draw rules are approximated).

---

## Priority order to close gaps (suggested)

1. **Correctness quick wins:** fix king activation (§1), deploying-unit
   targetability (§1).
2. **Per-card timing:** add `first_hit` / `load_time` fields and use them instead
   of the flat wind-up (§2, §3).
3. **Status-effect & spell framework:** slow / freeze / stun / rage / DoT /
   knockback / pull, so most spells and control cards become expressible (§4).
4. **Death-spawns & spawner buildings** (§3) — unlocks a large slice of the roster.
5. **Mass-weighted push** (§6) and **building-aware pathfinding** (§5).
6. **Verify stats** against authoritative tables (§12).

None of these block the hackathon: the current engine is a coherent, deterministic
RTS. They are what it would take to approach real-CR parity.

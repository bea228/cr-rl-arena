# Remaining fidelity gaps — work list (2026-11-12)

Authoritative checklist of what's left after the pathfinding rewrite, the 85→120
roster expansion, the observability guard, and the medium-gap pass. Work the
**actionable** items top-down; **scope decisions** are confirmed-and-documented,
not implemented.

## How to work this list (worker instructions)
- **Determinism is sacred.** All gameplay math is fixed-point Q16.16 (`src/fixed.rs`).
  No floats in gameplay state, no order-dependent `HashMap` iteration, no wall-clock,
  no unseeded RNG. Two runs of a seed must stay bit-identical.
- **Real values only.** Pull stats/ability data from the decoded tables at
  `/home/clio/projects/clashroyale/client-new/data/csv_logic/` (characters/*.toml,
  projectiles.toml, area_effect_objects.toml, spells_*.csv, actions.toml). Level-11 =
  `round(base × 2.56)`; times/1000, ranges/1000, Speed/60, Mass/10. A prebuilt index
  of every object is at the session scratchpad `game_index.json` (+ `resolve_stats.py`).
- **One item at a time.** For each item: make the real change, add/adjust a Rust
  `#[test]` proving it, run `cd gym && cargo test --release` (green), then tick the box
  here with a one-line note **and the commit SHA**, and `git commit` (author
  `Alex <42507764+ilumn@users.noreply.github.com>`, **no** Claude attribution footer).
- **Check a box only when the code actually does it.** A ticked box with no real diff +
  passing test behind it will be rejected on audit. If an item can't be done faithfully,
  mark it `[~]` DEFERRED with the concrete reason, don't fake a `[x]`.
- Do **not** run `maturin develop`/`pytest` (shared venv); `cargo test` is your gate.
- Keep edits off the pathfinding core (`PathGrid`/`PathCache`/`path_goal`) unless an item
  requires it.

---

## A. Actionable

- [x] **A1 — New champions' activated abilities.** Mighty Miner, Monk, Little Prince,
  Boss Bandit are deck-legal with real combat but no activated ability (`ability` = None).
  Model each ability faithfully using the decoded ability data (cost/cooldown/duration/
  damage from the character TOMLs + `ManaCostFromSummonerMana` in `spells_characters.csv`).
  Existing `AbilityKind` is `Cloak|Summon|Dash` (see `archer_queen`/`skeleton_king`/
  `golden_knight` in `cards.json` and `activate_ability` in `engine.rs`). Add new ability
  kinds if a champion needs one (e.g. Monk's deflect/pushback, Little Prince's power shot /
  Guardian, Mighty Miner's bomb-dash), or approximate to the nearest kind and document it.
  Acceptance: each of the 4 champions has a working `ability{...}` that fires via
  `Battle.activate_ability`, with a test per champion asserting the effect.
  DONE: added 3 new `AbilityKind`s (`LaneBomb`, `Deflect`, `Warp`) + reused `Summon`.
  Mighty Miner `lanebomb` = switch to mirror lane (reflect x) + drop a 333-dmg (L11 of
  MightyMinerExplosion BaseDamage 130) / 3.0-tile-radius / 1.8-knockback blast at the old
  spot; Monk `deflect` = 4.0s (AbilityStateDuration 4000) 65%-damage-reduction window
  (ShieldBoostMonk DamageReduction=65), applied in `apply_damage`; Little Prince `summon` =
  1 `guardian` (ChampionGuard 1600hp/233dmg, added as new `deployable:false` card); Boss
  Bandit `warp` = 6-tile (WarpY 6000) forward warp + 1.0s (SpawnTime 1000) cloak + target
  reset. Costs (ManaCost) exact: MM/Monk/BB=1, LP=3. Cooldowns: Boss Bandit 3.0s exact
  (Cooldown=3000); MM 3.5s / Monk 12.0s / LP 20.0s are documented APPROXIMATIONS — the
  decoded tables store only the leveled stat-tag `ability_cooldown`, not its numeric. Tests:
  `mighty_miner_lanebomb_damages_and_switches_lane`, `monk_deflect_reduces_incoming_damage`,
  `little_prince_summons_guardian`, `boss_bandit_warps_forward_and_cloaks`,
  `champion_abilities_are_deterministic`. `cargo test --release`: 29/29 pass. Commit: a8bfadb
- [ ] **A2 — Hunter shotgun burst + Firecracker recoil.** Hunter currently fires one
  weak projectile (per-pellet damage); real Hunter deals many pellets (big point-blank
  burst, falls off with range). Model its effective multi-pellet damage from the data
  (pellet count × per-pellet) without overpowering it at range. Firecracker should apply
  its self-recoil knockback on firing. Acceptance: Hunter kills far faster point-blank
  than at max range; a test covers both; Firecracker recoils. Resolve real numbers from
  the tables (HunterProjectile / `MultipleProjectiles` / Firecracker recoil fields).
- [ ] **A3 — Battle Healer heal aura.** Battle Healer continuously heals nearby allies
  (and herself) in a radius. Model it (reuse the area-effect/heal system — see `Heal`
  `SpellEffect` and `apply_area_effects`). Acceptance: allies in her radius regain hp over
  time; a test asserts a damaged nearby ally heals while she's alive. Pull heal rate/radius
  from the decoded data.
- [ ] **A4 — Mixed-composition cards (Goblin Gang, Rascals).** These deploy a MIX of unit
  types in one card (Goblin Gang = 3 stab goblins + 3 spear goblins; Rascals = 1 Rascal Boy
  tank + 2 Rascal Girls ranged). The engine spawns `count` of ONE def. Add a
  secondary-spawn-on-deploy mechanism (e.g. a `deploy_spawn{card,count}` field, or reuse the
  spawn system at deploy) so these are faithful, then add the two cards to `cards.json` with
  real stats. Acceptance: deploying each yields the correct mixed group; a test verifies the
  counts/types. (Add the summon-only sub-units as `deployable:false`.)
- [ ] **A5 — Targeting/aggro exactness.** M1 set the leash to `sight.max(range)`; refine
  acquisition toward CR's real rules where the data supports it: correct retarget timing,
  and not dropping a committed target prematurely. Keep it deterministic (nearest valid;
  ties → lower entity id). Acceptance: a test showing a unit holds its committed target
  within leash and only re-acquires when it should.
- [ ] **A6 — Crown-tower per-spell damage.** Verify each spell's `crown_tower_damage_pct`
  against the decoded `CrownTowerDamagePercent` / `{BaseDamage,TowerDamage}` structs and
  correct any that are placeholders. Acceptance: the spells' crown percentages match the
  tables; note any corrected in the checklist.
- [ ] **A7 — Curse / Clone / stretch cards (Mother Witch, Goblin Curse, Clone, Suspicious
  Bush, Goblinstein).** Implement the ones whose mechanic maps cleanly; for any that need a
  genuinely new system beyond reasonable scope, mark `[~]` DEFERRED with the specific reason
  (don't half-add a misleading version). Document each decision.
- [ ] **A8 — Dependabot vulnerabilities (3: 1 high / 1 moderate / 1 low).** Identify the
  flagged advisories (`gym` + `server` dependency manifests) and bump conservatively to
  patched versions; `cargo test` must stay green. If a bump can't be done safely, record the
  advisory IDs and why. Acceptance: the fixable advisories resolved; build + tests green.

## B. Scope decisions (confirm & document — do NOT implement)

- [ ] **B1 — Tick/decision granularity.** Keep the deterministic 20 Hz core tick (changing
  it rescales every tuned stat); decision cadence is already configurable (`CR_DECISION_EVERY`).
  Confirm + document as by-design.
- [ ] **B2 — Pathfinding sub-tile exactness / instant turning.** The tile flow field is
  faithful at tile granularity; sub-tile bank-hugging and facing/turn delay are intentionally
  not modelled. Confirm + document.
- [ ] **B3 — Game modes.** 1v1 only; no 2v2, draft, Touchdown, or event rulesets. By design
  for the hackathon. Confirm + document.
- [ ] **B4 — Matchmaking.** Server pairs by queue order with its own Elo ladder, not CR
  trophy bands. By design. Confirm + document.
- [ ] **B5 — Event-only card variants.** Super/Party/Chess variants are out of standard-
  ladder scope. Confirm + document.

---

## Audit log (filled by the supervising session)
- (pending)

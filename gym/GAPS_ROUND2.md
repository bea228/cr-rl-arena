# Remaining-gaps round 2 — work list (2026-11-12)

The last pass (REMAINING_GAPS.md) deferred two card mechanics and left two smaller
items open. This round closes them. Same discipline as before.

## How to work this list (worker instructions)
- **Determinism is sacred.** Fixed-point Q16.16 (`src/fixed.rs`); no floats in gameplay
  state, no order-dependent `HashMap` iteration in gameplay, no wall-clock/unseeded RNG.
  Seeds replay bit-identically. Add a determinism-guard test for any new stateful mechanic.
- **Real values only**, from the decoded tables at
  `/home/clio/projects/clashroyale/client-new/data/csv_logic/` (characters/*.toml,
  area_effect_objects.toml, projectiles.toml, actions.toml, spells_*.csv). Level-11 =
  `round(base × 2.56)`; times/1000, ranges/1000, Speed/60, Mass/10. Prebuilt index +
  resolver in the session scratchpad: `game_index.json`, `resolve_stats.py` (and grep the
  raw TOMLs directly — the index is known to miss some fields, e.g. ability cooldowns).
- **One item at a time**: real change + a Rust `#[test]` that proves it + full
  `cd gym && cargo test --release` green, then tick the box here with a one-line note **and
  the implementation commit SHA**, and commit (author
  `Alex <42507764+ilumn@users.noreply.github.com>`, **no** Claude attribution footer). One
  commit per item.
- **Tick `[x]` only when the code genuinely does it** (real diff + passing test). If an item
  truly can't be done faithfully, mark `[~]` DEFERRED with the specific technical reason —
  never fake a checkmark. This is audited.
- Do NOT run `maturin develop`/`pytest` (shared venv); `cargo test --release` is your gate.
- Patterns to reuse (all already in the engine): `deploy_spawn{card,count}` (mixed deploy),
  `death_spawn{card,count}`, on-hit status accumulators in `apply_hit`/`proj_impact`, the
  champion `ability{kind,...}` + `activate_ability` + `AbilityKind` enum, `SpellEffect`/
  `apply_area_effects`. Add new optional `CardDef`/`RawCard` fields with defaults so the
  other 128 cards keep loading. Summon-only units are `deployable:false`.

---

## Items

- [ ] **R1 — Mother Witch (curse-on-hit).** Internal name `WitchMother` (mana 4 Legendary;
  NOT `DarkWitch`, which is Night Witch and already in as `night_witch`). A ranged attacker
  whose hits **curse** the enemy; a cursed enemy that dies spawns a small "pig"/hog **for the
  Mother Witch's owner** (side-flips the kill). The prior pass deferred this because the curse
  must ride a projectile and spawn a unit on the victim's later death. Implement it: a
  projectile/hit that tags the victim with a `cursed_by<owner>` state, and on that unit's
  death spawn N curse-pigs for that owner. Resolve the real stats + pig unit + counts from the
  tables (WitchMother + its spawn/curse objects). Add `mother_witch` (+ the summon-only pig
  unit). Acceptance: a test where an enemy hit by Mother Witch then killed spawns the pig on
  HER side; an un-cursed death does not; determinism guard.
- [ ] **R2 — Goblinstein.** Internal `goblinstein` (mana 5 Champion). It deploys as **two
  linked parts** — the big melee **Monster** + the small **Doctor** that periodically zaps /
  revives. Model the champion-pet pairing (e.g. the champion's deploy also spawns its pet via
  the existing deploy-spawn/summon path), each with real stats, and wire its activated
  ability if the data defines one (else `ability` None + documented). Resolve the two parts'
  internal names + stats + the ability from the tables. Add `goblinstein` (+ the pet as a
  summon-only unit). Acceptance: deploying Goblinstein yields both parts on the owner's side;
  one-champion-per-deck still enforced; a test verifies the pairing; determinism guard. If the
  Doctor's revive/zap needs a genuinely new subsystem beyond reasonable scope, implement the
  pairing + combat faithfully and mark the specific sub-mechanic `[~]` with the reason.
- [ ] **R3 — Real champion ability cooldowns.** A1 left Mighty Miner / Monk / Little Prince
  cooldowns as approximations because the CSV `AbilityCooldown` column was empty. Dig deeper —
  check `actions.toml`, the `[ABILITY.*]`/`[ACTION.*]` blocks in each champion's TOML, and any
  leveled cooldown stat table — for the real numeric cooldowns, and correct `cards.json` if
  found. If they are genuinely absent everywhere in the decoded data, mark `[~]` with proof of
  where you looked and keep the documented approximations. Acceptance: either corrected values
  (with source) or a documented, evidenced deferral. (Boss Bandit's 3.0s is already real.)
- [ ] **R4 — pyo3 advisory RUSTSEC-2025-0020 (low).** Currently recorded-not-fixed because the
  patched line is a breaking 0.22→0.24 major bump. Attempt the upgrade conservatively: bump
  pyo3 to the patched release, fix the API breakage across `src/env.rs`/`src/lib.rs`/the
  `PlayerView` code, keep determinism and all tests green. If the migration proves invasive or
  risks the PyO3 surface, STOP, revert cleanly, and mark `[~]` DEFERRED documenting the exact
  breaking changes encountered and that the vulnerable API is unused (current mitigation). Do
  not leave a half-migrated build. This item is LAST.

---

## Audit log (filled by the supervising session)
- (pending)

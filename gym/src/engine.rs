//! The deterministic battle engine. One `tick()` = one simulation step
//! (1/TICKS_PER_SECOND seconds). All state transitions are pure functions of
//! (previous state, actions, rng) using fixed-point math, so a match replays
//! bit-identically from its seed + action log.

use crate::arena::*;
use crate::cards::{AbilityKind, CardType, Library, SpellEffect, Targets};
use crate::entity::{Entity, Projectile};
use crate::fixed::{Fx, V2};
use crate::rng::Rng;

/// [champ] Reserved `Action.slot` value meaning "activate my champion's ability".
/// Used so ability activation rides the same ordered action stream as deploys
/// (replayable/deterministic). `Battle.activate_ability(player)` is the direct API.
pub const ABILITY_SLOT: i32 = 100;

/// A per-tick action for one player. `slot < 0` means "do nothing".
#[derive(Clone, Copy)]
pub struct Action {
    pub slot: i32, // hand index 0..3, or <0 for no-op
    pub x: Fx,
    pub y: Fx,
}

impl Action {
    pub fn noop() -> Action {
        Action { slot: -1, x: Fx(0), y: Fx(0) }
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Outcome {
    Running,
    Win0,
    Win1,
    Draw,
}

const HAND: usize = 4;

// [inv] How long an invisible unit (Royal Ghost) stays targetable after it
// lands an attack. 15 ticks @ 20 tps = 0.75s, inside CR's ~0.5-1s reveal window.
const REVEAL_TICKS: i32 = 15;

#[derive(Clone, Copy, PartialEq)]
enum AeoKind {
    Damage, // DoT to enemies (poison, earthquake)
    Rage,   // buff allies
    Heal,   // heal allies
    Slow,   // slow enemies
    Freeze, // freeze enemies
    Pull,   // [kb] pull enemies toward center each tick + small DoT (Tornado)
    Spawn,  // [edep] periodically summon units over a radius (Graveyard)
}

// [kb] Knockback/pull tuning. A body at mass <= KB_REF_MASS takes the full push;
// heavier bodies resist in proportion (very-high-mass units are dragged least).
const KB_REF_MASS: Fx = Fx::from_raw(crate::fixed::ONE / 2); // 0.5 tiles of "mass"
// [kb] Tornado pull speed toward the eye, per tick (4.0 tiles/s at 20 ticks/s).
const TORNADO_PULL_PER_TICK: Fx = Fx::from_raw(crate::fixed::ONE / 5); // 0.2

/// [M3] A spell that has been paid for and committed to a ground point but has not
/// yet landed: it resolves `fuse` ticks later (its cast/travel window). Fixing the
/// target point at cast time but delaying the hit is what gives fast troops a
/// partial dodge chance in real CR. Part of reset() state so a (seed, action log)
/// replays bit-identically.
#[derive(Clone, Copy)]
struct PendingCast {
    owner: u8,
    def: usize,
    pos: V2,
    fuse: i32,
}

/// A lingering area effect on the field (poison/rage/heal zone, etc.).
struct AreaEffect {
    owner: u8,
    pos: V2,
    radius: Fx,
    ticks_left: i32,
    interval: i32,
    timer: i32,
    kind: AeoKind,
    power: i32,
    crown_pct: i32,
    // [edep] AeoKind::Spawn payload (Graveyard): which unit, how many per pulse
    spawn_def: usize,
    spawn_count: i32,
}

pub struct Engine {
    pub lib: Library,
    pub rng: Rng,
    pub tick: i64,
    pub entities: Vec<Entity>,
    pub projectiles: Vec<Projectile>,
    area_effects: Vec<AreaEffect>,
    pending_casts: Vec<PendingCast>, // [M3] spells in their cast/travel window
    next_id: u32,
    pub elixir: [Fx; 2],
    pub deck: [Vec<usize>; 2], // 8 def ids each
    pub queue: [Vec<usize>; 2],// rotating 8; front HAND = hand, [HAND] = next
    // [evo] per-player evolution charge, keyed by the base card's def id. A card
    // gains one charge each time it is cycled (played); once it reaches that
    // card's `evo_cycles` the next play spawns the evolved def and resets to 0.
    // Part of reset() state so a (seed, action log) replays bit-identically.
    evo_charge: [std::collections::HashMap<usize, i32>; 2], // [evo]
    pub crowns: [i32; 2],
    pub outcome: Outcome,
    king_down_dmg_seen: [bool; 2],
    last_crowns_for_ot: [i32; 2],
    // [eco] last card each player successfully deployed (for Mirror). Part of
    // reset() state so matches replay reproducibly. None = nothing played yet.
    last_played: [Option<usize>; 2],
    // [path] Cross-tick flow-field memo (NOT replay state): a pure cache of
    // BFS routing fields, rebuilt on demand. See PathCache.
    path_cache: PathCache,
}

impl Engine {
    pub fn new(lib: Library) -> Engine {
        Engine {
            lib,
            rng: Rng::new(0),
            tick: 0,
            entities: Vec::new(),
            projectiles: Vec::new(),
            area_effects: Vec::new(),
            pending_casts: Vec::new(), // [M3]
            next_id: 1,
            elixir: [Fx(0); 2],
            deck: [Vec::new(), Vec::new()],
            queue: [Vec::new(), Vec::new()],
            evo_charge: [std::collections::HashMap::new(), std::collections::HashMap::new()], // [evo]
            crowns: [0, 0],
            outcome: Outcome::Running,
            king_down_dmg_seen: [false, false],
            last_crowns_for_ot: [0, 0],
            last_played: [None, None], // [eco]
            path_cache: PathCache::default(), // [path]
        }
    }

    fn default_deck(&self) -> Vec<usize> {
        // first 8 deployable troops/buildings (skip spells so a random policy still
        // fights); at most one champion per deck ([champ] one-per-deck rule).
        let mut v: Vec<usize> = Vec::with_capacity(8);
        let mut has_champ = false;
        for &d in &self.lib.deployable {
            if self.lib.defs[d].ctype == CardType::Spell {
                continue;
            }
            if self.lib.defs[d].is_champion {
                if has_champ {
                    continue;
                }
                has_champ = true;
            }
            v.push(d);
            if v.len() == 8 {
                break;
            }
        }
        // pad (unreached in practice) with the first non-champion deployable
        let fallback = *self
            .lib
            .deployable
            .iter()
            .find(|&&d| !self.lib.defs[d].is_champion)
            .unwrap_or(&self.lib.deployable[0]);
        while v.len() < 8 {
            v.push(fallback);
        }
        v
    }

    pub fn reset(&mut self, seed: u64, decks: Option<[Vec<usize>; 2]>) {
        self.rng = Rng::new(seed);
        self.tick = 0;
        self.entities.clear();
        self.projectiles.clear();
        self.area_effects.clear();
        self.pending_casts.clear(); // [M3]
        self.next_id = 1;
        self.elixir = [Fx::from_int(START_ELIXIR); 2];
        self.evo_charge[0].clear(); // [evo] evolutions start uncharged every match
        self.evo_charge[1].clear(); // [evo]
        self.crowns = [0, 0];
        self.outcome = Outcome::Running;
        self.king_down_dmg_seen = [false, false];
        self.last_crowns_for_ot = [0, 0];
        self.last_played = [None, None]; // [eco]
        self.path_cache = PathCache::default(); // [path] drop any memo from a prior match

        self.deck = decks.unwrap_or_else(|| [self.default_deck(), self.default_deck()]);
        for p in 0..2 {
            let mut q = self.deck[p].clone();
            self.rng.shuffle(&mut q);
            self.queue[p] = q;
        }
        self.spawn_towers();
    }

    fn spawn_towers(&mut self) {
        for owner in 0u8..2 {
            for (c, half, is_king) in tower_layout_p0() {
                let pos = if owner == 0 { c } else { V2::new(c.x, mirror_y(c.y)) };
                let def = if is_king { self.lib.king_tower } else { self.lib.princess_tower };
                self.add_tower(owner, def, pos, half, is_king);
            }
        }
    }

    fn add_tower(&mut self, owner: u8, def: usize, pos: V2, half: Fx, is_king: bool) {
        let d = &self.lib.defs[def];
        let id = self.next_id;
        self.next_id += 1;
        self.entities.push(Entity {
            id,
            owner,
            def,
            pos,
            half: V2::new(half, half),
            hp: d.hp,
            max_hp: d.hp,
            target: None,
            cooldown: 0,
            deploy: 0,
            lifetime: -1,
            stun: 0,
            slow: 0,
            rage: 0,
            shield_hp: 0,
            reveal: 0, // [inv] towers are never invisible
            spawn_timer: -1,
            reflect_cd: 0, // [rfl]
            ability_cd: 0, // [champ]
            cloak: 0,      // [champ]
            deflect: 0,    // [champ]
            curse: 0,      // [curse]
            curse_spawn: -1, // [curse]
            charge_dist: Fx(0),
            dash: 0,    // [charge]
            dash_cd: 0, // [charge]
            ramp_timer: 0,   // [ramp]
            ramp_target: 0,  // [ramp]
            charge_shot: 0,  // [ramp]
            is_tower: true,
            is_king,
            activated: !(is_king && d.starts_dormant),
            engaged: false,
            revived: false, // [rev]
            hatch: false,   // [rev]
            dead: false,
        });
    }

    fn spawn_unit(&mut self, owner: u8, def: usize, pos: V2) {
        let d = &self.lib.defs[def];
        let id = self.next_id;
        self.next_id += 1;
        self.entities.push(Entity {
            id,
            owner,
            def,
            pos: Self::clamp_pos(pos),
            half: V2::default(),
            hp: d.hp,
            max_hp: d.hp,
            target: None,
            cooldown: 0,
            deploy: d.deploy_ticks.max(0),
            lifetime: d.lifetime_ticks,
            stun: 0,
            slow: 0,
            rage: 0,
            shield_hp: d.shield,
            reveal: 0, // [inv] invisible units spawn hidden until they first attack
            spawn_timer: if d.spawn_id.is_some() && d.spawn_interval_ticks > 0 {
                d.spawn_interval_ticks
            } else if d.elixir_per_pulse > 0 && d.elixir_pulse_ticks > 0 {
                d.elixir_pulse_ticks // [eco] reuse the spawner timer to pace elixir pulses
            } else {
                -1
            },
            reflect_cd: 0, // [rfl]
            ability_cd: 0, // [champ] ready on deploy
            cloak: 0,      // [champ]
            deflect: 0,    // [champ]
            curse: 0,      // [curse]
            curse_spawn: -1, // [curse]
            charge_dist: Fx(0),
            dash: 0,    // [charge]
            dash_cd: 0, // [charge]
            ramp_timer: 0,   // [ramp]
            ramp_target: 0,  // [ramp]
            charge_shot: 0,  // [ramp]
            is_tower: false,
            is_king: false,
            activated: true,
            engaged: false,
            revived: false, // [rev]
            hatch: false,   // [rev]
            dead: false,
        });
    }

    // [rev] Spawn a reborn unit (Phoenix hatch) at reduced hp, flagged one-shot so it
    // neither drops another egg nor revives again.
    fn spawn_revived(&mut self, owner: u8, def: usize, pos: V2, hp: i32) {
        self.spawn_unit(owner, def, pos);
        if let Some(e) = self.entities.last_mut() {
            let h = hp.max(1);
            e.hp = h;
            e.max_hp = h;
            e.revived = true;
        }
    }

    fn clamp_pos(p: V2) -> V2 {
        let cx = p.x.max(Fx(0)).min(WIDTH);
        let cy = p.y.max(Fx(0)).min(HEIGHT);
        V2::new(cx, cy)
    }

    // ---- hand / deck ----

    pub fn hand(&self, player: usize) -> [usize; HAND] {
        let q = &self.queue[player];
        [q[0], q[1], q[2], q[3]]
    }
    pub fn next_card(&self, player: usize) -> usize {
        self.queue[player][HAND]
    }

    fn cycle(&mut self, player: usize, slot: usize) {
        let card = self.queue[player].remove(slot);
        self.queue[player].push(card);
    }

    // ---- public step ----

    /// Advance one tick given both players' actions. Deploys are applied first,
    /// then the world simulates.
    pub fn step(&mut self, actions: [Action; 2]) -> Outcome {
        if self.outcome != Outcome::Running {
            return self.outcome;
        }
        for p in 0..2 {
            // [champ] the reserved slot activates the champion ability in-stream
            // (same tick ordering as a deploy) so replays stay deterministic.
            if actions[p].slot == ABILITY_SLOT {
                self.activate_ability(p);
            } else {
                self.try_deploy(p, actions[p]);
            }
        }
        self.simulate();
        self.tick += 1;
        self.resolve_time();
        self.outcome
    }

    fn try_deploy(&mut self, player: usize, a: Action) {
        if a.slot < 0 || a.slot as usize >= HAND {
            return;
        }
        let slot = a.slot as usize;
        let card = self.queue[player][slot];
        let is_mirror = self.lib.defs[card].is_mirror; // [eco]

        // [eco] Mirror resolves to the player's last-deployed card, cost +1 elixir
        // (nominally +1 level, which this level-less sim does not model).
        let (def, cost_elixir) = if is_mirror {
            match self.last_played[player] {
                Some(last) => (last, self.lib.defs[last].elixir + 1),
                None => return, // nothing deployed yet: Mirror is a no-op
            }
        } else {
            (card, self.lib.defs[card].elixir)
        };
        let cost = Fx::from_int(cost_elixir as i64);
        if self.elixir[player] < cost {
            return;
        }
        // deploy snaps to a tile; troops/buildings only on your own territory
        // (plus an opened pocket once you've taken an enemy princess tower),
        // unless the card is flagged deploy_anywhere ([edep]: Miner, Goblin Drill).
        let is_spell = self.lib.defs[def].ctype == CardType::Spell;
        let pos = snap_to_tile(Self::clamp_pos(V2::new(a.x, a.y)));
        if !is_spell && !self.deploy_allowed(player, pos, def) {
            return;
        }
        self.elixir[player] = self.elixir[player] - cost;
        self.cycle(player, slot);
        // [eco] remember the real card played; Mirror itself is never remembered,
        // so a Mirror always replays the last NON-Mirror deploy.
        if !is_mirror {
            self.last_played[player] = Some(def);
        }
        self.place_card(player, def, pos);
    }

    /// Place a resolved card: cast it if a spell, else spawn its troop/building
    /// cluster. Shared by normal deploys and Mirror. [eco]
    fn place_card(&mut self, player: usize, def: usize, pos: V2) {
        // [evo] advance this card's evolution charge and pick base-vs-evolved def.
        let spawn_def = self.evo_resolve(player, def);
        let (is_spell, count, land_damage, land_radius, land_targets) = {
            let d = &self.lib.defs[spawn_def];
            (d.ctype == CardType::Spell, d.count.max(1), d.land_damage, d.land_radius, d.targets)
        };
        if is_spell {
            self.cast_spell(player as u8, spawn_def, pos);
            return;
        }
        for k in 0..count {
            let off = Self::spawn_offset(k, count);
            self.spawn_unit(player as u8, spawn_def, V2::new(pos.x + off.x, pos.y + off.y));
        }
        // [mix] mixed-composition cards (Goblin Gang, Rascals): spawn the secondary unit
        // type beside the primary cluster, offset down-lane so the groups don't overlap.
        let (deploy_sid, deploy_cnt) = {
            let d = &self.lib.defs[spawn_def];
            (d.deploy_spawn_id, d.deploy_spawn_count.max(1))
        };
        if let Some(sid) = deploy_sid {
            // nudge the second group one tile toward the owner's own side so the ranged
            // backers sit behind the melee front without stacking on them.
            let back = if player == 0 { Fx::from_int(-1) } else { Fx::from_int(1) };
            let base = V2::new(pos.x, pos.y + back);
            for k in 0..deploy_cnt {
                let off = Self::spawn_offset(k, deploy_cnt);
                self.spawn_unit(player as u8, sid, V2::new(base.x + off.x, base.y + off.y));
            }
        }
        // [charge] Mega Knight deals area damage where it lands on deploy.
        if land_damage > 0 {
            self.projectiles
                .push(Self::instant_splash(player as u8, pos, land_damage, land_radius, land_targets));
        }
    }

    /// [evo] Resolve which def a play uses, advancing this card's evolution
    /// charge counter. A non-evolution card (no `evolved_id`, or `evo_cycles`
    /// == 0) always returns `base`. An evolution returns its evolved def once
    /// the counter has reached `evo_cycles` (then resets it to 0); otherwise it
    /// returns `base` and adds one charge. Deterministic: the counter is keyed
    /// by base def id and is only ever read/written by key (never iterated).
    fn evo_resolve(&mut self, player: usize, base: usize) -> usize {
        let (evolved, need) = {
            let d = &self.lib.defs[base];
            match d.evolved_id {
                Some(ev) if d.evo_cycles > 0 => (ev, d.evo_cycles),
                _ => return base,
            }
        };
        let charge = self.evo_charge[player].entry(base).or_insert(0);
        if *charge >= need {
            *charge = 0; // spent: evolved play, then recharge from scratch
            evolved
        } else {
            *charge += 1; // still charging: play the base card
            base
        }
    }

    fn enemy_princess_alive(&self, enemy: u8, lane: u8) -> bool {
        self.entities.iter().any(|e| {
            e.is_tower && !e.is_king && e.owner == enemy && e.alive() && lane_of(e.pos.x) == lane
        })
    }

    /// Whether `player` may deploy a troop/building (`def`) at `pos` this tick.
    pub fn deploy_allowed(&self, player: usize, pos: V2, def: usize) -> bool {
        // [edep] deploy-anywhere cards (Miner, Goblin Drill) may be placed on
        // either side of the arena. `pos` is already snapped+clamped into bounds,
        // so the only remaining gate (own-side / opened pocket) is lifted here.
        if self.lib.defs[def].deploy_anywhere {
            return true;
        }
        let lane = lane_of(pos.x);
        let enemy = 1 - player as u8;
        if player == 0 {
            pos.y < P0_DEPLOY_MAX_Y
                || (!self.enemy_princess_alive(enemy, lane) && pos.y < POCKET_FAR_Y_P0)
        } else {
            pos.y >= P1_DEPLOY_MIN_Y
                || (!self.enemy_princess_alive(enemy, lane) && pos.y >= POCKET_FAR_Y_P1)
        }
    }

    /// [M6] Deterministic per-count spawn formation offset for the k-th unit of a
    /// `count`-unit group, in tiles relative to the deploy tile centre. Real CR
    /// spawns grouped cards in compact clusters/arcs (pairs side-by-side, triangles,
    /// 2x2 squares, small grids) rather than one straight row, which changes how a
    /// group engages and gets splashed. The offsets are small (<~1 tile) so every
    /// unit stays on/next to the deploy tile; the collision solver then settles them.
    /// Pure fixed-point => fully deterministic.
    fn spawn_offset(k: i32, count: i32) -> V2 {
        if count <= 1 {
            return V2::new(Fx(0), Fx(0));
        }
        let k = k.rem_euclid(count);
        // Unit-cell coordinates (in grid units) per formation; scaled by SP tiles.
        let sp = Fx::from_f64(0.7); // spacing between neighbours (~1.4 collision radii)
        let (ux, uy): (f64, f64) = match count {
            2 => {
                // side-by-side pair
                [(-0.5, 0.0), (0.5, 0.0)][k as usize]
            }
            3 => {
                // triangle: apex forward, two behind
                [(0.0, 0.5), (-0.5, -0.4), (0.5, -0.4)][k as usize]
            }
            4 => {
                // 2x2 square / diamond
                [(-0.5, 0.5), (0.5, 0.5), (-0.5, -0.5), (0.5, -0.5)][k as usize]
            }
            5 => {
                // diamond + centre
                [(0.0, 0.0), (0.0, 0.8), (0.0, -0.8), (-0.8, 0.0), (0.8, 0.0)][k as usize]
            }
            6 => {
                // 2 rows of 3 (a tight block / horde)
                [
                    (-1.0, 0.45),
                    (0.0, 0.45),
                    (1.0, 0.45),
                    (-1.0, -0.45),
                    (0.0, -0.45),
                    (1.0, -0.45),
                ][k as usize]
            }
            _ => {
                // general case: centred square-ish grid, filled in index order.
                let cols = {
                    // smallest c with c*c >= count (compact grid)
                    let mut c = 1i32;
                    while c * c < count {
                        c += 1;
                    }
                    c
                };
                let rows = (count + cols - 1) / cols;
                let col = k % cols;
                let row = k / cols;
                let cx = (cols - 1) as f64 / 2.0;
                let cy = (rows - 1) as f64 / 2.0;
                (col as f64 - cx, row as f64 - cy)
            }
        };
        V2::new(sp.mul(Fx::from_f64(ux)), sp.mul(Fx::from_f64(uy)))
    }

    // ---- champions ---- [champ]

    /// Activate `player`'s on-field champion ability. Returns true if it fired.
    /// Fails (no elixir/cooldown spent) when: no active champion is deployed, its
    /// ability is still on cooldown, the player lacks the elixir, or — for Dash —
    /// there is no enemy in reach. Reuses the status/spawn systems for effects.
    pub fn activate_ability(&mut self, player: usize) -> bool {
        if self.outcome != Outcome::Running || player > 1 {
            return false;
        }
        let owner = player as u8;
        // locate the player's active champion with a real ability
        let mut idx = None;
        for i in 0..self.entities.len() {
            let e = &self.entities[i];
            if e.dead || e.owner != owner || !e.active() {
                continue;
            }
            let d = &self.lib.defs[e.def];
            if d.is_champion && d.ability != AbilityKind::None {
                idx = Some(i);
                break;
            }
        }
        let Some(i) = idx else { return false };
        if self.entities[i].ability_cd > 0 {
            return false;
        }
        let def = self.entities[i].def;
        // copy the (Copy) ability spec out so we can freely mutate entities below
        let kind = self.lib.defs[def].ability;
        let cost = Fx::from_int(self.lib.defs[def].ability_cost as i64);
        let cd = self.lib.defs[def].ability_cooldown_ticks.max(1);
        let dur = self.lib.defs[def].ability_duration_ticks.max(1);
        let admg = self.lib.defs[def].ability_damage;
        let acount = self.lib.defs[def].ability_count.max(1);
        let aradius = self.lib.defs[def].ability_radius;
        let arange = self.lib.defs[def].ability_range;
        let aknock = self.lib.defs[def].ability_knockback;
        let aspawn = self.lib.defs[def].ability_spawn_id;
        if self.elixir[player] < cost {
            return false;
        }
        let cpos = self.entities[i].pos;

        match kind {
            AbilityKind::None => false,
            AbilityKind::Cloak => {
                // become untargetable + attack/move boost (reuse cloak + rage status)
                self.elixir[player] = self.elixir[player] - cost;
                self.entities[i].ability_cd = cd;
                self.entities[i].cloak = self.entities[i].cloak.max(dur);
                self.entities[i].rage = self.entities[i].rage.max(dur);
                true
            }
            AbilityKind::Summon => {
                let Some(sid) = aspawn else { return false };
                self.elixir[player] = self.elixir[player] - cost;
                self.entities[i].ability_cd = cd;
                // ring of summoned units around the champion (reuse spawn system)
                for k in 0..acount {
                    let off = Self::spawn_offset(k, acount);
                    let p = V2::new(cpos.x + off.x.mul(aradius), cpos.y + off.y.mul(aradius));
                    self.spawn_unit(owner, sid, p);
                }
                true
            }
            AbilityKind::Dash => {
                // find the nearest enemy within dash reach; the lunge then closes
                // the gap to it (so the dash always connects) and damages every
                // enemy within radius of the travelled line.
                let search = arange;
                let mut best: Option<(V2, Fx)> = None;
                for e in self.entities.iter() {
                    if e.dead || e.owner == owner || !e.alive() || e.deploy > 0 {
                        continue;
                    }
                    let dist = e.pos.sub(cpos).len();
                    if dist <= search && best.map_or(true, |(_, bd)| dist < bd) {
                        best = Some((e.pos, dist));
                    }
                }
                let Some((tcenter, tdist)) = best else { return false };
                // feasible: consume and perform the dash
                self.elixir[player] = self.elixir[player] - cost;
                self.entities[i].ability_cd = cd;
                let dir = tcenter.sub(cpos).step_toward(Fx::from_int(1));
                let reach = if tdist < arange { tdist } else { arange };
                let endp =
                    Self::clamp_pos(V2::new(cpos.x + dir.x.mul(reach), cpos.y + dir.y.mul(reach)));
                // damage every enemy within radius of the dash line
                let r2 = aradius.mul(aradius);
                for j in 0..self.entities.len() {
                    if j == i {
                        continue;
                    }
                    let (dead, j_owner, alive, deploy, jpos, is_king) = {
                        let e = &self.entities[j];
                        (e.dead, e.owner, e.alive(), e.deploy, e.pos, e.is_king)
                    };
                    if dead || j_owner == owner || !alive || deploy > 0 {
                        continue;
                    }
                    let dpt = dist_point_segment(jpos, cpos, endp);
                    if dpt.mul(dpt) <= r2 {
                        let mut rem = admg;
                        if self.entities[j].shield_hp > 0 {
                            let ab = rem.min(self.entities[j].shield_hp);
                            self.entities[j].shield_hp -= ab;
                            rem -= ab;
                        }
                        if rem > 0 {
                            self.entities[j].hp -= rem;
                        }
                        // chipping the king wakes it, as with any damage
                        if is_king {
                            self.entities[j].activated = true;
                        }
                    }
                }
                self.entities[i].pos = endp;
                true
            }
            AbilityKind::LaneBomb => {
                // [champ] Mighty Miner: drop an area bomb (damage + knockback) at the
                // current spot, then dig across to the mirror lane (same row). The
                // bomb hits every enemy within the blast radius, air and ground.
                self.elixir[player] = self.elixir[player] - cost;
                self.entities[i].ability_cd = cd;
                let center = cpos;
                let r2 = aradius.mul(aradius);
                for j in 0..self.entities.len() {
                    if j == i {
                        continue;
                    }
                    let (dead, j_owner, alive, deploy, jpos, is_king) = {
                        let e = &self.entities[j];
                        (e.dead, e.owner, e.alive(), e.deploy, e.pos, e.is_king)
                    };
                    if dead || j_owner == owner || !alive || deploy > 0 {
                        continue;
                    }
                    let off = jpos.sub(center);
                    if off.len_sq() > r2 {
                        continue;
                    }
                    // area damage (shield soaks first), as with any blast
                    let mut rem = admg;
                    if self.entities[j].shield_hp > 0 {
                        let ab = rem.min(self.entities[j].shield_hp);
                        self.entities[j].shield_hp -= ab;
                        rem -= ab;
                    }
                    if rem > 0 {
                        self.entities[j].hp -= rem;
                    }
                    if is_king {
                        self.entities[j].activated = true;
                    }
                    // knockback: shove the victim directly away from the blast center
                    if aknock.0 > 0 {
                        let dir = off.step_toward(aknock);
                        let jdef = self.entities[j].def;
                        let np = Self::clamp_pos(V2::new(jpos.x + dir.x, jpos.y + dir.y));
                        let np = block_river(jpos, np, &self.lib.defs[jdef]);
                        self.entities[j].pos = np;
                    }
                }
                // dig to the mirror lane: reflect x across the arena centre, keep row.
                let newp = Self::clamp_pos(V2::new(WIDTH - cpos.x, cpos.y));
                self.entities[i].pos = newp;
                true
            }
            AbilityKind::Deflect => {
                // [champ] Monk: open a timed damage-reduction window on himself. While
                // active, apply_damage scales incoming hits by ability_reduction%.
                self.elixir[player] = self.elixir[player] - cost;
                self.entities[i].ability_cd = cd;
                self.entities[i].deflect = self.entities[i].deflect.max(dur);
                true
            }
            AbilityKind::Warp => {
                // [champ] Boss Bandit: warp a fixed distance toward the enemy side and
                // turn briefly untargetable/invisible, dropping the current target.
                self.elixir[player] = self.elixir[player] - cost;
                self.entities[i].ability_cd = cd;
                // forward is toward the opponent's half (p0 -> +y, p1 -> -y)
                let fwd = if owner == 0 { arange } else { Fx(0) - arange };
                let jdef = def;
                let np = Self::clamp_pos(V2::new(cpos.x, cpos.y + fwd));
                let np = block_river(cpos, np, &self.lib.defs[jdef]);
                self.entities[i].pos = np;
                self.entities[i].cloak = self.entities[i].cloak.max(dur);
                self.entities[i].target = None;
                true
            }
        }
    }

    /// (champion_on_field, ability_cooldown_ticks_remaining, ability_cost) for a
    /// player. on_field=false (and zeros) when no champion is deployed. [champ]
    pub fn champion_status(&self, player: usize) -> (bool, i32, i32) {
        if player > 1 {
            return (false, 0, 0);
        }
        let owner = player as u8;
        for e in self.entities.iter() {
            if e.dead || e.owner != owner || !e.active() {
                continue;
            }
            let d = &self.lib.defs[e.def];
            if d.is_champion && d.ability != AbilityKind::None {
                return (true, e.ability_cd, d.ability_cost);
            }
        }
        (false, 0, 0)
    }

    /// [M3] Schedule a spell: commit it to ground point `pos` now and let it land a
    /// short, data-plausible number of ticks later (its cast/travel window). The
    /// elixir was already spent at deploy time; only the *effect* is delayed, so
    /// damage/crown/effect behaviour is unchanged -- just time-shifted, which gives a
    /// fast troop a chance to walk out of the blast before it resolves.
    fn cast_spell(&mut self, owner: u8, def: usize, pos: V2) {
        let fuse = self.spell_cast_fuse(def);
        if fuse <= 0 {
            self.resolve_spell(owner, def, pos);
        } else {
            self.pending_casts.push(PendingCast { owner, def, pos, fuse });
        }
    }

    /// [M3] Cast/travel window in ticks for a spell. Faster projectiles land sooner
    /// (Zap/Lightning ~0.25s, Fireball/Rocket ~0.5s), matching how dodgeable each
    /// spell is in CR. Effect-only spells (Poison/Rage/Tornado/Graveyard) with no
    /// projectile use a small default. Derived from a fixed "drop" distance so the
    /// window scales with the card's own projectile speed (data-plausible).
    fn spell_cast_fuse(&self, def: usize) -> i32 {
        let d = &self.lib.defs[def];
        const MIN_FUSE: i32 = 3;
        const MAX_FUSE: i32 = 20;
        let drop = Fx::from_f64(3.0); // tiles the spell "falls"/travels before impact
        let f = if d.proj_speed.0 > 0 {
            drop.div(d.proj_speed).to_int_floor() as i32
        } else {
            6 // ~0.3s for effect/spawn-only spells with no projectile
        };
        f.clamp(MIN_FUSE, MAX_FUSE)
    }

    /// [M3] Resolve a spell at its committed ground point (was `cast_spell`'s body):
    /// the instant-damage projectile, lingering effect, and any unit spawns.
    fn resolve_spell(&mut self, owner: u8, def: usize, pos: V2) {
        let d = &self.lib.defs[def];
        let (damage, radius, proj_speed, crown_pct, stun_ticks) =
            (d.damage, d.spell_radius, d.proj_speed, d.crown_pct, d.stun_ticks);
        let (effect, eticks, einterval, epower) =
            (d.effect, d.effect_ticks, d.effect_interval, d.effect_power);
        let curse_spawn_id = d.curse_spawn_id; // [curse] Goblin Curse transform target
        let spell_targets = d.spell_max_targets; // [proj] Lightning: strike top-K by HP
        // [edep] unit-spawning spells: instant drop (Goblin Barrel) or a timed
        // spawn zone (Graveyard, via SpellEffect::Spawn below).
        let (spawn_def, spawn_count) = (d.spawn_id, d.spawn_count.max(1));
        let rad = if radius.0 > 0 { radius } else { Fx::from_f64(1.0) };

        // instant damage component (Fireball/Arrows/Zap/Lightning, and the hit of effect spells)
        if damage > 0 {
            // [kb] a rolling-pierce spell (the Log) travels in a straight line up the
            // lane away from the caster, knocking ground enemies back as it rolls.
            let is_roll = d.pierce && d.roll_range.0 > 0;
            let roll_dir = if owner == 0 {
                V2::new(Fx(0), Fx::from_int(1))
            } else {
                V2::new(Fx(0), Fx::from_int(-1))
            };
            self.projectiles.push(Projectile {
                owner,
                pos,
                target: None,
                aim: pos,
                speed: if proj_speed.0 > 0 { proj_speed } else { Fx::from_int(100) },
                damage,
                splash: rad,
                crown_pct,
                stun_ticks,
                // [kb] respect the spell's targets so a ground-only roller can't hit air
                hits_flying: matches!(d.targets, Targets::Air | Targets::Both),
                hits_ground: matches!(d.targets, Targets::Ground | Targets::Both),
                homing: false,        // [proj] spells commit to a ground point (dodgeable)
                top_k: spell_targets, // [proj] 0 = whole blast; >0 = K highest-HP enemies
                knockback: d.knockback,                                     // [kb]
                pierce: d.pierce,                                           // [kb]
                roll_dir: if is_roll { roll_dir } else { V2::default() },   // [kb]
                roll_remaining: if is_roll { d.roll_range } else { Fx(0) }, // [kb]
                ..Default::default()
            });
        }

        // effect component
        match effect {
            SpellEffect::None => {}
            SpellEffect::Freeze | SpellEffect::Slow => {
                let r2 = rad.mul(rad);
                for e in self.entities.iter_mut() {
                    if e.dead || e.owner == owner {
                        continue;
                    }
                    let dx = e.pos.x - pos.x;
                    let dy = e.pos.y - pos.y;
                    if dx.mul(dx) + dy.mul(dy) > r2 {
                        continue;
                    }
                    if effect == SpellEffect::Freeze {
                        e.stun = e.stun.max(eticks);
                    } else if !e.is_tower {
                        e.slow = e.slow.max(eticks);
                    }
                }
            }
            SpellEffect::Poison => self.area_effects.push(AreaEffect {
                owner, pos, radius: rad, ticks_left: eticks.max(1),
                interval: einterval.max(1), timer: einterval.max(1),
                kind: AeoKind::Damage, power: epower, crown_pct,
                spawn_def: 0, spawn_count: 0,
            }),
            SpellEffect::Rage => self.area_effects.push(AreaEffect {
                owner, pos, radius: rad, ticks_left: eticks.max(1),
                interval: 5, timer: 0, kind: AeoKind::Rage, power: 0, crown_pct: 100,
                spawn_def: 0, spawn_count: 0,
            }),
            SpellEffect::Heal => self.area_effects.push(AreaEffect {
                owner, pos, radius: rad, ticks_left: eticks.max(1),
                interval: einterval.max(1), timer: einterval.max(1),
                kind: AeoKind::Heal, power: epower, crown_pct: 100,
                spawn_def: 0, spawn_count: 0,
            }),
            // [kb] Tornado: a lingering zone that pulls enemies toward its eye every
            // tick (see apply_pull_movement) and chips them for `epower` each interval.
            SpellEffect::Tornado => self.area_effects.push(AreaEffect {
                owner, pos, radius: rad, ticks_left: eticks.max(1),
                interval: einterval.max(1), timer: einterval.max(1),
                kind: AeoKind::Pull, power: epower, crown_pct,
                spawn_def: 0, spawn_count: 0,
            }),
            // [edep] Graveyard: a zone that summons `spawn_count` units every
            // `effect_interval` ticks for `effect_duration`, scattered over radius.
            SpellEffect::Spawn => self.area_effects.push(AreaEffect {
                owner, pos, radius: rad, ticks_left: eticks.max(1),
                interval: einterval.max(1), timer: einterval.max(1),
                kind: AeoKind::Spawn, power: 0, crown_pct: 100,
                spawn_def: spawn_def.unwrap_or(0), spawn_count,
            }),
            // [curse] Clone: duplicate every friendly troop in radius at 1 hp.
            SpellEffect::Clone => {
                let r2 = rad.mul(rad);
                let mut copies: Vec<(usize, V2)> = Vec::new();
                for e in self.entities.iter() {
                    if e.dead || e.owner != owner {
                        continue;
                    }
                    if self.lib.defs[e.def].ctype != CardType::Troop {
                        continue; // buildings/towers are not cloned
                    }
                    let dx = e.pos.x - pos.x;
                    let dy = e.pos.y - pos.y;
                    if dx.mul(dx) + dy.mul(dy) <= r2 {
                        copies.push((e.def, e.pos));
                    }
                }
                for (cdef, cpos) in copies {
                    self.spawn_unit(owner, cdef, cpos);
                    if let Some(c) = self.entities.last_mut() {
                        c.hp = 1; // clones are fragile 1-hp duplicates
                        c.max_hp = self.lib.defs[cdef].hp.max(1);
                        c.shield_hp = 0;
                        c.deploy = 0; // clones appear ready (no re-deploy delay)
                    }
                }
            }
            // [curse] Goblin Curse: curse enemy troops in radius -- a DoT zone plus a
            // one-shot transform (on their death they become a unit for the CASTER).
            SpellEffect::Curse => {
                let r2 = rad.mul(rad);
                for idx in 0..self.entities.len() {
                    let (dead, eowner, is_tower, epos, is_troop) = {
                        let e = &self.entities[idx];
                        (e.dead, e.owner, e.is_tower, e.pos, self.lib.defs[e.def].ctype == CardType::Troop)
                    };
                    if dead || eowner == owner || is_tower || !is_troop {
                        continue;
                    }
                    let dx = epos.x - pos.x;
                    let dy = epos.y - pos.y;
                    if dx.mul(dx) + dy.mul(dy) > r2 {
                        continue;
                    }
                    self.entities[idx].curse = self.entities[idx].curse.max(eticks.max(1));
                    if let Some(cs) = curse_spawn_id {
                        self.entities[idx].curse_spawn = cs as i32;
                    }
                }
                // damage-over-time component (zone, like Poison)
                if epower > 0 {
                    self.area_effects.push(AreaEffect {
                        owner, pos, radius: rad, ticks_left: eticks.max(1),
                        interval: einterval.max(1), timer: einterval.max(1),
                        kind: AeoKind::Damage, power: epower, crown_pct,
                        spawn_def: 0, spawn_count: 0,
                    });
                }
            }
        }

        // [edep] Goblin Barrel: a spell that instantly lands `spawn_count` units
        // at the target (no lingering zone). Deterministic line spread.
        if effect != SpellEffect::Spawn {
            if let Some(sid) = spawn_def {
                for k in 0..spawn_count {
                    let off = Self::spawn_offset(k, spawn_count);
                    self.spawn_unit(owner, sid, V2::new(pos.x + off.x, pos.y + off.y));
                }
            }
        }
    }

    /// Apply lingering area effects (poison DoT, rage/heal zones). Deaths from
    /// DoT are picked up by the end-of-tick death pass in `apply_damage`.
    fn apply_area_effects(&mut self) {
        if self.area_effects.is_empty() {
            return;
        }
        let mut events: Vec<(AeoKind, u8, V2, Fx, i32, i32)> = Vec::new();
        // [edep] Graveyard-style spawn pulses: (owner, center, radius, def, count)
        let mut spawn_pulses: Vec<(u8, V2, Fx, usize, i32)> = Vec::new();
        for ae in self.area_effects.iter_mut() {
            if ae.ticks_left > 0 {
                ae.ticks_left -= 1;
            }
            ae.timer -= 1;
            if ae.timer <= 0 {
                if ae.kind == AeoKind::Spawn {
                    spawn_pulses.push((ae.owner, ae.pos, ae.radius, ae.spawn_def, ae.spawn_count.max(1)));
                } else {
                    events.push((ae.kind, ae.owner, ae.pos, ae.radius, ae.power, ae.crown_pct));
                }
                ae.timer = ae.interval.max(1);
            }
        }
        self.area_effects.retain(|ae| ae.ticks_left > 0);
        // [edep] apply spawn pulses: scatter units deterministically over the disc.
        for (owner, center, radius, sdef, count) in spawn_pulses {
            for _ in 0..count {
                let off = self.random_disc_offset(radius);
                self.spawn_unit(owner, sdef, V2::new(center.x + off.x, center.y + off.y));
            }
        }
        for (kind, owner, pos, radius, power, crown_pct) in events {
            let r2 = radius.mul(radius);
            for e in self.entities.iter_mut() {
                if e.dead {
                    continue;
                }
                let dx = e.pos.x - pos.x;
                let dy = e.pos.y - pos.y;
                if dx.mul(dx) + dy.mul(dy) > r2 {
                    continue;
                }
                let enemy = e.owner != owner;
                match kind {
                    AeoKind::Damage => {
                        if enemy {
                            e.hp -= if e.is_tower { power * crown_pct / 100 } else { power };
                            if e.is_king {
                                e.activated = true;
                            }
                        }
                    }
                    AeoKind::Heal => {
                        if !enemy && !e.is_tower {
                            e.hp = (e.hp + power).min(e.max_hp);
                        }
                    }
                    AeoKind::Rage => {
                        if !enemy {
                            e.rage = e.rage.max(7);
                        }
                    }
                    AeoKind::Slow => {
                        if enemy && !e.is_tower {
                            e.slow = e.slow.max(7);
                        }
                    }
                    AeoKind::Freeze => {
                        if enemy {
                            e.stun = e.stun.max(7);
                        }
                    }
                    // [kb] Tornado's damage-over-time component (the pull itself is
                    // applied every tick by apply_pull_movement, below).
                    AeoKind::Pull => {
                        if enemy {
                            e.hp -= if e.is_tower { power * crown_pct / 100 } else { power };
                            if e.is_king {
                                e.activated = true;
                            }
                        }
                    }
                    // [edep] spawn pulses are handled above (never queued as events).
                    AeoKind::Spawn => {}
                }
            }
        }
        // [kb] continuous pull toward each active Tornado eye (every tick, not gated
        // by the DoT interval, so the drag is smooth and deterministic).
        self.apply_pull_movement();
    }

    /// [heal] Battle Healer aura: every `heal_aura_interval` ticks, each alive healer
    /// restores `heal_aura_power` hp to friendly troops (itself included) within
    /// `heal_aura_radius`. Buildings/towers are not healed (CR's `no_buildings`
    /// filter). The aura is re-anchored to the healer each pulse, so it tracks her
    /// as she moves. Deterministic: healers snapshotted in index order, then applied.
    fn apply_heal_auras(&mut self) {
        let healers: Vec<(V2, u8, i32, Fx, i32)> = self
            .entities
            .iter()
            .filter_map(|e| {
                // must be a fully-deployed, living healer (a healer dying this tick must
                // not resurrect itself or keep healing via its own aura).
                if !e.alive() || e.deploy > 0 {
                    return None;
                }
                let d = &self.lib.defs[e.def];
                if d.heal_aura_power <= 0 {
                    return None;
                }
                Some((e.pos, e.owner, d.heal_aura_power, d.heal_aura_radius, d.heal_aura_interval.max(1)))
            })
            .collect();
        if healers.is_empty() {
            return;
        }
        // which entities are heal-eligible (friendly troops, no buildings/towers)
        let is_troop: Vec<bool> =
            self.entities.iter().map(|e| self.lib.defs[e.def].ctype == CardType::Troop).collect();
        for (pos, owner, power, radius, interval) in healers {
            if self.tick % interval as i64 != 0 {
                continue; // this healer does not pulse this tick
            }
            let r2 = radius.mul(radius);
            for idx in 0..self.entities.len() {
                if !is_troop[idx] {
                    continue;
                }
                let e = &self.entities[idx];
                // heal only living, not-full friendly troops (never resurrect hp<=0).
                if !e.alive() || e.owner != owner || e.hp >= e.max_hp {
                    continue;
                }
                let dx = e.pos.x - pos.x;
                let dy = e.pos.y - pos.y;
                if dx.mul(dx) + dy.mul(dy) > r2 {
                    continue;
                }
                let healed = (self.entities[idx].hp + power).min(self.entities[idx].max_hp);
                self.entities[idx].hp = healed;
            }
        }
    }

    /// [kb] Drag enemies toward every active `Pull` area effect's center. Snapshot-free
    /// but deterministic: entities are visited in index order and each reads only its
    /// own prior position. Towers/buildings are anchored (immune); heavier troops are
    /// dragged less (mass_resist). Ground units still obey the river banks.
    fn apply_pull_movement(&mut self) {
        let pulls: Vec<(u8, V2, Fx)> = self
            .area_effects
            .iter()
            .filter(|ae| matches!(ae.kind, AeoKind::Pull))
            .map(|ae| (ae.owner, ae.pos, ae.radius.mul(ae.radius)))
            .collect();
        if pulls.is_empty() {
            return;
        }
        let n = self.entities.len();
        for i in 0..n {
            let (owner, epos, anchored, resist, def) = {
                let e = &self.entities[i];
                if !e.active() {
                    continue;
                }
                let d = &self.lib.defs[e.def];
                let anchored = e.is_tower || d.ctype == CardType::Building;
                (e.owner, e.pos, anchored, mass_resist(d.mass), e.def)
            };
            if anchored {
                continue;
            }
            let step = TORNADO_PULL_PER_TICK.mul(resist);
            let mut disp = V2::default();
            for (powner, pos, r2) in &pulls {
                if owner == *powner {
                    continue;
                }
                let dvec = V2::new(pos.x - epos.x, pos.y - epos.y);
                if dvec.len_sq() > *r2 {
                    continue;
                }
                // move toward the eye, but never overshoot it
                let mv = if dvec.len() <= step { dvec } else { dvec.step_toward(step) };
                disp = disp.add(mv);
            }
            if disp.x.0 == 0 && disp.y.0 == 0 {
                continue;
            }
            let d = &self.lib.defs[def];
            let old = self.entities[i].pos;
            let np = block_river(old, V2::new(old.x + disp.x, old.y + disp.y), d);
            self.entities[i].pos = Self::clamp_pos(np);
        }
    }

    /// [edep] Deterministic random offset within a disc of `radius` (Graveyard
    /// skeleton scatter). Uses the integer engine RNG + fixed-point only, so the
    /// scatter is bit-identical from the seed, like every other stochastic draw.
    fn random_disc_offset(&mut self, radius: Fx) -> V2 {
        let r_raw = radius.0.max(1);
        let dx = Fx(self.rng.randint(-r_raw, r_raw));
        let dy = Fx(self.rng.randint(-r_raw, r_raw));
        let off = V2::new(dx, dy);
        // reject-by-clamp: points outside the disc are pulled onto its rim.
        if off.len_sq() > radius.mul(radius) {
            off.step_toward(radius)
        } else {
            off
        }
    }

    // ---- simulation ----

    fn elixir_rate(&self) -> Fx {
        if self.tick >= REGULATION_TICKS {
            elixir_per_tick_3x()
        } else if self.tick >= DOUBLE_ELIXIR_AT {
            elixir_per_tick_2x()
        } else {
            elixir_per_tick_1x()
        }
    }

    fn simulate(&mut self) {
        // 1. elixir regen
        let rate = self.elixir_rate();
        let cap = Fx::from_int(MAX_ELIXIR);
        for p in 0..2 {
            self.elixir[p] = (self.elixir[p] + rate).min(cap);
        }
        // 2. timers
        for e in self.entities.iter_mut() {
            let stunned = e.stun > 0; // [ramp] remember pre-decrement stun state
            if e.deploy > 0 {
                e.deploy -= 1;
            }
            if e.cooldown > 0 {
                e.cooldown -= 1;
            }
            if e.stun > 0 {
                e.stun -= 1;
                e.charge_dist = Fx(0); // [charge] a stun resets any built-up charge
            }
            if e.dash_cd > 0 {
                e.dash_cd -= 1; // [charge] leap cooldown
            }
            if e.slow > 0 {
                e.slow -= 1;
            }
            if e.rage > 0 {
                e.rage -= 1;
            }
            if e.reflect_cd > 0 {
                e.reflect_cd -= 1; // [rfl]
            }
            if e.ability_cd > 0 {
                e.ability_cd -= 1; // [champ]
            }
            if e.cloak > 0 {
                e.cloak -= 1; // [champ]
            }
            if e.deflect > 0 {
                e.deflect -= 1; // [champ] Monk Deflect window
            }
            if e.curse > 0 {
                e.curse -= 1; // [curse] Goblin Curse: expires if the unit survives it
                if e.curse == 0 {
                    e.curse_spawn = -1; // curse lifted: no transform on a later death
                }
            }
            if e.reveal > 0 {
                e.reveal -= 1; // [inv] fade back to invisible when the window elapses
            }
            if e.lifetime > 0 {
                e.lifetime -= 1;
                if e.lifetime == 0 {
                    e.hp = 0;
                    e.hatch = true; // [rev] lifetime-expiry death may hatch a revive (Phoenix egg)
                }
            }
            // [ramp] stun/freeze interrupts ramp-up and charge-shot; otherwise a
            // charge-shot weapon (Sparky) keeps charging while active.
            if stunned {
                e.ramp_timer = 0;
                e.ramp_target = 0;
                e.charge_shot = 0;
            } else {
                let cst = self.lib.defs[e.def].charge_shot_ticks;
                if cst > 0 && e.deploy <= 0 && !e.dead {
                    e.charge_shot = (e.charge_shot + 1).min(cst);
                }
            }
        }

        // [M3] 2a. resolve any spells whose cast/travel window has elapsed. Done
        // before the spawner/area/snapshot passes so a landed spell's projectile,
        // effect zone, and unit spawns are picked up this tick exactly as a fresh
        // cast would be. Deterministic: index order in, index order out.
        if !self.pending_casts.is_empty() {
            let mut ready: Vec<PendingCast> = Vec::new();
            let mut keep: Vec<PendingCast> = Vec::with_capacity(self.pending_casts.len());
            for mut pc in self.pending_casts.drain(..) {
                pc.fuse -= 1;
                if pc.fuse <= 0 {
                    ready.push(pc);
                } else {
                    keep.push(pc);
                }
            }
            self.pending_casts = keep;
            for pc in ready {
                self.resolve_spell(pc.owner, pc.def, pc.pos);
            }
        }

        // 2b. spawner buildings periodically produce units
        let mut spawns: Vec<(u8, usize, V2)> = Vec::new();
        for i in 0..self.entities.len() {
            if self.entities[i].spawn_timer < 0 || !self.entities[i].active() {
                continue;
            }
            self.entities[i].spawn_timer -= 1;
            if self.entities[i].spawn_timer <= 0 {
                let def = self.entities[i].def;
                let owner = self.entities[i].owner;
                let pos = self.entities[i].pos;
                let d = &self.lib.defs[def];
                if let Some(sid) = d.spawn_id {
                    let dir = if owner == 0 { Fx::from_int(1) } else { Fx::from_int(-1) };
                    let front = V2::new(pos.x, pos.y + dir);
                    for _ in 0..d.spawn_count.max(1) {
                        spawns.push((owner, sid, front));
                    }
                    self.entities[i].spawn_timer = d.spawn_interval_ticks.max(1);
                } else if d.elixir_per_pulse > 0 && d.elixir_pulse_ticks > 0 {
                    // [eco] Elixir Collector/Pump: grant the owner elixir instead of
                    // spawning a unit. Respect the elixir cap (overflow is lost).
                    let gain = d.elixir_per_pulse;
                    let interval = d.elixir_pulse_ticks.max(1);
                    let o = owner as usize;
                    let cap = Fx::from_int(MAX_ELIXIR);
                    self.elixir[o] = (self.elixir[o] + Fx::from_int(gain as i64)).min(cap);
                    self.entities[i].spawn_timer = interval;
                }
            }
        }
        for (owner, sid, pos) in spawns {
            self.spawn_unit(owner, sid, pos);
        }

        // 2c. lingering area effects (poison/rage/heal zones)
        self.apply_area_effects();
        // [heal] continuous heal auras (Battle Healer): re-anchored to each healer.
        self.apply_heal_auras();

        let n = self.entities.len();
        let snap = Snap::build(&self.entities, &self.lib);

        // [path] Solid structures (towers + buildings) that ground units route
        // around. Collected once per tick in entity order so the grid build stays
        // deterministic (tiny O(structures) list — not every troop).
        let obstacles: Vec<usize> = (0..n)
            .filter(|&j| snap.alive[j] && (snap.is_tower[j] || snap.is_building[j]))
            .collect();
        // [path] Per-tick tile passability grid + the persistent flow-field memo
        // (taken out of self so the movement loop can borrow it mutably alongside
        // the immutable entity/lib borrows; restored at the end of the tick).
        let grid = PathGrid::build(&snap, &obstacles);
        let mut path_cache = std::mem::take(&mut self.path_cache);
        path_cache.sync(&grid);

        let mut dmg = vec![0i32; n];
        let mut stun_add = vec![0i32; n];
        let mut slow_add = vec![0i32; n]; // [proj] on-hit slow accrued this tick
        let mut knock: Vec<V2> = vec![V2::default(); n]; // [kb] per-entity knockback displacement
        let mut set_pos: Vec<Option<V2>> = vec![None; n];
        let mut set_cd: Vec<Option<i32>> = vec![None; n];
        let mut set_reveal: Vec<Option<i32>> = vec![None; n]; // [inv] reveal window on attack
        let mut set_charge: Vec<Option<Fx>> = vec![None; n];
        let mut set_dash: Vec<Option<i32>> = vec![None; n];   // [charge] leap ticks remaining
        let mut set_dashcd: Vec<Option<i32>> = vec![None; n]; // [charge] leap cooldown
        let mut set_target: Vec<u32> = vec![0; n]; // locked target id (0 = none)
        let mut set_engaged: Vec<bool> = vec![false; n];
        let mut snare_add: Vec<i32> = vec![0; n];     // [charge] snare applied to a hit victim
        let mut self_kill: Vec<bool> = vec![false; n];// [charge] Battle Ram self-destructs on connect
        let mut set_ramp_t: Vec<Option<i32>> = vec![None; n];   // [ramp] ramp_timer
        let mut set_ramp_tgt: Vec<Option<u32>> = vec![None; n]; // [ramp] ramp_target
        let mut set_cshot: Vec<Option<i32>> = vec![None; n];    // [ramp] charge_shot
        let mut pull: Vec<Option<V2>> = vec![None; n]; // [rfl] hook displacement (final pos)
        let mut suicide_kill: Vec<bool> = vec![false; n]; // [eco] kamikaze: die after its hit
        let mut new_proj: Vec<Projectile> = Vec::new();

        for i in 0..n {
            let e = &self.entities[i];
            if !snap.alive[i] || !e.active() || e.stun > 0 {
                continue;
            }
            if e.is_king && !e.activated {
                continue;
            }
            let d = &self.lib.defs[e.def];
            let mover = d.speed.0 > 0 && !e.is_tower;

            // Target LOCK: keep the committed target while valid+in leash, else acquire.
            let Some(tj) = self.target_for(i, &snap) else {
                // [charge] lost the target mid-leap: cancel the leap so we don't freeze.
                if e.dash > 0 {
                    set_dash[i] = Some(0);
                    set_dashcd[i] = Some(d.dash_cooldown_ticks);
                }
                continue;
            };
            set_target[i] = snap.id[tj];

            let ipos = V2::new(snap.x[i], snap.y[i]);
            // aim for the nearest point on the target's hitbox (box-aware for towers)
            let tpos = snap.closest_point(ipos, tj);
            let surf = snap.surf_dist(ipos, tj);
            let gap = surf - d.radius; // surface-to-surface distance
            // [ramp] siege dead zone (Mortar): cannot strike a target inside min_range.
            let past_min = d.min_range.0 == 0 || surf >= d.min_range;
            let in_attack = gap <= d.range && past_min;

            // [charge] Leap state machine: Bandit dash (invuln, 2x on arrival) and
            // Mega Knight jump (area damage on landing). A leaping unit flies straight
            // to its target ignoring the river and collisions.
            if d.dash_speed.0 > 0 {
                if e.dash > 0 {
                    set_engaged[i] = true; // hold the lock; don't re-acquire while leaping
                    let mv = tpos.sub(ipos).step_toward(d.dash_speed);
                    let np = Self::clamp_pos(V2::new(ipos.x + mv.x, ipos.y + mv.y));
                    set_pos[i] = Some(np);
                    let newgap = snap.surf_dist(np, tj) - d.radius;
                    if newgap <= d.range || e.dash <= 1 {
                        // landing this tick
                        set_dash[i] = Some(0);
                        set_dashcd[i] = Some(d.dash_cooldown_ticks);
                        if d.land_damage > 0 {
                            // Mega Knight: area damage where it lands
                            new_proj.push(Self::instant_splash(
                                e.owner, np, d.land_damage, d.land_radius, d.targets,
                            ));
                        }
                        if d.dash_damage_mult > 1 {
                            // Bandit: 2x single-target strike on arrival
                            self.apply_hit(
                                &mut dmg, tj, d.damage * d.dash_damage_mult, d, e.owner, &snap, tpos,
                            );
                            if d.snare_ticks > 0 {
                                snare_add[tj] = snare_add[tj].max(d.snare_ticks);
                            }
                            set_cd[i] = Some(cadence(d.hit_ticks, e.rage > 0, e.slow > 0));
                        }
                    } else {
                        set_dash[i] = Some(e.dash - 1);
                    }
                    continue;
                } else if gap > d.range
                    && gap >= d.dash_min
                    && gap <= d.dash_max
                    && e.dash_cd == 0
                    && same_side(ipos.y, tpos.y)
                {
                    // launch a leap (begins moving next tick); cap ticks so a vanished
                    // target can't strand the unit mid-leap.
                    let cap = (d.dash_max.div(d.dash_speed)).to_int_floor() as i32 * 2 + 4;
                    set_dash[i] = Some(cap.max(1));
                    set_engaged[i] = true;
                    continue;
                }
            }

            // [ramp] ramp-up builds only while actually attacking in range; any tick
            // spent out of range / in the dead zone resets it (so chasing / losing
            // the lock drops the Inferno back to stage 0).
            if !d.ramp_damage.is_empty() && !in_attack {
                set_ramp_t[i] = Some(0);
                set_ramp_tgt[i] = Some(0);
            }

            if in_attack {
                // in attack range: hold position, strike on cadence (after wind-up)
                set_engaged[i] = true;
                // [ramp] ramp-up bookkeeping (Inferno Tower / Inferno Dragon):
                // advance while locked on the SAME target, restart on a target change.
                let mut ramp_stage = 0usize;
                if !d.ramp_damage.is_empty() {
                    let tid = snap.id[tj];
                    let nt = if e.ramp_target == tid { e.ramp_timer + 1 } else { 0 };
                    set_ramp_t[i] = Some(nt);
                    set_ramp_tgt[i] = Some(tid);
                    ramp_stage = ((nt / d.ramp_stage_ticks.max(1)) as usize).min(d.ramp_damage.len() - 1);
                }

                if d.charge_shot_ticks > 0 {
                    // [ramp] charge-shot (Sparky): fire the big area shot only once the
                    // charge is full, then reset the reload. Charge itself builds (and
                    // resets on stun) in the timers pass.
                    if e.charge_shot >= d.charge_shot_ticks {
                        if d.proj_speed.0 == 0 {
                            // melee / hitscan charge-shot hit (charge-shot weapons are normally ranged)
                            self.apply_hit(&mut dmg, tj, d.damage, d, e.owner, &snap, tpos);
                        } else {
                            // [proj] data-driven projectile: homing by default; commit-to-ground
                            // (lobbed) or line/pierce shots per the card's flags.
                            // [kb] a rolling-pierce attack (Bowler's boulder) travels in a straight
                            // line through the target and beyond, knocking ground enemies back.
                            let is_line = d.pierce;
                            let is_roll = d.pierce && d.roll_range.0 > 0;
                            // [M2] dodge windows: by default a projectile commits to the
                            // target's ground position at fire time (dodgeable) for
                            // splash/area shots, and homes (tracks the entity) only for
                            // single-target "tracking" shots. A per-card `homing` flag
                            // (real CR `Homing`) overrides. Pierce/roll never home.
                            let homing = proj_homes(d) && !is_line && !is_roll;
                            let roll_dir = if is_roll {
                                tpos.sub(ipos).step_toward(Fx::from_int(1))
                            } else {
                                V2::default()
                            };
                            new_proj.push(Projectile {
                                owner: e.owner,
                                pos: ipos,
                                target: if homing { Some(snap.id[tj]) } else { None },
                                aim: tpos,
                                speed: d.proj_speed,
                                damage: d.damage,
                                splash: d.splash,
                                crown_pct: 100,
                                stun_ticks: 0,
                                hits_flying: matches!(d.targets, Targets::Air | Targets::Both),
                                hits_ground: matches!(d.targets, Targets::Ground | Targets::Both),
                                homing,                           // [proj]
                                slow_ticks: d.on_hit_slow_ticks,  // [proj]
                                chain_count: d.chain_count,       // [proj]
                                chain_radius: d.chain_radius,     // [proj]
                                pierce: is_line || is_roll,       // [proj]/[kb]
                                line_range: d.line_range,         // [proj]
                                boomerang: d.boomerang,           // [proj]
                                vel: if is_line {
                                    tpos.sub(ipos).step_toward(d.proj_speed)
                                } else {
                                    V2::default()
                                },
                                knockback: d.knockback,                                     // [kb]
                                roll_dir,                                                   // [kb]
                                roll_remaining: if is_roll { d.roll_range } else { Fx(0) }, // [kb]
                                ..Default::default()
                            });
                        }
                        set_cshot[i] = Some(0);
                    }
                } else if (d.damage > 0 || !d.ramp_damage.is_empty()) && e.cooldown == 0 {
                    if !e.engaged {
                        // first contact: wait this card's load time before the first hit
                        set_cd[i] = Some(d.load_ticks.max(1));
                    } else {
                        // [ramp] per-hit damage is the ramp stage value if ramping,
                        // otherwise the normal (optionally Prince-charged) damage.
                        let per_hit = if d.ramp_damage.is_empty() {
                            let charged = d.charge.map_or(false, |c| e.charge_dist >= c.min_distance);
                            let mult = if charged { d.charge.unwrap().damage_mult } else { 1 };
                            d.damage * mult
                        } else {
                            d.ramp_damage[ramp_stage]
                        };
                        if d.proj_speed.0 == 0 {
                            self.apply_hit(&mut dmg, tj, per_hit, d, e.owner, &snap, tpos);
                            // [proj] on-contact slow/freeze (Ice Wizard-style melee, if any)
                            if d.on_hit_slow_ticks > 0 || d.on_hit_freeze_ticks > 0 {
                                self.apply_on_hit_status(
                                    &mut slow_add, &mut stun_add, tj, d, e.owner, &snap, tpos,
                                );
                            }
                            // [rfl] Fisherman hook: reel the target in (or reel self to a building).
                            if d.pull_tiles.0 > 0 {
                                if snap.is_tower[tj] || snap.is_building[tj] {
                                    if pull[i].is_none() {
                                        let dir = tpos.sub(ipos);
                                        let amt = d.pull_tiles.min(dir.len());
                                        let mv = dir.step_toward(amt);
                                        let np = block_river(ipos, V2::new(ipos.x + mv.x, ipos.y + mv.y), d);
                                        pull[i] = Some(Self::clamp_pos(np));
                                    }
                                } else if pull[tj].is_none() {
                                    let jpos = V2::new(snap.x[tj], snap.y[tj]);
                                    let dir = ipos.sub(jpos);
                                    let amt = d.pull_tiles.min(dir.len());
                                    let mv = dir.step_toward(amt);
                                    let td = &self.lib.defs[self.entities[tj].def];
                                    let np = block_river(jpos, V2::new(jpos.x + mv.x, jpos.y + mv.y), td);
                                    pull[tj] = Some(Self::clamp_pos(np));
                                }
                            }
                        } else {
                            // [proj]/[kb] line/pierce/roll shot flags (homing by default)
                            let is_line = d.pierce;
                            let is_roll = d.pierce && d.roll_range.0 > 0;
                            // [M2] dodge windows: by default a projectile commits to the
                            // target's ground position at fire time (dodgeable) for
                            // splash/area shots, and homes (tracks the entity) only for
                            // single-target "tracking" shots. A per-card `homing` flag
                            // (real CR `Homing`) overrides. Pierce/roll never home.
                            let homing = proj_homes(d) && !is_line && !is_roll;
                            let roll_dir = if is_roll {
                                tpos.sub(ipos).step_toward(Fx::from_int(1))
                            } else {
                                V2::default()
                            };
                            new_proj.push(Projectile {
                                owner: e.owner,
                                pos: ipos,
                                target: if homing { Some(snap.id[tj]) } else { None },
                                aim: tpos,
                                speed: d.proj_speed,
                                // [shot] Hunter shotgun: the shot carries per-hit x the
                                // number of pellets that connect at this firing range.
                                damage: Self::shotgun_damage(per_hit, d.pellets, gap, d.range),
                                splash: d.splash,
                                crown_pct: 100,
                                stun_ticks: d.on_hit_freeze_ticks, // [proj] Electro stun on hit
                                hits_flying: matches!(d.targets, Targets::Air | Targets::Both),
                                hits_ground: matches!(d.targets, Targets::Ground | Targets::Both),
                                homing,                           // [proj]
                                slow_ticks: d.on_hit_slow_ticks,  // [proj] Ice Wizard
                                chain_count: d.chain_count,       // [proj] Electro Wizard/Spirit
                                chain_radius: d.chain_radius,     // [proj]
                                pierce: is_line || is_roll,       // [proj] Magic Archer/Executioner / [kb] Bowler
                                line_range: d.line_range,         // [proj]
                                boomerang: d.boomerang,           // [proj] Executioner
                                vel: if is_line {                 // [proj] straight-line velocity
                                    tpos.sub(ipos).step_toward(d.proj_speed)
                                } else {
                                    V2::default()
                                },
                                knockback: d.knockback,                                     // [kb]
                                roll_dir,                                                   // [kb]
                                roll_remaining: if is_roll { d.roll_range } else { Fx(0) }, // [kb]
                                ..Default::default()
                            });
                        }
                        if d.suicide {
                            // [eco] kamikaze: its damage above already landed; also
                            // apply any area stun/freeze, then mark it to die this tick.
                            if d.stun_ticks > 0 {
                                self.apply_stun_area(&mut stun_add, d, e.owner, &snap, tpos);
                            }
                            suicide_kill[i] = true;
                        } else {
                            set_cd[i] = Some(cadence(d.hit_ticks, e.rage > 0, e.slow > 0));
                            set_charge[i] = Some(Fx(0));
                        }
                        // [charge] on-hit melee riders
                        if d.snare_ticks > 0 {
                            // Ram Rider: long snare (reuses the slow status) on the target
                            snare_add[tj] = snare_add[tj].max(d.snare_ticks);
                        }
                        if d.knockback_tiles.0 > 0 {
                            // Battle Ram: small knockback shoving the target away
                            let dir = V2::new(snap.x[tj] - ipos.x, snap.y[tj] - ipos.y)
                                .step_toward(d.knockback_tiles);
                            knock[tj] = knock[tj].add(dir);
                        }
                        if d.dies_on_hit {
                            // Battle Ram: self-destructs on connect -> triggers death-spawn
                            self_kill[i] = true;
                        }
                        // [inv] Royal Ghost becomes targetable for a short window around each hit.
                        if d.invisible {
                            set_reveal[i] = Some(REVEAL_TICKS);
                        }
                        // [shot] Firecracker self-recoil: jump backward (away from the
                        // target) by `recoil` tiles on each shot. Ground-river + bounds
                        // respected; applied here so it fires exactly once per shot.
                        if d.recoil.0 > 0 {
                            let back = ipos.sub(tpos).step_toward(d.recoil);
                            let np = block_river(
                                ipos,
                                Self::clamp_pos(V2::new(ipos.x + back.x, ipos.y + back.y)),
                                d,
                            );
                            set_pos[i] = Some(np);
                        }
                    }
                }
            } else if mover {
                // [path] True tile pathfinding: steer toward the next waypoint on
                // the shortest route to the target — around the river, towers and
                // buildings — read from a cached BFS flow field. Flyers go straight.
                let goal =
                    path_goal(ipos, tpos, tj, d.flying, d.jumps_river, &snap, &grid, &mut path_cache);
                let charged = d.charge.map_or(false, |c| e.charge_dist >= c.min_distance);
                let mut speed = if charged {
                    d.speed.mul(d.charge.unwrap().speed_mult)
                } else {
                    d.speed
                };
                if e.rage > 0 {
                    speed = speed.mul(Fx::from_f64(1.35));
                }
                if e.slow > 0 {
                    speed = speed.mul(Fx::from_f64(0.65));
                }
                let mv = goal.sub(ipos).step_toward(speed);
                let mut np = V2::new(ipos.x + mv.x, ipos.y + mv.y);
                np = block_river(ipos, np, d);
                np = Self::clamp_pos(np);
                if d.charge.is_some() && same_side(ipos.y, tpos.y) {
                    set_charge[i] = Some(e.charge_dist + np.sub(ipos).len());
                } else if d.charge.is_some() {
                    set_charge[i] = Some(Fx(0));
                }
                set_pos[i] = Some(np);
            }
        }

        // apply movement / cooldown / charge / target lock / engagement
        for i in 0..n {
            if let Some(p) = set_pos[i] {
                self.entities[i].pos = p;
            }
            if let Some(c) = set_cd[i] {
                self.entities[i].cooldown = c;
            }
            if let Some(r) = set_reveal[i] {
                self.entities[i].reveal = r; // [inv]
            }
            if let Some(ch) = set_charge[i] {
                self.entities[i].charge_dist = ch;
            }
            if let Some(dd) = set_dash[i] {
                self.entities[i].dash = dd; // [charge] leap ticks remaining
            }
            if let Some(dc) = set_dashcd[i] {
                self.entities[i].dash_cd = dc; // [charge] leap cooldown
            }
            if let Some(v) = set_ramp_t[i] {
                self.entities[i].ramp_timer = v; // [ramp]
            }
            if let Some(v) = set_ramp_tgt[i] {
                self.entities[i].ramp_target = v; // [ramp]
            }
            if let Some(v) = set_cshot[i] {
                self.entities[i].charge_shot = v; // [ramp]
            }
            self.entities[i].target = if set_target[i] == 0 { None } else { Some(set_target[i]) };
            self.entities[i].engaged = set_engaged[i];
            if suicide_kill[i] {
                // [eco] drop to 0 hp so the end-of-tick death pass removes it
                // (and runs any death_spawn) through the normal path.
                self.entities[i].hp = 0;
            }
        }

        // [charge] Battle Ram's on-hit melee knockback is accumulated into `knock` and
        // applied once (with projectile knockback) by apply_knockback after step_projectiles.
        // [charge] snare applied on hit (Ram Rider / Bandit) reuses the slow status
        for i in 0..n {
            if snare_add[i] > 0 {
                self.entities[i].slow = self.entities[i].slow.max(snare_add[i]);
            }
        }
        // [charge] Battle Ram self-destructs on connect; death-spawn fires in apply_damage
        for i in 0..n {
            if self_kill[i] {
                self.entities[i].hp = 0;
            }
        }
        // [rfl] hook pulls override this tick's movement for the yanked body.
        for i in 0..n {
            if let Some(p) = pull[i] {
                self.entities[i].pos = p;
            }
        }


        // separate overlapping bodies (units push each other; buildings are solid)
        self.resolve_collisions();

        // projectiles (use post-move positions)
        let snap2 = Snap::build(&self.entities, &self.lib);
        self.step_projectiles(&snap2, &mut dmg, &mut stun_add, &mut slow_add, &mut knock);
        self.projectiles.extend(new_proj);

        // [kb] apply accumulated knockback (push targets back; bounds + river respected)
        self.apply_knockback(&knock);


        // resolve damage, deaths, crowns, king activation
        self.apply_damage(&dmg, &stun_add, &slow_add);

        // [path] return the flow-field memo to self for reuse next tick
        self.path_cache = path_cache;
    }

    /// Target selection with commitment: keep the currently-locked target while
    /// it stays valid and within leash range; only otherwise acquire a new one
    /// (nearest valid enemy in sight, else march toward the nearest tower).
    fn target_for(&self, i: usize, snap: &Snap) -> Option<usize> {
        let e = &self.entities[i];
        let d = &self.lib.defs[e.def];
        let ipos = V2::new(snap.x[i], snap.y[i]);
        let valid = |j: usize| -> bool {
            j != i
                && snap.alive[j]
                && !snap.untargetable[j] // [champ] can't lock a cloaked enemy
                && !snap.hidden[j] // [inv] invisible+unrevealed units can't be acquired (troops & towers)
                && snap.owner[j] != e.owner
                && d.targets.can_hit(snap.flying[j])
                && (!d.buildings_only || snap.is_tower[j] || snap.is_building[j])
                // [ramp] siege dead zone (Mortar): targets inside min_range are unhittable
                && (d.min_range.0 == 0 || snap.surf_dist(ipos, j) >= d.min_range)
        };

        // 1. Keep the locked target if still valid and within leash.
        // [M1] CR-plausible retarget rule: a unit holds its committed target while
        // the target stays inside the unit's own aggro radius (its `sight`, or its
        // attack `range` for long-range shooters) -- NOT an invented sight+2.5. Once
        // the target leaves that radius the unit re-acquires (step 2), which for a
        // lane pusher means falling back to marching on the nearest tower. Building
        // targeters never drop a building objective; a building-PREFERRING unit that
        // is only holding a troop re-scans so a building in sight can take priority.
        if let Some(tid) = e.target {
            if let Some(&tj) = snap.id2idx.get(&tid) {
                if valid(tj) {
                    let gap = snap.surf_dist(ipos, tj);
                    let building = snap.is_tower[tj] || snap.is_building[tj];
                    let keep_forever = d.buildings_only && building;
                    let leash = d.sight.max(d.range);
                    let prefer_rescan = d.prefers_buildings && !building;
                    if (keep_forever || gap <= leash) && !prefer_rescan {
                        return Some(tj);
                    }
                }
            }
        }

        // 2. Acquire: nearest valid in sight, else nearest tower/building to march on.
        // [A5] Ties on distance break deterministically toward the LOWER entity id
        // (not merely scan order), so acquisition never depends on entity storage
        // order. `better(cur, j, gap)` == "(gap, id) is strictly ahead of cur".
        let sight = d.sight.max(d.range);
        let mut best_sight: Option<(usize, Fx)> = None;
        let mut best_march: Option<(usize, Fx)> = None;
        let mut best_bldg: Option<(usize, Fx)> = None; // [ramp] nearest building/tower in sight (X-Bow)
        let better = |cur: Option<(usize, Fx)>, j: usize, gap: Fx| -> bool {
            match cur {
                None => true,
                Some((cj, cg)) => gap.0 < cg.0 || (gap.0 == cg.0 && snap.id[j] < snap.id[cj]),
            }
        };
        for j in 0..snap.x.len() {
            if !valid(j) {
                continue;
            }
            let gap = snap.surf_dist(ipos, j);
            let march_ok = if d.buildings_only {
                snap.is_tower[j] || snap.is_building[j]
            } else {
                snap.is_tower[j]
            };
            if march_ok && better(best_march, j, gap) {
                best_march = Some((j, gap));
            }
            if gap <= sight {
                if better(best_sight, j, gap) {
                    best_sight = Some((j, gap));
                }
                // [ramp] track the nearest building/tower for building-priority targeters
                if (snap.is_tower[j] || snap.is_building[j]) && better(best_bldg, j, gap) {
                    best_bldg = Some((j, gap));
                }
            }
        }
        // [ramp] X-Bow-style: prefer a building/tower in sight over any troop.
        if d.prefers_buildings {
            if let Some((j, _)) = best_bldg {
                return Some(j);
            }
        }
        best_sight.or(best_march).map(|(j, _)| j)
    }

    /// Push overlapping bodies apart. Troops (and flyers among themselves)
    /// separate; towers and buildings are immovable obstacles. Deploying units
    /// are not yet solid. Two relaxation passes keep crowds stable.
    fn resolve_collisions(&mut self) {
        let n = self.entities.len();
        let pos: Vec<V2> = self.entities.iter().map(|e| e.pos).collect();
        let half: Vec<V2> = self.entities.iter().map(|e| e.half).collect();
        let rad: Vec<Fx> = self.entities.iter().map(|e| self.lib.defs[e.def].radius).collect();
        let mass: Vec<Fx> = self.entities.iter().map(|e| self.lib.defs[e.def].mass).collect();
        let flying: Vec<bool> = self.entities.iter().map(|e| self.lib.defs[e.def].flying).collect();
        let solid: Vec<bool> = self
            .entities
            .iter()
            // [charge] a unit mid-leap phases through bodies (Bandit dash / MK jump)
            .map(|e| !e.dead && e.deploy <= 0 && e.dash == 0)
            .collect();
        let immov: Vec<bool> = self
            .entities
            .iter()
            .map(|e| e.is_tower || self.lib.defs[e.def].ctype == CardType::Building)
            .collect();
        let mut disp = vec![V2::default(); n];

        for _pass in 0..2 {
            for a in 0..n {
                if !solid[a] {
                    continue;
                }
                for b in (a + 1)..n {
                    if !solid[b] || flying[a] != flying[b] {
                        continue;
                    }
                    if immov[a] && immov[b] {
                        continue;
                    }
                    // box obstacle (tower) vs circle unit: push the unit out of the
                    // box inflated by its radius, along the least-penetration axis.
                    let box_a = half[a].x.0 > 0 || half[a].y.0 > 0;
                    let box_b = half[b].x.0 > 0 || half[b].y.0 > 0;
                    if box_a || box_b {
                        let (bx, cx) = if box_a { (a, b) } else { (b, a) };
                        let cb = pos[bx].add(disp[bx]);
                        let cc = pos[cx].add(disp[cx]);
                        let ex = half[bx].x + rad[cx];
                        let ey = half[bx].y + rad[cx];
                        let dx = cc.x - cb.x;
                        let dy = cc.y - cb.y;
                        let px = ex - dx.abs();
                        let py = ey - dy.abs();
                        if px.0 > 0 && py.0 > 0 {
                            if px <= py {
                                disp[cx].x = disp[cx].x + if dx.0 >= 0 { px } else { -px };
                            } else {
                                disp[cx].y = disp[cx].y + if dy.0 >= 0 { py } else { -py };
                            }
                        }
                        continue;
                    }
                    let pa = pos[a].add(disp[a]);
                    let pb = pos[b].add(disp[b]);
                    let delta = pb.sub(pa);
                    let mind = rad[a] + rad[b];
                    let dist2 = delta.len_sq();
                    if dist2 >= mind.mul(mind) {
                        continue;
                    }
                    let dist = delta.len();
                    let (dirx, diry, overlap) = if dist.0 > 0 {
                        (delta.x.div(dist), delta.y.div(dist), mind - dist)
                    } else {
                        // exactly coincident: nudge along +y deterministically
                        (Fx(0), Fx::from_int(1), mind)
                    };
                    if immov[a] {
                        disp[b] = disp[b].add(V2::new(dirx.mul(overlap), diry.mul(overlap)));
                    } else if immov[b] {
                        disp[a] = disp[a].sub(V2::new(dirx.mul(overlap), diry.mul(overlap)));
                    } else {
                        // split the push inversely to mass: heavier bodies move less
                        let total = mass[a] + mass[b];
                        let fa = if total.0 > 0 { overlap.mul(mass[b]).div(total) } else { overlap.divi(2) };
                        let fb = if total.0 > 0 { overlap.mul(mass[a]).div(total) } else { overlap.divi(2) };
                        disp[a] = disp[a].sub(V2::new(dirx.mul(fa), diry.mul(fa)));
                        disp[b] = disp[b].add(V2::new(dirx.mul(fb), diry.mul(fb)));
                    }
                }
            }
        }

        for i in 0..n {
            if disp[i].x.0 == 0 && disp[i].y.0 == 0 {
                continue;
            }
            let old = self.entities[i].pos;
            let np = V2::new(old.x + disp[i].x, old.y + disp[i].y);
            let d = &self.lib.defs[self.entities[i].def];
            let np = block_river(old, np, d);
            self.entities[i].pos = Self::clamp_pos(np);
        }
    }

    /// [charge] Build an instantly-detonating splash projectile at `pos` (used for
    /// Mega Knight deploy damage and jump-landing area damage). It lands next tick,
    /// hitting enemies within `radius` per the caster's `targets`.
    /// [shot] Effective shotgun shot damage: `base` per pellet times the number of
    /// pellets that connect. All `pellets` land point-blank; the hit count falls
    /// linearly to 1 at max range (surface gap == range). `pellets <= 1` => the
    /// ordinary single-shot damage, so every non-shotgun card is unaffected.
    fn shotgun_damage(base: i32, pellets: i32, gap: Fx, range: Fx) -> i32 {
        if pellets <= 1 {
            return base;
        }
        if range.0 <= 0 {
            return base * pellets;
        }
        let g = gap.max(Fx(0)).min(range);
        let frac = Fx::from_int(1) - g.div(range); // 1 at point-blank, 0 at max range
        let hits_fx = Fx::from_int(pellets as i64).mul(frac) + Fx::from_f64(0.5);
        let hits = hits_fx.to_int_floor().clamp(1, pellets as i64) as i32;
        base * hits
    }

    fn instant_splash(owner: u8, pos: V2, damage: i32, radius: Fx, targets: Targets) -> Projectile {
        let r = if radius.0 > 0 { radius } else { Fx::from_f64(1.0) };
        Projectile {
            owner,
            pos,
            target: None,
            aim: pos,
            speed: Fx::from_int(100), // effectively instant
            damage,
            splash: r,
            crown_pct: 100,
            stun_ticks: 0,
            hits_flying: matches!(targets, Targets::Air | Targets::Both),
            hits_ground: matches!(targets, Targets::Ground | Targets::Both),
            ..Default::default()
        }
    }

    /// Apply a melee/hitscan hit to target `tj` plus splash around `center`.
    fn apply_hit(
        &self,
        dmg: &mut [i32],
        tj: usize,
        amount: i32,
        d: &crate::cards::CardDef,
        owner: u8,
        snap: &Snap,
        center: V2,
    ) {
        if d.splash.0 <= 0 {
            dmg[tj] += scaled_tower(amount, 100, snap.is_tower[tj]);
            return;
        }
        let r = d.splash;
        let r2 = r.mul(r);
        for j in 0..snap.x.len() {
            if !snap.alive[j] || snap.untargetable[j] || snap.owner[j] == owner {
                continue; // [champ] cloaked units take no splash
            }
            if !d.targets.can_hit(snap.flying[j]) {
                continue;
            }
            let dx = snap.x[j] - center.x;
            let dy = snap.y[j] - center.y;
            if dx.mul(dx) + dy.mul(dy) <= r2 {
                dmg[j] += scaled_tower(amount, 100, snap.is_tower[j]);
            }
        }
    }

    /// [proj] Apply a landed melee/hitscan troop attack's on-hit slow/freeze to the
    /// target (and splash area, if any). Mirrors `apply_hit`'s target set.
    fn apply_on_hit_status(
        &self,
        slow_add: &mut [i32],
        stun_add: &mut [i32],
        tj: usize,
        d: &crate::cards::CardDef,
        owner: u8,
        snap: &Snap,
        center: V2,
    ) {
        let mut hit = |j: usize| {
            if d.on_hit_slow_ticks > 0 {
                slow_add[j] = slow_add[j].max(d.on_hit_slow_ticks);
            }
            if d.on_hit_freeze_ticks > 0 {
                stun_add[j] = stun_add[j].max(d.on_hit_freeze_ticks);
            }
        };
        if d.splash.0 <= 0 {
            hit(tj);
            return;
        }
        let r2 = d.splash.mul(d.splash);
        for j in 0..snap.x.len() {
            if !snap.alive[j] || snap.owner[j] == owner {
                continue;
            }
            if !d.targets.can_hit(snap.flying[j]) {
                continue;
            }
            let dx = snap.x[j] - center.x;
            let dy = snap.y[j] - center.y;
            if dx.mul(dx) + dy.mul(dy) <= r2 {
                hit(j);
            }
        }
    }

    /// [eco] Apply a suicide unit's on-contact stun/freeze to every enemy inside
    /// its splash radius around `center` (Ice Spirit freeze, Electro Spirit stun).
    fn apply_stun_area(
        &self,
        stun_add: &mut [i32],
        d: &crate::cards::CardDef,
        owner: u8,
        snap: &Snap,
        center: V2,
    ) {
        let r = if d.splash.0 > 0 { d.splash } else { Fx::from_f64(0.5) };
        let r2 = r.mul(r);
        for j in 0..snap.x.len() {
            if !snap.alive[j] || snap.owner[j] == owner {
                continue;
            }
            if !d.targets.can_hit(snap.flying[j]) {
                continue;
            }
            let dx = snap.x[j] - center.x;
            let dy = snap.y[j] - center.y;
            if dx.mul(dx) + dy.mul(dy) <= r2 {
                stun_add[j] = stun_add[j].max(d.stun_ticks);
            }
        }
    }

    fn step_projectiles(
        &mut self,
        snap: &Snap,
        dmg: &mut [i32],
        stun_add: &mut [i32],
        slow_add: &mut [i32],
        knock: &mut [V2], // [kb] accumulated per-entity push
    ) {
        let id2idx = &snap.id2idx;
        for p in self.projectiles.iter_mut() {
            if p.dead {
                continue;
            }
            // [kb] rolling-pierce projectile (Log, Bowler boulder): advance along a fixed
            // line, hitting each enemy in its path once and knocking ground enemies back.
            if p.pierce && p.roll_remaining.0 > 0 {
                let r2 = p.splash.mul(p.splash);
                for j in 0..snap.x.len() {
                    if !snap.alive[j] || snap.owner[j] == p.owner {
                        continue;
                    }
                    let flying = snap.flying[j];
                    if (flying && !p.hits_flying) || (!flying && !p.hits_ground) {
                        continue;
                    }
                    let jid = snap.id[j];
                    if p.hit_ids.contains(&jid) {
                        continue;
                    }
                    let dx = snap.x[j] - p.pos.x;
                    let dy = snap.y[j] - p.pos.y;
                    if dx.mul(dx) + dy.mul(dy) > r2 {
                        continue;
                    }
                    dmg[j] += scaled_tower(p.damage, p.crown_pct, snap.is_tower[j]);
                    if p.stun_ticks > 0 {
                        stun_add[j] = stun_add[j].max(p.stun_ticks);
                    }
                    // [kb] push along the roll direction (down-lane, "back")
                    if p.knockback.0 > 0 && !snap.is_tower[j] && !snap.is_building[j] {
                        let eff = p.knockback.mul(mass_resist(snap.mass[j]));
                        knock[j] = knock[j].add(V2::new(p.roll_dir.x.mul(eff), p.roll_dir.y.mul(eff)));
                    }
                    p.hit_ids.push(jid);
                }
                // advance the roller; expire once its travel is spent
                p.pos = V2::new(p.pos.x + p.roll_dir.x.mul(p.speed), p.pos.y + p.roll_dir.y.mul(p.speed));
                p.roll_remaining = p.roll_remaining - p.speed;
                if p.roll_remaining.0 <= 0 {
                    p.dead = true;
                }
                continue;
            }
            // [proj] pierce/boomerang line shots travel straight and sweep a corridor.
            if p.pierce {
                proj_pierce(p, snap, dmg, stun_add, slow_add);
                continue;
            }
            // [proj] homing shots track the live target; commit-to-ground shots fly
            // to the fixed `aim` point so the target can walk out of the way.
            let tpos = if p.homing {
                match p.target.and_then(|id| id2idx.get(&id)) {
                    // [champ] a cloaked target can't be tracked; fall back to aim.
                    Some(&idx) if snap.alive[idx] && !snap.untargetable[idx] => {
                        V2::new(snap.x[idx], snap.y[idx])
                    }
                    _ => p.aim,
                }
            } else {
                p.aim
            };
            let delta = tpos.sub(p.pos);
            let dist = delta.len();
            if p.speed.0 == 0 || dist <= p.speed {
                proj_impact(p, tpos, snap, dmg, stun_add, slow_add, knock);
                p.dead = true;
            } else {
                let mv = delta.step_toward(p.speed);
                p.pos = V2::new(p.pos.x + mv.x, p.pos.y + mv.y);
            }
        }
        self.projectiles.retain(|p| !p.dead);
    }

    /// [kb] Apply accumulated knockback displacements to entities. Towers/buildings are
    /// anchored (filtered at accumulation time); ground units slide along the river bank.
    fn apply_knockback(&mut self, knock: &[V2]) {
        for i in 0..self.entities.len() {
            let k = knock[i];
            if k.x.0 == 0 && k.y.0 == 0 {
                continue;
            }
            if self.entities[i].dead {
                continue;
            }
            let d = &self.lib.defs[self.entities[i].def];
            let old = self.entities[i].pos;
            let np = block_river(old, V2::new(old.x + k.x, old.y + k.y), d);
            self.entities[i].pos = Self::clamp_pos(np);
        }
    }

    fn apply_damage(&mut self, dmg: &[i32], stun_add: &[i32], slow_add: &[i32]) {
        let mut activate = [false; 2];
        for i in 0..self.entities.len() {
            if stun_add[i] > 0 {
                let s = self.entities[i].stun.max(stun_add[i]);
                self.entities[i].stun = s;
            }
            if slow_add[i] > 0 {
                // [proj] on-hit slow (Ice Wizard etc.); refreshes to the longer duration
                self.entities[i].slow = self.entities[i].slow.max(slow_add[i]);
            }
            if dmg[i] > 0 && !self.entities[i].dead {
                let mut rem = dmg[i];
                // [champ] Monk Deflect: incoming damage reduced while the window is active
                if self.entities[i].deflect > 0 {
                    let red = self.lib.defs[self.entities[i].def].ability_reduction;
                    if red > 0 {
                        rem = ((rem as i64) * (100 - red as i64) / 100) as i32;
                    }
                }
                // shield absorbs next
                if self.entities[i].shield_hp > 0 {
                    let absorbed = rem.min(self.entities[i].shield_hp);
                    self.entities[i].shield_hp -= absorbed;
                    rem -= absorbed;
                }
                if rem > 0 {
                    self.entities[i].hp -= rem;
                }
                // The king wakes only when the KING ITSELF is damaged...
                if self.entities[i].is_king {
                    activate[self.entities[i].owner as usize] = true;
                }
            }
        }

        // [rfl] Reflect (Electro Giant): a unit that was hit this tick emits a zap
        // pulse to every enemy that is attacking it from within reflect_radius,
        // dealing reflect_damage + a brief stun, at most once per reflect interval.
        // Deterministic snapshot-then-apply: read positions, accumulate, then apply.
        let n = self.entities.len();
        let positions: Vec<V2> = self.entities.iter().map(|e| e.pos).collect();
        let mut rfl_dmg = vec![0i32; n];
        let mut rfl_stun = vec![0i32; n];
        for i in 0..n {
            let e = &self.entities[i];
            if e.dead || e.stun > 0 || e.reflect_cd > 0 || dmg[i] <= 0 {
                continue;
            }
            let d = &self.lib.defs[e.def];
            if d.reflect_damage <= 0 || d.reflect_radius.0 <= 0 {
                continue;
            }
            let center = positions[i];
            let r2 = d.reflect_radius.mul(d.reflect_radius);
            let owner = e.owner;
            let eid = e.id;
            let mut fired = false;
            for j in 0..n {
                if j == i {
                    continue;
                }
                let a = &self.entities[j];
                // only enemies currently locked onto (attacking) this unit, in range
                if a.dead || a.owner == owner || a.target != Some(eid) {
                    continue;
                }
                let dx = positions[j].x - center.x;
                let dy = positions[j].y - center.y;
                if dx.mul(dx) + dy.mul(dy) > r2 {
                    continue;
                }
                rfl_dmg[j] += d.reflect_damage;
                rfl_stun[j] = rfl_stun[j].max(d.reflect_stun_ticks);
                fired = true;
            }
            if fired {
                self.entities[i].reflect_cd = d.reflect_interval_ticks.max(1);
            }
        }
        for j in 0..n {
            if rfl_stun[j] > 0 {
                let s = self.entities[j].stun.max(rfl_stun[j]);
                self.entities[j].stun = s;
            }
            if rfl_dmg[j] > 0 && !self.entities[j].dead {
                let mut rem = rfl_dmg[j];
                if self.entities[j].shield_hp > 0 {
                    let absorbed = rem.min(self.entities[j].shield_hp);
                    self.entities[j].shield_hp -= absorbed;
                    rem -= absorbed;
                }
                if rem > 0 {
                    self.entities[j].hp -= rem;
                }
                if self.entities[j].is_king {
                    activate[self.entities[j].owner as usize] = true;
                }
            }
        }

        let mut death_spawns: Vec<(u8, usize, V2)> = Vec::new();
        // [curse] on a cursed troop's death, a unit spawns FOR THE OPPONENT (the caster).
        let mut curse_spawns: Vec<(u8, usize, V2)> = Vec::new();
        // [charge] on-death area effects (Lumberjack drops a Rage zone)
        let mut death_areas: Vec<(u8, V2, SpellEffect, i32, Fx, i32)> = Vec::new();
        let mut revives: Vec<(u8, usize, V2, i32)> = Vec::new(); // [rev]
        // [edep] on-death area burst (Goblin Drill): (owner, center, damage, radius)
        let mut death_blasts: Vec<(u8, V2, i32, Fx)> = Vec::new();
        for i in 0..self.entities.len() {
            let e = &self.entities[i];
            if e.dead || e.hp > 0 {
                continue;
            }
            let owner = e.owner as usize;
            let owner_u8 = e.owner;
            let is_tower = e.is_tower;
            let is_king = e.is_king;
            let revived = e.revived; // [rev]
            let hatch = e.hatch;     // [rev]
            let pos = e.pos;
            let curse = e.curse;                 // [curse]
            let curse_spawn = e.curse_spawn;     // [curse]
            let d = &self.lib.defs[e.def];
            let cv = d.crown_value;
            // [curse] Goblin Curse: a cursed troop dying transforms into one unit for
            // the OPPONENT (DeathSpawnIsEnemy). Towers never carry a curse.
            if !is_tower && curse > 0 && curse_spawn >= 0 {
                curse_spawns.push((1 - owner_u8, curse_spawn as usize, pos));
            }
            // on-death spawns (Golem -> golemites, which themselves split again, ...).
            // A reborn unit (revived) drops nothing, so the chain can't loop forever.
            if !is_tower && !revived {
                if let Some(did) = d.death_id {
                    for k in 0..d.death_count.max(1) {
                        let off = Self::spawn_offset(k, d.death_count.max(1));
                        death_spawns.push((owner_u8, did, V2::new(pos.x + off.x, pos.y + off.y)));
                    }
                }
                // [rev] Phoenix egg: only a lifetime-expiry death (hatch) rebirths; if the
                // egg is killed early it just dies. Reborn spawns at revive_hp_pct% hp.
                if hatch {
                    if let Some(rid) = d.revive_as {
                        let rhp = (self.lib.defs[rid].hp * d.revive_hp_pct / 100).max(1);
                        revives.push((owner_u8, rid, pos, rhp));
                    }
                }
                // [charge] on-death area effect (Lumberjack Rage)
                if d.death_effect != SpellEffect::None {
                    death_areas.push((
                        e.owner,
                        e.pos,
                        d.death_effect,
                        d.death_effect_ticks,
                        d.death_effect_radius,
                        d.death_effect_power,
                    ));
                }
                // [edep] death-damage burst (Goblin Drill digs away / is destroyed).
                if d.death_damage > 0 {
                    death_blasts.push((e.owner, e.pos, d.death_damage, d.death_radius));
                }
            }
            self.entities[i].dead = true;
            if is_tower {
                if is_king {
                    // [crown] Destroying the enemy KING tower is an instant win worth
                    // the full 3 crowns TOTAL. It does NOT add `cv` (=3) on top of the
                    // crowns already earned from that side's princess towers -- doing so
                    // let a player's total exceed the hard cap of 3 (e.g. 1 princess + king
                    // reported as 4). Set the total to the win value instead of adding.
                    self.crowns[1 - owner] = 3;
                    if self.outcome == Outcome::Running {
                        self.outcome = if owner == 0 { Outcome::Win1 } else { Outcome::Win0 };
                    }
                } else {
                    // A destroyed PRINCESS (arena) tower is 1 crown; clamp to the [0,3]
                    // cap defensively so crowns can never exceed 3.
                    self.crowns[1 - owner] = (self.crowns[1 - owner] + cv).min(3);
                    // ...destroying a princess tower also wakes the loser's king tower.
                    activate[owner] = true;
                }
            }
        }
        for e in self.entities.iter_mut() {
            if e.is_king && activate[e.owner as usize] {
                e.activated = true;
            }
        }
        self.entities.retain(|e| !e.dead);
        for (owner, did, pos) in death_spawns {
            self.spawn_unit(owner, did, pos);
        }
        // [curse] cursed-death transforms (spawn one unit for the opponent/caster)
        for (owner, did, pos) in curse_spawns {
            self.spawn_unit(owner, did, pos);
        }
        // [charge] spawn on-death area effects (reuses the AreaEffect framework)
        for (owner, pos, effect, ticks, radius, power) in death_areas {
            let rad = if radius.0 > 0 { radius } else { Fx::from_f64(1.0) };
            let t = ticks.max(1);
            match effect {
                SpellEffect::None => {}
                SpellEffect::Rage => self.area_effects.push(AreaEffect {
                    owner, pos, radius: rad, ticks_left: t,
                    interval: 5, timer: 0, kind: AeoKind::Rage, power: 0, crown_pct: 100,
                    spawn_def: 0, spawn_count: 0,
                }),
                SpellEffect::Heal => self.area_effects.push(AreaEffect {
                    owner, pos, radius: rad, ticks_left: t,
                    interval: 10, timer: 10, kind: AeoKind::Heal, power, crown_pct: 100,
                    spawn_def: 0, spawn_count: 0,
                }),
                SpellEffect::Poison => self.area_effects.push(AreaEffect {
                    owner, pos, radius: rad, ticks_left: t,
                    interval: 10, timer: 10, kind: AeoKind::Damage, power, crown_pct: 100,
                    spawn_def: 0, spawn_count: 0,
                }),
                SpellEffect::Slow => self.area_effects.push(AreaEffect {
                    owner, pos, radius: rad, ticks_left: t,
                    interval: 5, timer: 0, kind: AeoKind::Slow, power: 0, crown_pct: 100,
                    spawn_def: 0, spawn_count: 0,
                }),
                SpellEffect::Freeze => self.area_effects.push(AreaEffect {
                    owner, pos, radius: rad, ticks_left: t,
                    interval: 5, timer: 0, kind: AeoKind::Freeze, power: 0, crown_pct: 100,
                    spawn_def: 0, spawn_count: 0,
                }),
                // Tornado/Spawn (and None) are not meaningful as on-death area effects.
                _ => {}
            }
        }
        // [rev] hatch reborn units (Phoenix) last, at reduced hp and one-shot.
        for (owner, rid, pos, hp) in revives {
            self.spawn_revived(owner, rid, pos, hp);
        }
        // [edep] emit death bursts as instant (speed-0) splash projectiles; they
        // resolve on the next tick's projectile pass, hitting only enemies.
        for (owner, pos, dmg, rad) in death_blasts {
            let splash = if rad.0 > 0 { rad } else { Fx::from_f64(1.0) };
            self.projectiles.push(Projectile {
                owner,
                pos,
                target: None,
                aim: pos,
                speed: Fx(0),
                damage: dmg,
                splash,
                crown_pct: 100,
                stun_ticks: 0,
                hits_flying: true,
                hits_ground: true,
                ..Default::default()
            });
        }
    }

    fn min_tower_hp(&self, player: u8) -> i32 {
        self.entities
            .iter()
            .filter(|e| e.is_tower && e.owner == player && e.alive())
            .map(|e| e.hp)
            .min()
            .unwrap_or(0)
    }

    fn resolve_time(&mut self) {
        if self.outcome != Outcome::Running {
            return;
        }
        let reg = REGULATION_TICKS;
        let end = REGULATION_TICKS + OVERTIME_TICKS;
        if self.tick == reg {
            if self.crowns[0] != self.crowns[1] {
                self.finish_by_crowns();
            } else {
                self.last_crowns_for_ot = self.crowns; // enter overtime (sudden death)
            }
        } else if self.tick > reg && self.tick < end {
            // [M5] Sudden death: crowns were tied entering overtime, so the instant
            // one player pulls AHEAD on crowns (destroys the first tower) they win.
            // A simultaneous same-tick exchange that leaves crowns still tied does
            // NOT end the match -- play continues until someone leads or time runs
            // out (then the lowest-tower-HP fallback below decides).
            if self.crowns[0] != self.crowns[1] {
                self.finish_by_crowns(); // first player ahead in OT wins immediately
            }
        } else if self.tick >= end {
            if self.crowns[0] != self.crowns[1] {
                self.finish_by_crowns();
            } else {
                // [M5] final fallback only when OT expires still tied: lowest remaining
                // tower HP loses; a dead-even board is a draw.
                let a = self.min_tower_hp(0);
                let b = self.min_tower_hp(1);
                self.outcome = if a > b {
                    Outcome::Win0
                } else if b > a {
                    Outcome::Win1
                } else {
                    Outcome::Draw
                };
            }
        }
    }

    fn finish_by_crowns(&mut self) {
        self.outcome = if self.crowns[0] > self.crowns[1] {
            Outcome::Win0
        } else if self.crowns[1] > self.crowns[0] {
            Outcome::Win1
        } else {
            Outcome::Draw
        };
    }
}

/// [M2] Whether a card's ranged attack homes (tracks the target entity) vs. commits
/// to a fixed ground point (dodgeable). An explicit per-card `homing` flag (the real
/// CR `Homing` projectile field) wins; otherwise the default is: single-target
/// "tracking" shots home, splash/area shots commit to ground. `commit_ground` still
/// forces a ground commit. Pierce/roll shots are handled separately and never home.
#[inline]
fn proj_homes(d: &crate::cards::CardDef) -> bool {
    d.homing.unwrap_or(!d.commit_ground && d.splash.0 == 0)
}

/// Scale an attack interval for rage (faster) / slow (slower). Integer math.
#[inline]
fn cadence(base: i32, rage: bool, slow: bool) -> i32 {
    let mut v = base;
    if rage {
        v = v * 100 / 135;
    }
    if slow {
        v = v * 154 / 100;
    }
    v.max(1)
}

/// [kb] Knockback/pull resistance from body mass, in [0, 1]. A body at or below
/// KB_REF_MASS takes the full push (1.0); heavier bodies are pushed in inverse
/// proportion, so very-high-mass units (Golem/PEKKA) are shoved the least.
#[inline]
fn mass_resist(mass: Fx) -> Fx {
    let one = Fx::from_int(1);
    if mass.0 <= KB_REF_MASS.0 {
        return one;
    }
    KB_REF_MASS.div(mass) // mass > ref, so this is < 1
}

/// Reduce damage dealt to crown towers by `crown_pct` percent.
#[inline]
fn scaled_tower(amount: i32, crown_pct: i32, is_tower: bool) -> i32 {
    if is_tower {
        amount * crown_pct / 100
    } else {
        amount
    }
}

/// [proj] Half-width of the corridor a pierce/boomerang line shot sweeps.
/// 0.4 tiles (26214 in Q16.16) — wide enough that the shot cannot step over a
/// small body between ticks at the projectile speeds used.
const PIERCE_CORRIDOR: Fx = Fx::from_raw(26214);

/// [proj] True if projectile `p` is allowed to hit a (flying?) body `j`.
#[inline]
fn proj_can_hit(p: &Projectile, snap: &Snap, j: usize) -> bool {
    if !snap.alive[j] || snap.untargetable[j] || snap.owner[j] == p.owner {
        return false; // [champ] cloaked units take no projectile damage
    }
    let flying = snap.flying[j];
    !((flying && !p.hits_flying) || (!flying && !p.hits_ground))
}

/// [proj] Resolve a non-pierce projectile that reaches its aim/target this tick:
/// top-K-by-HP (Lightning), splash, or single-target + chain (Electro).
fn proj_impact(
    p: &mut Projectile,
    tpos: V2,
    snap: &Snap,
    dmg: &mut [i32],
    stun_add: &mut [i32],
    slow_add: &mut [i32],
    knock: &mut [V2], // [kb] accumulate knockback on hit (Fireball/Rocket/Snowball)
) {
    let n = snap.x.len();
    // [kb] owner-facing fallback direction when a knockback has no natural vector
    let odir = if p.owner == 0 { Fx::from_int(1) } else { Fx::from_int(-1) };

    // Lightning-style: strike the K highest-HP enemies in the blast radius.
    if p.top_k > 0 {
        let mut chosen: Vec<u32> = Vec::new();
        for _ in 0..p.top_k {
            let mut best: Option<usize> = None;
            for j in 0..n {
                if !proj_can_hit(p, snap, j) || chosen.contains(&snap.id[j]) {
                    continue;
                }
                if snap.surf_dist(tpos, j) > p.splash {
                    continue;
                }
                best = Some(match best {
                    None => j,
                    Some(b) => {
                        // highest HP wins; ties broken by lowest entity id (deterministic)
                        if snap.hp[j] > snap.hp[b]
                            || (snap.hp[j] == snap.hp[b] && snap.id[j] < snap.id[b])
                        {
                            j
                        } else {
                            b
                        }
                    }
                });
            }
            match best {
                Some(j) => {
                    dmg[j] += scaled_tower(p.damage, p.crown_pct, snap.is_tower[j]);
                    if p.stun_ticks > 0 {
                        stun_add[j] = stun_add[j].max(p.stun_ticks);
                    }
                    if p.slow_ticks > 0 {
                        slow_add[j] = slow_add[j].max(p.slow_ticks);
                    }
                    chosen.push(snap.id[j]);
                }
                None => break,
            }
        }
        return;
    }

    // Area splash (Fireball/Arrows/Zap/Wizard/Bomber).
    if p.splash.0 > 0 {
        let r2 = p.splash.mul(p.splash);
        for j in 0..n {
            if !proj_can_hit(p, snap, j) {
                continue;
            }
            let dx = snap.x[j] - tpos.x;
            let dy = snap.y[j] - tpos.y;
            if dx.mul(dx) + dy.mul(dy) <= r2 {
                dmg[j] += scaled_tower(p.damage, p.crown_pct, snap.is_tower[j]);
                if p.stun_ticks > 0 {
                    stun_add[j] = stun_add[j].max(p.stun_ticks);
                }
                if p.slow_ticks > 0 {
                    slow_add[j] = slow_add[j].max(p.slow_ticks);
                }
                // [kb] radial knockback outward from the blast center
                if p.knockback.0 > 0 && !snap.is_tower[j] && !snap.is_building[j] {
                    let out = V2::new(dx, dy);
                    let dir = if out.x.0 == 0 && out.y.0 == 0 {
                        V2::new(Fx(0), odir)
                    } else {
                        out.step_toward(Fx::from_int(1))
                    };
                    let eff = p.knockback.mul(mass_resist(snap.mass[j]));
                    knock[j] = knock[j].add(V2::new(dir.x.mul(eff), dir.y.mul(eff)));
                }
            }
        }
        return;
    }

    // Single target (archers, musketeer, Electro Wizard primary zap).
    let mut arc = tpos;
    if let Some(&idx) = p.target.and_then(|id| snap.id2idx.get(&id)) {
        if snap.alive[idx] && !snap.untargetable[idx] && snap.owner[idx] != p.owner {
            dmg[idx] += scaled_tower(p.damage, p.crown_pct, snap.is_tower[idx]);
            if p.stun_ticks > 0 {
                stun_add[idx] = stun_add[idx].max(p.stun_ticks);
            }
            if p.slow_ticks > 0 {
                slow_add[idx] = slow_add[idx].max(p.slow_ticks);
            }
            // [kb] single-target knockback along the projectile's travel line
            if p.knockback.0 > 0 && !snap.is_tower[idx] && !snap.is_building[idx] {
                let base = tpos.sub(p.pos);
                let dir = if base.x.0 == 0 && base.y.0 == 0 {
                    V2::new(Fx(0), odir)
                } else {
                    base.step_toward(Fx::from_int(1))
                };
                let eff = p.knockback.mul(mass_resist(snap.mass[idx]));
                knock[idx] = knock[idx].add(V2::new(dir.x.mul(eff), dir.y.mul(eff)));
            }
            arc = V2::new(snap.x[idx], snap.y[idx]);
        }
    }

    // [proj] Chain: arc from the last hit to successive nearest unhit enemies.
    if p.chain_count > 0 {
        let mut chosen: Vec<u32> = Vec::new();
        if let Some(id) = p.target {
            chosen.push(id);
        }
        let r2 = p.chain_radius.mul(p.chain_radius);
        for _ in 0..p.chain_count {
            let mut best: Option<(usize, Fx)> = None;
            for j in 0..n {
                if !proj_can_hit(p, snap, j) || chosen.contains(&snap.id[j]) {
                    continue;
                }
                let dx = snap.x[j] - arc.x;
                let dy = snap.y[j] - arc.y;
                let d2 = dx.mul(dx) + dy.mul(dy);
                if d2 > r2 {
                    continue;
                }
                best = Some(match best {
                    None => (j, d2),
                    Some((bj, bd)) => {
                        // nearest wins; ties broken by lowest entity id (deterministic)
                        if d2 < bd || (d2 == bd && snap.id[j] < snap.id[bj]) {
                            (j, d2)
                        } else {
                            (bj, bd)
                        }
                    }
                });
            }
            match best {
                Some((j, _)) => {
                    dmg[j] += scaled_tower(p.damage, p.crown_pct, snap.is_tower[j]);
                    if p.stun_ticks > 0 {
                        stun_add[j] = stun_add[j].max(p.stun_ticks);
                    }
                    chosen.push(snap.id[j]);
                    arc = V2::new(snap.x[j], snap.y[j]);
                }
                None => break,
            }
        }
    }
}

/// [proj] A piercing / boomerang line shot (Magic Archer, Executioner): travel a
/// straight corridor hitting every enemy once per leg; the boomerang reverses at
/// the end of its outbound leg and strikes again on the way back.
fn proj_pierce(
    p: &mut Projectile,
    snap: &Snap,
    dmg: &mut [i32],
    stun_add: &mut [i32],
    slow_add: &mut [i32],
) {
    for j in 0..snap.x.len() {
        if !proj_can_hit(p, snap, j) || p.hit_ids.contains(&snap.id[j]) {
            continue;
        }
        if snap.surf_dist(p.pos, j) <= PIERCE_CORRIDOR {
            dmg[j] += scaled_tower(p.damage, p.crown_pct, snap.is_tower[j]);
            if p.stun_ticks > 0 {
                stun_add[j] = stun_add[j].max(p.stun_ticks);
            }
            if p.slow_ticks > 0 {
                slow_add[j] = slow_add[j].max(p.slow_ticks);
            }
            p.hit_ids.push(snap.id[j]);
        }
    }
    p.pos = V2::new(p.pos.x + p.vel.x, p.pos.y + p.vel.y);
    p.traveled = p.traveled + p.vel.len();
    if p.traveled >= p.line_range {
        if p.boomerang && !p.return_leg {
            p.vel = V2::new(-p.vel.x, -p.vel.y);
            p.traveled = Fx(0);
            p.return_leg = true;
            p.hit_ids.clear(); // the return leg can strike the same bodies again
        } else {
            p.dead = true;
        }
    }
}

/// [champ] Shortest distance from point `p` to segment `a`-`b` (fixed-point).
/// Used by the Golden Knight dash to damage a line of enemies.
fn dist_point_segment(p: V2, a: V2, b: V2) -> Fx {
    let ab = b.sub(a);
    let len2 = ab.len_sq();
    if len2.0 <= 0 {
        return p.sub(a).len();
    }
    let ap = p.sub(a);
    let dot = ap.x.mul(ab.x) + ap.y.mul(ab.y);
    let mut t = dot.div(len2);
    let one = Fx::from_int(1);
    if t.0 < 0 {
        t = Fx(0);
    } else if t > one {
        t = one;
    }
    let proj = V2::new(a.x + ab.x.mul(t), a.y + ab.y.mul(t));
    p.sub(proj).len()
}

/// [path] How many tiles of the flow-field gradient to walk when choosing a
/// steering waypoint (the string-pull horizon for smoothing around obstacles).
const PATH_LOOKAHEAD: usize = 12;

/// [path] Fixed neighbour order (orthogonals first, then diagonals) so every
/// BFS expansion and gradient tie-break is deterministic.
const NEIGHBORS: [(i64, i64); 8] =
    [(0, -1), (0, 1), (-1, 0), (1, 0), (-1, -1), (1, -1), (-1, 1), (1, 1)];

/// [path] Steering waypoint for a ground unit at `ipos` whose target's nearest
/// surface point is `tpos` (target entity index `tj`). If the straight segment
/// to the target is clear it heads straight there; otherwise it descends a
/// cached BFS flow field toward the target — routing around the river, towers
/// and buildings — and string-pulls to the farthest visible tile on that route.
/// Flyers ignore terrain. All fixed-point integer math — bit-reproducible.
fn path_goal(
    ipos: V2,
    tpos: V2,
    tj: usize,
    flying: bool,
    jumps: bool,
    snap: &Snap,
    grid: &PathGrid,
    cache: &mut PathCache,
) -> V2 {
    if flying {
        return tpos;
    }
    // Fast path: an unobstructed straight shot (open lane, or lined up with a
    // bridge) needs no routing — this keeps open-field motion a straight line.
    if grid.los(ipos, tpos, jumps) {
        return tpos;
    }
    let (ucx, ucy) = PathGrid::tile_of(ipos);
    let field = cache.field_for(grid, snap, tj, jumps);
    if field[grid.idx(ucx, ucy)] == u16::MAX {
        return tpos; // unreachable (e.g. boxed in) — fall back to a straight push
    }
    // Walk the gradient down toward the target for up to PATH_LOOKAHEAD tiles.
    let mut tiles: Vec<(i64, i64)> = Vec::with_capacity(PATH_LOOKAHEAD);
    let (mut cx, mut cy) = (ucx, ucy);
    for _ in 0..PATH_LOOKAHEAD {
        if field[grid.idx(cx, cy)] == 0 {
            break;
        }
        match grid.best_neighbor(field, cx, cy, jumps) {
            Some((nx, ny)) if field[grid.idx(nx, ny)] < field[grid.idx(cx, cy)] => {
                tiles.push((nx, ny));
                cx = nx;
                cy = ny;
            }
            _ => break,
        }
    }
    if tiles.is_empty() {
        return tpos;
    }
    // String-pull: aim at the farthest traced tile still in clear line of sight.
    let mut goal = grid.tile_center(tiles[0].0, tiles[0].1);
    for &(tx, ty) in tiles.iter().rev() {
        let c = grid.tile_center(tx, ty);
        if grid.los(ipos, c, jumps) {
            goal = c;
            break;
        }
    }
    goal
}

/// Prevent ground units from entering the river except near a bridge.
fn block_river(old: V2, new: V2, d: &crate::cards::CardDef) -> V2 {
    if d.flying || d.jumps_river {
        return new;
    }
    if (new.y - RIVER_Y).abs() < RIVER_HALF {
        let near = (new.x - BRIDGE_L).abs().min((new.x - BRIDGE_R).abs());
        if near > BRIDGE_HALF {
            return V2::new(new.x, old.y); // slide along the bank toward a bridge
        }
    }
    new
}

/// [path] Tile passability grid for one tick: a tile is blocked when a solid
/// structure (tower or building) occupies it. The river is handled per unit
/// (river-jumpers ignore it), not baked into `blocked`. 18x32 tiles, row-major,
/// rebuilt each tick from the obstacle list.
struct PathGrid {
    blocked: Vec<bool>, // len COLS*ROWS, index = cy*COLS + cx
}

/// Inclusive tile ranges a footprint box [cx±ex] x [cy±ey] covers.
fn footprint(cx: Fx, cy: Fx, ex: Fx, ey: Fx) -> (i64, i64, i64, i64) {
    let ceil = |f: Fx| -((Fx(-f.0)).to_int_floor());
    let lo_x = (cx - ex).to_int_floor();
    let lo_y = (cy - ey).to_int_floor();
    let mut hi_x = ceil(cx + ex) - 1;
    let mut hi_y = ceil(cy + ey) - 1;
    if hi_x < lo_x {
        hi_x = lo_x;
    }
    if hi_y < lo_y {
        hi_y = lo_y;
    }
    (lo_x, hi_x, lo_y, hi_y)
}

impl PathGrid {
    fn build(snap: &Snap, obstacles: &[usize]) -> PathGrid {
        let mut blocked = vec![false; (COLS * ROWS) as usize];
        for &j in obstacles {
            let (ex, ey) = if snap.hx[j].0 > 0 || snap.hy[j].0 > 0 {
                (snap.hx[j], snap.hy[j]) // box (tower)
            } else {
                (snap.rad[j], snap.rad[j]) // circle (building)
            };
            let (lo_x, hi_x, lo_y, hi_y) = footprint(snap.x[j], snap.y[j], ex, ey);
            for ty in lo_y..=hi_y {
                for tx in lo_x..=hi_x {
                    if tx >= 0 && tx < COLS && ty >= 0 && ty < ROWS {
                        blocked[(ty * COLS + tx) as usize] = true;
                    }
                }
            }
        }
        PathGrid { blocked }
    }

    #[inline]
    fn idx(&self, cx: i64, cy: i64) -> usize {
        (cy * COLS + cx) as usize
    }

    #[inline]
    fn tile_of(p: V2) -> (i64, i64) {
        (
            p.x.to_int_floor().clamp(0, COLS - 1),
            p.y.to_int_floor().clamp(0, ROWS - 1),
        )
    }

    #[inline]
    fn tile_center(&self, cx: i64, cy: i64) -> V2 {
        let half = Fx::from_raw(crate::fixed::ONE / 2);
        V2::new(Fx::from_int(cx) + half, Fx::from_int(cy) + half)
    }

    /// A river tile that is not spanned by a bridge (impassable to non-jumpers).
    #[inline]
    fn river_blocked(cx: i64, cy: i64) -> bool {
        let half = Fx::from_raw(crate::fixed::ONE / 2);
        let center_y = Fx::from_int(cy) + half;
        if (center_y - RIVER_Y).abs() >= RIVER_HALF {
            return false;
        }
        let center_x = Fx::from_int(cx) + half;
        let on_bridge = (center_x - BRIDGE_L).abs() <= BRIDGE_HALF
            || (center_x - BRIDGE_R).abs() <= BRIDGE_HALF;
        !on_bridge
    }

    #[inline]
    fn passable(&self, cx: i64, cy: i64, jumps: bool) -> bool {
        cx >= 0
            && cx < COLS
            && cy >= 0
            && cy < ROWS
            && !self.blocked[(cy * COLS + cx) as usize]
            && (jumps || !Self::river_blocked(cx, cy))
    }

    /// Clear line of sight: every tile sampled along `a`→`b` is passable.
    fn los(&self, a: V2, b: V2, jumps: bool) -> bool {
        let d = b.sub(a);
        let step = Fx::from_raw(crate::fixed::ONE / 4); // 0.25 tile
        let len = d.len();
        let steps = (len.0 / step.0).max(1);
        let inc = d.step_toward(step);
        let mut p = a;
        for _ in 0..steps {
            let (cx, cy) = Self::tile_of(p);
            if !self.passable(cx, cy, jumps) {
                return false;
            }
            p = p.add(inc);
        }
        let (bx, by) = Self::tile_of(b);
        self.passable(bx, by, jumps)
    }

    /// BFS distance field (in tile steps) from target entity `tj`'s footprint to
    /// every reachable tile, 8-connected, without cutting blocked corners.
    fn field(&self, snap: &Snap, tj: usize, jumps: bool) -> Vec<u16> {
        let mut dist = vec![u16::MAX; (COLS * ROWS) as usize];
        let mut q = std::collections::VecDeque::new();
        let (ex, ey) = if snap.hx[tj].0 > 0 || snap.hy[tj].0 > 0 {
            (snap.hx[tj], snap.hy[tj])
        } else {
            (snap.rad[tj], snap.rad[tj])
        };
        let (lo_x, hi_x, lo_y, hi_y) = footprint(snap.x[tj], snap.y[tj], ex, ey);
        // Seed every footprint tile at distance 0 (even if blocked — a tower's
        // own tiles are the source; BFS flows out to the passable neighbours).
        for ty in lo_y..=hi_y {
            for tx in lo_x..=hi_x {
                if tx >= 0 && tx < COLS && ty >= 0 && ty < ROWS {
                    let i = (ty * COLS + tx) as usize;
                    if dist[i] != 0 {
                        dist[i] = 0;
                        q.push_back((tx, ty));
                    }
                }
            }
        }
        while let Some((cx, cy)) = q.pop_front() {
            let dcur = dist[(cy * COLS + cx) as usize];
            for &(dx, dy) in &NEIGHBORS {
                let nx = cx + dx;
                let ny = cy + dy;
                if !self.passable(nx, ny, jumps) {
                    continue;
                }
                if dx != 0
                    && dy != 0
                    && (!self.passable(cx + dx, cy, jumps) || !self.passable(cx, cy + dy, jumps))
                {
                    continue; // no diagonal slip through a blocked corner
                }
                let ni = (ny * COLS + nx) as usize;
                if dist[ni] == u16::MAX {
                    dist[ni] = dcur.saturating_add(1);
                    q.push_back((nx, ny));
                }
            }
        }
        dist
    }

    /// Passable, corner-safe neighbour of `(cx,cy)` with the lowest field value
    /// (ties broken by the fixed NEIGHBORS order for determinism).
    fn best_neighbor(&self, dist: &[u16], cx: i64, cy: i64, jumps: bool) -> Option<(i64, i64)> {
        let mut best: Option<((i64, i64), u16)> = None;
        for &(dx, dy) in &NEIGHBORS {
            let nx = cx + dx;
            let ny = cy + dy;
            if !self.passable(nx, ny, jumps) {
                continue;
            }
            if dx != 0
                && dy != 0
                && (!self.passable(cx + dx, cy, jumps) || !self.passable(cx, cy + dy, jumps))
            {
                continue;
            }
            let dv = dist[(ny * COLS + nx) as usize];
            if dv == u16::MAX {
                continue;
            }
            if best.map_or(true, |(_, bv)| dv < bv) {
                best = Some(((nx, ny), dv));
            }
        }
        best.map(|(c, _)| c)
    }
}

/// [path] Cross-tick flow-field memo. A field is a *pure function* of (blocked
/// tiles, target footprint, river-jump flag), so caching it cannot change the
/// simulation — only its speed. Invalidated wholesale whenever the blocked-tile
/// set changes (a building is built or destroyed), and per target whenever that
/// target moves to a new tile. Never part of replay state.
#[derive(Default)]
struct PathCache {
    prev_blocked: Vec<bool>,
    map: std::collections::HashMap<(u32, bool), (usize, Vec<u16>)>,
}

impl PathCache {
    /// Drop cached fields when the obstacle layout changed; bound memory growth
    /// (moving troop targets would otherwise accumulate stale fields).
    fn sync(&mut self, grid: &PathGrid) {
        if self.prev_blocked != grid.blocked {
            self.map.clear();
            self.prev_blocked = grid.blocked.clone();
        }
        if self.map.len() > 512 {
            self.map.clear();
        }
    }

    /// The flow field toward target `tj` (building it on a miss). Keyed by the
    /// target's id + jump flag; a cached field is reused only while the target
    /// still sits in the tile it was built from.
    fn field_for(&mut self, grid: &PathGrid, snap: &Snap, tj: usize, jumps: bool) -> &Vec<u16> {
        let id = snap.id[tj];
        let (cx, cy) = PathGrid::tile_of(V2::new(snap.x[tj], snap.y[tj]));
        let ttile = (cy * COLS + cx) as usize;
        let key = (id, jumps);
        let fresh = self.map.get(&key).map_or(false, |(bt, _)| *bt == ttile);
        if !fresh {
            let f = grid.field(snap, tj, jumps);
            self.map.insert(key, (ttile, f));
        }
        &self.map.get(&key).unwrap().1
    }
}

/// Per-tick immutable snapshot used for target queries and splash scans.
struct Snap {
    x: Vec<Fx>,
    y: Vec<Fx>,
    hx: Vec<Fx>, // box half-extents (0 => circle)
    hy: Vec<Fx>,
    owner: Vec<u8>,
    flying: Vec<bool>,
    rad: Vec<Fx>,
    mass: Vec<Fx>, // [kb] push weight, for knockback/pull resistance
    is_tower: Vec<bool>,
    is_building: Vec<bool>,
    alive: Vec<bool>,
    untargetable: Vec<bool>, // [champ] cloaked units: present, but can't be targeted/hit
    hidden: Vec<bool>, // [inv] invisible & unrevealed: excluded from target acquisition only
    id: Vec<u32>,
    hp: Vec<i32>, // [proj] current hp, for top-K-by-HP spell selection (Lightning)
    id2idx: std::collections::HashMap<u32, usize>,
}

impl Snap {
    /// Shortest distance from point `p` to entity `j`'s hitbox surface
    /// (0 if inside). Boxes for towers, circles otherwise.
    #[inline]
    fn surf_dist(&self, p: V2, j: usize) -> Fx {
        let cx = self.x[j];
        let cy = self.y[j];
        if self.hx[j].0 > 0 || self.hy[j].0 > 0 {
            let ex = ((p.x - cx).abs() - self.hx[j]).max(Fx(0));
            let ey = ((p.y - cy).abs() - self.hy[j]).max(Fx(0));
            (ex.mul(ex) + ey.mul(ey)).sqrt()
        } else {
            (V2::new(p.x - cx, p.y - cy)).len() - self.rad[j]
        }
    }

    /// Closest point on entity `j`'s hitbox to `p` (for walk-up targeting).
    #[inline]
    fn closest_point(&self, p: V2, j: usize) -> V2 {
        let c = V2::new(self.x[j], self.y[j]);
        if self.hx[j].0 > 0 || self.hy[j].0 > 0 {
            let x = p.x.max(c.x - self.hx[j]).min(c.x + self.hx[j]);
            let y = p.y.max(c.y - self.hy[j]).min(c.y + self.hy[j]);
            V2::new(x, y)
        } else {
            c
        }
    }
}

impl Snap {
    fn build(ents: &[Entity], lib: &Library) -> Snap {
        let n = ents.len();
        let mut s = Snap {
            x: Vec::with_capacity(n),
            y: Vec::with_capacity(n),
            hx: Vec::with_capacity(n),
            hy: Vec::with_capacity(n),
            owner: Vec::with_capacity(n),
            flying: Vec::with_capacity(n),
            rad: Vec::with_capacity(n),
            mass: Vec::with_capacity(n), // [kb]
            is_tower: Vec::with_capacity(n),
            is_building: Vec::with_capacity(n),
            alive: Vec::with_capacity(n),
            untargetable: Vec::with_capacity(n),
            hidden: Vec::with_capacity(n), // [inv]
            id: Vec::with_capacity(n),
            hp: Vec::with_capacity(n),
            id2idx: std::collections::HashMap::with_capacity(n),
        };
        for (i, e) in ents.iter().enumerate() {
            let d = &lib.defs[e.def];
            s.x.push(e.pos.x);
            s.y.push(e.pos.y);
            s.hx.push(e.half.x);
            s.hy.push(e.half.y);
            s.owner.push(e.owner);
            s.flying.push(d.flying);
            s.rad.push(d.radius);
            s.mass.push(d.mass); // [kb]
            s.is_tower.push(e.is_tower);
            s.is_building.push(d.ctype == CardType::Building);
            // A unit is a valid target only once placed (not during its deploy time).
            // [charge] a unit mid-leap with invuln (Bandit dash) is untargetable.
            let leaping_invuln = e.dash > 0 && d.dash_invuln;
            s.alive.push(e.alive() && e.deploy <= 0 && !leaping_invuln);
            // [champ] cloaked troops are on the field but cannot be targeted/hit.
            s.untargetable.push(e.cloak > 0);
            // [inv] invisible units are untargetable except in their post-attack reveal
            // window. This flag gates target ACQUISITION only; `alive` is untouched so
            // splash/area damage (apply_hit, projectile splash, area effects) still lands.
            s.hidden.push(d.invisible && e.reveal <= 0);
            s.id.push(e.id);
            s.hp.push(e.hp);
            s.id2idx.insert(e.id, i);
        }
        s
    }
}

// [rev] validation tests for revive + multi-stage death-spawn chaining
#[cfg(test)]
mod rev_tests {
    use super::*;
    use crate::cards::Library;

    fn kill_all(eng: &mut Engine) {
        for e in eng.entities.iter_mut() {
            e.hp = 0;
        }
    }
    fn count_def(eng: &Engine, name: &str) -> usize {
        let id = eng.lib.id_of(name).unwrap();
        eng.entities.iter().filter(|e| e.def == id).count()
    }

    #[test]
    fn elixir_golem_three_stage_split() {
        let mut eng = Engine::new(Library::default_lib());
        let gid = eng.lib.id_of("elixir_golem").unwrap();
        eng.spawn_unit(0, gid, V2::new(Fx::from_int(9), Fx::from_int(5)));
        // golem dies -> 2 golemites
        eng.entities[0].hp = 0;
        eng.step([Action::noop(), Action::noop()]);
        assert_eq!(count_def(&eng, "elixir_golemite"), 2);
        // each golemite dies -> 2 blobs each => 4 blobs
        kill_all(&mut eng);
        eng.step([Action::noop(), Action::noop()]);
        assert_eq!(count_def(&eng, "elixir_blob"), 4);
        // blobs have no death_spawn -> chain terminates
        kill_all(&mut eng);
        eng.step([Action::noop(), Action::noop()]);
        assert_eq!(eng.entities.len(), 0);
    }

    #[test]
    fn goblin_giant_death_spawns_two_spear_goblins() {
        let mut eng = Engine::new(Library::default_lib());
        let id = eng.lib.id_of("goblin_giant").unwrap();
        eng.spawn_unit(0, id, V2::new(Fx::from_int(9), Fx::from_int(5)));
        eng.entities[0].hp = 0;
        eng.step([Action::noop(), Action::noop()]);
        assert_eq!(count_def(&eng, "spear_goblins"), 2);
    }

    #[test]
    fn phoenix_revives_once_via_egg() {
        let mut eng = Engine::new(Library::default_lib());
        let pid = eng.lib.id_of("phoenix").unwrap();
        eng.spawn_unit(0, pid, V2::new(Fx::from_int(9), Fx::from_int(5)));
        // phoenix dies -> leaves one egg (not yet revived)
        eng.entities[0].hp = 0;
        eng.step([Action::noop(), Action::noop()]);
        assert_eq!(count_def(&eng, "phoenix_egg"), 1);
        assert!(eng.entities.iter().all(|e| !e.revived));
        // let the egg's 3.0s lifetime expire -> hatch reborn phoenix at revive_hp_pct% hp
        // (derive the expected hp from the card data so stat edits don't break this test)
        let egg_id = eng.lib.id_of("phoenix_egg").unwrap();
        let expect_hp = eng.lib.defs[pid].hp * eng.lib.defs[egg_id].revive_hp_pct / 100;
        for _ in 0..65 {
            eng.step([Action::noop(), Action::noop()]);
        }
        assert_eq!(count_def(&eng, "phoenix_egg"), 0);
        let reborn: Vec<&Entity> = eng.entities.iter().filter(|e| e.def == pid).collect();
        assert_eq!(reborn.len(), 1);
        assert!(reborn[0].revived);
        assert_eq!(reborn[0].hp, expect_hp);
        assert_eq!(reborn[0].max_hp, expect_hp);
        // reborn is one-shot: on death it leaves NO egg and does not revive
        kill_all(&mut eng);
        eng.step([Action::noop(), Action::noop()]);
        assert_eq!(count_def(&eng, "phoenix_egg"), 0);
        assert_eq!(eng.entities.len(), 0);
    }

    #[test]
    fn egg_killed_early_does_not_revive() {
        let mut eng = Engine::new(Library::default_lib());
        let pid = eng.lib.id_of("phoenix").unwrap();
        eng.spawn_unit(0, pid, V2::new(Fx::from_int(9), Fx::from_int(5)));
        eng.entities[0].hp = 0;
        eng.step([Action::noop(), Action::noop()]);
        assert_eq!(count_def(&eng, "phoenix_egg"), 1);
        // destroy the egg BEFORE its lifetime expires (hatch flag never set)
        kill_all(&mut eng);
        eng.step([Action::noop(), Action::noop()]);
        assert_eq!(count_def(&eng, "phoenix"), 0);
        assert_eq!(eng.entities.len(), 0);
    }

    // [crown] Regression test for the ">3 crowns" bug. Destroying one enemy
    // PRINCESS tower is 1 crown; then destroying the enemy KING tower is an
    // instant win worth the FULL 3 crowns TOTAL -- it must NOT stack +3 on top
    // of the princess crown (which produced an impossible total of 4). The
    // winner ends on exactly 3 and the loser's crowns are untouched.
    #[test]
    fn king_tower_kill_caps_crowns_at_three() {
        let mut eng = Engine::new(Library::default_lib());
        eng.reset(1, None); // spawn the four towers per side

        // Kill one of player 1's PRINCESS towers -> player 0 earns 1 crown.
        let pidx = eng
            .entities
            .iter()
            .position(|e| e.is_tower && !e.is_king && e.owner == 1)
            .expect("player 1 princess tower");
        eng.entities[pidx].hp = 0;
        eng.step([Action::noop(), Action::noop()]);
        assert_eq!(eng.crowns, [1, 0], "one princess tower = exactly 1 crown");
        assert_eq!(eng.outcome, Outcome::Running);

        // Now kill player 1's KING tower -> instant win at the full 3 crowns
        // TOTAL, never 1 + 3 = 4.
        let kidx = eng
            .entities
            .iter()
            .position(|e| e.is_tower && e.is_king && e.owner == 1)
            .expect("player 1 king tower");
        eng.entities[kidx].hp = 0;
        eng.step([Action::noop(), Action::noop()]);

        assert_eq!(eng.crowns[0], 3, "king kill caps the winner at 3 crowns, not 4");
        assert_eq!(eng.crowns[1], 0, "loser's crowns stay unchanged");
        assert_eq!(eng.outcome, Outcome::Win0);
    }
}

#[cfg(test)]
mod path_tests {
    // [path] Scenario tests for building-aware local avoidance.
    use super::*;
    use crate::cards::Library;
    use crate::fixed::{Fx, V2};

    fn at(x: f64, y: f64) -> V2 {
        V2::new(Fx::from_f64(x), Fx::from_f64(y))
    }

    /// A lone building squarely in a ground unit's path must be WALKED AROUND,
    /// not stalled against. A player-0 knight marches up-lane toward the enemy
    /// tower while a *friendly* (non-target) cannon sits directly in its path at
    /// (3.5, 11). The knight never targets its own cannon, so the cannon is pure
    /// obstacle: without avoidance the knight jams against the cannon's front
    /// face (centre pinned near y=10.0, x=3.5) forever; with avoidance it peels
    /// off sideways, rounds the cannon, and continues past it.
    #[test]
    fn walks_around_lone_building() {
        let lib = Library::default_lib();
        let mut eng = Engine::new(lib);
        eng.reset(7, None);

        let knight = eng.lib.id_of("knight").unwrap();
        let cannon = eng.lib.id_of("cannon").unwrap();
        eng.spawn_unit(0, knight, at(3.5, 8.0));
        eng.spawn_unit(0, cannon, at(3.5, 11.0));
        let kid = eng
            .entities
            .iter()
            .find(|e| eng.lib.defs[e.def].name == "knight")
            .unwrap()
            .id;

        let mut max_dx = 0.0f64; // how far it ever steered off the straight line
        let mut final_y = 8.0f64;
        for _ in 0..250 {
            eng.step([Action::noop(), Action::noop()]);
            if let Some(k) = eng.entities.iter().find(|e| e.id == kid) {
                max_dx = max_dx.max((k.pos.x.to_f64() - 3.5).abs());
                final_y = k.pos.y.to_f64();
            }
        }

        // Steered sideways to go around (a stalled unit would stay at x=3.5)...
        assert!(
            max_dx > 0.3,
            "knight never steered around the building (max_dx={max_dx})"
        );
        // ...and got past the building instead of jamming at its ~y=10 face.
        assert!(
            final_y > 12.0,
            "knight failed to pass the building (final_y={final_y})"
        );
    }

    /// End-to-end pathfinding guard: tanks pushed up each lane must ROUTE across
    /// the river (bridges) and reach the enemy towers — chipping them — and the
    /// whole engagement must replay bit-identically. If the flow field stranded
    /// units at the river bank or against their own towers, no enemy tower would
    /// ever take damage and this fails.
    #[test]
    fn cross_river_push_is_deterministic_and_progresses() {
        let run = || {
            let mut eng = Engine::new(Library::default_lib());
            eng.reset(42, None);
            let giant = eng.lib.id_of("giant").unwrap();
            let knight = eng.lib.id_of("knight").unwrap();
            // a tank + support up each lane, both sides (clear of their own towers)
            eng.spawn_unit(0, giant, at(3.5, 9.0));
            eng.spawn_unit(0, knight, at(14.5, 9.0));
            eng.spawn_unit(1, giant, at(14.5, 23.0));
            eng.spawn_unit(1, knight, at(3.5, 23.0));
            let init_tower_hp: i32 =
                eng.entities.iter().filter(|e| e.is_tower).map(|e| e.hp).sum();
            for _ in 0..1600 {
                eng.step([Action::noop(), Action::noop()]);
            }
            let final_tower_hp: i32 =
                eng.entities.iter().filter(|e| e.is_tower).map(|e| e.hp).sum();
            let snapshot: Vec<(i64, i64)> =
                eng.entities.iter().map(|e| (e.pos.x.0, e.pos.y.0)).collect();
            (init_tower_hp, final_tower_hp, snapshot)
        };
        let (init1, final1, snap1) = run();
        let (init2, final2, snap2) = run();
        assert_eq!(init1, init2);
        assert_eq!(final1, final2, "engagement diverged (tower hp)");
        assert_eq!(snap1, snap2, "engagement diverged (positions)");
        assert!(
            final1 < init1,
            "pushers never damaged a tower — units stranded by pathfinding \
             (init={init1}, final={final1})"
        );
    }

    /// Determinism guard: the same scenario replays bit-identically. The
    /// avoidance math is pure fixed-point, so two runs must agree exactly.
    #[test]
    fn avoidance_is_deterministic() {
        let run = || {
            let lib = Library::default_lib();
            let mut eng = Engine::new(lib);
            eng.reset(7, None);
            let knight = eng.lib.id_of("knight").unwrap();
            let cannon = eng.lib.id_of("cannon").unwrap();
            eng.spawn_unit(0, knight, at(3.5, 8.0));
            eng.spawn_unit(0, cannon, at(3.5, 11.0));
            let kid = eng
                .entities
                .iter()
                .find(|e| eng.lib.defs[e.def].name == "knight")
                .unwrap()
                .id;
            let mut trail = Vec::new();
            for _ in 0..250 {
                eng.step([Action::noop(), Action::noop()]);
                if let Some(k) = eng.entities.iter().find(|e| e.id == kid) {
                    trail.push((k.pos.x.0, k.pos.y.0)); // raw Q16.16 ints
                }
            }
            trail
        };
        assert_eq!(run(), run(), "avoidance path diverged between replays");
    }
}

// [M1-M6] Scenario tests for the medium-severity fidelity fixes.
#[cfg(test)]
mod medium_gap_tests {
    use super::*;
    use crate::arena::{DEPLOY_TICKS, OVERTIME_TICKS, REGULATION_TICKS};
    use crate::cards::Library;
    use crate::fixed::{Fx, V2};

    fn at(x: f64, y: f64) -> V2 {
        V2::new(Fx::from_f64(x), Fx::from_f64(y))
    }
    fn noop() -> [Action; 2] {
        [Action::noop(), Action::noop()]
    }

    // ---- M1: targeting leash is the card's own sight, not an invented sight+2.5 ----
    #[test]
    fn m1_leash_drops_target_beyond_sight() {
        let mut eng = Engine::new(Library::default_lib());
        eng.reset(5, None);
        let knight = eng.lib.id_of("knight").unwrap(); // sight 5.5, not buildings_only
        eng.spawn_unit(0, knight, at(9.0, 8.0));
        let aid = eng.entities.last().unwrap().id;
        eng.spawn_unit(1, knight, at(9.0, 11.5));
        let eid = eng.entities.last().unwrap().id;
        for e in eng.entities.iter_mut() {
            if e.id == aid || e.id == eid {
                e.deploy = 0;
            }
        }
        // in sight (gap ~3.0 <= 5.5): the attacker acquires the enemy troop.
        eng.step(noop());
        let a = eng.entities.iter().find(|e| e.id == aid).unwrap();
        assert_eq!(a.target, Some(eid), "enemy in sight should be acquired");

        // push the enemy just beyond the knight's sight (gap ~6.1 > 5.5). The old
        // sight+2.5 leash would have kept it; the CR-plausible sight leash drops it,
        // so the attacker falls back to marching on a tower (target != the troop).
        for e in eng.entities.iter_mut() {
            if e.id == aid {
                e.pos = at(9.0, 8.0);
            }
            if e.id == eid {
                e.pos = at(9.0, 14.6);
            }
        }
        eng.step(noop());
        let a = eng.entities.iter().find(|e| e.id == aid).unwrap();
        assert_ne!(a.target, Some(eid), "target beyond sight must be dropped");
        assert!(a.target.is_some(), "attacker should re-acquire a march target (tower)");
    }

    // ---- M2: splash shots commit to ground (dodgeable); single-target shots home ----
    fn first_proj_of(eng: &mut Engine, owner: u8) -> Option<(bool, bool)> {
        for _ in 0..90 {
            eng.step(noop());
            if let Some(p) = eng.projectiles.iter().find(|p| p.owner == owner) {
                return Some((p.homing, p.target.is_some()));
            }
        }
        None
    }

    #[test]
    fn m2_splash_shot_commits_to_ground() {
        let mut eng = Engine::new(Library::default_lib());
        eng.reset(3, None);
        let wiz = eng.lib.id_of("wizard").unwrap(); // splash 1.3 => dodgeable
        eng.spawn_unit(0, wiz, at(3.5, 21.0)); // in range of enemy princess tower
        let (homing, has_target) = first_proj_of(&mut eng, 0).expect("wizard fired");
        assert!(!homing, "a splash/area shot commits to a ground point");
        assert!(!has_target, "a committed shot tracks no entity");
    }

    #[test]
    fn m2_single_target_shot_homes() {
        let mut eng = Engine::new(Library::default_lib());
        eng.reset(3, None);
        let musk = eng.lib.id_of("musketeer").unwrap(); // splash 0 => tracking shot
        eng.spawn_unit(0, musk, at(3.5, 20.0));
        let (homing, has_target) = first_proj_of(&mut eng, 0).expect("musketeer fired");
        assert!(homing, "a single-target tracking shot homes");
        assert!(has_target, "a homing shot tracks the target entity");
    }

    // ---- M3: a spell lands only after its cast/travel window (dodge chance) ----
    #[test]
    fn m3_spell_has_cast_travel_delay() {
        // No towers (plain Engine::new) so only the spell can damage the dummy.
        let mut eng = Engine::new(Library::default_lib());
        let fid = eng.lib.id_of("fireball").unwrap();
        let dummy = eng.lib.id_of("knight").unwrap(); // 1766 hp, survives one fireball
        eng.spawn_unit(1, dummy, at(9.0, 10.0));
        let did = eng.entities.last().unwrap().id;
        for e in eng.entities.iter_mut() {
            e.deploy = 0;
        }
        let hp0 = eng.entities.iter().find(|e| e.id == did).unwrap().hp;

        eng.cast_spell(0, fid, at(9.0, 10.0));
        assert!(eng.projectiles.is_empty(), "a cast spell is not yet in flight");
        let fuse = eng.spell_cast_fuse(fid);
        assert!(fuse >= 3, "fireball should have a real cast window");

        // during the window the spell has NOT landed
        for _ in 0..(fuse - 1) {
            eng.step(noop());
        }
        let hp_mid = eng.entities.iter().find(|e| e.id == did).map(|e| e.hp).unwrap();
        assert_eq!(hp_mid, hp0, "spell must not deal damage during its cast window");

        // after the window it resolves and damages whoever is still at the point
        eng.step(noop());
        let hp_after = eng.entities.iter().find(|e| e.id == did).map(|e| e.hp).unwrap();
        assert!(hp_after < hp0, "spell lands after the cast window");
    }

    // ---- M4: spawn uses each card's own deploy time; else the DEPLOY_TICKS fallback ----
    #[test]
    fn m4_per_card_deploy_time() {
        let mut eng = Engine::new(Library::default_lib());
        let golem = eng.lib.id_of("golem").unwrap(); // deploy_time 3.0s
        eng.spawn_unit(0, golem, at(9.0, 5.0));
        let e = eng.entities.last().unwrap();
        assert_eq!(e.deploy, eng.lib.defs[golem].deploy_ticks);
        assert_eq!(eng.lib.defs[golem].deploy_ticks, 3 * TICKS_PER_SECOND as i32);
        // a 1.0s card resolves to exactly the named fallback constant
        let knight = eng.lib.id_of("knight").unwrap();
        assert_eq!(eng.lib.defs[knight].deploy_ticks, DEPLOY_TICKS);
    }

    // ---- M5: overtime is sudden death; HP is only the final expiry fallback ----
    #[test]
    fn m5_overtime_first_tower_wins() {
        let mut eng = Engine::new(Library::default_lib());
        eng.reset(1, None);
        eng.crowns = [0, 0];
        eng.tick = REGULATION_TICKS;
        eng.resolve_time(); // tie -> enter overtime
        assert_eq!(eng.outcome, Outcome::Running);
        // first player to go ahead on crowns in OT wins immediately
        eng.tick = REGULATION_TICKS + 40;
        eng.crowns = [1, 0];
        eng.resolve_time();
        assert_eq!(eng.outcome, Outcome::Win0, "first tower in OT wins immediately");
    }

    #[test]
    fn m5_overtime_simultaneous_exchange_continues() {
        let mut eng = Engine::new(Library::default_lib());
        eng.reset(1, None);
        eng.crowns = [1, 1];
        eng.tick = REGULATION_TICKS;
        eng.resolve_time(); // enter overtime
        // a same-tick exchange that leaves crowns tied does NOT end the match
        eng.tick = REGULATION_TICKS + 40;
        eng.crowns = [2, 2];
        eng.resolve_time();
        assert_eq!(eng.outcome, Outcome::Running, "tied exchange keeps sudden death alive");
    }

    #[test]
    fn m5_overtime_expiry_falls_back_to_tower_hp() {
        let mut eng = Engine::new(Library::default_lib());
        eng.reset(1, None);
        eng.crowns = [0, 0];
        eng.tick = REGULATION_TICKS;
        eng.resolve_time(); // enter overtime
        // run out the clock still tied, but player 1 has a wounded tower
        if let Some(e) = eng.entities.iter_mut().find(|e| e.is_tower && e.owner == 1) {
            e.hp = 1;
        }
        eng.tick = REGULATION_TICKS + OVERTIME_TICKS;
        eng.resolve_time();
        assert_eq!(eng.outcome, Outcome::Win0, "lowest-tower-HP loses at time expiry");
    }

    // ---- M6: grouped spawns use 2D formations, not a single row ----
    #[test]
    fn m6_formations_are_clusters_not_lines() {
        assert_eq!(Engine::spawn_offset(0, 1), V2::new(Fx(0), Fx(0)));
        for &count in &[3i32, 4, 5, 6] {
            let mut xs: Vec<i64> = Vec::new();
            let mut ys: Vec<i64> = Vec::new();
            let mut pts: Vec<(i64, i64)> = Vec::new();
            for k in 0..count {
                let o = Engine::spawn_offset(k, count);
                xs.push(o.x.0);
                ys.push(o.y.0);
                pts.push((o.x.0, o.y.0));
                // offsets stay compact (on/next to the deploy tile)
                assert!(o.x.to_f64().abs() < 1.2 && o.y.to_f64().abs() < 1.2);
            }
            xs.sort();
            xs.dedup();
            ys.sort();
            ys.dedup();
            assert!(xs.len() >= 2, "count {count}: formation spans >1 column");
            assert!(ys.len() >= 2, "count {count}: formation spans >1 row (not a line)");
            // no two units share the exact same slot
            let mut sorted = pts.clone();
            sorted.sort();
            sorted.dedup();
            assert_eq!(sorted.len(), pts.len(), "count {count}: slots must be distinct");
        }
    }

    // ---- determinism guard across the new spell-delay + formation paths ----
    #[test]
    fn medium_changes_are_deterministic() {
        let run = || {
            let mut eng = Engine::new(Library::default_lib());
            let fid = eng.lib.id_of("fireball").unwrap();
            let barbs = eng.lib.id_of("barbarians").unwrap();
            eng.spawn_unit(1, barbs, at(9.0, 20.0));
            eng.cast_spell(0, fid, at(9.0, 20.0));
            let mut trail = Vec::new();
            for _ in 0..40 {
                eng.step(noop());
                for e in &eng.entities {
                    trail.push((e.id, e.pos.x.0, e.pos.y.0, e.hp));
                }
            }
            trail
        };
        assert_eq!(run(), run(), "spell-delay + formation paths must replay identically");
    }
}

// [champ] A1: the four new champions' activated abilities. Each test deploys the
// champion, makes it active + funds the elixir, fires Battle.activate_ability, and
// asserts the real on-field effect (not just the return flag).
#[cfg(test)]
mod champ_ability_tests {
    use super::*;
    use crate::cards::Library;
    use crate::fixed::{Fx, V2};

    fn at(x: f64, y: f64) -> V2 {
        V2::new(Fx::from_f64(x), Fx::from_f64(y))
    }
    fn activate_ready(eng: &mut Engine, owner: usize, id: u32) {
        // make the champion active and give the player plenty of elixir
        for e in eng.entities.iter_mut() {
            if e.id == id {
                e.deploy = 0;
            }
        }
        eng.elixir[owner] = Fx::from_int(10);
    }
    fn count_def(eng: &Engine, name: &str) -> usize {
        let did = eng.lib.id_of(name).unwrap();
        eng.entities.iter().filter(|e| e.def == did && !e.dead).count()
    }

    // Mighty Miner: LaneBomb drops an area blast (damage + knockback) at the vacated
    // spot and digs the champion to the mirror lane (reflect x, same row).
    #[test]
    fn mighty_miner_lanebomb_damages_and_switches_lane() {
        let mut eng = Engine::new(Library::default_lib());
        let mm = eng.lib.id_of("mighty_miner").unwrap();
        let knight = eng.lib.id_of("knight").unwrap();
        eng.spawn_unit(0, mm, at(4.5, 9.0)); // left lane (x < 9)
        let mm_id = eng.entities.last().unwrap().id;
        eng.spawn_unit(1, knight, at(4.5, 11.0)); // 2.0 away, inside the 3.0 blast
        let k_id = eng.entities.last().unwrap().id;
        for e in eng.entities.iter_mut() {
            e.deploy = 0;
        }
        activate_ready(&mut eng, 0, mm_id);
        let k_hp0 = eng.entities.iter().find(|e| e.id == k_id).unwrap().hp;
        let k_pos0 = eng.entities.iter().find(|e| e.id == k_id).unwrap().pos;

        assert!(eng.activate_ability(0), "Mighty Miner ability should fire");

        let k = eng.entities.iter().find(|e| e.id == k_id).unwrap();
        assert_eq!(k_hp0 - k.hp, 333, "bomb deals its 333 (L11) area damage");
        assert!(k.pos.sub(k_pos0).len().to_f64() > 0.0, "enemy is knocked back");
        // the champion digs to the mirror lane: x -> WIDTH - x (18 - 4.5 = 13.5)
        let m = eng.entities.iter().find(|e| e.id == mm_id).unwrap();
        assert!((m.pos.x.to_f64() - 13.5).abs() < 1e-6, "mighty miner switched lanes");
        assert!(lane_of(m.pos.x) == 1, "now in the right lane");
    }

    // Monk: Deflect opens a timed damage-reduction window; incoming damage while it
    // is active is scaled by (100 - reduction)%.
    #[test]
    fn monk_deflect_reduces_incoming_damage() {
        let mut eng = Engine::new(Library::default_lib());
        let monk = eng.lib.id_of("monk").unwrap();
        eng.spawn_unit(0, monk, at(9.0, 8.0));
        let m_id = eng.entities.last().unwrap().id;
        activate_ready(&mut eng, 0, m_id);

        assert!(eng.activate_ability(0), "Monk ability should fire");
        let mi = eng.entities.iter().position(|e| e.id == m_id).unwrap();
        assert!(eng.entities[mi].deflect > 0, "deflect window is open");

        let hp0 = eng.entities[mi].hp;
        let n = eng.entities.len();
        let mut dmg = vec![0i32; n];
        dmg[mi] = 1000;
        let zero = vec![0i32; n];
        eng.apply_damage(&dmg, &zero, &zero);
        // 65% reduction -> 1000 * 35% = 350 taken
        assert_eq!(hp0 - eng.entities[mi].hp, 350, "Deflect cuts 65% of the hit");
    }

    // Little Prince: the Guardian summon spawns one ChampionGuard ally.
    #[test]
    fn little_prince_summons_guardian() {
        let mut eng = Engine::new(Library::default_lib());
        let lp = eng.lib.id_of("little_prince").unwrap();
        eng.spawn_unit(0, lp, at(9.0, 8.0));
        let lp_id = eng.entities.last().unwrap().id;
        activate_ready(&mut eng, 0, lp_id);
        assert_eq!(count_def(&eng, "guardian"), 0, "no guardian before the ability");

        assert!(eng.activate_ability(0), "Little Prince ability should fire");
        assert_eq!(count_def(&eng, "guardian"), 1, "exactly one guardian is summoned");
        // the guardian belongs to the Little Prince's owner
        let gid = eng.lib.id_of("guardian").unwrap();
        let g = eng.entities.iter().find(|e| e.def == gid).unwrap();
        assert_eq!(g.owner, 0, "guardian is an ally");
    }

    // Boss Bandit: Warp jumps a fixed 6 tiles toward the enemy side, turns briefly
    // untargetable (invisible), and resets the current target.
    #[test]
    fn boss_bandit_warps_forward_and_cloaks() {
        let mut eng = Engine::new(Library::default_lib());
        let bb = eng.lib.id_of("boss_bandit").unwrap();
        eng.spawn_unit(0, bb, at(9.0, 8.0)); // owner 0 -> forward is +y
        let bb_id = eng.entities.last().unwrap().id;
        // pretend it had locked a target before warping
        for e in eng.entities.iter_mut() {
            if e.id == bb_id {
                e.target = Some(999);
            }
        }
        activate_ready(&mut eng, 0, bb_id);

        assert!(eng.activate_ability(0), "Boss Bandit ability should fire");
        let b = eng.entities.iter().find(|e| e.id == bb_id).unwrap();
        assert!((b.pos.y.to_f64() - 14.0).abs() < 1e-6, "warped ~6 tiles forward (8 -> 14)");
        assert!(b.cloak > 0, "boss bandit is briefly untargetable/invisible");
        assert_eq!(b.target, None, "warp resets the target");
    }

    // Ability activations stay deterministic across identical runs.
    #[test]
    fn champion_abilities_are_deterministic() {
        let run = || {
            let mut eng = Engine::new(Library::default_lib());
            let mm = eng.lib.id_of("mighty_miner").unwrap();
            let knight = eng.lib.id_of("knight").unwrap();
            eng.spawn_unit(0, mm, at(4.5, 9.0));
            let mm_id = eng.entities.last().unwrap().id;
            eng.spawn_unit(1, knight, at(4.5, 11.0));
            for e in eng.entities.iter_mut() {
                e.deploy = 0;
            }
            eng.elixir[0] = Fx::from_int(10);
            eng.activate_ability(0);
            let mut trail = Vec::new();
            for _ in 0..20 {
                eng.step([Action::noop(), Action::noop()]);
                for e in &eng.entities {
                    trail.push((e.id, e.pos.x.0, e.pos.y.0, e.hp, e.cloak, e.deflect));
                }
            }
            let _ = mm_id;
            trail
        };
        assert_eq!(run(), run(), "champion ability paths must replay identically");
    }
}

// [shot] A2: Hunter shotgun burst (range falloff) + Firecracker self-recoil.
#[cfg(test)]
mod shot_tests {
    use super::*;
    use crate::cards::Library;
    use crate::fixed::{Fx, V2};

    fn at(x: f64, y: f64) -> V2 {
        V2::new(Fx::from_f64(x), Fx::from_f64(y))
    }
    fn noop() -> [Action; 2] {
        [Action::noop(), Action::noop()]
    }

    // The pellet model: all pellets land point-blank, falling to one at max range,
    // and non-shotgun cards (pellets <= 1) are untouched.
    #[test]
    fn hunter_shotgun_falloff_is_monotonic() {
        let base = 84;
        let range = Fx::from_f64(4.0);
        let pb = Engine::shotgun_damage(base, 10, Fx::from_f64(0.0), range);
        let mid = Engine::shotgun_damage(base, 10, Fx::from_f64(2.0), range);
        let far = Engine::shotgun_damage(base, 10, range, range);
        assert_eq!(pb, base * 10, "all 10 pellets connect point-blank");
        assert_eq!(far, base, "one pellet connects at max range");
        assert!(pb > mid && mid > far, "monotonic range falloff: {pb} {mid} {far}");
        // a plain single-shot card is unaffected by the shotgun path
        assert_eq!(Engine::shotgun_damage(100, 1, Fx::from_f64(0.0), range), 100);
    }

    // In the live engine the Hunter's fired shot carries a far larger payload
    // point-blank than at the edge of its range.
    #[test]
    fn hunter_fires_bigger_burst_point_blank_than_far() {
        fn shot_damage(center_gap: f64) -> i32 {
            let mut eng = Engine::new(Library::default_lib());
            let hunter = eng.lib.id_of("hunter").unwrap();
            let knight = eng.lib.id_of("knight").unwrap();
            eng.spawn_unit(0, hunter, at(9.0, 8.0));
            eng.spawn_unit(1, knight, at(9.0, 8.0 + center_gap));
            for e in eng.entities.iter_mut() {
                e.deploy = 0;
            }
            for _ in 0..90 {
                // pin both so the firing range equals `center_gap` exactly
                for e in eng.entities.iter_mut() {
                    if e.owner == 0 {
                        e.pos = at(9.0, 8.0);
                    } else {
                        e.pos = at(9.0, 8.0 + center_gap);
                    }
                }
                eng.step(noop());
                if let Some(p) = eng.projectiles.iter().find(|p| p.owner == 0) {
                    return p.damage;
                }
            }
            0
        }
        let pb = shot_damage(1.2); // point-blank
        let far = shot_damage(5.0); // near the edge of range
        assert!(pb > 0 && far > 0, "hunter fired in both cases: pb={pb} far={far}");
        assert!(pb >= 84 * 6, "point-blank fires most of the 10 pellets: {pb}");
        assert!(pb > far * 3, "point-blank burst dwarfs the max-range shot: {pb} vs {far}");
    }

    // Firecracker jumps backward (away from its target) each time it fires.
    #[test]
    fn firecracker_recoils_backward_on_firing() {
        let mut eng = Engine::new(Library::default_lib());
        let fc = eng.lib.id_of("firecracker").unwrap();
        let knight = eng.lib.id_of("knight").unwrap();
        eng.spawn_unit(0, fc, at(9.0, 8.0));
        let fc_id = eng.entities.last().unwrap().id;
        eng.spawn_unit(1, knight, at(9.0, 12.0)); // in range (toward +y)
        for e in eng.entities.iter_mut() {
            e.deploy = 0;
        }
        let fc_y = |eng: &Engine| eng.entities.iter().find(|e| e.id == fc_id).unwrap().pos.y.to_f64();
        for _ in 0..140 {
            let before = fc_y(&eng);
            eng.step(noop());
            if eng.projectiles.iter().any(|p| p.owner == 0) {
                let after = fc_y(&eng);
                assert!(
                    after < before - 0.5,
                    "firecracker recoils backward when it fires: {before} -> {after}"
                );
                return;
            }
        }
        panic!("firecracker never fired");
    }
}

// [heal] A3: Battle Healer continuous heal aura.
#[cfg(test)]
mod heal_tests {
    use super::*;
    use crate::cards::Library;
    use crate::fixed::{Fx, V2};

    fn at(x: f64, y: f64) -> V2 {
        V2::new(Fx::from_f64(x), Fx::from_f64(y))
    }
    fn noop() -> [Action; 2] {
        [Action::noop(), Action::noop()]
    }

    #[test]
    fn battle_healer_heals_nearby_ally_and_self() {
        let mut eng = Engine::new(Library::default_lib());
        let bh = eng.lib.id_of("battle_healer").unwrap();
        let knight = eng.lib.id_of("knight").unwrap();
        eng.spawn_unit(0, bh, at(9.0, 8.0));
        let bh_id = eng.entities.last().unwrap().id;
        eng.spawn_unit(0, knight, at(9.6, 8.4)); // ally ~0.7 away (inside the 3.0 aura)
        let ally_id = eng.entities.last().unwrap().id;
        // a far-away ally that should NOT be reached by the aura
        eng.spawn_unit(0, knight, at(9.0, 14.0));
        let far_id = eng.entities.last().unwrap().id;
        for e in eng.entities.iter_mut() {
            e.deploy = 0;
        }
        // wound everyone to half hp
        for e in eng.entities.iter_mut() {
            e.hp = e.max_hp / 2;
        }
        let hp_of = |eng: &Engine, id: u32| eng.entities.iter().find(|e| e.id == id).unwrap().hp;
        let ally0 = hp_of(&eng, ally_id);
        let self0 = hp_of(&eng, bh_id);
        let far0 = hp_of(&eng, far_id);

        for _ in 0..40 {
            eng.step(noop());
        }

        let ally1 = hp_of(&eng, ally_id);
        let self1 = hp_of(&eng, bh_id);
        let far1 = hp_of(&eng, far_id);
        assert!(ally1 > ally0, "nearby ally heals over time: {ally0} -> {ally1}");
        assert!(self1 > self0, "the healer heals herself: {self0} -> {self1}");
        assert_eq!(far1, far0, "an ally outside the radius is not healed");
        let ally_max = eng.entities.iter().find(|e| e.id == ally_id).unwrap().max_hp;
        assert!(ally1 <= ally_max, "heal never overshoots max hp");
    }

    // The aura stops the moment the healer dies.
    #[test]
    fn heal_aura_stops_when_healer_dies() {
        let mut eng = Engine::new(Library::default_lib());
        let bh = eng.lib.id_of("battle_healer").unwrap();
        let knight = eng.lib.id_of("knight").unwrap();
        eng.spawn_unit(0, bh, at(9.0, 8.0));
        let bh_id = eng.entities.last().unwrap().id;
        eng.spawn_unit(0, knight, at(9.5, 8.0));
        let ally_id = eng.entities.last().unwrap().id;
        for e in eng.entities.iter_mut() {
            e.deploy = 0;
            if e.id == ally_id {
                e.hp = e.max_hp / 2;
            }
        }
        // kill the healer, then step: the ally must not keep healing
        for e in eng.entities.iter_mut() {
            if e.id == bh_id {
                e.hp = 0;
            }
        }
        eng.step(noop()); // healer is removed by the death pass
        let before = eng.entities.iter().find(|e| e.id == ally_id).unwrap().hp;
        for _ in 0..20 {
            eng.step(noop());
        }
        let after = eng.entities.iter().find(|e| e.id == ally_id).unwrap().hp;
        assert_eq!(after, before, "no healer => no heal");
    }
}

// [mix] A4: mixed-composition deploys (Goblin Gang, Rascals).
#[cfg(test)]
mod mix_tests {
    use super::*;
    use crate::cards::Library;
    use crate::fixed::{Fx, V2};

    fn at(x: f64, y: f64) -> V2 {
        V2::new(Fx::from_f64(x), Fx::from_f64(y))
    }

    #[test]
    fn goblin_gang_deploys_three_stab_plus_three_spear() {
        let mut eng = Engine::new(Library::default_lib());
        let gg = eng.lib.id_of("goblin_gang").unwrap();
        let sg = eng.lib.id_of("spear_goblins").unwrap();
        eng.place_card(0, gg, at(9.0, 6.0));
        let stab = eng.entities.iter().filter(|e| e.def == gg).count();
        let spear = eng.entities.iter().filter(|e| e.def == sg).count();
        assert_eq!(stab, 3, "3 stab goblins (the primary cluster)");
        assert_eq!(spear, 3, "3 spear goblins (the secondary deploy-spawn)");
        assert_eq!(eng.entities.len(), 6, "exactly 6 units, nothing else");
        // every unit belongs to the deploying player
        assert!(eng.entities.iter().all(|e| e.owner == 0));
    }

    #[test]
    fn rascals_deploys_one_boy_plus_two_girls() {
        let mut eng = Engine::new(Library::default_lib());
        let r = eng.lib.id_of("rascals").unwrap();
        let girl = eng.lib.id_of("rascal_girl").unwrap();
        eng.place_card(0, r, at(9.0, 6.0));
        let boys = eng.entities.iter().filter(|e| e.def == r).count();
        let girls = eng.entities.iter().filter(|e| e.def == girl).count();
        assert_eq!(boys, 1, "1 Rascal Boy tank");
        assert_eq!(girls, 2, "2 Rascal Girls");
        assert_eq!(eng.entities.len(), 3);
    }

    // The gang cards are deck-legal; the summon-only sub-unit is not.
    #[test]
    fn mix_card_deck_legality() {
        let lib = Library::default_lib();
        let legal = |n: &str| lib.deployable.contains(&lib.id_of(n).unwrap());
        assert!(legal("goblin_gang"), "Goblin Gang is deck-legal");
        assert!(legal("rascals"), "Rascals is deck-legal");
        assert!(!legal("rascal_girl"), "rascal_girl is summon-only");
    }

    #[test]
    fn mixed_deploy_is_deterministic() {
        let run = || {
            let mut eng = Engine::new(Library::default_lib());
            let gg = eng.lib.id_of("goblin_gang").unwrap();
            eng.place_card(0, gg, at(9.0, 6.0));
            let mut v: Vec<(usize, i64, i64)> =
                eng.entities.iter().map(|e| (e.def, e.pos.x.0, e.pos.y.0)).collect();
            v.sort();
            v
        };
        assert_eq!(run(), run(), "mixed deploy must replay identically");
    }
}

// [A5] Targeting commitment + deterministic acquisition tie-break.
#[cfg(test)]
mod aggro_tests {
    use super::*;
    use crate::cards::Library;
    use crate::fixed::{Fx, V2};

    fn at(x: f64, y: f64) -> V2 {
        V2::new(Fx::from_f64(x), Fx::from_f64(y))
    }
    fn noop() -> [Action; 2] {
        [Action::noop(), Action::noop()]
    }
    fn pin(eng: &mut Engine, id: u32, p: V2) {
        for e in eng.entities.iter_mut() {
            if e.id == id {
                e.pos = p;
            }
        }
    }
    fn target_of(eng: &Engine, id: u32) -> Option<u32> {
        eng.entities.iter().find(|e| e.id == id).unwrap().target
    }

    // A committed target is held while it stays within leash, is NOT stolen by a
    // closer newcomer, and is only re-acquired once it leaves the leash.
    #[test]
    fn commitment_holds_then_reacquires_beyond_leash() {
        let mut eng = Engine::new(Library::default_lib());
        let knight = eng.lib.id_of("knight").unwrap(); // sight 5.5, range 1.2
        eng.spawn_unit(0, knight, at(9.0, 8.0));
        let aid = eng.entities.last().unwrap().id;
        eng.spawn_unit(1, knight, at(9.0, 12.0)); // surf ~3.5, inside sight
        let a_id = eng.entities.last().unwrap().id;
        for e in eng.entities.iter_mut() {
            e.deploy = 0;
        }
        pin(&mut eng, aid, at(9.0, 8.0));
        eng.step(noop());
        assert_eq!(target_of(&eng, aid), Some(a_id), "acquires the enemy in sight");

        // a closer enemy appears -- it must NOT steal the committed target.
        eng.spawn_unit(1, knight, at(9.0, 9.5)); // surf ~1.0, much closer
        let b_id = eng.entities.last().unwrap().id;
        for e in eng.entities.iter_mut() {
            e.deploy = 0;
        }
        for _ in 0..3 {
            pin(&mut eng, aid, at(9.0, 8.0));
            pin(&mut eng, a_id, at(9.0, 12.0));
            pin(&mut eng, b_id, at(9.0, 9.5));
            eng.step(noop());
            assert_eq!(target_of(&eng, aid), Some(a_id), "closer newcomer must not steal aggro");
        }

        // push the committed target out past the leash (surf ~6.5 > sight 5.5): now it
        // re-acquires -- the nearest valid enemy, which is B.
        for _ in 0..2 {
            pin(&mut eng, aid, at(9.0, 8.0));
            pin(&mut eng, a_id, at(9.0, 15.0));
            pin(&mut eng, b_id, at(9.0, 9.5));
            eng.step(noop());
        }
        assert_eq!(target_of(&eng, aid), Some(b_id), "re-acquires once the old target leaves leash");
    }

    // On a distance tie, acquisition is deterministic: the lower entity id wins,
    // regardless of entity storage order.
    #[test]
    fn acquisition_ties_break_to_lower_id() {
        let mut eng = Engine::new(Library::default_lib());
        let knight = eng.lib.id_of("knight").unwrap();
        eng.spawn_unit(0, knight, at(9.0, 8.0));
        let aid = eng.entities.last().unwrap().id;
        eng.spawn_unit(1, knight, at(8.0, 11.0)); // equal distance from (9,8)
        let e1 = eng.entities.last().unwrap().id;
        eng.spawn_unit(1, knight, at(10.0, 11.0)); // mirror: same distance
        let e2 = eng.entities.last().unwrap().id;
        assert!(e1 < e2);
        for e in eng.entities.iter_mut() {
            e.deploy = 0;
        }
        pin(&mut eng, aid, at(9.0, 8.0));
        pin(&mut eng, e1, at(8.0, 11.0));
        pin(&mut eng, e2, at(10.0, 11.0));
        eng.step(noop());
        assert_eq!(target_of(&eng, aid), Some(e1), "equidistant tie resolves to the lower id");
    }
}

// [A6] Crown-tower per-spell damage percentages match the decoded tables.
#[cfg(test)]
mod crown_tests {
    use super::*;
    use crate::cards::Library;

    #[test]
    fn spell_crown_percentages_match_decoded_tables() {
        let lib = Library::default_lib();
        let pct = |n: &str| lib.defs[lib.id_of(n).unwrap()].crown_pct;
        // Spells whose crown reduction is stored as `CrownTowerDamagePercent`
        // (negative): crown% = 100 + value.
        assert_eq!(pct("arrows"), 20, "arrows -80");
        assert_eq!(pct("fireball"), 23, "fireball -77");
        assert_eq!(pct("rocket"), 23, "rocket -77");
        assert_eq!(pct("lightning"), 25, "lightning -75");
        assert_eq!(pct("snowball"), 25, "snowball -75");
        assert_eq!(pct("the_log"), 13, "log -87");
        assert_eq!(pct("tornado"), 30, "tornado -70");
        assert_eq!(pct("poison"), 22, "poison -78 (corrected from 30)");
        assert_eq!(pct("earthquake"), 60, "earthquake -40 (corrected from 33)");
        // Spells whose crown damage is a {BaseDamage, TowerDamage} struct:
        // crown% = round(TowerDamage / BaseDamage * 100).
        assert_eq!(pct("zap"), 25, "zap 19/75");
        assert_eq!(pct("freeze"), 26, "freeze 15/58");
    }
}

// [curse] A7: Clone, Suspicious Bush, Goblin Curse (Mother Witch + Goblinstein deferred).
#[cfg(test)]
mod a7_tests {
    use super::*;
    use crate::cards::Library;
    use crate::fixed::{Fx, V2};

    fn at(x: f64, y: f64) -> V2 {
        V2::new(Fx::from_f64(x), Fx::from_f64(y))
    }
    fn noop() -> [Action; 2] {
        [Action::noop(), Action::noop()]
    }
    fn count(eng: &Engine, name: &str) -> usize {
        let id = eng.lib.id_of(name).unwrap();
        eng.entities.iter().filter(|e| e.def == id && !e.dead).count()
    }

    // Clone duplicates every friendly troop in its radius at 1 hp.
    #[test]
    fn clone_duplicates_friendly_troops_at_one_hp() {
        let mut eng = Engine::new(Library::default_lib());
        let knight = eng.lib.id_of("knight").unwrap();
        eng.spawn_unit(0, knight, at(9.0, 8.0));
        eng.spawn_unit(0, knight, at(9.3, 8.3));
        // an enemy knight in the blast must NOT be cloned (only the caster's troops)
        eng.spawn_unit(1, knight, at(9.0, 8.6));
        for e in eng.entities.iter_mut() {
            e.deploy = 0;
        }
        let clone = eng.lib.id_of("clone").unwrap();
        eng.resolve_spell(0, clone, at(9.1, 8.1));
        // 2 originals + 2 clones (owner 0) + 1 enemy original = 5 knights total
        assert_eq!(count(&eng, "knight"), 5);
        let clones = eng.entities.iter().filter(|e| e.def == knight && e.owner == 0 && e.hp == 1).count();
        assert_eq!(clones, 2, "exactly two 1-hp clones for the caster");
        let enemy_clones = eng.entities.iter().filter(|e| e.def == knight && e.owner == 1).count();
        assert_eq!(enemy_clones, 1, "the enemy troop is not cloned");
    }

    // Suspicious Bush pops into two Bush Goblins when it dies.
    #[test]
    fn suspicious_bush_pops_two_goblins_on_death() {
        let mut eng = Engine::new(Library::default_lib());
        let bush = eng.lib.id_of("suspicious_bush").unwrap();
        eng.place_card(0, bush, at(9.0, 6.0));
        assert_eq!(count(&eng, "suspicious_bush"), 1);
        for e in eng.entities.iter_mut() {
            if e.def == bush {
                e.hp = 0;
            }
        }
        eng.step(noop());
        assert_eq!(count(&eng, "suspicious_bush"), 0, "bush consumed");
        assert_eq!(count(&eng, "bush_goblin"), 2, "two Bush Goblins pop out");
    }

    // Goblin Curse: a cursed enemy that dies becomes a Goblin for the CASTER.
    #[test]
    fn goblin_curse_transforms_dead_enemy_for_caster() {
        let mut eng = Engine::new(Library::default_lib());
        let knight = eng.lib.id_of("knight").unwrap();
        eng.spawn_unit(1, knight, at(9.0, 10.0)); // enemy of caster 0
        let eid = eng.entities.last().unwrap().id;
        for e in eng.entities.iter_mut() {
            e.deploy = 0;
        }
        let gc = eng.lib.id_of("goblin_curse").unwrap();
        eng.resolve_spell(0, gc, at(9.0, 10.0));
        assert!(
            eng.entities.iter().find(|e| e.id == eid).unwrap().curse > 0,
            "enemy is cursed"
        );
        // kill the cursed enemy
        for e in eng.entities.iter_mut() {
            if e.id == eid {
                e.hp = 0;
            }
        }
        eng.step(noop());
        let gob = eng.lib.id_of("goblins").unwrap();
        let for_caster = eng.entities.iter().filter(|e| e.def == gob && e.owner == 0).count();
        assert_eq!(for_caster, 1, "a Goblin spawns for the caster (opponent of the dead unit)");
    }

    // An un-cursed enemy death spawns nothing extra.
    #[test]
    fn uncursed_death_does_not_transform() {
        let mut eng = Engine::new(Library::default_lib());
        let knight = eng.lib.id_of("knight").unwrap();
        eng.spawn_unit(1, knight, at(9.0, 10.0));
        for e in eng.entities.iter_mut() {
            e.deploy = 0;
            e.hp = 0;
        }
        eng.step(noop());
        assert_eq!(count(&eng, "goblins"), 0, "no curse => no transform");
    }

    #[test]
    fn curse_and_clone_are_deterministic() {
        let run = || {
            let mut eng = Engine::new(Library::default_lib());
            let knight = eng.lib.id_of("knight").unwrap();
            eng.spawn_unit(0, knight, at(9.0, 8.0));
            eng.spawn_unit(0, knight, at(9.4, 8.2));
            for e in eng.entities.iter_mut() {
                e.deploy = 0;
            }
            let clone = eng.lib.id_of("clone").unwrap();
            eng.resolve_spell(0, clone, at(9.2, 8.1));
            let mut v: Vec<(usize, i64, i64, i32)> =
                eng.entities.iter().map(|e| (e.def, e.pos.x.0, e.pos.y.0, e.hp)).collect();
            v.sort();
            v
        };
        assert_eq!(run(), run());
    }
}

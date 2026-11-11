//! The deterministic battle engine. One `tick()` = one simulation step
//! (1/TICKS_PER_SECOND seconds). All state transitions are pure functions of
//! (previous state, actions, rng) using fixed-point math, so a match replays
//! bit-identically from its seed + action log.

use crate::arena::*;
use crate::cards::{CardType, Library, SpellEffect, Targets};
use crate::entity::{Entity, Projectile};
use crate::fixed::{Fx, V2};
use crate::rng::Rng;

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

#[derive(Clone, Copy)]
enum AeoKind {
    Damage, // DoT to enemies (poison, earthquake)
    Rage,   // buff allies
    Heal,   // heal allies
    Slow,   // slow enemies
    Freeze, // freeze enemies
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
}

pub struct Engine {
    pub lib: Library,
    pub rng: Rng,
    pub tick: i64,
    pub entities: Vec<Entity>,
    pub projectiles: Vec<Projectile>,
    area_effects: Vec<AreaEffect>,
    next_id: u32,
    pub elixir: [Fx; 2],
    pub deck: [Vec<usize>; 2], // 8 def ids each
    pub queue: [Vec<usize>; 2],// rotating 8; front HAND = hand, [HAND] = next
    pub crowns: [i32; 2],
    pub outcome: Outcome,
    king_down_dmg_seen: [bool; 2],
    last_crowns_for_ot: [i32; 2],
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
            next_id: 1,
            elixir: [Fx(0); 2],
            deck: [Vec::new(), Vec::new()],
            queue: [Vec::new(), Vec::new()],
            crowns: [0, 0],
            outcome: Outcome::Running,
            king_down_dmg_seen: [false, false],
            last_crowns_for_ot: [0, 0],
        }
    }

    fn default_deck(&self) -> Vec<usize> {
        // first 8 deployable troops/buildings (skip spells so a random policy still fights)
        let mut v: Vec<usize> = self
            .lib
            .deployable
            .iter()
            .copied()
            .filter(|&d| self.lib.defs[d].ctype != CardType::Spell)
            .take(8)
            .collect();
        while v.len() < 8 {
            v.push(self.lib.deployable[0]);
        }
        v
    }

    pub fn reset(&mut self, seed: u64, decks: Option<[Vec<usize>; 2]>) {
        self.rng = Rng::new(seed);
        self.tick = 0;
        self.entities.clear();
        self.projectiles.clear();
        self.area_effects.clear();
        self.next_id = 1;
        self.elixir = [Fx::from_int(START_ELIXIR); 2];
        self.crowns = [0, 0];
        self.outcome = Outcome::Running;
        self.king_down_dmg_seen = [false, false];
        self.last_crowns_for_ot = [0, 0];

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
            spawn_timer: -1,
            charge_dist: Fx(0),
            is_tower: true,
            is_king,
            activated: !(is_king && d.starts_dormant),
            engaged: false,
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
            spawn_timer: if d.spawn_id.is_some() && d.spawn_interval_ticks > 0 {
                d.spawn_interval_ticks
            } else {
                -1
            },
            charge_dist: Fx(0),
            is_tower: false,
            is_king: false,
            activated: true,
            engaged: false,
            dead: false,
        });
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
            self.try_deploy(p, actions[p]);
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
        let def = self.queue[player][slot];
        let d = &self.lib.defs[def];
        let cost = Fx::from_int(d.elixir as i64);
        if self.elixir[player] < cost {
            return;
        }
        // deploy snaps to a tile; troops/buildings only on your own territory
        // (plus an opened pocket once you've taken an enemy princess tower).
        let pos = snap_to_tile(Self::clamp_pos(V2::new(a.x, a.y)));
        if d.ctype != CardType::Spell && !self.deploy_allowed(player, pos) {
            return;
        }
        let is_spell = d.ctype == CardType::Spell;
        let count = d.count.max(1);

        self.elixir[player] = self.elixir[player] - cost;
        self.cycle(player, slot);

        if is_spell {
            self.cast_spell(player as u8, def, pos);
            return;
        }
        for k in 0..count {
            let off = Self::spawn_offset(k, count);
            self.spawn_unit(player as u8, def, V2::new(pos.x + off.x, pos.y + off.y));
        }
    }

    fn enemy_princess_alive(&self, enemy: u8, lane: u8) -> bool {
        self.entities.iter().any(|e| {
            e.is_tower && !e.is_king && e.owner == enemy && e.alive() && lane_of(e.pos.x) == lane
        })
    }

    /// Whether `player` may deploy a troop/building at `pos` this tick.
    pub fn deploy_allowed(&self, player: usize, pos: V2) -> bool {
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

    fn spawn_offset(k: i32, count: i32) -> V2 {
        if count <= 1 {
            return V2::new(Fx(0), Fx(0));
        }
        // deterministic line spread, ~0.6 tile spacing, centered on the tile
        let step = Fx::from_f64(0.6);
        let i = (k - count / 2) as i64;
        let row = (k % 2) as i64; // two staggered rows
        V2::new(step.muli(i), step.muli(row).divi(2))
    }

    fn cast_spell(&mut self, owner: u8, def: usize, pos: V2) {
        let d = &self.lib.defs[def];
        let (damage, radius, proj_speed, crown_pct, stun_ticks) =
            (d.damage, d.spell_radius, d.proj_speed, d.crown_pct, d.stun_ticks);
        let (effect, eticks, einterval, epower) =
            (d.effect, d.effect_ticks, d.effect_interval, d.effect_power);
        let rad = if radius.0 > 0 { radius } else { Fx::from_f64(1.0) };

        // instant damage component (Fireball/Arrows/Zap, and the hit of effect spells)
        if damage > 0 {
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
                hits_flying: true,
                hits_ground: true,
                dead: false,
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
            }),
            SpellEffect::Rage => self.area_effects.push(AreaEffect {
                owner, pos, radius: rad, ticks_left: eticks.max(1),
                interval: 5, timer: 0, kind: AeoKind::Rage, power: 0, crown_pct: 100,
            }),
            SpellEffect::Heal => self.area_effects.push(AreaEffect {
                owner, pos, radius: rad, ticks_left: eticks.max(1),
                interval: einterval.max(1), timer: einterval.max(1),
                kind: AeoKind::Heal, power: epower, crown_pct: 100,
            }),
        }
    }

    /// Apply lingering area effects (poison DoT, rage/heal zones). Deaths from
    /// DoT are picked up by the end-of-tick death pass in `apply_damage`.
    fn apply_area_effects(&mut self) {
        if self.area_effects.is_empty() {
            return;
        }
        let mut events: Vec<(AeoKind, u8, V2, Fx, i32, i32)> = Vec::new();
        for ae in self.area_effects.iter_mut() {
            if ae.ticks_left > 0 {
                ae.ticks_left -= 1;
            }
            ae.timer -= 1;
            if ae.timer <= 0 {
                events.push((ae.kind, ae.owner, ae.pos, ae.radius, ae.power, ae.crown_pct));
                ae.timer = ae.interval.max(1);
            }
        }
        self.area_effects.retain(|ae| ae.ticks_left > 0);
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
                }
            }
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
            if e.deploy > 0 {
                e.deploy -= 1;
            }
            if e.cooldown > 0 {
                e.cooldown -= 1;
            }
            if e.stun > 0 {
                e.stun -= 1;
            }
            if e.slow > 0 {
                e.slow -= 1;
            }
            if e.rage > 0 {
                e.rage -= 1;
            }
            if e.lifetime > 0 {
                e.lifetime -= 1;
                if e.lifetime == 0 {
                    e.hp = 0;
                }
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
                }
            }
        }
        for (owner, sid, pos) in spawns {
            self.spawn_unit(owner, sid, pos);
        }

        // 2c. lingering area effects (poison/rage/heal zones)
        self.apply_area_effects();

        let n = self.entities.len();
        let snap = Snap::build(&self.entities, &self.lib);

        let mut dmg = vec![0i32; n];
        let mut stun_add = vec![0i32; n];
        let mut set_pos: Vec<Option<V2>> = vec![None; n];
        let mut set_cd: Vec<Option<i32>> = vec![None; n];
        let mut set_charge: Vec<Option<Fx>> = vec![None; n];
        let mut set_target: Vec<u32> = vec![0; n]; // locked target id (0 = none)
        let mut set_engaged: Vec<bool> = vec![false; n];
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
            let Some(tj) = self.target_for(i, &snap) else { continue };
            set_target[i] = snap.id[tj];

            let ipos = V2::new(snap.x[i], snap.y[i]);
            // aim for the nearest point on the target's hitbox (box-aware for towers)
            let tpos = snap.closest_point(ipos, tj);
            let gap = snap.surf_dist(ipos, tj) - d.radius; // surface-to-surface distance

            if gap <= d.range {
                // in attack range: hold position, strike on cadence (after wind-up)
                set_engaged[i] = true;
                if d.damage > 0 && e.cooldown == 0 {
                    if !e.engaged {
                        // first contact: wait this card's load time before the first hit
                        set_cd[i] = Some(d.load_ticks.max(1));
                    } else {
                        let charged = d.charge.map_or(false, |c| e.charge_dist >= c.min_distance);
                        let mult = if charged { d.charge.unwrap().damage_mult } else { 1 };
                        if d.proj_speed.0 == 0 {
                            self.apply_hit(&mut dmg, tj, d.damage * mult, d, e.owner, &snap, tpos);
                        } else {
                            new_proj.push(Projectile {
                                owner: e.owner,
                                pos: ipos,
                                target: Some(snap.id[tj]),
                                aim: tpos,
                                speed: d.proj_speed,
                                damage: d.damage * mult,
                                splash: d.splash,
                                crown_pct: 100,
                                stun_ticks: 0,
                                hits_flying: matches!(d.targets, Targets::Air | Targets::Both),
                                hits_ground: matches!(d.targets, Targets::Ground | Targets::Both),
                                dead: false,
                            });
                        }
                        set_cd[i] = Some(cadence(d.hit_ticks, e.rage > 0, e.slow > 0));
                        set_charge[i] = Some(Fx(0));
                    }
                }
            } else if mover {
                let goal = goal_point(ipos, d, tpos);
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
            if let Some(ch) = set_charge[i] {
                self.entities[i].charge_dist = ch;
            }
            self.entities[i].target = if set_target[i] == 0 { None } else { Some(set_target[i]) };
            self.entities[i].engaged = set_engaged[i];
        }

        // separate overlapping bodies (units push each other; buildings are solid)
        self.resolve_collisions();

        // projectiles (use post-move positions)
        let snap2 = Snap::build(&self.entities, &self.lib);
        self.step_projectiles(&snap2, &mut dmg, &mut stun_add);
        self.projectiles.extend(new_proj);

        // resolve damage, deaths, crowns, king activation
        self.apply_damage(&dmg, &stun_add);
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
                && snap.owner[j] != e.owner
                && d.targets.can_hit(snap.flying[j])
                && (!d.buildings_only || snap.is_tower[j] || snap.is_building[j])
        };

        // 1. Keep the locked target if still valid and within leash.
        if let Some(tid) = e.target {
            if let Some(&tj) = snap.id2idx.get(&tid) {
                if valid(tj) {
                    let gap = snap.surf_dist(ipos, tj);
                    let building = snap.is_tower[tj] || snap.is_building[tj];
                    // building-targeters never drop their objective; others leash to sight+margin
                    let keep_forever = d.buildings_only && building;
                    let leash = d.sight.max(d.range) + Fx::from_f64(2.5);
                    if keep_forever || gap <= leash {
                        return Some(tj);
                    }
                }
            }
        }

        // 2. Acquire: nearest valid in sight, else nearest tower/building to march on.
        let sight = d.sight.max(d.range);
        let mut best_sight: Option<(usize, Fx)> = None;
        let mut best_march: Option<(usize, Fx)> = None;
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
            if march_ok && best_march.map_or(true, |(_, bd)| gap < bd) {
                best_march = Some((j, gap));
            }
            if gap <= sight && best_sight.map_or(true, |(_, bd)| gap < bd) {
                best_sight = Some((j, gap));
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
            .map(|e| !e.dead && e.deploy <= 0)
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
            if !snap.alive[j] || snap.owner[j] == owner {
                continue;
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

    fn step_projectiles(&mut self, snap: &Snap, dmg: &mut [i32], stun_add: &mut [i32]) {
        let id2idx = &snap.id2idx;
        for p in self.projectiles.iter_mut() {
            if p.dead {
                continue;
            }
            // homing target position (fall back to fixed aim if gone)
            let tpos = match p.target.and_then(|id| id2idx.get(&id)) {
                Some(&idx) if snap.alive[idx] => V2::new(snap.x[idx], snap.y[idx]),
                _ => p.aim,
            };
            let delta = tpos.sub(p.pos);
            let dist = delta.len();
            if p.speed.0 == 0 || dist <= p.speed {
                // impact
                let center = tpos;
                if p.splash.0 <= 0 {
                    if let Some(&idx) = p.target.and_then(|id| id2idx.get(&id)) {
                        if snap.alive[idx] && snap.owner[idx] != p.owner {
                            dmg[idx] += scaled_tower(p.damage, p.crown_pct, snap.is_tower[idx]);
                            if p.stun_ticks > 0 {
                                stun_add[idx] = stun_add[idx].max(p.stun_ticks);
                            }
                        }
                    }
                } else {
                    let r2 = p.splash.mul(p.splash);
                    for j in 0..snap.x.len() {
                        if !snap.alive[j] || snap.owner[j] == p.owner {
                            continue;
                        }
                        let flying = snap.flying[j];
                        if (flying && !p.hits_flying) || (!flying && !p.hits_ground) {
                            continue;
                        }
                        let dx = snap.x[j] - center.x;
                        let dy = snap.y[j] - center.y;
                        if dx.mul(dx) + dy.mul(dy) <= r2 {
                            dmg[j] += scaled_tower(p.damage, p.crown_pct, snap.is_tower[j]);
                            if p.stun_ticks > 0 {
                                stun_add[j] = stun_add[j].max(p.stun_ticks);
                            }
                        }
                    }
                }
                p.dead = true;
            } else {
                let mv = delta.step_toward(p.speed);
                p.pos = V2::new(p.pos.x + mv.x, p.pos.y + mv.y);
            }
        }
        self.projectiles.retain(|p| !p.dead);
    }

    fn apply_damage(&mut self, dmg: &[i32], stun_add: &[i32]) {
        let mut activate = [false; 2];
        for i in 0..self.entities.len() {
            if stun_add[i] > 0 {
                let s = self.entities[i].stun.max(stun_add[i]);
                self.entities[i].stun = s;
            }
            if dmg[i] > 0 && !self.entities[i].dead {
                // shield absorbs first
                let mut rem = dmg[i];
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
        let mut death_spawns: Vec<(u8, usize, V2)> = Vec::new();
        for i in 0..self.entities.len() {
            let e = &self.entities[i];
            if e.dead || e.hp > 0 {
                continue;
            }
            let owner = e.owner as usize;
            let is_tower = e.is_tower;
            let is_king = e.is_king;
            let d = &self.lib.defs[e.def];
            let cv = d.crown_value;
            // on-death spawns (Golem -> golemites, barrels, skeletons, ...)
            if !is_tower {
                if let Some(did) = d.death_id {
                    for k in 0..d.death_count.max(1) {
                        let off = Self::spawn_offset(k, d.death_count.max(1));
                        death_spawns.push((e.owner, did, V2::new(e.pos.x + off.x, e.pos.y + off.y)));
                    }
                }
            }
            self.entities[i].dead = true;
            if is_tower {
                self.crowns[1 - owner] += cv;
                // ...or when one of its PRINCESS towers is destroyed (not merely chipped).
                if !is_king {
                    activate[owner] = true;
                }
                if is_king && self.outcome == Outcome::Running {
                    self.outcome = if owner == 0 { Outcome::Win1 } else { Outcome::Win0 };
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
            if self.crowns != self.last_crowns_for_ot {
                self.finish_by_crowns(); // first tower in OT decides
            }
        } else if self.tick >= end {
            if self.crowns[0] != self.crowns[1] {
                self.finish_by_crowns();
            } else {
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

/// Reduce damage dealt to crown towers by `crown_pct` percent.
#[inline]
fn scaled_tower(amount: i32, crown_pct: i32, is_tower: bool) -> i32 {
    if is_tower {
        amount * crown_pct / 100
    } else {
        amount
    }
}

fn goal_point(ipos: V2, d: &crate::cards::CardDef, tpos: V2) -> V2 {
    if d.flying || d.jumps_river || same_side(ipos.y, tpos.y) {
        return tpos;
    }
    let bx = nearest_bridge_x(ipos.x);
    let off = RIVER_HALF + Fx::from_f64(0.4);
    let mouth_y = if tpos.y > ipos.y { RIVER_Y + off } else { RIVER_Y - off };
    V2::new(bx, mouth_y)
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

/// Per-tick immutable snapshot used for target queries and splash scans.
struct Snap {
    x: Vec<Fx>,
    y: Vec<Fx>,
    hx: Vec<Fx>, // box half-extents (0 => circle)
    hy: Vec<Fx>,
    owner: Vec<u8>,
    flying: Vec<bool>,
    rad: Vec<Fx>,
    is_tower: Vec<bool>,
    is_building: Vec<bool>,
    alive: Vec<bool>,
    id: Vec<u32>,
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
            is_tower: Vec::with_capacity(n),
            is_building: Vec::with_capacity(n),
            alive: Vec::with_capacity(n),
            id: Vec::with_capacity(n),
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
            s.is_tower.push(e.is_tower);
            s.is_building.push(d.ctype == CardType::Building);
            // A unit is a valid target only once placed (not during its deploy time).
            s.alive.push(e.alive() && e.deploy <= 0);
            s.id.push(e.id);
            s.id2idx.insert(e.id, i);
        }
        s
    }
}

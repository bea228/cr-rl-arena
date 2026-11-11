//! Card/tower definitions: load JSON, resolve to fixed-point + tick units once.

use crate::arena::TICKS_PER_SECOND;
use crate::fixed::Fx;
use serde::Deserialize;

pub const DEFAULT_CARDS_JSON: &str = include_str!("../data/cards.json");

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum CardType {
    Troop,
    Building,
    Spell,
    Tower,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Targets {
    Ground,
    Air,
    Both,
}

/// Lingering or applied spell effect, beyond instant damage.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum SpellEffect {
    None,
    Poison, // damage-over-time zone (enemies)
    Rage,   // speed/attack buff zone (allies)
    Heal,   // heal-over-time zone (allies)
    Freeze, // area freeze (enemies, duration)
    Slow,   // area slow (enemies, duration)
}

impl SpellEffect {
    fn parse(s: &str) -> SpellEffect {
        match s {
            "poison" => SpellEffect::Poison,
            "rage" => SpellEffect::Rage,
            "heal" => SpellEffect::Heal,
            "freeze" => SpellEffect::Freeze,
            "slow" => SpellEffect::Slow,
            _ => SpellEffect::None,
        }
    }
}

impl Targets {
    fn parse(s: &str) -> Targets {
        match s {
            "air" => Targets::Air,
            "both" => Targets::Both,
            _ => Targets::Ground,
        }
    }
    /// Can an attacker with `self` hit a (flying?) victim?
    #[inline]
    pub fn can_hit(self, victim_flying: bool) -> bool {
        match self {
            Targets::Both => true,
            Targets::Air => victim_flying,
            Targets::Ground => !victim_flying,
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct Charge {
    pub min_distance: Fx,
    pub speed_mult: Fx,
    pub damage_mult: i32, // integer multiplier (2x)
}

/// Fully resolved, hot-loop-ready card definition.
#[derive(Clone, Debug)]
pub struct CardDef {
    pub name: String,
    pub ctype: CardType,
    pub elixir: i32,
    pub hp: i32,
    pub damage: i32,
    pub hit_ticks: i32,
    pub load_ticks: i32, // delay from entering range to the FIRST hit landing
    pub range: Fx,
    pub sight: Fx,
    pub radius: Fx,
    pub mass: Fx,  // push weight: heavier bodies move less in a collision
    pub speed: Fx, // tiles per tick
    pub targets: Targets,
    pub count: i32,
    pub flying: bool,
    pub splash: Fx,
    pub proj_speed: Fx, // tiles per tick, 0 = instant
    pub buildings_only: bool,
    pub jumps_river: bool,
    pub lifetime_ticks: i32, // -1 = infinite
    pub spell_radius: Fx,
    pub crown_pct: i32,
    pub stun_ticks: i32,
    pub charge: Option<Charge>,
    pub crown_value: i32, // towers: crowns awarded when destroyed
    pub starts_dormant: bool,
    pub deployable: bool,  // false = summoned only (golemite, hut spawns), not deck-legal
    pub deploy_ticks: i32, // time from placement to becoming active
    pub shield: i32,       // starting shield hp absorbed before body hp
    // spawner building: periodically produces `spawn_count` of `spawn_name`
    pub spawn_count: i32,
    pub spawn_interval_ticks: i32,
    pub spawn_name: Option<String>,
    pub spawn_id: Option<usize>,
    // on death, spawn `death_count` of `death_name` at this unit's position
    pub death_count: i32,
    pub death_name: Option<String>,
    pub death_id: Option<usize>,
    // spell lingering/applied effect
    pub effect: SpellEffect,
    pub effect_ticks: i32,    // duration
    pub effect_interval: i32, // ticks between DoT/heal applications
    pub effect_power: i32,    // damage or heal per application
}

// ---- raw JSON shapes ----

#[derive(Deserialize, Default)]
struct RawCharge {
    min_distance: f64,
    speed_mult: f64,
    damage_mult: f64,
}

#[derive(Deserialize, Default)]
struct RawSpawn {
    card: String,
    count: Option<i64>,
    interval: Option<f64>,
}

#[derive(Deserialize, Default)]
struct RawDeath {
    card: String,
    count: Option<i64>,
}

#[derive(Deserialize, Default)]
struct RawCard {
    elixir: Option<i64>,
    #[serde(rename = "type")]
    ctype: Option<String>,
    hp: Option<i64>,
    damage: Option<i64>,
    hit_speed: Option<f64>,
    load_time: Option<f64>,
    range: Option<f64>,
    sight: Option<f64>,
    speed: Option<f64>,
    collision_radius: Option<f64>,
    mass: Option<f64>,
    targets: Option<String>,
    count: Option<i64>,
    flying: Option<bool>,
    splash_radius: Option<f64>,
    projectile_speed: Option<f64>,
    target_buildings_only: Option<bool>,
    jumps_river: Option<bool>,
    lifetime: Option<f64>,
    radius: Option<f64>,
    crown_tower_damage_pct: Option<i64>,
    stun: Option<f64>,
    charge: Option<RawCharge>,
    starts_dormant: Option<bool>,
    deploy_time: Option<f64>,
    shield: Option<i64>,
    spawn: Option<RawSpawn>,
    death_spawn: Option<RawDeath>,
    effect: Option<String>,
    effect_duration: Option<f64>,
    effect_interval: Option<f64>,
    effect_power: Option<i64>,
    deployable: Option<bool>,
}

#[derive(Deserialize)]
struct RawFile {
    cards: std::collections::BTreeMap<String, RawCard>,
    towers: std::collections::BTreeMap<String, RawCard>,
}

fn secs_to_ticks(s: f64) -> i32 {
    ((s * TICKS_PER_SECOND as f64).round() as i32).max(0)
}

fn per_tick(tiles_per_sec: f64) -> Fx {
    Fx::from_f64(tiles_per_sec / TICKS_PER_SECOND as f64)
}

fn resolve(name: &str, r: &RawCard, forced: Option<CardType>) -> CardDef {
    let ctype = forced.unwrap_or_else(|| match r.ctype.as_deref() {
        Some("spell") => CardType::Spell,
        Some("building") => CardType::Building,
        _ => CardType::Troop,
    });
    let charge = r.charge.as_ref().map(|c| Charge {
        min_distance: Fx::from_f64(c.min_distance),
        speed_mult: Fx::from_f64(c.speed_mult),
        damage_mult: c.damage_mult.round() as i32,
    });
    let lifetime = r.lifetime.unwrap_or(0.0);
    CardDef {
        name: name.to_string(),
        ctype,
        elixir: r.elixir.unwrap_or(0) as i32,
        hp: r.hp.unwrap_or(0) as i32,
        damage: r.damage.unwrap_or(0) as i32,
        hit_ticks: r.hit_speed.map(secs_to_ticks).unwrap_or(0).max(1),
        load_ticks: r
            .load_time
            .map(secs_to_ticks)
            .unwrap_or(TICKS_PER_SECOND as i32 / 5)
            .max(1),
        range: Fx::from_f64(r.range.unwrap_or(0.0)),
        sight: Fx::from_f64(r.sight.unwrap_or(5.5)),
        radius: Fx::from_f64(r.collision_radius.unwrap_or(0.5)),
        mass: Fx::from_f64(r.mass.unwrap_or_else(|| r.collision_radius.unwrap_or(0.5))),
        speed: per_tick(r.speed.unwrap_or(0.0)),
        targets: Targets::parse(r.targets.as_deref().unwrap_or("ground")),
        count: r.count.unwrap_or(1).max(1) as i32,
        flying: r.flying.unwrap_or(false),
        splash: Fx::from_f64(r.splash_radius.unwrap_or(0.0)),
        proj_speed: per_tick(r.projectile_speed.unwrap_or(0.0)),
        buildings_only: r.target_buildings_only.unwrap_or(false),
        jumps_river: r.jumps_river.unwrap_or(false),
        lifetime_ticks: if lifetime > 0.0 { secs_to_ticks(lifetime) } else { -1 },
        spell_radius: Fx::from_f64(r.radius.unwrap_or(0.0)),
        crown_pct: r.crown_tower_damage_pct.unwrap_or(100) as i32,
        stun_ticks: r.stun.map(secs_to_ticks).unwrap_or(0),
        charge,
        crown_value: 0,
        starts_dormant: r.starts_dormant.unwrap_or(false),
        deployable: r.deployable.unwrap_or(true),
        deploy_ticks: r.deploy_time.map(secs_to_ticks).unwrap_or(TICKS_PER_SECOND as i32),
        shield: r.shield.unwrap_or(0) as i32,
        spawn_count: r.spawn.as_ref().and_then(|s| s.count).unwrap_or(1) as i32,
        spawn_interval_ticks: r
            .spawn
            .as_ref()
            .and_then(|s| s.interval)
            .map(secs_to_ticks)
            .unwrap_or(0),
        spawn_name: r.spawn.as_ref().map(|s| s.card.clone()),
        spawn_id: None,
        death_count: r.death_spawn.as_ref().and_then(|s| s.count).unwrap_or(1) as i32,
        death_name: r.death_spawn.as_ref().map(|s| s.card.clone()),
        death_id: None,
        effect: r.effect.as_deref().map(SpellEffect::parse).unwrap_or(SpellEffect::None),
        effect_ticks: r.effect_duration.map(secs_to_ticks).unwrap_or(0),
        effect_interval: r.effect_interval.map(secs_to_ticks).unwrap_or(TICKS_PER_SECOND as i32),
        effect_power: r.effect_power.unwrap_or(0) as i32,
    }
}

/// Resolved library: deployable cards plus the two tower defs, in one table.
#[derive(Clone)]
pub struct Library {
    pub defs: Vec<CardDef>,          // index = def id
    pub index: std::collections::HashMap<String, usize>,
    pub deployable: Vec<usize>,      // def ids a player may put in a deck
    pub princess_tower: usize,
    pub king_tower: usize,
}

impl Library {
    pub fn from_json(s: &str) -> Result<Library, String> {
        let raw: RawFile = serde_json::from_str(s).map_err(|e| e.to_string())?;
        let mut defs = Vec::new();
        let mut index = std::collections::HashMap::new();
        let mut deployable = Vec::new();
        for (name, rc) in &raw.cards {
            let id = defs.len();
            let def = resolve(name, rc, None);
            let is_deployable = def.deployable;
            defs.push(def);
            index.insert(name.clone(), id);
            if is_deployable {
                deployable.push(id);
            }
        }
        let mut princess_tower = usize::MAX;
        let mut king_tower = usize::MAX;
        for (name, rc) in &raw.towers {
            let id = defs.len();
            let mut d = resolve(name, rc, Some(CardType::Tower));
            d.crown_value = if name == "king_tower" { 3 } else { 1 };
            if name == "king_tower" {
                king_tower = id;
            } else if name == "princess_tower" {
                princess_tower = id;
            }
            defs.push(d);
            index.insert(name.clone(), id);
        }
        if princess_tower == usize::MAX || king_tower == usize::MAX {
            return Err("cards.json towers must define princess_tower and king_tower".into());
        }
        // Link inter-card references now that every name is in the index.
        for i in 0..defs.len() {
            if let Some(n) = defs[i].spawn_name.clone() {
                defs[i].spawn_id = index.get(&n).copied();
            }
            if let Some(n) = defs[i].death_name.clone() {
                defs[i].death_id = index.get(&n).copied();
            }
        }
        Ok(Library { defs, index, deployable, princess_tower, king_tower })
    }

    pub fn default_lib() -> Library {
        Library::from_json(DEFAULT_CARDS_JSON).expect("bundled cards.json is valid")
    }

    pub fn id_of(&self, name: &str) -> Option<usize> {
        self.index.get(name).copied()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn bundled_loads() {
        let lib = Library::default_lib();
        assert!(lib.deployable.len() >= 15, "cards {}", lib.deployable.len());
        assert!(lib.id_of("knight").is_some());
        let k = &lib.defs[lib.id_of("knight").unwrap()];
        assert_eq!(k.elixir, 3);
        assert!(k.hit_ticks > 0);
        // king tower resolved with 3 crowns
        assert_eq!(lib.defs[lib.king_tower].crown_value, 3);
    }
}

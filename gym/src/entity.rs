//! Runtime entities: troops, buildings, and towers on the field.
//! Projectiles are tracked separately in the engine (see engine.rs).

use crate::fixed::{Fx, V2};

#[derive(Clone)]
pub struct Entity {
    pub id: u32,
    pub owner: u8,      // 0 or 1
    pub def: usize,     // index into Library.defs
    pub pos: V2,
    pub half: V2,       // box half-extents for towers; zero => circle hitbox (radius from def)
    pub hp: i32,
    pub max_hp: i32,
    pub target: Option<u32>,
    pub cooldown: i32,  // ticks until able to attack again
    pub deploy: i32,    // ticks until active (0 = active)
    pub lifetime: i32,  // ticks remaining (-1 = infinite)
    pub stun: i32,      // ticks of stun/freeze remaining (can't move or attack)
    pub slow: i32,      // ticks slowed (reduced move + attack speed)
    pub rage: i32,      // ticks raged (increased move + attack speed)
    pub shield_hp: i32, // absorbs damage before hp (Dark Prince, Guards, Recruits)
    pub spawn_timer: i32, // ticks until this spawner building produces units
    pub charge_dist: Fx,// distance moved toward target unobstructed (prince)
    pub is_tower: bool,
    pub is_king: bool,
    pub activated: bool,// king tower: starts dormant until activated
    pub engaged: bool,  // in attack range of its target last tick (drives wind-up)
    pub dead: bool,
}

impl Entity {
    #[inline]
    pub fn active(&self) -> bool {
        self.deploy <= 0 && !self.dead
    }
    #[inline]
    pub fn alive(&self) -> bool {
        !self.dead && self.hp > 0
    }
}

/// An in-flight projectile (ranged attack or spell delivery).
#[derive(Clone)]
pub struct Projectile {
    pub owner: u8,
    pub pos: V2,
    pub target: Option<u32>, // homing target; None = fixed-point ground strike
    pub aim: V2,             // fallback impact point if target dies
    pub speed: Fx,           // tiles per tick
    pub damage: i32,
    pub splash: Fx,          // 0 = single target
    pub crown_pct: i32,
    pub stun_ticks: i32,
    pub hits_flying: bool,
    pub hits_ground: bool,
    pub dead: bool,
}

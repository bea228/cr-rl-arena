//! Python-facing environment: `cr_sim._core.Battle`.
//!
//! Minimal, dependency-free surface an RL agent drives:
//!   b = Battle()
//!   b.reset(seed, deck0, deck1)
//!   out = b.step(action0, action1)     # action = (slot, x, y) or None
//!   obs = b.observe(player)            # dict of scalars + entity lists
//!
//! Observations are returned as plain Python tuples/lists/dicts so there are no
//! third-party deps; the optional numpy/gym adapters live in Python.
//!
//! Throughput primitives (pure speed -- they decide nothing about rewards,
//! observations or action encodings):
//!   bb   = BatchBattle(n, threads=None)    # n independent battles, stepped in parallel
//!   outs = bb.step(actions0, actions1)     # one tick each, all cores, GIL released
//!   bb[i]                                  # an ordinary `Battle` (every accessor works)

use crate::arena::{HEIGHT, OVERTIME_TICKS, REGULATION_TICKS, TICKS_PER_SECOND, WIDTH};
use crate::cards::Library;
use crate::engine::{Action, Engine, Outcome};
use crate::fixed::{Fx, V2};
use pyo3::prelude::*;
use pyo3::types::PyDict;
use rayon::prelude::*;

#[pyclass]
pub struct Battle {
    eng: Engine,
}

fn outcome_code(o: Outcome) -> i32 {
    match o {
        Outcome::Running => 0,
        Outcome::Win0 => 1,
        Outcome::Win1 => 2,
        Outcome::Draw => 3,
    }
}

#[pymethods]
impl Battle {
    #[new]
    #[pyo3(signature = (cards_json=None))]
    fn new(cards_json: Option<String>) -> PyResult<Battle> {
        let lib = match cards_json {
            Some(path) => {
                let s = std::fs::read_to_string(&path)
                    .map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))?;
                Library::from_json(&s)
                    .map_err(pyo3::exceptions::PyValueError::new_err)?
            }
            None => Library::default_lib(),
        };
        Ok(Battle { eng: Engine::new(lib) })
    }

    /// Reset to tick 0. Decks are lists of card names (length 8); None = default.
    #[pyo3(signature = (seed=0, deck0=None, deck1=None))]
    fn reset(
        &mut self,
        seed: u64,
        deck0: Option<Vec<String>>,
        deck1: Option<Vec<String>>,
    ) -> PyResult<()> {
        let decks = match (deck0, deck1) {
            (Some(a), Some(b)) => {
                let da = self.resolve_deck(&a)?;
                let db = self.resolve_deck(&b)?;
                Some([da, db])
            }
            (None, None) => None,
            _ => return Err(pyo3::exceptions::PyValueError::new_err(
                "provide both decks or neither",
            )),
        };
        self.eng.reset(seed, decks);
        Ok(())
    }

    /// Check that `deck` is legal without starting a match: exactly 8 different,
    /// playable card names with at most one champion. Raises ValueError saying
    /// what is wrong; returns None if the deck is fine. `reset` applies the same
    /// rules, so the competition server calls this when an agent joins.
    fn check_deck(&self, deck: Vec<String>) -> PyResult<()> {
        self.resolve_deck(&deck).map(|_| ())
    }

    /// Advance one tick. Each action is (hand_slot, x, y) or None for no-op.
    /// Returns the outcome code: 0 running, 1 player0 win, 2 player1 win, 3 draw.
    #[pyo3(signature = (action0=None, action1=None))]
    fn step(
        &mut self,
        action0: Option<(i32, f64, f64)>,
        action1: Option<(i32, f64, f64)>,
    ) -> i32 {
        let a0 = to_action(action0);
        let a1 = to_action(action1);
        outcome_code(self.eng.step([a0, a1]))
    }

    /// [champ] Activate `player`'s on-field champion ability. Returns True if it
    /// fired (champion deployed + active, off cooldown, enough elixir, and — for
    /// the dash — an enemy in reach); False otherwise, spending nothing.
    ///
    /// Equivalent to passing the reserved slot `cr_sim.ABILITY_SLOT` in `step`,
    /// e.g. `b.step((cr_sim.ABILITY_SLOT, 0.0, 0.0), None)`.
    fn activate_ability(&mut self, player: usize) -> bool {
        self.eng.activate_ability(player.min(1))
    }

    /// [champ] (on_field, cooldown_ticks_remaining, ability_cost) for a player's
    /// champion; (False, 0, 0) when none is deployed.
    fn champion_status(&self, player: usize) -> (bool, i32, i32) {
        self.eng.champion_status(player.min(1))
    }

    /// Return a read-only [`PlayerView`] exposing **only** what `player` may
    /// legally see under Clash Royale's hidden-information model.
    ///
    /// A competition harness should hand each agent `battle.view(me)` — never the
    /// raw `Battle`. The view lets the agent read the full board, its own
    /// hand/next-card/elixir/crowns, and the opponent's crowns + tower state, but
    /// has no accessor at all for the opponent's hand, next card, or elixir, so an
    /// agent structurally cannot peek at the opponent's hidden information. The
    /// raw `Battle` getters stay available for the trusted harness, debugging, and
    /// replay. The view is a live window: re-reading it after `step(...)` reflects
    /// the new state, and it never mutates the simulation.
    fn view(slf: &Bound<'_, Self>, player: usize) -> PlayerView {
        PlayerView {
            battle: slf.clone().unbind(),
            player: player.min(1),
        }
    }

    // ---- scalar accessors ----

    #[getter]
    fn tick(&self) -> i64 {
        self.eng.tick
    }
    fn elapsed_seconds(&self) -> f64 {
        self.eng.tick as f64 / TICKS_PER_SECOND as f64
    }
    fn outcome(&self) -> i32 {
        outcome_code(self.eng.outcome)
    }
    fn done(&self) -> bool {
        self.eng.outcome != Outcome::Running
    }
    fn phase(&self) -> &'static str {
        let t = self.eng.tick;
        if t >= REGULATION_TICKS {
            "overtime"
        } else if t >= crate::arena::DOUBLE_ELIXIR_AT {
            "double"
        } else {
            "single"
        }
    }
    fn elixir(&self, player: usize) -> f64 {
        self.eng.elixir[player.min(1)].to_f64()
    }
    fn crowns(&self) -> (i32, i32) {
        (self.eng.crowns[0], self.eng.crowns[1])
    }

    fn hand(&self, player: usize) -> Vec<String> {
        self.eng
            .hand(player.min(1))
            .iter()
            .map(|&d| self.eng.lib.defs[d].name.clone())
            .collect()
    }
    fn hand_cost(&self, player: usize) -> Vec<i32> {
        self.eng
            .hand(player.min(1))
            .iter()
            .map(|&d| self.eng.lib.defs[d].elixir)
            .collect()
    }
    fn next_card(&self, player: usize) -> String {
        self.eng.lib.defs[self.eng.next_card(player.min(1))].name.clone()
    }

    /// Per-hand-slot affordability mask (true = enough elixir to play it now).
    fn legal(&self, player: usize) -> Vec<bool> {
        let p = player.min(1);
        let e = self.eng.elixir[p].to_f64();
        self.eng
            .hand(p)
            .iter()
            .map(|&d| self.eng.lib.defs[d].elixir as f64 <= e)
            .collect()
    }

    /// All field entities as tuples:
    /// (owner, card, x, y, hp, max_hp, flying, is_tower, deploying, id, half_w, half_h)
    /// half_w/half_h are box half-extents in tiles (0 => circle; radius via card).
    fn entities(
        &self,
    ) -> Vec<(u8, String, f64, f64, i32, i32, bool, bool, bool, u32, f64, f64)> {
        self.eng
            .entities
            .iter()
            .map(|e| {
                let d = &self.eng.lib.defs[e.def];
                (
                    e.owner,
                    d.name.clone(),
                    e.pos.x.to_f64(),
                    e.pos.y.to_f64(),
                    e.hp,
                    e.max_hp,
                    d.flying,
                    e.is_tower,
                    e.deploy > 0,
                    e.id,
                    e.half.x.to_f64(),
                    e.half.y.to_f64(),
                )
            })
            .collect()
    }

    /// Projectiles as (owner, x, y) for the viewer.
    fn projectiles(&self) -> Vec<(u8, f64, f64)> {
        self.eng
            .projectiles
            .iter()
            .map(|p| (p.owner, p.pos.x.to_f64(), p.pos.y.to_f64()))
            .collect()
    }

    /// Structured per-player observation dict (scalars + entity lists).
    fn observe<'py>(&self, py: Python<'py>, player: usize) -> PyResult<Bound<'py, PyDict>> {
        let p = player.min(1);
        let d = PyDict::new(py);
        d.set_item("tick", self.eng.tick)?;
        d.set_item("player", p)?;
        d.set_item("elixir", self.elixir(p))?;
        d.set_item("phase", self.phase())?;
        d.set_item("crowns", self.crowns())?;
        d.set_item("hand", self.hand(p))?;
        d.set_item("hand_cost", self.hand_cost(p))?;
        d.set_item("next_card", self.next_card(p))?;
        d.set_item("legal", self.legal(p))?;
        d.set_item("entities", self.entities())?;
        d.set_item("done", self.done())?;
        d.set_item("outcome", self.outcome())?;
        Ok(d)
    }

    /// Card names a deck may draw from.
    fn deployable_cards(&self) -> Vec<String> {
        self.eng
            .lib
            .deployable
            .iter()
            .map(|&d| self.eng.lib.defs[d].name.clone())
            .collect()
    }
}

impl Battle {
    fn resolve_deck(&self, names: &[String]) -> PyResult<Vec<usize>> {
        if names.len() != 8 {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "deck must have 8 cards, got {}",
                names.len()
            )));
        }
        let mut out = Vec::with_capacity(8);
        for n in names {
            match self.eng.lib.id_of(n) {
                Some(id) => out.push(id),
                None => {
                    return Err(pyo3::exceptions::PyValueError::new_err(format!(
                        "unknown card '{n}'"
                    )))
                }
            }
        }
        // Deck-legal cards only: towers, evolved variants and summon-only units
        // (golemite, hut spawns, ...) exist in the library but are not cards.
        for (n, &id) in names.iter().zip(out.iter()) {
            if !self.eng.lib.deployable.contains(&id) {
                return Err(pyo3::exceptions::PyValueError::new_err(format!(
                    "'{n}' is not a playable card and cannot be put in a deck"
                )));
            }
        }
        // No duplicates: a deck is 8 different cards.
        for (i, n) in names.iter().enumerate() {
            if out[..i].contains(&out[i]) {
                return Err(pyo3::exceptions::PyValueError::new_err(format!(
                    "duplicate card '{n}': a deck must be 8 different cards"
                )));
            }
        }
        // [champ] enforce the one-champion-per-deck rule
        let champs = out.iter().filter(|&&id| self.eng.lib.defs[id].is_champion).count();
        if champs > 1 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "a deck may contain at most one champion",
            ));
        }
        Ok(out)
    }
}

type BatchActions = Option<Vec<Option<(i32, f64, f64)>>>;

/// `n` independent battles stepped together in one native call, across threads,
/// with the GIL released.
///
/// This is purely a throughput tool. Each element is an ordinary [`Battle`]
/// (`bb[i]`), so every accessor -- `observe`, `view`, `entities`, `legal`,
/// `reset`, ... -- works per battle exactly as it does standalone, and a battle
/// stepped here plays out bit-identically to the same battle stepped alone.
/// Nothing about rewards, observation encoding or when to reset is decided here.
///
///   bb = BatchBattle(256)                    # threads=None: one per CPU core
///   bb.reset(range(256))                     # one seed per battle
///   outs = bb.step(acts0, acts1)             # lists of (slot, x, y) | None
///   for i, o in enumerate(outs):
///       if o: bb[i].reset(new_seed)          # finished battles are left as-is
#[pyclass]
pub struct BatchBattle {
    battles: Vec<Py<Battle>>,
    pool: Option<rayon::ThreadPool>, // None => step sequentially on the caller's thread
    threads: usize,
}

impl BatchBattle {
    fn batch_actions(&self, a: BatchActions, which: &str) -> PyResult<Vec<Action>> {
        match a {
            None => Ok(vec![Action::noop(); self.battles.len()]),
            Some(v) => {
                if v.len() != self.battles.len() {
                    return Err(pyo3::exceptions::PyValueError::new_err(format!(
                        "{which} has {} entries, expected {} (one per battle)",
                        v.len(),
                        self.battles.len()
                    )));
                }
                Ok(v.into_iter().map(to_action).collect())
            }
        }
    }
}

#[pymethods]
impl BatchBattle {
    /// `n` battles. `threads`: None/0 = one worker per CPU core, 1 = no worker
    /// threads (step on the calling thread), N = exactly N workers.
    #[new]
    #[pyo3(signature = (n, threads=None, cards_json=None))]
    fn new(py: Python<'_>, n: usize, threads: Option<usize>, cards_json: Option<String>) -> PyResult<BatchBattle> {
        if n == 0 {
            return Err(pyo3::exceptions::PyValueError::new_err("n must be >= 1"));
        }
        let mut battles = Vec::with_capacity(n);
        for _ in 0..n {
            battles.push(Py::new(py, Battle::new(cards_json.clone())?)?);
        }
        let want = threads.unwrap_or(0);
        let pool = if want == 1 {
            None
        } else {
            Some(
                rayon::ThreadPoolBuilder::new()
                    .num_threads(want) // 0 => rayon picks the CPU count
                    .build()
                    .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?,
            )
        };
        let threads = pool.as_ref().map_or(1, |p| p.current_num_threads());
        Ok(BatchBattle { battles, pool, threads })
    }

    /// Reset every battle to tick 0. `seeds` has one seed per battle; the decks
    /// (both or neither, as in `Battle.reset`) are shared by all of them. To reset
    /// a single battle -- e.g. one that just finished -- call `bb[i].reset(...)`.
    #[pyo3(signature = (seeds, deck0=None, deck1=None))]
    fn reset(
        &self,
        py: Python<'_>,
        seeds: Vec<u64>,
        deck0: Option<Vec<String>>,
        deck1: Option<Vec<String>>,
    ) -> PyResult<()> {
        if seeds.len() != self.battles.len() {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "seeds has {} entries, expected {} (one per battle)",
                seeds.len(),
                self.battles.len()
            )));
        }
        for (b, seed) in self.battles.iter().zip(seeds) {
            b.bind(py).try_borrow_mut()?.reset(seed, deck0.clone(), deck1.clone())?;
        }
        Ok(())
    }

    /// Advance every battle by one tick, exactly as `Battle.step` would: battle
    /// `i` gets `actions0[i]` / `actions1[i]`. Either list may be None (all
    /// no-ops). Battles that have already finished are left untouched. Returns one
    /// outcome code per battle. Runs on the worker threads with the GIL released;
    /// results do not depend on the thread count.
    #[pyo3(signature = (actions0=None, actions1=None))]
    fn step(
        &self,
        py: Python<'_>,
        actions0: BatchActions,
        actions1: BatchActions,
    ) -> PyResult<Vec<i32>> {
        let a0 = self.batch_actions(actions0, "actions0")?;
        let a1 = self.batch_actions(actions1, "actions1")?;
        // Exclusive borrows of every battle for the duration of the call; the
        // engines are then plain `&mut` data, safe to hand to other threads.
        let mut guards: Vec<PyRefMut<'_, Battle>> = Vec::with_capacity(self.battles.len());
        for b in &self.battles {
            guards.push(b.bind(py).try_borrow_mut()?);
        }
        let mut work: Vec<(&mut Engine, Action, Action)> = guards
            .iter_mut()
            .zip(a0.into_iter().zip(a1))
            .map(|(g, (x, y))| (&mut g.eng, x, y))
            .collect();
        let pool = self.pool.as_ref();
        let outs: Vec<Outcome> = py.allow_threads(|| match pool {
            Some(p) => p.install(|| {
                work.par_iter_mut().map(|(e, x, y)| e.step([*x, *y])).collect()
            }),
            None => work.iter_mut().map(|(e, x, y)| e.step([*x, *y])).collect(),
        });
        Ok(outs.into_iter().map(outcome_code).collect())
    }

    /// Number of worker threads `step` uses (1 = the calling thread only).
    #[getter]
    fn threads(&self) -> usize {
        self.threads
    }

    fn __len__(&self) -> usize {
        self.battles.len()
    }

    /// Battle `i` as an ordinary `Battle` (negative indices count from the end).
    fn __getitem__(&self, py: Python<'_>, i: isize) -> PyResult<Py<Battle>> {
        let n = self.battles.len() as isize;
        let j = if i < 0 { i + n } else { i };
        if j < 0 || j >= n {
            return Err(pyo3::exceptions::PyIndexError::new_err("battle index out of range"));
        }
        Ok(self.battles[j as usize].clone_ref(py))
    }

    /// All battles, in order, as a list of `Battle` objects.
    fn battles(&self, py: Python<'_>) -> Vec<Py<Battle>> {
        self.battles.iter().map(|b| b.clone_ref(py)).collect()
    }

    // ---- bulk raw state (one call instead of n) ----

    /// Outcome code per battle (0 running, 1 player0 win, 2 player1 win, 3 draw).
    fn outcomes(&self, py: Python<'_>) -> Vec<i32> {
        self.battles.iter().map(|b| b.bind(py).borrow().outcome()).collect()
    }
    /// Whether each battle has finished.
    fn done(&self, py: Python<'_>) -> Vec<bool> {
        self.battles.iter().map(|b| b.bind(py).borrow().done()).collect()
    }
    /// Current tick of each battle.
    fn ticks(&self, py: Python<'_>) -> Vec<i64> {
        self.battles.iter().map(|b| b.bind(py).borrow().tick()).collect()
    }
    /// (player0_crowns, player1_crowns) per battle.
    fn crowns(&self, py: Python<'_>) -> Vec<(i32, i32)> {
        self.battles.iter().map(|b| b.bind(py).borrow().crowns()).collect()
    }
    /// `Battle.entities()` for every battle: a list (per battle) of entity tuples.
    fn entities(
        &self,
        py: Python<'_>,
    ) -> Vec<Vec<(u8, String, f64, f64, i32, i32, bool, bool, bool, u32, f64, f64)>> {
        self.battles.iter().map(|b| b.bind(py).borrow().entities()).collect()
    }

    fn __repr__(&self) -> String {
        format!("BatchBattle(n={}, threads={})", self.battles.len(), self.threads)
    }
}

/// Clamp a player index to a valid seat (0 or 1).
#[inline]
fn view_player(player: usize) -> usize {
    player.min(1)
}

/// The opposing seat for a (clamped) player index.
#[inline]
fn view_opponent(player: usize) -> usize {
    1 - player.min(1)
}

/// A read-only, per-player window onto a [`Battle`] that enforces Clash Royale's
/// hidden-information model at the API boundary.
///
/// What this player **may** see (and this view exposes):
///   * the full board — every deployed unit, tower, and projectile of *both*
///     sides is public (`entities`, `towers`, `projectiles`),
///   * its own hand, next card, elixir, crown count, and champion status,
///   * the opponent's crown count and tower HP/state (via the public board),
///   * the fog-of-war `observe()` observation and the `legal()` action mask.
///
/// What it **cannot** see: there is deliberately no accessor for the opponent's
/// hand, next card, or elixir. The per-player getters (`elixir`, `hand`,
/// `hand_cost`, `next_card`, `legal`, `champion_status`) take **no player
/// argument** — they always report this view's own seat — so an agent handed
/// `battle.view(me)` structurally cannot read the opponent's private state, and
/// the view exposes no mutators (`step`/`reset`/`activate_ability`) either.
///
/// The raw `Battle` getters remain for the trusted harness, debugging, and
/// replay. The view is a live window onto the same `Battle`, so re-reading it
/// after `Battle.step(...)` reflects the new tick; it never mutates the sim.
#[pyclass]
pub struct PlayerView {
    battle: Py<Battle>,
    player: usize,
}

#[pymethods]
impl PlayerView {
    /// This view's own seat (0 or 1).
    #[getter]
    fn player(&self) -> usize {
        self.player
    }

    /// The opponent's seat (0 or 1).
    #[getter]
    fn opponent(&self) -> usize {
        view_opponent(self.player)
    }

    // ---- public match state (both players see these) ----

    #[getter]
    fn tick(&self, py: Python<'_>) -> i64 {
        self.battle.borrow(py).tick()
    }
    fn elapsed_seconds(&self, py: Python<'_>) -> f64 {
        self.battle.borrow(py).elapsed_seconds()
    }
    fn phase(&self, py: Python<'_>) -> &'static str {
        self.battle.borrow(py).phase()
    }
    fn outcome(&self, py: Python<'_>) -> i32 {
        self.battle.borrow(py).outcome()
    }
    fn done(&self, py: Python<'_>) -> bool {
        self.battle.borrow(py).done()
    }

    /// (player0_crowns, player1_crowns). The crown score is public to both sides.
    fn crowns(&self, py: Python<'_>) -> (i32, i32) {
        self.battle.borrow(py).crowns()
    }
    /// This player's own crown count.
    fn my_crowns(&self, py: Python<'_>) -> i32 {
        let c = self.battle.borrow(py).crowns();
        if self.player == 0 { c.0 } else { c.1 }
    }
    /// The opponent's crown count (public).
    fn opponent_crowns(&self, py: Python<'_>) -> i32 {
        let c = self.battle.borrow(py).crowns();
        if self.player == 0 { c.1 } else { c.0 }
    }

    /// Full board — every deployed unit and tower of *both* sides (public).
    /// Same tuple shape as `Battle.entities()`.
    fn entities(
        &self,
        py: Python<'_>,
    ) -> Vec<(u8, String, f64, f64, i32, i32, bool, bool, bool, u32, f64, f64)> {
        self.battle.borrow(py).entities()
    }

    /// The tower subset of `entities()` (both sides' crown/king towers are
    /// public, including their HP) — convenience for reading tower state.
    fn towers(
        &self,
        py: Python<'_>,
    ) -> Vec<(u8, String, f64, f64, i32, i32, bool, bool, bool, u32, f64, f64)> {
        self.battle
            .borrow(py)
            .entities()
            .into_iter()
            .filter(|e| e.7) // e.7 == is_tower
            .collect()
    }

    /// Projectiles in flight for both sides (public).
    fn projectiles(&self, py: Python<'_>) -> Vec<(u8, f64, f64)> {
        self.battle.borrow(py).projectiles()
    }

    // ---- private: this player's OWN state only (never the opponent's) ----

    /// This player's own elixir. The opponent's elixir is hidden — there is no
    /// accessor for it, and this getter takes no player argument.
    fn elixir(&self, py: Python<'_>) -> f64 {
        self.battle.borrow(py).elixir(self.player)
    }
    /// This player's own hand (4 card names).
    fn hand(&self, py: Python<'_>) -> Vec<String> {
        self.battle.borrow(py).hand(self.player)
    }
    /// Elixir cost of each card in this player's own hand.
    fn hand_cost(&self, py: Python<'_>) -> Vec<i32> {
        self.battle.borrow(py).hand_cost(self.player)
    }
    /// This player's own next card.
    fn next_card(&self, py: Python<'_>) -> String {
        self.battle.borrow(py).next_card(self.player)
    }
    /// This player's own champion status: (on_field, cooldown_ticks, cost). The
    /// opponent's ability cooldown is hidden.
    fn champion_status(&self, py: Python<'_>) -> (bool, i32, i32) {
        self.battle.borrow(py).champion_status(self.player)
    }
    /// Per-hand-slot affordability mask for this player.
    fn legal(&self, py: Python<'_>) -> Vec<bool> {
        self.battle.borrow(py).legal(self.player)
    }

    /// The fog-of-war observation dict for this player (identical to
    /// `Battle.observe(player)`): own hand/next/elixir/crowns + the full board,
    /// with the opponent's private state omitted.
    fn observe<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        self.battle.borrow(py).observe(py, self.player)
    }

    fn __repr__(&self) -> String {
        format!("PlayerView(player={})", self.player)
    }
}

#[cfg(test)]
mod view_tests {
    use super::{view_opponent, view_player};

    #[test]
    fn seat_clamps_to_valid_range() {
        assert_eq!(view_player(0), 0);
        assert_eq!(view_player(1), 1);
        // out-of-range seats clamp to seat 1, matching the Battle getters.
        assert_eq!(view_player(2), 1);
        assert_eq!(view_player(usize::MAX), 1);
    }

    #[test]
    fn opponent_is_the_other_seat() {
        assert_eq!(view_opponent(0), 1);
        assert_eq!(view_opponent(1), 0);
        assert_eq!(view_opponent(2), 0); // clamped first
        // a view's own seat and its opponent are always distinct
        for p in 0..2usize {
            assert_ne!(view_player(p), view_opponent(p));
        }
    }
}

fn to_action(a: Option<(i32, f64, f64)>) -> Action {
    match a {
        None => Action::noop(),
        Some((slot, x, y)) => Action {
            slot,
            x: Fx::from_f64(x),
            y: Fx::from_f64(y),
        },
    }
}

/// Register the arena dimensions on the module so Python/viewer can read them.
pub fn register_constants(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("WIDTH", WIDTH.to_f64())?;
    m.add("HEIGHT", HEIGHT.to_f64())?;
    m.add("TICKS_PER_SECOND", TICKS_PER_SECOND)?;
    m.add("REGULATION_TICKS", REGULATION_TICKS)?;
    m.add("OVERTIME_TICKS", OVERTIME_TICKS)?;
    m.add("ABILITY_SLOT", crate::engine::ABILITY_SLOT)?; // [champ]
    m.add("RIVER_Y", crate::arena::RIVER_Y.to_f64())?;
    m.add("RIVER_HALF", crate::arena::RIVER_HALF.to_f64())?;
    m.add("BRIDGE_HALF", crate::arena::BRIDGE_HALF.to_f64())?;
    m.add(
        "BRIDGES",
        vec![crate::arena::BRIDGE_L.to_f64(), crate::arena::BRIDGE_R.to_f64()],
    )?;
    let _ = V2::default();
    Ok(())
}

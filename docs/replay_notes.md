# Replay notes: scripted v0 vs random on default.json (J2 / C12)

Source: `python -m evals.gauntlet --a scripted --b random --games 400 --seed 0 --a-deck bot/decks/default.json`
(v0, 92.2%), the 10 worst losses, read with `scripts/show_replay.py`. Read in full: g2, g82, g213, g110
(plus g13 from the post-fix run). Action list + tower events only: g53, g42, g162, g93, g118, g310.
Engine check: `gym/src/engine.rs` `[inv]` — an invisible unit (Royal Ghost) can't be *acquired* by troops
or towers except for `REVEAL_TICKS` = 15 ticks after each of its hits (hit speed 1.8 s = 36 ticks);
a unit already locked on keeps it, and splash still hits it.

## Losses (v0, seed 0)

| game | seat | first tower lost | what went wrong | rule that would have prevented it |
|---|---|---|---|---|
| g2 | 0 | L t1312, royal_ghost (+ronin) | Ghost sits on L tower for 250 ticks; our valkyrie/knight in the lane count as cover so defend is silent, then melee picks walk past the hidden ghost. Elixir ~0 all game from defending ice spirit / skeletons. | ranged defender vs invisible threat; melee hp not counted as cover (kept fix) |
| g82 | 0 | L t902, ronin + ghost | Ghost + ronin on L tower while elixir is spent on R; musketeer used on the R ghost dies; L defended late with archers at 0.6 elixir. | ranged vs ghost (kept fix); elixir starvation |
| g42 | 0 | L t1649, royal_ghost + skeleton_barrel | Defend fires every ~100–200 ticks in L with cheapest card; ghost keeps hitting. | ranged vs ghost (kept fix) |
| g110 | 0 | R t2926, executioner + skeleton_dragons | Splash air+ground push at y 12–13; we have 0.7 elixir after minions + archers + musketeer chain; musketeer alone dies. | elixir starvation; no rule (would need holding a wizard for it) |
| g213 | 1 | L t892, miner + royal_ghost + spear_goblins | 9 spear goblins + ghost + miner; archers placed alone at 0.6 elixir; later the ghost kills the king while our minipekka at y 8–10 never hits it. | ranged vs ghost (kept fix); splash vs swarm needs elixir |
| g53 | 1 | L t1506, battle_ram + royal_ghost + spear_goblins | We took their R tower, then defended L with cheapest cards; king fell 8 ticks after R. | ranged vs ghost (kept fix) |
| g162 | 0 | R t1600, royal_ghost | Only a ghost in R; minipekka/minions/musketeer rotate in; ghost survives. | ranged vs ghost (kept fix) |
| g93 | 1 | R t1651, battle_ram + royal_ghost | Two attack plays at bridge_L early, then archers+minions double-drop vs ram; ghost takes R. | ranged vs ghost (kept fix) |
| g118 | 0 | R t1940, mega_minion + royal_ghost | Defend fires steadily, but melee picks (knight, valkyrie, minipekka) vs the ghost. | ranged vs ghost (kept fix) |
| g310 | 0 | R t2398, executioner + skeleton_dragons + guards + evo_knight | Splash/air push; we had just spent on an attack valkyrie; elixir 0–2 while the tower drops. | elixir starvation; no single rule |

## Most common causes (most common first)

1. **Royal Ghost not answered (8 of 10).** Melee defenders can't acquire it except in a 15-tick reveal
   window, and v0 counted their hp as cover. Royal-ghost decks (3 of 40) were 54% of all losses at
   1,000 games × 2 seeds (81 of 151). Confirmed by a placement experiment: archers / musketeer at def_
   kill a lone ghost; knight, valkyrie, minipekka, giant, minions do not.
2. **Elixir starvation (all 10, decisive in ~4).** Instrumented over 1,000 games: while one of our towers
   was taking damage, in losses the bot had no affordable card 52% of the time (wins: 20%); defend spends
   20 elixir / 1,000 ticks in losses vs 13.5 in wins. Defend reacts to small threats (ice spirit,
   guards, skeletons) and double-drops.
3. **Defense counted but not engaging ("covered", 29% of tower-damage decisions in losses).** Our units
   in the lane count as cover while walking up past the threats.
4. **King falls right after a princess tower** (g53, g162, g118): defenders still go to def_ of the dead
   lane instead of near the king. Not attempted.

Suspicions from the brief: *flying threat with no hits_air card* → rejected as a main cause (4% of
tower-damage decisions in losses). *Cheapest-card vs big push* → C9(a) scoring tried, made it worse.
*Late threat / buildings near the bridge* → goblin_hut decks show up after the ghost fix (g13, g575,
g453, g855) but the losses there are starvation, not detection. *No counter-push / overtime* → none of the 10
losses reached overtime (longest 3,558 ticks); not the cause here. *Wrong lane* → not seen; defend always picked the right lane.

## Attempts

Measured with `--a scripted --b random --a-deck bot/decks/default.json --games 1000`, seeds 0 and 50000.
Keep rule: mean up ≥ 0.7 points and neither seed drops.

| # | change | seed 0 | seed 50000 | mean | result |
|---|---|---|---|---|---|
| base | v0 | 91.8% | 93.1% | 92.45 | — |
| 1 | defend: vs invisible threat, prefer ranged (range ≥ 5) cards | 92.3% | 93.2% | 92.75 | reverted (+0.3) |
| 2 | 1 + vs invisible threat only ranged units count as defense | 93.4% | 94.2% | 93.80 | **kept** (+1.35); rotating 66.0% (was 66.2%), vs v0 72.5% |
| 3 | 2 + ignore lane threat < 300 hp (MIN_THREAT) | 93.1% | 92.3% | 92.70 | reverted (both drop) |
| 4 | 2 + C9(a) defender scoring (dps/elixir, ×2 splash vs 3+) | 91.8% | 92.8% | 92.30 | reverted (both drop) |
| 5 | 2 + vs ghost with no affordable ranged card but one in hand: wait | 92.7% | 94.1% | 93.40 | reverted (both drop) |
| 6 | 2 + defense ignores our units > 2 tiles past the deepest threat | 93.1% | 94.5% | 93.80 | reverted (+0.0, seed 0 drops) |

Stopped after 6 attempts. Worst opposing deck (#3, ghost deck): 16% / 20% → 48% / 44%.
Next ideas (not tried): defend at king_pull after a princess tower falls; a per-lane cooldown so
defend doesn't double-drop while the first card is still deploying.

## Deck choice (C13 + B16 + C14)

Candidates: `evals/candidate_decks.json` (10 decks: the default baseline, A–E hand-picked, F–I new).
All of A–E already met C13 with `cards.role`/`elixir` (air and splash counted on troops only), so no
swaps. The baseline breaks C13 (no spell, avg elixir 3.875). F giant wizard = default with
musketeer/minipekka → fireball/zap; G ram rider and H battle ram = deck A around another win condition;
I hog wizard. No held-out cards, mirror, champions or placement buildings.

`python -m evals.deck_rr`: scripted bot on the candidate deck vs random, scripted and scripted_v0, each
opponent rotating through the 40 gauntlet decks. Score = mean − population std across opponents.
Wall time 181 s on 16 cores (screen 76 s, final 105 s).

Phase 1 screen: 300 games per opponent, seed 0

| deck | avg elx | vs random | vs scripted | vs v0 | mean | lowest | score | ignored |
|---|---|---|---|---|---|---|---|---|
| B hog cycle | 3.12 | 87.7% | 72.7% | 72.7% | 77.7% | 72.7% | 0.706 | 0.00% |
| default | 3.88 | 92.3% | 72.0% | 72.0% | 78.8% | 72.0% | 0.692 | 0.00% |
| A default + spells | 3.50 | 88.0% | 70.0% | 70.0% | 76.0% | 70.0% | 0.675 | 0.00% |
| D royal giant | 3.62 | 86.7% | 67.0% | 67.0% | 73.6% | 67.0% | 0.643 | 0.00% |
| I hog wizard | 3.50 | 83.3% | 64.7% | 64.7% | 70.9% | 64.7% | 0.621 | 0.00% |
| E balloon | 3.62 | 81.3% | 63.7% | 63.7% | 69.6% | 63.7% | 0.612 | 0.00% |
| F giant wizard | 3.62 | 82.3% | 62.7% | 62.7% | 69.2% | 62.7% | 0.600 | 0.00% |
| H battle ram | 3.62 | 80.3% | 59.3% | 59.3% | 66.3% | 59.3% | 0.564 | 0.00% |
| G ram rider | 3.50 | 84.3% | 58.3% | 58.3% | 67.0% | 58.3% | 0.547 | 0.00% |
| C giant beatdown | 3.75 | 78.3% | 50.0% | 50.0% | 59.4% | 50.0% | 0.461 | 0.00% |

Phase 2 final: top 3 + baseline, 1000 games per opponent, seed 50000

| deck | avg elx | vs random | vs scripted | vs v0 | mean | lowest | score | ignored |
|---|---|---|---|---|---|---|---|---|
| default | 3.88 | 94.2% | 74.0% | 74.0% | 80.7% | 74.0% | 0.712 | 0.00% |
| B hog cycle | 3.12 | 88.6% | 72.3% | 72.3% | 77.7% | 72.3% | 0.700 | 0.00% |
| A default + spells | 3.50 | 86.8% | 63.4% | 63.4% | 71.2% | 63.4% | 0.602 | 0.00% |
| D royal giant | 3.62 | 86.2% | 62.4% | 62.4% | 70.3% | 62.4% | 0.591 | 0.00% |

**Choice: keep the default deck** (`bot/decks/tournament.json` = `default.json`). It has the highest
final score (0.712 vs B hog cycle 0.700) and the best win rate vs random (94.2%); B led the screen but
lost 5.6 points vs random in the final. Check: `evals.gauntlet --a scripted --b random --games 1000
--a-deck bot/decks/tournament.json` → 93.4% (seed 0, same as default.json), ignored 0.00%.

Caveats:
- vs scripted and vs v0 are identical in every row. Not a bug: the two bots differ only in the
  royal_ghost defense rule, and no candidate deck has royal_ghost, so they play the same games. In
  effect there are two opponents and the mirror is weighted twice; scoring with only random and
  scripted still ranks default (0.740) above B (0.723).
- Adding spells (A vs default) cost 5–10 points. Likely (not measured) because the spell rule fires rarely and a spell in hand is a
  card that can't defend, which makes the elixir-starvation losses worse. Worth revisiting after the
  defense fixes, and once the learned bot (bc_v1 / snapshots in the original B16) exists to test against.

# cr-rl-arena — Progress log

Live version of `docs/IMPLEMENTATION.md` section 9 (the build order). Where they differ, this file is newer.
Last updated Fri Nov 20, 13:05 from `origin/main` (HEAD `e966b6c`).

## Ownership change (Thu Nov 19, 18:18)
Chris pushed nothing between Mon 20:17 and Thu 18:18, so **Ben wrote Chris's track himself** from impl-1 on.
Ben now owns every file under `bot/`, `evals/`, `train/`, `scripts/`, `models/`, `docs/replay_notes.md`
and `run_bot.ps1`. If Chris picks up a file again, agree on it first so nobody writes the same file twice.

## Hardware
- Ben's laptop: Windows, 16 logical cores, no CUDA. Runs the live bot, the gauntlets and BC training.
- Lab machine (via SSH): Linux, 16 cores, 64 GB RAM, RTX 3060. Long PPO runs in `tmux`.
  Setup: `bash install.sh` (not `--torch`, which installs the CPU build on Linux), then
  `./.venv/bin/python -m pip install torch` (CUDA wheel). The lab machine only `git pull`s and runs jobs.

## How impl prompts are run
Prompts are `.md` files; in Claude Code: `Read <path> and do everything it says` (plan mode for large ones).
Each file ends with an `END OF PROMPT` line (a pasted prompt was once cut off, which skipped its push step).
There is no `CLAUDE.md` in the repo; each prompt carries the rules (owned files only, never `git add .`,
ask before `git push`).

## Status

| impl | Commit | When | Numbers / notes |
|---|---|---|---|
| 1 (C1) | `1fb1aa6` | Thu 18:21 | `bot/cards.py`. **FEAT_DIM = 24.** |
| 2 (B1) | `7c1759f` | Mon 21:05 | Policy, DECIDE_EVERY = 10, RandomPolicy. |
| 3 (C2, C3, W2) | `2819049` | Thu 18:28 | `bot/state.py`, `bot/placements.py` (53 actions, mask, mask_reason, legal_indices); RandomPolicy on the mask. |
| 4 (B2–B4) | `03da3e9` | Mon 21:14 | Deck sampler (40 gauntlet / 12 held-out cards / 40 held-out decks), match runner, replays. ~15 ms per random game. |
| — | `4bf5053` | Mon 21:18 | `.gitattributes`: `*.json text eol=lf`. |
| 5 (C4, C6) | `f1f4de9` | Thu 18:47 | Scripted bot v0, default deck, `scripts/quick_loop.py`. 40 games 92.5%, 400 games 92.8%, random deck 67.5%, 0 ignored moves. |
| 7 (C5) | `ba09108` | Thu 18:53 | `bot/client.py`. Local J1: 5/5 vs random_agent, re-queue and reconnect OK, 0 missed / malformed / ignored, slowest decision 0.8 ms. |
| 6 (B5) | `050bf50` | Thu 19:07 | `evals/gauntlet.py` (+ `--a-deck`), `evals/policies.py`. Random vs random 51.7%, 400 games in 19 s on 15 workers. |
| 8 (J2 + fix pass, C12) | `ccc6797`, `9f511bd` | Thu 19:34 | J2 v0: default deck 92.2% (worst opposing deck 30%), rotating 66.2%, held-out 72.2%. After the fix pass: **default deck 93.8%** (avg of 2 × 1,000 games), worst opposing deck ~46%, rotating 66.0%, vs v0 72.5%. Kept 1 of 6 changes (Royal Ghost: ranged defenders only). `scripts/show_replay.py`, `bot/scripted_v0.py`. Remaining loss cause: running out of elixir on defense. **J2 target (95%) not met.** |
| 18 / 20 / 21 (C13, B16, C14) | `33c8514` | Thu 20:13 | 10 candidate decks, `evals/deck_rr.py`. **Default deck won** (score 0.712, 94.2% vs random); hog cycle 0.700 (88.6%); every spell deck lower. `bot/decks/tournament.json` = default deck. |
| 24 (C16, C17) | `3a4d647` | Thu 20:32 | `run_bot.ps1` (`$Mode` ship / rollback_scripted; restart within 5.2 s), `scripts/soak.ps1`. Soak 200 matches per client with a server restart: PASS, 0 missed, 0 malformed, 0 / 99,863 ignored. ~1.25 s per match. Client takes any policy spec and `--summary-json`. |
| 9 + 10 (C7, C8, B6, B7) | `d35258b` | Thu 22:27 | Trackers: enemy elixir within 1.0 on 49.4% of decision ticks overall, 100% vs decks without spells (enemy spells are invisible). Encoder: globals 16, hand 5×26, units 48×31, enemy_deck 8×25, mask 53. Net 217,494 params, 0.29 ms CPU forward. |
| 12 + 14 (B8–B10) | `8cf51ab` | Thu 22:42 | BC data 1,031,105 decisions / 13,000 games, 57.2% wait. **bc_v1**: held-out non-wait accuracy 70.4% (top-3 94.3%). vs random 92.5% default deck / 66.5% rotating; vs scripted 71.0%. LearnedPolicy runs without guardrails (C10 not built). |
| 15 + 16 (B11–B13) | `b0258d6` | Thu 23:22 | Env with worker processes: 36,010 decisions/s vs random, 29,096 vs scripted (14 workers). PPO ~3,100 decisions/s on CPU. 20-min CPU run **regressed** vs bc_v1 (87.0% / 53.2% / 61.0%), env verified, so PPO drift. |
| B12 stabilisation | `e966b6c` | Fri 00:18 | KL-to-BC penalty (`--kl-bc`), `--ent-coef 0.001`, `--lr 1e-4`, `--value-warmup 60`, `--shape-start 0.5`; `train/eval_snaps.py` gauntlets every snapshot. **Results of the cpu20b test and of any lab run not reported yet.** |
| J3 (go live) | — | — | Not confirmed. Needs the organizers' arena `wss://` address and team name. |

Not built (cut for time, per handoff §11): 11 (scripted v1), 13 (guardrails), 17 / 22 (GPU overnight / specialist), 19 (promotion script), 23 (latency + final gauntlet), 25 (freeze + Saturday run sheet).

## Key numbers to beat
- Scripted (ships unless beaten): **93.8%** vs random on the default deck (2 × 1,000 games), 92.8% at seed 0 / 400.
- bc_v1: 92.5% vs random (default deck), 71.0% vs scripted.
- A learned model replaces scripted on the ladder only if it wins the 400-game gauntlet against scripted.

## Facts found while building

| Found in | Fact | Affects |
|---|---|---|
| impl-2 | Local `Battle.step` needs `(int, float, float)` tuples; policies return lists (server JSON). | all local runners |
| repo | `Battle.step` returns 0 running, 1 seat 0 won, 2 seat 1 won, 3 draw. `b.done()`, `b.crowns()`, `b.last_action(seat)`, `view.result()` are methods; `b.tick` is a property. `cr_sim.ABILITY_SLOT = 100`. | |
| arena.rs | Elixir starts at 5, cap 10; regen per tick 1/56 single, 1/28 double (tick ≥ 2,400), 3/56 overtime (≥ 3,600). | trackers |
| engine.rs | Positions are snapped to tile centres by the simulator; troops and buildings need y < 15; miner and goblin_drill deploy anywhere; mirror is treated as a ground card and never picked. | placements |
| impl-5 | `back_` placements strand ground troops behind our own princess tower, so the bot uses `def_` instead. | scripted, BC |
| server | The server wants a reply every tick (2 s limit, 10 missed in a row = removed), an int `tick` and a JSON-int slot. Re-queue is automatic. Replays at `/replay?id=<match>`. | client |
| impl-8 | Royal Ghost is untargetable except just after it hits, so only ranged defenders help. | scripted |
| decks | Spell decks score lower with the current rules (spells only on clusters of 2+). | deck choice |
| impl-15 | The env is fast (~30k decisions/s); PPO on CPU is limited by the network update (~3.1k/s). | run length |

## Open questions for the organizers
1. Do tournament matches use our join deck, or assign random decks? (Rotating-deck strength is only 66%.)
2. Arena `wss://` address and when rated play starts.
3. Does the team name have to be registered?
4. Which simulator commit does the server run?

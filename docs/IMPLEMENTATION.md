# cr-rl-arena — Implementation Overview

For **Chris** (game logic + live bot) and **Ben** (ML + measurement).
Companion to `docs/Bot_Handoff.pdf` (the spec) and the Winning Bot Playbook (the strategy).
Written Mon Nov 16 2026, evening. Hackathon ends Sat Nov 21.

> **The handoff is the spec. This file is the build order.** Every impl below points at
> handoff step IDs (C1–C17, B1–B19, J1–J4). Where the handoff is ambiguous or two steps
> collide, section 3 of this file says how to resolve it, and this file wins.

---

## 0. How to use this document

**The loop, for each impl:**

1. In the **Clash Royale Bot** Claude project, type: `impl-1` (or whichever number is next on your track).
2. Claude reads this file plus the handoff and replies with **one copy-paste prompt** in a code block.
3. You open Claude Code in the repo (section 5), paste the prompt, and let it work.
4. Claude Code writes the code and tests, runs the checks, and commits/pushes **only your files**.
5. Come back to the project chat and report: `impl-1 done, <any numbers it printed>`, or paste the error if it failed.
   Claude updates the progress log (section 9) so later prompts use your real numbers.

**Which number do I ask for?** Look at your track in section 6. Numbers are global (1, 2, 3, …)
so you can talk about them unambiguously, but Chris and Ben work through their own lists **in parallel**.
Chris starts at **impl-1**, Ben starts at **impl-2**, at the same time.

**Before any impl:** finish section 4 (one-time setup) on your laptop.

---

## 1. Where things stand right now

| Item | State |
|---|---|
| G1 Chris has push access | ✅ done |
| G2 Handoff PDF committed (`docs/Bot_Handoff.pdf`) | ✅ done (repo HEAD `ae6269d`) |
| G3 Chris clone + install | ⬜ section 4 |
| G4 Ben adds PyTorch to his venv | ⬜ section 4 |
| G5 Both git identity | ⬜ section 4 |
| `bot/`, `evals/`, `train/`, `tests/`, `models/` | Do not exist yet. Everything is built from scratch. |
| Organizer questions (handoff §11) | ⬜ unanswered. Placeholders `<arena-host>` and `<team>` are used below. |
| Both laptops | Windows, Claude Code installed |

---

## 2. Facts checked against the real code (Nov 16)

These were measured by building the simulator from the repo and probing it. They match the handoff,
with additions marked **NEW**.

| Fact | Value |
|---|---|
| Deployable cards | 116 (`cr_sim.Battle().deployable_cards()`); `cards.json` has 132 under key `cards`, plus `towers` = `princess_tower`, `king_tower` |
| `observe()` keys | `crowns, done, elixir, entities, hand, hand_cost, last_action, legal, next_card, outcome, phase, tick` |
| Entity | tuple locally, list from server: `(owner, card, x, y, hp, max_hp, flying, is_tower, deploying, id, half_w, half_h)` |
| Tower half-sizes | king 2.0 × 2.0, princess 1.5 × 1.5 |
| `view.result()` | a **method**; returns `None` while running, else `"win"/"loss"/"draw"` |
| Placements 0–8 (handoff §3.2) | all nine tiles return `last_action == "ok"` for a troop ✅ |
| Average random game | ~2,430 ticks |
| `targets` field values | only `"ground"` or `"both"` (so `hits_air = targets == "both"`) |
| Spawned-unit link | `spawn = {"card", "count", "interval"}`, `death_spawn = {"card", "count"}` (e.g. golem → golemite) |
| **NEW: enemy spells are invisible** | fireball, poison, rocket, arrows, earthquake and zap played by seat 1 **never appear** in seat 0's `observe()["entities"]`. The elixir tracker (C7) cannot see spell spending directly. See W8. |
| **NEW: `view.projectiles()` exists locally** | but the server only sends `observe()`, so it is **off-limits** as bot input (same rule as `b.elixir(1)`). |
| **NEW: Claude Code on Windows** | runs shell commands through Git Bash or PowerShell, so prompts always call the venv's Python by explicit path: `./.venv/Scripts/python` (works in both shells). |

---

## 3. Wrinkles in the handoff and how we resolve them

Claude includes the relevant resolution in each impl prompt. You don't have to remember these.

| # | Wrinkle | Resolution |
|---|---|---|
| W1 | `bot/policy.py` (Ben) calls `placements.decode` (Chris), which doesn't exist until C3. | `policy.py` imports `placements` **inside** `act()`, not at the top of the file. B1 pushes without waiting. |
| W2 | `RandomPolicy` must use `placements.mask` (C3), but B1 comes first. | impl-2 writes a fallback `RandomPolicy.act()` (random_agent-style). impl-4 switches it to the mask once C3 is on main. |
| W3 | C10 guardrails needs "the scripted defend choice". | impl-5 (C4) builds `scripted.py` as one function per rule: `defend_index(obs, state)`, `spell_index(...)`, `attack_index(...)`, `noleak_index(...)`. `guardrails.py` imports `defend_index`. |
| W4 | C9(e) champion ability returns `[100, 0, 0]`, which is not in the 53-action table. | `ScriptedPolicy` overrides `act()` to return the ability action; `act_index()` stays in the table. BC can't learn the ability. Accepted. |
| W5 | `--decks heldout` (B5, B15) needs a **deck list**, but B2 only makes a **card list**. | impl-4 (B2) also writes `evals/heldout_decks.json`: 40 decks, each with ≥ 3 held-out cards, seed 0. |
| W6 | B17 / C11 commands use `--deck` and start from `best.pt`, but B12's spec has neither flag. | impl-16 (B12) adds `--init <model path>` (default `models/bc_v1.pt`) and `--deck <json>` (our side's deck mix, per B17). |
| W7 | C13 writes `evals/candidate_decks.json` inside Ben's `evals/` folder. | Allowed exception: **Chris owns that one file**. Ben never edits it. |
| W8 | Enemy spells are invisible (section 2), so C7's 90% target may be unreachable. | impl-9 measures and reports the number. Optional heuristic: when one of our towers or units loses hp in a single tick with no enemy unit in range, charge the cheapest spell from `enemy_cards()` that could explain it. Keep it only if the test number goes up. |
| W9 | B6 encode needs `Trackers` (C8, Tuesday). | impl-10 codes against the two-call interface (handoff §3.4). Tests use a tiny stub class with `enemy_elixir()` and `enemy_cards()`. The real tracker is dropped in when impl-9 is on main. |
| W10 | `tests/` is shared. | Each person only creates `tests/test_<their file>.py`. **Nobody creates `tests/conftest.py` or `tests/__init__.py`.** |
| W11 | Windows multiprocessing (gauntlet, BC recording) re-imports the main module in each worker. | Every script that uses `multiprocessing.Pool` has an `if __name__ == "__main__":` guard and top-level worker functions. Workers build their policies from a string spec (`"scripted"`, `"learned:models/x.pt"`) and call `torch.set_num_threads(1)`. |
| W12 | `CLAUDE.md` and this file are shared docs. | Chris commits both in setup step S8. After that, change them only after telling the other person (like the contract). |

---

## 4. One-time setup (by hand, before any impl)

Do these in order. Each step says **who**, the exact commands, and how to know it worked.
Use **Windows PowerShell** (Start menu → type "PowerShell" → Windows PowerShell). Paste one line at a time.

### S1 — Git for Windows (both; skip if `git --version` already prints a version)

Claude Code on Windows also needs Git for Windows, so you probably have it already.

```powershell
git --version
```

If that prints `git version 2.x`, skip to S2. If it says "not recognized":

```powershell
winget install --id Git.Git -e --source winget
```

Then **close PowerShell and open a new one** (so it sees the new program) and run `git --version` again.

### S2 — Allow PowerShell to run the venv activation script (both)

Windows blocks `.ps1` scripts by default, and activating the venv is a `.ps1` script.

```powershell
Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned
```

Type `Y` and Enter if it asks. **Done when** `Get-ExecutionPolicy -Scope CurrentUser` prints `RemoteSigned`.

### S3 — Put the repo somewhere OneDrive can't touch (Chris; Ben optional)

OneDrive locks files while it syncs, which breaks git and training runs (handoff §4). Clone into `C:\dev` instead of Documents or Desktop.

**Chris:**
```powershell
mkdir C:\dev
cd C:\dev
git clone https://github.com/bea228/cr-rl-arena.git
cd cr-rl-arena
```

**Done when** `git log -1 --oneline` prints a commit (at least `ae6269d Mark PDFs as binary…`).

**Ben (optional, recommended):** your repo lives in OneDrive. Either pause OneDrive whenever you train or run gauntlets
(taskbar cloud icon → gear → Pause syncing → 24 hours), or re-clone into `C:\dev` the same way and
re-run S4 there. If you re-clone, first make sure everything in the old copy is pushed (`git status` says "nothing to commit"
and `git log origin/main..HEAD` prints nothing).

From here on, **every command runs from the repo root** (`C:\dev\cr-rl-arena`, or wherever Ben's lives).

### S4 — Run the installer (Chris: full; Ben: adds PyTorch)

```powershell
powershell -ExecutionPolicy Bypass -File .\install.ps1 -WithTorch
```

- Takes 5–15 minutes the first time. Most of it is compiling the simulator in Rust. It may install Python 3.12 and Rust for you through `winget`. If it tells you to open a new terminal and run it again, do exactly that.
- **Done when** it prints **`All set.`** in green, plus a line like `torch 2.x ok`.

**If it fails with "An Application Control policy has blocked this file"** (Windows Smart App Control):
Ben builds a wheel on his laptop and sends it to Chris. Both of you must have the **same Python version**, so first each run
`.\.venv\Scripts\python --version` and compare.

*Ben:*
```powershell
cd gym
..\.venv\Scripts\maturin build --release
cd ..
dir gym\target\wheels
```
Send Chris the `.whl` file that `dir` shows (Discord, Drive, USB, anything).

*Chris:*
```powershell
.\.venv\Scripts\python -m pip install C:\path\to\cr_sim-0.1.0-<...>.whl
.\.venv\Scripts\python scripts\check_install.py
```
`check_install.py` should print `played a full game…`, `numpy ok`, `websockets ok`, `torch … ok`.

**Re-run `install.ps1` after every `git pull` that changes anything under `gym/`.** It rebuilds the simulator so yours matches the server's.

### S5 — Check the venv (both)

```powershell
.\.venv\Scripts\Activate.ps1
python -c "import torch, cr_sim; print('ok', torch.__version__)"
```

Your prompt should now start with `(.venv)`, and the second line prints `ok 2.x…`.
You have to activate in **every new PowerShell window** you open. If you skip it, `python` means the wrong Python.

### S6 — Git identity and pull behaviour (both, once per laptop)

```powershell
git config user.name "Chris Byrnes"          # Ben: "Ben <last name>"
git config user.email "<the email on your GitHub account>"
git config pull.rebase true
git config pull.rebase                        # must print: true
```

The first time you `git push`, a browser window opens asking you to sign in to GitHub (Git Credential Manager).
Sign in with the account Ben invited. It remembers you after that.

### S7 — Open Claude Code in the repo (both)

```powershell
cd C:\dev\cr-rl-arena
claude
```

- The first time, it asks whether you trust this folder. Say yes.
- Type `/status` to confirm it's signed in, then `/exit` or Ctrl+C twice to leave.
- **Plan mode:** press **Shift+Tab** to cycle modes. In plan mode Claude Code proposes a plan and waits before editing anything. Use it for impls the table marks as *large*.
- Claude Code asks before running commands or editing files. Read what it wants to run. Approving `pytest`, `python -m …` and `git add <named file>` is normal. **Refuse** `git add .`, `git add -A`, `git push --force`, or any edit to a file you don't own (section 6).

### S8 — Commit the two shared docs (Chris, once; Ben pulls)

Copy `IMPLEMENTATION.md` (this file) into `docs/` and `CLAUDE.md` into the repo root (both come with this plan). Then:

```powershell
git pull --rebase origin main
git add CLAUDE.md docs/IMPLEMENTATION.md
git commit -m "[S8] implementation overview + CLAUDE.md"
git push origin main
```

**Ben:** `git pull --rebase origin main`. **Done when** both laptops have `CLAUDE.md` at the root.
Claude Code reads `CLAUDE.md` automatically every time it starts in the repo, so the ownership and git rules apply to every impl.

### Setup checklist

| | Chris | Ben |
|---|---|---|
| S1 git works | ⬜ | ⬜ |
| S2 execution policy | ⬜ | ⬜ |
| S3 repo outside OneDrive | ⬜ | ⬜ (or pause OneDrive) |
| S4 `All set.` with torch | ⬜ | ⬜ |
| S5 `import torch, cr_sim` | ⬜ | ⬜ |
| S6 git identity, `pull.rebase true` | ⬜ | ⬜ |
| S7 Claude Code opens in repo | ⬜ | ⬜ |
| S8 CLAUDE.md on main | ⬜ (commits) | ⬜ (pulls) |

---

## 5. Running one impl (every time)

```powershell
# 1. Fresh terminal, repo root, venv on
cd C:\dev\cr-rl-arena
.\.venv\Scripts\Activate.ps1

# 2. Get the other person's work first
git pull --rebase origin main
#    If anything under gym/ changed: powershell -ExecutionPolicy Bypass -File .\install.ps1 -WithTorch

# 3. Make sure main is healthy before you start
python -m pytest tests -q          # (skip until a tests/ folder exists)

# 4. Start Claude Code and paste the impl prompt
claude
```

While Claude Code works:
- It writes the files, writes tests, and runs them. Let it fix its own failures.
- At the end, every prompt tells it to: run the full test suite, run the smoke test (once the gauntlet exists),
  show `git status`, `git add` **only the named files**, commit with the `[ID]` message, and **ask you before `git push`**.
- **Before you approve the push:** skim the summary and check that `git status` lists only files you own.
- If the push is **rejected** ("fetch first" / "non-fast-forward"), Ben or Chris pushed in the meantime. That's normal.
  Tell Claude Code: *"pull --rebase, re-run the tests, then push."*
- If a rebase reports a **conflict**, someone edited the other person's file. Stop, run `git rebase --abort`, and message each other.

Afterwards, report back in the project chat: `impl-N done` plus any numbers it printed (win rates, accuracy, decisions/sec).

**If Claude Code gets stuck** (same error three times, or it wants to edit someone else's file):
press Esc, then tell it *"stop, summarise what's failing and what you've tried."* Paste that summary into the project chat.

---

## 6. The build sequence

### 6.1 Master table

Size: S ≈ under 30 min of Claude Code, M ≈ 30–60 min, L ≈ use plan mode.

| impl | Owner | Day | Handoff steps | Writes | Needs on main first | Size |
|---|---|---|---|---|---|---|
| 1 | Chris | Mon | C1 | `bot/cards.py`, `tests/test_cards.py` | — | M |
| 2 | Ben | Mon | B1 | `bot/__init__.py`, `bot/policy.py`, `evals/__init__.py`, `train/__init__.py`, `.gitignore` | — | S |
| 3 | Chris | Mon | C2, C3 | `bot/state.py`, `bot/placements.py`, tests | impl-1 | M |
| 4 | Ben | Mon | B2, B3, B4 (+W2, W5) | `evals/decks.py`, `evals/match.py`, `evals/replay.py`, 3 JSON files, `tests/test_replay.py`, `tests/test_policy.py` | impl-2, impl-3 (for W2 only) | M |
| 5 | Chris | Mon | C4, C6 | `bot/scripted.py`, `bot/decks/default.json`, `scripts/quick_loop.py`, `tests/test_scripted.py` | impl-2, impl-3 | L |
| 6 | Ben | Mon | B5 | `evals/gauntlet.py`, `evals/policies.py`, `tests/test_gauntlet.py` | impl-4 | M |
| 7 | Chris | Mon | C5 | `bot/client.py`, `tests/test_client.py` | impl-5 | M |
| 8 | Both | Mon | J1, J2, J3 | (fixes in `scripted.py` only) | impl-6, impl-7 | M |
| 9 | Chris | Tue | C7, C8 | `bot/trackers.py`, `tests/test_trackers.py`, ScriptedPolicy.reset hook | impl-8 | L |
| 10 | Ben | Tue | B6, B7 (+W9) | `bot/encode.py`, `bot/model.py`, tests | impl-3 | M |
| 11 | Chris | Tue | C9 | `bot/scripted.py`, `bot/scripted_v0.py` | impl-9 | L |
| 12 | Ben | Tue | B8, B9 | `train/bc_record.py`, `train/bc_train.py`, `models/bc_v1.pt` | impl-9, impl-10 (ideally impl-11) | L |
| 13 | Chris | Tue | C10 | `bot/guardrails.py`, `tests/test_guardrails.py` | impl-9 | M |
| 14 | Ben | Tue | B10 | `bot/learned.py`, `tests/test_learned.py` | impl-12, impl-13 | M |
| 15 | Ben | Wed | B11 | `bot/env.py`, `tests/test_env.py` | impl-14 | L |
| 16 | Ben | Wed | B12, B13 (+W6) | `train/ppo.py`, `train/league.py` | impl-15 | L |
| 17 | Chris | Wed | C11 | (setup + overnight run; no code) | impl-16 | S |
| 18 | Chris | Wed | C12, C13 | `docs/replay_notes.md`, `evals/candidate_decks.json`, scripted fixes | impl-8 | M |
| 19 | Ben | Wed | B14 | `evals/promote.py`, `models/VERSIONS.md` | impl-16 | S |
| 20 | Ben | Thu | B15, B16 | `evals/deck_rr.py` | impl-18 (C13), impl-19 | M |
| 21 | Chris | Thu | C14, C15 | `bot/decks/tournament.json`, `bot/client.py` | impl-20 | M |
| 22 | Ben → Chris runs | Thu | B17 | `train/ppo.py` deck-mix changes | impl-21 | S |
| 23 | Ben | Fri | B18, B19 | `evals/latency.py`, `models/ship.pt`, `models/VERSIONS.md` | impl-22 | M |
| 24 | Chris | Fri | C16, C17 | `run_bot.ps1`, `scripts/soak.ps1` | impl-21 | M |
| 25 | Both | Fri/Sat | J4 + tournament run sheet | tag `ship-v1` | everything | S |

### 6.2 Chris's track (game logic + live bot)

| Order | impl | What you get | Wait for |
|---|---|---|---|
| 1 | **impl-1** | Card stat vectors + role flags (`FEAT_DIM` for Ben) | nothing, start now |
| 2 | impl-3 | Board parser + the 53-action table and mask | impl-1 pushed (yours) |
| 3 | impl-5 | Scripted bot v0 + default deck | Ben's **impl-2** pushed (`git pull`, check `bot/policy.py` exists) |
| 4 | impl-7 | Tournament websocket client | impl-5 |
| 5 | impl-8 | **Go live tonight** (with Ben) | Ben's impl-6 |
| Tue | impl-9 → impl-11 → impl-13 | Trackers, scripted v1, guardrails | — |
| Wed | impl-17, impl-18 | GPU overnight training, replay review, candidate decks | Ben's impl-16 for impl-17 |
| Thu | impl-21 (+ run impl-22 overnight) | Tournament deck, event trigger | Ben's impl-20 |
| Fri | impl-24, impl-25 | Soak test, auto-restart script, freeze | — |

### 6.3 Ben's track (ML + measurement)

| Order | impl | What you get | Wait for |
|---|---|---|---|
| 1 | **impl-2** | Policy contract, `.gitignore`. **Push this first, Chris needs it.** | nothing, start now |
| 2 | impl-4 | Deck sampler, match runner, replays (+ switch RandomPolicy to the mask) | Chris's impl-3 for the mask part only; do the rest first |
| 3 | impl-6 | The gauntlet | impl-4 |
| 4 | impl-8 | Baseline J2 (with Chris) | Chris's impl-5 and impl-7 |
| Tue | impl-10 → impl-12 → impl-14 | Encoder, model, BC data + training, LearnedPolicy | Chris's impl-9 for impl-12, impl-13 for impl-14 |
| Wed | impl-15 → impl-16 → impl-19 | Env, PPO + league, promotion routine | — |
| Thu | impl-20, impl-22 | Held-out check, deck round robin, specialist | Chris's impl-18 (candidate decks) for impl-20 |
| Fri | impl-23, impl-25 | Latency, final gauntlet, freeze | — |

**Monday critical path to the ladder:** impl-1 + impl-2 → impl-3 → impl-5 → impl-7 → impl-8 (J3).
If it's getting late, impl-4/impl-6 (Ben's measurement) can finish after Chris is live. The ladder bot only needs Chris's track plus `bot/policy.py`.

---

## 7. Impl cards

Each card says what the prompt must contain beyond the handoff text. Claude copies the handoff step text verbatim
into the prompt and adds the card's notes.

### impl-1 · Chris · C1 card stats (`bot/cards.py`)
- Load `gym/data/cards.json` once at import (path relative to the repo root via `pathlib`, not the current directory). Keys: `cards` (132) and `towers` (2).
- `feat(name)`: the exact feature list in handoff C1. Facts from section 2: `hits_air = targets == "both"`, `hits_ground = targets in ("ground","both")` (buildings with no `targets` → 0). "has splash" = `splash_radius` **or** `radius` present. "spawn" = `spawn` **or** `death_spawn`. Unknown name → `KeyError` with a clear message.
- `FEAT_DIM = len(feat("knight"))`, computed, never typed in.
- `role(name)` returns a dict of bools: spell, building, flying, hits_air, splash, swarm (count ≥ 3), tank (hp ≥ 2500), win_condition (target_buildings_only), cheap (elixir ≤ 2).
- Also expose `elixir(name)`, `DEPLOYABLE` (from `cr_sim.Battle().deployable_cards()`), `NON_DECK` (the 16 spawn-only), `spawned_by(name)` (cards it spawns) for C7 later.
- Tests: handoff C1 checks + `FEAT_DIM` matches the vector length + roles for knight, giant, minions, fireball, cannon.
- Commit: `[C1] card stat table, FEAT_DIM, role flags`. After pushing, tell Ben the `FEAT_DIM` value.

### impl-2 · Ben · B1 the Policy contract
- `bot/policy.py` exactly as in handoff §3.1, with `placements` imported inside `act()` (W1).
- `RandomPolicy`: until `bot/placements.py` exists, `act()` behaves like `random_agent.py` (60% wait, random affordable slot, x ∈ [2,16], y ∈ [9,14]). Use `random.Random(seed)` passed in the constructor so games are reproducible.
- `.gitignore`: append `data/`, `runs/`, `evals/results/`. Don't remove existing lines.
- Tests: `tests/test_policy.py`: import works; `RandomPolicy` returns None or a valid `[int, float, float]` on 100 real observations.
- Commit: `[B1] Policy base class, DECIDE_EVERY, RandomPolicy`. **Message Chris when pushed.**

### impl-3 · Chris · C2 state parser + C3 placements
- `state.py` exactly per C2. Use a `dataclass`. Lane rule: `L if x < 9 else R`. Towers are found by `is_tower` + card name + x.
- `placements.py` exactly per handoff §3.2: `PLACEMENTS` list of 13 `(name, x, y)` entries (computed ones `None`), `N_ACTIONS = 53`, `encode(slot, p) = 1 + slot*13 + p`, `decode(index, obs) -> [slot, x, y] | None` (slot as **int**), `mask(obs) -> list[bool]` of 53.
- Spells are detected with `cards.role(name)["spell"]`. `cluster` per C3. `on_threat`: enemy unit nearest any of our living towers; clamp y to ≤ 14.5 for troops/buildings.
- `mask` rules exactly as listed in §3.2. Also: computed placements decode to a position snapped inside 0.5..17.5 × 0.5..31.5.
- Tests: C2 + C3 checks, plus: every index the mask allows decodes without error on 200 random-play observations, and `decode` + `encode` round-trip.
- Commit: `[C2][C3] state parser, 53-action table and mask`.

### impl-4 · Ben · B2 decks, B3 match runner, B4 replay
- `evals/decks.py`: `random_deck(rng, exclude=())`. Writes `evals/gauntlet_decks.json` (40 decks, no held-out cards), `evals/heldout_cards.json` (12 deck-able cards, not champions), **and `evals/heldout_decks.json`** (40 decks with ≥ 3 held-out cards, W5). A `python -m evals.decks --write` command regenerates all three with seed 0. Choose held-out cards first, then gauntlet decks avoid them.
- `evals/match.py` per B3. Policy objects get `reset(deck)` before the game. Count ignored moves = `last_action` not in `("ok", "none")`.
- `evals/replay.py` per B4. JSON format: `{seed, deck0, deck1, actions: [[tick, seat, [slot,x,y]], ...]}`.
- W2: if `bot/placements.py` is on main, update `RandomPolicy` to `act_index()` = random true entry of `mask(obs)` (wait stays possible). Otherwise leave a TODO and do it in impl-6.
- Commit: `[B2][B3][B4] deck sampler, match runner, replays`.

### impl-5 · Chris · C4 scripted bot v0 + C6 default deck
- Rules exactly as in C4, in that order. **Structure (W3):** one module-level function per rule, each `(obs, state) -> index or None`. `ScriptedPolicy.act_index` calls them in order and returns the first non-None, else 0. Every returned index must pass `placements.mask(obs)`.
- `bot/decks/default.json`: the 8 cards from C6. Test that `check_deck` passes.
- `scripts/quick_loop.py`: 40 games vs `RandomPolicy`, random decks (use a local copy of the deck sampler until Ben's `evals/decks.py` exists), act every 10 ticks. Prints win rate and `last_action` counts.
- Done when: > 85% wins over 40 games and **zero** `not_your_side`.
- Commit: `[C4][C6] scripted bot v0, default deck`.

### impl-6 · Ben · B5 gauntlet
- Arguments and outputs exactly per B5. Policy spec strings: `random`, `scripted`, `learned:<path>` (lazy import, since `learned.py` doesn't exist yet: clear error message), and a generic `class:<module>.<Class>` (e.g. `class:bot.scripted_v0.ScriptedPolicyV0`) so later variants never need a gauntlet edit. Put the spec parser in `evals/policies.py` (`make_policy(spec)`) so `match.py`, the BC recorder and the env reuse it.
- **Windows (W11):** `if __name__ == "__main__":`, top-level worker, `Pool(workers)` default `os.cpu_count() - 1`.
- Seat fairness: alternate which seat `a` plays (even i: a is seat 0) so seat effects cancel out.
- Done when: random vs random, 400 games, 43–57%, under 2 minutes. Print the wall time.
- Commit: `[B5] gauntlet`.

### impl-7 · Chris · C5 tournament client
- Start from `server/examples/random_agent.py`. All of C5. Also handle `aborted` (log it, keep going), `error` (log it, then the reconnect loop takes over), `joined`, `start`.
- `--policy scripted|learned|random`. `learned` is imported lazily.
- Policy errors **and** slow decisions must never miss a tick. Wrap `act` in try/except, always send a reply, and log any decision slower than 200 ms.
- Print per-match: result, crowns, rating, delta, replay id, and the `last_action` counts.
- Log lines also go to `runs/client.log`.
- Test: a local unit test with a fake websocket isn't required. The acceptance check is J1 (5 matches in a row against `random_agent.py`).
- Commit: `[C5] tournament client with reconnect`.

### impl-8 · Both · J1 local server, J2 baseline, J3 go live
Mostly a run sheet. The prompt is a *diagnose-and-fix* prompt for Chris if J2 < 95%.
1. Either laptop: `python -m pip install ./server`, then `cr-arena` in terminal 1. Leave it running.
2. Terminal 2: `python -m bot.client --url ws://localhost:8765 --name dev-scripted --policy scripted --deck bot/decks/default.json`
3. Terminal 3: `python server/examples/random_agent.py ws://localhost:8765 rando`
4. Browser: http://localhost:8765 shows dev-scripted winning; ignored-move share under 2%.
5. Ben: `python -m evals.gauntlet --a scripted --b random --games 400`. **Post the win rate in team chat and in the project chat (progress log).**
6. If under 95%: Ben sends Chris the 10 worst-loss replay files; Chris asks for **impl-8** and Claude builds a prompt that replays them and fixes the failing rule.
7. J3, once the organizers answer: `python -m bot.client --url wss://<arena-host> --name <team> --policy scripted --deck bot/decks/default.json`. Leave it running (sleep off; see impl-17 for how).

### impl-9 · Chris · C7 + C8 trackers
- First the probe in C7. Section 2 already shows enemy spells are invisible, so confirm and write down what *is* visible (spawned units, buildings).
- `EnemyElixir`, `EnemyCycle`, `Trackers(reset, update(obs), enemy_elixir(), enemy_cards())`, exactly per handoff §3.4. Regen rates per handoff §1, phase from `obs["phase"]`.
- Don't charge for `NON_DECK` cards, or for cards spawned by an enemy unit already on the board (`cards.spawned_by`).
- `ScriptedPolicy.reset` makes a fresh `Trackers`, and `act_index` calls `update(obs)` first.
- Optional W8 heuristic, kept only if it improves the test number.
- Tests per C7 + C8. **Report the C7 percentage** in the project chat.
- Commit: `[C7][C8] enemy elixir and cycle trackers`.

### impl-10 · Ben · B6 encoder + B7 model
- Shapes per B6. Use `cards.FEAT_DIM` and `placements.N_ACTIONS`, never literals. Units sorted by distance to (9, 3); pad/truncate to 48.
- Trackers stub for tests (W9).
- Model per B7, plus a `batch_encode(list_of_dicts)` helper for training. Print parameter count and CPU forward time in the test.
- Commit: `[B6][B7] observation encoder and network`.

### impl-11 · Chris · C9 scripted v1
- First `copy bot/scripted.py bot/scripted_v0.py` (class renamed `ScriptedPolicyV0`), committed **unchanged**. Compare with `python -m evals.gauntlet --a scripted --b class:bot.scripted_v0.ScriptedPolicyV0 --games 400` (the generic spec from impl-6, so no edit to Ben's file).
- (a)–(e) per C9. (e) uses the W4 override.
- Done when per C9 (≥ J2 number vs random; > 55% vs v0, 400 games).
- Commit: `[C9] scripted bot v1`.

### impl-12 · Ben · B8 BC recording + B9 BC training
- Recording runs scripted with real `Trackers` (needs impl-9). If impl-11 lands later, re-record (about 30 min). Note which scripted version was recorded in the `.npz` metadata.
- Multiprocessing per W11. Shards of ~50k decisions each in `data/bc/`.
- Training per B9. Save `{"state_dict", "FEAT_DIM", "N_ACTIONS", "config"}`. `models/bc_v1.pt` must be under 10 MB (it will be at < 1M params).
- Commit: `[B8][B9] BC recording and training, bc_v1`.

### impl-13 · Chris · C10 guardrails
- Per C10. Imports `defend_index` from `scripted.py` (W3). `runs/overrides.jsonl` (create `runs/` if missing).
- Commit: `[C10] guardrails`.

### impl-14 · Ben · B10 LearnedPolicy
- Per B10. Device CPU. `torch.no_grad()`, `torch.set_num_threads(1)` in the client. Must load in under 2 s.
- Add `learned:<path>` to the gauntlet. Report the number vs J2.
- Commit: `[B10] LearnedPolicy`.

### impl-15 · Ben · B11 batched env
- Per B11 over `cr_sim.BatchBattle`. Opponent actions come from policy objects on `bb[i].view(1).observe()`. The learner only ever sees `view(0)`.
- Print decisions/sec. Target > 2,000.
- Commit: `[B11] batched training env`.

### impl-16 · Ben · B12 PPO + B13 league
- Per B12 + B13, plus `--init` and `--deck` (W6), and `--minutes` for short tests. Checkpoint every 50 updates so an overnight crash loses little. Log to `runs/<run>/log.csv`.
- Opponent-policy inference batched where possible; CPU-bound sim is the bottleneck.
- Done when per B12 + B13 (20-minute CPU run).
- Commit: `[B12][B13] PPO with league opponents`.

### impl-17 · Chris · C11 GPU setup + overnight run (thorough setup steps)
The prompt for this one is a **setup guide for you**, plus a short Claude Code check. Outline:
1. `nvidia-smi` in PowerShell. It shows your GPU and the highest CUDA version the driver supports (top right). If "not recognized", update the NVIDIA driver from nvidia.com first.
2. pytorch.org/get-started/locally → Stable, Windows, Pip, Python, the newest CUDA ≤ what `nvidia-smi` showed. Copy the `--index-url` it shows.
3. Replace the CPU build (pip won't swap it on its own, since the version number matches):
   `.\.venv\Scripts\python -m pip uninstall -y torch` then `.\.venv\Scripts\python -m pip install torch --index-url <url from step 2>`
4. `python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"` → `True <your GPU>`.
5. From now on run `install.ps1` **without** `-WithTorch` on this laptop.
6. Sleep off while plugged in: `powercfg /change standby-timeout-ac 0` and `powercfg /change hibernate-timeout-ac 0`. Control Panel → Power Options → "Choose what closing the lid does" → Plugged in: **Do nothing**. Undo after the hackathon with `powercfg /change standby-timeout-ac 30`.
7. Pause OneDrive. `git pull --rebase origin main`. `python -m train.ppo --device cuda --hours 8 --run wed`.
8. Morning: `dir runs\wed` shows `snap_*.pt`. Tell Ben. Don't commit `runs/`.

### impl-18 · Chris · C12 replay notes + C13 candidate decks
- C12: needs the 10 worst-loss replays (from the latest gauntlet in `evals/results/`). The prompt writes a small `scripts/show_replay.py` (Chris-owned) that prints a replay as a readable timeline of plays and tower hp. Chris writes the 10 lines; Claude Code fixes the scripted causes.
- C13 per spec (W7). Use `cards.role` to check each deck's composition in a test.
- Commit: `[C12][C13] replay notes, scripted fixes, candidate decks`.

### impl-19 · Ben · B14 promotion routine
- `evals/promote.py --candidate <path>`: runs both 400-game gauntlets, applies the exit rule, and if it passes copies `best.pt → prev_best.pt`, the candidate → `best.pt`, and appends a line to `models/VERSIONS.md`. Prints the git commands, doesn't run them.
- Commit: `[B14] promotion routine`.

### impl-20 · Ben · B15 held-out check + B16 deck round robin
- B15 uses `evals/heldout_decks.json` (W5). B16 per spec. Opponents on random decks.
- Commit: `[B15][B16] held-out check, deck round robin`.

### impl-21 · Chris · C14 deck choice + C15 event trigger
- C14 per spec. C15 behind `--event-trigger` flag, default off. Keep trackers correct when decisions are irregular (they already use tick differences).
- Commit: `[C14][C15] tournament deck, event trigger`.

### impl-22 · Ben (code) → Chris (overnight run) · B17 specialist
- Deck-mix changes in `env.py`/`ppo.py` per B17. Chris runs it with the impl-17 command plus `--run thu --deck bot/decks/tournament.json --init models/best.pt`.
- Commit: `[B17] specialist fine-tune option`.

### impl-23 · Ben · B18 latency + B19 final gauntlet
- Per spec. Record observations from gauntlet games. The final table goes into `models/VERSIONS.md`, `models/ship.pt` is committed.
- Commit: `[B18][B19] latency check, final gauntlet, ship.pt`.

### impl-24 · Chris · C16 soak test + C17 `run_bot.ps1`
- `run_bot.ps1`: `$Mode = "ship"` / `"rollback_scripted"` / `"rollback_prev"` at the top. Activates the venv, loops forever, restarts 5 s after any exit, logs to `runs/run_bot.log`.
- `scripts/soak.ps1` starts the local server + two clients and stops at 1,000 matches. Restart the server once by hand midway.
- Commit: `[C16][C17] soak test, run_bot.ps1`.

### impl-25 · Both · J4 freeze + Saturday run sheet
- Run sheet only (handoff §9 J4 + §10). Claude produces a printable one-page checklist with the rollback rules filled in with the real gauntlet numbers.

---

## 8. Prompt template (Claude follows this when asked for `impl-N`)

When a user in this project asks for `impl-N`, Claude:
1. Reads this file (card N + section 3 wrinkles that apply), the handoff text for the step IDs, and the progress log (section 9).
2. If the card's "Needs on main first" items aren't confirmed done in the progress log, asks in one line whether they are pushed.
3. Replies with a short note (who runs it, plan mode or not, anything to do by hand first) and **one fenced code block** containing the prompt, in this structure:

```
# impl-N · <Owner> · <step IDs> · <title>

## Context
You are working in the cr-rl-arena repo for <Owner>, who owns: <file list from handoff §2>.
CLAUDE.md has the repo rules; follow them. Spec source: docs/Bot_Handoff.pdf steps <IDs>.
Already on main that you can rely on: <modules + exact names>.

## Before you start
git pull --rebase origin main, then check that <dependency files> exist. If not, stop and tell me.

## Files you may create or edit (nothing else)
<list>

## What to build
<handoff step text, verbatim, then the card's notes and applicable W-resolutions>

## Tests to write
<file + exact assertions>

## Checks to run (all must pass)
./.venv/Scripts/python -m pytest tests -q
<step-specific commands + the number each must hit>

## Finish
1. Show `git status` and a short summary of what you built and the numbers from the checks.
2. git add <exact files>   (never `git add .` or `-A`)
3. git commit -m "<message>"
4. Ask me before running: git push origin main. If it's rejected, pull --rebase, re-run tests, push.
5. Print a 3-line report I can paste back into the project chat.
```

---

## 9. Progress log (Claude updates this as you report back)

| impl | Status | Date | Numbers / notes |
|---|---|---|---|
| setup Chris | ⬜ | | |
| setup Ben | ⬜ | | |
| 1 | ⬜ | | FEAT_DIM = |
| 2 | ⬜ | | |
| 3 | ⬜ | | |
| 4 | ⬜ | | |
| 5 | ⬜ | | quick-loop win rate = |
| 6 | ⬜ | | random vs random = , wall time = |
| 7 | ⬜ | | |
| 8 | ⬜ | | **J2 baseline (scripted vs random, 400) =** |

Key numbers later impls depend on: **J2 baseline**, C7 elixir-tracker %, bc_v1 vs random, env decisions/sec, best.pt vs scripted.

---

## 10. Open organizer questions (handoff §11), ask tonight

| # | Question | Answer |
|---|---|---|
| 1 | Will some tournament matches assign a random deck? How is the bot told its cards? | |
| 2 | Arena `wss://` host, and when rated play starts | |
| 3 | Team name: must it be registered? (We'll use `<team>` and `<team>-dev`) | |
| 4 | Which simulator commit does the server run? | |

---

## 11. Beginner troubleshooting

| Symptom | Fix |
|---|---|
| `python` runs but `import cr_sim` fails | The venv isn't active. `.\.venv\Scripts\Activate.ps1`. Prompt must show `(.venv)`. |
| "running scripts is disabled on this system" | Do S2. |
| `ModuleNotFoundError: No module named 'bot'` | You're not in the repo root, or you ran `python bot\client.py` instead of `python -m bot.client`. |
| `git push` rejected (fetch first) | `git pull --rebase origin main`, re-run tests, push again. Normal. |
| Rebase conflict | `git rebase --abort`, then message each other. It means someone edited the other's file. |
| "Permission denied" / file in use during training | OneDrive or an open editor has the file. Pause OneDrive; close the file. |
| Gauntlet hangs or spawns endless processes on Windows | Missing `if __name__ == "__main__":` guard (W11). Tell Claude Code that. |
| `install.ps1` fails with Application Control | See S4's wheel fallback. |
| Claude Code wants to edit a file you don't own | Refuse. Message the owner with what you need. |
| You accidentally committed `data/` or a big file | Don't push. `git reset --soft HEAD~1`, then `git restore --staged <file>`, and re-commit only your files. |

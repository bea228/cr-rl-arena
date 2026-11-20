# Model versions

One line per model file that has been on main. Gauntlet numbers: 400 games, seed 0,
`python -m evals.gauntlet`; "default deck" = `--a-deck bot/decks/default.json`, "rotating" = gauntlet decks.

| Date | File | Code commit | Training data | Held-out non-wait acc | vs random (default deck) | vs random (rotating) | vs scripted (default deck) |
|---|---|---|---|---|---|---|---|
| 2026-11-19 | `models/bc_v1.pt` | d35258b | BC: 1,031,105 scripted decisions from 13,000 games (57.2% wait), 5 epochs, CPU | 70.4% (top-3 94.3%) | 92.5% (scripted 92.8%) | 66.5% (scripted 66.0%) | 71.0% (scripted itself on default deck: 72.5%) |

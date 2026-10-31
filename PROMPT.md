\# Project: Premier League Player Rating System (SofaScore-style)



\## Your role

You are the lead engineer and orchestrator. Break the work into phases, use subagents

for independent work, verify everything with tests, and stop at the checkpoints marked

⏸ CHECKPOINT to show me results before continuing. Do not skip checkpoints.



\## Goal

Build a system that gives every player a 3–10 rating for each match, anchored so the

average is 6.5 and a 10 is extremely rare. Build two rating methods and compare them:

1\. \*\*Baseline:\*\* hand-weighted stats, position-specific weights.

2\. \*\*VAEP:\*\* action values computed with the `socceraction` library.



Then build a match-ratings web page, a grounded "why this rating?" explanation feature,

and a gameweek pipeline for the current season.



\## Data (use only these sources)

\- \*\*Development and validation:\*\* StatsBomb Open Data, Premier League 2015/16

&#x20; (competition\_id=2, season\_id=27), loaded via `statsbombpy` or `socceraction`'s

&#x20; StatsBombLoader. Credit StatsBomb as the data source in the README and on the web page.

\- \*\*Current season (step 6 only):\*\* the official Fantasy Premier League API (per-match

&#x20; player stats). It has no event-level data, so only the baseline method runs on it.

\- Do NOT use FBref for advanced stats (removed in 2026). Do NOT scrape SofaScore or

&#x20; WhoScored. Never invent or simulate data. If a source fails, stop and tell me.



\## Interface contract (every rating method must output this)

A table with columns: `match\_id, player\_id, player\_name, team, position\_group

(GK/DEF/MID/FWD), minutes, raw\_score`. The scale module turns raw\_score into `rating`.

Build against this contract so components can be developed in parallel.



\## Rules (each one must become a test)

\- Only players with 10+ minutes get a rating.

\- Raw scores are compared within position groups, never across them.

\- Over a full season: mean rating 6.5 ± 0.1, ratings bounded 3.0–10.0,

&#x20; fewer than 0.1% of ratings equal 10.0, one decimal place.

\- Every match has 11 rated starters per team unless a player got under 10 minutes

&#x20; (e.g. early injury or red card); the test must account for that.

\- No rating function may read data from a different match than the one it is rating.



\## Repo structure

rating-system/

├── CLAUDE.md          # write this first: goal, rules, contract, conventions

├── README.md

├── data/raw/, data/processed/

├── src/ingest/, src/baseline/, src/vaep/, src/scale/, src/explain/, src/pipeline/

├── web/

├── tests/

└── reports/



\## Phases



\### Step 1: Scaffold

Create the repo structure, CLAUDE.md (containing this goal, data rules, interface

contract, and test rules), a pinned requirements file, and the full test suite for the

rules above. The tests should fail at this point, since nothing is implemented.

⏸ CHECKPOINT: show me CLAUDE.md and the list of tests.



\### Step 2: Ingestion

Load all 380 matches of 2015/16 into processed Parquet files (events, lineups, minutes

played, positions mapped to GK/DEF/MID/FWD). Add a test that checks 380 matches and

20 teams. Print a summary: matches, events, players.



\### Step 3: Parallel build (use four subagents, each on its own git branch)

\- \*\*Baseline agent:\*\* \~25–35 stats, separate weights per position group, documented

&#x20; in a weights file with a one-line justification per weight.

\- \*\*VAEP agent:\*\* train VAEP with socceraction (train on part of the season, score the

&#x20; rest, then also produce full-season values with clearly documented train/score splits

&#x20; so no match is scored by a model trained on that same match). Sum action values per

&#x20; player per match as raw\_score.

\- \*\*Scale agent:\*\* percentile-based mapping from raw\_score to 3–10 within position

&#x20; group, satisfying the distribution tests. Build against fake data in the contract format.

\- \*\*Frontend agent:\*\* a single-page app: pick a match, see both lineups on a pitch

&#x20; with color-coded ratings (red <6, orange 6–6.9, green 7–7.9, blue 8+) and a toggle

&#x20; between baseline and VAEP. Build against sample JSON in the contract format.

Merge branches only after each agent's tests pass.



\### Step 4: Verification (fresh subagent that has NOT seen the build code's history)

The verifier must:

1\. Run the full test suite and report failures.

2\. Check position-group handling throughout.

3\. Eye test: top 20 single-match ratings of the season, and full ratings for 5

&#x20;  notable 2015/16 matches (include at least two Leicester games).

4\. Compare methods: rank correlation between baseline and VAEP, and the 10 players

&#x20;  where they disagree most, with a hypothesis for each.

5\. Stability: correlation of each player's average rating between the first and

&#x20;  second half of the season (higher = more reliable signal).

Write everything to reports/validation.md. Then fix all failures and re-run until clean.

⏸ CHECKPOINT: show me the validation report. I will decide whether to adjust weights.



\### Step 5: Explanation feature

For any player-match, produce a short explanation of the rating using ONLY that

player's top positive and negative actions and their values (VAEP) or stat

contributions (baseline). Pass that structured data to an LLM with instructions not to

mention anything absent from the data. Make the LLM call pluggable via an environment

variable; ask me before using any API key. Add a test that every number in an

explanation appears in the input data. Show 3 example explanations.



\### Step 6: Current-season pipeline

Build a script that: pulls the latest finished gameweek from the FPL API, maps its stats

to the baseline model (document which baseline stats are unavailable and how weights

were adjusted), rates every player, runs validity checks (no missing teams, distribution

within bounds), and writes output the web page can load. If any check fails, stop and

report instead of writing output. Run it once on the latest gameweek.

⏸ CHECKPOINT: show me the ratings for one current match and ask whether to set it up

as a scheduled weekly task.



\## Working rules

\- Keep me updated with a task list. Keep narration short.

\- Never mark a step done without running its tests and showing the output.

\- If a library API differs from what you expect, read its docs/source; don't guess.

\- If you're blocked (network, data, missing package), say exactly what's blocking you

&#x20; and what you need from me.

\- At the end: a README with setup, how to run each step, method descriptions,

&#x20; known limitations, and data credits.


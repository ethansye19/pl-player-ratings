# Premier League player ratings

Match-by-match player ratings on a **3.0–10.0 scale** for every Premier League match of
2015/16, built two independent ways and compared:

1. **Baseline** – 35 per-match stats, hand-weighted with a separate weight profile per
   position group.
2. **VAEP** – the value of every on-the-ball action, learned from the event data with
   [`socceraction`](https://github.com/ML-KULeuven/socceraction), summed per player per match.

Around them: a match page that draws both lineups on a pitch, a "why this rating?"
explanation feature that is checked against the player's own numbers, and a pipeline that
rates the current season from the official Fantasy Premier League API.

**Live page (2015/16 season):** https://ethansye19.github.io/pl-player-ratings/

![The match page: Manchester City 6–1 Newcastle United with both lineups on a pitch and the player table](assets/match-page.png)

*Manchester City 6–1 Newcastle United, 3 October 2015. Agüero's five goals earn one of the
three 10.0s of the season.*

## What a rating means

- Every player who was on the pitch for **10 minutes or more** gets a rating for that match.
- Ratings are **anchored**: over the season each position group (GK, DEF, MID, FWD) averages
  6.5, about a quarter of appearances fall below 6.0, and a **10.0 is extremely rare** (3 of
  10,133 rated appearances per method).
- A rating only compares a player with **other players of the same position group**. A
  7.5 for a centre-back and a 7.5 for a striker mean "better than the same share of their
  peers", not the same actions.
- Colours on the page: red below 6.0, orange 6.0–6.9, green 7.0–7.9, blue 8.0 and above,
  grey for players under 10 minutes.

The development and validation data is StatsBomb Open Data for the 2015/16 season:

| quantity | value |
|---|---|
| matches / teams | 380 / 20 |
| events | 1,313,773 |
| SPADL actions (VAEP input) | 757,641 |
| players who appeared | 550 |
| player-match appearances | 10,469 (10,133 with 10+ minutes) |

## How the two methods work

### Baseline: weighted match stats

Thirty-five stats are counted for each player from that match's events only, from goals and
expected goals down to pressures and goalkeeper claims. Each position group has its own
weight profile (GK 28 stats, DEF 32, MID 32, FWD 30), and the raw score is the weighted sum.
Every weight carries a one-line justification in `src/baseline/weights.py`; a few examples:

| stat | GK | DEF | MID | FWD |
|---|---:|---:|---:|---:|
| non-penalty goal | 4.0 | 3.5 | 3.0 | 3.0 |
| penalty scored | 1.25 | 1.25 | 1.25 | 1.25 |
| assist | 2.0 | 2.0 | 2.0 | 1.8 |
| non-penalty xG (per 1.0) | 1.0 | 1.0 | 1.5 | 1.5 |
| tackle won | 0.2 | 0.3 | 0.3 | 0.2 |
| possession lost | -0.5 | -0.25 | -0.25 | -0.15 |
| goal conceded while on the pitch | -1.0 | -0.4 | -0.3 | -0.2 |
| clean sheet (60+ minutes) | 1.5 | 0.5 | 0.25 | – |
| save / penalty save | 0.5 / 2.0 | – | – | – |
| yellow card / red card / own goal | -0.5 / -3 / -3 | same | same | same |

Team context (goals conceded, xG faced, clean sheets) is counted only while the player was on
the pitch. There is no minutes term: a substitute with nothing to show scores about 0, below
a quiet full-match starter and above anyone with a clearly negative game. The full table with
definitions is in `reports/baseline.md`.

### VAEP: the value of every action

VAEP (Decroos, Bransen, Van Haaren and Davis, *Actions Speak Louder than Goals*, KDD 2019)
values each on-the-ball action by how much it changes the team's chance of scoring, minus how
much it changes the chance of conceding, within the next ten actions. Two gradient-boosted
models (`lightgbm`) predict those probabilities from 568 features describing the action and
its two predecessors; a player's raw score is the sum of their action values in the match.

Two choices matter for the results:

- **Shots are valued by chance quality, not outcome.** The training labels are built from
  StatsBomb's expected goals (the probability that at least one chance in the window is
  converted) instead of whether a goal was scored. A goal is worth the chance it came from
  rather than a flat +0.8, a missed penalty is no longer a catastrophe, and goalkeepers and
  blockers are credited with the chances they stop. `reports/vaep_splits.md` documents the
  formula and its effects.
- **No match is scored by a model that saw it.** A chronological holdout (train matchweeks
  1–19, score 20–38) and 5-fold cross-fitting over the whole season are both recorded with
  their full match lists in `vaep_splits.json`, and a test checks that no scored match is in
  its model's training set. Held-out AUC is about 0.81 for both the scoring and the
  conceding model.

VAEP sees only on-the-ball actions, so it undervalues defensive work: clearances, aerial
duels, blocks and clean sheets carry little or no value, and a goalkeeper's saves are not
actions of the keeper in the SPADL representation. The page therefore shows goalkeepers'
baseline ratings in VAEP mode (marked "B"), and VAEP ratings for defenders are best read as
attacking contribution.

### From raw score to rating

Raw scores are only comparable within a position group, so each raw score becomes a
percentile within its group, and a fixed curve maps percentiles to ratings: 3.0 at the
bottom, 6.0 around the 25th percentile, a mean of 6.5, and 10.0 only above roughly the
99.96th percentile. The percentile tables are fitted once per season and frozen as JSON, so a
single match can be rated without touching any other match. `reports/scale.md` has the curve
and its rationale.

## Validation, and the weight fix before and after

After the first full run, an independent verification pass (`src/validation/`, report in
`reports/validation.md`) re-derived every rating rule from the output files, eye-tested the
top-20 ratings and five notable matches, compared the two methods, and measured whether a
player's ratings in the first half of the season predict the second half. It found no rule
violations, but it did find that defenders' baseline ratings were dominated by the scoreline,
that forwards were punished too hard for lost duels, and that outcome-labelled VAEP was a
goals-and-misses detector.

Five decisions were applied: team-context weights for DEF and MID halved; forwards' penalties
for lost aerials and lost possession eased; penalties separated from open-play goals (and
assists from key passes); VAEP shots valued through xG labels; goalkeepers shown with their
baseline rating in VAEP mode. The verifier then re-ran on the new outputs:

| quantity | before | after |
|---|---|---|
| Spearman(baseline, VAEP), all player-matches | 0.448 | 0.536 |
| ... GK / DEF / MID / FWD | 0.16 / 0.32 / 0.55 / 0.56 | 0.44 / 0.43 / 0.62 / 0.58 |
| ... season averages (393 players) | 0.613 | 0.677 |
| split-half stability (Pearson), baseline / VAEP | 0.544 / 0.411 | 0.564 / 0.518 |
| team-context share of baseline raw variance, DEF / MID / GK | 40% / 13% / 50% | 15% / 4% / 50% |
| Spearman(baseline rating, goal difference), DEF | 0.515 | 0.373 |
| VAEP value of an open-play goal / miss / keeper save | +0.838 / -0.053 / -0.004 | +0.120 / +0.020 / +0.085 |
| VAEP value of a converted / missed penalty to the taker | +0.063 / -0.752 | -0.112 / -0.298 |
| baseline 10–20 minute cameos: mean / share below 6.0 | 6.01 / 43% | 5.85 / 65% |

The two methods now agree more and both are more stable from one half of the season to the
other, while defenders' ratings depend less on the result. Two things got worse and are
documented rather than hidden: VAEP's split-half stability for defenders fell (0.44 to 0.29)
because xG labels moved value from the shot to the build-up, and short baseline cameos
drifted down once the team-context terms that used to prop them up were halved. A converted
penalty is still worth slightly less than nothing to the taker under VAEP, because the models
under-predict the 91 penalty states in the data. Section 7 of `reports/validation.md` has the
full before/after tables, including every player in the five notable matches.

## Why this rating?

Every rating can be explained in a few sentences. The explanation is built from **only that
player's data in that match**: the rating, the raw score, and the five largest positive and
negative items (stat contributions for the baseline, individual actions for VAEP). A checker
then confirms that **every number in the text appears in that input**, so an explanation
cannot quote a stat the player does not have.

```powershell
.venv\Scripts\python -m src.explain.explain --match 3754079 --player 3237 --method both
```

> Sergio Agüero (Manchester City, FWD) rated 10.0 with the baseline method in Manchester City
> 6-1 Newcastle United, playing 67.8 minutes. Agüero's 10.0 came mostly from 5 non-penalty
> goals (+15.00), non-penalty xG of 2.03 (+3.05) and 5 shots on target (+2.00). Against that,
> 5 possessions lost (-0.75) and 1 goal conceded while on the pitch (-0.20) pulled the score
> down. Raw score +19.89.

> Robert Huth (Leicester City, DEF) rated 10.0 with the baseline method in Manchester City 1-3
> Leicester City, playing 97.2 minutes. Huth's 10.0 came mostly from 2 non-penalty goals
> (+7.00), 5 blocks (+1.00) and 5 aerial duels won (+1.00). Against that, 1 goal conceded
> while on the pitch (-0.40) and 1 possession lost (-0.25) pulled the score down. Raw score
> +11.44.

> Roberto Firmino (Liverpool, FWD) rated 3.5 with the VAEP method in Leicester City 2-0
> Liverpool, playing 87.7 minutes. Firmino's 3.5 rests on a successful pass in minute 83
> (+0.06), a successful pass in minute 38 (+0.02) and an unsuccessful shot in minute 57
> (+0.02). The costliest moments were a bad touch in minute 77 (-0.24) and a bad touch in
> minute 67 (-0.11). Raw score -0.79.

The text above comes from the deterministic **template provider**, which needs no API key
and is the default. An **LLM provider** (Claude, through the official SDK) is wired up behind
`RATING_LLM_PROVIDER=anthropic` with `ANTHROPIC_API_KEY`; its output goes through the same
number check and falls back to the template text if it fails. It is not used by default.

## The current season: a gameweek pipeline on the FPL API

`python -m src.pipeline.gameweek` rates the latest finished gameweek of the current Premier
League season from three endpoints of the official Fantasy Premier League API:
`bootstrap-static` (gameweeks, teams, players), `fixtures/?event=<gw>` and
`event/<gw>/live` (every player's stats, itemised per fixture). FPL publishes per-gameweek
totals rather than events, so only the baseline method runs, and only **15 of its 35 stats**
exist there (`reports/fpl_mapping.md` lists every stat, its FPL field or the reason FPL lacks
it, and the two weight adjustments this forces). The scaler is fitted on all finished
gameweeks of the season so far, and the gameweek is published only if every check passes:
all 20 teams present, every starter rated unless under 10 minutes, ratings finite and in
range with one decimal, gameweek mean within 6.5 ± 0.3.

A PowerShell wrapper (`scripts/run_gameweek.ps1`) runs it from Task Scheduler once a day. It
does nothing when the latest finished gameweek is already published, writes nothing when a
check fails, and refuses to replace published output with an older gameweek or with fewer
fixtures or rated rows. Current-season output lives under `web/data/current/`, which is
local only: the public page carries the 2015/16 season.

## Known limitations

- **Current-season ratings use FPL data only.** The FPL API has no passing, shot,
  dribble or duel counts, so 20 of the 35 baseline stats are missing there and
  midfielders are rated mainly on defensive actions (tackles, clearances/blocks/
  interceptions, recoveries) and team context (goals conceded, clean sheets, xG conceded);
  forwards mainly on goals, assists and xG. Treat current-season ratings as coarser than
  the 2015/16 ones, and expect quiet midfielders to cluster near the middle.
- Only the baseline method runs on the current season: VAEP needs event data.
- FPL positions follow the game's classification (wingers are usually MID), and FPL
  player names are short display names.
- VAEP undervalues defensive work (see the caveat above); goalkeepers are shown
  with their baseline rating in VAEP mode.
- Explanations default to the template provider. The Anthropic provider is wired up
  behind `RATING_LLM_PROVIDER=anthropic` but has not been used or tested with a key.
- Ratings are percentile-anchored within each season: a 7.0 in 2015/16 and a 7.0 in the
  current season are both "better than about three quarters of the group", not the same
  underlying performance.
- The public page (`docs/`, served by GitHub Pages) contains the 2015/16 season only.
  Current-season ratings are not published.

## Setup

Python **3.11** is required: `socceraction` 1.4.2 pins `pandas<2` and `numpy<2`, which have
no wheels for newer interpreters. The commands below are for Windows PowerShell; on macOS or
Linux use `.venv/bin/python` instead of `.venv\Scripts\python`.

```powershell
py -3.11 -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt   # fully pinned
.venv\Scripts\python -m pytest -q -m "not data"           # tests that need no downloaded data
```

Reproduce the 2015/16 ratings from scratch (the download is about 1.1 GB; VAEP training
takes a few minutes):

```powershell
.venv\Scripts\python -m src.ingest.download               # StatsBomb open data -> data/raw/statsbomb
.venv\Scripts\python -m src.ingest.build                  # -> data/processed/*.parquet, reports/ingest.md
.venv\Scripts\python -m src.baseline.rate                 # baseline stats and raw scores
.venv\Scripts\python -m src.vaep.train                    # VAEP features, models, action values, raw scores
.venv\Scripts\python -m src.scale.apply --method baseline # raw scores -> ratings
.venv\Scripts\python -m src.scale.apply --method vaep
.venv\Scripts\python -m src.pipeline.export_web           # JSON payloads for the page -> web/data/
.venv\Scripts\python -m http.server 8810 --directory web  # open http://localhost:8810
.venv\Scripts\python -m pytest -q                         # the full suite, including the data-backed rule tests
```

Other entry points:

```powershell
.venv\Scripts\python -m src.explain.explain --match <match_id> --player <player_id> --method baseline|vaep|both
.venv\Scripts\python -m src.pipeline.gameweek [--gw 8] [--if-new]      # current season -> web/data/current/
.venv\Scripts\python -m src.pipeline.export_site                        # rebuild the public site in docs/
.venv\Scripts\python -m src.validation.validate                         # regenerate reports/validation.md
```

`export_site` copies the page files and writes the 2015/16 payloads into `docs/`, then
verifies that the folder holds nothing else; `tests/test_site.py` repeats that check on the
committed folder. GitHub Pages serves `docs/` from the `main` branch.

To schedule the gameweek pipeline on Windows, register `scripts\run_gameweek.ps1` with Task
Scheduler (daily, with "run as soon as possible after a missed start"); the script header
explains its exit codes and the log it writes to `logs/pipeline.log`.

## Repository layout

```
src/contract.py     the interface every rating method implements (columns, constants, validate_contract)
src/ingest/         download StatsBomb open data and build the processed parquet files
src/baseline/       stats, weights with justifications, raw scores, report
src/vaep/           splits, xG labels, lightgbm models, action values, report
src/scale/          percentile scaler, the target curve, colour bands
src/explain/        explanation inputs, number verification, template and LLM providers
src/pipeline/       web export, public-site build, FPL gameweek pipeline and stat mapping
src/validation/     the independent verifier behind reports/validation.md
web/                the match page (vanilla HTML/CSS/JS, no build step); web/data/ is generated
docs/               the public GitHub Pages site: the page plus the 2015/16 payloads
tests/              pytest suite; synthetic fixtures only
reports/            baseline, VAEP, scale, validation (with before/after) and FPL mapping reports
scripts/            the scheduled-task wrapper for the gameweek pipeline
```

Raw data, processed data, logs and the current-season output are never committed
(`data/`, `logs/`, `web/data/` are gitignored).

## Data and credits

<a href="https://github.com/statsbomb/open-data"><img src="web/img/statsbomb-logo.png" alt="StatsBomb" width="220"></a>

**Data provided by StatsBomb.** The 2015/16 ratings, the reports and the public page are
built from [StatsBomb Open Data](https://github.com/statsbomb/open-data) (Premier League
2015/16, `competition_id=2`, `season_id=27`), used under the
[StatsBomb Public Data User Agreement](https://github.com/statsbomb/open-data/blob/master/LICENSE.pdf),
which asks anyone publishing work based on the data to state StatsBomb as the source and to
use their logo. The ratings are the output of the two models described above, not StatsBomb's
own ratings.

Current-season data comes from the official
[Fantasy Premier League API](https://fantasy.premierleague.com/api/) and stays local.

VAEP is implemented with [`socceraction`](https://github.com/ML-KULeuven/socceraction)
(Decroos, Bransen, Van Haaren and Davis, "Actions Speak Louder than Goals: Valuing Player
Actions in Soccer", KDD 2019).

# CLAUDE.md — Premier League Player Rating System

Read this before touching anything. `PROMPT.md` is the original brief; this file is the
working contract derived from it. If the two disagree, PROMPT.md wins and this file must
be fixed.

## Goal

Give every Premier League player a **3.0–10.0 rating for every match**, anchored so a
full season averages **6.5** and a **10.0 is extremely rare**. Two methods, compared side
by side:

1. **Baseline** — hand-weighted per-match stats, separate weights per position group.
2. **VAEP** — action values from `socceraction`, summed per player per match.

Then: a match-ratings web page, a grounded "why this rating?" explanation feature, and a
current-season gameweek pipeline (baseline only, from the FPL API).

## Non-negotiable rules

- **Data sources.** Development/validation: StatsBomb Open Data, Premier League 2015/16
  (`competition_id=2`, `season_id=27`). Current season (Step 6 only): the official
  Fantasy Premier League API. Nothing else. No FBref. No SofaScore/WhoScored scraping.
- **Never invent, simulate or fill in data.** Synthetic data is allowed only inside
  `tests/` (fixtures, fakes) and must be labelled synthetic. Nothing synthetic goes into
  `data/` or `web/data/`. If a source fails, stop and report; do not substitute.
- **Credit StatsBomb** in the README and on the web page.
- **Checkpoints.** Stop at every ⏸ CHECKPOINT in PROMPT.md and show results. Never skip.
- **No step is done until its tests ran and the output was shown.**
- **Do not weaken, skip or delete a test to make it pass.** If a test is wrong, say so
  and get approval first. Subagents may add tests in their area; they may not change
  assertions in existing ones.
- **No LLM API key is used without explicit approval** (Step 5 checkpoint).
- **If a library API differs from what you expect, read its source**
  (`.venv/Lib/site-packages/socceraction`, `.../statsbombpy`). Don't guess.

## Environment

- Python **3.11** venv at `.venv`. Reason: `socceraction==1.4.2` pins `pandas<2`,
  `numpy<2`, `pandera<0.14`, `lxml<5`; pandas 1.5.3 has wheels up to CPython 3.11.
  The system Python is 3.14 and must not be used for this project.
- Always run through the venv: `.venv/Scripts/python -m <module>` (Windows paths).
- Install: `.venv/Scripts/python -m pip install -r requirements.txt` (fully pinned).
- Tests: `.venv/Scripts/python -m pytest -q` (everything),
  `-m "not data"` (no processed data needed), `-m data` (data-backed rule tests).
- pandas is **1.5.x**: no pandas-2-only APIs (no `pd.ArrowDtype`, no copy-on-write,
  `DataFrame.map` doesn't exist — use `applymap`), numpy is 1.26.
- VAEP learner: `lightgbm` (installed). xgboost is not installed.
- Determinism: every stochastic step uses `SEED` from `src/contract.py`.
- Tooling quirk: Bash commands longer than ~8,000 characters are truncated on this
  machine. Write long files with the Write tool, not heredocs.

## Repo layout

```
rating-system/
├── CLAUDE.md, PROMPT.md, README.md, requirements.txt, pyproject.toml (pytest config)
├── data/raw/          downloaded StatsBomb JSON (gitignored)
├── data/processed/    parquet + json produced by the steps (gitignored)
├── src/contract.py    column names, constants, validate_contract()  <- single source of truth
├── src/ingest/        Step 2: download + build processed parquet files
├── src/baseline/      hand-weighted stats method
├── src/vaep/          socceraction VAEP method (splits, training, scoring)
├── src/scale/         raw_score -> rating (percentile scaler), colour bands
├── src/explain/       grounded explanations, pluggable LLM provider
├── src/pipeline/      FPL current-season pipeline + web JSON export
├── web/               static single-page app (index.html, app.js, styles.css); web/data/ is generated
├── tests/             pytest suite; tests/fixtures/ holds synthetic samples
└── reports/           validation.md, vaep_splits.md, fpl_mapping.md, ...
```

## Interface contract (every rating method outputs exactly this)

| column           | dtype   | meaning |
|------------------|---------|---------|
| `match_id`       | int64   | StatsBomb `match_id` (FPL fixture id in the current-season pipeline) |
| `player_id`      | int64   | StatsBomb `player_id` (FPL element id in the pipeline) |
| `player_name`    | str     | |
| `team`           | str     | team name |
| `position_group` | str     | one of `GK`, `DEF`, `MID`, `FWD` |
| `minutes`        | float64 | minutes on the pitch **including stoppage time**, 1 decimal |
| `raw_score`      | float64 | finite; higher is better; **only comparable within `position_group`** |

- One row per `(match_id, player_id)`. Include **every player who appeared**
  (`minutes > 0`), including those under 10 minutes; the scale module drops them.
- Extra columns are allowed and ignored by consumers.
- Constants (`CONTRACT_COLUMNS`, `POSITION_GROUPS`, `MIN_MINUTES`, `RATING_MIN/MAX`,
  `TARGET_MEAN`, `MEAN_TOLERANCE`, `MAX_SHARE_OF_TENS`, `SEED`) and
  `validate_contract(df, with_rating=...)` live in `src/contract.py`. Import them;
  never redefine them.
- The scale module adds `rating` (float64, one decimal, 3.0–10.0) and returns **only
  rows with `minutes >= MIN_MINUTES`**.

## Rating rules -> tests

| rule (PROMPT.md)                                              | test |
|---------------------------------------------------------------|------|
| only players with 10+ minutes get a rating                    | `test_scale.py::test_players_under_10_minutes_are_not_rated`, `test_rules_season.py::test_only_players_with_10_plus_minutes_are_rated` |
| raw scores compared within position groups, never across     | `test_scale.py::test_groups_are_scaled_independently`, `::test_same_raw_score_rates_differently_in_different_groups`, `::test_each_position_group_mean_is_6_5`, `test_rules_season.py::test_each_position_group_is_anchored_separately` |
| full-season mean 6.5 ± 0.1                                    | `test_scale.py::test_season_mean_is_6_5`, `test_rules_season.py::test_season_mean_is_6_5` |
| ratings bounded 3.0–10.0                                      | `test_scale.py::test_ratings_bounded_3_to_10`, `test_rules_season.py::test_ratings_bounded_3_to_10` |
| fewer than 0.1% of ratings equal 10.0                         | `test_scale.py::test_tens_are_extremely_rare`, `test_rules_season.py::test_tens_are_extremely_rare` |
| one decimal place                                             | `test_scale.py::test_ratings_have_one_decimal`, `test_rules_season.py::test_ratings_have_one_decimal` |
| 11 rated starters per team unless a starter got < 10 minutes  | `test_rules_season.py::test_eleven_rated_starters_per_team_unless_under_10_minutes` |
| no rating function reads data from another match             | `test_isolation.py` (all tests), `test_scale.py::test_transform_is_row_wise` |
| 380 matches, 20 teams                                         | `test_ingest.py::test_380_matches`, `::test_20_teams` |
| no match scored by a VAEP model trained on it                | `test_vaep.py::test_recorded_splits_exclude_each_scored_match_from_its_training_set` |
| every number in an explanation appears in its input          | `test_explain.py::test_every_number_in_the_explanation_appears_in_the_input` |
| FPL pipeline: checks fail -> no output written               | `test_pipeline.py::test_nothing_is_written_when_checks_fail` |

### The isolation rule, precisely

Two kinds of code exist:

- **Calibration** (may read many matches): training VAEP models, fitting the scaler.
  Both must be split so that no match's raw score comes from a model trained on it
  (VAEP), and the scaler is fit once on a reference set and then frozen.
- **Per-match rating functions** (may read only the match being rated plus frozen
  artefacts): `src.baseline.rate.rate_match`, `src.vaep.rate.rate_match`,
  `PercentileScaler.transform` (row-wise), `src.explain.inputs.explanation_input`.
  Each must select rows with the requested `match_id` from whatever it is given, raise
  `ValueError` if that match is absent, and produce identical output whether or not
  other matches' rows are present.

## Processed data (`data/processed/`, written by `python -m src.ingest.build`)

Raw JSON is downloaded once into `data/raw/statsbomb/` mirroring the open-data repo
layout (`matches/2/27.json`, `events/<match_id>.json`, `lineups/<match_id>.json`) so
that `socceraction.data.statsbomb.StatsBombLoader(getter="local", root=...)` can read it.
`data/raw/statsbomb/MANIFEST.json` records the upstream commit and download time;
`reports/ingest.md` records the build summary, the minutes cross-check and anomalies.

| file | required columns |
|------|------------------|
| `matches.parquet` | `match_id, match_date (YYYY-MM-DD), kick_off, match_week, home_team_id, home_team, away_team_id, away_team, home_score, away_score, home_formation, away_formation, stadium, referee` |
| `teams.parquet` | `team_id, team_name` |
| `players.parquet` | `player_id, player_name, player_full_name, appearances` (`player_name` is StatsBomb's nickname when present, else the full name; the same rule is used everywhere) |
| `lineups.parquet` | one row per **player who appeared** (`minutes > 0`): `match_id, team_id, team, player_id, player_name, player_full_name, jersey_number, is_starter (bool), starting_position_id, position_id, position, position_group, minutes, minute_on, minute_off, red_card_minute (null unless sent off)` |
| `events.parquet` | all 380 matches, statsbombpy-style flattened events (`sb.events()` naming) plus `match_id`; coordinate lists split into `location_x, location_y`, `pass_end_x, pass_end_y`, `carry_end_x, carry_end_y`, `shot_end_x, shot_end_y, shot_end_z`, `goalkeeper_end_x, goalkeeper_end_y`; dict-valued columns dropped (`tactics`, `shot_freeze_frame`) |
| `actions.parquet` | SPADL actions from `socceraction.spadl.statsbomb.convert_to_actions`, socceraction's native names (`game_id, original_event_id, action_id, period_id, time_seconds, team_id, player_id, start_x, start_y, end_x, end_y, type_id, result_id, bodypart_id`) plus `type_name, result_name, bodypart_name` |

Written by later steps (same folder):

| file | producer | content |
|------|----------|---------|
| `baseline_stats.parquet` | baseline | `match_id, player_id, position_group, minutes` + one column per stat in `STATS` |
| `raw_baseline.parquet` | baseline | contract |
| `vaep_splits.json`, `models/vaep_*.pkl` | vaep | splits (see below) and fitted models |
| `action_values.parquet` | vaep | `game_id, action_id, player_id, team_id, period_id, time_seconds, type_name, result_name, offensive_value, defensive_value, vaep_value, fold_id` |
| `raw_vaep_holdout.parquet` | vaep | contract, holdout score matches only |
| `raw_vaep.parquet` | vaep | contract, full season (cross-fitted) |
| `scaler_<method>.json` | scale | frozen `PercentileScaler` |
| `ratings_<method>.parquet` | scale | contract + `rating`, rated rows only |

### Position groups (StatsBomb `position_id` -> group)

| group | position ids (names) |
|-------|----------------------|
| GK  | 1 Goalkeeper |
| DEF | 2 Right Back, 3 Right Center Back, 4 Center Back, 5 Left Center Back, 6 Left Back, 7 Right Wing Back, 8 Left Wing Back |
| MID | 9 Right Defensive Midfield, 10 Center Defensive Midfield, 11 Left Defensive Midfield, 12 Right Midfield, 13 Right Center Midfield, 14 Center Midfield, 15 Left Center Midfield, 16 Left Midfield, 18 Right Attacking Midfield, 19 Center Attacking Midfield, 20 Left Attacking Midfield |
| FWD | 17 Right Wing, 21 Left Wing, 22 Right Center Forward, 23 Center Forward, 24 Left Center Forward, 25 Secondary Striker |

A player's `position_group` for a match is the group of the position in which they
spent the **most minutes in that match** (from the lineup `positions` intervals).
Wingers are forwards; attacking midfielders are midfielders. Keep this table in
`src/ingest/positions.py` and nowhere else.

### Minutes

`minutes` = total time on the pitch including stoppage time, derived from the **events**:
`Starting XI` (on at kick-off), `Substitution` (one off, the replacement on),
`Player Off` / `Player On` (treatment breaks; `permanent` means no return) and a Red Card
or Second Yellow (off for good). Period lengths come from the `Half End` timestamps.
The lineup file's `positions` intervals are used only to pick the position a player
spent the most time in (clipped to the on-pitch spans): they do not stop at a dismissal,
and near half-time substitutions the last interval often runs to the final whistle.
`reports/ingest.md` lists every such correction (65 in 2015/16) and a cross-check
against socceraction's `minutes_played`, which is whole-minute, counts a half-time
substitution as 45 minutes and overwrites substitutes who later go off; ours is the
reference. Rounded to 1 decimal (minimum 0.1 for anyone who stepped on the pitch).
Eligible for a rating iff `minutes >= MIN_MINUTES` (10.0).

## Module APIs (build against these; they are what the tests import)

### `src/baseline/`
- `weights.py`: `STATS: list[str]` (25–35 names, unique), `Weight(weight: float, why: str)`
  (dataclass/namedtuple; `why` is one line, >= 15 chars), `WEIGHTS: dict[group, dict[stat, Weight]]`
  with an entry for every group in `POSITION_GROUPS` (at least 8 stats each); weight
  vectors differ between groups. Document the stat definitions in the module docstring.
  `LABELS[stat]` gives the fan-facing label; `weight_of(group, stat)` returns 0.0 for stats
  outside a group's profile (they are absent from `WEIGHTS[group]`, so iterate
  `WEIGHTS[group]`, not `STATS`, when listing a player's contributions).
- `stats.py`: `player_match_stats(match_id, events, lineups) -> DataFrame` with
  `match_id, player_id, position_group, minutes` + one column per stat, one row per
  player in that match's lineups (zeros for players with no events).
- `rate.py`: `rate_match(match_id, events, lineups) -> DataFrame` (contract;
  `raw_score = sum(WEIGHTS[group][stat] * stat)`), `rate_season(processed_dir) -> DataFrame`,
  CLI `python -m src.baseline.rate` writes `baseline_stats.parquet` and `raw_baseline.parquet`.
- Known pitfall: substitutes with few minutes have few actions; decide explicitly how
  the raw score treats low activity and document it in `weights.py`.
- After the Step 4 validation (2026-10-31): `goals` counts non-penalty goals only and
  `penalties_scored` (weight 1.25 in every group) counts penalty goals; `key_passes`
  excludes assists; `fouls_won` was retired to stay within 35 stats; DEF/MID team-context
  weights were halved. Anything summing a player's goals must use `goals + penalties_scored`.

### `src/vaep/`
- `splits.py`: `Split(name: str, train: list[int], score: list[int])`;
  `make_cv_splits(match_ids, n_folds=5, seed=SEED) -> list[Split]` (each match scored in
  exactly one fold, never in its own training set, deterministic);
  `make_holdout_split(matches: DataFrame) -> Split` (train `match_week <= 19`, score the rest).
- `train.py`: builds SPADL game states/features/labels with `socceraction.vaep`
  (`nb_prev_actions=3`, lightgbm learner), fits one model per split, writes
  `vaep_splits.json` = `{"holdout": {"train": [...], "score": [...]}, "folds": [{"fold_id": 0, "train": [...], "score": [...]}, ...]}`,
  `action_values.parquet` (full season, cross-fitted, `fold_id` per game), the models,
  `raw_vaep_holdout.parquet` and `raw_vaep.parquet`. CLI `python -m src.vaep.train`.
  Document the splits in `reports/vaep_splits.md`. Features and labels are cached in
  `vaep_features.parquet` / `vaep_labels.parquet`; settings and held-out AUC/Brier per split
  are in `models/vaep_metrics.json`; `python -m src.vaep.report` regenerates the report.
- `rate.py`: `load_model_for_match(match_id)` (the fold model whose training set
  excludes that match), `rate_match(match_id, actions, lineups, model) -> DataFrame`
  (contract; `raw_score` = sum of `vaep_value` over the player's actions in that match;
  players in the lineup with no actions get 0.0).

### `src/scale/`
- `scaler.py`: `class PercentileScaler` with `fit(df) -> self`, `transform(df) -> DataFrame`
  (input columns + `rating`; rows with `minutes < MIN_MINUTES` dropped), `fit_transform`,
  `save(path)`, `PercentileScaler.load(path)`, `to_dict()/from_dict()`.
  Behaviour: per `position_group`, `fit` stores a reference quantile table of `raw_score`
  (rated rows only; fit validates the contract and raises `ValueError` otherwise);
  `transform` is **row-wise** (a row's rating depends only on its own `raw_score`, its
  group's frozen table and the fixed target curve). Equal raw scores -> equal ratings.
  Raw scores beyond the reference range clip to 3.0 / 10.0. Unknown group ->
  `ValueError`. Output rounded to `RATING_DECIMALS`.
- `curve.py`: the fixed percentile -> rating target curve, documented with its
  rationale. It must give mean 6.5 ± 0.1 overall **and per group**, 3.0 at p = 0,
  10.0 only above roughly p = 0.9995 (so < 0.1% of ratings are tens), monotone.
- `bands.py`: `rating_band(rating) -> "red" | "orange" | "green" | "blue" | "unrated"`
  (red < 6.0, orange 6.0–6.9, green 7.0–7.9, blue >= 8.0, `None` -> unrated).
- CLI `python -m src.scale.apply --method baseline|vaep` fits on `raw_<method>.parquet`,
  writes `scaler_<method>.json` and `ratings_<method>.parquet`. The ratings file keeps any extra
  input columns; a saved scaler embeds the curve anchors and refuses to load under a different
  curve, so re-run `apply` after editing the curve.

### `src/explain/`
- `inputs.py`: `explanation_input(match_id, player_id, method, k=5) -> dict`:
  ```
  {"method", "match_id", "player_id", "player_name", "team", "opponent", "position_group",
   "minutes", "raw_score", "rating",
   "top_positive": [item, ...],   # <= k, descending value, all value > 0
   "top_negative": [item, ...]}   # <= k, ascending value, all value < 0
  item (vaep):     {"action_id", "label", "minute", "value"}        # label = type (+ result)
  item (baseline): {"stat", "label", "count", "weight", "value"}    # value = count * weight
  ```
  Uses only that player's rows from `action_values.parquet` / `baseline_stats.parquet`
  for that match (per-match function: the isolation rule applies).
- `verify.py`: `numbers_in_text(text) -> list[float]` (in order of appearance),
  `verify_numbers(text, data) -> bool` (every number in the text equals some numeric
  leaf of `data`, allowing rounding to 0–2 decimals; text without numbers passes).
  Used by tests and at runtime.
- `llm.py`: `get_provider()` reads `RATING_LLM_PROVIDER` (`template` default ->
  `TemplateProvider`, deterministic, offline; `anthropic` -> `AnthropicProvider`, which
  requires `ANTHROPIC_API_KEY` and raises an error naming that variable when missing;
  anything else -> `ValueError`). `RATING_LLM_MODEL` overrides the model (default
  `claude-opus-5-5`; the `anthropic` package is not in the pinned environment and is imported
  lazily). The LLM prompt forbids mentioning anything absent from the input, and the output is
  checked with `verify_numbers` before it is shown (template fallback on failure).
- `explain.py`: `explain(match_id, player_id, method) -> Explanation(text, input, provider, verified)`.
  If the provider's text fails `verify_numbers`, fall back to the template text and set
  `verified=False`; the template text itself always verifies.

### `src/pipeline/`
- `fpl.py`: `FplClient` protocol — `bootstrap() -> dict` (`/api/bootstrap-static/`),
  `fixtures(gw) -> list[dict]` (`/api/fixtures/?event=<gw>`), `live(gw) -> dict`
  (`/api/event/<gw>/live/`); `HttpFplClient` implements it;
  `latest_finished_gameweek(bootstrap) -> int` (highest event with `finished` and
  `data_checked` true). Tests use a fake client (`tests/test_pipeline.py`) whose payload
  shapes mirror the real API; the pipeline agent may add stat fields to the fake but
  not change assertions.
- `fpl_mapping.py`: `FPL_POSITION_GROUPS = {1: "GK", 2: "DEF", 3: "MID", 4: "FWD"}`;
  `FPL_STAT_MAP: dict[baseline_stat, fpl_field]`; `UNAVAILABLE_STATS: dict[baseline_stat, note]`
  (`FPL_STAT_MAP.keys() | UNAVAILABLE_STATS.keys() == set(STATS)`, disjoint);
  `FPL_WEIGHTS: dict[group, dict[stat, Weight]]` using only mapped stats, with the
  adjustments explained in `reports/fpl_mapping.md`.
- `gameweek.py`: `run_checks(ratings, fixtures, teams) -> list[str]` (empty = pass).
  `ratings` is the gameweek frame: contract + `rating`, `team_id`, `is_starter`
  (`match_id` = FPL fixture id); `fixtures` and `teams` are the FPL API dicts. Checks:
  all 20 teams present (a failure message names the missing team), every fixture has
  both teams with their starters rated unless < 10 minutes, ratings finite, within
  bounds, one decimal, gameweek mean within 6.5 ± 0.3 (single-gameweek tolerance,
  documented). `run(client, out_dir, gw=None) -> RunResult(gameweek, ok, failures, written)`:
  fits the scaler on all finished gameweeks of the season so far, rates the target
  gameweek (latest finished when `gw` is None), runs the checks, and **writes nothing
  if any check fails**; on success writes `<out_dir>/index.json` (one entry per
  fixture) and one payload per fixture. CLI `python -m src.pipeline.gameweek` exits 1
  with the failures printed.
- `export_web.py`: `match_payload(match_id) -> dict` (2015/16, both methods),
  `validate_payload(payload)` (raises `ValueError`), `export_season(out_dir=web/data)`
  writing `web/data/index.json` and one JSON per match.

### Web payload (what `web/` loads; `tests/fixtures/sample_match.json` is a synthetic example)

```
<out_dir>/index.json : [{"season", "match_id", "match_date", "matchweek", "home", "away",
                         "home_score", "away_score", "methods", "path"}, ...]
<out_dir>/<season>/<match_id>.json :
{"schema_version": 1, "source": "StatsBomb Open Data" | "Fantasy Premier League API",
 "season", "competition", "match_id", "match_date", "matchweek",
 "home": {"team_id", "team", "score", "formation"}, "away": {...},
 "methods": ["baseline", "vaep"] | ["baseline"],
 "players": [{"player_id", "player_name", "team_id", "team", "side": "home"|"away",
              "position_id", "position", "position_group", "is_starter", "jersey_number",
              "minutes", "ratings": {method: rating | null}, "raw_scores": {method: raw | null}}],
 "credits": "Data: StatsBomb Open Data (https://github.com/statsbomb/open-data)"}
```
Rating is `null` exactly when `minutes < 10`. Colours: red < 6.0, orange 6.0–6.9,
green 7.0–7.9, blue >= 8.0, grey = unrated. A toggle switches between methods. In VAEP
mode goalkeepers display their baseline rating with a "B" tag and a note, because VAEP
does not value shot-stopping (validation finding A); payloads are unchanged.
`validate_payload` (src/pipeline/export_web.py) also requires `schema_version == 1`,
`raw_scores` keys equal to `methods`, numeric non-negative `minutes`, boolean `is_starter`
and 11 starters per side; per-player `minute_on` / `minute_off` are optional (the page shows
a substitute's entry minute when present). A single-method payload (`["baseline"]`) is valid;
the FPL pipeline must produce exactly this shape. `python -m src.pipeline.export_web` writes
`web/data/` for 2015/16; serve with `python -m http.server 8810 --directory web`.
Regenerate the sample with `python tests/fixtures/make_sample_match.py`. Serve with
`.venv/Scripts/python -m http.server 8810 --directory web`.

## Conventions

- Parquet via pyarrow; ids are int64; names are plain `str`; sort outputs by
  `match_id, team, player_id` before writing.
- Pure functions for per-match work; CLIs (`python -m src.<area>.<module>`) only glue
  I/O together and print a short summary at the end.
- `logging` for progress, never bare prints inside library code.
- Tests make **no network calls** and **no LLM calls** unless marked `network` / `llm`.
  Data-backed tests are marked `data` and **fail (not skip)** when the data is missing.
- Git: `main` is the integration branch. Step 3 agents work on `feat/baseline`,
  `feat/vaep`, `feat/scale`, `feat/web`, stay inside their own `src/<area>/`,
  `tests/test_<area>*.py` and (web) `web/`, and never edit `src/contract.py`, this
  file or other areas' tests — ask the orchestrator instead. Merge into `main` only
  after that agent's tests pass. Commit messages: `area: imperative summary`.
- Reports go in `reports/` as Markdown with the numbers that justify each claim.

## Checkpoints (from PROMPT.md)

1. ⏸ after Step 1: show CLAUDE.md and the list of tests.
2. ⏸ after Step 4: show `reports/validation.md`; weights may be adjusted.
3. ⏸ after Step 6: show the ratings for one current match; ask about scheduling.
Also: ask before any API key is used (Step 5).

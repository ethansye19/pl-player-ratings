# Baseline model on the Fantasy Premier League API

Current-season ratings use the baseline method only: the FPL API has per-player, per-gameweek totals and no event data, so VAEP cannot run.

## Endpoints

| endpoint | used for |
|---|---|
| `GET /api/bootstrap-static/` | gameweeks (`events`: `finished`, `data_checked`), `teams`, `elements` (players: team, element_type, names, squad number) |
| `GET /api/fixtures/?event=<gw>` | the gameweek's fixtures with team ids, scores, kickoff time and `finished` |
| `GET /api/event/<gw>/live/` | every player's stats for the gameweek (one total per player) plus `explain`, which names the fixture(s) he was involved in; it itemises only stats that earned points, so it is not used for numbers |
| `GET /api/element-summary/<id>/` | one player's `history`, a row per fixture (starts, minutes, every mapped stat, `opponent_team`, `was_home`); fetched only for players who need it (below) |

The latest finished gameweek is the highest `event` with both `finished` and `data_checked` true. Responses are mirrored under `data/raw/fpl/` on every run (`element-summary/<id>.json` for the histories). Requests go at most one per second, with a generic User-Agent (`rating-system/0.2 (football player ratings)`), three attempts with back-off, and `Retry-After` honoured on HTTP 429.

## One row per player per fixture

The live feed has one total per player per gameweek and only knows the player's current club. That is enough when he played exactly one fixture of the gameweek for the club he had already joined (`team_join_date` before the kickoff): the totals are that fixture's numbers. Every other player who played takes the gameweek's rows of his element-summary `history` instead:

- **two fixtures in the gameweek** (a double gameweek): starts, minutes, saves, goals conceded, tackles, recoveries, CBI, xG, xG conceded and every other mapped stat come from his row for each fixture, so a player who started one fixture and came off the bench in the other is a starter in one and a substitute in the other, and saves or goals conceded that earned no FPL points still count;
- **a club that may have changed**: his current club did not play the fixture, he joined it on or after the fixture, or bootstrap has no `team_join_date` for him. His club for the fixture is the home side if `was_home`, else the away side, and `opponent_team` must be the other side.

The history rows must add up to his gameweek totals (exactly for counts, to 0.015 for the 2-decimal xG fields). A missing response, row or field, an `opponent_team` that does not match, or totals that do not add up is a check failure: nothing is written. Nothing is split between fixtures, estimated from the gameweek total or read as 0, and the same applies to a field missing from the live feed. Each player's history is fetched at most once per run.

## Stat mapping

| baseline stat | FPL field | GK | DEF | MID | FWD | note |
|---|---|---|---|---|---|---|
| goals | `goals_scored` | 4 | 3.4 | 2.85 | 2.85 | includes penalties; weight blended (baseline 4/3.5/3/3) |
| penalties_scored | **not in FPL** | - | - | - | - | FPL reports goals_scored with penalties included and has no penalties-scored field; the goal weight is blended instead |
| assists | `assists` | 2 | 2 | 2 | 1.8 |  |
| xg | `expected_goals` | 1 | 1 | 1.5 | 1.5 |  |
| shots_on_target | **not in FPL** | - | - | - | - | FPL publishes no shot counts (threat is an index, not a count) |
| key_passes | **not in FPL** | - | - | - | - | FPL publishes no key-pass count; expected_assists is a different quantity and not a baseline stat |
| dribbles_completed | **not in FPL** | - | - | - | - | FPL publishes no dribble or take-on counts |
| progressive_carries | **not in FPL** | - | - | - | - | FPL publishes no carry data |
| penalties_won | **not in FPL** | - | - | - | - | FPL does not record who won a penalty |
| penalties_missed | `penalties_missed` | - | -2 | -2 | -2 |  |
| passes_completed | **not in FPL** | - | - | - | - | FPL publishes no passing counts |
| passes_incomplete | **not in FPL** | - | - | - | - | FPL publishes no passing counts |
| progressive_passes | **not in FPL** | - | - | - | - | FPL publishes no passing counts or locations |
| possession_lost | **not in FPL** | - | - | - | - | FPL publishes no dispossession or miscontrol counts |
| tackles_won | `tackles` | 0.2 | 0.3 | 0.3 | 0.2 |  |
| dribbled_past | **not in FPL** | - | - | - | - | FPL publishes no defensive-duel outcomes |
| interceptions | **not in FPL** | - | - | - | - | counted inside clearances_blocks_interceptions, which is mapped through 'clearances' with a blended weight |
| clearances | `clearances_blocks_interceptions` | 0.06 | 0.15 | 0.15 | 0.09 | FPL combines clearances, blocks and interceptions; blended weight (baseline clearances 0.1/0.1/0.1/0.05) |
| blocks | **not in FPL** | - | - | - | - | counted inside clearances_blocks_interceptions, which is mapped through 'clearances' with a blended weight |
| ball_recoveries | `recoveries` | - | 0.06 | 0.06 | 0.05 |  |
| aerials_won | **not in FPL** | - | - | - | - | FPL publishes no aerial-duel counts |
| aerials_lost | **not in FPL** | - | - | - | - | FPL publishes no aerial-duel counts |
| pressures | **not in FPL** | - | - | - | - | FPL publishes no pressure events |
| fouls_committed | **not in FPL** | - | - | - | - | FPL publishes no foul counts (only cards) |
| yellow_cards | `yellow_cards` | -0.5 | -0.5 | -0.5 | -0.5 |  |
| red_cards | `red_cards` | -3 | -3 | -3 | -3 |  |
| errors | **not in FPL** | - | - | - | - | FPL publishes no error-leading-to-shot events |
| own_goals | `own_goals` | -3 | -3 | -3 | -3 |  |
| penalties_conceded | **not in FPL** | - | - | - | - | FPL does not record who conceded a penalty |
| goals_conceded | `goals_conceded` | -1 | -0.4 | -0.3 | -0.2 |  |
| clean_sheet | `clean_sheets` | 1.5 | 0.5 | 0.25 | - |  |
| xg_faced | `expected_goals_conceded` | 0.8 | -0.2 | -0.1 | - |  |
| saves | `saves` | 0.5 | - | - | - |  |
| penalties_saved | `penalties_saved` | 2 | - | - | - |  |
| keeper_claims | **not in FPL** | - | - | - | - | FPL publishes no claims, punches or sweeps for goalkeepers |

15 of 35 baseline stats are available (two of them, blocks and interceptions, inside the combined CBI field); 20 are not.

## What the missing stats mean

- The FPL baseline has far less resolution for midfielders and forwards (no passing, dribbling, shots, key passes, duels); their raw scores rest on goals, assists, xG, tackles, CBI, recoveries, cards and team context.
- The percentile scaler is fit on all finished gameweeks of the season so far, within position group, so every group is re-anchored at 6.5; the single-gameweek validity check allows a mean of 6.5 +/- 0.3.
- Positions follow FPL's `element_type`, so wingers are usually MID.

## Validity checks (run before anything is written)

every player's per-fixture numbers are complete and his element-summary rows, where used, add up to his gameweek totals (see above); every team with a fixture has rated players (teams without a fixture are not required, so a blank gameweek publishes); each side of every fixture has exactly 11 starters in the FPL data, and every starter with 10+ minutes is rated (the same 11-starters rule `validate_payload` enforces, checked before any payload is built); ratings finite, within 3.0-10.0, one decimal; no rated row under 10 minutes; gameweek mean within 6.5 +/- 0.3. If any check fails nothing is written and the CLI exits 1.

## Replay of gameweeks 1-8 with per-fixture rows (2026-11-03)

The mirrored responses for gameweeks 1-8 were replayed offline through the code before and after the change, with the same inputs (five element-summary responses fetched once and mirrored). Gameweeks 1-8 have no double gameweek; five players changed club on 31 August or 1 September after playing for their old one.

| gameweek | before: checks | after: checks | rows with a changed club | ratings changed |
|---|---|---|---|---|
| 1 | failed: Everton and Crystal Palace had 10 starters in fixture 3 | passed | 3 rated (Ndiaye to Everton, Muñoz to Crystal Palace, Enzo to Chelsea) and Grealish (8 min, unrated) to Man City | 0 |
| 2 | failed: Everton had 10 starters in fixture 12 | passed | 1 rated (Ndiaye to Everton) and Iroegbunam (4 min, unrated) to Everton | 0 |
| 3-8 | passed | passed | 0 | 0 |

No rating changed: a rating depends on the player's stats and position group, not his club, and these players' stats are the same in both sources. Gameweek 8 is identical before and after (279 rated, mean 6.58, the numbers below).

## Latest run: gameweek 8 (2026/27)

- fixtures: 10; rated player-fixtures: 279; mean rating 6.58; by group DEF 6.70, FWD 6.63, GK 6.56, MID 6.49
- ratings of 10.0: 0; below 5.0: 4; min 3.7, max 9.0

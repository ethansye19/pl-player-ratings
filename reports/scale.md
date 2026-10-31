# Scale module: raw score to a 3.0-10.0 rating

**Status.** Built and tested against the **SYNTHETIC** season from
`tests/conftest.py::make_fake_season()` (seed 7): 380 matches x 2 teams x 14 players =
10,640 raw rows, of which 10,141 have 10 or more minutes and are rated. Sections 2 to 5
describe that synthetic data pushed through the fixed curve, not real ratings; section 7
summarises a read-only rehearsal on the preliminary real raw files that appeared in the shared
`data/processed` while this module was being built. Step 4 re-measures on the final real
ratings. Nothing synthetic is written to `data/`, and this module wrote nothing to `data/` at
all (the CLI test and the rehearsal use temporary folders).

Every table in this report is generated from the code and checked verbatim by
`tests/test_scale_report.py`, so it cannot drift from the implementation.

## 1. What was built

| file | role |
|------|------|
| `src/scale/curve.py` | the fixed percentile-to-rating curve: 46 anchors joined by straight lines (`numpy.interp`) |
| `src/scale/scaler.py` | `PercentileScaler`: `fit`, `transform`, `fit_transform`, `to_dict`/`from_dict`, `save`/`load` (JSON, no pickle) |
| `src/scale/apply.py` | CLI `python -m src.scale.apply --method baseline\|vaep [--processed data/processed]` |
| `src/scale/summary.py` | plain-text summaries shared by the CLI and this report |
| `src/scale/bands.py` | colour bands (already existed, untouched) |

Tests: `test_scale.py` (the 17 gate tests), `test_scale_curve.py`, `test_scale_scaler.py`,
`test_scale_cli.py`, `test_scale_summary.py`, `test_scale_report.py`, plus
`test_web.py::test_rating_band_thresholds` and `test_contract.py`.

## 2. The target curve

A raw score becomes a percentile `p` within its position group (section 4), and `p`
becomes a rating through this table (linear between anchors, rounded to one decimal
afterwards):

| percentile p | curve(p) |
|-------------:|---------:|
| 0.0000 | 3.000 |
| 0.0001 | 3.349 |
| 0.0003 | 3.633 |
| 0.0010 | 3.956 |
| 0.0030 | 4.267 |
| 0.0060 | 4.475 |
| 0.0100 | 4.636 |
| 0.0150 | 4.770 |
| 0.0200 | 4.868 |
| 0.0250 | 4.951 |
| 0.0375 | 5.094 |
| 0.0500 | 5.203 |
| 0.0750 | 5.373 |
| 0.1000 | 5.497 |
| 0.1250 | 5.595 |
| 0.1500 | 5.687 |
| 0.1750 | 5.775 |
| 0.2000 | 5.853 |
| 0.2250 | 5.922 |
| 0.2500 | 5.986 |
| 0.3000 | 6.098 |
| 0.3500 | 6.196 |
| 0.4000 | 6.285 |
| 0.4500 | 6.370 |
| 0.5000 | 6.453 |
| 0.5500 | 6.537 |
| 0.6000 | 6.625 |
| 0.6500 | 6.721 |
| 0.7000 | 6.828 |
| 0.7500 | 6.954 |
| 0.7750 | 7.025 |
| 0.8000 | 7.105 |
| 0.8250 | 7.195 |
| 0.8500 | 7.298 |
| 0.8750 | 7.418 |
| 0.9000 | 7.574 |
| 0.9200 | 7.754 |
| 0.9400 | 7.951 |
| 0.9550 | 8.100 |
| 0.9700 | 8.270 |
| 0.9800 | 8.421 |
| 0.9900 | 8.679 |
| 0.9950 | 8.951 |
| 0.9985 | 9.424 |
| 0.9996 | 9.951 |
| 1.0000 | 10.000 |

What the table means for the rounded ratings, when percentiles are spread uniformly (which is
what a fitted scaler produces on its own reference rows):

| statistic | value |
|-----------|------:|
| mean rating | 6.500 |
| median rating | 6.5 |
| most common rating | 6.4 (6.0% of ratings) |
| standard deviation | 0.83 |
| 6.0 to 7.5 | 66.0% |
| 7.0 or better | 25.2% |
| 8.0 or better | 6.0% |
| 9.0 or better | 0.502% |
| 10.0 | 0.040% |
| below 5.0 | 2.5% |
| below 4.0 | 0.099% |
| red band (below 6.0) | 23.6% |
| orange band (6.0 to 6.9) | 51.2% |
| green band (7.0 to 7.9) | 19.1% |
| blue band (8.0 or better) | 6.0% |

**Shape.** Most ratings (66%) fall between 6.0 and 7.5; 8.0 or better is the top 6%, 9.0 or
better the top 0.5%, and 10.0 only the top 0.04% (`p >= 0.9996`, about 4 ratings in
10,000). Below 5.0 is the worst 2.5%, below 4.0 about 1 in 1,000, and 3.0 is reached only
by `p = 0` or a raw score below every reference score. The curve is right-skewed: the long
upper tail lifts the mean (6.500) slightly above the median (6.5, curve value 6.453). The
rating histogram is unimodal at 0.1 resolution, with every rating from 3.0 to 10.0
reachable. A linear map `3 + 7p` would have made a 3.0 as common as a 6.5 and one rating in
seven a 9 or better.

**Four design points** carry the shape promises: `(0.025, 4.951)`, `(0.94, 7.951)`,
`(0.995, 8.951)` and `(0.9996, 9.951)`. They sit at x.x51 rather than x.x5 on purpose: a
rating of 5.0 needs the curve to reach 4.95, and landing exactly on a rounding boundary would
let floating point decide between 4.9 and 5.0 for a row whose percentile equals the anchor.

**Mean.** The mean is the point of the exercise. The scaler fits and applies on the same
reference rows, whose mid-rank percentiles are `(i + 0.5) / n`, so the reference season's mean
rating is the mean of `round(curve(p), 1)` over those percentiles. The anchors were tuned so
that this mean is 6.5000 +/- 0.0001 on a fine uniform grid. For a group of `n` reference rows
it is within 0.001 of 6.5 for every `n` from 750 to 12,000 (each `n` is checked by
`test_scale_curve.py`), within 0.002 from 300 rows and within 0.003 from 200 rows. That is
well inside the requested 6.5 +/- 0.02 per group.

**Derivation** (the anchors themselves are frozen literals; do not nudge single entries).
A Johnson SU quantile function `Q(z) = xi + lambda * sinh((z - gamma) / delta)`, with
`z = Phi^-1(p)` and `(xi, lambda, gamma, delta) = (5.6566, 3.1274, -1.0068, 3.9009)`, was solved
so that `Q(0.03) = 4.95`, `Q(0.95) = 7.95`, `Q(0.995) = 8.95` and the mean is 6.5. Its core
(`|z| < 1.645`) was then tightened by 20% around the median with the smooth window
`(1 - (z / 1.645)^2)^2`, and the percentile axis was eased slightly (below 0.15 and around
0.94) so that the two share targets sit mid-range: 2.5% below 5.0 and 6% at 8.0 or better.
The result was sampled at the anchor percentiles, rounded to **three** decimals (with two, the
0.1-wide histogram jitters by 10 to 20% from one rating to the next), the design anchors set
to x.x51, and the centre shifted by -0.02 so that the rounded mean is exactly 6.5. Anchors are
denser (down to every 0.025) where the curve bends most, which keeps the piecewise-linear histogram
within about 12% of the smooth one at every rating that holds at least 0.4% of the rows.

## 3. Rating distribution on the synthetic season

`make_fake_season()` gives each position group a deliberately different raw distribution
(GK normal(0, 1), DEF normal(50, 20), MID lognormal, FWD exponential minus 1) and sets 3% of all
rows to exactly 0.0 raw score, so any cross-group comparison or tie mishandling shows up as a
group mean far from 6.5. It contains 499 appearances under 10 minutes, which are dropped.
Rated rows: 10,141 (GK 760, DEF 3,646, MID 3,598, FWD 2,137).

```
rating    count   share
3.0-3.4       1   0.01%  #
3.5-3.9       6   0.06%  #
4.0-4.4      30   0.30%  #
4.5-4.9     275   2.71%  ####
5.0-5.4     606   5.98%  #########
5.5-5.9    1474  14.54%  ######################
6.0-6.4    2665  26.28%  ########################################
6.5-6.9    2533  24.98%  ######################################
7.0-7.4    1336  13.17%  ####################
7.5-7.9     606   5.98%  #########
8.0-8.4     418   4.12%  ######
8.5-8.9     140   1.38%  ##
9.0-9.4      37   0.36%  #
9.5-9.9      11   0.11%  #
10.0          3   0.03%  #
```

Per position group:

| group | rated rows | mean rating | 10.0s | min | max |
|-------|-----------:|------------:|------:|----:|----:|
| GK | 760 | 6.499 | 0 | 3.8 | 9.8 |
| DEF | 3646 | 6.501 | 1 | 3.4 | 10.0 |
| MID | 3598 | 6.504 | 1 | 4.8 | 10.0 |
| FWD | 2137 | 6.500 | 1 | 3.5 | 10.0 |
| all | 10141 | 6.502 | 3 | 3.4 | 10.0 |

What the CLI prints for this season (`--method test`, the synthetic raw file written to a
temporary folder):

```
scale[test]
  rows rated     : 10141 of 10640 raw rows (minutes >= 10)
  mean rating    : 6.502 (target 6.5 +/- 0.1)
  mean by group  : GK 6.499 (n=760) | DEF 6.501 (n=3646) | MID 6.504 (n=3598) | FWD 6.500 (n=2137)
  share of 10.0  : 0.030% (3 of 10141 rows; limit < 0.1%)
  min / max      : 3.4 / 10.0
```

Rules checked on this season (`tests/test_scale.py`, all passing): only players with 10 or
more minutes are rated, and all of them are; ratings are bounded 3.0 to 10.0 (here 3.4 to
10.0); the season mean and every position-group mean are within 6.5 +/- 0.1 (6.499 to 6.504);
3 of 10,141 ratings (0.03%) are 10.0, against a limit of 0.1%; every rating has one decimal;
rating is monotone in raw score within a group and equal raw scores get equal ratings;
shifting one group's raw scores by 1,000 changes no rating; and transforming one match alone
reproduces its rows from the whole-season transform.

Two features of the synthetic data are visible in the table and are not defects. The lowest
MID rating is 4.8, not 3.x: MID raw scores are lognormal (all positive), so the injected rows at
exactly 0.0 are the lowest MID scores. They form a block of about 3% of the group, which
mid-rank places at `p` of about 0.015 (rating 4.8), and that block lifts the MID mean by about
0.004. And the lowest rating a reference row can get is `curve(0.5 / n)` (3.0 is reserved for
scores below every reference score): 3.8 for the 760 GK rows, 3.4 for the 3,646 DEF rows.

## 4. Design decisions

1. **Percentile within position group, then a fixed curve.** Raw scores are only comparable
   inside a position group (a goalkeeper's raw score is on another scale from a forward's), so
   each group has its own reference table and is anchored to the same mean separately. A
   percentile makes no assumption about the shape of the raw distribution (the synthetic groups
   are normal, lognormal and exponential and all land on 6.5), and the curve, not the data,
   decides the mean and the tails.
2. **Mid-rank ties.** `p = (count_below + 0.5 * count_equal) / n`, the "mean" kind of
   `scipy.stats.percentileofscore`. Equal raw scores get equal percentiles and therefore equal
   ratings, and the reference rows' percentiles are spread evenly over (0, 1) on average, which
   is what makes the reference mean equal the curve's mean. Lower or upper ranks would bias the
   mean, and random tie-breaking would give equal scores different ratings.
3. **Interpolation and clipping.** For a raw score strictly between two reference scores the
   percentile is interpolated linearly, so unseen raw scores get continuous percentiles. Below
   the lowest reference score the rating is exactly 3.0 and above the highest exactly 10.0: the
   scale never extrapolates. A fitted row itself is never clipped (the extreme reference rows sit
   at `0.5 / n` and `1 - 0.5 / n`); clipping only matters for data outside the reference.
4. **One decimal, rounded after the curve.** `rating = round(curve(p), 1)`. The anchors are
   tuned on the rounded ratings, so the mean the tests see is the mean of what is published.
5. **Clipping to [3.0, 10.0].** A final `clip` to `RATING_MIN` and `RATING_MAX` is defensive:
   the curve is already bounded (and validated at import), so it can only matter if the anchors
   are edited wrongly.
6. **Row-wise transform, and why it matters for the isolation rule.** A row's rating depends on
   three things only: its own `raw_score`, its own `position_group`, and the frozen artefacts
   (reference table and fixed curve). `transform` never ranks a row against the other rows of
   the frame it receives, so rating one match alone, or mixed with any number of other
   matches, gives identical numbers (`test_transform_is_row_wise`,
   `test_transform_is_permutation_equivariant`, and the data-backed
   `test_isolation.py::test_scaler_transform_of_one_match_matches_the_season_ratings`). The
   calibration step (`fit`, and `apply` which calls it) is the only code that reads many matches,
   and it runs once on the reference set. If `transform` ranked rows within the frame it was
   given, a rating would change with whoever else played that day.
7. **Frozen, exact, refuses to go stale.** The saved table holds the distinct reference scores
   and their counts per group (not a coarse quantile grid), written as JSON floats with `repr`
   precision, so a reload reproduces every rating bit for bit and ties survive. The curve
   anchors are embedded and compared on load: a scaler saved under a different curve is
   refused with an instruction to re-run `apply`, rather than silently disagreeing with the
   ratings file next to it. Files are written atomically.
8. **Rows under 10 minutes are neither fitted on nor rated.** The 10-minute boundary is
   inclusive (10.0 is rated, 9.9 is not). The reference set must be exactly the set of rated
   rows: cameo appearances have raw scores clustered near zero, and including them would put a
   large block at the bottom of every group, push every real performance's percentile up, and
   move the rated mean away from 6.5.
9. **Strict `fit`, lenient `transform` on rows it drops.** `fit` validates the full interface
   contract (a violation is a `ValueError`) because it defines the artefact. `transform` checks
   the contract columns, requires numeric `minutes` without NaN (otherwise eligibility is
   unknown), and requires a numeric, finite `raw_score` and a known `position_group` on the rows
   it will rate. Garbage in a row that is dropped for having under 10 minutes cannot affect any
   output, so it is ignored.
10. **Extra columns pass through.** `transform` returns the input columns in their order plus
    `rating`, and `apply` writes them to `ratings_<method>.parquet`. The FPL pipeline's
    `team_id` and `is_starter` need this, and `test_isolation.py` compares `transform(raw)` with the
    ratings file column by column, which requires the file to have every column of the raw
    input.

## 5. Known limitations and what to check on real data

**Large blocks of exactly equal raw scores move the mean.** Mid-rank gives a tied block the
curve value at its middle rather than the average of the curve over its span. Mean rating of a
fitted group (SYNTHETIC normal scores, n = 2000) when a block of its rows shares one raw score:

| rows sharing one raw score | block at the bottom | block in the middle | block at the top |
|---------------------------:|--------------------:|--------------------:|-----------------:|
| 2% | 6.501 | 6.501 | 6.498 |
| 5% | 6.507 | 6.502 | 6.492 |
| 10% | 6.509 | 6.505 | 6.497 |
| 20% | 6.521 | 6.509 | 6.476 |
| 30% | 6.533 | 6.514 | 6.450 |

Ties up to 10% stay within 0.01 of 6.5; even 30% stays within 0.055, inside the 0.1 tolerance.
A discrete score where two thirds of the group are identical can reach about 0.09. Baseline raw
scores are weighted sums of many stats and VAEP sums are continuous, so large exact ties are
unlikely, but read the per-group means that `apply` prints. If it ever matters, giving a tied
block the average of the curve over its percentile span would make the fit-set mean exactly the
curve's mean; it was not done because the brief specifies mid-rank.

Other limitations:

- **The mean and the tens share are exact on the fit set only.** On any other set (one
  gameweek, a holdout) they reflect how that set differs from the reference. In particular
  a few hundred rows can contain a 10.0 by chance (one in 300 is 0.33%), which the 0.1% rule
  does not cover.
- **A percentile scale is relative.** A 7.0 means "better than about 75% of the reference
  performances of this position group", not an absolute standard, and it says nothing about how
  large the raw gap is.
- **Small groups are coarse.** A group with fewer than 50 reference rows logs a warning; the
  lowest rating a reference row can get is `curve(0.5 / n)`.
- **Editing the curve invalidates saved scalers** (by design, see decision 7): re-run
  `python -m src.scale.apply` for each method afterwards.
- `position_group` is per match (the position played longest in that match), so a player can be
  rated against different groups in different matches.

## 6. How to run

```
.venv/Scripts/python -m src.scale.apply --method baseline
.venv/Scripts/python -m src.scale.apply --method vaep
```

Each reads `data/processed/raw_<method>.parquet` and writes `scaler_<method>.json` and
`ratings_<method>.parquet` (contract plus `rating`, rated rows only, sorted by `match_id`,
`team`, `player_id`). Per-match rating from the frozen scaler:

```python
scaler = PercentileScaler.load("data/processed/scaler_baseline.json")
rated = scaler.transform(raw[raw["match_id"] == match_id])
```

Tests for this module:

```
.venv/Scripts/python -m pytest -q tests/test_scale.py tests/test_scale_curve.py tests/test_scale_scaler.py tests/test_scale_cli.py tests/test_scale_summary.py tests/test_scale_report.py tests/test_web.py::test_rating_band_thresholds tests/test_contract.py
```

The data-backed season rules (`test_rules_season.py`) and `test_isolation.py` run against the real
`ratings_<method>.parquet` and `scaler_<method>.json` once the rating steps and `apply` have
produced them.

## 7. Rehearsal on the preliminary real raw files

`raw_baseline.parquet` and `raw_vaep.parquet` appeared in the shared `data/processed` (written
by the baseline and VAEP agents) while this module was being built. Both satisfy the interface
contract: 10,469 rows, 380 matches, 10,133 rows with 10 or more minutes (GK 766, DEF 3,281,
MID 3,487, FWD 2,599), and near-continuous scores (for example 2,539 distinct values among the
2,599 baseline FWD rows), so ties are not an issue.

To leave `data/` untouched, copies of these two files (with `matches.parquet` and
`lineups.parquet`) were scaled by the real CLI in a temporary folder, and the data-backed tests
`test_rules_season.py` (both methods) and
`test_isolation.py::test_scaler_transform_of_one_match_matches_the_season_ratings` were run
against that folder with `tests.conftest.PROCESSED` redirected: all 26 pass.

| method | rated rows | mean by group (GK / DEF / MID / FWD) | 10.0s | min / max |
|--------|-----------:|--------------------------------------|------:|----------:|
| baseline | 10,133 | 6.500 / 6.500 / 6.500 / 6.500 | 3 (0.030%) | 3.4 / 10.0 |
| vaep | 10,133 | 6.500 / 6.500 / 6.500 / 6.500 | 3 (0.030%) | 3.4 / 10.0 |

A percentile scale depends only on ranks, so with tie-free scores both methods have exactly the
same rating distribution on the fit set; they differ only in who gets which rating. Treat this
section as a rehearsal: those raw files may still change (baseline weights, VAEP models), and,
unlike sections 2 to 5, its numbers are not regenerated by the test suite because `data/` is
gitignored. Run `python -m src.scale.apply --method baseline` and `--method vaep` once the raw
files are final.

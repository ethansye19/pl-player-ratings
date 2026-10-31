# Premier League Player Rating System

Match-by-match player ratings on a 3.0–10.0 scale (season mean 6.5, a 10 is extremely
rare), built two ways and compared:

1. **Baseline** – hand-weighted per-match stats with position-specific weights.
2. **VAEP** – action values from [`socceraction`](https://github.com/ML-KULeuven/socceraction),
   summed per player per match.

Plus a match-ratings web page, a grounded "why this rating?" explanation feature, and a
current-season gameweek pipeline.

> **Status:** Step 1 (scaffold) complete. Nothing is implemented yet; the test suite
> encodes the rules and fails by design. See `CLAUDE.md` for the working contract.

## Setup

Python **3.11** is required (`socceraction` 1.4.2 pins `pandas<2` and `numpy<2`, which
have no wheels for newer interpreters).

```powershell
py -3.11 -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt
.venv\Scripts\python -m pytest -q            # full suite
.venv\Scripts\python -m pytest -q -m "not data"   # tests that need no processed data
```

## Data credits

- Development and validation data: **StatsBomb Open Data**, Premier League 2015/16
  (competition_id=2, season_id=27). https://github.com/statsbomb/open-data
  Used under the StatsBomb Public Data User Agreement.
- Current-season data: the official **Fantasy Premier League API**.

_Setup for each step, method descriptions and known limitations are added as the steps land._

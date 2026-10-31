# Ingestion report: StatsBomb Open Data, Premier League 2015/16

Built from `data\raw\statsbomb` (competition 2, season 27) into `data\processed`.
Data: StatsBomb Open Data (https://github.com/statsbomb/open-data).

| quantity | value |
|---|---|
| matches | 380 |
| teams | 20 |
| events | 1,313,773 (125 columns) |
| SPADL actions | 757,641 |
| players who appeared | 550 |
| player-match appearances | 10,469 |
| team-matches with 11 starters | 760/760 |
| appearances by position group | {'GK': 766, 'DEF': 3313, 'MID': 3652, 'FWD': 2738} |

## Minutes cross-check against socceraction's `minutes_played`

Ours: on-pitch spans from the Starting XI / Substitution / Player Off / Player On / red
card events, period lengths from the Half End timestamps (stoppage time counts, to the
second; treatment time off the pitch does not).
socceraction: whole minutes from the Starting XI / Substitution / Half End events, which
misreads substitutions at exactly 45:00 of the second half and ignores Player Off gaps.

- players compared: 10469 (only in ours: 0, only in socceraction: 0)
- |difference|: mean 0.65 min, median 0.5 min, max 63.9 min; more than 2 min apart: 160, more than 10 min apart: 12
- starter flag disagreements: 0

## Position names seen in the data vs `src/ingest/positions.py`

none

## Anomalies

none

## Lineup-interval inconsistencies corrected from the events

Minutes come from the events; these players' lineup position intervals disagreed by
more than a minute (intervals that ignore a dismissal or a half-time substitution).

- match 3753984: Mark Noble (West Ham United) lineup intervals say 101.2 min, events say 82.0 min; events used
- match 3753984: Philippe Coutinho (Liverpool) lineup intervals say 101.2 min, events say 55.6 min; events used
- match 3753997: Jordan Ayew (Aston Villa) lineup intervals say 95.1 min, events say 16.3 min; events used
- match 3754007: Charlie Adam (Stoke City) lineup intervals say 98.3 min, events say 30.9 min; events used
- match 3754007: Ibrahim Afellay (Stoke City) lineup intervals say 98.3 min, events say 24.5 min; events used
- match 3754018: Bradley Shaun Smith (Liverpool) lineup intervals say 95.2 min, events say 76.2 min; events used
- match 3754032: Gary O'Neil (Norwich City) lineup intervals say 95.5 min, events say 30.2 min; events used
- match 3754039: Pape Souaré (Crystal Palace) lineup intervals say 100.4 min, events say 96.1 min; events used
- match 3754048: James McCarthy (Everton) lineup intervals say 95.5 min, events say 52.2 min; events used
- match 3754058: Simon Francis (AFC Bournemouth) lineup intervals say 95.8 min, events say 58.0 min; events used
- match 3754059: John Terry (Chelsea) lineup intervals say 97.1 min, events say 55.3 min; events used
- match 3754063: Aly Cissokho (Aston Villa) lineup intervals say 98.2 min, events say 75.5 min; events used
- match 3754067: Dwight Gayle (Crystal Palace) lineup intervals say 96.4 min, events say 43.3 min; events used
- match 3754071: Kyle Naughton (Swansea City) lineup intervals say 98.2 min, events say 36.6 min; events used
- match 3754075: Alan Hutton (Aston Villa) lineup intervals say 96.8 min, events say 87.8 min; events used
- match 3754076: Jeremain Lens (Sunderland) lineup intervals say 99.0 min, events say 59.8 min; events used
- match 3754078: Thibaut Courtois (Chelsea) lineup intervals say 97.5 min, events say 54.6 min; events used
- match 3754081: Juan Mata (Manchester United) lineup intervals say 95.4 min, events say 25.5 min; events used
- match 3754086: Nordin Amrabat (Watford) lineup intervals say 94.9 min, events say 93.3 min; events used
- match 3754099: Saido Berahino (West Bromwich Albion) lineup intervals say 96.3 min, events say 47.2 min; events used
- match 3754101: Victor Wanyama (Southampton) lineup intervals say 94.1 min, events say 73.7 min; events used
- match 3754108: John Terry (Chelsea) lineup intervals say 99.4 min, events say 97.8 min; events used
- match 3754111: Per Mertesacker (Arsenal) lineup intervals say 96.4 min, events say 17.7 min; events used
- match 3754119: Fabricio Coloccini (Newcastle United) lineup intervals say 97.6 min, events say 45.6 min; events used
- match 3754120: Victor Wanyama (Southampton) lineup intervals say 96.9 min, events say 54.0 min; events used
- match 3754122: Geoff Cameron (Stoke City) lineup intervals say 97.0 min, events say 83.3 min; events used
- match 3754131: Carl Jenkinson (West Ham United) lineup intervals say 100.6 min, events say 82.0 min; events used
- match 3754148: Ryan Mason (Tottenham Hotspur) lineup intervals say 95.6 min, events say 85.1 min; events used
- match 3754150: Ryan Shawcross (Stoke City) lineup intervals say 97.1 min, events say 49.3 min; events used
- match 3754164: José Fonte (Southampton) lineup intervals say 96.3 min, events say 81.0 min; events used
- match 3754165: Danny Drinkwater (Leicester City) lineup intervals say 95.1 min, events say 86.8 min; events used
- match 3754167: Russell Martin (Norwich City) lineup intervals say 97.9 min, events say 88.2 min; events used
- match 3754169: Younès Kaboul (Sunderland) lineup intervals say 97.7 min, events say 76.9 min; events used
- match 3754176: Cheikhou Kouyaté (West Ham United) lineup intervals say 96.4 min, events say 69.0 min; events used
- match 3754191: Victor Wanyama (Southampton) lineup intervals say 95.3 min, events say 78.6 min; events used
- match 3754196: Sadio Mané (Southampton) lineup intervals say 21.7 min, events say 15.5 min; events used
- match 3754203: Adrián (West Ham United) lineup intervals say 98.8 min, events say 94.6 min; events used
- match 3754205: Aleksandar Mitrović (Newcastle United) lineup intervals say 95.3 min, events say 15.5 min; events used
- match 3754213: Valon Behrami (Watford) lineup intervals say 98.6 min, events say 64.4 min; events used
- match 3754217: Francis Joseph Coquelin (Arsenal) lineup intervals say 99.9 min, events say 48.2 min; events used
- match 3754217: Mesut Özil (Arsenal) lineup intervals say 79.4 min, events say 78.2 min; events used
- match 3754217: Aaron Ramsey (Arsenal) lineup intervals say 100.4 min, events say 99.3 min; events used
- match 3754217: Gabriel Paulista (Arsenal) lineup intervals say 100.4 min, events say 45.7 min; events used
- match 3754219: James McClean (West Bromwich Albion) lineup intervals say 98.4 min, events say 33.5 min; events used
- match 3754219: Salomón Rondón (West Bromwich Albion) lineup intervals say 98.4 min, events say 96.4 min; events used
- match 3754220: James Milner (Liverpool) lineup intervals say 97.6 min, events say 62.2 min; events used
- match 3754227: Jamie Vardy (Leicester City) lineup intervals say 96.9 min, events say 56.9 min; events used
- match 3754230: Sadio Mané (Southampton) lineup intervals say 96.1 min, events say 94.1 min; events used
- match 3754252: Jamaal Lascelles (Newcastle United) lineup intervals say 53.5 min, events say 52.0 min; events used
- match 3754261: Danny Simpson (Leicester City) lineup intervals say 92.4 min, events say 56.1 min; events used
- match 3754273: James Collins (West Ham United) lineup intervals say 96.2 min, events say 84.6 min; events used
- match 3754274: Marc Muniesa (Stoke City) lineup intervals say 98.2 min, events say 47.1 min; events used
- match 3754277: Kevin Mirallas (Everton) lineup intervals say 98.2 min, events say 33.2 min; events used
- match 3754288: Nemanja Matić (Chelsea) lineup intervals say 97.4 min, events say 43.2 min; events used
- match 3754291: Nathan Aké (Watford) lineup intervals say 95.6 min, events say 62.7 min; events used
- match 3754294: Ramiro Funes Mori (Everton) lineup intervals say 99.9 min, events say 51.8 min; events used
- match 3754297: Gareth McAuley (West Bromwich Albion) lineup intervals say 96.9 min, events say 92.5 min; events used
- match 3754298: Kevin Mirallas (Everton) lineup intervals say 3.6 min, events say 1.9 min; events used
- match 3754318: Francis Joseph Coquelin (Arsenal) lineup intervals say 95.2 min, events say 55.2 min; events used
- match 3754322: Thibaut Courtois (Chelsea) lineup intervals say 95.4 min, events say 78.9 min; events used
- match 3754333: Aleksandar Mitrović (Newcastle United) lineup intervals say 95.1 min, events say 66.8 min; events used
- match 3754336: Séamus Coleman (Everton) lineup intervals say 95.4 min, events say 77.7 min; events used
- match 3754337: Steven Whittaker (Norwich City) lineup intervals say 95.5 min, events say 30.0 min; events used
- match 3754339: Xherdan Shaqiri (Stoke City) lineup intervals say 103.6 min, events say 52.3 min; events used
- match 3754344: Daryl Janmaat (Newcastle United) lineup intervals say 96.4 min, events say 40.5 min; events used

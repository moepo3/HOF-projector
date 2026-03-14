# NBA Hall of Fame Projector

A data engineering and machine learning project that models Hall of Fame probability for NBA players — past, present, and future. The system identifies statistical archetypes among inducted Hall of Famers, builds empirical aging curves, and projects whether current players are on a Hall of Fame trajectory.

Eventually expanding to MLB and NFL.

---

## Motivation

The NBA Hall of Fame has no objective criteria. Voters are inconsistent, era bias is real, and there is no agreed-upon framework for what a Hall of Fame career looks like. This project attempts to build one — data-first, era-adjusted, and archetype-driven rather than relying on a single statistical threshold.

The goal is to answer questions like:
- Will your favorite player be a hall of famer? What would he need to do to become one?
- Who are the biggest statistical snubs in HOF history?
- Who is in the Hall of Fame that does not belong?

---

## Current Status - 03/14/2026

**Phase 1 — Data Infrastructure: Complete**

- Ingestion pipeline loads the full NBA/ABA historical dataset (5,396 players, 33,278 player-seasons) from [this Kaggle dataset](https://www.kaggle.com/datasets/sumitrodatta/nba-aba-baa-stats)
- All stats normalized to **per-75 possessions** using league-average pace by season, making players comparable across eras
- Awards table populated: All-Star (2,058), All-NBA 1st/2nd/3rd, All-Defense 1st/2nd, MVP, DPOY, ROY
- 177 Hall of Famers identified in dataset
- SQLite database for local development, designed to migrate to BigQuery for production

**Phase 2 — Aging Curves: Complete**

- Empirical aging curves built using the **delta method**: for every player with consecutive seasons, compute the year-over-year change in each stat at each age
- Curves stratified by size bucket (Guard / Wing / Big) — larger players peak earlier and decline faster
- 16,105 consecutive-season pairs across 23,300 qualifying player-seasons
- Key findings:
  - Scoring peaks ~age 26
  - Playmaking peaks ~age 27
  - Rebounding declines from ~age 22 (athleticism-dependent)
  - Overall value (VORP, BPM, WS) peaks ~age 25
- Plots saved to `data/processed/`

---

## Roadmap

**Phase 3 — Feature Engineering (Next)**
- Peak score: best 7 consecutive seasons by VORP
- Career award score: weighted sum of All-NBA, All-Star, MVP, DPOY, Finals MVP selections
- Championship value score: contextually adjusted (distinguishes star players from role players on championship teams)
- Per-75 career and peak averages for scoring, playmaking, rebounding, and two-way impact

**Phase 4 — HOF Archetype Clustering**
- Cluster current Hall of Famers into statistical archetypes using k-means
- Expected archetypes: dominant scorer, two-way star, playmaker, interior anchor, champion/role player
- A player needs to match at least one archetype to project as a HOF candidate — this accounts for players like Robert Horry (championships) and DeMar DeRozan (volume scoring) who represent different paths to induction

**Phase 5 — HOF Classifier**
- Binary classification model (logistic regression / random forest) trained on completed careers
- Features: peak score, award score, championship score, per-75 career averages, archetype cluster
- Validated against known snubs (Chris Webber) and known locks (LeBron James, Steph Curry)

**Phase 6 — Career Projection**
- Apply aging curves to active players to project remaining career value
- Two modes:
  - **Auto projection**: career ends at statistically average retirement age for their size bucket
  - **Manual override**: user sets a retirement age (e.g. "Bam Adebayo plays until 42")
- Projected career fed into classifier to produce HOF probability

**Phase 7 — Frontend**
- Streamlit web app
- Search any player, see their HOF probability, archetype, career trajectory chart, and comparable Hall of Famers
- Adjust retirement age slider and watch the probability update in real time

**Phase 8 — Expand to MLB and NFL**

---

## Setup

**1. Clone the repo**
```bash
git clone https://github.com/moepo3/HOF-projector.git
cd HOF-projector
```

**2. Install dependencies**
```bash
pip install -r requirements.txt
```

**3. Download the dataset**

Download from Kaggle: [NBA/ABA/BAA Stats by sumitrodatta](https://www.kaggle.com/datasets/sumitrodatta/nba-aba-baa-stats)

Unzip and place all CSV files in `data/raw/`

**4. Run the pipeline**
```bash
# Initialize database and load all data
python src/schema.py
python src/ingest.py

# Build aging curves
python src/aging_curve.py
```

---

## Stack

| Layer | Technology |
|---|---|
| Language | Python |
| Data storage | SQLite (local) / BigQuery (production) |
| Transformations | dbt Core |
| Orchestration | Apache Airflow |
| Data processing | Pandas, NumPy |
| Modeling | scikit-learn |
| Frontend (planned) | Streamlit |
| Version control | Git / GitHub |

---

## Project Structure

```
hof_projector/
├── src/
│   ├── schema.py          # Database table definitions
│   ├── ingest.py          # Kaggle dataset ingestion pipeline
│   ├── player_index.py    # Player list via nba_api
│   ├── aging_curve.py     # Empirical aging curves (delta method)
│   └── features.py        # Feature engineering (coming Phase 3)
├── data/
│   ├── raw/               # Kaggle CSVs (not committed — see Setup)
│   └── processed/         # Aging curve outputs and plots
├── dbt/                   # dbt models (coming Phase 3)
├── pipelines/             # Airflow DAGs (coming Phase 3)
├── notebooks/             # Exploratory analysis (coming)
└── requirements.txt
```

---

## Data Source

[NBA/ABA/BAA Stats — Kaggle (sumitrodatta)](https://www.kaggle.com/datasets/sumitrodatta/nba-aba-baa-stats)

Sourced from Basketball Reference. Covers every NBA and ABA player season in history.

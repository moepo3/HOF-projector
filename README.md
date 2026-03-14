# Hall of Fame Projector

A data engineering and machine learning project that scrapes historical NBA (and eventually MLB, NFL) statistics to identify Hall of Fame archetypes, build aging curves, and project whether current players will be inducted.

## Stack
- **Python** — scraping, feature engineering, modeling
- **SQLite** (local dev) / **BigQuery** (production) — storage
- **dbt** — transformations and feature models
- **Apache Airflow** — pipeline orchestration
- **Streamlit** — frontend (planned)

## Project Structure
```
hof_projector/
├── src/
│   ├── scraper.py         # Basketball Reference scraper
│   ├── schema.py          # Database schema definitions
│   ├── aging_curve.py     # Aging curve via delta method
│   └── features.py        # Feature engineering (coming soon)
├── data/
│   ├── raw/               # Raw scraped data
│   └── processed/         # Cleaned, normalized data
├── notebooks/
│   └── eda.ipynb          # Exploratory analysis (coming soon)
├── dbt/                   # dbt models (coming soon)
├── pipelines/             # Airflow DAGs (coming soon)
└── requirements.txt
```

## Setup
```bash
pip install -r requirements.txt

# Step 1: Scrape Basketball Reference
python src/scraper.py

# Step 2: Build aging curve
python src/aging_curve.py
```

## Methodology

### Era Normalization
Stats are normalized to **per-75 possessions** using league-average pace by season, making players comparable across eras.

### Hall of Fame Archetypes
Rather than a single HOF threshold, players are clustered into archetypes (dominant scorer, two-way star, playmaker, champion/role player, etc.). A player needs to match at least one archetype profile to project as a HOF candidate.

### Peak Score
A player's best **7 consecutive seasons** by VORP are used as the peak score. This guards against compilers who accumulate counting stats over long mediocre careers.

### Aging Curve
Built using the **delta method**: for every player-season pair, compute the year-over-year change in each normalized stat at each age. Averaged across all players, this produces an empirical aging curve. Stratified by size bucket (guard / wing / big).

### Career Projection
Given a player's current age and stats, the aging curve is applied forward to project remaining career value. The projection feeds into the HOF classifier to produce a probability estimate.

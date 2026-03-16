# NBA Hall of Fame Predictor

A full data engineering and machine learning pipeline that predicts NBA Hall of Fame probability for every player in history — and produces an all-time ranking of the greatest players ever.

---

## What This Does

- **HOF Probability** — for every player with 400+ games and 5+ career Win Shares, the model outputs a probability (0–100%) of HOF induction based on their statistical profile
- **Career Projection** — active players are projected forward using aging curves before scoring, so LeBron at age 40 isn't penalized for not yet being inducted
- **All-Time Ranking** — an uncapped backend score ranks every player in history by a peak-weighted composite of stats, accolades, and ring quality
- **Archetypes** — 16 modern + 4 pre-1974 statistical archetypes cluster players by playing style

---

## All-Time Top 10 (Backend Score)

| Rank | Player | Backend Score |
|------|--------|---------------|
| 1 | Michael Jordan | 413 |
| 2 | LeBron James | 407 |
| 3 | Kareem Abdul-Jabbar | 385 |
| 4 | Tim Duncan | 358 |
| 5 | Kobe Bryant | 346 |
| 6 | Kevin Garnett | 319 |
| 7 | Magic Johnson | 310 |
| 8 | Karl Malone | 309 |
| 9 | Hakeem Olajuwon | 305 |
| 10 | Larry Bird | 301 |

---

## Stack

| Layer | Technology |
|-------|-----------|
| Database | SQLite (local) |
| Data pipeline | Python, Pandas, NumPy |
| Feature engineering | Custom peak windows, per-75 normalization, aging curves |
| Clustering | K-Means (scikit-learn) |
| Classification | Soft voting ensemble: Logistic Regression + Random Forest + Gradient Boosting |
| Data source | [Kaggle: NBA Player Stats](https://www.kaggle.com/datasets/sumitrodatta/nba-aba-baa-stats) |

---

## Pipeline

```
src/schema.py      → Initialize SQLite database
src/ingest.py      → Load Kaggle CSVs, flag HOF players, compute per-75 stats
src/features.py    → Build peak windows (3yr, 7yr), aging curves, championship scores
src/archetypes.py  → K-Means clustering into 16 modern + 4 pre-1974 archetypes
src/classifier.py  → Train ensemble, compute HOF probability + backend score
```

Run the full pipeline:
```bash
python3 src/schema.py
python3 src/ingest.py
python3 src/features.py
python3 src/archetypes.py
python3 src/classifier.py
```

---

## Key Design Decisions

### Feature Engineering

**Per-75 possession normalization** — all rate stats are normalized to per-75 possessions using league pace for each season. This makes cross-era comparisons valid: a 1965 player averaging 25 PPG in a 110-pace league is not the same as a 2005 player averaging 25 PPG in a 90-pace league.

**Peak windows** — rather than career averages, the model uses the player's best consecutive 3-year and 7-year windows by VORP sum. This correctly identifies players with elite peaks but shorter careers (e.g., Derrick Rose).

**Career minutes over games played** — longevity is measured in career minutes rather than games played. A player who averaged 35 minutes for 15 seasons is valued appropriately over someone who played 20 seasons at declining minutes.

### HOF Label Quality

The HOF flag in the raw dataset includes coaches, contributors, and referees. We override this with a hand-verified player-only list (`src/hof_players.py`) sourced from the official Naismith Memorial Basketball Hall of Fame, cross-referenced against Basketball Reference IDs.

**International players** — five players (Radja, Petrovic, Sabonis, Marciulionis, Yao Ming) are flagged as `PRIMARILY_INTERNATIONAL` and excluded from classifier training. Their NBA stats alone cannot explain their induction, which was based largely on non-NBA careers.

### Two-Model Architecture

**Modern model (post-1974):** Uses VORP, Win Shares, All-NBA, DPOY, Finals MVP, ring quality, defensive scores, and efficiency metrics. AUC: 0.965.

**Pre-1974 model:** Blocks and steals were not recorded before 1974, making VORP/BPM unavailable. This model relies heavily on All-Star selections, All-NBA teams, and Win Shares, with an `award_weight_boost` feature that multiplies award values to compensate for missing advanced stats. AUC: 0.967.

> **Caveat:** Pre-1974 HOF probability should be interpreted with caution. The absence of blocks, steals, and advanced metrics means the model cannot fully evaluate these players. This is by design — we would rather be honest about data limitations than produce false precision.

### Ring Quality Score

```
ring_quality = opponent_win_pct × player_vorp_share
```

We use only the Finals opponent's regular season win percentage — not the champion's. This correctly rewards beating a great opponent regardless of how good your own team was. LeBron's 2016 ring (beating the 73-win Warriors) scores higher than most Jordan rings because the opponent was historically dominant.

### Backend Score Formula

The all-time ranking uses an uncapped composite score:

```
backend_score = (adj_prob × 150)
              + (peak7_vorp × 1.5)
              + (total_vorp × 0.1)
              + (peak7_ws_per_48 × total_ws × 0.5)
              + accolade_bonus
```

**Accolade weights:**

| Award | Points |
|-------|--------|
| All-NBA 1st Team | 5 |
| MVP | 3 |
| All-NBA 2nd Team | 3 |
| Finals MVP | 3 |
| All-Defense 1st Team | 3 |
| DPOY | 2 |
| All-NBA 3rd Team | 1 |
| All-Defense 2nd Team | 1 |
| Ring quality score | ×12 |

All-Star selections are intentionally excluded from the backend score — they are a popularity vote and do not add signal beyond what VORP and All-NBA already capture.

### Why Jordan #1 Over LeBron

Jordan's `peak7_vorp_sum` of 75.3 is the highest of any player in the dataset — nearly double Kevin Garnett's 55.5 and significantly above LeBron's 66.3. The backend score weights peak7 at ×1.5, which properly rewards the most dominant 7-year stretch ever recorded. LeBron's extraordinary career longevity (158 total VORP) keeps him within 6 points of Jordan. The near-tie accurately reflects the genuine debate.

### Why Jordan's HOF Probability Is ~72%

The model's raw probability is a known limitation. The classifier was trained on Win Shares and All-NBA signals, and Jordan's shorter career (1037 games vs. e.g. KG's 1504) gives him slightly lower career totals than some HOFers with longer careers. The backend score — which heavily weights peak performance — is the correct ranking metric. The HOF probability is shown as-is because it reflects a real pattern in the training data, not a data error.

---

## Model Performance

| Model | AUC | Training Set |
|-------|-----|--------------|
| Modern (post-1974) | 0.965 | 1,347 players (78 HOF) |
| Pre-1974 | 0.967 | 126 players (34 HOF) |

Ensemble: Soft voting average of calibrated Logistic Regression, Random Forest, and Gradient Boosting.

---

## Known Limitations

1. **No playoff stats** — players with outsized playoff performance (e.g. Jimmy Butler) are undervalued. The model only sees regular season statistics.
2. **Injury-shortened careers** — players who retired early due to injury (Rose, Grant Hill, Penny Hardaway) are scored on actual career totals, not projected healthy careers. Rose's ~5% reflects this honestly.
3. **Pre-1974 era** — advanced metrics unavailable; probability estimates for this era are approximate.
4. **International careers** — players inducted primarily for non-NBA careers will show low HOF probabilities. This is expected and documented.

---

## Pending Development

- [ ] Playoff performance feature (Jimmy Butler problem)
- [ ] Westbrook vs. Jokic archetype naming distinction (same cluster, different efficiency profiles)
- [ ] Streamlit frontend — player search, HOF probability, archetype card, career trajectory
- [ ] GridSearchCV hyperparameter tuning
- [ ] BigQuery + dbt migration for production
- [ ] MLB / NFL expansion (Phase 2)

---

## Data

Raw CSVs not committed (Kaggle terms). Download from:
https://www.kaggle.com/datasets/sumitrodatta/nba-aba-baa-stats

Place in `data/raw/` before running the pipeline.

Required files: `Player Per Game.csv`, `Advanced.csv`, `Player Career Info.csv`,
`All-Star Selections.csv`, `End of Season Teams.csv`, `Player Award Shares.csv`,
`Team Summaries.csv`

---

## Author

Moe Posner — Data Engineer
[GitHub](https://github.com/moepo3)

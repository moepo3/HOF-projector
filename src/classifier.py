"""
classifier.py

Hall of Fame probability classifier with career projection.

Architecture:
    1. Score current career stats
    2. Project remaining career using aging curves (delta method)
    3. Score projected complete career
    4. Output raw backend score (uncapped) + display probability (0-100%)
    5. Players exceeding HOF threshold score 100% display

Key design decisions:
    - Min 400 games AND min 5 career WS to filter noise
    - International players excluded from training (NBA stats don't capture their value)
    - Pre-1974 players get separate model with higher award weighting
    - No MVP in features (already captured in VORP — avoids Rose double-count)
    - Active players projected forward via aging curves before scoring
    - Backend score uncapped — enables all-time ranking

Usage:
    python src/classifier.py
"""

import logging
import sqlite3
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings("ignore")

sys.path.insert(0, "src")
from hof_players import get_hof_player_ids, get_hof_player_names, get_international_players

Path("data/processed").mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
)
log = logging.getLogger(__name__)

DB_PATH       = "data/hof.db"
FEATURES_PATH = "data/processed/player_features_with_archetypes.csv"

# Minimum thresholds to be included in predictions
MIN_GAMES = 400
MIN_WS    = 5.0   # career win shares — filters pre-VORP era noise

# HOF display threshold — composite above this AND at least 1 All-NBA = 100%
# Raised to 100 to prevent good-but-not-great players (Porter, Lowry, Marion)
# from being auto-promoted. Composite of 100 requires e.g.:
#   peak7_vorp ~30 + total_vorp ~50 + 8 All-Stars + 2 All-NBA-1 = ~105
HOF_THRESHOLD  = 100.0
# Floor — composite below this AND backend < 10 caps display at 5%
HOF_FLOOR      = 12.0

# ---------------------------------------------------------------------------
# Championship and Finals MVP reference
# ---------------------------------------------------------------------------
NBA_CHAMPIONS = {
    1947: "PHW", 1948: "BAL", 1949: "MNL", 1950: "MNL", 1951: "ROC",
    1952: "MNL", 1953: "MNL", 1954: "MNL", 1955: "SYR", 1956: "PHW",
    1957: "BOS", 1958: "STL", 1959: "BOS", 1960: "BOS", 1961: "BOS",
    1962: "BOS", 1963: "BOS", 1964: "BOS", 1965: "BOS", 1966: "BOS",
    1967: "PHW", 1968: "BOS", 1969: "BOS", 1970: "NYK", 1971: "MIL",
    1972: "LAL", 1973: "NYK", 1974: "BOS", 1975: "GSW", 1976: "BOS",
    1977: "POR", 1978: "WSB", 1979: "SEA", 1980: "LAL", 1981: "BOS",
    1982: "LAL", 1983: "PHI", 1984: "BOS", 1985: "LAL", 1986: "BOS",
    1987: "LAL", 1988: "LAL", 1989: "DET", 1990: "DET", 1991: "CHI",
    1992: "CHI", 1993: "CHI", 1994: "HOU", 1995: "HOU", 1996: "CHI",
    1997: "CHI", 1998: "CHI", 1999: "SAS", 2000: "LAL", 2001: "LAL",
    2002: "LAL", 2003: "SAS", 2004: "DET", 2005: "SAS", 2006: "MIA",
    2007: "SAS", 2008: "BOS", 2009: "LAL", 2010: "LAL", 2011: "DAL",
    2012: "MIA", 2013: "MIA", 2014: "SAS", 2015: "GSW", 2016: "CLE",
    2017: "GSW", 2018: "GSW", 2019: "TOR", 2020: "LAL", 2021: "MIL",
    2022: "GSW", 2023: "DEN", 2024: "BOS",
}

FINALS_OPPONENTS = {
    1947: "CHI", 1948: "PHW", 1949: "WSC", 1950: "SYR", 1951: "NYK",
    1952: "NYK", 1953: "NYK", 1954: "SYR", 1955: "FTW", 1956: "FTW",
    1957: "STL", 1958: "BOS", 1959: "MNL", 1960: "STL", 1961: "STL",
    1962: "LAL", 1963: "LAL", 1964: "SFW", 1965: "LAL", 1966: "LAL",
    1967: "SFW", 1968: "LAL", 1969: "LAL", 1970: "LAL", 1971: "BAL",
    1972: "NYK", 1973: "LAL", 1974: "MIL", 1975: "WSB", 1976: "PHO",
    1977: "PHI", 1978: "SEA", 1979: "WSB", 1980: "PHI", 1981: "HOU",
    1982: "PHI", 1983: "LAL", 1984: "LAL", 1985: "BOS", 1986: "HOU",
    1987: "BOS", 1988: "DET", 1989: "LAL", 1990: "POR", 1991: "LAL",
    1992: "POR", 1993: "PHO", 1994: "NYK", 1995: "ORL", 1996: "SEA",
    1997: "UTA", 1998: "UTA", 1999: "NYK", 2000: "IND", 2001: "PHI",
    2002: "NJN", 2003: "NJN", 2004: "LAL", 2005: "DET", 2006: "DAL",
    2007: "CLE", 2008: "LAL", 2009: "ORL", 2010: "BOS", 2011: "MIA",
    2012: "OKC", 2013: "SAS", 2014: "MIA", 2015: "CLE", 2016: "GSW",
    2017: "CLE", 2018: "CLE", 2019: "GSW", 2020: "MIA", 2021: "PHO",
    2022: "BOS", 2023: "MIA", 2024: "DAL",
}

FINALS_MVP = {
    1969: "westje01",   1970: "reedwi01",   1971: "abdulka01",
    1972: "chambwi01",  1973: "reedwi01",   1974: "cowenda01",
    1975: "barryri01",  1976: "whitejo01",  1977: "waltobi01",
    1978: "unselde01",  1979: "johnsde02",  1980: "johnsma02",
    1981: "maxwece01",  1982: "johnsma02",  1983: "malonmo01",
    1984: "birdla01",   1985: "abdulka01",  1986: "birdla01",
    1987: "johnsma02",  1988: "worthja01",  1989: "thomais02",
    1990: "thomais02",  1991: "jordami01",  1992: "jordami01",
    1993: "jordami01",  1994: "olajuha01",  1995: "olajuha01",
    1996: "jordami01",  1997: "jordami01",  1998: "jordami01",
    1999: "duncati01",  2000: "onealsh01",  2001: "onealsh01",
    2002: "onealsh01",  2003: "duncati01",  2004: "chaunde01",
    2005: "duncati01",  2006: "wadedw01",   2007: "parketo01",
    2008: "piercpa01",  2009: "bryanko01",  2010: "bryanko01",
    2011: "nowitdi01",  2012: "jamesle01",  2013: "jamesle01",
    2014: "leonaka01",  2015: "iguodan01",  2016: "jamesle01",
    2017: "duranke01",  2018: "duranke01",  2019: "leonaka01",
    2020: "jamesle01",  2021: "antetgi01",  2022: "curryst01",
    2023: "jokicni01",  2024: "tatumja01",
}

# ---------------------------------------------------------------------------
# Ring quality
# ---------------------------------------------------------------------------
def compute_ring_quality(seasons, summaries):
    """
    Ring quality per player per championship season:
        ring_quality = opponent_win_pct × min(player_vorp_share, 0.40)

    opponent_win_pct: how strong was the opponent you beat in the Finals.
        This is the primary measure of ring prestige. The 2016 Warriors
        (.890) are the hardest Finals opponent ever — LeBron's 2016 ring
        should be his best.

    vorp_share capped at 0.40: prevents individual dominance from creating
        outliers. Jordan had 58% VORP share on some Bulls teams — without
        the cap his ring score would double a player on a balanced team.

    champ_wp intentionally excluded: we don't reward teams for being great,
        we reward players for beating great opponents. LeBron winning as a
        Cavs underdog (.695) against the 73-win Warriors (.890) should not
        be penalized just because his own team's win pct was lower.
    """
    summaries = summaries.copy()
    summaries.columns = [c.lower() for c in summaries.columns]
    if "w" in summaries.columns and "l" in summaries.columns:
        summaries["win_pct"] = summaries["w"] / (summaries["w"] + summaries["l"])
    else:
        summaries["win_pct"] = 0.5

    def get_win_pct(season, abbrev):
        rows = summaries[
            (summaries["season"] == season) &
            (summaries["abbreviation"] == abbrev)
        ]
        return float(rows["win_pct"].values[0]) if not rows.empty else 0.5

    VORP_SHARE_CAP = 1.0  # no cap — full credit for genuine dominance

    records = []
    for season, champ in NBA_CHAMPIONS.items():
        team_s = seasons[
            (seasons["season"] == season) &
            (seasons["team"] == champ) &
            (seasons["vorp"].notna())
        ]
        if team_s.empty:
            continue

        opp_abbrev  = FINALS_OPPONENTS.get(season)
        opp_wp      = get_win_pct(season, opp_abbrev) if opp_abbrev else 0.5
        total_vorp  = max(team_s["vorp"].sum(), 0.1)

        for _, row in team_s.iterrows():
            share     = max(row["vorp"], 0) / total_vorp
            records.append({
                "player_id":        row["player_id"],
                "ring_quality":     opp_wp * share,
                "opponent_win_pct": opp_wp,
            })

    if not records:
        return pd.DataFrame(columns=["player_id", "ring_quality_score",
                                      "ring_quality_rings", "best_ring_difficulty"])
    df = pd.DataFrame(records)
    return df.groupby("player_id").agg(
        ring_quality_score  =("ring_quality",     "sum"),
        ring_quality_rings  =("ring_quality",     "count"),
        best_ring_difficulty=("opponent_win_pct", "max"),
        best_ring_win_pct   =("opponent_win_pct", "max"),
    ).reset_index()


# ---------------------------------------------------------------------------
# Career projection using aging curves
# ---------------------------------------------------------------------------
def project_career(player_row: pd.Series, aging_curves: pd.DataFrame,
                   current_season: int = 2025) -> dict:
    """
    Project a player's remaining career using aging curves.
    Returns projected additional totals to add to current stats.

    For retired players, returns zeros (no projection needed).
    For active players, projects from current age to expected retirement.
    """
    if player_row.get("retired", 1):
        return {"proj_vorp": 0, "proj_ws": 0, "proj_allstar": 0, "proj_seasons": 0}

    size_bucket = player_row.get("size_bucket", "All")
    age         = player_row.get("peak7_seasons_n", 0)

    # Get current age approximation from career length
    # Approximate: most players enter at 20, so current age ~ 20 + seasons
    seasons_played = player_row.get("seasons_played", 0)
    est_age        = min(20 + seasons_played, 42)

    # Get aging curve for this size bucket
    curve = aging_curves[
        aging_curves["size_bucket"].isin([size_bucket, "All"])
    ].copy()

    if curve.empty:
        return {"proj_vorp": 0, "proj_ws": 0, "proj_allstar": 0, "proj_seasons": 0}

    # Average expected retirement age by size
    retirement_age = {"Guard": 35, "Wing": 34, "Big": 33}.get(size_bucket, 34)

    proj_vorp     = 0.0
    proj_ws       = 0.0
    proj_seasons  = 0
    current_vorp  = player_row.get("peak7_per_season", 2.0)

    for age in range(int(est_age), retirement_age):
        age_curve = curve[curve["age"] == age]
        if age_curve.empty:
            continue
        delta_vorp = float(age_curve["mean_delta_vorp"].values[0]) if "mean_delta_vorp" in age_curve.columns else -0.3
        delta_ws   = float(age_curve["mean_delta_ws_per_48"].values[0]) * 48 if "mean_delta_ws_per_48" in age_curve.columns else -0.2
        current_vorp = max(current_vorp + delta_vorp, 0)
        proj_vorp   += current_vorp
        proj_ws     += max(current_vorp * 2, 0)  # approximate WS from VORP
        proj_seasons += 1

    # Project All-Star appearances — roughly 1 per season if above average
    proj_allstar = max(0, int(proj_seasons * (current_vorp > 2.0)))

    return {
        "proj_vorp":    proj_vorp,
        "proj_ws":      proj_ws,
        "proj_allstar": proj_allstar,
        "proj_seasons": proj_seasons,
    }


# ---------------------------------------------------------------------------
# Build classifier features
# ---------------------------------------------------------------------------
def build_features(features, awards, ring_quality, aging_curves=None,
                   pre_1974: bool = False):
    df = features.copy()

    # Accolade counts — no MVP (already in VORP)
    accolade_map = {
        "All_NBA_1":     "all_nba_1_count",
        "All_NBA_2":     "all_nba_2_count",
        "All_NBA_3":     "all_nba_3_count",
        "All_Star":      "all_star_count",
        "DPOY":          "dpoy_count",
        "All_Defense_1": "all_def_1_count",
        "All_Defense_2": "all_def_2_count",
        "MVP":           "mvp_count",   # not used in classifier features (Rose problem)
                                        # but used in backend score accolade bonus
    }
    for award_type, col in accolade_map.items():
        counts = (
            awards[awards["award"] == award_type]
            .groupby("player_id").size()
            .reset_index(name=col)
        )
        df = df.merge(counts, on="player_id", how="left")
        df[col] = df[col].fillna(0).astype(int)

    # Finals MVP
    fmvp = {}
    for pid in FINALS_MVP.values():
        fmvp[pid] = fmvp.get(pid, 0) + 1
    df["finals_mvp_count"] = df["player_id"].map(fmvp).fillna(0).astype(int)

    # Ring quality
    df = df.merge(ring_quality, on="player_id", how="left")
    for col in ["ring_quality_score", "ring_quality_rings",
                "best_ring_difficulty", "best_ring_win_pct"]:
        df[col] = df.get(col, pd.Series(0, index=df.index)).fillna(0)

    # Career projection for active players
    if aging_curves is not None and not aging_curves.empty:
        proj_cols = ["proj_vorp", "proj_ws", "proj_allstar", "proj_seasons"]
        projs = df.apply(
            lambda r: project_career(r, aging_curves), axis=1
        )
        proj_df = pd.DataFrame(list(projs), index=df.index)
        df = pd.concat([df, proj_df], axis=1)
    else:
        df["proj_vorp"]    = 0
        df["proj_ws"]      = 0
        df["proj_allstar"] = 0
        df["proj_seasons"] = 0

    # Projected totals
    df["total_vorp"]     = df["career_vorp"].fillna(0) + df["proj_vorp"]
    df["total_ws"]       = df["career_ws"].fillna(0)   + df["proj_ws"]
    df["total_allstar"]  = df["all_star_count"].fillna(0) + df["proj_allstar"]

    # Derived features
    career_safe = df["total_vorp"].abs().replace(0, 1)
    df["peak_longevity_ratio"] = df["peak7_vorp_sum"].fillna(0) / career_safe
    df["peak7_per_season"]     = df["peak7_vorp_sum"].fillna(0) / df["peak7_seasons_n"].replace(0, 1).fillna(1)
    df["all_nba_weighted"]     = (
        df["all_nba_1_count"] * 3 +
        df["all_nba_2_count"] * 2 +
        df["all_nba_3_count"] * 1
    )
    df["elite_allnba_flag"]    = (df["all_nba_1_count"] >= 8).astype(int)
    # allstar_longevity — uses career_minutes instead of games_played
    # so end-of-career bench stints don't inflate the metric
    df["allstar_longevity"] = df["total_allstar"] * df["career_minutes"].fillna(0) / 30000

    # Pre-1974 specific: boost award weight since we can't use advanced stats
    if pre_1974:
        df["award_weight_boost"] = (
            df["all_star_count"]    * 3.0 +
            df["all_nba_1_count"]   * 8.0 +
            df["all_nba_2_count"]   * 5.0 +
            df["all_nba_3_count"]   * 2.0 +
            df["finals_mvp_count"]  * 10.0
        )
    else:
        df["award_weight_boost"] = 0

    for col in ["peak7_vorp_sum", "peak3_vorp_sum", "peak7_pts_per75",
                "peak7_ast_per75", "peak7_trb_per75", "peak7_ws_per_48",
                "peak7_ts_pct", "total_vorp", "total_ws",
                "defensive_score", "ts_vs_era_avg", "championship_score"]:
        if col in df.columns:
            df[col] = df[col].fillna(0)

    return df


# ---------------------------------------------------------------------------
# Feature column sets
# ---------------------------------------------------------------------------
MODERN_FEATURES = [
    # Peak value
    "peak7_vorp_sum", "peak3_vorp_sum", "peak7_per_season",
    "peak7_pts_per75", "peak7_ws_per_48", "peak7_ts_pct",
    # Longevity
    "total_vorp", "total_ws", "career_minutes",
    "peak_longevity_ratio",
    # Accolades — All-Star removed, model relies on VORP/WS/All-NBA/awards
    "mvp_count",                                        # added per user request
    "all_nba_1_count", "all_nba_2_count", "all_nba_3_count",
    "all_nba_weighted", "elite_allnba_flag",
    "dpoy_count", "finals_mvp_count",
    "all_def_1_count", "all_def_2_count",               # added all_def_2
    # Team success
    "ring_quality_score", "ring_quality_rings", "best_ring_difficulty",
    # Defense / efficiency
    "defensive_score", "ts_vs_era_avg", "championship_score",
]

PRE74_FEATURES = [
    "peak7_pts_per75", "peak7_trb_per75", "peak7_ast_per75",
    "peak7_ts_pct", "total_ws", "games_played",
    "all_star_count", "all_nba_1_count", "all_nba_2_count",
    "all_nba_weighted", "finals_mvp_count",
    "ring_quality_score", "ring_quality_rings",
    "award_weight_boost",   # heavily weighted for pre-1974
    "peak_longevity_ratio",
]


# ---------------------------------------------------------------------------
# Train model
# ---------------------------------------------------------------------------
def train_model(df_train, feat_cols, label="Modern"):
    X = df_train[feat_cols].fillna(0).values
    y = df_train["hof_inducted"].astype(int).values

    log.info(f"\n[{label}] Training on {len(df_train)} players "
             f"({y.sum()} HOF, {(1-y).sum()} non-HOF)")

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    pipes = {
        "LR": Pipeline([("s", StandardScaler()),
                        ("m", LogisticRegression(C=0.1, max_iter=1000, random_state=42))]),
        "RF": Pipeline([("s", StandardScaler()),
                        ("m", RandomForestClassifier(n_estimators=500, max_depth=6,
                                                      min_samples_leaf=5, random_state=42,
                                                      class_weight="balanced"))]),
        "GB": Pipeline([("s", StandardScaler()),
                        ("m", GradientBoostingClassifier(n_estimators=300, max_depth=4,
                                                          learning_rate=0.05, subsample=0.8,
                                                          random_state=42))]),
    }

    results = {}
    for name, pipe in pipes.items():
        scores = cross_val_score(pipe, X, y, cv=cv, scoring="roc_auc")
        results[name] = scores.mean()
        log.info(f"  {name} AUC: {scores.mean():.3f} (+/- {scores.std():.3f})")

    # Fit all three models with calibration for soft voting ensemble
    fitted = {}
    for name, pipe in pipes.items():
        cal = CalibratedClassifierCV(pipe, cv=5, method="isotonic")
        cal.fit(X, y)
        fitted[name] = cal

    # Feature importance from RF
    pipes["RF"].fit(X, y)
    importances = pd.DataFrame({
        "feature":    feat_cols,
        "importance": pipes["RF"].named_steps["m"].feature_importances_,
    }).sort_values("importance", ascending=False)
    log.info(f"\n  Top 10 features ({label}):")
    log.info(importances.head(10).to_string(index=False))

    ensemble_auc = max(results.values())
    log.info(f"  Best single model AUC: {ensemble_auc:.3f} — using soft voting ensemble")

    return fitted, importances


# ---------------------------------------------------------------------------
# Compute backend score and display probability
# ---------------------------------------------------------------------------
def compute_scores(df, models, feat_cols, era="modern"):
    """
    Soft voting ensemble — average probabilities from LR, RF, GB.

    Backend score: uncapped, enables all-time ranking.
    - Modern players: raw_prob * 200
    - Pre-1974 players: raw_prob * 150 (normalized down to prevent
      award-heavy pre-1974 model from swamping modern all-time rankings)

    Display probability: 0-100%, with floor and ceiling logic.
    - Floor: players with composite < HOF_FLOOR get capped at 5%
      (truly marginal players — not short-career legitimate cases)
    - Ceiling: players with composite > HOF_THRESHOLD show 100%
    - Minimum backend score check: if backend_score < 10, cap display at 5%
      regardless of composite (catches Sam Lacey types)
    - Rose fix: floor only applies to composite < HOF_FLOOR, not to
      legitimate short careers. Rose has composite ~35 so floor doesn't
      apply — his ~12-18% reflects a real but incomplete case.
    """
    X = df[feat_cols].fillna(0).values

    if isinstance(models, dict):
        all_probs = np.array([m.predict_proba(X)[:, 1] for m in models.values()])
        probs     = all_probs.mean(axis=0)
    else:
        probs = models.predict_proba(X)[:, 1]

    df = df.copy()
    df["raw_prob"] = probs

    # Backend score — peak-weighted composite for all-time ranking.
    #
    # Key design decisions:
    # 1. Floor raw_prob at 0.80 for players above HOF_THRESHOLD composite.
    #    Active players like LeBron get suppressed model probabilities because
    #    the training set treats them as non-HOF (not yet inducted). Flooring
    #    ensures all clear HOF locks start from the same base and are then
    #    differentiated by peak/career/accolades — not by eligibility timing.
    # 2. peak7_vorp × 1.5 — Jordan's peak of 75.3 is the highest ever (nearly
    #    double KG's 55.5). Peak is how most analysts evaluate greatness.
    # 3. all_nba_1 × 2.0 — Kareem's 10 first teams and LeBron's 13 should
    #    distinguish them from KG's 4, preventing KG from ranking ahead of Kareem.
    # 4. career_vorp × 0.1 — rewards longevity at a lower rate so compilers
    #    don't outrank peak performers.
    # 5. all_star × 0.3 — reduced from 0.5 to prevent pure longevity from
    #    dominating over peak quality.
    #
    # Expected top 5: Jordan → LeBron → Kareem → KG → Duncan (any tight order)
    scale = 100 if era == "pre_1974" else 150

    # Floor raw_prob for clear HOF cases so active players aren't penalized
    composite_vals = (
        df["peak7_vorp_sum"].fillna(0)   * 1.0 +
        df["total_vorp"].fillna(0)       * 0.3 +
        df["all_star_count"].fillna(0)   * 1.5 +
        df.get("all_nba_1_count", pd.Series(0, index=df.index)).fillna(0) * 4.0 +
        df.get("career_minutes", df["games_played"] * 30).fillna(0) * 0.001
    )
    adj_probs = np.where(composite_vals > HOF_THRESHOLD,
                         np.maximum(probs, 0.80), probs)

    peak7_vorp  = df["peak7_vorp_sum"].fillna(0)
    career_vorp = df["total_vorp"].fillna(0)
    peak_bonus   = peak7_vorp  * 1.5
    career_bonus = career_vorp * 0.1

    def get(col):
        return df.get(col, pd.Series(0, index=df.index)).fillna(0)

    # Accolade bonus — award weights:
    #   MVP=3, Finals MVP=3  (equal; ring score handles championship context)
    #   All-NBA 1st=5, 2nd=3, 3rd=1
    #   All-Defense 1st=3, DPOY=2  (combined DPOY+All-Def-1st = 5, same as All-NBA 1st)
    #   All-Defense 2nd=1
    #   Ring quality ×8: reduced from 30 to prevent ring-heavy players like Pippen
    #   from ranking ahead of better individual players. Average ring ≈ 0.8pts.
    #   Jordan's 6 dominant rings ≈ +12, Pippen's 6 rings ≈ +7.
    peak7_ws   = df.get("peak7_ws_per_48", pd.Series(0, index=df.index)).fillna(0)
    total_ws   = df["total_ws"].fillna(0)

    # Peak WS scaler: peak7_ws_per_48 × total_ws × 0.5
    # Rewards players who were both efficient at their peak AND played a lot.
    # Jordan (~0.250 × 214 × 0.5 = +26.8) benefits more than KG
    # (~0.210 × 220 × 0.5 = +23.1) because Jordan's peak efficiency was higher.
    peak_ws_bonus = peak7_ws * total_ws * 0.5

    accolade_bonus = (
        get("mvp_count")           * 3  +
        get("all_nba_1_count")     * 5  +
        get("all_nba_2_count")     * 3  +
        get("all_nba_3_count")     * 1  +
        get("all_def_1_count")     * 3  +
        get("dpoy_count")          * 2  +
        get("all_def_2_count")     * 1  +
        get("finals_mvp_count")    * 3  +
        get("ring_quality_score")  * 12
    )

    df["backend_score"] = (adj_probs * scale) + peak_bonus + career_bonus + peak_ws_bonus + accolade_bonus

    # Composite for floor/ceiling decisions (modern stats only)
    composite = (
        df["peak7_vorp_sum"].fillna(0)   * 1.0 +
        df["total_vorp"].fillna(0)       * 0.3 +
        df["all_star_count"].fillna(0)   * 1.5 +
        df.get("all_nba_1_count", pd.Series(0, index=df.index)).fillna(0) * 4.0 +
        df.get("career_minutes", df["games_played"] * 30).fillna(0) * 0.001
    )

    display_prob = probs * 100

    # Players with very high backend scores are clear HOF locks regardless
    # of whether they're inducted yet. LeBron, Jokic, Giannis, Durant, Curry
    # all have backend scores well above any reasonable HOF threshold.
    # Threshold set at 280 — everyone above this is an unambiguous HOFer
    # or certain future HOFer based on their statistical profile.
    high_backend = df["backend_score"] >= 280
    display_prob = np.where(high_backend, 100.0, display_prob)

    # Floor: truly marginal non-inducted players cap at 5%
    inducted = df["hof_inducted"].fillna(0).astype(int).values
    truly_marginal = (composite < HOF_FLOOR) | (df["backend_score"] < 10)
    display_prob = np.where(
        (inducted == 0) & ~high_backend & truly_marginal,
        np.minimum(display_prob, 5.0),
        display_prob
    )

    display_prob = np.clip(display_prob, 0, 100)

    df["hof_probability"] = display_prob.round(1)
    df["composite_score"] = composite.round(1)
    return df


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------
def validate(df):
    log.info("\n" + "="*65)
    log.info("VALIDATION — KNOWN CASES")
    log.info("="*65)

    groups = {
        "Retired HOF locks": [
            "jordami01", "bryanko01", "duncati01", "johnsma02",
            "birdla01",  "abdulka01", "onealsh01", "wadedw01",
            "garneke01", "malonka01", "nowitdi01",  # Dirk retired 2019, inducted 2023
        ],
        "Active / recent locks (not yet eligible)": [
            "jamesle01", "curryst01", "jokicni01", "duranke01",
            "antetgi01",
        ],
        "Borderline / snubs": [
            "cartevi01", "rosede01", "paulch01", "westbru01",
            "howardw01", "irvinkyr01", "hardeja01",
        ],
    }

    cols = ["name", "hof_inducted", "hof_probability", "backend_score",
            "composite_score", "peak7_vorp_sum", "total_vorp",
            "all_star_count", "games_played"]
    cols = [c for c in cols if c in df.columns]

    for group_name, pids in groups.items():
        subset = df[df["player_id"].isin(pids)].sort_values(
            "hof_probability", ascending=False
        )
        log.info(f"\n{group_name}:")
        log.info(subset[cols].to_string(index=False))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def run():
    # Load data
    features  = pd.read_csv(FEATURES_PATH)
    conn      = sqlite3.connect(DB_PATH)
    awards    = pd.read_sql("SELECT * FROM awards", conn)
    seasons   = pd.read_sql("SELECT * FROM seasons", conn)
    player_meta = pd.read_sql("SELECT player_id, to_year FROM players", conn)

    try:
        summaries = pd.read_csv("data/raw/Team Summaries.csv")
    except FileNotFoundError:
        summaries = pd.DataFrame()
        log.warning("No team summaries — ring quality using defaults")

    # Load aging curves if available
    aging_curves = pd.DataFrame()
    try:
        aging_curves = pd.read_csv("data/processed/aging_curves.csv")
        log.info(f"Loaded aging curves: {len(aging_curves)} rows")
    except FileNotFoundError:
        log.warning("No aging curves found — active players won't be projected")

    # Ring quality
    log.info("Computing ring quality...")
    ring_quality = compute_ring_quality(seasons, summaries)

    # Apply thresholds
    df = features.copy()
    df = df[df["games_played"] >= MIN_GAMES].copy()
    df = df[df["career_ws"].fillna(0) >= MIN_WS].copy()
    log.info(f"After thresholds (min {MIN_GAMES} games, min {MIN_WS} WS): {len(df)} players")

    # Split pre-1974 and modern
    pre74_mask = df["peak7_stl_per75"].isna() & df["peak7_blk_per75"].isna()
    pre74_df   = df[pre74_mask].copy()
    modern_df  = df[~pre74_mask].copy()
    log.info(f"Pre-1974: {len(pre74_df)}, Modern: {len(modern_df)}")

    # International players excluded from training
    intl = get_international_players()

    # ---- Modern model ----
    modern_df = build_features(modern_df, awards, ring_quality, aging_curves, pre_1974=False)

    train_modern = modern_df[
        (modern_df["retired"] == 1) &
        (~modern_df["player_id"].isin(intl))
    ].copy()

    modern_feat_cols = [c for c in MODERN_FEATURES if c in modern_df.columns]
    modern_models, modern_imp = train_model(train_modern, modern_feat_cols, "Modern")
    modern_imp.to_csv("data/processed/feature_importances_modern.csv", index=False)

    modern_df = compute_scores(modern_df, modern_models, modern_feat_cols, era="modern")

    # ---- Pre-1974 model ----
    pre74_df  = build_features(pre74_df, awards, ring_quality, pre_1974=True)

    train_pre74 = pre74_df[pre74_df["retired"] == 1].copy()

    pre74_feat_cols = [c for c in PRE74_FEATURES if c in pre74_df.columns]
    if len(train_pre74[train_pre74["hof_inducted"] == 1]) >= 5:
        pre74_models, pre74_imp = train_model(train_pre74, pre74_feat_cols, "Pre-1974")
        pre74_df = compute_scores(pre74_df, pre74_models, pre74_feat_cols, era="pre_1974")
    else:
        log.warning("Not enough pre-1974 HOF players — using modern model")
        pre74_df = compute_scores(pre74_df, modern_models, modern_feat_cols, era="pre_1974")

    # ---- Combine and save ----
    all_df = pd.concat([modern_df, pre74_df], ignore_index=True)

    # Merge to_year for eligibility filtering
    all_df = all_df.merge(player_meta, on="player_id", how="left")

    out_cols = ["player_id", "name", "hof_inducted", "retired",
                "hof_probability", "backend_score", "composite_score",
                "archetype_name", "peak7_vorp_sum", "total_vorp",
                "career_vorp", "games_played", "all_star_count",
                "all_nba_1_count", "ring_quality_score"]
    out_cols = [c for c in out_cols if c in all_df.columns]
    all_df[out_cols].sort_values("backend_score", ascending=False).to_csv(
        "data/processed/hof_predictions.csv", index=False
    )
    log.info("\nSaved to data/processed/hof_predictions.csv")

    # Validate
    validate(all_df)

    # All-time ranking (backend score)
    log.info("\n" + "="*65)
    log.info("ALL-TIME RANKING (backend score — uncapped)")
    log.info("="*65)
    top50 = all_df.nlargest(50, "backend_score")[
        ["name", "backend_score", "hof_probability", "peak7_vorp_sum",
         "total_vorp", "all_star_count", "hof_inducted"]
    ]
    log.info(top50.to_string(index=False))

    # Top snubs — only HOF-eligible players
    # Eligible = retired AND last season was 5+ years ago (2020 or earlier for 2025)
    # This correctly excludes LeBron, Durant, Curry, Giannis, Harden, CP3
    # while including Dirk (2019), Kobe (2016), Wade (2019) etc.
    CURRENT_SEASON = 2025
    HOF_ELIGIBILITY_YEARS = 5

    if "to_year" in all_df.columns:
        eligible_mask = (
            (all_df["retired"] == 1) &
            (all_df["to_year"] <= CURRENT_SEASON - HOF_ELIGIBILITY_YEARS)
        )
    else:
        eligible_mask = (all_df.get("hof_eligible", all_df["retired"]) == 1)

    log.info("\n" + "="*65)
    log.info("TOP SNUBS — HOF-eligible retired non-HOF players")
    log.info("="*65)
    snubs = all_df[
        eligible_mask &
        (all_df["hof_inducted"] == 0) &
        (~all_df["player_id"].isin(intl)) &
        # Exclude pre-1974 players — they lack advanced stats and belong
        # in the pre-1974 model results, not the modern snubs list
        (all_df["career_vorp"].fillna(0) != 0)
    ].nlargest(20, "hof_probability")
    log.info(snubs[["name", "hof_probability", "backend_score",
                     "peak7_vorp_sum", "total_vorp",
                     "all_star_count"]].to_string(index=False))

    # Weakest HOF inductees
    log.info("\n" + "="*65)
    log.info("WEAKEST HOF INDUCTEES by probability")
    log.info("="*65)
    weakest = all_df[all_df["hof_inducted"] == 1].nsmallest(15, "hof_probability")
    log.info(weakest[["name", "hof_probability", "backend_score",
                       "peak7_vorp_sum", "total_vorp",
                       "all_star_count"]].to_string(index=False))

    return all_df


if __name__ == "__main__":
    run()

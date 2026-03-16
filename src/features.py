"""
features.py

Builds the full feature vector for every player in the database.
These features feed directly into the archetype clustering and HOF classifier.

Feature groups:
    1. Peak 3-year window  — absolute ceiling (best consecutive 3 seasons)
    2. Peak 7-year window  — sustained prime (best consecutive 7 seasons)
    3. Career averages     — longevity and total contribution
    4. Efficiency metrics  — era-adjusted TS%, scoring quality
    5. Award score         — weighted accolades
    6. Championship score  — VORP-share of championships
    7. Three-point profile — era-adjusted, only for post-1980 seasons
    8. Defensive profile   — position-aware blocks/steals vs DBPM

Usage:
    python src/features.py
"""

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sqlalchemy import create_engine, text

sys.path.insert(0, "src")

DB_PATH = "data/hof.db"
Path("data/processed").mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Championship reference — every NBA champion by season
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

# ---------------------------------------------------------------------------
# Finals MVP reference — every winner since 1969 mapped to BBref player_id
# ---------------------------------------------------------------------------
FINALS_MVP = {
    1969: "westje01",      # Jerry West (only Finals MVP from losing team)
    1970: "reedwi01",      # Willis Reed
    1971: "abdulka01",     # Kareem Abdul-Jabbar (then Lew Alcindor)
    1972: "chambwi01",     # Wilt Chamberlain
    1973: "reedwi01",      # Willis Reed
    1974: "cowenda01",     # Dave Cowens
    1975: "barryri01",     # Rick Barry
    1976: "coweda01",      # Dave Cowens -- note: Jo Jo White won this, correcting below
    1976: "whitejo01",     # Jo Jo White
    1977: "waltobi01",     # Bill Walton
    1978: "unselde01",     # Wes Unseld
    1979: "guidede01",     # Dennis Johnson
    1980: "johnsma02",     # Magic Johnson
    1981: "maxwece01",     # Cedric Maxwell
    1982: "johnsma02",     # Magic Johnson
    1983: "malonmo01",     # Moses Malone
    1984: "birdla01",      # Larry Bird
    1985: "abdulka01",     # Kareem Abdul-Jabbar
    1986: "birdla01",      # Larry Bird
    1987: "johnsma02",     # Magic Johnson
    1988: "worthja01",     # James Worthy
    1989: "thomais02",     # Isiah Thomas
    1990: "thomais02",     # Isiah Thomas
    1991: "jordami01",     # Michael Jordan
    1992: "jordami01",     # Michael Jordan
    1993: "jordami01",     # Michael Jordan
    1994: "olajuha01",     # Hakeem Olajuwon
    1995: "olajuha01",     # Hakeem Olajuwon
    1996: "jordami01",     # Michael Jordan
    1997: "jordami01",     # Michael Jordan
    1998: "jordami01",     # Michael Jordan
    1999: "duncati01",     # Tim Duncan
    2000: "onealsh01",     # Shaquille O'Neal
    2001: "onealsh01",     # Shaquille O'Neal
    2002: "onealsh01",     # Shaquille O'Neal
    2003: "duncati01",     # Tim Duncan
    2004: "chaunde01",     # Chauncey Billups
    2005: "duncati01",     # Tim Duncan
    2006: "wadedw01",      # Dwyane Wade
    2007: "parketo01",     # Tony Parker
    2008: "piercpa01",     # Paul Pierce
    2009: "bryankobe01",   # Kobe Bryant -- corrected below
    2009: "bryanko01",     # Kobe Bryant
    2010: "bryanko01",     # Kobe Bryant
    2011: "nowitdi01",     # Dirk Nowitzki
    2012: "jamesle01",     # LeBron James
    2013: "jamesle01",     # LeBron James
    2014: "leonaka01",     # Kawhi Leonard
    2015: "iguodan01",     # Andre Iguodala
    2016: "jamesle01",     # LeBron James
    2017: "duranke01",     # Kevin Durant
    2018: "duranke01",     # Kevin Durant
    2019: "leonaka01",     # Kawhi Leonard
    2020: "jamesle01",     # LeBron James
    2021: "antetgi01",     # Giannis Antetokounmpo
    2022: "curryst01",     # Stephen Curry
    2023: "jokicni01",     # Nikola Jokic
    2024: "tatumja01",     # Jayson Tatum
}

# Award weights
AWARD_WEIGHTS = {
    "MVP":          10.0,
    "Finals_MVP":    7.0,
    "DPOY":          5.0,
    "Scoring_Champ": 3.0,
    "ROY":           2.0,
    "All_NBA_1":     5.0,
    "All_NBA_2":     3.0,
    "All_NBA_3":     1.5,
    "All_Defense_1": 3.0,
    "All_Defense_2": 1.5,
    "All_Star":      1.0,
}

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_data(db_path: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    engine  = create_engine(f"sqlite:///{db_path}")
    players = pd.read_sql("SELECT * FROM players", engine)
    seasons = pd.read_sql("SELECT * FROM seasons", engine)
    awards  = pd.read_sql("SELECT * FROM awards",  engine)
    log.info(f"Loaded {len(players)} players, {len(seasons)} seasons, {len(awards)} awards")
    return players, seasons, awards


# ---------------------------------------------------------------------------
# Era-adjusted league averages
# Used to normalize stats relative to the league in a given season
# ---------------------------------------------------------------------------
def compute_era_averages(seasons: pd.DataFrame) -> pd.DataFrame:
    """
    Compute league-average per-75 stats by season.
    Used to express individual stats as deviations from league average.
    """
    nba = seasons[seasons["lg"].isin(["NBA", "BAA"]) | seasons["lg"].isna()]

    # Only use players with meaningful minutes
    nba = nba[nba["g"] >= 20]

    era_cols = ["pts_per75", "ast_per75", "trb_per75", "stl_per75",
                "blk_per75", "ts_pct", "fg3a", "fg3"]

    era_avgs = (
        nba.groupby("season")[era_cols]
        .mean()
        .reset_index()
        .rename(columns={c: f"lg_{c}" for c in era_cols})
    )

    # Era-adjusted 3-point rate — fg3a per 75 relative to league average
    # This lets us compare 1990s shooters to 2020s shooters fairly
    if "lg_fg3a" in era_avgs.columns:
        era_avgs["lg_3p_rate"] = era_avgs["lg_fg3a"]

    return era_avgs


# ---------------------------------------------------------------------------
# Peak window calculation
# Core of the feature set — find best N consecutive seasons
# ---------------------------------------------------------------------------
def best_consecutive_window(
    player_seasons: pd.DataFrame,
    n: int,
    value_col: str = "vorp"
) -> pd.DataFrame:
    """
    Find the best N consecutive seasons for a player by sum of value_col.
    Returns the rows corresponding to that window.

    'Consecutive' means consecutive calendar seasons (no gaps).
    If a player has fewer than N qualifying seasons, use all available.
    """
    df = player_seasons.sort_values("season").reset_index(drop=True)
    df = df[df[value_col].notna()]

    if len(df) == 0:
        return pd.DataFrame()

    if len(df) <= n:
        return df

    best_sum   = -np.inf
    best_start = 0

    for i in range(len(df) - n + 1):
        window = df.iloc[i:i + n]

        # Check that seasons are truly consecutive (no gaps > 1 year)
        season_diffs = window["season"].diff().dropna()
        if (season_diffs > 2).any():  # allow 1-year gap for strikes/lockouts
            continue

        window_sum = window[value_col].sum()
        if window_sum > best_sum:
            best_sum   = window_sum
            best_start = i

    return df.iloc[best_start:best_start + n]


# ---------------------------------------------------------------------------
# Per-75 averages for a window of seasons
# ---------------------------------------------------------------------------
def window_averages(window: pd.DataFrame) -> dict:
    """Compute weighted averages for a peak window, weighted by games played."""
    if window.empty:
        return {}

    g = window["g"].fillna(0)
    total_g = g.sum()
    if total_g == 0:
        return {}

    stats = ["pts_per75", "ast_per75", "trb_per75", "stl_per75", "blk_per75",
             "bpm", "obpm", "dbpm", "ws_per_48", "ts_pct", "usg_pct", "per",
             "fg3a", "fg3", "fg3_pct"]

    result = {}
    for stat in stats:
        if stat in window.columns and window[stat].notna().any():
            result[stat] = np.average(
                window[stat].fillna(0),
                weights=g
            )

    result["vorp_sum"]   = window["vorp"].sum()   if "vorp" in window.columns else np.nan
    result["ws_sum"]     = window["ws"].sum()     if "ws"   in window.columns else np.nan
    result["seasons_n"]  = len(window)

    return result


# ---------------------------------------------------------------------------
# Three-point era adjustment
# ---------------------------------------------------------------------------
def compute_3p_era_adjusted(
    seasons: pd.DataFrame,
    era_avgs: pd.DataFrame
) -> pd.DataFrame:
    """
    For each player-season after 1980, compute era-adjusted 3-point rate.

    era_adj_3p_rate = (player fg3a_per75 - league_avg_fg3a) / league_std_fg3a

    This puts 1990s shooters and 2020s shooters on the same scale.
    Pre-1980 seasons get NaN (3-point line didn't exist).
    """
    s = seasons.merge(era_avgs[["season", "lg_fg3a"]], on="season", how="left")

    # Compute league std by season
    nba = seasons[seasons["g"] >= 20]
    lg_std = (
        nba.groupby("season")["fg3a"]
        .std()
        .reset_index()
        .rename(columns={"fg3a": "lg_fg3a_std"})
    )
    s = s.merge(lg_std, on="season", how="left")

    # Only apply for post-1980 seasons where 3s existed
    mask = s["season"] >= 1980
    s.loc[mask, "fg3a_era_adj"] = (
        (s.loc[mask, "fg3a"].fillna(0) - s.loc[mask, "lg_fg3a"].fillna(0)) /
        s.loc[mask, "lg_fg3a_std"].replace(0, np.nan)
    )
    s.loc[~mask, "fg3a_era_adj"] = np.nan

    # Era-adjusted 3P makes
    s.loc[mask, "fg3_era_adj"] = (
        (s.loc[mask, "fg3"].fillna(0) - s.loc[mask, "lg_fg3"].fillna(0))
        if "lg_fg3" in s.columns
        else s.loc[mask, "fg3a_era_adj"] * s.loc[mask, "fg3_pct"].fillna(0.35)
    )

    return s


# ---------------------------------------------------------------------------
# Award score
# ---------------------------------------------------------------------------
def compute_award_score(player_id: str, awards: pd.DataFrame) -> float:
    player_awards = awards[awards["player_id"] == player_id]
    base_score = player_awards["award_weight"].sum() if not player_awards.empty else 0.0

    # Add Finals MVP from hardcoded dict (not in Kaggle dataset)
    finals_mvp_count = sum(1 for pid in FINALS_MVP.values() if pid == player_id)
    finals_mvp_score = finals_mvp_count * AWARD_WEIGHTS["Finals_MVP"]

    return base_score + finals_mvp_score


# ---------------------------------------------------------------------------
# Championship score
# VORP share of team total VORP in each championship season
# ---------------------------------------------------------------------------
def compute_championship_scores(seasons: pd.DataFrame) -> pd.DataFrame:
    """
    For each championship season, compute each player's VORP share
    of their team's total VORP. Sum across all rings.

    championship_score = sum(player_vorp / max(team_vorp_total, 1))
    """
    champ_records = []

    for season, team_abbrev in NBA_CHAMPIONS.items():
        # Get all players on championship team that season
        team_season = seasons[
            (seasons["season"] == season) &
            (seasons["team"] == team_abbrev) &
            (seasons["vorp"].notna())
        ]

        if team_season.empty:
            continue

        team_vorp_total = max(team_season["vorp"].sum(), 0.1)

        for _, row in team_season.iterrows():
            player_vorp  = max(row["vorp"], 0)  # negative VORP gets 0 share
            vorp_share   = player_vorp / team_vorp_total
            champ_records.append({
                "player_id":        row["player_id"],
                "season":           season,
                "ring_vorp_share":  vorp_share,
                "ring_vorp":        player_vorp,
                "champion_team":    team_abbrev,
            })

    champ_df = pd.DataFrame(champ_records)
    if champ_df.empty:
        return pd.DataFrame(columns=["player_id", "championship_score", "rings"])

    summary = (
        champ_df.groupby("player_id")
        .agg(
            championship_score=("ring_vorp_share", "sum"),
            rings=("season", "count"),
        )
        .reset_index()
    )
    return summary


# ---------------------------------------------------------------------------
# Defensive profile score
# Position-aware: cross-references blocks/steals with DBPM
# ---------------------------------------------------------------------------
def compute_defensive_score(peak_window: pd.DataFrame) -> float:
    """
    Composite defensive score that avoids rewarding gambling defenders.

    defensive_score = (
        dbpm_avg * 2.0
        + blk_per75 * blk_weight    (high if dbpm > 0, penalized if dbpm < -1)
        + stl_per75 * stl_weight    (same logic)
    )

    The idea: blocks and steals only count positively if the player's
    overall defensive impact (DBPM) is also positive.
    """
    if peak_window.empty:
        return 0.0

    avg = peak_window.mean(numeric_only=True)

    dbpm    = avg.get("dbpm",      0.0) or 0.0
    blk     = avg.get("blk_per75", 0.0) or 0.0
    stl     = avg.get("stl_per75", 0.0) or 0.0

    # Weight blocks and steals by defensive impact
    # Positive DBPM amplifies them, negative DBPM discounts them
    dbpm_multiplier = max(0.5 + dbpm * 0.2, 0.1)

    defensive_score = (
        dbpm    * 2.0 +
        blk     * dbpm_multiplier +
        stl     * dbpm_multiplier
    )

    return round(defensive_score, 4)


# ---------------------------------------------------------------------------
# Main feature builder
# ---------------------------------------------------------------------------
def build_features(db_path: str = DB_PATH) -> pd.DataFrame:
    players, seasons, awards = load_data(db_path)

    # Only NBA/BAA seasons
    seasons = seasons[seasons["lg"].isin(["NBA", "BAA"]) | seasons["lg"].isna()]

    # Minimum games filter
    seasons = seasons[seasons["g"] >= 20]

    # Era averages for normalization
    era_avgs = compute_era_averages(seasons)

    # Era-adjusted 3-point stats
    seasons  = compute_3p_era_adjusted(seasons, era_avgs)

    # Championship scores for all players
    log.info("Computing championship scores...")
    champ_scores = compute_championship_scores(seasons)

    # Era-average TS% by season for efficiency comparison
    era_ts = era_avgs[["season", "lg_ts_pct"]].copy() if "lg_ts_pct" in era_avgs.columns else None

    feature_rows = []
    total = len(players)

    log.info(f"Building features for {total} players...")

    for i, player in players.iterrows():
        pid = player["player_id"]

        # Get this player's seasons, sorted by season
        ps = seasons[seasons["player_id"] == pid].sort_values("season")

        # Skip players with no season data
        if ps.empty:
            continue

        # ---- Peak windows ----
        # Use vorp as the ranking column if available, fall back to pts_per75
        # for pre-1974 players who have no VORP data
        rank_col = "vorp" if ps["vorp"].notna().any() else "pts_per75"
        peak3_window  = best_consecutive_window(ps, 3, rank_col)
        peak7_window  = best_consecutive_window(ps, 7, rank_col)

        peak3_avgs    = window_averages(peak3_window)
        peak7_avgs    = window_averages(peak7_window)
        career_avgs   = window_averages(ps)

        # ---- Award score ----
        award_score   = compute_award_score(pid, awards)

        # ---- Championship score ----
        champ_row     = champ_scores[champ_scores["player_id"] == pid]
        champ_score   = float(champ_row["championship_score"].values[0]) if not champ_row.empty else 0.0
        rings         = int(champ_row["rings"].values[0])                 if not champ_row.empty else 0

        # ---- Defensive score (from peak 7) ----
        def_score     = compute_defensive_score(peak7_window)

        # ---- Career TS% vs era average ----
        ts_vs_era = np.nan
        if "ts_pct" in ps.columns and era_ts is not None:
            ps_ts = ps[ps["ts_pct"].notna()].merge(era_ts, on="season", how="left")
            if not ps_ts.empty:
                ts_vs_era = (ps_ts["ts_pct"] - ps_ts["lg_ts_pct"]).mean()

        # ---- Assemble feature row ----
        row = {
            "player_id":   pid,
            "name":        player.get("name"),
            "hof_inducted": int(player.get("hof_inducted", 0) or 0),
            "retired":     int(player.get("retired", 1) or 1),
            "seasons_played": len(ps),
            "games_played": ps["g"].sum(),
            # career_minutes is a better longevity proxy than games_played —
            # it discounts end-of-career bench stints (e.g. late-career KG)
            # and properly weights players who played heavy minutes vs. spot duty.
            "career_minutes": (ps["g"] * ps["mp_per_game"]).sum() if "mp_per_game" in ps.columns else ps["g"].sum() * 30,
            "award_score":  award_score,
            "championship_score": champ_score,
            "rings":        rings,
            "defensive_score": def_score,
            "ts_vs_era_avg": ts_vs_era,
            "career_vorp":  ps["vorp"].sum() if "vorp" in ps.columns else np.nan,
            "career_ws":    ps["ws"].sum()   if "ws"   in ps.columns else np.nan,
        }

        # Add peak3, peak7, career averages with prefixes
        for prefix, avgs in [("peak3", peak3_avgs), ("peak7", peak7_avgs), ("career", career_avgs)]:
            for stat, val in avgs.items():
                row[f"{prefix}_{stat}"] = val

        feature_rows.append(row)

    features = pd.DataFrame(feature_rows)
    log.info(f"Features built: {len(features)} players")

    # Save
    out_path = "data/processed/player_features.csv"
    features.to_csv(out_path, index=False)
    log.info(f"Saved to {out_path}")

    # Quick sanity check
    hof_features = features[features["hof_inducted"] == 1]
    log.info(f"\n=== HOF Player Averages (sanity check) ===")
    check_cols = ["peak7_vorp_sum", "peak7_pts_per75", "award_score",
                  "championship_score", "rings", "career_vorp"]
    check_cols = [c for c in check_cols if c in hof_features.columns]
    log.info(f"\n{hof_features[check_cols].mean().round(2).to_string()}")

    log.info(f"\n=== Non-HOF Player Averages ===")
    non_hof = features[features["hof_inducted"] == 0]
    log.info(f"\n{non_hof[check_cols].mean().round(2).to_string()}")

    return features


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    build_features()

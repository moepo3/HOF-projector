"""
ingest.py

Loads the Kaggle 'NBA/ABA/BAA Stats' dataset CSVs into the local SQLite database.

Files used:
    data/raw/Player Per Game.csv       — per-game counting stats by season
    data/raw/Advanced.csv              — advanced metrics (VORP, BPM, WS, etc.)
    data/raw/Player Career Info.csv    — bio info (height, weight, HOF status)
    data/raw/End of Season Teams.csv   — All-NBA, All-Defense selections
    data/raw/All-Star Selections.csv   — All-Star appearances
    data/raw/Player Award Shares.csv   — MVP, DPOY, ROY voting (includes winners)
    data/raw/Team Summaries.csv        — team pace by season (for per-75 normalization)

Usage:
    python src/ingest.py
"""

import logging
import sys
from pathlib import Path

import pandas as pd
import numpy as np
from sqlalchemy import create_engine, text

sys.path.insert(0, "src")
from schema import init_db

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
DB_PATH = "data/hof.db"
RAW_DIR = Path("data/raw")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
)
log = logging.getLogger(__name__)

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
# Helpers
# ---------------------------------------------------------------------------
def read_csv(filename: str) -> pd.DataFrame:
    path = RAW_DIR / filename
    if not path.exists():
        log.error(f"File not found: {path}")
        return pd.DataFrame()
    df = pd.read_csv(path)
    log.info(f"Loaded {filename}: {len(df)} rows, columns: {list(df.columns)}")
    return df

def size_bucket(height_inches) -> str:
    if pd.isna(height_inches):
        return "Unknown"
    h = float(height_inches)
    if h <= 77:
        return "Guard"
    if h <= 81:
        return "Wing"
    return "Big"

# ---------------------------------------------------------------------------
# Players table
# ---------------------------------------------------------------------------
def ingest_players(engine):
    log.info("--- Ingesting players ---")
    df = read_csv("Player Career Info.csv")
    if df.empty:
        return

    log.info(f"Columns: {list(df.columns)}")

    # Standardize column names — lowercase and strip whitespace
    df.columns = [c.lower().strip() for c in df.columns]

    # Actual columns: player, player_id, pos, ht_in_in, wt, birth_date, colleges, from, to, debut, hof
    rename = {
        "player":     "name",
        "player_id":  "player_id",
        "hof":        "hof_inducted",
        "ht_in_in":   "height_inches",
        "wt":         "weight_lbs",
        "from":       "from_year",
        "to":         "to_year",
        "birth_date": "birth_date",
    }
    df = df.rename(columns={k: v for k, v in rename.items() if k in df.columns})

    # Extract birth year from birth_date string
    if "birth_date" in df.columns:
        df["birth_year"] = pd.to_datetime(df["birth_date"], errors="coerce").dt.year

    # Convert height
    if "height_inches" in df.columns:
        df["height_inches"] = pd.to_numeric(df["height_inches"], errors="coerce")
        df["size_bucket"]   = df["height_inches"].apply(size_bucket)
    else:
        df["size_bucket"] = "Unknown"

    # HOF flag — column contains True/False as strings
    if "hof_inducted" in df.columns:
        hof_str = df["hof_inducted"].astype(str).str.strip().str.lower()
        df["hof_inducted"] = hof_str == "true"
    else:
        df["hof_inducted"] = False

    # Active / retired
    current_season   = 2025
    to_year_series   = pd.to_numeric(df.get("to_year", pd.Series(dtype=float)), errors="coerce")
    df["retired"]      = to_year_series < current_season
    df["hof_eligible"] = df["retired"] & (to_year_series <= 2019)

    # BBref URL
    df["bbref_url"] = df["player_id"].apply(
        lambda pid: f"https://www.basketball-reference.com/players/{str(pid)[0]}/{pid}.html"
        if pd.notna(pid) else None
    )

    keep = ["player_id", "name", "hof_inducted", "hof_eligible", "retired",
            "from_year", "to_year", "birth_year", "height_inches", "weight_lbs",
            "size_bucket", "bbref_url"]
    keep = [c for c in keep if c in df.columns]

    df = df[keep].drop_duplicates(subset=["player_id"])
    df.to_sql("players", engine, if_exists="replace", index=False)
    log.info(f"Players ingested: {len(df)} rows")
    hof_count = df["hof_inducted"].sum() if "hof_inducted" in df.columns else 0
    log.info(f"Hall of Famers in dataset: {hof_count}")


# ---------------------------------------------------------------------------
# Seasons table — merge per-game + advanced
# ---------------------------------------------------------------------------
def ingest_seasons(engine):
    log.info("--- Ingesting seasons ---")

    pg  = read_csv("Player Per Game.csv")
    adv = read_csv("Advanced.csv")

    if pg.empty and adv.empty:
        return

    # Standardize columns
    for df in [pg, adv]:
        df.columns = [c.lower().strip() for c in df.columns]

    # Both files share: player_id, season, team, age
    merge_cols = ["player_id", "season", "team"]

    if not pg.empty and not adv.empty:
        # Drop duplicate columns that appear in both (keep from pg)
        adv_only = [c for c in adv.columns if c not in pg.columns or c in merge_cols]
        merged   = pg.merge(adv[adv_only], on=merge_cols, how="left")
    elif not pg.empty:
        merged = pg
    else:
        merged = adv

    log.info(f"Merged seasons shape: {merged.shape}")

    # Rename to match schema
    rename = {
        "tm":       "team",
        "lg":       "lg",
        "g":        "g",
        "gs":       "gs",
        "mp_per_game": "mp",
        "mp":       "mp",
        "pts_per_game": "pts",
        "pts":      "pts",
        "ast_per_game": "ast",
        "ast":      "ast",
        "trb_per_game": "trb",
        "trb":      "trb",
        "orb_per_game": "orb",
        "orb":      "orb",
        "drb_per_game": "drb",
        "drb":      "drb",
        "stl_per_game": "stl",
        "stl":      "stl",
        "blk_per_game": "blk",
        "blk":      "blk",
        "tov_per_game": "tov",
        "tov":      "tov",
        "pf_per_game":  "pf",
        "pf":       "pf",
        "fg_per_game":  "fg",
        "fg":       "fg",
        "fga_per_game": "fga",
        "fga":      "fga",
        "fg_percent":   "fg_pct",
        "fg_pct":       "fg_pct",
        "x3p_per_game": "fg3",
        "x3p":      "fg3",
        "x3pa_per_game": "fg3a",
        "x3pa":     "fg3a",
        "x3p_percent":  "fg3_pct",
        "x3p_pct":      "fg3_pct",
        "ft_per_game":  "ft",
        "ft":       "ft",
        "fta_per_game": "fta",
        "fta":      "fta",
        "ft_percent":   "ft_pct",
        "ft_pct":       "ft_pct",
        "per":      "per",
        "ts_percent":   "ts_pct",
        "ts_pct":       "ts_pct",
        "usg_percent":  "usg_pct",
        "usg_pct":      "usg_pct",
        "ows":      "ows",
        "dws":      "dws",
        "ws":       "ws",
        "ws_48":    "ws_per_48",
        "obpm":     "obpm",
        "dbpm":     "dbpm",
        "bpm":      "bpm",
        "vorp":     "vorp",
    }
    merged = merged.rename(columns={k: v for k, v in rename.items() if k in merged.columns})

    # Drop rows with no player_id or season
    merged = merged[merged["player_id"].notna() & merged["season"].notna()]

    # Remove duplicate columns from merge
    merged = merged.loc[:, ~merged.columns.duplicated()]

    schema_cols = [
        "player_id", "season", "age", "team", "lg", "g", "gs", "mp",
        "pts", "ast", "trb", "orb", "drb", "stl", "blk", "tov", "pf",
        "fg", "fga", "fg_pct", "fg3", "fg3a", "fg3_pct", "ft", "fta", "ft_pct",
        "per", "ts_pct", "usg_pct", "ows", "dws", "ws", "ws_per_48",
        "obpm", "dbpm", "bpm", "vorp",
    ]
    keep = [c for c in schema_cols if c in merged.columns]
    merged[keep].to_sql("seasons", engine, if_exists="replace", index=False)
    log.info(f"Seasons ingested: {len(merged)} rows")


# ---------------------------------------------------------------------------
# Awards table
# ---------------------------------------------------------------------------
def ingest_awards(engine):
    log.info("--- Ingesting awards ---")
    records = []

    # All-Star selections
    allstar = read_csv("All-Star Selections.csv")
    if not allstar.empty:
        allstar.columns = [c.lower().strip() for c in allstar.columns]
        id_col   = "player_id" if "player_id" in allstar.columns else "player"
        seas_col = "season"    if "season"    in allstar.columns else "year"
        for _, row in allstar.iterrows():
            records.append({
                "player_id":    row.get(id_col),
                "season":       row.get(seas_col),
                "award":        "All_Star",
                "award_weight": AWARD_WEIGHTS["All_Star"],
            })
        log.info(f"All-Star records: {len(allstar)}")

    # End of Season Teams (All-NBA, All-Defense)
    eost = read_csv("End of Season Teams.csv")
    if not eost.empty:
        eost.columns = [c.lower().strip() for c in eost.columns]
        log.info(f"End of Season Teams columns: {list(eost.columns)}")

        team_map = {
            "all-nba":            {"1st": "All_NBA_1", "2nd": "All_NBA_2", "3rd": "All_NBA_3"},
            "all-defensive":      {"1st": "All_Defense_1", "2nd": "All_Defense_2"},
            "all-defense":        {"1st": "All_Defense_1", "2nd": "All_Defense_2"},
        }

        id_col   = "player_id" if "player_id" in eost.columns else "player"
        seas_col = "season"

        for _, row in eost.iterrows():
            team_type = str(row.get("type", "")).lower()
            number    = str(row.get("number_tm", row.get("number", ""))).lower()

            for team_key, number_map in team_map.items():
                if team_key in team_type:
                    for num_key, award_label in number_map.items():
                        if num_key in number:
                            records.append({
                                "player_id":    row.get(id_col),
                                "season":       row.get(seas_col),
                                "award":        award_label,
                                "award_weight": AWARD_WEIGHTS[award_label],
                            })

    # Player Award Shares — captures MVP, DPOY, ROY winners
    award_shares = read_csv("Player Award Shares.csv")
    if not award_shares.empty:
        award_shares.columns = [c.lower().strip() for c in award_shares.columns]
        log.info(f"Award Shares columns: {list(award_shares.columns)}")

        award_map = {
            "nba mvp":    "MVP",
            "dpoy":       "DPOY",
            "roy":        "ROY",
            "finals mvp": "Finals_MVP",
        }

        id_col   = "player_id" if "player_id" in award_shares.columns else "player"
        seas_col = "season"
        award_col = "award" if "award" in award_shares.columns else "award_id"

        # Only keep winners (first place / winner flag)
        winner_col = None
        for col in ["winner", "first", "won"]:
            if col in award_shares.columns:
                winner_col = col
                break

        for _, row in award_shares.iterrows():
            # Skip non-winners if we have a winner column
            if winner_col and not row.get(winner_col):
                continue

            raw_award = str(row.get(award_col, "")).lower()
            for key, label in award_map.items():
                if key in raw_award:
                    records.append({
                        "player_id":    row.get(id_col),
                        "season":       row.get(seas_col),
                        "award":        label,
                        "award_weight": AWARD_WEIGHTS[label],
                    })
                    break

    if records:
        awards_df = pd.DataFrame(records).dropna(subset=["player_id"])
        awards_df.to_sql("awards", engine, if_exists="replace", index=False)
        log.info(f"Awards ingested: {len(awards_df)} rows")
        log.info(f"Award breakdown:\n{awards_df['award'].value_counts()}")


# ---------------------------------------------------------------------------
# League pace (from Team Summaries)
# ---------------------------------------------------------------------------
def ingest_pace(engine):
    log.info("--- Ingesting league pace ---")
    df = read_csv("Team Summaries.csv")
    if df.empty:
        return

    df.columns = [c.lower().strip() for c in df.columns]
    log.info(f"Team Summaries columns: {list(df.columns)}")

    pace_col = None
    for col in ["pace", "pace_factor"]:
        if col in df.columns:
            pace_col = col
            break

    if not pace_col:
        log.warning("No pace column found in Team Summaries")
        return

    # Average pace across all teams per season
    pace = (
        df[df["lg"] == "NBA"]
        .groupby("season")[pace_col]
        .mean()
        .reset_index()
        .rename(columns={pace_col: "pace"})
    )
    pace["lg"] = "NBA"
    pace.to_sql("league_seasons", engine, if_exists="replace", index=False)
    pace.to_csv(RAW_DIR / "league_pace.csv", index=False)
    log.info(f"League pace ingested: {len(pace)} seasons")


# ---------------------------------------------------------------------------
# Compute per-75 possession stats
# ---------------------------------------------------------------------------
def compute_per75(engine):
    log.info("--- Computing per-75 possession stats ---")

    seasons = pd.read_sql("SELECT * FROM seasons", engine)
    pace    = pd.read_sql("SELECT season, pace FROM league_seasons", engine)

    if pace.empty:
        log.warning("No pace data — skipping per-75 computation")
        return

    seasons = seasons.merge(pace, on="season", how="left")

    # per_75 = (stat_per_game / possessions_per_game) * 75
    # possessions_per_game ≈ pace * mp / 48
    mp_safe = seasons["mp"].replace(0, np.nan)
    poss_per_game = seasons["pace"] * mp_safe / 48

    for stat in ["pts", "ast", "trb", "stl", "blk"]:
        col = f"{stat}_per75"
        if stat in seasons.columns:
            seasons[col] = (seasons[stat] / poss_per_game) * 75

    # Write back
    seasons.to_sql("seasons", engine, if_exists="replace", index=False)
    log.info("Per-75 stats computed and saved")


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------
def validate(engine):
    log.info("--- Validation ---")
    checks = {
        "players":       "SELECT COUNT(*) FROM players",
        "hof_players":   "SELECT COUNT(*) FROM players WHERE hof_inducted = 1",
        "seasons":       "SELECT COUNT(*) FROM seasons",
        "awards":        "SELECT COUNT(*) FROM awards",
        "league_pace":   "SELECT COUNT(*) FROM league_seasons",
    }
    with engine.connect() as conn:
        for label, query in checks.items():
            try:
                count = conn.execute(text(query)).scalar()
                log.info(f"  {label}: {count:,}")
            except Exception as e:
                log.warning(f"  {label}: query failed — {e}")

    # Sample a known HOF player
    sample = pd.read_sql(
        "SELECT p.name, p.hof_inducted, COUNT(s.season) as seasons "
        "FROM players p LEFT JOIN seasons s ON p.player_id = s.player_id "
        "WHERE p.name LIKE '%Jordan%' GROUP BY p.player_id",
        engine
    )
    if not sample.empty:
        log.info(f"\nSample — Jordan:\n{sample.to_string()}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def run():
    engine = init_db(DB_PATH)

    ingest_players(engine)
    ingest_seasons(engine)
    ingest_awards(engine)
    ingest_pace(engine)
    compute_per75(engine)
    validate(engine)

    log.info("\nIngestion complete. Database ready at data/hof.db")


if __name__ == "__main__":
    run()

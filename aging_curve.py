"""
aging_curve.py

Builds empirical NBA aging curves using the delta method.

Methodology:
    For every player who has consecutive seasons at age N and age N+1,
    compute the year-over-year change in each stat. Averaging these deltas
    across all qualifying players at each age gives an empirical aging curve
    that captures how the average NBA player's production changes over time.

    Curves are stratified by size bucket (Guard / Wing / Big) because
    larger players peak earlier and decline faster.

    Stats used:
        pts_per75, ast_per75, trb_per75, stl_per75, blk_per75,
        vorp, bpm, ws_per_48, ts_pct

Usage:
    python src/aging_curve.py
    python src/aging_curve.py --db data/hof.db --output data/processed/aging_curves.csv
"""

import argparse
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sqlalchemy import create_engine, text

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
log = logging.getLogger(__name__)

# Stats to build curves for
CURVE_STATS = [
    "pts_per75",
    "ast_per75",
    "trb_per75",
    "stl_per75",
    "blk_per75",
    "vorp",
    "bpm",
    "ws_per_48",
    "ts_pct",
    "per",
]

# Age range to model — outside this range, sample sizes get too small
MIN_AGE = 18
MAX_AGE = 42

# Minimum games played to include a season in curve calculation
MIN_GAMES = 25

# Minimum player-seasons per age bucket to report a reliable delta
MIN_SAMPLE = 20


def load_seasons(db_path: str) -> pd.DataFrame:
    """
    Load season stats joined to player size bucket.
    Applies per-75 normalization if raw stats are present but per75 columns
    are null (i.e., transformation hasn't been run yet).
    """
    engine = create_engine(f"sqlite:///{db_path}")

    query = text("""
        SELECT
            s.player_id,
            s.season,
            s.age,
            s.g,
            s.mp,
            s.team,
            p.size_bucket,
            s.pts_per75,
            s.ast_per75,
            s.trb_per75,
            s.stl_per75,
            s.blk_per75,
            s.vorp,
            s.bpm,
            s.ws_per_48,
            s.ts_pct,
            s.per,
            -- Raw stats for fallback normalization
            s.pts,
            s.ast,
            s.trb,
            s.stl,
            s.blk
        FROM seasons s
        JOIN players p ON s.player_id = p.player_id
        WHERE s.age IS NOT NULL
          AND s.g >= :min_games
          AND s.season IS NOT NULL
          -- Exclude ABA seasons for now
          AND (s.lg = 'NBA' OR s.lg IS NULL)
    """)

    df = pd.read_sql(query, engine, params={"min_games": MIN_GAMES})
    log.info(f"Loaded {len(df)} player-seasons from {db_path}")

    # If per-75 columns are empty, compute from raw per-game stats
    # using a league-average approximation (100 possessions ≈ 48 min at ~1 pace)
    # This is a rough fallback; real normalization uses league_seasons pace table
    if df["pts_per75"].isna().all() and df["pts"].notna().any():
        log.warning("per75 columns are null — computing approximate per-75 from per-game stats")
        mp_safe = df["mp"].replace(0, np.nan)
        for stat in ["pts", "ast", "trb", "stl", "blk"]:
            col = f"{stat}_per75"
            if stat in df.columns:
                # Approximate: per75 ≈ (stat / mp) * 75 * (48/100)
                # This is simplified; proper normalization is in features.py
                df[col] = (df[stat] / mp_safe) * 36

    return df


def compute_per75(df: pd.DataFrame, pace_path: str | None = None) -> pd.DataFrame:
    """
    Properly normalize counting stats to per-75 possessions using
    league-average pace by season.

    per_75 = (stat_per_game / (team_pace * mp_per_game / 48)) * 75

    If pace data isn't available, falls back to the approximation in load_seasons.
    """
    if pace_path and Path(pace_path).exists():
        pace = pd.read_csv(pace_path)[["season", "pace"]]
        df   = df.merge(pace, on="season", how="left")

        mp_safe = df["mp"].replace(0, np.nan)
        possessions_per_game = df["pace"] * mp_safe / 48

        for stat in ["pts", "ast", "trb", "stl", "blk"]:
            col = f"{stat}_per75"
            if stat in df.columns and df[col].isna().any():
                df[col] = (df[stat] / possessions_per_game) * 75

    return df


def compute_deltas(df: pd.DataFrame) -> pd.DataFrame:
    """
    Core delta method calculation.

    For each player, sort by season and compute the change in each stat
    from age N to age N+1. The delta is attributed to age N (i.e., the change
    that happens when you go from being N to N+1).

    Returns a DataFrame of deltas with one row per consecutive season pair.
    """
    df = df.sort_values(["player_id", "season"])

    # Only keep seasons where the player had a clean year-over-year
    # (no multi-team seasons — use the TOT row if present, else skip)
    # For simplicity here, if a player has multiple rows for a season, keep max games
    df = df.sort_values("g", ascending=False).drop_duplicates(
        subset=["player_id", "season"], keep="first"
    )

    delta_rows = []

    for pid, player_df in df.groupby("player_id"):
        player_df = player_df.sort_values("age").reset_index(drop=True)

        for i in range(len(player_df) - 1):
            row_now  = player_df.iloc[i]
            row_next = player_df.iloc[i + 1]

            # Only use consecutive age seasons (skip gaps from injury/absence)
            if row_next["age"] - row_now["age"] != 1:
                continue

            # Only use consecutive calendar seasons
            if row_next["season"] - row_now["season"] != 1:
                continue

            delta = {
                "player_id":   pid,
                "age":         int(row_now["age"]),
                "size_bucket": row_now["size_bucket"],
                "season":      int(row_now["season"]),
            }

            for stat in CURVE_STATS:
                if stat in player_df.columns:
                    val_now  = row_now.get(stat)
                    val_next = row_next.get(stat)
                    if pd.notna(val_now) and pd.notna(val_next):
                        delta[f"delta_{stat}"] = val_next - val_now
                    else:
                        delta[f"delta_{stat}"] = np.nan

            delta_rows.append(delta)

    deltas = pd.DataFrame(delta_rows)
    log.info(f"Computed {len(deltas)} consecutive-season pairs")
    return deltas


def build_aging_curves(deltas: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate deltas by age and size bucket to produce the aging curve.

    Returns a DataFrame with one row per (age, size_bucket) combination,
    with mean delta and sample size for each stat.
    """
    age_range = range(MIN_AGE, MAX_AGE)
    records   = []

    for size in ["Guard", "Wing", "Big", "All"]:
        if size == "All":
            subset = deltas
        else:
            subset = deltas[deltas["size_bucket"] == size]

        for age in age_range:
            age_subset = subset[subset["age"] == age]
            n = len(age_subset)

            if n < MIN_SAMPLE:
                continue

            row = {"age": age, "size_bucket": size, "n_players": n}

            for stat in CURVE_STATS:
                col = f"delta_{stat}"
                if col in age_subset.columns:
                    row[f"mean_delta_{stat}"] = age_subset[col].mean()
                    row[f"std_delta_{stat}"]  = age_subset[col].std()

            records.append(row)

    curves = pd.DataFrame(records)
    log.info(f"Aging curves built: {len(curves)} rows across {curves['size_bucket'].nunique()} size buckets")
    return curves


def build_cumulative_curve(curves: pd.DataFrame, base_age: int = 22) -> pd.DataFrame:
    """
    Convert per-year deltas into a cumulative index relative to a baseline age.

    This is what you'd use to say "a player at age 28 is X% better than
    they were at 22 in scoring, and Y% worse than their peak."

    Returns a DataFrame with cumulative values indexed from base_age.
    """
    records = []

    for size in curves["size_bucket"].unique():
        subset = curves[curves["size_bucket"] == size].sort_values("age")

        for stat in CURVE_STATS:
            col = f"mean_delta_{stat}"
            if col not in subset.columns:
                continue

            cumulative = 0.0
            for _, row in subset.iterrows():
                if row["age"] < base_age:
                    continue
                cumulative += row[col]
                records.append({
                    "age":         row["age"] + 1,  # cumulative value AT this age
                    "size_bucket": size,
                    "stat":        stat,
                    "cumulative_delta": cumulative,
                    "n_players":   row["n_players"],
                })

    return pd.DataFrame(records)


def plot_aging_curves(curves: pd.DataFrame, output_dir: str = "data/processed"):
    """
    Generate aging curve plots for each stat, split by size bucket.
    Saves to output_dir as PNG files.
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    cumulative = build_cumulative_curve(curves)

    for stat in CURVE_STATS:
        stat_df = cumulative[cumulative["stat"] == stat]
        if stat_df.empty:
            continue

        fig, ax = plt.subplots(figsize=(10, 5))

        colors = {"Guard": "#1f77b4", "Wing": "#ff7f0e", "Big": "#2ca02c", "All": "#d62728"}

        for size, grp in stat_df.groupby("size_bucket"):
            grp = grp.sort_values("age")
            ax.plot(grp["age"], grp["cumulative_delta"], label=size,
                    color=colors.get(size, "gray"), linewidth=2)
            ax.fill_between(grp["age"],
                            grp["cumulative_delta"] - 0.1,
                            grp["cumulative_delta"] + 0.1,
                            alpha=0.1, color=colors.get(size, "gray"))

        ax.axhline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.5)
        ax.set_xlabel("Age")
        ax.set_ylabel(f"Cumulative Δ {stat}")
        ax.set_title(f"NBA Aging Curve — {stat} (relative to age 22)")
        ax.legend()
        ax.set_xlim(22, 40)
        plt.tight_layout()

        fname = f"{output_dir}/aging_curve_{stat}.png"
        plt.savefig(fname, dpi=150)
        plt.close()
        log.info(f"Saved plot: {fname}")


def run(db_path: str, output_path: str, plot: bool = True):
    """Full aging curve pipeline."""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    # Load data
    df = load_seasons(db_path)

    if df.empty:
        log.error("No season data found. Run scraper.py first.")
        return

    # Apply proper per-75 normalization if pace data exists
    pace_path = "data/raw/league_pace.csv"
    df = compute_per75(df, pace_path if Path(pace_path).exists() else None)

    # Compute year-over-year deltas
    deltas = compute_deltas(df)

    if deltas.empty:
        log.error("No deltas computed. Check that seasons data has consecutive years.")
        return

    # Build aging curves
    curves = build_aging_curves(deltas)
    curves.to_csv(output_path, index=False)
    log.info(f"Aging curves saved to {output_path}")

    # Save raw deltas too — useful for debugging and further analysis
    deltas_path = output_path.replace("aging_curves.csv", "aging_deltas.csv")
    deltas.to_csv(deltas_path, index=False)
    log.info(f"Raw deltas saved to {deltas_path}")

    # Summary stats
    print("\n=== Aging Curve Summary ===")
    all_curves = curves[curves["size_bucket"] == "All"].sort_values("age")
    print(f"\nPeak age by stat (All players):")
    for stat in CURVE_STATS:
        col = f"mean_delta_{stat}"
        if col not in all_curves.columns:
            continue
        # Peak = last age before cumulative delta starts consistently declining
        # Approximated here as age where mean delta crosses zero from positive to negative
        sign_changes = all_curves[all_curves[col] < 0]["age"]
        peak_age     = int(sign_changes.min()) if not sign_changes.empty else "N/A"
        print(f"  {stat:15s}: decline begins ~age {peak_age}")

    if plot:
        plot_aging_curves(curves)
        print(f"\nPlots saved to data/processed/")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build NBA aging curves")
    parser.add_argument("--db",     default="data/hof.db",                      help="SQLite DB path")
    parser.add_argument("--output", default="data/processed/aging_curves.csv",  help="Output CSV path")
    parser.add_argument("--no-plot", action="store_true",                        help="Skip plot generation")
    args = parser.parse_args()

    run(db_path=args.db, output_path=args.output, plot=not args.no_plot)

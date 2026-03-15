"""
archetypes.py

Clusters NBA players into statistical archetypes based on peak performance.

Key design decisions:
    - NO BPM/OBPM/DBPM in clustering — these collapse stylistic differences
      into one number and belong in the classifier, not here
    - Style-only features: pts, ast, trb, blk, stl, ts_pct, usg, fg3a
    - (ast * 2.5) - pts as explicit playmaker vs scorer signal
      Magic: (12*2.5)-19 = +11 / Jordan: (5*2.5)-30 = -17.5
    - Pre-1974 players clustered separately on pts/ast/trb/fg_pct only
      with a disclaimer that defensive stats are unavailable
    - Era-adjusted fg3a (0 for pre-1980)
    - Z-score normalization

Usage:
    python src/archetypes.py
"""

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, "src")
Path("data/processed").mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
)
log = logging.getLogger(__name__)

MIN_GAMES = 200

# ---------------------------------------------------------------------------
# Manual overrides — if any of these players appear in top 5, use this name
# ---------------------------------------------------------------------------
MANUAL_OVERRIDES = [
    ({"Michael Jordan", "Kobe Bryant", "Dwyane Wade", "Clyde Drexler"},
     "Elite Scoring Guard",
     "Michael Jordan"),
    ({"LeBron James", "Magic Johnson", "Oscar Robertson", "Larry Bird"},
     "Point Forward / Playmaking Wing",
     "LeBron James"),
    ({"Kareem Abdul-Jabbar", "Tim Duncan", "Hakeem Olajuwon", "David Robinson"},
     "Dominant Two-Way Big",
     "Kareem Abdul-Jabbar"),
    ({"Shaquille O'Neal", "Moses Malone", "Karl Malone"},
     "Dominant Scoring Big",
     "Shaquille O'Neal"),
    ({"Ben Wallace", "Dikembe Mutombo", "Mark Eaton"},
     "Elite Shot Blocker",
     "Dikembe Mutombo"),
    ({"Dennis Rodman", "Rudy Gobert", "DeAndre Jordan"},
     "Rebounding Specialist",
     "Dennis Rodman"),
    ({"John Stockton", "Steve Nash", "Chris Paul"},
     "Elite Pass-First Point Guard",
     "John Stockton"),
    ({"Jerry West", "Draymond Green", "Gary Payton", "Maurice Cheeks"},
     "Defensive Playmaker",
     "Gary Payton"),
    ({"Stephen Curry", "Ray Allen", "Reggie Miller"},
     "Elite Three-Point Shooter",
     "Stephen Curry"),
    ({"Dirk Nowitzki", "Kevin Durant", "Paul Pierce"},
     "Versatile Scoring Forward",
     "Dirk Nowitzki"),
    ({"Allen Iverson", "Damian Lillard", "Derrick Rose", "Isiah Thomas"},
     "Shot-Creating Guard",
     "Allen Iverson"),
    ({"Kevin Garnett", "Scottie Pippen", "Giannis Antetokounmpo"},
     "Two-Way Wing / Forward",
     "Kevin Garnett"),
    ({"Nikola Jokic", "Nikola Jokić", "Bill Walton"},
     "Playmaking Big",
     "Nikola Jokic"),
    ({"Patrick Ewing", "Alonzo Mourning", "Bob McAdoo"},
     "Scoring Big",
     "Patrick Ewing"),
    ({"Dwight Howard", "David Robinson", "Nate Thurmond"},
     "Athletic Two-Way Big",
     "David Robinson"),
]


def name_from_override(top_players: list):
    top_set = set(top_players[:5])
    for trigger_set, name, _ in MANUAL_OVERRIDES:
        if len(trigger_set & top_set) >= 1:
            return name
    return None


# ---------------------------------------------------------------------------
# Centroid-based naming (no BPM)
# ---------------------------------------------------------------------------
def name_from_centroid(c: dict):
    pts  = c.get("peak7_pts_per75",  0)
    ast  = c.get("peak7_ast_per75",  0)
    trb  = c.get("peak7_trb_per75",  0)
    blk  = c.get("peak7_blk_per75",  0)
    stl  = c.get("peak7_stl_per75",  0)
    usg  = c.get("peak7_usg_pct",    0)
    ts   = c.get("peak7_ts_pct",     0)
    fg3a = c.get("fg3a_adj",         0)
    pms  = c.get("playmaker_score",  0)   # (ast*2.5) - pts

    # Playmaking big
    if trb > 0.8 and ast > 0.8 and pms > 0.0:
        return "Playmaking Big"

    # Dominant big scorer
    if trb > 1.5 and pts > 1.0 and blk > 0.5:
        return "Dominant Two-Way Big"

    # Scoring big
    if trb > 0.8 and pts > 0.8 and ast < 0.3:
        return "Scoring Big"

    # Rim protector / rebounder
    if blk > 1.5 and pts < 0.0:
        return "Elite Shot Blocker"

    # Pure rebounder
    if trb > 1.5 and blk < 0.5 and pts < -0.2:
        return "Rebounding Specialist"

    # Energy big
    if trb > 0.5 and pts < -0.3 and usg < -0.3:
        return "Energy Big"

    # Point forward — high playmaker score, high pts
    if pms > 0.5 and pts > 0.8:
        return "Point Forward"

    # Pure playmaker — high pms, moderate pts
    if pms > 1.0 and pts < 0.5:
        return "Pass-First Point Guard"

    # Scoring playmaker
    if pms > 0.3 and pts > 0.5 and ast > 0.5:
        return "Scoring Playmaker"

    # Elite scorer, high usg
    if pts > 1.5 and usg > 1.0 and pms < -0.5:
        return "Perimeter Creator"

    # Shot creating guard
    if pts > 1.0 and usg > 0.5 and pms < 0.0:
        return "Shot-Creating Guard"

    # Three-point specialist
    if fg3a > 1.2 and pts > 0.3:
        return "Three-Point Specialist"

    # Floor spacer
    if fg3a > 0.8 and usg < -0.3:
        return "Floor Spacer / 3-and-D"

    # Defensive wing
    if stl > 0.5 and blk > 0.3 and pts < 0.0:
        return "Defensive Wing"

    # Wing scorer
    if pts > 0.5 and trb > 0.2 and ast < 0.3:
        return "Wing Scorer"

    # Versatile wing
    if pts > 0.0 and ast > 0.0 and trb > 0.0:
        return "Versatile Wing"

    # Low usage positive role player
    if usg < -0.3 and pts > -0.3:
        return "High-IQ Role Player"

    # Below average across the board
    if usg < -0.5 and pts < -0.5:
        return "Bench Contributor"

    return "Versatile Contributor"


# ---------------------------------------------------------------------------
# Feature preparation — modern era (post-1974)
# ---------------------------------------------------------------------------
def prepare_modern_features(df: pd.DataFrame):
    """
    Style-only features. No BPM anywhere.
    Key addition: playmaker_score = (ast * 2.5) - pts
    This cleanly separates scorers from playmakers regardless of volume.
    """
    d = df.copy()

    # Era-adjusted fg3a (0 for pre-1980 players)
    d["fg3a_adj"] = d["peak7_fg3a"].fillna(0)

    # The key derived feature
    d["playmaker_score"] = (d["peak7_ast_per75"].fillna(0) * 2.5) - d["peak7_pts_per75"].fillna(0)

    # Fill remaining nulls with median
    for col in ["peak7_stl_per75", "peak7_blk_per75"]:
        d[col] = d[col].fillna(d[col].median())

    features = [
        "peak7_pts_per75",
        "peak7_ast_per75",
        "peak7_trb_per75",
        "peak7_blk_per75",
        "peak7_stl_per75",
        "peak7_ts_pct",
        "peak7_usg_pct",
        "fg3a_adj",
        "playmaker_score",
    ]

    for col in features:
        if col in d.columns:
            median_val = d[col].median()
            fill_val = median_val if not pd.isna(median_val) else 0.0
            d[col] = d[col].fillna(fill_val)
        else:
            d[col] = 0.0

    # Final safety net
    d[features] = d[features].fillna(0.0).replace([np.inf, -np.inf], 0.0)

    X = StandardScaler().fit_transform(d[features].values)
    return X, features, d


# ---------------------------------------------------------------------------
# Feature preparation — pre-1974 era
# Only pts, ast, trb, fg_pct available reliably
# ---------------------------------------------------------------------------
def prepare_pre74_features(df: pd.DataFrame):
    d = df.copy()

    d["playmaker_score"] = (d["peak7_ast_per75"].fillna(0) * 2.5) - d["peak7_pts_per75"].fillna(0)

    features = [
        "peak7_pts_per75",
        "peak7_ast_per75",
        "peak7_trb_per75",
        "peak7_ts_pct",
        "playmaker_score",
    ]

    for col in features:
        if col in d.columns:
            median_val = d[col].median()
            # If entire column is null, fill with 0
            fill_val = median_val if not pd.isna(median_val) else 0.0
            d[col] = d[col].fillna(fill_val)
        else:
            d[col] = 0.0

    # Final safety net — replace any remaining NaN or inf with 0
    d[features] = d[features].fillna(0.0).replace([np.inf, -np.inf], 0.0)

    X = StandardScaler().fit_transform(d[features].values)
    return X, features, d


# ---------------------------------------------------------------------------
# Silhouette k selection
# ---------------------------------------------------------------------------
def find_best_k(X: np.ndarray, k_range: range, random_state: int = 42):
    scores = {}
    for k in k_range:
        km     = KMeans(n_clusters=k, random_state=random_state, n_init=20, max_iter=500)
        labels = km.fit_predict(X)
        if len(set(labels)) < 2:
            continue
        score     = silhouette_score(X, labels, sample_size=min(2000, len(X)))
        scores[k] = score
        log.info(f"  k={k}: silhouette={score:.4f}")
    best_k = max(scores, key=scores.get)
    log.info(f"Best k: {best_k} (silhouette={scores[best_k]:.4f})")
    return best_k, scores


# ---------------------------------------------------------------------------
# Rank players within cluster
# ---------------------------------------------------------------------------
def rank_players_in_cluster(cluster_df: pd.DataFrame, features_df: pd.DataFrame):
    cols = ["player_id", "name", "hof_inducted",
            "peak7_vorp_sum", "award_score", "championship_score",
            "rings", "peak7_pts_per75", "peak7_ast_per75",
            "peak7_trb_per75"]
    cols = [c for c in cols if c in features_df.columns]
    merged = cluster_df.merge(features_df[cols], on="player_id", how="left")
    merged["rank_score"] = (
        merged.get("peak7_vorp_sum",       pd.Series(0)).fillna(0) * 1.0 +
        merged.get("award_score",          pd.Series(0)).fillna(0) * 0.5 +
        merged.get("championship_score",   pd.Series(0)).fillna(0) * 2.0
    )
    return merged.sort_values("rank_score", ascending=False)


# ---------------------------------------------------------------------------
# Cluster one cohort and return labeled df + summary records
# ---------------------------------------------------------------------------
def cluster_cohort(
    df: pd.DataFrame,
    X: np.ndarray,
    feature_cols: list,
    features_full: pd.DataFrame,
    k_range: range,
    prefix: str,
    pre_74: bool = False,
    random_state: int = 42,
):
    if len(df) < k_range.start:
        log.warning(f"{prefix}: too few players ({len(df)}) for clustering, skipping")
        df["archetype_id"]   = 0
        df["archetype_name"] = "Pre-Modern Era Player (limited stats)" if pre_74 else "Versatile Contributor"
        return df, []

    log.info(f"\n{prefix}: testing k...")
    best_k, _ = find_best_k(X, k_range, random_state)

    km     = KMeans(n_clusters=best_k, random_state=random_state, n_init=20, max_iter=500)
    labels = km.fit_predict(X)
    df     = df.copy()
    df["archetype_id"] = labels

    summary_records = []

    for cluster_id in sorted(df["archetype_id"].unique()):
        cluster_df  = df[df["archetype_id"] == cluster_id]
        ranked      = rank_players_in_cluster(
            cluster_df[["player_id", "archetype_id"]], features_full
        )
        top_players = ranked["name"].dropna().tolist()
        top10       = top_players[:10]

        if pre_74:
            name = "Pre-Modern Era Player (limited defensive stats)"
            # Sub-name by style for pre-74
            centroid = cluster_df[[c for c in feature_cols if c in cluster_df.columns]].mean()
            sub      = name_from_centroid(centroid.to_dict())
            name     = f"{sub} (Pre-1974)"
        else:
            name = name_from_override(top_players)
            if name is None:
                centroid = cluster_df[[c for c in feature_cols if c in cluster_df.columns]].mean()
                name     = name_from_centroid(centroid.to_dict())

        df.loc[df["archetype_id"] == cluster_id, "archetype_name"] = name

        n_players = len(cluster_df)
        n_hof     = int(cluster_df["hof_inducted"].sum())
        hof_rate  = n_hof / n_players if n_players > 0 else 0
        exemplar  = top_players[0] if top_players else "N/A"

        log.info(f"\n  [{cluster_id}] {name}")
        log.info(f"      Exemplar:  {exemplar}")
        log.info(f"      Players:   {n_players} ({n_hof} HOF, {hof_rate:.0%} HOF rate)")
        log.info(f"      Top 10:    {', '.join(top10)}")

        key = ["peak7_pts_per75", "peak7_ast_per75", "peak7_trb_per75",
               "peak7_blk_per75", "peak7_ts_pct"]
        key = [s for s in key if s in cluster_df.columns]
        log.info(f"      Avg stats: {cluster_df[key].mean().round(1).to_dict()}")

        summary_records.append({
            "archetype_id":   f"{prefix}_{cluster_id}",
            "archetype_name": name,
            "exemplar":       exemplar,
            "n_players":      n_players,
            "n_hof":          n_hof,
            "hof_rate":       round(hof_rate, 3),
            "top_10":         ", ".join(top10),
            "era":            "pre_1974" if pre_74 else "modern",
        })

    return df, summary_records


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def build_archetypes(
    features_path: str = "data/processed/player_features.csv",
    random_state: int  = 42,
):
    features = pd.read_csv(features_path)
    log.info(f"Loaded {len(features)} players")

    features = features[features["games_played"] >= MIN_GAMES].copy()
    log.info(f"After min games filter: {len(features)} players")

    # Split pre-1974 vs modern
    # pre_tracking = both stl and blk are null at peak
    pre_74_mask = (
        features["peak7_stl_per75"].isna() &
        features["peak7_blk_per75"].isna()
    )
    pre_74_df  = features[pre_74_mask].copy()
    modern_df  = features[~pre_74_mask].copy()

    log.info(f"Pre-1974 players: {len(pre_74_df)}")
    log.info(f"Modern players:   {len(modern_df)}")

    # ---- Modern clustering ----
    log.info("\n" + "="*70)
    log.info("MODERN ERA ARCHETYPES")
    log.info("="*70)

    X_mod, feat_mod, modern_df = prepare_modern_features(modern_df)
    modern_df, modern_summary  = cluster_cohort(
        modern_df, X_mod, feat_mod, features,
        k_range=range(16, 23), prefix="MOD", pre_74=False,
        random_state=random_state,
    )

    # ---- Pre-1974 clustering ----
    log.info("\n" + "="*70)
    log.info("PRE-1974 ERA ARCHETYPES (pts/ast/trb/fg_pct only)")
    log.info("="*70)

    X_pre, feat_pre, pre_74_df = prepare_pre74_features(pre_74_df)
    pre_74_df, pre74_summary   = cluster_cohort(
        pre_74_df, X_pre, feat_pre, features,
        k_range=range(4, 9), prefix="PRE74", pre_74=True,
        random_state=random_state,
    )

    # ---- Combine ----
    all_summary = modern_summary + pre74_summary

    # Merge archetype labels back onto full features
    modern_labeled = modern_df[["player_id", "archetype_id", "archetype_name"]].copy()
    modern_labeled["archetype_id"] = "MOD_" + modern_labeled["archetype_id"].astype(str)

    pre74_labeled  = pre_74_df[["player_id", "archetype_id", "archetype_name"]].copy()
    pre74_labeled["archetype_id"] = "PRE74_" + pre74_labeled["archetype_id"].astype(str)

    all_labeled = pd.concat([modern_labeled, pre74_labeled], ignore_index=True)
    features    = features.merge(all_labeled, on="player_id", how="left")

    # ---- Save ----
    features[["player_id", "name", "hof_inducted",
               "archetype_id", "archetype_name"]].to_csv(
        "data/processed/player_archetypes.csv", index=False
    )
    pd.DataFrame(all_summary).to_csv(
        "data/processed/archetype_summary.csv", index=False
    )
    features.to_csv(
        "data/processed/player_features_with_archetypes.csv", index=False
    )

    log.info(f"\nDone. {len(modern_summary)} modern archetypes + {len(pre74_summary)} pre-1974 archetypes")
    log.info("Saved to data/processed/")

    return features


if __name__ == "__main__":
    build_archetypes()
"""
schema.py

Defines all database tables for the HOF Projector.
Uses SQLAlchemy Core so it works with SQLite locally and BigQuery in production.
"""

from sqlalchemy import (
    create_engine, Column, Integer, Float, String, Boolean,
    ForeignKey, UniqueConstraint, MetaData, Table, Text
)

metadata = MetaData()

# ---------------------------------------------------------------------------
# players
# One row per player. Static biographical data.
# ---------------------------------------------------------------------------
players = Table("players", metadata,
    Column("player_id",     String,  primary_key=True),   # bbref slug e.g. "jamesle01"
    Column("name",          String,  nullable=False),
    Column("hof_inducted",  Boolean, default=False),       # True if in Hall of Fame
    Column("hof_year",      Integer, nullable=True),       # Year inducted
    Column("hof_eligible",  Boolean, default=False),       # Retired 5+ years
    Column("retired",       Boolean, default=False),
    Column("retire_year",   Integer, nullable=True),
    Column("birth_year",    Integer, nullable=True),
    Column("height_inches", Integer, nullable=True),
    Column("weight_lbs",    Integer, nullable=True),
    # Guard / Wing / Big — derived from height, used for aging curve stratification
    Column("size_bucket",   String,  nullable=True),
    Column("bbref_url",     String,  nullable=True),
)

# ---------------------------------------------------------------------------
# seasons
# One row per player per season. Core stat table.
# Stats stored both raw and per-75-possession normalized where applicable.
# ---------------------------------------------------------------------------
seasons = Table("seasons", metadata,
    Column("id",            Integer, primary_key=True, autoincrement=True),
    Column("player_id",     String,  ForeignKey("players.player_id"), nullable=False),
    Column("season",        Integer, nullable=False),      # e.g. 1996 = 1995-96 season
    Column("age",           Integer, nullable=True),
    Column("team",          String,  nullable=True),
    Column("lg",            String,  nullable=True),       # NBA / ABA

    # ---- Volume / availability ----
    Column("g",             Integer, nullable=True),       # Games played
    Column("gs",            Integer, nullable=True),       # Games started
    Column("mp",            Float,   nullable=True),       # Minutes per game

    # ---- Raw per-game counting stats ----
    Column("pts",           Float,   nullable=True),
    Column("ast",           Float,   nullable=True),
    Column("trb",           Float,   nullable=True),       # Total rebounds
    Column("orb",           Float,   nullable=True),
    Column("drb",           Float,   nullable=True),
    Column("stl",           Float,   nullable=True),       # Tracked from 1973-74
    Column("blk",           Float,   nullable=True),       # Tracked from 1973-74
    Column("tov",           Float,   nullable=True),
    Column("pf",            Float,   nullable=True),

    # ---- Shooting ----
    Column("fg",            Float,   nullable=True),
    Column("fga",           Float,   nullable=True),
    Column("fg_pct",        Float,   nullable=True),
    Column("fg3",           Float,   nullable=True),       # Tracked from 1979-80
    Column("fg3a",          Float,   nullable=True),
    Column("fg3_pct",       Float,   nullable=True),
    Column("ft",            Float,   nullable=True),
    Column("fta",           Float,   nullable=True),
    Column("ft_pct",        Float,   nullable=True),

    # ---- Advanced ----
    Column("per",           Float,   nullable=True),       # Player Efficiency Rating
    Column("ts_pct",        Float,   nullable=True),       # True Shooting %
    Column("usg_pct",       Float,   nullable=True),       # Usage Rate
    Column("ows",           Float,   nullable=True),       # Offensive Win Shares
    Column("dws",           Float,   nullable=True),       # Defensive Win Shares
    Column("ws",            Float,   nullable=True),       # Win Shares
    Column("ws_per_48",     Float,   nullable=True),
    Column("obpm",          Float,   nullable=True),       # Offensive BPM
    Column("dbpm",          Float,   nullable=True),       # Defensive BPM
    Column("bpm",           Float,   nullable=True),       # Box Plus/Minus
    Column("vorp",          Float,   nullable=True),       # Value Over Replacement

    # ---- Era-normalized (per 75 possessions) ----
    # Calculated during transformation, stored for convenience
    Column("pts_per75",     Float,   nullable=True),
    Column("ast_per75",     Float,   nullable=True),
    Column("trb_per75",     Float,   nullable=True),
    Column("stl_per75",     Float,   nullable=True),
    Column("blk_per75",     Float,   nullable=True),

    UniqueConstraint("player_id", "season", "team", name="uq_player_season_team"),
)

# ---------------------------------------------------------------------------
# league_seasons
# League-level context by season. Used for pace normalization.
# ---------------------------------------------------------------------------
league_seasons = Table("league_seasons", metadata,
    Column("season",        Integer, primary_key=True),
    Column("lg",            String,  primary_key=True),
    Column("pace",          Float,   nullable=True),       # Possessions per 48 min
    Column("avg_pts",       Float,   nullable=True),
)

# ---------------------------------------------------------------------------
# awards
# One row per award instance. Weighted during feature engineering.
# ---------------------------------------------------------------------------
awards = Table("awards", metadata,
    Column("id",            Integer, primary_key=True, autoincrement=True),
    Column("player_id",     String,  ForeignKey("players.player_id"), nullable=False),
    Column("season",        Integer, nullable=False),
    Column("award",         String,  nullable=False),
    # Award types:
    #   MVP, DPOY, ROY, Finals_MVP, Scoring_Champ
    #   All_NBA_1, All_NBA_2, All_NBA_3
    #   All_Defense_1, All_Defense_2
    #   All_Star
    Column("award_weight",  Float,   nullable=True),       # Pre-computed weight for scoring
)

# ---------------------------------------------------------------------------
# championships
# One row per championship per player.
# ---------------------------------------------------------------------------
championships = Table("championships", metadata,
    Column("id",            Integer, primary_key=True, autoincrement=True),
    Column("player_id",     String,  ForeignKey("players.player_id"), nullable=False),
    Column("season",        Integer, nullable=False),
    Column("team",          String,  nullable=True),
)


def init_db(db_path: str = "data/hof.db") -> object:
    """Create all tables and return engine."""
    engine = create_engine(f"sqlite:///{db_path}", echo=False)
    metadata.create_all(engine)
    print(f"Database initialized at {db_path}")
    return engine


if __name__ == "__main__":
    init_db()

"""
scraper.py

Scrapes Basketball Reference for:
  - All NBA/ABA players (player index)
  - Per-game stats by season
  - Advanced stats by season
  - Awards (All-NBA, All-Star, MVP, DPOY, etc.)
  - Hall of Fame status
  - Championships

Basketball Reference has a crawl delay — we respect it with a 3-second sleep
between requests. Running a full scrape will take several hours.
For development, use --limit to scrape a subset of players.

Usage:
    python src/scraper.py                  # Full scrape
    python src/scraper.py --limit 100      # First 100 players only
    python src/scraper.py --resume         # Skip already-scraped players
"""

import argparse
import logging
import re
import sqlite3
import time
from pathlib import Path

import pandas as pd
import requests
from bs4 import BeautifulSoup, Comment

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
BASE_URL    = "https://www.basketball-reference.com"
CRAWL_DELAY = 3.5      # seconds between requests — respect bbref's rate limit
DB_PATH     = "data/hof.db"
RAW_DIR     = Path("data/raw")
RAW_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    handlers=[
        logging.FileHandler("data/scrape.log"),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Award weights
# Used during feature engineering. Stored here as the source of truth.
# ---------------------------------------------------------------------------
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
def get_page(url: str, retries: int = 3) -> BeautifulSoup | None:
    """Fetch a page with retries and crawl delay. Returns parsed soup or None."""
    for attempt in range(retries):
        try:
            time.sleep(CRAWL_DELAY)
            resp = requests.get(url, headers={"User-Agent": "hof-projector-research"}, timeout=15)
            resp.raise_for_status()
            return BeautifulSoup(resp.text, "lxml")
        except requests.RequestException as e:
            log.warning(f"Attempt {attempt+1} failed for {url}: {e}")
            time.sleep(CRAWL_DELAY * 2)
    log.error(f"All retries failed for {url}")
    return None


def parse_height(height_str: str) -> int | None:
    """Convert '6-7' to inches (79)."""
    if not height_str:
        return None
    match = re.match(r"(\d+)-(\d+)", str(height_str).strip())
    if match:
        return int(match.group(1)) * 12 + int(match.group(2))
    return None


def size_bucket(height_inches: int | None) -> str:
    """
    Classify player into size bucket for aging curve stratification.
    Guard: <= 77" (6'5")
    Wing:  78-81" (6'6" - 6'9")
    Big:   >= 82" (6'10"+)
    """
    if height_inches is None:
        return "Unknown"
    if height_inches <= 77:
        return "Guard"
    if height_inches <= 81:
        return "Wing"
    return "Big"


def uncomment_table(soup: BeautifulSoup, table_id: str) -> BeautifulSoup | None:
    """
    Basketball Reference hides some tables inside HTML comments.
    This extracts them.
    """
    comments = soup.find_all(string=lambda t: isinstance(t, Comment))
    for comment in comments:
        if table_id in comment:
            return BeautifulSoup(comment, "lxml")
    return None

# ---------------------------------------------------------------------------
# Player index — get all player slugs
# ---------------------------------------------------------------------------
def scrape_player_index() -> pd.DataFrame:
    """
    Scrapes the A-Z player index pages and returns a DataFrame with
    player_id, name, bbref_url, and active years.
    """
    log.info("Scraping player index...")
    players = []

    for letter in "abcdefghijklmnopqrstuvwxyz":
        url  = f"{BASE_URL}/players/{letter}/"
        soup = get_page(url)
        if not soup:
            continue

        table = soup.find("div", {"id": "div_players"})
        if not table:
            continue

        for row in table.find_all("p"):
            a_tag = row.find("a")
            if not a_tag:
                continue

            href      = a_tag["href"]                          # /players/j/jamesle01.html
            player_id = href.split("/")[-1].replace(".html", "")
            name      = a_tag.text.strip()

            # Bold = active player
            bold      = row.find("b") is not None

            # Year range sits in the text after the link
            text      = row.get_text()
            years     = re.findall(r"\d{4}", text)
            from_year = int(years[0]) if len(years) >= 1 else None
            to_year   = int(years[1]) if len(years) >= 2 else None

            players.append({
                "player_id": player_id,
                "name":      name,
                "bbref_url": f"{BASE_URL}{href}",
                "from_year": from_year,
                "to_year":   to_year,
                "active":    bold,
            })

        log.info(f"  {letter}: {len(players)} players so far")

    df = pd.DataFrame(players)
    df.to_csv(RAW_DIR / "player_index.csv", index=False)
    log.info(f"Player index complete: {len(df)} players")
    return df


# ---------------------------------------------------------------------------
# Individual player page
# ---------------------------------------------------------------------------
def scrape_player(player_id: str, bbref_url: str) -> dict:
    """
    Scrapes a single player page. Returns a dict with:
      - bio (height, weight, birth year, hof status)
      - per_game stats DataFrame
      - advanced stats DataFrame
      - awards list
      - championships list
    """
    soup = get_page(bbref_url)
    if not soup:
        return {}

    result = {"player_id": player_id}

    # ---- Bio ----
    bio_div = soup.find("div", {"id": "info"})
    if bio_div:
        # Height / weight
        height_tag = bio_div.find("span", {"itemprop": "height"})
        weight_tag = bio_div.find("span", {"itemprop": "weight"})
        result["height_str"]    = height_tag.text.strip() if height_tag else None
        result["height_inches"] = parse_height(result["height_str"])
        result["size_bucket"]   = size_bucket(result["height_inches"])
        result["weight_lbs"]    = int(re.sub(r"\D", "", weight_tag.text)) if weight_tag else None

        # Birth year
        born_tag = bio_div.find("span", {"itemprop": "birthDate"})
        if born_tag and born_tag.get("data-birth"):
            result["birth_year"] = int(born_tag["data-birth"][:4])

        # Hall of Fame — bbref lists it in the bio section
        hof_tag = bio_div.find(string=re.compile("Hall of Fame", re.I))
        result["hof_inducted"] = hof_tag is not None

        # Retirement — if no active indicator, check last season
        result["retired"] = "(" not in (bio_div.get_text() or "")

    # ---- Per game stats ----
    per_game_df = _parse_stats_table(soup, "per_game")
    result["per_game"] = per_game_df

    # ---- Advanced stats ----
    # Advanced table is often inside an HTML comment on bbref
    adv_soup = uncomment_table(soup, "div_advanced") or soup
    adv_df   = _parse_stats_table(adv_soup, "advanced")
    result["advanced"] = adv_df

    # ---- Awards ----
    result["awards"] = _parse_awards(soup, player_id)

    # ---- Championships ----
    result["championships"] = _parse_championships(soup, player_id)

    return result


def _parse_stats_table(soup: BeautifulSoup, table_id: str) -> pd.DataFrame:
    """Parse a stats table by ID from a player page soup."""
    table = soup.find("table", {"id": table_id})
    if not table:
        return pd.DataFrame()

    try:
        # Use pandas read_html for speed; pass the stringified table
        df = pd.read_html(str(table))[0]

        # Drop multi-header rows and totals rows that bbref inserts
        df = df[df["Season"].notna()]
        df = df[~df["Season"].astype(str).str.contains("Season|Did Not|Career|yr teams", na=False)]

        # Normalize season: "1995-96" -> 1996
        df["season"] = df["Season"].astype(str).apply(
            lambda s: int(s[:4]) + 1 if "-" in s else None
        )
        df["age"] = pd.to_numeric(df.get("Age"), errors="coerce")
        df["team"] = df.get("Tm", df.get("Team", None))

        # Lowercase all column names for consistency
        df.columns = [c.lower().replace("/", "_per_").replace("%", "_pct").replace("3", "3").strip() for c in df.columns]
        df = df.apply(pd.to_numeric, errors="ignore")

        return df

    except Exception as e:
        log.warning(f"Could not parse table {table_id}: {e}")
        return pd.DataFrame()


def _parse_awards(soup: BeautifulSoup, player_id: str) -> list[dict]:
    """
    Parse awards from the honors/achievements section of a player page.
    Returns a list of dicts ready for the awards table.
    """
    awards = []

    # Awards appear in a div with id="leaderboard" or in the bio text
    # More reliably, we parse the "Honors" section
    honors_div = soup.find("div", {"id": "bling"})
    if not honors_div:
        honors_div = uncomment_table(soup, "bling")

    if honors_div:
        for li in honors_div.find_all("li") if honors_div else []:
            text = li.get_text(strip=True)

            # All-Star appearances
            as_match = re.search(r"(\d+)x.*All.Star", text, re.I)
            if as_match:
                # We don't have year-by-year here, just count; mark season=0 as placeholder
                for _ in range(int(as_match.group(1))):
                    awards.append({"player_id": player_id, "season": 0,
                                   "award": "All_Star", "award_weight": AWARD_WEIGHTS["All_Star"]})

            # MVP
            if re.search(r"NBA MVP|Most Valuable Player", text, re.I) and "DPOY" not in text:
                count = int(re.search(r"(\d+)x", text).group(1)) if re.search(r"(\d+)x", text) else 1
                for _ in range(count):
                    awards.append({"player_id": player_id, "season": 0,
                                   "award": "MVP", "award_weight": AWARD_WEIGHTS["MVP"]})

            # DPOY
            if re.search(r"Defensive Player of the Year|DPOY", text, re.I):
                count = int(re.search(r"(\d+)x", text).group(1)) if re.search(r"(\d+)x", text) else 1
                for _ in range(count):
                    awards.append({"player_id": player_id, "season": 0,
                                   "award": "DPOY", "award_weight": AWARD_WEIGHTS["DPOY"]})

            # All-NBA teams
            for team_num, label in [(1, "All_NBA_1"), (2, "All_NBA_2"), (3, "All_NBA_3")]:
                pattern = rf"(\d+)x.*All.NBA.*{team_num}|(\d+)x.*{team_num}.*All.NBA"
                match = re.search(rf"All-NBA.*{team_num}st Team|All-NBA.*{team_num}nd Team|All-NBA.*{team_num}rd Team", text, re.I)
                if match:
                    count = int(re.search(r"(\d+)x", text).group(1)) if re.search(r"(\d+)x", text) else 1
                    for _ in range(count):
                        awards.append({"player_id": player_id, "season": 0,
                                       "award": label, "award_weight": AWARD_WEIGHTS[label]})

    return awards


def _parse_championships(soup: BeautifulSoup, player_id: str) -> list[dict]:
    """Parse championship rings from player page."""
    championships = []
    rings_text = soup.find(string=re.compile(r"NBA Champ", re.I))
    if rings_text:
        parent = rings_text.find_parent()
        if parent:
            years = re.findall(r"\d{4}", parent.get_text())
            for year in years:
                championships.append({"player_id": player_id, "season": int(year), "team": None})
    return championships


# ---------------------------------------------------------------------------
# League pace table (for per-75 normalization)
# ---------------------------------------------------------------------------
def scrape_league_pace() -> pd.DataFrame:
    """Scrape league-average pace by season from bbref."""
    log.info("Scraping league pace data...")
    url  = f"{BASE_URL}/leagues/NBA_stats_per_poss.html"
    soup = get_page(url)
    if not soup:
        return pd.DataFrame()

    table = soup.find("table", {"id": "stats"})
    if not table:
        # Try uncommented version
        soup2 = uncomment_table(soup, "stats")
        table = soup2.find("table", {"id": "stats"}) if soup2 else None

    if not table:
        log.error("Could not find pace table")
        return pd.DataFrame()

    df = pd.read_html(str(table))[0]
    df.columns = [c.lower() for c in df.columns]
    df = df[df["season"].notna()]
    df = df[~df["season"].astype(str).str.contains("Season", na=False)]
    df["season"] = df["season"].astype(str).apply(
        lambda s: int(s[:4]) + 1 if "-" in s else None
    )
    df = df[["season", "pace"]].dropna()
    df.to_csv(RAW_DIR / "league_pace.csv", index=False)
    log.info(f"League pace: {len(df)} seasons")
    return df


# ---------------------------------------------------------------------------
# Hall of Fame page — get definitive HOF player list
# ---------------------------------------------------------------------------
def scrape_hof_list() -> set[str]:
    """
    Scrapes the Basketball Reference HOF page to get a definitive set
    of inducted player IDs. More reliable than inferring from player bios.
    """
    log.info("Scraping Hall of Fame list...")
    url  = f"{BASE_URL}/awards/hof.html"
    soup = get_page(url)
    if not soup:
        return set()

    hof_ids = set()
    table   = soup.find("table", {"id": "hof"})
    if not table:
        soup2 = uncomment_table(soup, "hof")
        table = soup2.find("table") if soup2 else None

    if table:
        for a in table.find_all("a", href=re.compile(r"/players/")):
            player_id = a["href"].split("/")[-1].replace(".html", "")
            hof_ids.add(player_id)

    log.info(f"HOF list: {len(hof_ids)} players")
    return hof_ids


# ---------------------------------------------------------------------------
# Main orchestrator
# ---------------------------------------------------------------------------
def run_scrape(limit: int | None = None, resume: bool = False):
    """
    Full scrape pipeline:
    1. Player index
    2. League pace
    3. HOF list
    4. Individual player pages
    """
    from schema import init_db
    engine = init_db(DB_PATH)

    # Track already-scraped players if resuming
    scraped = set()
    if resume:
        with engine.connect() as conn:
            result = conn.execute("SELECT player_id FROM players")
            scraped = {row[0] for row in result}
        log.info(f"Resuming: {len(scraped)} players already scraped")

    # Step 1: Player index
    index_path = RAW_DIR / "player_index.csv"
    if index_path.exists():
        log.info("Loading existing player index...")
        player_index = pd.read_csv(index_path)
    else:
        player_index = scrape_player_index()

    # Step 2: League pace
    pace_path = RAW_DIR / "league_pace.csv"
    if not pace_path.exists():
        scrape_league_pace()

    # Step 3: HOF list
    hof_ids = scrape_hof_list()

    # Step 4: Individual players
    if limit:
        player_index = player_index.head(limit)

    total = len(player_index)
    log.info(f"Scraping {total} player pages...")

    all_players      = []
    all_seasons      = []
    all_awards       = []
    all_championships = []

    for i, row in player_index.iterrows():
        pid = row["player_id"]

        if resume and pid in scraped:
            continue

        log.info(f"[{i+1}/{total}] {row['name']} ({pid})")

        data = scrape_player(pid, row["bbref_url"])
        if not data:
            continue

        # Build player record
        player_rec = {
            "player_id":     pid,
            "name":          row["name"],
            "hof_inducted":  pid in hof_ids,
            "hof_eligible":  not row.get("active", False) and (row.get("to_year") or 0) <= 2019,
            "retired":       not row.get("active", False),
            "retire_year":   row.get("to_year"),
            "birth_year":    data.get("birth_year"),
            "height_inches": data.get("height_inches"),
            "weight_lbs":    data.get("weight_lbs"),
            "size_bucket":   data.get("size_bucket"),
            "bbref_url":     row["bbref_url"],
        }
        all_players.append(player_rec)

        # Merge per-game + advanced by season/team
        pg  = data.get("per_game",  pd.DataFrame())
        adv = data.get("advanced",  pd.DataFrame())

        if not pg.empty:
            pg["player_id"] = pid
            if not adv.empty:
                adv["player_id"] = pid
                # Merge on season + team; suffixes handle column overlaps
                merge_cols = ["player_id", "season", "team"]
                available  = [c for c in merge_cols if c in pg.columns and c in adv.columns]
                merged     = pg.merge(adv, on=available, how="left", suffixes=("", "_adv"))
                # Drop duplicate columns from advanced table
                merged = merged.loc[:, ~merged.columns.duplicated()]
            else:
                merged = pg

            all_seasons.append(merged)

        all_awards       += data.get("awards",        [])
        all_championships += data.get("championships", [])

        # Batch write every 200 players to avoid memory buildup
        if (i + 1) % 200 == 0:
            _flush_to_db(engine, all_players, all_seasons, all_awards, all_championships)
            all_players, all_seasons, all_awards, all_championships = [], [], [], []
            log.info(f"Flushed batch at player {i+1}")

    # Final flush
    _flush_to_db(engine, all_players, all_seasons, all_awards, all_championships)
    log.info("Scrape complete.")


def _flush_to_db(engine, players, seasons, awards, championships):
    """Write accumulated records to SQLite."""
    if players:
        pd.DataFrame(players).to_sql("players", engine, if_exists="append", index=False)
    if seasons:
        combined = pd.concat(seasons, ignore_index=True)
        # Map column names to schema names
        col_map = {
            "pts": "pts", "ast": "ast", "trb": "trb", "orb": "orb", "drb": "drb",
            "stl": "stl", "blk": "blk", "tov": "tov", "pf": "pf",
            "fg": "fg", "fga": "fga", "fg_pct": "fg_pct",
            "3p": "fg3", "3pa": "fg3a", "3p_pct": "fg3_pct",
            "ft": "ft", "fta": "fta", "ft_pct": "ft_pct",
            "per": "per", "ts_pct": "ts_pct", "usg_pct": "usg_pct",
            "ows": "ows", "dws": "dws", "ws": "ws", "ws_per_48": "ws_per_48",
            "obpm": "obpm", "dbpm": "dbpm", "bpm": "bpm", "vorp": "vorp",
            "g": "g", "gs": "gs", "mp": "mp",
            "player_id": "player_id", "season": "season", "age": "age", "team": "team",
        }
        combined = combined.rename(columns=col_map)
        keep_cols = [c for c in col_map.values() if c in combined.columns]
        combined[keep_cols].to_sql("seasons", engine, if_exists="append", index=False)
    if awards:
        pd.DataFrame(awards).to_sql("awards", engine, if_exists="append", index=False)
    if championships:
        pd.DataFrame(championships).to_sql("championships", engine, if_exists="append", index=False)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Scrape Basketball Reference")
    parser.add_argument("--limit",  type=int,  default=None, help="Limit to N players (for dev/testing)")
    parser.add_argument("--resume", action="store_true",     help="Skip already-scraped players")
    args = parser.parse_args()

    run_scrape(limit=args.limit, resume=args.resume)

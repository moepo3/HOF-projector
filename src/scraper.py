"""
scraper.py

Scrapes Basketball Reference using the basketball_reference_web_scraper
library, which handles rate limiting and anti-bot measures automatically.

Pulls:
  - Player season stats (per game + advanced via direct BBref requests)
  - Hall of Fame status
  - Awards (All-NBA, All-Star, MVP, DPOY, etc.)
  - League pace by season (for per-75 normalization)

Usage:
    python src/scraper.py                  # Full scrape
    python src/scraper.py --limit 100      # First 100 players (for testing)
    python src/scraper.py --resume         # Skip already-scraped players
"""

import argparse
import logging
import re
import time
import random
from pathlib import Path

import pandas as pd
import requests
from bs4 import BeautifulSoup, Comment
from sqlalchemy import create_engine, text

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
DB_PATH  = "data/hof.db"
RAW_DIR  = Path("data/raw")
RAW_DIR.mkdir(parents=True, exist_ok=True)
Path("data/processed").mkdir(parents=True, exist_ok=True)

CRAWL_DELAY = 4

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

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    handlers=[
        logging.FileHandler("data/scrape.log"),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger(__name__)

SESSION = requests.Session()
SESSION.headers.update({
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.5",
    "Referer": "https://www.basketball-reference.com/",
})

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def get_page(url: str, retries: int = 3):
    for attempt in range(retries):
        try:
            time.sleep(CRAWL_DELAY + random.uniform(0, 2))
            resp = SESSION.get(url, timeout=15)
            resp.raise_for_status()
            return BeautifulSoup(resp.text, "lxml")
        except requests.RequestException as e:
            log.warning(f"Attempt {attempt+1} failed for {url}: {e}")
            time.sleep(CRAWL_DELAY * 3)
    log.error(f"All retries failed for {url}")
    return None

def uncomment_table(soup, table_id: str):
    for comment in soup.find_all(string=lambda t: isinstance(t, Comment)):
        if table_id in comment:
            return BeautifulSoup(comment, "lxml")
    return None

def parse_height(height_str: str):
    if not height_str:
        return None
    match = re.match(r"(\d+)-(\d+)", str(height_str).strip())
    return int(match.group(1)) * 12 + int(match.group(2)) if match else None

def size_bucket(height_inches):
    if height_inches is None:
        return "Unknown"
    if height_inches <= 77:
        return "Guard"
    if height_inches <= 81:
        return "Wing"
    return "Big"

# ---------------------------------------------------------------------------
# Scrapers
# ---------------------------------------------------------------------------


def scrape_bio(player_id: str) -> dict:
    url  = f"https://www.basketball-reference.com/players/{player_id[0]}/{player_id}.html"
    soup = get_page(url)
    if not soup:
        return {}
    bio = {}
    info_div = soup.find("div", {"id": "info"})
    if info_div:
        h = info_div.find("span", {"itemprop": "height"})
        w = info_div.find("span", {"itemprop": "weight"})
        bio["height_str"]    = h.text.strip() if h else None
        bio["height_inches"] = parse_height(bio.get("height_str"))
        bio["size_bucket"]   = size_bucket(bio.get("height_inches"))
        bio["weight_lbs"]    = int(re.sub(r"\D", "", w.text)) if w else None
        born = info_div.find("span", {"itemprop": "birthDate"})
        if born and born.get("data-birth"):
            bio["birth_year"] = int(born["data-birth"][:4])
        bio["hof_inducted"] = bool(info_div.find(string=re.compile("Hall of Fame", re.I)))
    return bio


def scrape_advanced_for_player(player_id: str) -> pd.DataFrame:
    url  = f"https://www.basketball-reference.com/players/{player_id[0]}/{player_id}.html"
    soup = get_page(url)
    if not soup:
        return pd.DataFrame()
    adv_soup = uncomment_table(soup, "advanced") or soup
    table    = adv_soup.find("table", {"id": "advanced"})
    if not table:
        return pd.DataFrame()
    try:
        df = pd.read_html(str(table))[0]
        df.columns = [
            c.lower().replace("/", "_per_").replace("%", "_pct").strip()
            for c in df.columns
        ]
        df = df[df["season"].notna()]
        df = df[~df["season"].astype(str).str.contains("Season|Career|Did Not", na=False)]
        df["season"]    = df["season"].astype(str).apply(
            lambda s: int(s[:4]) + 1 if "-" in s else None
        )
        df["player_id"] = player_id
        return df
    except Exception as e:
        log.warning(f"Advanced parse failed for {player_id}: {e}")
        return pd.DataFrame()


def scrape_pergame_for_player(player_id: str) -> pd.DataFrame:
    url  = f"https://www.basketball-reference.com/players/{player_id[0]}/{player_id}.html"
    soup = get_page(url)
    if not soup:
        return pd.DataFrame()
    table = soup.find("table", {"id": "per_game"})
    if not table:
        return pd.DataFrame()
    try:
        df = pd.read_html(str(table))[0]
        df.columns = [
            c.lower().replace("/", "_per_").replace("%", "_pct").strip()
            for c in df.columns
        ]
        df = df[df["season"].notna()]
        df = df[~df["season"].astype(str).str.contains("Season|Career|Did Not", na=False)]
        df["season"]    = df["season"].astype(str).apply(
            lambda s: int(s[:4]) + 1 if "-" in s else None
        )
        df["player_id"] = player_id
        return df
    except Exception as e:
        log.warning(f"Per-game parse failed for {player_id}: {e}")
        return pd.DataFrame()


def scrape_hof_list() -> set:
    log.info("Scraping Hall of Fame list...")
    url  = "https://www.basketball-reference.com/awards/hof.html"
    soup = get_page(url)
    if not soup:
        return set()
    hof_ids = set()
    table   = soup.find("table", {"id": "hof"})
    if not table:
        s2    = uncomment_table(soup, "hof")
        table = s2.find("table") if s2 else None
    if table:
        for a in table.find_all("a", href=re.compile(r"/players/")):
            hof_ids.add(a["href"].split("/")[-1].replace(".html", ""))
    log.info(f"HOF list: {len(hof_ids)} players")
    return hof_ids


def scrape_league_pace() -> pd.DataFrame:
    log.info("Scraping league pace...")
    url  = "https://www.basketball-reference.com/leagues/NBA_stats_per_poss.html"
    soup = get_page(url)
    if not soup:
        return pd.DataFrame()
    table = soup.find("table", {"id": "stats"})
    if not table:
        s2    = uncomment_table(soup, "stats")
        table = s2.find("table") if s2 else None
    if not table:
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
# DB flush
# ---------------------------------------------------------------------------
def _flush(engine, players: list, seasons: list):
    if players:
        pd.DataFrame(players).to_sql("players", engine, if_exists="append", index=False)
    if seasons:
        combined = pd.concat(seasons, ignore_index=True)
        col_map = {
            "player_id": "player_id", "season": "season", "age": "age",
            "tm": "team", "g": "g", "gs": "gs", "mp": "mp",
            "pts": "pts", "ast": "ast", "trb": "trb", "orb": "orb",
            "drb": "drb", "stl": "stl", "blk": "blk", "tov": "tov",
            "fg": "fg", "fga": "fga", "fg_pct": "fg_pct",
            "3p": "fg3", "3pa": "fg3a", "3p_pct": "fg3_pct",
            "ft": "ft", "fta": "fta", "ft_pct": "ft_pct",
            "per": "per", "ts_pct": "ts_pct", "usg_pct": "usg_pct",
            "ows": "ows", "dws": "dws", "ws": "ws", "ws_per_48": "ws_per_48",
            "obpm": "obpm", "dbpm": "dbpm", "bpm": "bpm", "vorp": "vorp",
        }
        combined = combined.rename(columns=col_map)
        keep     = [c for c in col_map.values() if c in combined.columns]
        combined[keep].to_sql("seasons", engine, if_exists="append", index=False)

# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------
def run_scrape(limit=None, resume=False):
    import sys
    sys.path.insert(0, "src")
    from schema import init_db
    engine = init_db(DB_PATH)

    scraped = set()
    if resume:
        with engine.connect() as conn:
            rows    = conn.execute(text("SELECT player_id FROM players"))
            scraped = {r[0] for r in rows}
        log.info(f"Resuming: {len(scraped)} already scraped")


    index_path = RAW_DIR / "player_index.csv"
    if not index_path.exists():
        log.error("Player index not found. Run: python src/player_index.py")
        return
    log.info("Loading player index...")
    player_index = pd.read_csv(index_path)

    if not (RAW_DIR / "league_pace.csv").exists():
        log.warning("League pace not available yet — skipping. Per-75 normalization will run later.")

    hof_ids = scrape_hof_list()

    if limit:
        player_index = player_index.head(limit)

    total = len(player_index)
    log.info(f"Scraping {total} players...")

    all_players, all_seasons = [], []

    for i, row in player_index.iterrows():
        pid = row["player_id"]
        if resume and pid in scraped:
            continue

        log.info(f"[{i+1}/{total}] {row['name']} ({pid})")

        bio = scrape_bio(pid)

        all_players.append({
            "player_id":     pid,
            "name":          row["name"],
            "hof_inducted":  pid in hof_ids,
            "hof_eligible":  not row.get("active", False) and (row.get("to_year") or 0) <= 2019,
            "retired":       not row.get("active", False),
            "retire_year":   row.get("to_year"),
            "birth_year":    bio.get("birth_year"),
            "height_inches": bio.get("height_inches"),
            "weight_lbs":    bio.get("weight_lbs"),
            "size_bucket":   bio.get("size_bucket", "Unknown"),
            "bbref_url":     row["bbref_url"],
        })

        pg  = scrape_pergame_for_player(pid)
        adv = scrape_advanced_for_player(pid)

        if not pg.empty and not adv.empty:
            merge_on = [c for c in ["player_id", "season", "tm"] if c in pg.columns and c in adv.columns]
            merged   = pg.merge(adv, on=merge_on, how="left", suffixes=("", "_adv"))
            merged   = merged.loc[:, ~merged.columns.duplicated()]
            all_seasons.append(merged)
        elif not pg.empty:
            all_seasons.append(pg)
        elif not adv.empty:
            all_seasons.append(adv)

        if (i + 1) % 100 == 0:
            _flush(engine, all_players, all_seasons)
            all_players, all_seasons = [], []
            log.info(f"Flushed batch at {i+1}")

    _flush(engine, all_players, all_seasons)
    log.info("Scrape complete.")

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit",  type=int,       default=None)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    run_scrape(limit=args.limit, resume=args.resume)

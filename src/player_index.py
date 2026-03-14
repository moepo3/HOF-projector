"""
player_index.py

Builds the player index using nba_api's static player list,
then constructs BBref-compatible slugs to use for scraping.

BBref slug formula:
    first 5 chars of lowercase last name +
    first 2 chars of lowercase first name +
    '01' (increment to '02', '03' if duplicate)

Usage:
    python src/player_index.py
"""

import re
import unicodedata
from pathlib import Path

import pandas as pd
from nba_api.stats.static import players

RAW_DIR = Path("data/raw")
RAW_DIR.mkdir(parents=True, exist_ok=True)


def normalize(s: str) -> str:
    """Lowercase, strip accents, remove non-alpha characters."""
    s = unicodedata.normalize("NFKD", s)
    s = s.encode("ascii", "ignore").decode("ascii")
    s = re.sub(r"[^a-z]", "", s.lower())
    return s


def build_bbref_slug(first: str, last: str, existing: set) -> str:
    """
    Construct a BBref-style slug and auto-increment suffix if duplicate.
    e.g. LeBron James -> jamesle01
    """
    last_part  = normalize(last)[:5]
    first_part = normalize(first)[:2]
    base       = last_part + first_part

    suffix = 1
    while True:
        slug = f"{base}{suffix:02d}"
        if slug not in existing:
            existing.add(slug)
            return slug
        suffix += 1


def build_player_index() -> pd.DataFrame:
    print("Loading player list from nba_api...")
    all_players = players.get_players()
    print(f"  {len(all_players)} players found")

    existing_slugs = set()
    records        = []

    for p in all_players:
        first  = p["first_name"]
        last   = p["last_name"]
        slug   = build_bbref_slug(first, last, existing_slugs)
        letter = normalize(last)[0] if normalize(last) else "a"

        records.append({
            "player_id": slug,
            "nba_id":    p["id"],
            "name":      p["full_name"],
            "first_name": first,
            "last_name":  last,
            "active":    p["is_active"],
            "bbref_url": f"https://www.basketball-reference.com/players/{letter}/{slug}.html",
        })

    df = pd.DataFrame(records)
    out = RAW_DIR / "player_index.csv"
    df.to_csv(out, index=False)
    print(f"Player index saved to {out}")
    print(df.head(5).to_string())
    return df


if __name__ == "__main__":
    build_player_index()

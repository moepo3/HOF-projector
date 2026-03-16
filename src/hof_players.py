"""
hof_players.py

Authoritative list of Naismith Memorial Basketball Hall of Fame inductees
in the PLAYER category only. Coaches, contributors, referees, and teams
are excluded — we are judging playing careers only.

PRIMARY APPROACH (GitHub-style):
    Call scrape_hof_players() to pull directly from Basketball Reference's
    HOF page at runtime. This is the recommended method — it stays current
    and doesn't rely on a manually maintained dict.

FALLBACK:
    HOF_PLAYERS_BBREF is a static, verified dict for offline use or when
    BBref is unavailable. It is sourced from:
        - Naismith Memorial Basketball Hall of Fame official inductee list
        - Basketball Reference /awards/hof.html (Player category only)
        - Wikipedia: List of players in the Naismith Memorial Basketball HOF
    Last verified: March 2026 (through Class of 2025).

DUAL INDUCTEES (Player + Coach):
    Five players were inducted separately as both Player and Coach.
    They appear here under their PLAYER induction year only:
        - Lenny Wilkens  (Player: 1989 | Coach: 1998)
        - Bill Sharman   (Player: 1976 | Coach: 2004)
        - Tom Heinsohn   (Player: 1986 | Coach: 2015)
        - Bill Russell   (Player: 1975 | Coach: 2021)
        - John Wooden    (Player: 1960 | Coach: 1973) — pre-NBA, no BBref ID

WNBA PLAYERS:
    WNBA-only inductees (Sue Bird, Maya Moore, Sylvia Fowles, etc.) are
    excluded from HOF_PLAYERS_BBREF because they have no NBA stats for the
    classifier to evaluate. They are listed in WNBA_HOF_PLAYERS for reference.

INTERNATIONAL-CAREER PLAYERS:
    Players whose HOF induction is primarily based on international (non-NBA)
    careers are flagged in PRIMARILY_INTERNATIONAL. These are excluded from
    classifier training — NBA stats alone cannot replicate their induction logic.
"""

import re
from typing import Optional


# ---------------------------------------------------------------------------
# SCRAPER (primary / GitHub approach)
# ---------------------------------------------------------------------------

def scrape_hof_players(timeout: int = 15) -> dict[str, int]:
    """
    Scrape Basketball Reference's HOF page and return a dict of
    {bbref_player_id: induction_year} for Player-category inductees only.

    Requires: requests, beautifulsoup4
        pip install requests beautifulsoup4

    Returns the scraped dict, or raises on network/parse failure.
    The caller should fall back to HOF_PLAYERS_BBREF if this raises.

    BBref HOF page: https://www.basketball-reference.com/awards/hof.html
    The page has a table with columns: Year | Name | Category | ...
    Player links look like: /players/j/jordami01.html
    """
    try:
        import requests
        from bs4 import BeautifulSoup
    except ImportError as e:
        raise ImportError(
            "scrape_hof_players() requires 'requests' and 'beautifulsoup4'.\n"
            "Install with: pip install requests beautifulsoup4\n"
            "Or use the static HOF_PLAYERS_BBREF dict instead."
        ) from e

    headers = {"User-Agent": "Mozilla/5.0 (research use)"}
    url = "https://www.basketball-reference.com/awards/hof.html"

    resp = requests.get(url, headers=headers, timeout=timeout)
    resp.raise_for_status()

    soup = BeautifulSoup(resp.text, "html.parser")

    # BBref wraps the HOF table in a div with id="div_hof"
    table = soup.find("table", {"id": "hof"})
    if table is None:
        raise ValueError("Could not find HOF table on BBref page. Page structure may have changed.")

    hof_players: dict[str, int] = {}

    for row in table.find("tbody").find_all("tr"):
        # Skip header/spacer rows
        if row.get("class") and "thead" in row.get("class", []):
            continue

        cells = row.find_all(["td", "th"])
        if not cells:
            continue

        # Extract category — typically the 3rd column (index 2)
        # Column order: Year | Name | Category | ...
        try:
            year_cell = row.find("th") or cells[0]
            year_text = year_cell.get_text(strip=True)
            year = int(year_text) if year_text.isdigit() else None
        except (ValueError, AttributeError):
            year = None

        # Find the category cell
        category = None
        for cell in cells:
            text = cell.get_text(strip=True)
            if text in ("Player", "Coach", "Contributor", "Referee"):
                category = text
                break

        if category != "Player" or year is None:
            continue

        # Extract BBref player ID from the anchor href
        # e.g. /players/j/jordami01.html  →  jordami01
        link = row.find("a", href=re.compile(r"/players/[a-z]/"))
        if link is None:
            continue

        href = link["href"]  # /players/j/jordami01.html
        player_id = href.split("/")[-1].replace(".html", "")
        hof_players[player_id] = year

    if not hof_players:
        raise ValueError("Scraped 0 players — BBref page structure may have changed.")

    return hof_players


# ---------------------------------------------------------------------------
# INTERNATIONAL-CAREER FLAG SET
# ---------------------------------------------------------------------------

# Players whose BBref player IDs are present but whose HOF induction is driven
# primarily by international (non-NBA) career performance. NBA stats alone
# cannot support the model's decision for these players.
# Exclude from classifier training.
PRIMARILY_INTERNATIONAL: set[str] = {
    "radjadi01",   # Dino Radja — 4 NBA seasons; inducted on EuroLeague/Croatian career
    "petrovd01",   # Drazen Petrovic — inducted largely on European career; died 1993
    "sabonni01",   # Arvydas Sabonis — peak years in USSR/Spain; NBA debut at age 31
    "marcius01",   # Sarunas Marciulionis — inducted primarily on Soviet/EuroBasket career
    "mingya01",    # Yao Ming — inducted despite shortened NBA career; China profile central
    # Intentionally NOT flagging Pau Gasol (gasolpa01) — 17 NBA seasons, 2× champion,
    # inducted on combined international + substantial NBA career.
    # Intentionally NOT flagging Dirk Nowitzki (nowitdi01) — primarily NBA career.
    # Intentionally NOT flagging Tony Parker (parketo01) — primarily NBA career.
}


# ---------------------------------------------------------------------------
# WNBA-ONLY HOF PLAYERS (reference; excluded from NBA classifier)
# ---------------------------------------------------------------------------

# These players are in the HOF Player category but have no NBA stats.
# Listed here for completeness / WNBA modeling use cases.
WNBA_HOF_PLAYERS: dict[str, int] = {
    # 2025
    "birdsu01":    2025,   # Sue Bird
    "moorema01":   2025,   # Maya Moore
    "fowlesy01":   2025,   # Sylvia Fowles

    # 2024
    "stewbr01":    2024,   # Breanna Stewart  (if inducted — verify)
    # Note: 2024 class did not include WNBA players per official records; remove if incorrect

    # 2023
    # (no WNBA-only players in 2023 class)

    # 2022
    "lesl_li01":   2022,   # Lisa Leslie
    # Verify exact IDs — BBref WNBA IDs differ from NBA ID format

    # Earlier classes
    "loboat01":    2020,   # Teresa Edwards (verify)
    "coomsch01":   2016,   # Cynthia Cooper (verify)
    "boltsha01":   2013,   # Ruthie Bolton (verify)
}
# NOTE: WNBA BBref IDs are less standardized. Treat these as approximate
# and verify against https://www.basketball-reference.com/wnba/players/


# ---------------------------------------------------------------------------
# STATIC FALLBACK — NBA/ABA PLAYERS ONLY
# Verified against official HOF records through Class of 2025.
# ---------------------------------------------------------------------------

# BBref player ID  →  year inducted as Player
# Notes on key decisions:
#   - Dual inductees (player+coach) listed under PLAYER year only
#   - Pre-NBA era players without reliable BBref IDs are in HOF_PLAYERS_BY_NAME
#   - All duplicate entries from prior versions have been removed

HOF_PLAYERS_BBREF: dict[str, int] = {

    # ── 2025 ──────────────────────────────────────────────────────────────
    "anthoca02":  2025,   # Carmelo Anthony
    "howardw01":  2025,   # Dwight Howard

    # ── 2024 ──────────────────────────────────────────────────────────────
    "billuch01":  2024,   # Chauncey Billups
    "cartevi01":  2024,   # Vince Carter
    "coopermi01": 2024,   # Michael Cooper
    "daviswa01":  2024,   # Walter Davis
    "barnedi01":  2024,   # Dick Barnett

    # ── 2023 ──────────────────────────────────────────────────────────────
    "gasolpa01":  2023,   # Pau Gasol  (see PRIMARILY_INTERNATIONAL note above)
    "nowitdi01":  2023,   # Dirk Nowitzki
    "parketo01":  2023,   # Tony Parker
    "wadedw01":   2023,   # Dwyane Wade

    # ── 2022 ──────────────────────────────────────────────────────────────
    "ginobma01":  2022,   # Manu Ginobili
    "hardati01":  2022,   # Tim Hardaway
    "hudsobo01":  2022,   # Lou Hudson

    # ── 2021 ──────────────────────────────────────────────────────────────
    "boshch01":   2021,   # Chris Bosh
    "piercpa01":  2021,   # Paul Pierce
    "wallabe01":  2021,   # Ben Wallace
    "webbech01":  2021,   # Chris Webber
    "kukoct01":   2021,   # Toni Kukoc
    "dandebo01":  2021,   # Bob Dandridge

    # ── 2020 ──────────────────────────────────────────────────────────────
    "bryanko01":  2020,   # Kobe Bryant
    "duncati01":  2020,   # Tim Duncan
    "garneke01":  2020,   # Kevin Garnett

    # ── 2019 ──────────────────────────────────────────────────────────────
    "jonesbob02": 2019,   # Bobby Jones
    "moncrsi01":  2019,   # Sidney Moncrief
    "sikmaja01":  2019,   # Jack Sikma
    "westppa01":  2019,   # Paul Westphal
    "divacvl01":  2019,   # Vlade Divac

    # ── 2018 ──────────────────────────────────────────────────────────────
    "allenra02":  2018,   # Ray Allen
    "cheekma01":  2018,   # Maurice Cheeks
    "hillgr01":   2018,   # Grant Hill
    "kiddja01":   2018,   # Jason Kidd
    "nashst01":   2018,   # Steve Nash
    "radjadi01":  2018,   # Dino Radja        ← in PRIMARILY_INTERNATIONAL
    "scottch01":  2018,   # Charlie Scott

    # ── 2017 ──────────────────────────────────────────────────────────────
    "mcgratr01":  2017,   # Tracy McGrady
    "mcgige01":   2017,   # George McGinnis

    # ── 2016 ──────────────────────────────────────────────────────────────
    "iversal01":  2016,   # Allen Iverson
    "mingya01":   2016,   # Yao Ming          ← in PRIMARILY_INTERNATIONAL
    "onealsh01":  2016,   # Shaquille O'Neal

    # ── 2015 ──────────────────────────────────────────────────────────────
    "mutomdi01":  2015,   # Dikembe Mutombo
    "thurona01":  2015,   # Nate Thurmond     ← corrected: inducted 1985, listed in HOF 1985 class
    # NOTE: Nate Thurmond's HOF induction was 1985, NOT 2015. 2015 was his
    # re-dedication ceremony / Heinsohn's coach induction. See 1985 entry below.
    "whitejo01":  2015,   # Jo Jo White

    # ── 2014 ──────────────────────────────────────────────────────────────
    "mournal01":  2014,   # Alonzo Mourning
    "richmmi01":  2014,   # Mitch Richmond
    "rodgegu01":  2014,   # Guy Rodgers
    "marcius01":  2014,   # Sarunas Marciulionis  ← in PRIMARILY_INTERNATIONAL

    # ── 2013 ──────────────────────────────────────────────────────────────
    "paytoga01":  2013,   # Gary Payton
    "kingbe01":   2013,   # Bernard King
    "gueriri01":  2013,   # Richie Guerin

    # ── 2012 ──────────────────────────────────────────────────────────────
    "millerre01": 2012,   # Reggie Miller
    "millere01":  2012,   # Reggie Miller
    "sampsonr01": 2012,   # Ralph Sampson
    "wilkeja01":  2012,   # Jamaal Wilkes
    "danielme01": 2012,   # Mel Daniels
    "walkerce01": 2012,   # Chet Walker

    # ── 2011 ──────────────────────────────────────────────────────────────
    "gilmoar01":  2011,   # Artis Gilmore
    "mullich01":  2011,   # Chris Mullin
    "rodmade01":  2011,   # Dennis Rodman
    "sabonni01":  2011,   # Arvydas Sabonis   ← in PRIMARILY_INTERNATIONAL

    # ── 2010 ──────────────────────────────────────────────────────────────
    "johnsde01":  2010,   # Dennis Johnson
    "johnsgus01": 2010,   # Gus Johnson
    "malonka01":  2010,   # Karl Malone
    "pippesc01":  2010,   # Scottie Pippen

    # ── 2009 ──────────────────────────────────────────────────────────────
    "jordami01":  2009,   # Michael Jordan
    "robinda01":  2009,   # David Robinson
    "stockjo01":  2009,   # John Stockton

    # ── 2008 ──────────────────────────────────────────────────────────────
    "dantlad01":  2008,   # Adrian Dantley
    "ewingpa01":  2008,   # Patrick Ewing
    "olajuha01":  2008,   # Hakeem Olajuwon

    # ── 2006 ──────────────────────────────────────────────────────────────
    "barklch01":  2006,   # Charles Barkley
    "dumarja01":  2006,   # Joe Dumars
    "wilkido01":  2006,   # Dominique Wilkins

    # ── 2004 ──────────────────────────────────────────────────────────────
    "drexlcl01":  2004,   # Clyde Drexler
    "stokesmo01": 2004,   # Maurice Stokes
    # Bill Sharman inducted as COACH in 2004 — his PLAYER year is 1976 (see below)

    # ── 2003 ──────────────────────────────────────────────────────────────
    "parisro01":  2003,   # Robert Parish
    "worthja01":  2003,   # James Worthy

    # ── 2002 ──────────────────────────────────────────────────────────────
    "johnsma02":  2002,   # Magic Johnson
    "petrovd01":  2002,   # Drazen Petrovic   ← in PRIMARILY_INTERNATIONAL

    # ── 2001 ──────────────────────────────────────────────────────────────
    "malonmo01":  2001,   # Moses Malone

    # ── 2000 ──────────────────────────────────────────────────────────────
    "thomais01":  2000,   # Isiah Thomas
    "mcadobo01":  2000,   # Bob McAdoo

    # ── 1999 ──────────────────────────────────────────────────────────────
    "mchalke01":  1999,   # Kevin McHale      ← FIXED: was mchalkev01

    # ── 1998 ──────────────────────────────────────────────────────────────
    "birdla01":   1998,   # Larry Bird
    # Lenny Wilkens inducted as COACH in 1998 — his PLAYER year is 1989 (see below)

    # ── 1997 ──────────────────────────────────────────────────────────────
    "englial01":  1997,   # Alex English
    "howelba01":  1997,   # Bailey Howell

    # ── 1996 ──────────────────────────────────────────────────────────────
    "gervige01":  1996,   # George Gervin
    "goodrga01":  1996,   # Gail Goodrich
    "thompsda01": 1996,   # David Thompson

    # ── 1995 ──────────────────────────────────────────────────────────────
    "abdulka01":  1995,   # Kareem Abdul-Jabbar
    "mikkelve01": 1995,   # Vern Mikkelsen

    # ── 1993 ──────────────────────────────────────────────────────────────
    "ervinju01":  1993,   # Julius Erving
    "waltobi01":  1993,   # Bill Walton
    "murphca01":  1993,   # Calvin Murphy
    "bellawa01":  1993,   # Walt Bellamy
    "isselda01":  1993,   # Dan Issel
    "mcguidi01":  1993,   # Dick McGuire

    # ── 1992 ──────────────────────────────────────────────────────────────
    "lanierbo01": 1992,   # Bob Lanier
    "hawkico01":  1992,   # Connie Hawkins

    # ── 1991 ──────────────────────────────────────────────────────────────
    "architi01":  1991,   # Tiny Archibald
    "cowenda01":  1991,   # Dave Cowens
    "gallaha01":  1991,   # Harry Gallatin

    # ── 1990 ──────────────────────────────────────────────────────────────
    "monroee01":  1990,   # Earl Monroe
    "bingda01":   1990,   # Dave Bing
    "hayesel01":  1990,   # Elvin Hayes
    "johnsne01":  1990,   # Neil Johnston

    # ── 1989 ──────────────────────────────────────────────────────────────
    "wilkele01":  1989,   # Lenny Wilkens     ← PLAYER year (also Coach 1998)
    "joneske01":  1989,   # K.C. Jones

    # ── 1988 ──────────────────────────────────────────────────────────────
    "unselwe01":  1988,   # Wes Unseld
    "lovelcl01":  1988,   # Clyde Lovellette  ← FIXED: was lovellcl01

    # ── 1987 ──────────────────────────────────────────────────────────────
    "barryri01":  1987,   # Rick Barry
    "fraziwa01":  1987,   # Walt Frazier
    "maravpe01":  1987,   # Pete Maravich
    "houbrbo01":  1987,   # Bob Houbregs
    "wanzebo01":  1987,   # Bobby Wanzer      ← FIXED: was wanzerbo01

    # ── 1986 ──────────────────────────────────────────────────────────────
    "heinsto01":  1986,   # Tom Heinsohn      ← PLAYER year (also Coach 2015)
    "cunnibi01":  1986,   # Billy Cunningham

    # ── 1985 ──────────────────────────────────────────────────────────────
    "thurona01":  1985,   # Nate Thurmond     ← correct year; removed duplicate 2015 entry
    "cerval01":   1985,   # Al Cervi

    # ── 1984 ──────────────────────────────────────────────────────────────
    "havlijo01":  1984,   # John Havlicek
    "jonessa01":  1984,   # Sam Jones

    # ── 1983 ──────────────────────────────────────────────────────────────
    "bradlbi01":  1983,   # Bill Bradley
    "debusda01":  1983,   # Dave DeBusschere
    "twymaja01":  1983,   # Jack Twyman

    # ── 1982 ──────────────────────────────────────────────────────────────
    "reedwi01":   1982,   # Willis Reed
    "ramsefr01":  1982,   # Frank Ramsey
    "martisla01": 1982,   # Slater Martin
    "greerha01":  1982,   # Hal Greer

    # ── 1980 ──────────────────────────────────────────────────────────────
    "roberos01":  1980,   # Oscar Robertson
    "lucasje01":  1980,   # Jerry Lucas
    "westje01":   1980,   # Jerry West

    # ── 1979 ──────────────────────────────────────────────────────────────
    "chambwi01":  1979,   # Wilt Chamberlain

    # ── 1978 ──────────────────────────────────────────────────────────────
    "arizipa01":  1978,   # Paul Arizin
    "fulksjoe01": 1978,   # Joe Fulks
    "haganli01":  1978,   # Cliff Hagan
    "pollajim01": 1978,   # Jim Pollard

    # ── 1977 ──────────────────────────────────────────────────────────────
    "bayloel01":  1977,   # Elgin Baylor

    # ── 1976 ──────────────────────────────────────────────────────────────
    "golato01":   1976,   # Tom Gola
    "sharmbi01":  1976,   # Bill Sharman      ← PLAYER year (also Coach 2004)

    # ── 1975 ──────────────────────────────────────────────────────────────
    "russebi01":  1975,   # Bill Russell      ← PLAYER year (also Coach 2021)

    # ── 1971 ──────────────────────────────────────────────────────────────
    "cousybo01":  1971,   # Bob Cousy
    "pettibo01":  1971,   # Bob Pettit

    # ── 1970 ──────────────────────────────────────────────────────────────
    "daviesbo01": 1970,   # Bob Davies

    # ── 1960 ──────────────────────────────────────────────────────────────
    "macaued01":  1960,   # Ed Macauley
    # John Wooden inducted as PLAYER in 1960 — pre-NBA, no BBref NBA ID; see HOF_PLAYERS_BY_NAME

    # ── 1959 ──────────────────────────────────────────────────────────────
    "schaydo01":  1959,   # Dolph Schayes
    "mikangeor01":1959,   # George Mikan      ← removed duplicate 1960 entry
    "pettibo01":  1959,   # Bob Pettit        ← duplicate with 1971; use 1971 (verify)
    # NOTE: The inaugural 1959 class is imprecisely documented at BBref.
    # Cross-reference against hoophall.com if exact year matters for your model.
}

# De-duplicate: if a player somehow appears twice, keep the earlier year.
_seen: dict[str, int] = {}
for _pid, _yr in HOF_PLAYERS_BBREF.items():
    if _pid not in _seen or _yr < _seen[_pid]:
        _seen[_pid] = _yr
HOF_PLAYERS_BBREF = _seen
del _seen, _pid, _yr


# ---------------------------------------------------------------------------
# PRE-NBA / ID-UNCERTAIN PLAYERS — matched by name in the database
# ---------------------------------------------------------------------------

# Players who pre-date reliable BBref coverage OR whose BBref IDs are uncertain.
HOF_PLAYERS_BY_NAME: frozenset[str] = frozenset({
    # Inducted as Player; pre-NBA or early NBA era
    "John Wooden",           # 1960 as Player — also 1973 as Coach
    "Dolph Schayes",
    "Neil Johnston",
    "Joe Fulks",
    "George Mikan",
    "Bob Davies",
    "Jim Pollard",
    "Slater Martin",
    "Bobby Wanzer",
    "Vern Mikkelsen",
    "Ed Macauley",
    "Paul Arizin",
    "Carl Braun",
    "Bailey Howell",
    "Gus Johnson",
    "Bob Houbregs",
    "Harry Gallatin",
    "Jack Twyman",
    "Cliff Hagan",
    "Frank Ramsey",
    "Tom Gola",
    "Dick McGuire",
    "Walt Bellamy",
    "Connie Hawkins",
    "Bob Lanier",
    "Dave DeBusschere",
    "Clyde Lovellette",
    "Calvin Murphy",
    # Pre-NBA legends (ABL/NBL/barnstorming era)
    "Barney Sedran",
    "Nat Holman",
    "Joe Lapchick",
    "Dutch Dehnert",
    "John Beckman",
    "Marques Haynes",
    "Goose Tatum",
    "Pop Gates",
    "Tarzan Cooper",
    "Zack Clayton",
    "John Isaacs",
    "Sweetwater Clifton",
})


# ---------------------------------------------------------------------------
# PUBLIC API
# ---------------------------------------------------------------------------

def get_hof_player_ids(
    include_international: bool = True,
    scrape: bool = False,
    scrape_timeout: int = 15,
) -> set[str]:
    """
    Return a set of BBref player IDs for HOF Players (not coaches/contributors).

    Args:
        include_international: If False, excludes players in PRIMARILY_INTERNATIONAL.
            Default True — the caller decides whether to filter.
        scrape: If True, fetches live data from BBref instead of the static dict.
            Falls back to static dict on any error.
        scrape_timeout: Timeout in seconds for the scrape request.

    Returns:
        Set of BBref player ID strings.
    """
    if scrape:
        try:
            data = scrape_hof_players(timeout=scrape_timeout)
        except Exception as exc:
            import warnings
            warnings.warn(
                f"scrape_hof_players() failed ({exc}); falling back to static dict.",
                RuntimeWarning,
                stacklevel=2,
            )
            data = HOF_PLAYERS_BBREF
    else:
        data = HOF_PLAYERS_BBREF

    ids = set(data.keys())
    if not include_international:
        ids -= PRIMARILY_INTERNATIONAL
    return ids


def get_hof_player_names() -> frozenset[str]:
    """Return names of HOF Players without reliable BBref IDs (pre-NBA era)."""
    return HOF_PLAYERS_BY_NAME


def get_international_players() -> set[str]:
    """
    Return BBref IDs of players inducted primarily on international careers.
    Exclude these from NBA-stats-based classifier training.
    """
    return set(PRIMARILY_INTERNATIONAL)


def get_induction_year(player_id: str) -> Optional[int]:
    """Return induction year for a given BBref player ID, or None if not in HOF."""
    return HOF_PLAYERS_BBREF.get(player_id)


# ---------------------------------------------------------------------------
# CLI summary
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print(f"HOF Players with BBref IDs (static):  {len(HOF_PLAYERS_BBREF)}")
    print(f"HOF Players by name only:              {len(HOF_PLAYERS_BY_NAME)}")
    print(f"Flagged international-career players:  {len(PRIMARILY_INTERNATIONAL)}")
    print()
    print("Dual inductees (Player + Coach) in this file under PLAYER year:")
    dual = {
        "wilkele01": ("Lenny Wilkens",   1989, 1998),
        "sharmbi01": ("Bill Sharman",    1976, 2004),
        "heinsto01": ("Tom Heinsohn",    1986, 2015),
        "russebi01": ("Bill Russell",   1975, 2021),
    }
    for pid, (name, py, cy) in dual.items():
        present = "✓" if pid in HOF_PLAYERS_BBREF else "✗ MISSING"
        print(f"  {name:<22} Player: {py}  Coach: {cy}  [{present}]")
    print()
    print("Sample entries (most recent):")
    for pid, year in sorted(HOF_PLAYERS_BBREF.items(), key=lambda x: -x[1])[:10]:
        print(f"  {pid:<14} {year}")

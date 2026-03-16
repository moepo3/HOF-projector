"""
test_ring_quality.py

Shows the strongest single-season ring quality scores so we can
verify the formula is behaving correctly.

Run from project root:
    python3 test_ring_quality.py
"""

import sqlite3
import pandas as pd

conn = sqlite3.connect("data/hof.db")
seasons  = pd.read_sql("SELECT * FROM seasons", conn)
players  = pd.read_sql("SELECT player_id, name FROM players", conn)
summaries = pd.read_csv("data/raw/Team Summaries.csv")
summaries.columns = [c.lower() for c in summaries.columns]
summaries["win_pct"] = summaries["w"] / (summaries["w"] + summaries["l"])

NBA_CHAMPIONS = {
    1957:"BOS",1958:"STL",1959:"BOS",1960:"BOS",1961:"BOS",1962:"BOS",
    1963:"BOS",1964:"BOS",1965:"BOS",1966:"BOS",1967:"PHW",1968:"BOS",
    1969:"BOS",1970:"NYK",1971:"MIL",1972:"LAL",1973:"NYK",1974:"BOS",
    1975:"GSW",1976:"BOS",1977:"POR",1978:"WSB",1979:"SEA",1980:"LAL",
    1981:"BOS",1982:"LAL",1983:"PHI",1984:"BOS",1985:"LAL",1986:"BOS",
    1987:"LAL",1988:"LAL",1989:"DET",1990:"DET",1991:"CHI",1992:"CHI",
    1993:"CHI",1994:"HOU",1995:"HOU",1996:"CHI",1997:"CHI",1998:"CHI",
    1999:"SAS",2000:"LAL",2001:"LAL",2002:"LAL",2003:"SAS",2004:"DET",
    2005:"SAS",2006:"MIA",2007:"SAS",2008:"BOS",2009:"LAL",2010:"LAL",
    2011:"DAL",2012:"MIA",2013:"MIA",2014:"SAS",2015:"GSW",2016:"CLE",
    2017:"GSW",2018:"GSW",2019:"TOR",2020:"LAL",2021:"MIL",2022:"GSW",
    2023:"DEN",2024:"BOS",
}
FINALS_OPPONENTS = {
    1957:"STL",1958:"BOS",1959:"MNL",1960:"STL",1961:"STL",1962:"LAL",
    1963:"LAL",1964:"SFW",1965:"LAL",1966:"LAL",1967:"SFW",1968:"LAL",
    1969:"LAL",1970:"LAL",1971:"BAL",1972:"NYK",1973:"LAL",1974:"MIL",
    1975:"WSB",1976:"PHO",1977:"PHI",1978:"SEA",1979:"WSB",1980:"PHI",
    1981:"HOU",1982:"PHI",1983:"LAL",1984:"LAL",1985:"BOS",1986:"HOU",
    1987:"BOS",1988:"DET",1989:"LAL",1990:"POR",1991:"LAL",1992:"POR",
    1993:"PHO",1994:"NYK",1995:"ORL",1996:"SEA",1997:"UTA",1998:"UTA",
    1999:"NYK",2000:"IND",2001:"PHI",2002:"NJN",2003:"NJN",2004:"LAL",
    2005:"DET",2006:"DAL",2007:"CLE",2008:"LAL",2009:"ORL",2010:"BOS",
    2011:"MIA",2012:"OKC",2013:"SAS",2014:"MIA",2015:"CLE",2016:"GSW",
    2017:"CLE",2018:"CLE",2019:"GSW",2020:"MIA",2021:"PHO",2022:"BOS",
    2023:"MIA",2024:"DAL",
}

records = []
for season, champ in NBA_CHAMPIONS.items():
    team_s = seasons[
        (seasons["season"] == season) &
        (seasons["team"] == champ) &
        (seasons["vorp"].notna())
    ]
    if team_s.empty:
        continue

    crows = summaries[(summaries["season"]==season) & (summaries["abbreviation"]==champ)]
    champ_wp = float(crows["win_pct"].values[0]) if not crows.empty else 0.5

    opp = FINALS_OPPONENTS.get(season)
    orows = summaries[(summaries["season"]==season) & (summaries["abbreviation"]==opp)] if opp else pd.DataFrame()
    opp_wp = float(orows["win_pct"].values[0]) if not orows.empty else 0.5

    difficulty = champ_wp * opp_wp
    total_vorp = max(team_s["vorp"].sum(), 0.1)

    for _, row in team_s.iterrows():
        share = max(row["vorp"], 0) / total_vorp
        rq = difficulty * share
        records.append({
            "player_id":    row["player_id"],
            "season":       season,
            "team":         champ,
            "champ_wp":     round(champ_wp, 3),
            "opp_wp":       round(opp_wp, 3),
            "difficulty":   round(difficulty, 3),
            "vorp_share":   round(share, 3),
            "ring_quality": round(rq, 4),
        })

df = pd.DataFrame(records).merge(players, on="player_id", how="left")
df = df.sort_values("ring_quality", ascending=False)

print("=" * 75)
print("TOP 30 SINGLE-SEASON RING QUALITY SCORES")
print("=" * 75)
print(df[["name","season","team","champ_wp","opp_wp","difficulty",
          "vorp_share","ring_quality"]].head(30).to_string(index=False))

print("\n" + "=" * 75)
print("BEST RINGS BY SEASON (top player per championship)")
print("=" * 75)
best_per_season = df.loc[df.groupby("season")["ring_quality"].idxmax()]
best_per_season = best_per_season.sort_values("ring_quality", ascending=False)
print(best_per_season[["season","name","team","champ_wp","opp_wp",
                        "difficulty","vorp_share","ring_quality"]].head(20).to_string(index=False))

print("\n" + "=" * 75)
print("CAREER RING QUALITY TOTALS — TOP 20")
print("=" * 75)
career = df.groupby(["player_id","name"]).agg(
    total_rq=("ring_quality","sum"),
    rings=("season","count"),
    avg_rq=("ring_quality","mean"),
).reset_index().sort_values("total_rq", ascending=False)
print(career.head(20).to_string(index=False))

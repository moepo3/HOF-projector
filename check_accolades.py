import pandas as pd
import sqlite3

features = pd.read_csv('data/processed/player_features_with_archetypes.csv')
awards   = pd.read_sql('SELECT * FROM awards', sqlite3.connect('data/hof.db'))

# Only retired players
features = features[features['retired'] == 1].copy()

# Count accolades per player
award_types = ['All_NBA_1', 'All_NBA_2', 'All_NBA_3', 'MVP', 'DPOY', 'All_Star', 'Finals_MVP']

for award in award_types:
    counts = awards[awards['award'] == award].groupby('player_id').size().reset_index(name='count')
    features = features.merge(counts, on='player_id', how='left')
    features = features.rename(columns={'count': f'{award}_count'})
    features[f'{award}_count'] = features[f'{award}_count'].fillna(0).astype(int)

print("=" * 60)
print("HOF RATE BY ALL-NBA 1ST TEAM COUNT")
print("=" * 60)
for n in range(0, 12):
    subset = features[features['All_NBA_1_count'] == n]
    if len(subset) == 0:
        continue
    hof_rate = subset['hof_inducted'].mean()
    non_hof  = subset[subset['hof_inducted']==0]['name'].tolist()[:4]
    print(f"  {n}x: {len(subset):4d} players, {hof_rate:.0%} HOF  |  non-HOF: {non_hof}")

print("\n" + "=" * 60)
print("HOF RATE BY MVP COUNT")
print("=" * 60)
for n in range(0, 8):
    subset = features[features['MVP_count'] == n]
    if len(subset) == 0:
        continue
    hof_rate = subset['hof_inducted'].mean()
    non_hof  = subset[subset['hof_inducted']==0]['name'].tolist()[:4]
    print(f"  {n}x: {len(subset):4d} players, {hof_rate:.0%} HOF  |  non-HOF: {non_hof}")

print("\n" + "=" * 60)
print("HOF RATE BY ALL-STAR COUNT")
print("=" * 60)
for n in [0,1,2,3,4,5,6,7,8,10,12]:
    subset = features[features['All_Star_count'] >= n]
    if len(subset) == 0:
        continue
    hof_rate = subset['hof_inducted'].mean()
    print(f"  {n}+ All-Stars: {len(subset):4d} players, {hof_rate:.0%} HOF rate")

print("\n" + "=" * 60)
print("HOF RATE BY DPOY COUNT")
print("=" * 60)
for n in range(0, 5):
    subset = features[features['DPOY_count'] == n]
    if len(subset) == 0:
        continue
    hof_rate = subset['hof_inducted'].mean()
    non_hof  = subset[subset['hof_inducted']==0]['name'].tolist()[:4]
    print(f"  {n}x: {len(subset):4d} players, {hof_rate:.0%} HOF  |  non-HOF: {non_hof}")

print("\n" + "=" * 60)
print("KEY PLAYER COMPARISON")
print("=" * 60)
players_check = ['rosede01', 'cartevi01', 'wadedw01', 'jamesle01', 'jordami01', 'bryanko01', 'curryst01']
cols = ['name', 'peak7_vorp_sum', 'career_vorp', 'games_played', 'hof_inducted']
cols = [c for c in cols if c in features.columns]
check = features[features['player_id'].isin(players_check)][cols]
print(check.to_string())

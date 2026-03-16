"""
Run this to find natural HOF score thresholds.
Helps us set the floor (0%) and ceiling (100%) for the classifier.
"""
import pandas as pd
import sqlite3
import numpy as np

features = pd.read_csv('data/processed/player_features_with_archetypes.csv')
awards   = pd.read_sql('SELECT * FROM awards', sqlite3.connect('data/hof.db'))

# Only players (not coaches) — retired
df = features[features['retired'] == 1].copy()
df = df[df['games_played'] >= 200].copy()

# Count accolades
for award, col in [('All_NBA_1','all_nba_1'), ('All_Star','all_star'),
                   ('All_NBA_2','all_nba_2'), ('All_NBA_3','all_nba_3')]:
    counts = awards[awards['award']==award].groupby('player_id').size().reset_index(name=col)
    df = df.merge(counts, on='player_id', how='left')
    df[col] = df[col].fillna(0).astype(int)

# Build a simple composite score to find thresholds
# This is not the final model — just for threshold exploration
df['composite'] = (
    df['peak7_vorp_sum'].fillna(0)      * 1.0  +
    df['career_vorp'].fillna(0)         * 0.3  +
    df['all_star'].fillna(0)            * 1.5  +
    df['all_nba_1'].fillna(0)           * 4.0  +
    df['all_nba_2'].fillna(0)           * 2.0  +
    df['all_nba_3'].fillna(0)           * 1.0  +
    df['games_played'].fillna(0)        * 0.02 +
    df['championship_score'].fillna(0)  * 10.0
)

print("=" * 65)
print("COMPOSITE SCORE DISTRIBUTION — HOF vs NON-HOF")
print("=" * 65)
hof    = df[df['hof_inducted'] == 1]['composite']
non    = df[df['hof_inducted'] == 0]['composite']
print(f"\nHOF players  (n={len(hof)}): min={hof.min():.1f} median={hof.median():.1f} max={hof.max():.1f}")
print(f"Non-HOF      (n={len(non)}): min={non.min():.1f} median={non.median():.1f} max={non.max():.1f}")

print("\n--- HOF score percentiles ---")
for p in [5, 10, 25, 50, 75, 90, 95]:
    print(f"  {p}th: {np.percentile(hof, p):.1f}")

print("\n--- Non-HOF score percentiles ---")
for p in [75, 90, 95, 99]:
    print(f"  {p}th: {np.percentile(non, p):.1f}")

print("\n" + "=" * 65)
print("BOTTOM 15 HOF PLAYERS BY COMPOSITE (candidates for floor)")
print("=" * 65)
bottom = df[df['hof_inducted']==1].nsmallest(15, 'composite')
print(bottom[['name','composite','peak7_vorp_sum','career_vorp',
              'all_star','all_nba_1','games_played']].to_string(index=False))

print("\n" + "=" * 65)
print("TOP 15 NON-HOF PLAYERS BY COMPOSITE (snub candidates)")
print("=" * 65)
top_non = df[df['hof_inducted']==0].nlargest(15, 'composite')
print(top_non[['name','composite','peak7_vorp_sum','career_vorp',
               'all_star','all_nba_1','games_played']].to_string(index=False))

print("\n" + "=" * 65)
print("KEY REFERENCE PLAYERS")
print("=" * 65)
refs = ['jordami01','jamesle01','bryanko01','wadedw01','cartevi01',
        'rosede01','smithjr02','paulch01','westbru01','jokicni01']
ref_df = df[df['player_id'].isin(refs)][
    ['name','composite','peak7_vorp_sum','career_vorp','all_star','games_played','hof_inducted']
].sort_values('composite', ascending=False)
print(ref_df.to_string(index=False))

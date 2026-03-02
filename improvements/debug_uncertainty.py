"""Quick check: what does the DB have for uncertainty-relevant features?"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
from src.db import connect

con = connect()

df = con.execute("""
    SELECT ms.musher_id, m.name_canonical as name,
           ms.n_finishes, ms.is_rookie, ms.years_since_last_entry,
           ms.pct_finished, ms.best_finish_place, ms.w_avg_finish_place,
           ms.pct_top10, ms.n_entries,
           ms.w_n_entries
    FROM musher_strength ms
    LEFT JOIN mushers m ON ms.musher_id = m.musher_id
    WHERE ms.year = 2026
    ORDER BY COALESCE(ms.n_finishes, 0) DESC
""").df()

# Compute uncertainty multiplier
n_fin = pd.to_numeric(df["n_finishes"], errors="coerce").fillna(0).values
is_rook = pd.to_numeric(df["is_rookie"], errors="coerce").fillna(0).values
ysl = pd.to_numeric(df["years_since_last_entry"], errors="coerce").fillna(0).values

unc = np.sqrt(8.0 / np.maximum(n_fin, 1.0))
unc = np.where(is_rook == 1, 1.8, unc)
absence_add = np.minimum(np.maximum(ysl - 1, 0) * 0.05, 0.4)
unc = np.clip(unc + absence_add, 0.85, 2.0)

df["unc_mult"] = unc.round(3)

print("2026 FIELD: UNCERTAINTY MULTIPLIER INPUTS")
print("=" * 100)
print(df[["musher_id", "name", "n_finishes", "n_entries", "is_rookie",
          "years_since_last_entry", "pct_finished", "best_finish_place", "unc_mult"
          ]].to_string(index=False))

# Flag suspicious cases
print("\n\nFLAGGED: unc_mult >= 1.8 but n_entries looks high:")
for _, row in df.iterrows():
    n_ent = row.get("n_entries") or row.get("w_n_entries") or 0
    if row["unc_mult"] >= 1.8 and not pd.isna(n_ent) and float(n_ent) > 3:
        print(f"  {row['name']} (id={row['musher_id']}): "
              f"n_finishes={row['n_finishes']}, n_entries={n_ent}, "
              f"pct_finished={row['pct_finished']}, unc={row['unc_mult']}")

# Also check historical_results directly for Redington and Porsild
print("\n\n=== HISTORICAL RESULTS: Ryan Redington ===")
hr = con.execute("""
    SELECT year, finish_place, status 
    FROM historical_results 
    WHERE musher_id = '83' 
    ORDER BY year
""").df()
if hr.empty:
    hr = con.execute("""
        SELECT year, finish_place, status 
        FROM entries 
        WHERE musher_id = '83' 
        ORDER BY year
    """).df()
print(hr.to_string(index=False))

print("\n=== HISTORICAL RESULTS: Mille Porsild ===")
hr2 = con.execute("""
    SELECT year, finish_place, status 
    FROM historical_results 
    WHERE musher_id = '1077' 
    ORDER BY year
""").df()
if hr2.empty:
    hr2 = con.execute("""
        SELECT year, finish_place, status 
        FROM entries 
        WHERE musher_id = '1077' 
        ORDER BY year
    """).df()
print(hr2.to_string(index=False))
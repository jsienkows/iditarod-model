# src/eval/diagnose_regression.py
"""
Diagnose remaining-time regression errors.
Run: python -m src.eval.diagnose_regression
"""
import numpy as np
import pandas as pd
from src.db import connect


def main():
    con = connect()

    # 1. Remaining time distribution by checkpoint_pct bucket
    print("=" * 70)
    print("REMAINING TIME BY CHECKPOINT_PCT BUCKET (training data, 2016-2024)")
    print("=" * 70)
    df = con.execute("""
        SELECT
            ROUND(s.checkpoint_pct * 5) / 5 AS pct_bucket,
            COUNT(*) AS n,
            ROUND(AVG(s.finish_time_seconds - s.cum_elapsed_seconds) / 3600, 1) AS avg_remaining_hrs,
            ROUND(STDDEV(s.finish_time_seconds - s.cum_elapsed_seconds) / 3600, 1) AS std_remaining_hrs,
            ROUND(MIN(s.finish_time_seconds - s.cum_elapsed_seconds) / 3600, 1) AS min_remaining_hrs,
            ROUND(MAX(s.finish_time_seconds - s.cum_elapsed_seconds) / 3600, 1) AS max_remaining_hrs
        FROM snapshots s
        WHERE s.year BETWEEN 2016 AND 2024
          AND s.finished = 1
          AND s.finish_time_seconds IS NOT NULL
          AND s.cum_elapsed_seconds IS NOT NULL
          AND s.cum_elapsed_seconds <= s.finish_time_seconds
          AND s.checkpoint_order >= 2
        GROUP BY pct_bucket
        ORDER BY pct_bucket
    """).df()
    print(df.to_string(index=False))

    # 2. Check for NULL checkpoint_pct
    print("\n" + "=" * 70)
    print("NULL CHECKPOINT_PCT ROWS (training data)")
    print("=" * 70)
    nulls = con.execute("""
        SELECT year, checkpoint_order, COUNT(*) as n
        FROM snapshots
        WHERE year BETWEEN 2016 AND 2024
          AND checkpoint_pct IS NULL
        GROUP BY year, checkpoint_order
        ORDER BY year, checkpoint_order
    """).df()
    if nulls.empty:
        print("None — all rows have checkpoint_pct ✅")
    else:
        print(f"WARNING: {nulls['n'].sum()} rows with NULL checkpoint_pct")
        print(nulls.to_string(index=False))

    # 3. Checkpoint_pct vs checkpoint_order correlation
    print("\n" + "=" * 70)
    print("CHECKPOINT_PCT SPREAD PER CHECKPOINT_ORDER")
    print("(If spread is large, same cp_order maps to very different race positions)")
    print("=" * 70)
    spread = con.execute("""
        SELECT
            checkpoint_order,
            COUNT(DISTINCT ROUND(checkpoint_pct, 2)) AS n_distinct_pcts,
            ROUND(MIN(checkpoint_pct), 3) AS min_pct,
            ROUND(MAX(checkpoint_pct), 3) AS max_pct,
            ROUND(MAX(checkpoint_pct) - MIN(checkpoint_pct), 3) AS pct_spread,
            COUNT(*) AS n_rows
        FROM snapshots
        WHERE year BETWEEN 2016 AND 2024
          AND checkpoint_pct IS NOT NULL
        GROUP BY checkpoint_order
        ORDER BY checkpoint_order
    """).df()
    print(spread.to_string(index=False))

    # 4. Dogs data availability
    print("\n" + "=" * 70)
    print("DOGS FEATURE AVAILABILITY")
    print("=" * 70)
    dogs = con.execute("""
        SELECT
            year,
            COUNT(*) as total_rows,
            COUNT(dogs_out) as has_dogs_out,
            COUNT(dogs_dropped) as has_dogs_dropped,
            ROUND(COUNT(dogs_out) * 100.0 / COUNT(*), 1) as pct_dogs_out
        FROM snapshots
        WHERE year BETWEEN 2016 AND 2025
        GROUP BY year
        ORDER BY year
    """).df()
    print(dogs.to_string(index=False))

    # 5. Musher strength join check
    print("\n" + "=" * 70)
    print("MUSHER STRENGTH FEATURE AVAILABILITY IN SNAPSHOTS")
    print("=" * 70)
    ms_check = con.execute("""
        SELECT
            s.year,
            COUNT(*) as snapshot_rows,
            COUNT(ms.best_finish_place) as has_best_finish,
            COUNT(ms.is_rookie) as has_is_rookie,
            ROUND(COUNT(ms.best_finish_place) * 100.0 / COUNT(*), 1) as pct_joined
        FROM snapshots s
        LEFT JOIN musher_strength ms ON s.year = ms.year AND s.musher_id = ms.musher_id
        WHERE s.year BETWEEN 2016 AND 2025
        GROUP BY s.year
        ORDER BY s.year
    """).df()
    print(ms_check.to_string(index=False))

    # 6. 2025 specifically - what does the test set look like?
    print("\n" + "=" * 70)
    print("2025 TEST SET SUMMARY")
    print("=" * 70)
    test_2025 = con.execute("""
        SELECT
            checkpoint_order,
            ROUND(checkpoint_pct, 3) AS cp_pct,
            COUNT(*) AS n_mushers,
            ROUND(AVG(cum_elapsed_seconds) / 3600, 1) AS avg_elapsed_hrs,
            COUNT(dogs_out) AS has_dogs,
            ROUND(AVG(finish_time_seconds) / 3600, 1) AS avg_actual_finish_hrs
        FROM snapshots
        WHERE year = 2025
        GROUP BY checkpoint_order, cp_pct
        ORDER BY checkpoint_order
    """).df()
    print(test_2025.to_string(index=False))


if __name__ == "__main__":
    main()
# src/features/race_context.py
"""
Year-level race context features: route regime, year difficulty proxies.

Usage:
    python -m src.features.race_context
    # Populates/updates the races table with route_regime and computed difficulty proxies.
"""

import argparse
from src.db import connect
from src.features.checkpoint_distances import get_route_for_year


def route_regime_for_year(year: int) -> str:
    return get_route_for_year(year)


def populate_race_metadata(con, year_min: int = 2000, year_max: int = 2026):
    """
    Ensure the races table has route_regime set for each year, and compute
    year-difficulty proxy features from entries/finishers data.
    """
    # Ensure columns exist
    for col, ctype in [
        ("median_finish_time_seconds", "DOUBLE"),
        ("pct_finishers", "DOUBLE"),
        ("spread_1st_10th_seconds", "DOUBLE"),
    ]:
        try:
            con.execute(f"ALTER TABLE races ADD COLUMN IF NOT EXISTS {col} {ctype}")
        except Exception:
            try:
                con.execute(f"ALTER TABLE races ADD COLUMN {col} {ctype}")
            except Exception:
                pass

    # Get years that have entries
    years_with_data = con.execute(
        "SELECT DISTINCT year FROM entries WHERE year BETWEEN ? AND ? ORDER BY year",
        [year_min, year_max],
    ).fetchall()

    for (year,) in years_with_data:
        regime = route_regime_for_year(year)

        # Upsert route_regime
        existing = con.execute(
            "SELECT COUNT(*) FROM races WHERE year = ?", [year]
        ).fetchone()[0]

        if existing == 0:
            con.execute(
                "INSERT INTO races (year, route_regime) VALUES (?, ?)",
                [year, regime],
            )
        else:
            con.execute(
                "UPDATE races SET route_regime = ? WHERE year = ?",
                [regime, year],
            )

        # Compute year-difficulty proxies from entries
        stats = con.execute("""
            SELECT
                MEDIAN(finish_time_seconds) FILTER (WHERE finish_place IS NOT NULL) AS median_ft,
                COUNT(*) FILTER (WHERE finish_place IS NOT NULL) * 1.0 / NULLIF(COUNT(*), 0) AS pct_fin,
                (
                    MAX(finish_time_seconds) FILTER (WHERE finish_place <= 10)
                    - MIN(finish_time_seconds) FILTER (WHERE finish_place <= 10)
                ) AS spread_1_10
            FROM entries
            WHERE year = ?
        """, [year]).fetchone()

        if stats:
            median_ft, pct_fin, spread = stats
            con.execute("""
                UPDATE races
                SET median_finish_time_seconds = ?,
                    pct_finishers = ?,
                    spread_1st_10th_seconds = ?
                WHERE year = ?
            """, [median_ft, pct_fin, spread, year])

    print(f"Race metadata populated for years {year_min}–{year_max}")

    # Show summary
    summary = con.execute("""
        SELECT year, route_regime, pct_finishers,
               ROUND(median_finish_time_seconds / 3600.0, 1) AS median_hours,
               ROUND(spread_1st_10th_seconds / 3600.0, 1) AS spread_hours
        FROM races
        WHERE year BETWEEN ? AND ?
        ORDER BY year
    """, [year_min, year_max]).df()
    print(summary.to_string(index=False))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--year_min", type=int, default=2016)
    ap.add_argument("--year_max", type=int, default=2025)
    args = ap.parse_args()

    con = connect()
    populate_race_metadata(con, args.year_min, args.year_max)


if __name__ == "__main__":
    main()
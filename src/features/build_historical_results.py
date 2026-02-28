from src.db import connect
import pandas as pd

def main():
    con = connect()

    df = con.execute("""
        SELECT
            year,
            musher_id,
            finish_place,
            finish_time_seconds,
            status
        FROM entries
        WHERE year BETWEEN 2006 AND 2025
    """).df()

    # Only consider finishers for ranking metrics
    finishers = df[df["finish_place"].notnull()].copy()

    # Compute percentile within each year
    finishers["finish_pct"] = (
        finishers.groupby("year")["finish_place"]
        .rank(method="max", pct=True)
    )

    # Compute winner time per year
    winner_times = (
        finishers.loc[finishers["finish_place"] == 1]
        .set_index("year")["finish_time_seconds"]
    )

    finishers["gap_to_winner_seconds"] = (
        finishers["finish_time_seconds"]
        - finishers["year"].map(winner_times)
    )

    # Merge back into full dataset
    df = df.merge(
        finishers[
            ["year", "musher_id", "finish_pct", "gap_to_winner_seconds"]
        ],
        on=["year", "musher_id"],
        how="left"
    )

    # Save as table
    con.execute("DROP TABLE IF EXISTS historical_results")

    con.execute("""
        CREATE TABLE historical_results AS
        SELECT * FROM df
    """)

    print("Built historical_results table ✅")
    print(con.execute("""
        SELECT year,
               COUNT(*) AS entries,
               COUNT(*) FILTER (WHERE finish_place IS NOT NULL) AS finishers
        FROM historical_results
        GROUP BY year
        ORDER BY year
    """).df())

if __name__ == "__main__":
    main()

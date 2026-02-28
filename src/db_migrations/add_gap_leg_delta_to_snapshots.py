"""Migration: add rank_pct, gap_delta, and leg_delta columns to the snapshots table."""

from src.db import connect

def main():
    con = connect()

    con.execute("ALTER TABLE snapshots ADD COLUMN IF NOT EXISTS rank_pct DOUBLE")
    con.execute("ALTER TABLE snapshots ADD COLUMN IF NOT EXISTS gap_delta DOUBLE")
    con.execute("ALTER TABLE snapshots ADD COLUMN IF NOT EXISTS leg_delta DOUBLE")

    print("Added rank_pct, gap_delta, leg_delta to snapshots (if missing).")

if __name__ == "__main__":
    main()

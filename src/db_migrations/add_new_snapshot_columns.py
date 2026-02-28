# src/db_migrations/add_new_snapshot_columns.py
"""
Migration: Add checkpoint_pct, dogs_in, dogs_dropped, pct_dogs_remaining
to existing snapshots table.

Safe to re-run (uses IF NOT EXISTS / try-except).
"""

from src.db import connect


def _ensure_column(con, table: str, col: str, col_type: str):
    try:
        con.execute(f"ALTER TABLE {table} ADD COLUMN IF NOT EXISTS {col} {col_type}")
        return
    except Exception:
        pass
    try:
        con.execute(f"ALTER TABLE {table} ADD COLUMN {col} {col_type}")
    except Exception:
        pass  # Column already exists


def main():
    con = connect()

    # New snapshot columns
    _ensure_column(con, "snapshots", "checkpoint_pct", "DOUBLE")
    _ensure_column(con, "snapshots", "dogs_in", "INTEGER")
    _ensure_column(con, "snapshots", "dogs_dropped", "INTEGER")
    _ensure_column(con, "snapshots", "pct_dogs_remaining", "DOUBLE")

    # Checkpoint distances table
    con.execute("""
        CREATE TABLE IF NOT EXISTS checkpoint_distances (
            year INTEGER,
            checkpoint_order INTEGER,
            checkpoint_name TEXT,
            cumulative_miles DOUBLE,
            checkpoint_pct DOUBLE,
            route TEXT,
            PRIMARY KEY (year, checkpoint_order)
        )
    """)

    # Ensure races table has year-difficulty columns
    _ensure_column(con, "races", "median_finish_time_seconds", "DOUBLE")
    _ensure_column(con, "races", "pct_finishers", "DOUBLE")
    _ensure_column(con, "races", "spread_1st_10th_seconds", "DOUBLE")

    print("Migration complete: new snapshot columns + checkpoint_distances table + races columns ✅")


if __name__ == "__main__":
    main()

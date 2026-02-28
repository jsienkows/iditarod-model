"""
Database connection and schema definition.

Provides a single connect() function that returns a DuckDB connection
with all tables created if they don't already exist. All scripts import
this module to get a consistent database handle.

Database location: data/db/iditarod.duckdb (relative to project root).
"""

import duckdb
from pathlib import Path

DB_PATH = Path("data/db/iditarod.duckdb")

def connect():
    # Make sure the data/db folder exists
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)

    con = duckdb.connect(str(DB_PATH))

    con.execute("""
    CREATE TABLE IF NOT EXISTS snapshots (
        year INTEGER,
        musher_id TEXT,
        checkpoint_order INTEGER,
        checkpoint_pct DOUBLE,
        asof_time_utc TIMESTAMP,

        rank_at_checkpoint INTEGER,
        rank_pct DOUBLE,

        dogs_in INTEGER,
        dogs_out INTEGER,
        dogs_dropped INTEGER,
        pct_dogs_remaining DOUBLE,

        rest_cum_seconds DOUBLE,
        rest_last_seconds DOUBLE,

        last_leg_seconds DOUBLE,
        leg_delta DOUBLE,

        cum_elapsed_seconds DOUBLE,

        rank_delta DOUBLE,

        gap_to_leader_seconds DOUBLE,
        gap_delta DOUBLE,

        gap_to_10th_seconds DOUBLE,

        pace_last_leg_vs_median DOUBLE,
        pace_cum_vs_median DOUBLE,

        won INTEGER,
        top10 INTEGER,
        finished INTEGER,
        finish_time_seconds DOUBLE,

        PRIMARY KEY (year, musher_id, checkpoint_order)
    );
CREATE TABLE IF NOT EXISTS raw_pages (
        url TEXT PRIMARY KEY,
        fetched_at TIMESTAMP,
        page_type TEXT,
        year INTEGER,
        checkpoint_name TEXT,
        html TEXT
    );

    CREATE TABLE IF NOT EXISTS races (
        year INTEGER PRIMARY KEY,
        route_regime TEXT,
        route_name TEXT,
        notes TEXT,
        distance_miles DOUBLE
    );

    CREATE TABLE IF NOT EXISTS mushers (
        musher_id TEXT PRIMARY KEY,
        name_canonical TEXT,
        first_race_year INTEGER,
        latest_race_year INTEGER,
        total_winnings DOUBLE,
        is_champion BOOLEAN
    );

    CREATE TABLE IF NOT EXISTS entries (
        year INTEGER,
        musher_id TEXT,
        bib INTEGER,
        finish_place INTEGER,
        finish_time_seconds BIGINT,
        status TEXT,
        PRIMARY KEY (year, musher_id)
    );

    CREATE TABLE IF NOT EXISTS checkpoints (
        year INTEGER,
        checkpoint_order INTEGER,
        checkpoint_name TEXT,
        PRIMARY KEY (year, checkpoint_order)
    );

    CREATE TABLE IF NOT EXISTS splits (
        year INTEGER,
        musher_id TEXT,
        checkpoint_order INTEGER,
        checkpoint_name TEXT,
        in_time_utc TIMESTAMP,
        out_time_utc TIMESTAMP,
        rest_seconds BIGINT,
        time_en_route_seconds BIGINT,
        dogs_in INTEGER,
        dogs_out INTEGER,
        rank_at_checkpoint INTEGER,
        PRIMARY KEY (year, musher_id, checkpoint_order)
    );
    """)

    return con

from src.db import connect

def main():
    con = connect()

    con.execute("DROP TABLE IF EXISTS snapshots")

    con.execute("""
    CREATE TABLE snapshots (
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
        rest_cum_seconds BIGINT,
        rest_last_seconds BIGINT,
        last_leg_seconds BIGINT,
        leg_delta BIGINT,
        cum_elapsed_seconds BIGINT,
        rank_delta INTEGER,

        gap_to_leader_seconds BIGINT,
        gap_delta BIGINT,
        gap_to_10th_seconds BIGINT,
        pace_last_leg_vs_median BIGINT,
        pace_cum_vs_median BIGINT,

        won INTEGER,
        top10 INTEGER,
        finished INTEGER,
        finish_time_seconds BIGINT,

        PRIMARY KEY (year, musher_id, checkpoint_order)
    );
    """)

    print("snapshots table reset ✅")

if __name__ == "__main__":
    main()

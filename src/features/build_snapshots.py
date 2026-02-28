"""
Build checkpoint-level snapshot features for the in-race model.

Transforms raw split times into feature vectors at each checkpoint:
pace vs. median, gap to leader, rest strategy, dogs remaining, rank
deltas, and pre-race musher strength priors.

Usage:
    python -m src.features.build_snapshots --year 2025
"""

import argparse
import pandas as pd
from src.db import connect


DEFAULT_MAX_LEG_SECONDS = 36 * 3600  # sanity window (36h)


def build_snapshots_for_year(con, year: int, max_leg_seconds: int = DEFAULT_MAX_LEG_SECONDS) -> pd.DataFrame:
    splits = con.execute(
        """
        SELECT
          year, musher_id, checkpoint_order, checkpoint_name,
          in_time_utc,
          out_time_utc,
          COALESCE(rest_seconds, 0) AS rest_seconds,
          time_en_route_seconds,
          dogs_in,
          dogs_out
        FROM splits
        WHERE year = ?
        """,
        [year],
    ).df()

    if splits.empty:
        raise RuntimeError(f"No splits found for year={year}. Did you build/scrape splits for this year?")

    # --- Parse timestamps early and harden dtype ---
    splits["in_time_utc"] = pd.to_datetime(splits["in_time_utc"], utc=True, errors="coerce")
    splits["out_time_utc"] = pd.to_datetime(splits["out_time_utc"], utc=True, errors="coerce")

    # Force ns precision + tz-aware dtype (prevents datetime dtype warnings)
    splits["in_time_utc"] = splits["in_time_utc"].astype("datetime64[ns, UTC]")
    splits["out_time_utc"] = splits["out_time_utc"].astype("datetime64[ns, UTC]")

    splits = splits.sort_values(["musher_id", "checkpoint_order"]).reset_index(drop=True)

    # If scraper duplicated rows per musher/checkpoint, keep first occurrence
    splits = splits.drop_duplicates(subset=["musher_id", "checkpoint_order"])

    # Drop rows with no arrival time (can't build timeline without it)
    splits = splits[splits["in_time_utc"].notna()].copy()

    # --- Fix out_time: keep only plausible out>=in and within MAX_LEG_SECONDS ---
    out_ok = (
        splits["out_time_utc"].notna()
        & (splits["out_time_utc"] >= splits["in_time_utc"])
        & ((splits["out_time_utc"] - splits["in_time_utc"]).dt.total_seconds() <= max_leg_seconds)
    )

    # Create fixed-out column with correct dtype from the start
    splits["out_time_fixed_utc"] = pd.Series(pd.NaT, index=splits.index, dtype="datetime64[ns, UTC]")
    splits.loc[out_ok, "out_time_fixed_utc"] = splits.loc[out_ok, "out_time_utc"]

    # Robust as-of timestamp: prefer fixed OUT, else IN
    splits["asof_time_utc"] = splits["out_time_fixed_utc"].combine_first(splits["in_time_utc"])
    splits["asof_time_utc"] = splits["asof_time_utc"].astype("datetime64[ns, UTC]")

    # --- Time features derived from asof_time_utc ---
    START_CP = 2  # restart anchor (avoids ceremonial start mismatch)

    start_times = (
        splits.loc[splits["checkpoint_order"] >= START_CP]
        .groupby("musher_id")["asof_time_utc"]
        .min()
    )

    splits["start_time_utc"] = splits["musher_id"].map(start_times)

    # Drop mushers where we can't determine restart-anchored start_time
    splits = splits[splits["start_time_utc"].notna()].copy()

    splits["cum_elapsed_seconds"] = (splits["asof_time_utc"] - splits["start_time_utc"]).dt.total_seconds()

    # Optional but recommended: don’t allow negative elapsed at early checkpoints
    splits.loc[splits["cum_elapsed_seconds"] < 0, "cum_elapsed_seconds"] = 0

    splits["cum_elapsed_seconds"] = (splits["asof_time_utc"] - splits["start_time_utc"]).dt.total_seconds()

    splits["prev_asof_time_utc"] = splits.groupby(["musher_id"])["asof_time_utc"].shift(1)
    splits["last_leg_seconds"] = (splits["asof_time_utc"] - splits["prev_asof_time_utc"]).dt.total_seconds()

    # If negative (bad ordering), drop to NA then fill 0 (prevents nonsense legs)
    splits.loc[splits["last_leg_seconds"] < 0, "last_leg_seconds"] = pd.NA
    splits["last_leg_seconds"] = splits["last_leg_seconds"].fillna(0)

    # --- Rest features ---
    splits["rest_seconds"] = pd.to_numeric(splits["rest_seconds"], errors="coerce").fillna(0)
    splits["rest_cum_seconds"] = splits.groupby(["musher_id"])["rest_seconds"].cumsum()
    splits["rest_last_seconds"] = splits["rest_seconds"]

    # --- Recompute time_en_route_seconds (prefer IN - prev fixed OUT; fallback to last_leg_seconds) ---
    splits["prev_out_fixed_utc"] = splits.groupby(["musher_id"])["out_time_fixed_utc"].shift(1)
    splits["prev_out_fixed_utc"] = splits["prev_out_fixed_utc"].astype("datetime64[ns, UTC]")

    calc_enroute = (splits["in_time_utc"] - splits["prev_out_fixed_utc"]).dt.total_seconds()

    splits["time_en_route_seconds"] = pd.to_numeric(
        splits["time_en_route_seconds"], errors="coerce"
    ).astype("Float64")

    use_calc = calc_enroute.between(0, max_leg_seconds)
    splits.loc[use_calc, "time_en_route_seconds"] = calc_enroute[use_calc]
    splits["time_en_route_seconds"] = splits["time_en_route_seconds"].fillna(splits["last_leg_seconds"])

    # --- Rank features (1 = leader) using cum_elapsed_seconds ---
    splits["rank_at_checkpoint"] = (
        splits.groupby(["year", "checkpoint_order"])["cum_elapsed_seconds"]
        .rank(method="min")
        .astype("Int64")
    )

    splits["rank_prev"] = splits.groupby(["musher_id"])["rank_at_checkpoint"].shift(1)
    splits["rank_delta"] = splits["rank_at_checkpoint"] - splits["rank_prev"]

    splits["n_at_checkpoint"] = splits.groupby(["year", "checkpoint_order"])["musher_id"].transform("count")
    splits["rank_pct"] = (splits["rank_at_checkpoint"] - 1) / (splits["n_at_checkpoint"] - 1)
    splits.loc[splits["n_at_checkpoint"] <= 1, "rank_pct"] = 0.0

    # --- Relative-to-field features (NO apply; uses transform to preserve columns) ---
    gcp = splits.groupby("checkpoint_order")

    # leader gap
    leader_time = gcp["cum_elapsed_seconds"].transform("min")
    splits["gap_to_leader_seconds"] = splits["cum_elapsed_seconds"] - leader_time

    # "10th place" time at each checkpoint:
    rank_within_cp = gcp["cum_elapsed_seconds"].rank(method="first", ascending=True)
    n_within_cp = gcp["musher_id"].transform("count")
    kth = n_within_cp.clip(upper=10)

    tenth_time = (
        splits["cum_elapsed_seconds"]
        .where(rank_within_cp == kth)
        .groupby(splits["checkpoint_order"])
        .transform("max")
    )
    splits["gap_to_10th_seconds"] = splits["cum_elapsed_seconds"] - tenth_time

    # pace vs median
    med_last = gcp["last_leg_seconds"].transform("median")
    med_cum = gcp["cum_elapsed_seconds"].transform("median")
    splits["pace_last_leg_vs_median"] = splits["last_leg_seconds"] - med_last
    splits["pace_cum_vs_median"] = splits["cum_elapsed_seconds"] - med_cum

    # --- Momentum / delta features ---
    splits["gap_prev"] = splits.groupby(["musher_id"])["gap_to_leader_seconds"].shift(1)
    splits["gap_delta"] = (splits["gap_to_leader_seconds"] - splits["gap_prev"]).fillna(0)

    splits["leg_prev"] = splits.groupby(["musher_id"])["last_leg_seconds"].shift(1)
    splits["leg_delta"] = (splits["last_leg_seconds"] - splits["leg_prev"]).fillna(0)

    # --- Dog team features ---
    splits["dogs_in"] = pd.to_numeric(splits["dogs_in"], errors="coerce").astype("Int64")
    splits["dogs_out"] = pd.to_numeric(splits["dogs_out"], errors="coerce").astype("Int64")

    # dogs_dropped: how many dogs have been dropped since the start (use dogs_out relative to max team size)
    # Most teams start with 14 dogs; use each musher's first recorded dogs_in as their starting count
    first_dogs = (
        splits.loc[splits["dogs_in"].notna()]
        .groupby("musher_id")["dogs_in"]
        .first()
    )
    splits["start_dogs"] = splits["musher_id"].map(first_dogs)
    splits["dogs_dropped"] = (splits["start_dogs"] - splits["dogs_out"]).clip(lower=0)
    splits["pct_dogs_remaining"] = (
        splits["dogs_out"].astype("Float64") / splits["start_dogs"].astype("Float64")
    )

    # --- Checkpoint distance normalization (Priority #1 fix) ---
    # Join checkpoint_distances for checkpoint_pct (fraction of total race distance)
    try:
        cp_dist = con.execute(
            """
            SELECT checkpoint_order, cumulative_miles, checkpoint_pct
            FROM checkpoint_distances
            WHERE year = ?
            """,
            [year],
        ).df()

        if not cp_dist.empty:
            splits = splits.merge(cp_dist, on="checkpoint_order", how="left")
        else:
            splits["cumulative_miles"] = pd.NA
            splits["checkpoint_pct"] = pd.NA
    except Exception:
        # Table may not exist yet — graceful fallback
        splits["cumulative_miles"] = pd.NA
        splits["checkpoint_pct"] = pd.NA

    # --- Join entries / labels (keep your existing semantics) ---
    entries = con.execute(
        """
        SELECT year, musher_id, finish_place, finish_time_seconds
        FROM entries
        WHERE year = ?
        """,
        [year],
    ).df()

    df = splits.merge(entries, on=["year", "musher_id"], how="left")

    # Keep EXACT same labeling behavior as build_snapshots_2025.py:
    # - finish_place NULL => 999 (DNF/unknown treated as not-finished)
    df["finish_place"] = df["finish_place"].fillna(999)
    df["won"] = (df["finish_place"] == 1).astype("int64")
    df["top10"] = (df["finish_place"] <= 10).astype("int64")
    df["finished"] = (df["finish_place"] != 999).astype("int64")

    snaps = df[
        [
            "year",
            "musher_id",
            "checkpoint_order",
            "checkpoint_pct",
            "asof_time_utc",
            "rank_at_checkpoint",
            "rank_pct",
            "dogs_in",
            "dogs_out",
            "dogs_dropped",
            "pct_dogs_remaining",
            "rest_cum_seconds",
            "rest_last_seconds",
            "last_leg_seconds",
            "leg_delta",
            "cum_elapsed_seconds",
            "rank_delta",
            "gap_to_leader_seconds",
            "gap_delta",
            "gap_to_10th_seconds",
            "pace_last_leg_vs_median",
            "pace_cum_vs_median",
            "won",
            "top10",
            "finished",
            "finish_time_seconds",
        ]
    ].copy()

    snaps = snaps[snaps["asof_time_utc"].notna()].copy()
    # --- Drop post-finish snapshots (caused by OUT/ceremonial times being later than official finish_time_seconds) ---
    bad_post_finish = (
        (snaps["finished"] == 1)
        & snaps["finish_time_seconds"].notna()
        & snaps["cum_elapsed_seconds"].notna()
        & (snaps["cum_elapsed_seconds"] > snaps["finish_time_seconds"])
    )

    n_bad = int(bad_post_finish.sum())
    if n_bad:
        print(f"WARNING: dropping {n_bad} post-finish snapshot rows where cum_elapsed > finish_time_seconds")
        snaps = snaps.loc[~bad_post_finish].copy()


    return snaps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--year", type=int, required=True, help="Race year to build snapshots for")
    ap.add_argument(
        "--max_leg_hours",
        type=float,
        default=36.0,
        help="Sanity cap for a single leg duration (hours). Default=36.",
    )
    args = ap.parse_args()

    year = int(args.year)
    max_leg_seconds = int(args.max_leg_hours * 3600)

    con = connect()
    snaps = build_snapshots_for_year(con, year=year, max_leg_seconds=max_leg_seconds)

    con.execute("DELETE FROM snapshots WHERE year = ?", [year])
    con.register("snaps_df", snaps)

    con.execute(
        """
        INSERT INTO snapshots (
          year,
          musher_id,
          checkpoint_order,
          checkpoint_pct,
          asof_time_utc,
          rank_at_checkpoint,
          rank_pct,
          dogs_in,
          dogs_out,
          dogs_dropped,
          pct_dogs_remaining,
          rest_cum_seconds,
          rest_last_seconds,
          last_leg_seconds,
          leg_delta,
          cum_elapsed_seconds,
          rank_delta,
          gap_to_leader_seconds,
          gap_delta,
          gap_to_10th_seconds,
          pace_last_leg_vs_median,
          pace_cum_vs_median,
          won,
          top10,
          finished,
          finish_time_seconds
        )
        SELECT
          year,
          musher_id,
          checkpoint_order,
          checkpoint_pct,
          asof_time_utc,
          rank_at_checkpoint,
          rank_pct,
          dogs_in,
          dogs_out,
          dogs_dropped,
          pct_dogs_remaining,
          rest_cum_seconds,
          rest_last_seconds,
          last_leg_seconds,
          leg_delta,
          cum_elapsed_seconds,
          rank_delta,
          gap_to_leader_seconds,
          gap_delta,
          gap_to_10th_seconds,
          pace_last_leg_vs_median,
          pace_cum_vs_median,
          won,
          top10,
          finished,
          finish_time_seconds
        FROM snaps_df
        """
    )

    print(f"Built snapshots for year={year} - rows inserted: {len(snaps)} ✅")


if __name__ == "__main__":
    main()

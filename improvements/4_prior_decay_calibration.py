"""
4_prior_decay_calibration.py
============================
Empirical analysis of how much pre-race musher priors contribute to
prediction accuracy at each checkpoint, with recommendations for
optimal decay.

BACKGROUND:
  Your HistGBT model includes musher prior features (best_finish_place,
  w_pct_top10, etc.) alongside in-race snapshot features. The model
  learns implicitly how much to weight priors at each checkpoint. But
  this implicit weighting may not be optimal — especially early in the
  race where priors should dominate, and late where they should fade.

  The Reddit suggestion was:
    weight_prior = 1 / (1 + n_checkpoints_seen)
  And to plot realized rank accuracy vs. prior weight across checkpoints
  to see if accuracy improves faster than the weight decays.

WHAT THIS SCRIPT DOES:
  1. Runs LOOCV backtest at each checkpoint with TWO model variants:
     (a) Full model (priors + snapshots)
     (b) Snapshot-only model (no priors)
  2. Measures the marginal value of priors at each checkpoint.
  3. Fits an empirical decay curve showing when priors stop helping.
  4. Suggests optimal prior fade schedule.

  The output tells you whether the HistGBT is already handling this well
  or if explicit prior blending would help.

Usage:
    python improvements/4_prior_decay_calibration.py
    python improvements/4_prior_decay_calibration.py --start_year 2016 --end_year 2024
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from scipy.optimize import minimize_scalar

from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor

from src.db import connect


SNAPSHOT_FEATURES = [
    "checkpoint_pct", "rank_at_checkpoint", "rank_pct",
    "dogs_out", "dogs_dropped", "pct_dogs_remaining",
    "rest_cum_seconds", "rest_last_seconds", "last_leg_seconds", "leg_delta",
    "cum_elapsed_seconds", "rank_delta", "gap_to_leader_seconds", "gap_delta",
    "gap_to_10th_seconds", "pace_last_leg_vs_median", "pace_cum_vs_median",
]

MUSHER_PRIOR_FEATURES = [
    "best_finish_place", "pct_top10", "pct_finished", "n_finishes",
    "last3_avg_finish_place", "w_pct_top10", "w_avg_finish_place",
    "years_since_last_entry", "is_rookie", "career_race_number",
    "trajectory", "last_year_improvement",
]

RACE_CONTEXT_FEATURES = ["is_northern_route"]

FULL_FEATURES = SNAPSHOT_FEATURES + MUSHER_PRIOR_FEATURES + RACE_CONTEXT_FEATURES
SNAPSHOT_ONLY = SNAPSHOT_FEATURES + RACE_CONTEXT_FEATURES


def _load_all_snapshots(con, year_min, year_max):
    df = con.execute("""
        SELECT
          s.year, s.musher_id, s.checkpoint_order, s.checkpoint_pct,
          s.rank_at_checkpoint, s.rank_pct, s.dogs_in, s.dogs_out,
          s.dogs_dropped, s.pct_dogs_remaining,
          s.rest_cum_seconds, s.rest_last_seconds, s.last_leg_seconds, s.leg_delta,
          s.cum_elapsed_seconds, s.rank_delta, s.gap_to_leader_seconds, s.gap_delta,
          s.gap_to_10th_seconds, s.pace_last_leg_vs_median, s.pace_cum_vs_median,
          s.finished, s.finish_time_seconds, s.won, s.top10,
          ms.best_finish_place, ms.pct_top10, ms.pct_finished, ms.n_finishes,
          ms.last3_avg_finish_place, ms.w_pct_top10, ms.w_avg_finish_place,
          ms.years_since_last_entry, ms.is_rookie, ms.career_race_number,
          ms.trajectory, ms.last_year_improvement,
          CASE WHEN r.route_regime IN ('northern') THEN 1 ELSE 0 END AS is_northern_route,
          -- Actual finish place for evaluation
          e.finish_place AS actual_finish_place
        FROM snapshots s
        LEFT JOIN musher_strength ms ON s.year = ms.year AND s.musher_id = ms.musher_id
        LEFT JOIN races r ON s.year = r.year
        LEFT JOIN entries e ON s.year = e.year AND s.musher_id = e.musher_id
        WHERE s.year BETWEEN ? AND ?
    """, [year_min, year_max]).df()

    for c in FULL_FEATURES:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    df["actual_finish_place"] = pd.to_numeric(df["actual_finish_place"], errors="coerce")

    return df


def evaluate_at_checkpoint(df_all, test_year, cp, feature_sets):
    """
    Train remaining-time regression on all years except test_year,
    predict at given checkpoint, evaluate ranking accuracy.

    feature_sets: dict of {name: list_of_feature_cols}
    Returns dict of {name: {spearman, p_at_10, winner_rank}}
    """
    d_train = df_all[df_all["year"] != test_year].copy()
    d_test = df_all[(df_all["year"] == test_year) & (df_all["checkpoint_order"] == cp)].copy()

    if d_test.empty or len(d_test) < 5:
        return None

    # Filter to finishers with valid data for training regression
    reg_mask = (
        (d_train["finished"] == 1)
        & d_train["finish_time_seconds"].notna()
        & d_train["cum_elapsed_seconds"].notna()
        & (d_train["cum_elapsed_seconds"] <= d_train["finish_time_seconds"])
        & (d_train["checkpoint_order"] >= 2)
    )
    d_train_reg = d_train[reg_mask].copy()
    d_train_reg["remaining_seconds"] = d_train_reg["finish_time_seconds"] - d_train_reg["cum_elapsed_seconds"]
    d_train_reg = d_train_reg[d_train_reg["remaining_seconds"] >= 0].copy()

    if d_train_reg.empty or len(d_train_reg) < 20:
        return None

    y_train_log = np.log1p(d_train_reg["remaining_seconds"].to_numpy(dtype=float))

    results = {}
    for name, features in feature_sets.items():
        avail = [f for f in features if f in d_train_reg.columns]
        if not avail:
            continue

        reg = HistGradientBoostingRegressor(
            max_depth=3, learning_rate=0.05, max_iter=800,
            min_samples_leaf=20, random_state=42,
        )
        reg.fit(d_train_reg[avail], y_train_log)

        # Predict on test checkpoint
        pred_log = reg.predict(d_test[avail])
        pred_remaining = np.expm1(pred_log)
        cum = pd.to_numeric(d_test["cum_elapsed_seconds"], errors="coerce").to_numpy(dtype=float)
        pred_finish = cum + np.clip(pred_remaining, 0, None)

        d_eval = d_test[["musher_id", "actual_finish_place"]].copy()
        d_eval["pred_finish_time"] = pred_finish
        d_eval = d_eval[d_eval["actual_finish_place"].notna()].copy()

        if len(d_eval) < 5:
            continue

        # Spearman correlation
        rho, _ = spearmanr(d_eval["pred_finish_time"], d_eval["actual_finish_place"])

        # Precision@10
        pred_top10 = set(d_eval.nsmallest(10, "pred_finish_time")["musher_id"])
        actual_top10 = set(d_eval.nsmallest(10, "actual_finish_place")["musher_id"])
        p_at_10 = len(pred_top10 & actual_top10) / 10

        # Winner rank
        winner = d_eval[d_eval["actual_finish_place"] == 1]
        if not winner.empty:
            winner_pred_rank = d_eval["pred_finish_time"].rank(ascending=True, method="min").loc[winner.index[0]]
        else:
            winner_pred_rank = np.nan

        results[name] = {
            "spearman": rho,
            "p_at_10": p_at_10,
            "winner_rank": float(winner_pred_rank) if np.isfinite(winner_pred_rank) else np.nan,
        }

    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start_year", type=int, default=2019)
    ap.add_argument("--end_year", type=int, default=2024)
    ap.add_argument("--checkpoints", type=str, default="3,5,8,10,13,15,18,20,22",
                    help="Checkpoints to evaluate at")
    args = ap.parse_args()

    eval_cps = [int(x.strip()) for x in args.checkpoints.split(",")]

    con = connect()
    df_all = _load_all_snapshots(con, args.start_year - 1, args.end_year)
    print(f"Loaded {len(df_all)} snapshot rows")

    feature_sets = {
        "full": FULL_FEATURES,
        "snapshot_only": SNAPSHOT_ONLY,
        "prior_only": MUSHER_PRIOR_FEATURES,
    }

    all_rows = []

    for test_year in range(args.start_year, args.end_year + 1):
        for cp in eval_cps:
            result = evaluate_at_checkpoint(df_all, test_year, cp, feature_sets)
            if result is None:
                continue

            row = {"year": test_year, "checkpoint": cp}
            for model_name, metrics in result.items():
                for metric, value in metrics.items():
                    row[f"{model_name}_{metric}"] = value

            # Marginal value of priors = full - snapshot_only
            if "full_spearman" in row and "snapshot_only_spearman" in row:
                row["prior_marginal_spearman"] = row["full_spearman"] - row["snapshot_only_spearman"]
            if "full_p_at_10" in row and "snapshot_only_p_at_10" in row:
                row["prior_marginal_p10"] = row["full_p_at_10"] - row["snapshot_only_p_at_10"]

            all_rows.append(row)

    results = pd.DataFrame(all_rows)

    if results.empty:
        print("No results generated.")
        return

    # Aggregate by checkpoint
    print(f"\n{'='*90}")
    print("PRIOR DECAY ANALYSIS: Marginal value of musher priors by checkpoint")
    print(f"{'='*90}")

    agg = results.groupby("checkpoint").agg({
        "full_spearman": "mean",
        "snapshot_only_spearman": "mean",
        "prior_marginal_spearman": "mean",
        "full_p_at_10": "mean",
        "snapshot_only_p_at_10": "mean",
        "prior_marginal_p10": "mean",
    }).round(3)

    print(f"\n{'CP':>4} | {'Full':>8} {'SnapOnly':>8} {'Δ Spear':>8} | {'Full':>8} {'SnapOnly':>8} {'Δ P@10':>8}")
    print(f"{'':>4} | {'Spearman':>8} {'Spearman':>8} {'':>8} | {'P@10':>8} {'P@10':>8} {'':>8}")
    print("-" * 72)

    for cp, row in agg.iterrows():
        delta_s = row["prior_marginal_spearman"]
        delta_p = row["prior_marginal_p10"]
        s_sign = "+" if delta_s > 0 else ""
        p_sign = "+" if delta_p > 0 else ""
        print(f"{cp:>4} | {row['full_spearman']:>8.3f} {row['snapshot_only_spearman']:>8.3f} {s_sign}{delta_s:>7.3f} | "
              f"{row['full_p_at_10']:>8.3f} {row['snapshot_only_p_at_10']:>8.3f} {p_sign}{delta_p:>7.3f}")

    # Suggested decay: the checkpoint where prior marginal value drops below 0
    crossover_spearman = None
    crossover_p10 = None
    for cp in sorted(agg.index):
        if agg.loc[cp, "prior_marginal_spearman"] <= 0 and crossover_spearman is None:
            crossover_spearman = cp
        if agg.loc[cp, "prior_marginal_p10"] <= 0 and crossover_p10 is None:
            crossover_p10 = cp

    print(f"\nPrior marginal value → 0 crossover:")
    print(f"  Spearman: {'CP ' + str(crossover_spearman) if crossover_spearman else 'priors helpful at all checkpoints'}")
    print(f"  P@10:     {'CP ' + str(crossover_p10) if crossover_p10 else 'priors helpful at all checkpoints'}")

    # Theoretical decay curves for comparison
    print(f"\nDecay curve comparison (prior weight at each checkpoint):")
    print(f"{'CP':>4} | {'Linear':>8} | {'Hyperbolic':>10} | {'Exponential':>11}")
    print("-" * 48)
    max_cp = max(eval_cps)
    for cp in eval_cps:
        linear = max(0, 1.0 - cp / max_cp)
        hyperbolic = 1.0 / (1.0 + cp)
        exponential = np.exp(-0.15 * cp)
        print(f"{cp:>4} | {linear:>8.3f} | {hyperbolic:>10.3f} | {exponential:>11.3f}")

    print(f"\nRECOMMENDATION:")
    print(f"  If priors add value at all checkpoints → current implicit HistGBT weighting is fine.")
    print(f"  If priors hurt after CP {crossover_spearman or crossover_p10 or '?'} → consider explicit")
    print(f"  prior fading: blend pred = w*prior_pred + (1-w)*snapshot_pred")
    print(f"  where w = 1/(1+n_checkpoints) or matched to the empirical crossover above.")


if __name__ == "__main__":
    main()
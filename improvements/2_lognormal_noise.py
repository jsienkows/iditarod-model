"""
2_lognormal_noise.py
====================
A/B comparison: Gaussian vs Log-normal Monte Carlo noise on the
in-race backtest pipeline.

WHY LOG-NORMAL:
  Gaussian noise is symmetric — equally likely faster or slower.
  But remaining race times are right-skewed: a musher can lose hours to
  blizzards/injuries/fatigue but can't go faster than terrain allows.
  Log-normal naturally produces this asymmetry and never goes negative.

THE MATH:
  Gaussian:    sim_time = T + noise          (additive)
  Log-normal:  sim_time = T * exp(noise)     (multiplicative)

  Where noise ~ N(-sigma_log^2/2, sigma_log^2).
  The bias correction ensures E[sim_time] = T.

Usage:
    python improvements/2_lognormal_noise.py
    python improvements/2_lognormal_noise.py --start_year 2019 --end_year 2024
    python improvements/2_lognormal_noise.py --checkpoints 5,10,15,20
    python improvements/2_lognormal_noise.py --sigma_hours 10.0

After reviewing results, see INTEGRATION section at bottom for how to
permanently switch predict_inrace.py to log-normal.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
import numpy as np
import pandas as pd

from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor

from src.db import connect


# Mirror feature sets from backtest_inrace.py
SNAPSHOT_FEATURES = [
    "checkpoint_pct", "rank_at_checkpoint", "rank_pct",
    "dogs_out", "dogs_dropped", "pct_dogs_remaining",
    "rest_cum_seconds", "rest_last_seconds",
    "last_leg_seconds", "leg_delta", "cum_elapsed_seconds",
    "rank_delta", "gap_to_leader_seconds", "gap_delta",
    "gap_to_10th_seconds", "pace_last_leg_vs_median", "pace_cum_vs_median",
]

MUSHER_PRIOR_FEATURES = [
    "best_finish_place", "pct_top10", "pct_finished", "n_finishes",
    "last3_avg_finish_place", "w_pct_top10", "w_avg_finish_place",
    "years_since_last_entry", "is_rookie", "career_race_number",
    "trajectory", "last_year_improvement",
]

RACE_CONTEXT_FEATURES = ["is_northern_route"]

ALL_FEATURES = SNAPSHOT_FEATURES + MUSHER_PRIOR_FEATURES + RACE_CONTEXT_FEATURES


def _load_all_snapshots(con, year_min, year_max):
    df = con.execute("""
        SELECT
          s.year, s.musher_id, s.checkpoint_order, s.checkpoint_pct,
          s.rank_at_checkpoint, s.rank_pct,
          s.dogs_in, s.dogs_out, s.dogs_dropped, s.pct_dogs_remaining,
          s.rest_cum_seconds, s.rest_last_seconds,
          s.last_leg_seconds, s.leg_delta, s.cum_elapsed_seconds,
          s.rank_delta, s.gap_to_leader_seconds, s.gap_delta,
          s.gap_to_10th_seconds, s.pace_last_leg_vs_median, s.pace_cum_vs_median,
          s.finished, s.finish_time_seconds, s.won, s.top10,
          ms.best_finish_place, ms.pct_top10, ms.pct_finished, ms.n_finishes,
          ms.last3_avg_finish_place, ms.w_pct_top10, ms.w_avg_finish_place,
          ms.years_since_last_entry, ms.is_rookie, ms.career_race_number,
          ms.trajectory, ms.last_year_improvement,
          CASE WHEN r.route_regime IN ('northern') THEN 1 ELSE 0 END AS is_northern_route
        FROM snapshots s
        LEFT JOIN musher_strength ms ON s.year = ms.year AND s.musher_id = ms.musher_id
        LEFT JOIN races r ON s.year = r.year
        WHERE s.year BETWEEN ? AND ?
    """, [year_min, year_max]).df()
    return df


def _coerce_numeric(df, cols):
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


# ==============================================================================
# SIMULATION FUNCTIONS
# ==============================================================================

def simulate_gaussian(pred_finish_hours, p_finish, n_sims, sigma_hours,
                      shared_frac, rng):
    """Original Gaussian (additive) noise simulation."""
    n = len(pred_finish_hours)

    max_ft = np.nanmax(pred_finish_hours)
    if not np.isfinite(max_ft):
        max_ft = 300.0
    dnf_penalty = max_ft + 500.0

    shared_sigma = sigma_hours * np.sqrt(shared_frac)
    indiv_sigma = sigma_hours * np.sqrt(1.0 - shared_frac)

    shared = rng.normal(0, shared_sigma, (n_sims, 1))
    indiv = rng.normal(0, indiv_sigma, (n_sims, n))

    sim_finish = pred_finish_hours[None, :] + shared + indiv

    u = rng.random((n_sims, n))
    sim_is_finish = u < p_finish[None, :]
    sim_finish = np.where(sim_is_finish, sim_finish, dnf_penalty)

    order = np.argsort(sim_finish, axis=1)
    ranks = np.empty_like(order)
    rows = np.arange(n_sims)[:, None]
    ranks[rows, order] = np.arange(n)[None, :]
    place = ranks + 1

    return {
        "p_win": (place == 1).mean(axis=0),
        "p_top10": (place <= 10).mean(axis=0),
        "exp_place": place.mean(axis=0).astype(float),
        "p_finish": p_finish,
    }


def simulate_lognormal(pred_finish_hours, p_finish, n_sims, sigma_hours,
                        shared_frac, rng):
    """Log-normal (multiplicative) noise simulation."""
    n = len(pred_finish_hours)

    max_ft = np.nanmax(pred_finish_hours)
    if not np.isfinite(max_ft):
        max_ft = 300.0
    dnf_penalty = max_ft + 500.0

    # Convert sigma from hours-space to log-space
    T_mean = np.nanmean(pred_finish_hours)
    if T_mean <= 0 or not np.isfinite(T_mean):
        T_mean = 240.0
    sigma_log = sigma_hours / T_mean
    sigma_log = np.clip(sigma_log, 0.01, 0.5)

    shared_sigma_log = sigma_log * np.sqrt(shared_frac)
    indiv_sigma_log = sigma_log * np.sqrt(1.0 - shared_frac)

    # Bias correction: E[exp(X)] = 1 when X ~ N(-s^2/2, s^2)
    shared = rng.normal(-shared_sigma_log**2 / 2, shared_sigma_log, (n_sims, 1))
    indiv = rng.normal(-indiv_sigma_log**2 / 2, indiv_sigma_log, (n_sims, n))

    # Multiplicative: sim_time = pred_time * exp(noise)
    sim_finish = pred_finish_hours[None, :] * np.exp(shared + indiv)

    u = rng.random((n_sims, n))
    sim_is_finish = u < p_finish[None, :]
    sim_finish = np.where(sim_is_finish, sim_finish, dnf_penalty)

    order = np.argsort(sim_finish, axis=1)
    ranks = np.empty_like(order)
    rows = np.arange(n_sims)[:, None]
    ranks[rows, order] = np.arange(n)[None, :]
    place = ranks + 1

    return {
        "p_win": (place == 1).mean(axis=0),
        "p_top10": (place <= 10).mean(axis=0),
        "exp_place": place.mean(axis=0).astype(float),
        "p_finish": p_finish,
    }


# ==============================================================================
# SCORING
# ==============================================================================

def score_simulation(sim, d_cp):
    """Score a simulation result against actual outcomes."""
    actual_winner_mask = d_cp["won"] == 1
    actual_top10_mask = d_cp["top10"] == 1

    winner_p_win = float(sim["p_win"][actual_winner_mask.to_numpy()].mean()) if actual_winner_mask.any() else np.nan
    winner_exp_place = float(sim["exp_place"][actual_winner_mask.to_numpy()].mean()) if actual_winner_mask.any() else np.nan

    pred_top10_ids = set(d_cp.iloc[np.argsort(-sim["p_top10"])[:10]]["musher_id"])
    actual_top10_ids = set(d_cp.loc[actual_top10_mask, "musher_id"])
    overlap = len(pred_top10_ids & actual_top10_ids)
    precision_at_10 = overlap / min(10, len(actual_top10_ids)) if actual_top10_ids else np.nan

    # Log-likelihood of actual winner's p_win (higher = more confident)
    winner_log_prob = float(np.log(sim["p_win"][actual_winner_mask.to_numpy()] + 1e-10).mean()) if actual_winner_mask.any() else np.nan

    # Check for negative sim times (only possible with Gaussian)
    # We can't directly check here since we don't store raw sim times,
    # but we can flag if any p_win is exactly 0 for all mushers (degenerate)

    return {
        "winner_p_win": winner_p_win,
        "winner_exp_place": winner_exp_place,
        "precision_at_10": precision_at_10,
        "top10_overlap": overlap,
        "winner_log_prob": winner_log_prob,
    }


# ==============================================================================
# MAIN: A/B COMPARISON
# ==============================================================================

def main():
    ap = argparse.ArgumentParser(
        description="A/B comparison: Gaussian vs Log-normal noise in in-race backtest"
    )
    ap.add_argument("--start_year", type=int, default=2019)
    ap.add_argument("--end_year", type=int, default=2024)
    ap.add_argument("--checkpoints", type=str, default="5,10,15,20",
                    help="Comma-separated checkpoint orders to evaluate")
    ap.add_argument("--n_sims", type=int, default=5000)
    ap.add_argument("--sigma_hours", type=float, default=12.0)
    ap.add_argument("--shared_noise_frac", type=float, default=0.6)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    eval_cps = [int(x.strip()) for x in args.checkpoints.split(",")]

    con = connect()
    df_all = _load_all_snapshots(con, args.start_year - 1, args.end_year)

    feature_cols = [c for c in ALL_FEATURES if c in df_all.columns]
    df_all = _coerce_numeric(df_all, feature_cols)

    print(f"In-race noise model A/B comparison")
    print(f"  Years: {args.start_year}-{args.end_year}")
    print(f"  Checkpoints: {eval_cps}")
    print(f"  Sims: {args.n_sims}, sigma: {args.sigma_hours}h, shared_frac: {args.shared_noise_frac}")
    print(f"  Features: {len(feature_cols)}")
    print(f"  Total snapshot rows: {len(df_all)}")

    results_gauss = []
    results_lognorm = []

    for test_year in range(args.start_year, args.end_year + 1):
        d_train = df_all[df_all["year"] != test_year].copy()
        d_test = df_all[df_all["year"] == test_year].copy()

        if d_train.empty or d_test.empty:
            continue

        # Train finish model
        d_train_fin = d_train[d_train["finished"].notna()].copy()
        y_train_fin = d_train_fin["finished"].astype(int).to_numpy()
        if len(np.unique(y_train_fin)) < 2:
            continue

        finish_model = HistGradientBoostingClassifier(
            max_depth=3, learning_rate=0.05, max_iter=400,
            min_samples_leaf=20, random_state=42,
        )
        finish_model.fit(d_train_fin[feature_cols], y_train_fin)

        # Train remaining-time model
        d_train_reg = d_train[
            (d_train["finished"] == 1)
            & d_train["finish_time_seconds"].notna()
            & d_train["cum_elapsed_seconds"].notna()
            & (d_train["cum_elapsed_seconds"] <= d_train["finish_time_seconds"])
            & (d_train["checkpoint_order"] >= 2)
        ].copy()
        d_train_reg["remaining_seconds"] = (
            d_train_reg["finish_time_seconds"] - d_train_reg["cum_elapsed_seconds"]
        )
        d_train_reg = d_train_reg[d_train_reg["remaining_seconds"] >= 0].copy()

        if d_train_reg.empty:
            continue

        y_train_log = np.log1p(d_train_reg["remaining_seconds"].to_numpy(dtype=float))
        reg_model = HistGradientBoostingRegressor(
            max_depth=3, learning_rate=0.05, max_iter=800,
            min_samples_leaf=20, random_state=42,
        )
        reg_model.fit(d_train_reg[feature_cols], y_train_log)

        # Evaluate at each checkpoint
        for cp in eval_cps:
            d_cp = d_test[d_test["checkpoint_order"] == cp].copy()
            if d_cp.empty or len(d_cp) < 5:
                continue

            cum_elapsed = pd.to_numeric(d_cp["cum_elapsed_seconds"], errors="coerce").to_numpy(dtype=float)

            # Shared prediction inputs
            p_finish = finish_model.predict_proba(d_cp[feature_cols])[:, 1]
            pred_log = reg_model.predict(d_cp[feature_cols])
            pred_remaining = np.expm1(pred_log)
            pred_remaining = np.clip(pred_remaining, 0, None)
            pred_remaining_hours = pred_remaining / 3600.0
            cum_hours = cum_elapsed / 3600.0
            pred_finish_hours = cum_hours + pred_remaining_hours

            # Use SAME seed for both so differences are purely from noise model
            rng_g = np.random.default_rng(args.seed + test_year * 100 + cp)
            rng_l = np.random.default_rng(args.seed + test_year * 100 + cp)

            sim_g = simulate_gaussian(
                pred_finish_hours, p_finish, args.n_sims,
                args.sigma_hours, args.shared_noise_frac, rng_g,
            )
            sim_l = simulate_lognormal(
                pred_finish_hours, p_finish, args.n_sims,
                args.sigma_hours, args.shared_noise_frac, rng_l,
            )

            score_g = score_simulation(sim_g, d_cp)
            score_l = score_simulation(sim_l, d_cp)

            score_g.update({"test_year": test_year, "checkpoint": cp})
            score_l.update({"test_year": test_year, "checkpoint": cp})

            results_gauss.append(score_g)
            results_lognorm.append(score_l)

            # Per-checkpoint output
            d_wp = score_l["winner_p_win"] - score_g["winner_p_win"]
            d_p10 = score_l["precision_at_10"] - score_g["precision_at_10"]
            tag = "LN+" if d_wp > 0.001 else ("G+" if d_wp < -0.001 else "  =")
            print(f"  {test_year} cp={cp:2d}: "
                  f"win_p G={score_g['winner_p_win']:.3f} LN={score_l['winner_p_win']:.3f} ({tag}) | "
                  f"P@10 G={score_g['precision_at_10']:.2f} LN={score_l['precision_at_10']:.2f}")

    # Build summary
    df_g = pd.DataFrame(results_gauss)
    df_l = pd.DataFrame(results_lognorm)

    if df_g.empty:
        print("\nNo results generated.")
        return

    print(f"\n{'='*90}")
    print("A/B SUMMARY: GAUSSIAN vs LOG-NORMAL NOISE")
    print(f"{'='*90}")

    # Per-checkpoint summary
    print(f"\n{'Checkpoint':>12} | {'winner_p_win':>24} | {'winner_exp_place':>24} | {'precision@10':>24}")
    print(f"{'':>12} | {'Gauss':>8} {'LogN':>8} {'Delta':>7} | "
          f"{'Gauss':>8} {'LogN':>8} {'Delta':>7} | "
          f"{'Gauss':>8} {'LogN':>8} {'Delta':>7}")
    print("-" * 90)

    for cp in eval_cps:
        g_cp = df_g[df_g["checkpoint"] == cp]
        l_cp = df_l[df_l["checkpoint"] == cp]
        if g_cp.empty:
            continue

        g_wp = g_cp["winner_p_win"].mean()
        l_wp = l_cp["winner_p_win"].mean()
        g_ep = g_cp["winner_exp_place"].mean()
        l_ep = l_cp["winner_exp_place"].mean()
        g_p10 = g_cp["precision_at_10"].mean()
        l_p10 = l_cp["precision_at_10"].mean()

        print(f"{'CP '+str(cp):>12} | "
              f"{g_wp:>8.3f} {l_wp:>8.3f} {l_wp-g_wp:>+7.3f} | "
              f"{g_ep:>8.1f} {l_ep:>8.1f} {l_ep-g_ep:>+7.1f} | "
              f"{g_p10:>8.2f} {l_p10:>8.2f} {l_p10-g_p10:>+7.2f}")

    # Overall
    g_wp = df_g["winner_p_win"].mean()
    l_wp = df_l["winner_p_win"].mean()
    g_ep = df_g["winner_exp_place"].mean()
    l_ep = df_l["winner_exp_place"].mean()
    g_p10 = df_g["precision_at_10"].mean()
    l_p10 = df_l["precision_at_10"].mean()

    print("-" * 90)
    print(f"{'OVERALL':>12} | "
          f"{g_wp:>8.3f} {l_wp:>8.3f} {l_wp-g_wp:>+7.3f} | "
          f"{g_ep:>8.1f} {l_ep:>8.1f} {l_ep-g_ep:>+7.1f} | "
          f"{g_p10:>8.2f} {l_p10:>8.2f} {l_p10-g_p10:>+7.2f}")

    # Win count
    merged = df_g[["test_year", "checkpoint", "winner_p_win", "precision_at_10"]].merge(
        df_l[["test_year", "checkpoint", "winner_p_win", "precision_at_10"]],
        on=["test_year", "checkpoint"], suffixes=("_g", "_l"),
    )
    ln_better_wp = (merged["winner_p_win_l"] > merged["winner_p_win_g"] + 0.001).sum()
    g_better_wp = (merged["winner_p_win_g"] > merged["winner_p_win_l"] + 0.001).sum()
    tied_wp = len(merged) - ln_better_wp - g_better_wp

    ln_better_p10 = (merged["precision_at_10_l"] > merged["precision_at_10_g"] + 0.001).sum()
    g_better_p10 = (merged["precision_at_10_g"] > merged["precision_at_10_l"] + 0.001).sum()
    tied_p10 = len(merged) - ln_better_p10 - g_better_p10

    print(f"\nWin count (across {len(merged)} year-checkpoint combos):")
    print(f"  winner_p_win:  LogNorm={ln_better_wp}  Gauss={g_better_wp}  Tied={tied_wp}")
    print(f"  precision@10:  LogNorm={ln_better_p10}  Gauss={g_better_p10}  Tied={tied_p10}")

    # Winner log-probability (calibration metric)
    g_lp = df_g["winner_log_prob"].mean()
    l_lp = df_l["winner_log_prob"].mean()
    print(f"\nAvg winner log-probability (higher = better calibrated):")
    print(f"  Gaussian:   {g_lp:.3f}")
    print(f"  Log-normal: {l_lp:.3f}")
    print(f"  Delta:      {l_lp - g_lp:+.3f}")


if __name__ == "__main__":
    main()
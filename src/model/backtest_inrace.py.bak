# src/model/backtest_inrace.py
"""
Leave-one-year-out backtest for the in-race prediction pipeline.

For each test year:
  1. Train finish model + remaining-time model on all other years
  2. At each checkpoint in the test year, generate predictions
  3. Score: did actual winner have high p_win? Did actual top-10 have high p_top10?

Usage:
    python -m src.model.backtest_inrace --start_year 2019 --end_year 2024
"""

import argparse
import numpy as np
import pandas as pd

from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.metrics import log_loss, roc_auc_score, brier_score_loss

from src.db import connect


# Mirror the feature sets from train_inrace_model.py
SNAPSHOT_FEATURES = [
    "checkpoint_pct",
    "rank_at_checkpoint",
    "rank_pct",
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
]

MUSHER_PRIOR_FEATURES = [
    "best_finish_place",
    "pct_top10",
    "pct_finished",
    "n_finishes",
    "last3_avg_finish_place",
    "w_pct_top10",
    "w_avg_finish_place",
    "years_since_last_entry",
    "is_rookie",
    "career_race_number",
    "trajectory",
    "last_year_improvement",
]

RACE_CONTEXT_FEATURES = [
    "is_northern_route",
]

ALL_FEATURES = SNAPSHOT_FEATURES + MUSHER_PRIOR_FEATURES + RACE_CONTEXT_FEATURES


def _load_all_snapshots(con, year_min: int, year_max: int) -> pd.DataFrame:
    """Load snapshots with musher_strength and races joins."""
    df = con.execute("""
        SELECT
          s.year,
          s.musher_id,
          s.checkpoint_order,
          s.checkpoint_pct,
          s.rank_at_checkpoint,
          s.rank_pct,
          s.dogs_in,
          s.dogs_out,
          s.dogs_dropped,
          s.pct_dogs_remaining,
          s.rest_cum_seconds,
          s.rest_last_seconds,
          s.last_leg_seconds,
          s.leg_delta,
          s.cum_elapsed_seconds,
          s.rank_delta,
          s.gap_to_leader_seconds,
          s.gap_delta,
          s.gap_to_10th_seconds,
          s.pace_last_leg_vs_median,
          s.pace_cum_vs_median,
          s.finished,
          s.finish_time_seconds,
          s.won,
          s.top10,
          -- Musher priors
          ms.best_finish_place,
          ms.pct_top10,
          ms.pct_finished,
          ms.n_finishes,
          ms.last3_avg_finish_place,
          ms.w_pct_top10,
          ms.w_avg_finish_place,
          ms.years_since_last_entry,
          ms.is_rookie,
          ms.career_race_number,
          ms.trajectory,
          ms.last_year_improvement,
          -- Race context
          CASE WHEN r.route_regime IN ('northern') THEN 1 ELSE 0 END AS is_northern_route
        FROM snapshots s
        LEFT JOIN musher_strength ms ON s.year = ms.year AND s.musher_id = ms.musher_id
        LEFT JOIN races r ON s.year = r.year
        WHERE s.year BETWEEN ? AND ?
    """, [year_min, year_max]).df()
    return df


def _coerce_numeric(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def _simulate_predictions(
    finish_model,
    reg_model,
    X: pd.DataFrame,
    cum_elapsed: np.ndarray,
    feature_cols: list[str],
    n_sims: int = 5000,
    sigma_hours: float = 12.0,
    shared_frac: float = 0.6,
    seed: int = 42,
) -> dict:
    """Run Monte Carlo simulation and return p_win, p_top10, exp_place arrays."""
    rng = np.random.default_rng(seed)

    p_finish = finish_model.predict_proba(X[feature_cols])[:, 1]

    pred_log = reg_model.predict(X[feature_cols])
    pred_remaining = np.expm1(pred_log)
    pred_remaining = np.clip(pred_remaining, 0, None)
    pred_remaining_hours = pred_remaining / 3600.0
    cum_hours = cum_elapsed / 3600.0
    pred_finish_hours = cum_hours + pred_remaining_hours

    max_ft = np.nanmax(pred_finish_hours)
    if not np.isfinite(max_ft):
        max_ft = 300.0
    dnf_penalty = max_ft + 500.0

    shared_sigma = sigma_hours * np.sqrt(shared_frac)
    indiv_sigma = sigma_hours * np.sqrt(1.0 - shared_frac)

    shared = rng.normal(0, shared_sigma, (n_sims, 1))
    indiv = rng.normal(0, indiv_sigma, (n_sims, len(X)))

    sim_finish = pred_finish_hours[None, :] + shared + indiv
    u = rng.random((n_sims, len(X)))
    sim_is_finish = u < p_finish[None, :]
    sim_finish = np.where(sim_is_finish, sim_finish, dnf_penalty)

    order = np.argsort(sim_finish, axis=1)
    ranks = np.empty_like(order)
    rows = np.arange(n_sims)[:, None]
    ranks[rows, order] = np.arange(len(X))[None, :]
    place = ranks + 1

    return {
        "p_win": (place == 1).mean(axis=0),
        "p_top10": (place <= 10).mean(axis=0),
        "exp_place": place.mean(axis=0).astype(float),
        "p_finish": p_finish,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start_year", type=int, default=2019)
    ap.add_argument("--end_year", type=int, default=2024)
    ap.add_argument("--checkpoints", type=str, default="5,10,15,20",
                    help="Comma-separated checkpoint orders to evaluate")
    ap.add_argument("--n_sims", type=int, default=5000)
    ap.add_argument("--sigma_hours", type=float, default=12.0)
    ap.add_argument("--shared_noise_frac", type=float, default=0.6)
    args = ap.parse_args()

    eval_cps = [int(x.strip()) for x in args.checkpoints.split(",")]

    con = connect()
    df_all = _load_all_snapshots(con, args.start_year - 1, args.end_year)

    feature_cols = [c for c in ALL_FEATURES if c in df_all.columns]
    df_all = _coerce_numeric(df_all, feature_cols)

    print(f"Features available: {len(feature_cols)}")
    print(f"Total snapshot rows: {len(df_all)}")

    results = []

    for test_year in range(args.start_year, args.end_year + 1):
        d_train = df_all[df_all["year"] != test_year].copy()
        d_test = df_all[df_all["year"] == test_year].copy()

        if d_train.empty or d_test.empty:
            print(f"  SKIP year={test_year} (no data)")
            continue

        # --- Train finish model ---
        d_train_fin = d_train[d_train["finished"].notna()].copy()
        y_train_fin = d_train_fin["finished"].astype(int).to_numpy()
        if len(np.unique(y_train_fin)) < 2:
            print(f"  SKIP year={test_year} (no class variation in finish labels)")
            continue

        finish_model = HistGradientBoostingClassifier(
            max_depth=3, learning_rate=0.05, max_iter=400,
            min_samples_leaf=20, random_state=42,
        )
        finish_model.fit(d_train_fin[feature_cols], y_train_fin)

        # --- Train global remaining-time model ---
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
            print(f"  SKIP year={test_year} (no regression training data)")
            continue

        y_train_log = np.log1p(d_train_reg["remaining_seconds"].to_numpy(dtype=float))
        reg_model = HistGradientBoostingRegressor(
            max_depth=3, learning_rate=0.05, max_iter=800,
            min_samples_leaf=20, random_state=42,
        )
        reg_model.fit(d_train_reg[feature_cols], y_train_log)

        # --- Evaluate at each checkpoint ---
        for cp in eval_cps:
            d_cp = d_test[d_test["checkpoint_order"] == cp].copy()
            if d_cp.empty or len(d_cp) < 5:
                continue

            cum_elapsed = pd.to_numeric(d_cp["cum_elapsed_seconds"], errors="coerce").to_numpy(dtype=float)

            sim = _simulate_predictions(
                finish_model, reg_model, d_cp, cum_elapsed,
                feature_cols, n_sims=args.n_sims,
                sigma_hours=args.sigma_hours,
                shared_frac=args.shared_noise_frac,
            )

            # Score: did the actual winner get high p_win?
            actual_winner_mask = d_cp["won"] == 1
            actual_top10_mask = d_cp["top10"] == 1

            winner_p_win = float(sim["p_win"][actual_winner_mask.to_numpy()].mean()) if actual_winner_mask.any() else np.nan
            winner_rank = float(sim["exp_place"][actual_winner_mask.to_numpy()].mean()) if actual_winner_mask.any() else np.nan

            # How many of actual top-10 were in predicted top-10?
            pred_top10_ids = set(d_cp.iloc[np.argsort(-sim["p_top10"])[:10]]["musher_id"])
            actual_top10_ids = set(d_cp.loc[actual_top10_mask, "musher_id"])
            overlap = len(pred_top10_ids & actual_top10_ids)
            precision_at_10 = overlap / min(10, len(actual_top10_ids)) if actual_top10_ids else np.nan

            results.append({
                "test_year": test_year,
                "checkpoint": cp,
                "n_mushers": len(d_cp),
                "winner_p_win": winner_p_win,
                "winner_exp_place": winner_rank,
                "precision_at_10": precision_at_10,
                "top10_overlap": overlap,
            })

            print(f"  year={test_year} cp={cp:2d}: winner_p_win={winner_p_win:.3f} "
                  f"winner_exp_place={winner_rank:.1f} precision@10={precision_at_10:.2f} "
                  f"({overlap}/10 overlap)")

    # Summary
    res = pd.DataFrame(results)
    if res.empty:
        print("\nNo results generated.")
        return

    print("\n" + "=" * 80)
    print("BACKTEST SUMMARY")
    print("=" * 80)

    for cp in eval_cps:
        cp_res = res[res["checkpoint"] == cp]
        if cp_res.empty:
            continue
        print(f"\nCheckpoint {cp}:")
        print(f"  Avg winner_p_win:    {cp_res['winner_p_win'].mean():.3f}")
        print(f"  Avg winner_exp_place: {cp_res['winner_exp_place'].mean():.1f}")
        print(f"  Avg precision@10:    {cp_res['precision_at_10'].mean():.2f}")

    print("\nOverall averages:")
    print(f"  winner_p_win:    {res['winner_p_win'].mean():.3f}")
    print(f"  winner_exp_place: {res['winner_exp_place'].mean():.1f}")
    print(f"  precision@10:    {res['precision_at_10'].mean():.2f}")


if __name__ == "__main__":
    main()
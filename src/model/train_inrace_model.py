"""
Train the in-race HistGradientBoosting models.

Trains two models from checkpoint-level snapshot data:
  1. Finish classifier: P(musher finishes the race | checkpoint features)
  2. Remaining time regressor: predicted seconds to finish line

Outputs trained models to models/ directory as .joblib files.

Usage:
    python -m src.model.train_inrace_model --train_start 2016 --train_end 2024 --test_year 2025
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import joblib

from sklearn.metrics import (
    log_loss,
    roc_auc_score,
    brier_score_loss,
    mean_absolute_error,
    mean_squared_error,
)
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor

from src.db import connect


# These are the in-race snapshot features (per checkpoint observation)
SNAPSHOT_FEATURES = [
    "checkpoint_pct",
    # "checkpoint_order" removed — conflicts with checkpoint_pct across routes
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

# Pre-race musher strength features to inject (from musher_strength table)
# These give the in-race model historical context about each musher
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

# Year/race-level context features (from races table)
RACE_CONTEXT_FEATURES = [
    "is_northern_route"
]

DEFAULT_FEATURES = SNAPSHOT_FEATURES + MUSHER_PRIOR_FEATURES + RACE_CONTEXT_FEATURES


def _safe_auc(y_true: np.ndarray, p: np.ndarray) -> float:
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(roc_auc_score(y_true, p))


def _load_snapshots(
    con,
    year_min: int,
    year_max: int,
    checkpoint_min: int | None,
    checkpoint_max: int | None,
) -> pd.DataFrame:
    df = con.execute(
        """
        SELECT
          s.year,
          s.musher_id,
          s.checkpoint_order,
          s.checkpoint_pct,
          s.asof_time_utc,
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
          -- Musher strength priors (joined from pre-race features)
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
          -- Year/race context
          CASE WHEN r.route_regime IN ('northern') THEN 1 ELSE 0 END AS is_northern_route,
          r.pct_finishers AS year_pct_finishers
        FROM snapshots s
        LEFT JOIN musher_strength ms
          ON s.year = ms.year AND s.musher_id = ms.musher_id
        LEFT JOIN races r
          ON s.year = r.year
        WHERE s.year BETWEEN ? AND ?
        """,
        [year_min, year_max],
    ).df()

    if checkpoint_min is not None:
        df = df[df["checkpoint_order"] >= checkpoint_min].copy()
    if checkpoint_max is not None:
        df = df[df["checkpoint_order"] <= checkpoint_max].copy()

    return df


def _coerce_numeric(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    for c in cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def _hgb_regressor(random_state: int = 42, **overrides) -> HistGradientBoostingRegressor:
    params = dict(
        max_depth=3,
        learning_rate=0.05,
        max_iter=800,
        min_samples_leaf=20,
        l2_regularization=0.0,
        random_state=random_state,
    )
    params.update(overrides)
    return HistGradientBoostingRegressor(**params)


def _hgb_classifier(random_state: int = 42, **overrides) -> HistGradientBoostingClassifier:
    params = dict(
        max_depth=3,
        learning_rate=0.05,
        max_iter=400,
        min_samples_leaf=20,
        l2_regularization=0.0,
        random_state=random_state,
    )
    params.update(overrides)
    return HistGradientBoostingClassifier(**params)


def _tune_hgb_regressor(X, y, feature_cols, random_state=42):
    """Grid search over key HGB hyperparameters using 5-fold year-aware CV."""
    from sklearn.model_selection import GridSearchCV, GroupKFold

    param_grid = {
        "max_depth": [2, 3, 4, 5],
        "learning_rate": [0.02, 0.05, 0.1],
        "min_samples_leaf": [10, 20, 40],
        "l2_regularization": [0.0, 0.1, 1.0],
        "max_iter": [500, 800, 1200],
    }

    base = HistGradientBoostingRegressor(random_state=random_state)

    # Use year as group for GroupKFold so we don't leak within-year data
    if "year" in X.columns:
        groups = X["year"]
        cv = GroupKFold(n_splits=min(5, groups.nunique()))
    else:
        from sklearn.model_selection import KFold
        cv = KFold(n_splits=5, shuffle=True, random_state=random_state)
        groups = None

    gs = GridSearchCV(
        base, param_grid,
        cv=cv,
        scoring="neg_mean_absolute_error",
        n_jobs=-1,
        verbose=1,
        refit=True,
    )
    gs.fit(X[feature_cols], y, groups=groups)

    print(f"\n  Best regressor params: {gs.best_params_}")
    print(f"  Best CV MAE: {-gs.best_score_ / 3600:.2f} hours")
    return gs.best_estimator_, gs.best_params_


def _tune_hgb_classifier(X, y, feature_cols, random_state=42):
    """Grid search over key HGB hyperparameters for the finish classifier."""
    from sklearn.model_selection import GridSearchCV, GroupKFold

    param_grid = {
        "max_depth": [2, 3, 4],
        "learning_rate": [0.02, 0.05, 0.1],
        "min_samples_leaf": [10, 20, 40],
        "l2_regularization": [0.0, 0.1, 1.0],
        "max_iter": [200, 400, 600],
    }

    base = HistGradientBoostingClassifier(random_state=random_state)

    if "year" in X.columns:
        groups = X["year"]
        cv = GroupKFold(n_splits=min(5, groups.nunique()))
    else:
        from sklearn.model_selection import KFold
        cv = KFold(n_splits=5, shuffle=True, random_state=random_state)
        groups = None

    gs = GridSearchCV(
        base, param_grid,
        cv=cv,
        scoring="neg_brier_score",
        n_jobs=-1,
        verbose=1,
        refit=True,
    )
    gs.fit(X[feature_cols], y, groups=groups)

    print(f"\n  Best classifier params: {gs.best_params_}")
    print(f"  Best CV Brier: {-gs.best_score_:.4f}")
    return gs.best_estimator_, gs.best_params_


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_start", type=int, default=2016)
    ap.add_argument("--train_end", type=int, default=2024)
    ap.add_argument("--test_year", type=int, default=2025)

    ap.add_argument("--checkpoint_min", type=int, default=None)
    ap.add_argument("--checkpoint_max", type=int, default=None)

    ap.add_argument("--model_dir", type=str, default="models")

    # Optional: exclude CP1 artifacts (start can be messy / inconsistent in older years)
    ap.add_argument("--min_checkpoint_for_reg", type=int, default=2)

    # Regression mode
    ap.add_argument(
        "--reg_mode",
        choices=["global", "per_checkpoint"],
        default="global",
        help=(
            "How to train remaining-time regression. "
            "'global' (recommended): single model with checkpoint_pct feature for checkpoint awareness, "
            "pools all data for larger training set. "
            "'per_checkpoint': separate model per checkpoint_order (legacy, small training sets)."
        ),
    )

    # Avoid training tiny per-checkpoint models
    ap.add_argument(
        "--min_train_rows_per_cp",
        type=int,
        default=50,
        help="Min training rows required to fit a checkpoint-specific regressor (per_checkpoint mode).",
    )

    # In per_checkpoint mode, use a global fallback model for checkpoints without a dedicated model
    ap.add_argument(
        "--use_fallback",
        action="store_true",
        help="In per_checkpoint mode, also train a global fallback and use it for missing checkpoints.",
    )

    # Hyperparameter tuning
    ap.add_argument(
        "--tune",
        action="store_true",
        help="Run GridSearchCV to find optimal HGB hyperparameters. Slower but can improve accuracy.",
    )

    args = ap.parse_args()

    con = connect()

    year_min = min(args.train_start, args.test_year)
    year_max = max(args.train_end, args.test_year)

    df = _load_snapshots(
        con,
        year_min=year_min,
        year_max=year_max,
        checkpoint_min=args.checkpoint_min,
        checkpoint_max=args.checkpoint_max,
    )

    print(f"Snapshots rows: {len(df)}")
    print(f"Train years: {args.train_start}..{args.train_end} | Test year: {args.test_year}")

    # --- Basic hygiene / types ---
    df["finished"] = pd.to_numeric(df["finished"], errors="coerce").astype("Int64")
    df["finish_time_seconds"] = pd.to_numeric(df["finish_time_seconds"], errors="coerce")
    df["cum_elapsed_seconds"] = pd.to_numeric(df["cum_elapsed_seconds"], errors="coerce")

    feature_cols = [c for c in DEFAULT_FEATURES if c in df.columns]
    df = _coerce_numeric(df, feature_cols)

    # Guard against inf/-inf sneaking in from bad coercions
    for c in feature_cols:
        if df[c].dtype.kind in {"f", "i"}:
            arr = df[c].to_numpy(dtype=float, copy=False)
            df.loc[~np.isfinite(arr), c] = np.nan

    # Split train/test
    d_train = df[(df["year"] >= args.train_start) & (df["year"] <= args.train_end)].copy()
    d_test = df[df["year"] == args.test_year].copy()

    print(f"Train rows: {len(d_train)} | Test rows: {len(d_test)}")
    print(f"Features used: {len(feature_cols)}")
    print(f"Checkpoint filter: {args.checkpoint_min} .. {args.checkpoint_max}")

    if d_train.empty or d_test.empty:
        raise RuntimeError("Train or test split is empty. Check year ranges / snapshot availability.")

    # =========================
    # 1) FINISH MODEL (binary)
    # =========================
    d_train_finish = d_train[d_train["finished"].notna()].copy()
    d_test_finish = d_test[d_test["finished"].notna()].copy()

    y_train_finish = d_train_finish["finished"].astype(int).to_numpy()
    y_test_finish = d_test_finish["finished"].astype(int).to_numpy()

    X_train_finish = d_train_finish[feature_cols]
    X_test_finish = d_test_finish[feature_cols]

    if len(np.unique(y_train_finish)) < 2:
        raise RuntimeError(
            "Not enough class variation in training finished labels: "
            f"{pd.Series(y_train_finish).value_counts().to_dict()}"
        )

    if args.tune:
        print("\nTuning finish classifier hyperparameters (GridSearchCV)...")
        finish_model, best_clf_params = _tune_hgb_classifier(
            d_train_finish, y_train_finish, feature_cols
        )
    else:
        finish_model = _hgb_classifier(random_state=42)
        finish_model.fit(X_train_finish, y_train_finish)

    p_finish = finish_model.predict_proba(X_test_finish)[:, 1]

    ll = float(log_loss(y_test_finish, p_finish))
    auc = _safe_auc(y_test_finish, p_finish)
    brier = float(brier_score_loss(y_test_finish, p_finish))

    print("\nFinished model (P(finish)) metrics on test:")
    print(f"  logloss: {ll:.4f}")
    print(f"  auc:     {auc:.4f}")
    print(f"  brier:   {brier:.4f}")

    # ==========================================
    # 2) REMAINING TIME MODEL (regression)
    # ==========================================
    # Target: remaining_seconds = finish_time_seconds - cum_elapsed_seconds
    # Train/eval ONLY on finishers with valid finish_time and snapshots that occur BEFORE finish.

    def _remaining_mask(d: pd.DataFrame) -> pd.Series:
        ft = d["finish_time_seconds"]
        ce = d["cum_elapsed_seconds"]
        cp = d["checkpoint_order"]
        return (
            d["finished"].eq(1)
            & ft.notna()
            & ce.notna()
            & cp.notna()
            & (ft > 0)
            & (ce >= 0)
            & (ce <= ft)  # pre-finish snapshots only
            & (cp >= args.min_checkpoint_for_reg)
        )

    def _count_post_finish(d: pd.DataFrame) -> int:
        m = (
            d["finished"].eq(1)
            & d["finish_time_seconds"].notna()
            & d["cum_elapsed_seconds"].notna()
            & (d["cum_elapsed_seconds"] > d["finish_time_seconds"])
        )
        return int(m.sum())

    print("\nSanity: post-finish snapshot rows (should be 0 after filtering):")
    print(f"  train_post_finish_rows: {_count_post_finish(d_train)}")
    print(f"  test_post_finish_rows:  {_count_post_finish(d_test)}")

    d_train_reg_all = d_train[_remaining_mask(d_train)].copy()
    d_test_reg_all = d_test[_remaining_mask(d_test)].copy()

    if d_train_reg_all.empty or d_test_reg_all.empty:
        print("\nRemaining-time model skipped (no usable pre-finish finisher snapshots).")
        reg_model = None
        reg_metrics = None
    else:
        d_train_reg_all["remaining_seconds"] = (
            d_train_reg_all["finish_time_seconds"] - d_train_reg_all["cum_elapsed_seconds"]
        )
        d_test_reg_all["remaining_seconds"] = (
            d_test_reg_all["finish_time_seconds"] - d_test_reg_all["cum_elapsed_seconds"]
        )

        d_train_reg_all = d_train_reg_all[d_train_reg_all["remaining_seconds"] >= 0].copy()
        d_test_reg_all = d_test_reg_all[d_test_reg_all["remaining_seconds"] >= 0].copy()

        if d_train_reg_all.empty or d_test_reg_all.empty:
            print("\nRemaining-time model skipped (no usable remaining_seconds after final filtering).")
            reg_model = None
            reg_metrics = None
        else:
            # Common test arrays for implied finish-time
            y_test_finish_time_all = d_test_reg_all["finish_time_seconds"].to_numpy(dtype=float)
            cum_test_all = d_test_reg_all["cum_elapsed_seconds"].to_numpy(dtype=float)

            if args.reg_mode == "global":
                # -------------------------
                # GLOBAL regression
                # -------------------------
                y_train_log = np.log1p(d_train_reg_all["remaining_seconds"].to_numpy(dtype=float))
                y_test_remaining = d_test_reg_all["remaining_seconds"].to_numpy(dtype=float)

                X_train_reg = d_train_reg_all[feature_cols]
                X_test_reg = d_test_reg_all[feature_cols]

                if args.tune:
                    print("\nTuning remaining-time regressor hyperparameters (GridSearchCV)...")
                    reg_model, best_reg_params = _tune_hgb_regressor(
                        d_train_reg_all, y_train_log, feature_cols
                    )
                else:
                    reg_model = _hgb_regressor(random_state=42)
                    reg_model.fit(X_train_reg, y_train_log)

                pred_log = reg_model.predict(X_test_reg)
                pred_remaining = np.expm1(pred_log)
                pred_remaining = np.clip(pred_remaining, 0, None)

                mae_sec = float(mean_absolute_error(y_test_remaining, pred_remaining))
                rmse_sec = float(np.sqrt(mean_squared_error(y_test_remaining, pred_remaining)))

                print("\nRemaining-time model metrics on test (finishers only, pre-finish snapshots) [GLOBAL]:")
                print(f"  MAE:  {mae_sec/3600.0:.2f} hours")
                print(f"  RMSE: {rmse_sec/3600.0:.2f} hours")

                pred_finish_time = cum_test_all + pred_remaining
                mae_ft = float(mean_absolute_error(y_test_finish_time_all, pred_finish_time))
                rmse_ft = float(np.sqrt(mean_squared_error(y_test_finish_time_all, pred_finish_time)))

                print("\nImplied finish-time metrics on test (finishers only) [GLOBAL]:")
                print(f"  MAE:  {mae_ft/3600.0:.2f} hours")
                print(f"  RMSE: {rmse_ft/3600.0:.2f} hours")

                reg_metrics = {
                    "mode": "global",
                    "remaining_mae_seconds": mae_sec,
                    "remaining_rmse_seconds": rmse_sec,
                    "implied_finish_time_mae_seconds": mae_ft,
                    "implied_finish_time_rmse_seconds": rmse_ft,
                    "n_train_rows": int(len(d_train_reg_all)),
                    "n_test_rows": int(len(d_test_reg_all)),
                    "min_checkpoint_for_reg": int(args.min_checkpoint_for_reg),
                    "target": "log1p(remaining_seconds)",
                    "notes": "remaining_seconds = finish_time_seconds - cum_elapsed_seconds (finishers only; pre-finish snapshots only)",
                }

            else:
                # -------------------------
                # PER-CHECKPOINT regression (+ optional global fallback)
                # -------------------------
                reg_models: dict[int, HistGradientBoostingRegressor] = {}
                skipped_cps: list[int] = []

                # Fit fallback if requested (recommended)
                fallback_model = None
                if args.use_fallback:
                    y_train_all_log = np.log1p(d_train_reg_all["remaining_seconds"].to_numpy(dtype=float))
                    X_train_all = d_train_reg_all[feature_cols]
                    fallback_model = _hgb_regressor(random_state=42)
                    fallback_model.fit(X_train_all, y_train_all_log)

                train_cps = sorted(d_train_reg_all["checkpoint_order"].dropna().astype(int).unique().tolist())
                for cp in train_cps:
                    d_cp = d_train_reg_all[d_train_reg_all["checkpoint_order"].astype(int) == cp].copy()
                    if len(d_cp) < args.min_train_rows_per_cp:
                        skipped_cps.append(int(cp))
                        continue

                    y_cp_log = np.log1p(d_cp["remaining_seconds"].to_numpy(dtype=float))
                    X_cp = d_cp[feature_cols]

                    m = _hgb_regressor(random_state=42)
                    m.fit(X_cp, y_cp_log)
                    reg_models[int(cp)] = m

                if len(reg_models) == 0:
                    print("\nRemaining-time model skipped (no checkpoints met min_train_rows_per_cp).")
                    reg_model = None
                    reg_metrics = None
                else:
                    test_cps = sorted(d_test_reg_all["checkpoint_order"].dropna().astype(int).unique().tolist())
                    missing_cps = sorted(set(test_cps) - set(reg_models.keys()))

                    preds = []
                    missing_model_rows = 0
                    fallback_rows = 0

                    for cp in test_cps:
                        d_cp_test = d_test_reg_all[d_test_reg_all["checkpoint_order"].astype(int) == cp].copy()
                        model_cp = reg_models.get(int(cp))

                        X_cp_test = d_cp_test[feature_cols]

                        if model_cp is None:
                            if fallback_model is None:
                                missing_model_rows += len(d_cp_test)
                                continue
                            # fallback prediction
                            pred_log = fallback_model.predict(X_cp_test)
                            fallback_rows += len(d_cp_test)
                        else:
                            pred_log = model_cp.predict(X_cp_test)

                        pred_remaining = np.expm1(pred_log)
                        pred_remaining = np.clip(pred_remaining, 0, None)

                        d_cp_test = d_cp_test.assign(pred_remaining_seconds=pred_remaining)
                        preds.append(d_cp_test)

                    if len(preds) == 0:
                        print("\nRemaining-time model skipped (no test rows had an available regressor).")
                        reg_model = None
                        reg_metrics = None
                    else:
                        d_pred = pd.concat(preds, axis=0, ignore_index=True)

                        y_true_rem = d_pred["remaining_seconds"].to_numpy(dtype=float)
                        y_pred_rem = d_pred["pred_remaining_seconds"].to_numpy(dtype=float)

                        mae_sec = float(mean_absolute_error(y_true_rem, y_pred_rem))
                        rmse_sec = float(np.sqrt(mean_squared_error(y_true_rem, y_pred_rem)))

                        mode_label = "PER_CHECKPOINT+FALLBACK" if fallback_model is not None else "PER_CHECKPOINT"
                        print(f"\nRemaining-time model metrics on test (finishers only, pre-finish snapshots) [{mode_label}]:")
                        if fallback_model is None:
                            print(
                                f"  coverage_test_rows: {len(d_pred)} / {len(d_test_reg_all)} "
                                f"(missing_model_rows={missing_model_rows})"
                            )
                        else:
                            print(
                                f"  coverage_test_rows: {len(d_pred)} / {len(d_test_reg_all)} "
                                f"(fallback_rows={fallback_rows})"
                            )
                        print(f"  n_checkpoint_models: {len(reg_models)}")
                        if missing_cps:
                            msg = "used fallback" if fallback_model is not None else "no model (dropped)"
                            print(f"  missing_checkpoint_models ({msg}): {missing_cps}")
                        if skipped_cps:
                            print(f"  skipped_checkpoint_models_train_too_small: {sorted(set(skipped_cps))}")
                        print(f"  MAE:  {mae_sec/3600.0:.2f} hours")
                        print(f"  RMSE: {rmse_sec/3600.0:.2f} hours")

                        # Implied finish-time on covered rows
                        y_true_ft = d_pred["finish_time_seconds"].to_numpy(dtype=float)
                        cum_test = d_pred["cum_elapsed_seconds"].to_numpy(dtype=float)
                        y_pred_ft = cum_test + y_pred_rem

                        mae_ft = float(mean_absolute_error(y_true_ft, y_pred_ft))
                        rmse_ft = float(np.sqrt(mean_squared_error(y_true_ft, y_pred_ft)))

                        print(f"\nImplied finish-time metrics on test (finishers only) [{mode_label}]:")
                        print(f"  MAE:  {mae_ft/3600.0:.2f} hours")
                        print(f"  RMSE: {rmse_ft/3600.0:.2f} hours")

                        # Per-checkpoint breakdown (includes fallback-predicted CPs if enabled)
                        per_cp_rows = []
                        for cp, grp in d_pred.groupby(d_pred["checkpoint_order"].astype(int)):
                            ytr = grp["remaining_seconds"].to_numpy(dtype=float)
                            ypr = grp["pred_remaining_seconds"].to_numpy(dtype=float)
                            per_cp_rows.append(
                                {
                                    "checkpoint_order": int(cp),
                                    "n": int(len(grp)),
                                    "used_fallback": bool((fallback_model is not None) and (int(cp) not in reg_models)),
                                    "mae_hours": float(mean_absolute_error(ytr, ypr) / 3600.0),
                                    "rmse_hours": float(np.sqrt(mean_squared_error(ytr, ypr)) / 3600.0),
                                }
                            )
                        per_cp_df = pd.DataFrame(per_cp_rows).sort_values("checkpoint_order")

                        print("\nPer-checkpoint remaining-time error (hours):")
                        print(per_cp_df.to_string(index=False))

                        # Save reg_model in an explicit structure to avoid ambiguity later
                        reg_model = {
                            "mode": "per_checkpoint_with_fallback" if fallback_model is not None else "per_checkpoint",
                            "per_checkpoint": reg_models,
                            "fallback": fallback_model,  # can be None
                        }

                        reg_metrics = {
                            "mode": reg_model["mode"],
                            "remaining_mae_seconds": mae_sec,
                            "remaining_rmse_seconds": rmse_sec,
                            "implied_finish_time_mae_seconds": mae_ft,
                            "implied_finish_time_rmse_seconds": rmse_ft,
                            "n_train_rows": int(len(d_train_reg_all)),
                            "n_test_rows_total": int(len(d_test_reg_all)),
                            "n_test_rows_covered": int(len(d_pred)),
                            "missing_model_rows": int(missing_model_rows),
                            "fallback_rows": int(fallback_rows),
                            "missing_checkpoint_models": missing_cps,
                            "skipped_checkpoint_models_train_too_small": sorted(set(skipped_cps)),
                            "min_checkpoint_for_reg": int(args.min_checkpoint_for_reg),
                            "min_train_rows_per_cp": int(args.min_train_rows_per_cp),
                            "n_checkpoint_models": int(len(reg_models)),
                            "per_checkpoint_metrics": per_cp_df.to_dict(orient="records"),
                            "target": "log1p(remaining_seconds)",
                            "notes": "remaining_seconds = finish_time_seconds - cum_elapsed_seconds (finishers only; pre-finish snapshots only)",
                        }

    # =========================
    # Save artifacts
    # =========================
    model_dir = Path(args.model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)

    finish_path = model_dir / "inrace_finish_model.joblib"
    reg_path = model_dir / "inrace_remaining_time_model.joblib"
    meta_path = model_dir / "inrace_metadata.json"

    joblib.dump(finish_model, finish_path)
    joblib.dump(reg_model, reg_path)  # can be None, model, or dict bundle

    metadata = {
        "train_start": args.train_start,
        "train_end": args.train_end,
        "test_year": args.test_year,
        "checkpoint_min": args.checkpoint_min,
        "checkpoint_max": args.checkpoint_max,
        "min_checkpoint_for_reg": args.min_checkpoint_for_reg,
        "reg_mode": args.reg_mode,
        "min_train_rows_per_cp": args.min_train_rows_per_cp,
        "use_fallback": bool(args.use_fallback),
        "tuned": bool(args.tune),
        "feature_cols": feature_cols,
        "finish_metrics": {"logloss": ll, "auc": auc, "brier": brier},
        "remaining_time_metrics": reg_metrics,
        "notes": {
            "finish_model": "Binary classifier predicting P(finish).",
            "remaining_time_model": (
                "Target is log1p(remaining_seconds). "
                "If reg_mode=per_checkpoint, saved reg_model is a bundle dict with keys: "
                "mode, per_checkpoint (dict[int, model]), fallback (model or None)."
            ),
            "sanity_filter": "Only train/eval rows where cum_elapsed_seconds <= finish_time_seconds.",
            "implied_finish_time": "cum_elapsed_seconds + predicted_remaining_seconds.",
        },
    }
    meta_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print("\nSaved:")
    print(f"  {finish_path}")
    print(f"  {reg_path}")
    print(f"  {meta_path}")


if __name__ == "__main__":
    main()
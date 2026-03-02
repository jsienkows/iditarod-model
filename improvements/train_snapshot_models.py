"""
train_snapshot_models.py
========================
Train snapshot-only (no musher priors) models for prior decay blending.

These are global HistGBT models trained on ONLY the in-race checkpoint
features + race context. They complement the full models (which include
musher priors) and are blended at prediction time with a checkpoint-
dependent weight that fades priors as in-race data accumulates.

Saves:
    models/inrace_finish_model_snapshot.joblib
    models/inrace_remaining_time_model_snapshot.joblib

Usage:
    python improvements/train_snapshot_models.py
    python improvements/train_snapshot_models.py --train_start 2016 --train_end 2025
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
import json
import numpy as np
import pandas as pd
import joblib

from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor

from src.db import connect


# In-race snapshot features ONLY (no musher priors)
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

RACE_CONTEXT_FEATURES = [
    "is_northern_route",
]

SNAPSHOT_ONLY_COLS = SNAPSHOT_FEATURES + RACE_CONTEXT_FEATURES


def load_data(con, year_min, year_max):
    df = con.execute("""
        SELECT
          s.year, s.musher_id, s.checkpoint_order, s.checkpoint_pct,
          s.rank_at_checkpoint, s.rank_pct,
          s.dogs_out, s.dogs_dropped, s.pct_dogs_remaining,
          s.rest_cum_seconds, s.rest_last_seconds,
          s.last_leg_seconds, s.leg_delta,
          s.cum_elapsed_seconds, s.rank_delta,
          s.gap_to_leader_seconds, s.gap_delta,
          s.gap_to_10th_seconds,
          s.pace_last_leg_vs_median, s.pace_cum_vs_median,
          s.finished, s.finish_time_seconds,
          CASE WHEN r.route_regime IN ('northern') THEN 1 ELSE 0 END AS is_northern_route
        FROM snapshots s
        LEFT JOIN races r ON s.year = r.year
        WHERE s.year BETWEEN ? AND ?
    """, [year_min, year_max]).df()

    # Coerce numeric
    for c in SNAPSHOT_ONLY_COLS:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_start", type=int, default=2016)
    ap.add_argument("--train_end", type=int, default=2025)
    ap.add_argument("--model_dir", type=str, default="models")
    args = ap.parse_args()

    con = connect()
    df = load_data(con, args.train_start, args.train_end)
    print(f"Loaded {len(df)} snapshot rows ({args.train_start}-{args.train_end})")

    feature_cols = [c for c in SNAPSHOT_ONLY_COLS if c in df.columns]
    print(f"Features: {len(feature_cols)} (snapshot + race context, NO priors)")

    # ---- Finish classifier ----
    d_finish = df[df["finished"].notna()].copy()
    y_finish = d_finish["finished"].astype(int).values
    X_finish = d_finish[feature_cols]

    finish_model = HistGradientBoostingClassifier(
        max_depth=3, learning_rate=0.05, max_iter=400,
        min_samples_leaf=20, random_state=42,
    )
    finish_model.fit(X_finish, y_finish)

    p_hat = finish_model.predict_proba(X_finish)[:, 1]
    from sklearn.metrics import roc_auc_score, brier_score_loss
    auc = roc_auc_score(y_finish, p_hat)
    brier = brier_score_loss(y_finish, p_hat)
    print(f"\nFinish classifier (train fit):")
    print(f"  AUC={auc:.3f}, Brier={brier:.4f}, n={len(y_finish)}")

    # ---- Remaining time regressor (global, log1p target) ----
    d_reg = df[
        (df["finished"] == 1)
        & df["finish_time_seconds"].notna()
        & df["cum_elapsed_seconds"].notna()
    ].copy()
    d_reg["remaining_seconds"] = d_reg["finish_time_seconds"] - d_reg["cum_elapsed_seconds"]
    d_reg = d_reg[d_reg["remaining_seconds"] >= 0].copy()

    y_reg = np.log1p(d_reg["remaining_seconds"].values.astype(float))
    X_reg = d_reg[feature_cols]

    reg_model = HistGradientBoostingRegressor(
        max_depth=3, learning_rate=0.05, max_iter=800,
        min_samples_leaf=20, random_state=42,
    )
    reg_model.fit(X_reg, y_reg)

    pred_log = reg_model.predict(X_reg)
    pred_sec = np.expm1(pred_log)
    residuals = d_reg["remaining_seconds"].values - pred_sec
    mae_h = np.mean(np.abs(residuals)) / 3600
    rmse_h = np.sqrt(np.mean(residuals**2)) / 3600
    print(f"\nRemaining time regressor (train fit):")
    print(f"  MAE={mae_h:.2f}h, RMSE={rmse_h:.2f}h, n={len(y_reg)}")

    # ---- Save ----
    model_dir = Path(args.model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)

    finish_path = model_dir / "inrace_finish_model_snapshot.joblib"
    reg_path = model_dir / "inrace_remaining_time_model_snapshot.joblib"

    joblib.dump(finish_model, finish_path)
    joblib.dump(reg_model, reg_path)

    # Update metadata with snapshot_only_cols
    meta_path = model_dir / "inrace_metadata.json"
    if meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        meta["snapshot_only_cols"] = feature_cols
        meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
        print(f"\nUpdated {meta_path} with snapshot_only_cols")

    print(f"\nSaved:")
    print(f"  {finish_path}")
    print(f"  {reg_path}")
    print(f"\nNext: run predict_inrace.py as usual — it will auto-detect")
    print(f"snapshot models and blend with prior decay weight.")


if __name__ == "__main__":
    main()
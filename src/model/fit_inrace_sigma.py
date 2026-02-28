"""
Calibrate prediction uncertainty (sigma) from in-race model residuals.

Fits a MAD-based sigma estimate from leave-one-year-out residuals,
used by the Monte Carlo simulator to convert point predictions into
probability distributions.

Usage:
    python -m src.model.fit_inrace_sigma --train_start 2016 --train_end 2024
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import joblib

from src.db import connect


def _remaining_mask(d: pd.DataFrame, min_checkpoint_for_reg: int) -> pd.Series:
    """
    Mask for *finisher* snapshots where remaining time is defined and non-negative,
    and checkpoint_order is at/after the minimum checkpoint where your remaining-time reg is valid.
    """
    ft = pd.to_numeric(d.get("finish_time_seconds"), errors="coerce")
    ce = pd.to_numeric(d.get("cum_elapsed_seconds"), errors="coerce")
    cp = pd.to_numeric(d.get("checkpoint_order"), errors="coerce")
    finished = pd.to_numeric(d.get("finished"), errors="coerce")

    return (
        (finished == 1)
        & ft.notna()
        & ce.notna()
        & cp.notna()
        & (ft > 0)
        & (ce >= 0)
        & (ce <= ft)
        & (cp >= min_checkpoint_for_reg)
    )


def _coerce_numeric_cols(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
            # normalize non-finite -> NaN
            arr = df[c].to_numpy(dtype=float, copy=False)
            df.loc[~np.isfinite(arr), c] = np.nan
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_start", type=int, default=2016)
    ap.add_argument("--train_end", type=int, default=2024)
    ap.add_argument("--min_checkpoint_for_reg", type=int, default=2)
    ap.add_argument("--model_dir", type=str, default="models")

    # How to compute sigma from residuals
    ap.add_argument(
        "--sigma_method",
        choices=["std", "mad"],
        default="std",
        help="std = standard deviation of residuals; mad = robust MAD->sigma (1.4826 * median(|resid - median|)).",
    )

    # Optional caps to avoid insane noise
    ap.add_argument("--sigma_floor_hours", type=float, default=2.0)
    ap.add_argument("--sigma_cap_hours", type=float, default=96.0)

    # Also write a top-level key for convenience / backward-compat
    ap.add_argument(
        "--write_top_level",
        action="store_true",
        help="If set, also writes meta['sigma_hours_by_cp'] in addition to remaining_time_metrics['sigma_hours_by_cp'].",
    )

    args = ap.parse_args()

    model_dir = Path(args.model_dir)
    meta_path = model_dir / "inrace_metadata.json"
    reg_path = model_dir / "inrace_remaining_time_model.joblib"

    if not meta_path.exists():
        raise FileNotFoundError(f"Missing metadata: {meta_path}")
    if not reg_path.exists():
        raise FileNotFoundError(f"Missing remaining-time model: {reg_path}")

    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    feature_cols = meta.get("feature_cols", None)
    if not feature_cols:
        raise RuntimeError("metadata missing 'feature_cols'")

    # Load regression model bundle (global model or dict[int, model])
    reg_model = joblib.load(reg_path)

    con = connect()

    # Re-query snapshots including:
    # - checkpoint_order (ALWAYS)
    # - finish_time_seconds / cum_elapsed_seconds / finished (ALWAYS)
    # - the feature columns used by the remaining-time model
    #
    # This avoids subtle failures if checkpoint_order isn't part of feature_cols.
    base_cols = ["year", "musher_id", "checkpoint_order", "finished", "finish_time_seconds", "cum_elapsed_seconds"]

    # Keep ordering stable, avoid duplicates
    all_cols = []
    for c in base_cols + list(feature_cols):
        if c not in all_cols:
            all_cols.append(c)

    cols_sql = ",\n          ".join(f"s.{c}" if c not in ("best_finish_place", "pct_top10", "pct_finished", "n_finishes",
        "last3_avg_finish_place", "w_pct_top10", "w_avg_finish_place",
        "years_since_last_entry", "is_rookie") else f"ms.{c}" for c in all_cols
        if c != "is_northern_route")

    # Build the SELECT with proper table prefixes
    select_parts = []
    ms_cols = {"best_finish_place", "pct_top10", "pct_finished", "n_finishes",
               "last3_avg_finish_place", "w_pct_top10", "w_avg_finish_place",
               "years_since_last_entry", "is_rookie"}
    for c in all_cols:
        if c == "is_northern_route":
            select_parts.append("CASE WHEN r.route_regime IN ('northern') THEN 1 ELSE 0 END AS is_northern_route")
        elif c in ms_cols:
            select_parts.append(f"ms.{c}")
        else:
            select_parts.append(f"s.{c}")

    cols_sql = ",\n          ".join(select_parts)

    df = con.execute(
        f"""
        SELECT
          {cols_sql}
        FROM snapshots s
        LEFT JOIN musher_strength ms ON s.year = ms.year AND s.musher_id = ms.musher_id
        LEFT JOIN races r ON s.year = r.year
        WHERE s.year BETWEEN ? AND ?
        """,
        [args.train_start, args.train_end],
    ).df()

    if df.empty:
        raise RuntimeError("No snapshots loaded for sigma fitting.")

    # Filter to valid pre-finish finisher snapshots
    m = _remaining_mask(df, args.min_checkpoint_for_reg)
    df = df[m].copy()
    if df.empty:
        raise RuntimeError("No usable rows after finisher/remaining mask filtering.")

    # Coerce key columns
    df["checkpoint_order"] = pd.to_numeric(df["checkpoint_order"], errors="coerce").astype(int)
    df["finish_time_seconds"] = pd.to_numeric(df["finish_time_seconds"], errors="coerce")
    df["cum_elapsed_seconds"] = pd.to_numeric(df["cum_elapsed_seconds"], errors="coerce")

    # Remaining seconds (truth)
    df["remaining_seconds"] = df["finish_time_seconds"] - df["cum_elapsed_seconds"]
    df = df[df["remaining_seconds"].notna() & (df["remaining_seconds"] >= 0)].copy()
    if df.empty:
        raise RuntimeError("No usable rows after remaining_seconds filtering.")

    # Ensure feature columns are numeric + finite
    df = _coerce_numeric_cols(df, list(feature_cols))

    # Predict remaining_seconds for each row using the saved model(s)
    preds = []
    missing_model_rows = 0

    if isinstance(reg_model, dict):
        # per-checkpoint models
        cps = sorted(df["checkpoint_order"].unique().tolist())
        for cp in cps:
            dcp = df[df["checkpoint_order"] == int(cp)].copy()
            mcp = reg_model.get(int(cp))
            if mcp is None:
                missing_model_rows += len(dcp)
                continue

            X = dcp[feature_cols]
            pred_log = mcp.predict(X)
            pred_rem = np.expm1(pred_log)
            pred_rem = np.clip(pred_rem, 0, None)
            dcp["pred_remaining_seconds"] = pred_rem
            preds.append(dcp)
    else:
        # global model
        X = df[feature_cols]
        pred_log = reg_model.predict(X)
        pred_rem = np.expm1(pred_log)
        pred_rem = np.clip(pred_rem, 0, None)
        df["pred_remaining_seconds"] = pred_rem
        preds.append(df)

    if not preds:
        raise RuntimeError("No predictions generated (no checkpoint models available for any rows).")

    d_pred = pd.concat(preds, ignore_index=True)

    # Residuals in seconds
    resid = (
        d_pred["remaining_seconds"].to_numpy(dtype=float)
        - d_pred["pred_remaining_seconds"].to_numpy(dtype=float)
    )
    d_pred["resid_seconds"] = resid

    # Compute sigma per checkpoint
    rows = []
    for cp, grp in d_pred.groupby("checkpoint_order"):
        r = grp["resid_seconds"].to_numpy(dtype=float)
        r = r[np.isfinite(r)]
        if len(r) < 5:
            continue

        if args.sigma_method == "std":
            sigma_sec = float(np.std(r, ddof=1))
        else:
            med = float(np.median(r))
            mad = float(np.median(np.abs(r - med)))
            sigma_sec = float(1.4826 * mad)

        sigma_hours = sigma_sec / 3600.0
        sigma_hours = float(np.clip(sigma_hours, args.sigma_floor_hours, args.sigma_cap_hours))

        rows.append(
            {
                "checkpoint_order": int(cp),
                "n": int(len(grp)),
                "sigma_hours": sigma_hours,
            }
        )

    sigma_df = pd.DataFrame(rows).sort_values("checkpoint_order")
    if sigma_df.empty:
        raise RuntimeError("sigma_df is empty; not enough data per checkpoint.")

    # NOTE: JSON keys will be strings after dump, which predict_inrace handles (tries str and int keys).
    sigma_map = {int(r["checkpoint_order"]): float(r["sigma_hours"]) for r in sigma_df.to_dict(orient="records")}

    # Write back to metadata (canonical location)
    meta.setdefault("remaining_time_metrics", {})
    meta["remaining_time_metrics"]["sigma_method"] = args.sigma_method
    meta["remaining_time_metrics"]["sigma_floor_hours"] = args.sigma_floor_hours
    meta["remaining_time_metrics"]["sigma_cap_hours"] = args.sigma_cap_hours
    meta["remaining_time_metrics"]["sigma_hours_by_cp"] = sigma_map
    meta["remaining_time_metrics"]["sigma_rows_covered"] = int(len(d_pred))
    meta["remaining_time_metrics"]["sigma_missing_model_rows"] = int(missing_model_rows)
    meta["remaining_time_metrics"]["sigma_train_year_start"] = int(args.train_start)
    meta["remaining_time_metrics"]["sigma_train_year_end"] = int(args.train_end)
    meta["remaining_time_metrics"]["sigma_min_checkpoint_for_reg"] = int(args.min_checkpoint_for_reg)

    # Optional top-level convenience key (predict_inrace supports either)
    if args.write_top_level:
        meta["sigma_hours_by_cp"] = sigma_map

    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print("Wrote sigma_hours_by_cp into:", meta_path)
    if args.write_top_level:
        print("Also wrote top-level sigma_hours_by_cp into metadata.")
    print("Sigma by checkpoint (hours):")
    print(sigma_df.to_string(index=False))


if __name__ == "__main__":
    main()

"""
3_bootstrap_ci.py
=================
Add block bootstrap confidence intervals to backtest reporting.

WHY:
  With n=11 backtest years, point estimates (P@5=0.545, Spearman=0.668)
  have wide uncertainty. Without CIs, you can't judge whether differences
  between model versions are meaningful.

  Block bootstrap resamples at the YEAR level (not individual predictions),
  preserving within-year correlation — the 5 picks in a given year aren't
  independent, so we treat each year as one unit.

USAGE:
  This provides two things:
    1. A standalone function you can call from any backtest script.
    2. A runnable script that reads your existing LOOCV predictions
       and adds CIs to the metrics.

  To integrate into backtest_prerace_baseline.py:
    from improvements.bootstrap_ci import compute_bootstrap_cis, format_ci
    # ... after collecting per-year metrics:
    cis = compute_bootstrap_cis(year_metrics_df)
    print(format_ci(cis))

Usage:
    python improvements/3_bootstrap_ci.py              # runs on pre-race backtest
    python improvements/3_bootstrap_ci.py --n_boot 10000
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import roc_auc_score

from src.db import connect


# ====================================================================
# CORE BOOTSTRAP FUNCTIONS (import these into your backtest scripts)
# ====================================================================

def block_bootstrap_ci(
    year_values: dict[int, float],
    n_boot: int = 10000,
    ci_level: float = 0.95,
    seed: int = 42,
    statistic: str = "mean",
) -> dict:
    """
    Block bootstrap CI treating each year as one observation.

    Parameters
    ----------
    year_values : dict
        {year: metric_value} for each backtest year.
        E.g. {2015: 0.6, 2016: 0.4, ...}
    n_boot : int
        Number of bootstrap resamples.
    ci_level : float
        Confidence level (default 0.95 for 95% CI).
    seed : int
        Random seed.
    statistic : str
        'mean' or 'median'

    Returns
    -------
    dict with keys: point, ci_low, ci_high, se, n_years
    """
    rng = np.random.default_rng(seed)
    years = list(year_values.keys())
    values = np.array([year_values[y] for y in years])

    # Remove NaN years
    valid = ~np.isnan(values)
    values = values[valid]
    n = len(values)

    if n < 3:
        return {"point": np.nanmean(values), "ci_low": np.nan, "ci_high": np.nan,
                "se": np.nan, "n_years": n}

    stat_fn = np.mean if statistic == "mean" else np.median

    # Bootstrap: resample years with replacement
    boot_stats = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        boot_stats[i] = stat_fn(values[idx])

    alpha = 1 - ci_level
    ci_low = np.percentile(boot_stats, 100 * alpha / 2)
    ci_high = np.percentile(boot_stats, 100 * (1 - alpha / 2))

    return {
        "point": float(stat_fn(values)),
        "ci_low": float(ci_low),
        "ci_high": float(ci_high),
        "se": float(np.std(boot_stats)),
        "n_years": n,
    }


def compute_bootstrap_cis(
    year_metrics: pd.DataFrame,
    metric_cols: list[str] = None,
    n_boot: int = 10000,
    ci_level: float = 0.95,
) -> dict[str, dict]:
    """
    Compute bootstrap CIs for multiple metrics from a per-year results DataFrame.

    Parameters
    ----------
    year_metrics : DataFrame
        Must have a 'year' or 'test_year' column and metric columns.
    metric_cols : list of str
        Which columns to compute CIs for. If None, auto-detect numeric cols.
    n_boot : int
    ci_level : float

    Returns
    -------
    dict of {metric_name: {point, ci_low, ci_high, se, n_years}}
    """
    year_col = "year" if "year" in year_metrics.columns else "test_year"

    if metric_cols is None:
        metric_cols = [c for c in year_metrics.columns
                       if c != year_col and year_metrics[c].dtype in [np.float64, np.int64, float]]

    results = {}
    for col in metric_cols:
        vals = year_metrics.set_index(year_col)[col].to_dict()
        results[col] = block_bootstrap_ci(vals, n_boot=n_boot, ci_level=ci_level)

    return results


def format_ci(cis: dict[str, dict], ci_level: float = 0.95) -> str:
    """Pretty-print bootstrap CIs."""
    pct = int(ci_level * 100)
    lines = [f"{'Metric':>25} | {'Point':>8} | {pct}% CI             | {'SE':>6} | n"]
    lines.append("-" * 72)

    for metric, vals in cis.items():
        p = vals["point"]
        lo = vals["ci_low"]
        hi = vals["ci_high"]
        se = vals["se"]
        n = vals["n_years"]

        if np.isnan(p):
            lines.append(f"{metric:>25} | {'N/A':>8} |")
            continue

        # Auto-detect if this looks like a percentage metric
        is_pct = "winner_in" in metric
        if is_pct:
            lines.append(f"{metric:>25} | {p:>7.1%} | [{lo:>6.1%}, {hi:>6.1%}] | {se:>6.3f} | {n}")
        else:
            lines.append(f"{metric:>25} | {p:>8.3f} | [{lo:>7.3f}, {hi:>7.3f}] | {se:>6.3f} | {n}")

    return "\n".join(lines)


# ====================================================================
# STANDALONE RUNNER: Pre-race backtest with CIs
# ====================================================================

WIN_FEATURES = [
    "w_avg_finish_place", "w_pct_top10", "w_pct_finished", "pct_top5",
    "w_avg_time_behind_winner_seconds", "n_finishes", "w_n_entries",
    "years_since_last_entry", "is_rookie",
]
RANK_FEATURES = WIN_FEATURES + ["last_year_finish_place"]


def _make_calibrated_model(class_weight="balanced", cv=5):
    base = Pipeline([
        ("imputer", SimpleImputer(strategy="constant", fill_value=0, keep_empty_features=True)),
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=4000, class_weight=class_weight)),
    ])
    try:
        return CalibratedClassifierCV(estimator=base, method="sigmoid", cv=cv)
    except TypeError:
        return CalibratedClassifierCV(base_estimator=base, method="sigmoid", cv=cv)


def _model_cols(features):
    return features + [f"{c}_missing" for c in features]


def derive_win_probability(p_top5):
    raw = np.power(np.clip(p_top5, 1e-6, 1.0), 3.0)
    return raw / raw.sum()


def run_prerace_backtest_with_cis(con, train_start, train_end, start_year, n_boot):
    """Full pre-race backtest collecting per-year metrics for CI computation."""
    all_features = list(dict.fromkeys(RANK_FEATURES + WIN_FEATURES))
    win_mcols = _model_cols(WIN_FEATURES)
    rank_mcols = _model_cols(RANK_FEATURES)

    # Load data
    ms = con.execute(
        f"SELECT year, musher_id, {', '.join(all_features)} "
        f"FROM musher_strength WHERE year BETWEEN ? AND ?",
        [train_start, train_end],
    ).df()
    hr = con.execute(
        "SELECT year, musher_id, finish_place, status "
        "FROM historical_results WHERE year BETWEEN ? AND ?",
        [train_start, train_end],
    ).df()
    df = ms.merge(hr, on=["year", "musher_id"], how="inner")

    df["won"] = (df["finish_place"] == 1).astype("Int64")
    df["top5"] = (df["finish_place"].notna() & (df["finish_place"] <= 5)).astype("Int64")
    df["top10"] = (df["finish_place"].notna() & (df["finish_place"] <= 10)).astype("Int64")
    df["finished"] = (df["finish_place"].notna()).astype("Int64")

    df[all_features] = df[all_features].apply(pd.to_numeric, errors="coerce")
    for f in all_features:
        df[f"{f}_missing"] = df[f].isna().astype("int64")

    year_results = []

    for test_yr in range(start_year, train_end + 1):
        tr = df[df["year"] < test_yr].copy()
        te = df[df["year"] == test_yr].copy()

        if tr.empty or te.empty or tr["top5"].sum() < 3:
            continue

        # Train 4 sub-models, generate held-out predictions
        preds = te[["year", "musher_id", "finish_place"]].copy()

        # Win model
        mask = tr["top5"].notna()
        model = _make_calibrated_model(class_weight="balanced")
        model.fit(tr.loc[mask, win_mcols], tr.loc[mask, "top5"].astype(int))
        preds["p_won"] = derive_win_probability(model.predict_proba(te[win_mcols])[:, 1])

        # Rank models
        for target, col in [("top5", "p_top5"), ("top10", "p_top10"), ("finished", "p_finished")]:
            mask = tr[target].notna()
            cw = "balanced" if target != "finished" else None
            m = _make_calibrated_model(class_weight=cw)
            m.fit(tr.loc[mask, rank_mcols], tr.loc[mask, target].astype(int))
            preds[col] = m.predict_proba(te[rank_mcols])[:, 1]

        # Composite ranking
        for t in ["won", "top5", "top10", "finished"]:
            preds[f"rank_{t}"] = preds[f"p_{t}"].rank(ascending=False, method="min")
        preds["composite"] = (
            0.10 * preds["rank_won"] + 0.25 * preds["rank_top5"]
            + 0.40 * preds["rank_top10"] + 0.25 * preds["rank_finished"]
        )

        # Evaluate
        valid = preds[preds["finish_place"].notna()].copy()
        valid["finish_place"] = pd.to_numeric(valid["finish_place"])

        if len(valid) < 5:
            continue

        # Spearman
        rho, _ = spearmanr(valid["composite"], valid["finish_place"])

        # P@5 and P@10
        pred_t5 = set(valid.nsmallest(5, "composite")["musher_id"])
        actual_t5 = set(valid.nsmallest(5, "finish_place")["musher_id"])
        p_at_5 = len(pred_t5 & actual_t5) / 5

        pred_t10 = set(valid.nsmallest(10, "composite")["musher_id"])
        actual_t10 = set(valid.nsmallest(10, "finish_place")["musher_id"])
        p_at_10 = len(pred_t10 & actual_t10) / 10

        # AUC for top-10 classification
        y_true = (valid["finish_place"] <= 10).astype(int)
        if y_true.nunique() >= 2:
            auc = roc_auc_score(y_true, valid["p_top10"])
        else:
            auc = np.nan

        # Winner identification
        winner = valid[valid["finish_place"] == 1]
        if not winner.empty:
            winner_comp_rank = valid["composite"].rank(ascending=True, method="min").loc[winner.index[0]]
            winner_in_top3 = 1 if winner_comp_rank <= 3 else 0
            winner_in_top5 = 1 if winner_comp_rank <= 5 else 0
        else:
            winner_in_top3 = np.nan
            winner_in_top5 = np.nan

        year_results.append({
            "year": test_yr,
            "spearman": rho,
            "P@5": p_at_5,
            "P@10": p_at_10,
            "AUC": auc,
            "winner_in_top3": winner_in_top3,
            "winner_in_top5": winner_in_top5,
        })

    results_df = pd.DataFrame(year_results)

    # Compute CIs
    metric_cols = ["spearman", "P@5", "P@10", "AUC", "winner_in_top3", "winner_in_top5"]
    cis = compute_bootstrap_cis(results_df, metric_cols=metric_cols, n_boot=n_boot)

    return results_df, cis


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_start", type=int, default=2006)
    ap.add_argument("--train_end", type=int, default=2025)
    ap.add_argument("--start_year", type=int, default=2015)
    ap.add_argument("--n_boot", type=int, default=10000)
    args = ap.parse_args()

    con = connect()

    print(f"Running pre-race backtest with bootstrap CIs...")
    print(f"  Train window: {args.train_start}–{args.train_end}")
    print(f"  Backtest years: {args.start_year}–{args.train_end}")
    print(f"  Bootstrap resamples: {args.n_boot}")

    results_df, cis = run_prerace_backtest_with_cis(
        con, args.train_start, args.train_end, args.start_year, args.n_boot
    )

    print(f"\nPer-year results:")
    print(results_df.to_string(index=False, float_format="%.3f"))

    print(f"\n{'='*72}")
    print("BACKTEST METRICS WITH 95% BOOTSTRAP CONFIDENCE INTERVALS")
    print(f"(block bootstrap, n_boot={args.n_boot}, resampling at year level)")
    print(f"{'='*72}")
    print(format_ci(cis))


if __name__ == "__main__":
    main()
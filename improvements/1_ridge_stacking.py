"""
1_ridge_stacking.py
====================
Replace hand-tuned composite ranking weights (10/25/40/25) with
a Ridge regression meta-learner trained on LOOCV held-out predictions.

HOW IT WORKS:
  1. For each backtest year, train 4 sub-models (win, top5, top10, finish)
     on all other years, then generate held-out probability predictions.
  2. Convert probabilities -> ranks (per year).
  3. Train a Ridge regression on the held-out ranks to predict actual finish place.
  4. The learned Ridge coefficients ARE the new blend weights.

TARGET MODES:
  The key insight: what Ridge optimizes for determines what weights it learns.
  Predicting raw finish place treats musher #35 vs #40 the same as #1 vs #5.
  Alternative targets upweight the top of the field:

    raw     - predict finish_place (default, full-field MSE)
    inverse - predict 1/finish_place (heavily upweights top positions)
    log     - predict log(finish_place) (moderate upweighting)
    top15   - predict finish_place, trained only on top-15 finishers
    top10   - predict finish_place, trained only on top-10 finishers

Usage:
    python improvements/1_ridge_stacking.py --compare_targets
    python improvements/1_ridge_stacking.py --alpha_search --target_mode inverse
    python improvements/1_ridge_stacking.py --target_mode top10 --alpha 1.0
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
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.calibration import CalibratedClassifierCV

from src.db import connect


# Same feature sets as predict_prerace_2026.py
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


def _model_cols(feature_cols):
    return feature_cols + [f"{c}_missing" for c in feature_cols]


def derive_win_probability(p_top5):
    EXPONENT = 3.0
    raw = np.power(np.clip(p_top5, 1e-6, 1.0), EXPONENT)
    return raw / raw.sum()


def build_data(con, year_min, year_max):
    """Load musher_strength + historical_results, create labels and model columns."""
    all_features = list(dict.fromkeys(RANK_FEATURES + WIN_FEATURES))

    ms = con.execute(
        f"SELECT year, musher_id, {', '.join(all_features)} "
        f"FROM musher_strength WHERE year BETWEEN ? AND ?",
        [year_min, year_max],
    ).df()
    hr = con.execute(
        "SELECT year, musher_id, finish_place, status "
        "FROM historical_results WHERE year BETWEEN ? AND ?",
        [year_min, year_max],
    ).df()
    df = ms.merge(hr, on=["year", "musher_id"], how="inner")

    df["won"] = (df["finish_place"] == 1).astype("Int64")
    df["top5"] = (df["finish_place"].notna() & (df["finish_place"] <= 5)).astype("Int64")
    df["top10"] = (df["finish_place"].notna() & (df["finish_place"] <= 10)).astype("Int64")
    df["finished"] = (df["finish_place"].notna()).astype("Int64")

    df[all_features] = df[all_features].apply(pd.to_numeric, errors="coerce")
    for f in all_features:
        df[f"{f}_missing"] = df[f].isna().astype("int64")

    return df


def run_loocv(df, start_year, end_year):
    """
    Leave-one-year-out CV: for each test year, train 4 sub-models on remaining
    years and collect held-out probability predictions.
    """
    win_mcols = _model_cols(WIN_FEATURES)
    rank_mcols = _model_cols(RANK_FEATURES)

    all_preds = []

    for test_yr in range(start_year, end_year + 1):
        tr = df[df["year"] < test_yr].copy()
        te = df[df["year"] == test_yr].copy()

        if tr.empty or te.empty:
            continue
        if tr["top5"].sum() < 3 or tr["top10"].sum() < 3:
            continue

        year_preds = te[["year", "musher_id", "finish_place"]].copy()

        # Win model: v7 features -> P(top5) -> derived win prob
        mask = tr["top5"].notna()
        model_win = _make_calibrated_model(class_weight="balanced")
        model_win.fit(tr.loc[mask, win_mcols], tr.loc[mask, "top5"].astype(int))
        p_top5_for_win = model_win.predict_proba(te[win_mcols])[:, 1]
        year_preds["p_won"] = derive_win_probability(p_top5_for_win)

        # Rank models: v8 features
        for target, col_name in [("top5", "p_top5"), ("top10", "p_top10"), ("finished", "p_finished")]:
            mask = tr[target].notna()
            cw = "balanced" if target != "finished" else None
            model = _make_calibrated_model(class_weight=cw)
            model.fit(tr.loc[mask, rank_mcols], tr.loc[mask, target].astype(int))
            year_preds[col_name] = model.predict_proba(te[rank_mcols])[:, 1]

        # Convert probs -> within-year ranks
        for target in ["won", "top5", "top10", "finished"]:
            year_preds[f"rank_{target}"] = year_preds[f"p_{target}"].rank(ascending=False, method="min")

        all_preds.append(year_preds)

    return pd.concat(all_preds, ignore_index=True)


def fit_ridge_weights(preds_df, alpha=1.0, non_negative=True, target_mode="raw"):
    """
    Fit a Ridge regression on held-out ranks to predict actual finish place.

    target_mode controls what Ridge optimizes for:
      "raw"       - predict finish_place directly (favors full-field accuracy)
      "inverse"   - predict 1/finish_place (upweights getting top positions right)
      "top15"     - predict finish_place but only train on top-15 finishers
      "top10"     - predict finish_place but only train on top-10 finishers
      "log"       - predict log(finish_place) (moderate upweighting of top)
    """
    valid = preds_df[preds_df["finish_place"].notna()].copy()
    valid["finish_place"] = pd.to_numeric(valid["finish_place"], errors="coerce")
    valid = valid[valid["finish_place"].notna()].copy()

    rank_cols = ["rank_won", "rank_top5", "rank_top10", "rank_finished"]

    # --- Build training subset and target based on mode ---
    if target_mode == "top15":
        train = valid[valid["finish_place"] <= 15].copy()
        y = train["finish_place"].values
    elif target_mode == "top10":
        train = valid[valid["finish_place"] <= 10].copy()
        y = train["finish_place"].values
    elif target_mode == "inverse":
        train = valid.copy()
        y = 1.0 / train["finish_place"].values
    elif target_mode == "log":
        train = valid.copy()
        y = np.log(train["finish_place"].values)
    else:  # "raw"
        train = valid.copy()
        y = train["finish_place"].values

    X = train[rank_cols].values

    if len(X) < 10:
        return ({"won": 0.25, "top5": 0.25, "top10": 0.25, "finished": 0.25},
                None, {"ridge": {}, "original": {}})

    if non_negative:
        from sklearn.linear_model import Ridge as SkRidge
        ridge = SkRidge(alpha=alpha, fit_intercept=False, positive=True)
    else:
        ridge = Ridge(alpha=alpha, fit_intercept=False)

    ridge.fit(X, y)

    raw_coefs = ridge.coef_
    total = np.sum(np.abs(raw_coefs))
    if total > 0:
        norm_coefs = np.abs(raw_coefs) / total
    else:
        norm_coefs = np.array([0.25, 0.25, 0.25, 0.25])

    weights = {
        "won": float(norm_coefs[0]),
        "top5": float(norm_coefs[1]),
        "top10": float(norm_coefs[2]),
        "finished": float(norm_coefs[3]),
    }

    # --- Diagnostics: always evaluate on ALL finishers (apples-to-apples) ---
    valid["composite_learned"] = sum(
        norm_coefs[i] * valid[rank_cols[i]] for i in range(4)
    )
    valid["composite_original"] = (
        0.10 * valid["rank_won"]
        + 0.25 * valid["rank_top5"]
        + 0.40 * valid["rank_top10"]
        + 0.25 * valid["rank_finished"]
    )

    diag = {}
    for label, col in [("ridge", "composite_learned"), ("original", "composite_original")]:
        year_spearman = []
        year_winner_rank = []
        year_p5 = []
        year_p10 = []

        for yr, grp in valid.groupby("year"):
            if len(grp) < 5:
                continue
            rho, _ = spearmanr(grp[col], grp["finish_place"])
            year_spearman.append(rho)

            winner = grp[grp["finish_place"] == 1]
            if not winner.empty:
                all_ranks = grp[col].rank(ascending=True, method="min")
                winner_composite_rank = all_ranks.loc[winner.index[0]]
                year_winner_rank.append(winner_composite_rank)

            pred_top5_ids = set(grp.nsmallest(5, col)["musher_id"])
            actual_top5_ids = set(grp.nsmallest(5, "finish_place")["musher_id"])
            year_p5.append(len(pred_top5_ids & actual_top5_ids) / 5)

            pred_top10_ids = set(grp.nsmallest(10, col)["musher_id"])
            actual_top10_ids = set(grp.nsmallest(10, "finish_place")["musher_id"])
            year_p10.append(len(pred_top10_ids & actual_top10_ids) / 10)

        diag[label] = {
            "spearman": np.mean(year_spearman) if year_spearman else np.nan,
            "avg_winner_rank": np.mean(year_winner_rank) if year_winner_rank else np.nan,
            "P@5": np.mean(year_p5) if year_p5 else np.nan,
            "P@10": np.mean(year_p10) if year_p10 else np.nan,
            "winner_in_top3": np.mean([1 if r <= 3 else 0 for r in year_winner_rank]) if year_winner_rank else np.nan,
            "winner_in_top5": np.mean([1 if r <= 5 else 0 for r in year_winner_rank]) if year_winner_rank else np.nan,
            "n_years": len(year_spearman),
        }

    return weights, ridge, diag


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_start", type=int, default=2006)
    ap.add_argument("--train_end", type=int, default=2025)
    ap.add_argument("--start_year", type=int, default=2015,
                    help="First year to include in LOOCV backtest (must have enough training history)")
    ap.add_argument("--alpha", type=float, default=1.0,
                    help="Ridge regularization strength. Higher = more conservative/even weights.")
    ap.add_argument("--no_non_negative", action="store_true",
                    help="Allow negative coefficients (default: non-negative constraint)")
    ap.add_argument("--alpha_search", action="store_true",
                    help="Search over multiple alpha values and report results")
    ap.add_argument("--target_mode", type=str, default="raw",
                    choices=["raw", "inverse", "top15", "top10", "log"],
                    help="Ridge target: raw, inverse (1/place), top15, top10, log")
    ap.add_argument("--compare_targets", action="store_true",
                    help="Compare all target modes side-by-side at a fixed alpha")
    args = ap.parse_args()

    con = connect()
    df = build_data(con, args.train_start, args.train_end)
    print(f"Loaded {len(df)} musher-years ({args.train_start}-{args.train_end})")

    # Run LOOCV to collect held-out predictions
    print(f"\nRunning LOOCV from {args.start_year} to {args.train_end}...")
    preds = run_loocv(df, args.start_year, args.train_end)
    n_years = preds["year"].nunique()
    print(f"Collected {len(preds)} held-out predictions across {n_years} years")

    non_negative = not args.no_non_negative

    if args.compare_targets:
        # ---- Compare all target modes at fixed alpha ----
        print(f"\n{'='*95}")
        print(f"TARGET MODE COMPARISON (alpha={args.alpha}, non_negative={non_negative})")
        print(f"{'='*95}")
        print(f"{'Mode':>10} | {'W_win':>6} {'W_t5':>6} {'W_t10':>6} {'W_fin':>6} | "
              f"{'Spear':>6} {'P@5':>6} {'P@10':>6} {'W_top3':>7} {'W_top5':>7}")
        print("-" * 95)

        for mode in ["raw", "inverse", "log", "top15", "top10"]:
            w, _, d = fit_ridge_weights(preds, alpha=args.alpha,
                                        non_negative=non_negative, target_mode=mode)
            r = d["ridge"]
            print(f"{mode:>10} | {w['won']:>6.3f} {w['top5']:>6.3f} {w['top10']:>6.3f} {w['finished']:>6.3f} | "
                  f"{r['spearman']:>6.3f} {r['P@5']:>6.3f} {r['P@10']:>6.3f} "
                  f"{r['winner_in_top3']:>7.1%} {r['winner_in_top5']:>7.1%}")

        # Original hand-tuned weights for comparison
        _, _, d_orig = fit_ridge_weights(preds, alpha=1.0, non_negative=non_negative, target_mode="raw")
        o = d_orig["original"]
        print(f"{'ORIG':>10} | {'0.100':>6} {'0.250':>6} {'0.400':>6} {'0.250':>6} | "
              f"{o['spearman']:>6.3f} {o['P@5']:>6.3f} {o['P@10']:>6.3f} "
              f"{o['winner_in_top3']:>7.1%} {o['winner_in_top5']:>7.1%}")

        print(f"\nModes:")
        print(f"  raw     = predict finish_place for all finishers (default)")
        print(f"  inverse = predict 1/finish_place (heavily upweights top positions)")
        print(f"  log     = predict log(finish_place) (moderate top upweighting)")
        print(f"  top15   = predict finish_place, trained only on top-15 finishers")
        print(f"  top10   = predict finish_place, trained only on top-10 finishers")

    elif args.alpha_search:
        # ---- Alpha sweep for a single target mode ----
        print(f"\n{'='*95}")
        print(f"ALPHA SEARCH (target_mode={args.target_mode})")
        print(f"{'='*95}")
        print(f"{'Alpha':>8} | {'W_win':>6} {'W_t5':>6} {'W_t10':>6} {'W_fin':>6} | "
              f"{'Spear':>6} {'P@5':>6} {'P@10':>6} {'W_top3':>7} {'W_top5':>7}")
        print("-" * 95)

        for alpha in [0.01, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 50.0, 100.0]:
            w, _, d = fit_ridge_weights(preds, alpha=alpha,
                                        non_negative=non_negative, target_mode=args.target_mode)
            r = d["ridge"]
            print(f"{alpha:>8.2f} | {w['won']:>6.3f} {w['top5']:>6.3f} {w['top10']:>6.3f} {w['finished']:>6.3f} | "
                  f"{r['spearman']:>6.3f} {r['P@5']:>6.3f} {r['P@10']:>6.3f} "
                  f"{r['winner_in_top3']:>7.1%} {r['winner_in_top5']:>7.1%}")

        _, _, d_orig = fit_ridge_weights(preds, alpha=1.0, non_negative=non_negative, target_mode=args.target_mode)
        o = d_orig["original"]
        print(f"{'ORIG':>8} | {'0.100':>6} {'0.250':>6} {'0.400':>6} {'0.250':>6} | "
              f"{o['spearman']:>6.3f} {o['P@5']:>6.3f} {o['P@10']:>6.3f} "
              f"{o['winner_in_top3']:>7.1%} {o['winner_in_top5']:>7.1%}")

    else:
        # ---- Single run ----
        weights, ridge, diag = fit_ridge_weights(
            preds, alpha=args.alpha, non_negative=non_negative, target_mode=args.target_mode
        )

        print(f"\n{'='*70}")
        print(f"RIDGE STACKING (alpha={args.alpha}, target={args.target_mode}, non_neg={non_negative})")
        print(f"{'='*70}")
        print(f"\nLearned weights (normalized to sum=1):")
        print(f"  Win:      {weights['won']:.4f}  (was 0.10)")
        print(f"  Top 5:    {weights['top5']:.4f}  (was 0.25)")
        print(f"  Top 10:   {weights['top10']:.4f}  (was 0.40)")
        print(f"  Finish:   {weights['finished']:.4f}  (was 0.25)")

        if ridge is not None:
            print(f"\nRaw Ridge coefficients: {ridge.coef_}")

        print(f"\nComparison (on {diag['ridge']['n_years']} LOOCV years):")
        print(f"{'Metric':>20} | {'Ridge':>10} | {'Original':>10} | {'Delta':>10}")
        print("-" * 58)
        for metric in ["spearman", "P@5", "P@10", "avg_winner_rank", "winner_in_top3", "winner_in_top5"]:
            r = diag["ridge"][metric]
            o = diag["original"][metric]
            delta = r - o if metric != "avg_winner_rank" else o - r
            sign = "+" if delta > 0 else ""
            fmt = ".1%" if "winner" in metric else ".3f"
            print(f"{metric:>20} | {r:>10{fmt}} | {o:>10{fmt}} | {sign}{delta:>9{fmt}}")

        print(f"\n--- Copy this into predict_prerace_2026.py ---")
        print(f'results["composite_rank"] = (')
        print(f"    {weights['won']:.4f} * results[\"rank_won\"]")
        print(f"    + {weights['top5']:.4f} * results[\"rank_top5\"]")
        print(f"    + {weights['top10']:.4f} * results[\"rank_top10\"]")
        print(f"    + {weights['finished']:.4f} * results[\"rank_finished\"]")
        print(f")")


if __name__ == "__main__":
    main()
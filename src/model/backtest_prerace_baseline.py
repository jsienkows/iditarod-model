"""
Rolling backtest for the pre-race logistic regression model.

Runs leave-one-year-out evaluation, reporting AUC, Brier score, precision@K,
and rank correlation for each held-out year.

Usage:
    python -m src.model.backtest_prerace_baseline --start_year 2015 --end_year 2025
"""

import argparse
import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import log_loss, roc_auc_score, brier_score_loss

from src.db import connect
from src.model.calibration_utils import reliability_table


# ---- Bucket feature set (explicit, conceptual) ----
BUCKET_FEATURES = [
    # Peak ability
    "peak_best_finish_place",
    "peak_pct_top10",
    "peak_pct_top5",
    "peak_pct_win",
    # Current form
    "form_last3_avg_finish_place",
    "form_last5_avg_finish_place",
    "form_last3_pct_top10",
    "form_last5_pct_top10",
    "form_w_avg_finish_place",
    "form_w_pct_top10",
    # Experience / volume
    "exp_n_entries",
    "exp_n_finishes",
    "exp_n_years",
    "exp_w_entries",
    # Rust / comeback
    "rust_years_since_last_entry",
    "rust_is_rookie",
    # Consistency / reliability
    "cons_pct_finished",
    "cons_last5_pct_finished",
    "cons_w_pct_finished",
]

# ---- Slim feature set: non-redundant, from raw columns ----
SLIM_FEATURES = [
    "best_finish_place",
    "w_avg_finish_place",
    "w_pct_top10",
    "n_finishes",
    "w_n_entries",
    "w_pct_finished",
    "years_since_last_entry",
    "is_rookie",
    "last_year_finish_place",
]


def make_model(class_weight="balanced"):
    return Pipeline([
        ("imputer", SimpleImputer(strategy="constant", fill_value=0, keep_empty_features=True)),
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=4000, class_weight=class_weight)),
    ])


def _wrap_with_calibration(base_model, method="sigmoid", cv=5):
    try:
        return CalibratedClassifierCV(estimator=base_model, method=method, cv=cv)
    except TypeError:
        return CalibratedClassifierCV(base_estimator=base_model, method=method, cv=cv)


def _table_columns(con, table: str) -> list[str]:
    info = con.execute(f"PRAGMA table_info('{table}')").df()
    return info["name"].tolist()


def load_feature_cols(con, feature_set: str) -> list[str]:
    cols = _table_columns(con, "musher_strength")

    if feature_set == "slim":
        return [c for c in SLIM_FEATURES if c in cols]

    if feature_set == "buckets":
        return [c for c in BUCKET_FEATURES if c in cols]

    exclude = {
        "year", "musher_id", "name", "name_canonical",
        "finish_place", "finish_time_seconds", "status",
        "won", "top10", "top5",
    }

    raw = [c for c in cols if c not in exclude]
    raw = [c for c in raw if c not in BUCKET_FEATURES]
    raw = [c for c in raw if not c.endswith("_missing")]
    return raw


def build_dataset(con, year_min: int, year_max: int, feature_set: str) -> tuple[pd.DataFrame, list[str], list[str]]:
    feature_cols = load_feature_cols(con, feature_set=feature_set)
    if not feature_cols:
        raise RuntimeError(
            f"No feature columns found for feature_set='{feature_set}'. "
            f"Do the bucket columns exist/populate in musher_strength?"
        )

    print(f"[build_dataset] feature_set={feature_set} | n_features={len(feature_cols)}")

    ms = con.execute(
        f"""
        SELECT year, musher_id, {", ".join(feature_cols)}
        FROM musher_strength
        WHERE year BETWEEN ? AND ?
        """,
        [year_min, year_max],
    ).df()

    hr = con.execute(
        """
        SELECT year, musher_id, finish_place, finish_time_seconds, status
        FROM historical_results
        WHERE year BETWEEN ? AND ?
        """,
        [year_min, year_max],
    ).df()

    df = ms.merge(hr, on=["year", "musher_id"], how="left")

    # --- Labels (IMPORTANT: coerce finish_place numeric first) ---
    df["finish_place"] = pd.to_numeric(df["finish_place"], errors="coerce")

    # DNFs: finish_place is NULL -> should be 0 for won/top10/top5 (i.e., did not achieve)
    df["won"] = (df["finish_place"] == 1).astype("Int64").fillna(0)
    df["top10"] = (df["finish_place"].notna() & (df["finish_place"] <= 10)).astype("Int64").fillna(0)
    df["top5"] = (df["finish_place"].notna() & (df["finish_place"] <= 5)).astype("Int64").fillna(0)

    # Coerce numeric features (vectorized)
    df[feature_cols] = df[feature_cols].apply(pd.to_numeric, errors="coerce")

    # Missing flags in one shot (prevents fragmentation)
    missing = df[feature_cols].isna().astype("int64")
    missing.columns = [f"{c}_missing" for c in feature_cols]
    df = pd.concat([df, missing], axis=1)

    model_cols = feature_cols + missing.columns.tolist()
    return df, model_cols, feature_cols


def eval_binary(y_true, p_pred):
    y_true = pd.Series(y_true).astype(int)
    ll = log_loss(y_true, p_pred)
    auc = roc_auc_score(y_true, p_pred) if y_true.nunique() > 1 else np.nan
    brier = brier_score_loss(y_true, p_pred)
    return ll, auc, brier


def precision_at_n(df: pd.DataFrame, target_col: str, p_col: str, n: int) -> float:
    df = df[df[target_col].notna()].copy()
    if df.empty:
        return np.nan
    n = min(n, len(df))
    true_set = set(df.loc[df[target_col] == 1, "musher_id"])
    pred_set = set(df.sort_values(p_col, ascending=False).head(n)["musher_id"])
    return len(true_set & pred_set) / (n if n > 0 else 1)


def _parse_class_weight(x: str):
    if x is None:
        return None
    if isinstance(x, str) and x.strip().lower() in {"none", "null", ""}:
        return None
    return x


def _print_topn(d_test: pd.DataFrame, target: str, p_col: str, topn: int, label_only: bool):
    d = d_test.copy()
    if label_only:
        d = d[d[target].notna()].copy()

    if d.empty:
        scope = "LABELED-ONLY" if label_only else "ALL"
        print(f"\nTop {topn} predicted ({scope}): <none>")
        return

    n = min(topn, len(d))
    scope = "LABELED-ONLY" if label_only else "ALL"

    show_cols = ["year", "musher_id", p_col, "finish_place", "finish_time_seconds", "status"]
    show_cols = [c for c in show_cols if c in d.columns]

    print(f"\nTop {n} predicted ({scope}, p_col={p_col}):")
    print(d.sort_values(p_col, ascending=False).head(n)[show_cols].to_string(index=False))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start_year", type=int, default=2016)
    ap.add_argument("--end_year", type=int, default=2025)
    ap.add_argument("--target", choices=["won", "top10", "top5"], default="top10")
    ap.add_argument("--class_weight", default="balanced", help="balanced or None")
    ap.add_argument("--topn", type=int, default=10)

    ap.add_argument(
        "--feature_set",
        choices=["raw", "buckets", "slim"],
        default="slim",
        help="Use all musher_strength columns (raw), explicit conceptual buckets (buckets), or curated slim set (slim).",
    )

    ap.add_argument("--calibrate", choices=["none", "sigmoid", "isotonic"], default="none")
    ap.add_argument("--calibration_cv", type=int, default=5)
    ap.add_argument("--reliability_bins", type=int, default=0, help="0 disables; otherwise number of bins")

    ap.add_argument(
        "--show_topn",
        choices=["none", "all", "labeled", "both"],
        default="none",
        help="Print top-N predictions per fold: none|all|labeled|both",
    )

    args = ap.parse_args()
    con = connect()

    test_years = list(range(args.start_year + 1, args.end_year + 1))

    df, MODEL_COLS, base_features = build_dataset(con, args.start_year, args.end_year, feature_set=args.feature_set)
    print(f"[build_dataset] base features: {len(base_features)} | model cols: {len(MODEL_COLS)} (incl. missing flags)")

    rows = []
    for test_year in test_years:
        train_start = args.start_year
        train_end = test_year - 1

        d_train = df[(df["year"] >= train_start) & (df["year"] <= train_end)].copy()
        d_test = df[df["year"] == test_year].copy()

        # Drop unknown labels from training (should be none for top10/top5 given our label creation,
        # but keep this for won or any future targets)
        y_train = d_train[args.target]
        train_mask = y_train.notna()
        d_train = d_train[train_mask].copy()
        y_train = d_train[args.target].astype(int)

        if y_train.nunique() < 2 or d_test.empty:
            rows.append({
                "test_year": test_year,
                "train_years": f"{train_start}..{train_end}",
                "train_rows": len(d_train),
                "test_rows": len(d_test),
                "n_labeled": 0,
                "labeled_rate": np.nan,
                "base_rate": np.nan,
                "avg_pred_p": np.nan,
                "avg_pred_p_labeled": np.nan,
                "calibration_error": np.nan,
                "logloss": np.nan,
                "auc": np.nan,
                "brier": np.nan,
                "precision_at_n": np.nan,
            })
            continue

        X_train = d_train[MODEL_COLS]
        cw = _parse_class_weight(args.class_weight)
        base_model = make_model(class_weight=cw)

        model = base_model
        if args.calibrate != "none":
            model = _wrap_with_calibration(base_model, method=args.calibrate, cv=args.calibration_cv)

        model.fit(X_train, y_train)

        X_test = d_test[MODEL_COLS]
        p = model.predict_proba(X_test)[:, 1]
        d_test = d_test.assign(p=p)

        y_test = d_test[args.target]
        test_mask = y_test.notna()

        # --- Calibration stability stats (use LABELED subset consistently) ---
        n_labeled = int(test_mask.sum())
        labeled_rate = float(n_labeled / len(d_test)) if len(d_test) else np.nan

        base_rate = float((d_test.loc[test_mask, args.target] == 1).mean()) if test_mask.any() else np.nan
        avg_pred_p = float(d_test["p"].mean()) if len(d_test) else np.nan
        avg_pred_p_labeled = float(d_test.loc[test_mask, "p"].mean()) if test_mask.any() else np.nan

        calibration_error = float(abs(avg_pred_p_labeled - base_rate)) if test_mask.any() else np.nan

        if test_mask.sum() > 0 and y_test[test_mask].nunique() >= 2:
            ll, auc, brier = eval_binary(y_test[test_mask].astype(int), d_test.loc[test_mask, "p"])
        else:
            ll, auc, brier = np.nan, np.nan, np.nan

        p_at_n = precision_at_n(d_test, args.target, "p", args.topn)

        out = {
            "test_year": test_year,
            "train_years": f"{train_start}..{train_end}",
            "train_rows": len(d_train),
            "test_rows": len(d_test),
            "n_labeled": n_labeled,
            "labeled_rate": labeled_rate,
            "base_rate": base_rate,
            "avg_pred_p": avg_pred_p,
            "avg_pred_p_labeled": avg_pred_p_labeled,
            "calibration_error": calibration_error,
            "logloss": ll,
            "auc": auc,
            "brier": brier,
            "precision_at_n": p_at_n,
        }

        if args.reliability_bins and test_mask.sum() > 0 and y_test[test_mask].nunique() >= 2:
            try:
                out["reliability"] = reliability_table(
                    y_test[test_mask].astype(int),
                    d_test.loc[test_mask, "p"],
                    n_bins=args.reliability_bins
                )
            except Exception:
                out["reliability"] = None

        rows.append(out)

        if args.show_topn != "none":
            print(f"\n=== Fold test_year={test_year} (feature_set={args.feature_set}, calibrate={args.calibrate}) ===")
            if args.show_topn in {"all", "both"}:
                _print_topn(d_test, args.target, "p", args.topn, label_only=False)
            if args.show_topn in {"labeled", "both"}:
                _print_topn(d_test, args.target, "p", args.topn, label_only=True)

    res = pd.DataFrame(rows)

    print(f"\nBacktest results (target={args.target}, calibrate={args.calibrate}, feature_set={args.feature_set})")
    cols = [
        "test_year", "train_years", "train_rows", "test_rows",
        "n_labeled", "labeled_rate",
        "base_rate", "avg_pred_p", "avg_pred_p_labeled", "calibration_error",
        "logloss", "auc", "brier", "precision_at_n",
    ]
    cols = [c for c in cols if c in res.columns]
    print(res[cols].to_string(index=False))

    print("\nSummary (mean ± std over folds with non-null metrics):")
    for m in ["logloss", "auc", "brier", "precision_at_n", "calibration_error"]:
        if m not in res.columns:
            continue
        v = res[m].dropna()
        if len(v) == 0:
            print(f"  {m}: (no valid folds)")
        else:
            print(f"  {m}: {v.mean():.4f} ± {v.std(ddof=1):.4f} (n={len(v)})")

    if args.reliability_bins and "reliability" in res.columns:
        print("\nReliability (last 2 folds with data):")
        last = res.dropna(subset=["reliability"]).tail(2)
        for _, r in last.iterrows():
            print(f"\nYear {int(r['test_year'])} reliability:")
            print(r["reliability"])


if __name__ == "__main__":
    main()

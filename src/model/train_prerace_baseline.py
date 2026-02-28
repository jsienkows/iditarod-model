"""
Train the pre-race logistic regression model.

Trains a calibrated logistic regression to predict top-5, top-10, or
finish probability from musher strength features. Supports multiple
feature sets (slim, full) for comparison.

Usage:
    python -m src.model.train_prerace_baseline --train_start 2006 --train_end 2024 --test_year 2025
"""

import argparse
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import log_loss, roc_auc_score, brier_score_loss

from src.model.calibration_utils import reliability_table
from src.db import connect


DEFAULT_TRAIN_START = 2016
DEFAULT_TRAIN_END = 2024
DEFAULT_TEST_YEAR = 2025


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

# ---- Slim feature set: non-redundant, directly from raw columns ----
# These are the raw source columns (not the bucket renames, which are identical copies).
# Selected for low correlation, high stability, and domain relevance.
SLIM_FEATURES = [
    # Peak ability (1 feature — best finish is the single strongest predictor)
    "best_finish_place",
    # Recent form (2 features — weighted avg is better than fixed windows)
    "w_avg_finish_place",
    "w_pct_top10",
    # Experience (2 features — total finishes + weighted entries)
    "n_finishes",
    "w_n_entries",
    # Consistency (1 feature — weighted finish rate)
    "w_pct_finished",
    # Recency / rust (2 features)
    "years_since_last_entry",
    "is_rookie",
    # Last year snapshot (1 feature — strong recent signal)
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
    """
    feature_set:
      - "raw": all musher_strength columns except IDs + buckets + obvious labels
      - "buckets": explicit conceptual bucket columns only
      - "slim": curated non-redundant set from raw columns (recommended)
    """
    cols = _table_columns(con, "musher_strength")

    if feature_set == "slim":
        return [c for c in SLIM_FEATURES if c in cols]

    if feature_set == "buckets":
        return [c for c in BUCKET_FEATURES if c in cols]

    exclude = {
        "year", "musher_id", "name", "name_canonical",
        # future-proof: if labels ever appear in musher_strength, keep them out
        "finish_place", "finish_time_seconds", "status",
        "won", "top10", "top5",
    }

    raw = [c for c in cols if c not in exclude]
    raw = [c for c in raw if c not in BUCKET_FEATURES]
    raw = [c for c in raw if not c.endswith("_missing")]  # defensive
    return raw


def build_dataset(con, year_min: int, year_max: int, feature_set: str) -> tuple[pd.DataFrame, list[str], list[str]]:
    """
    Returns:
      df: musher-year rows with features + labels
      model_cols: feature cols + missing flags
      feature_cols: base feature cols only (no missing flags)
    """
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

    # Labels
    df["won"]   = (df["finish_place"] == 1).astype("Int64")
    df["top10"] = (df["finish_place"].notna() & (df["finish_place"] <= 10)).astype("Int64")
    df["top5"]  = (df["finish_place"].notna() & (df["finish_place"] <= 5)).astype("Int64")

    # Coerce features numeric (vectorized)
    df[feature_cols] = df[feature_cols].apply(pd.to_numeric, errors="coerce")

    # Missing flags in one shot (avoids fragmentation)
    missing = df[feature_cols].isna().astype("int64")
    missing.columns = [f"{c}_missing" for c in feature_cols]
    df = pd.concat([df, missing], axis=1)

    model_cols = feature_cols + missing.columns.tolist()
    return df, model_cols, feature_cols


def eval_binary(y_true, p_pred):
    y_true = pd.Series(y_true).astype(int)
    ll = log_loss(y_true, p_pred)
    auc = roc_auc_score(y_true, p_pred) if y_true.nunique() > 1 else None
    brier = brier_score_loss(y_true, p_pred)
    return ll, auc, brier


def _parse_class_weight(x: str):
    if x is None:
        return None
    if isinstance(x, str) and x.strip().lower() in {"none", "null", ""}:
        return None
    return x


def _print_topn(d_test: pd.DataFrame, target: str, p_col: str, topn: int, label_only: bool, test_year: int):
    """
    Print top-N predictions. If label_only=True, restrict to rows where target label exists.
    """
    d = d_test.copy()
    if label_only:
        d = d[d[target].notna()].copy()

    scope = "LABELED-ONLY" if label_only else "ALL"
    if d.empty:
        print(f"\nTop {topn} predicted ({scope}, year={test_year}): <none>")
        return

    n = min(topn, len(d))

    show_cols = ["year", "musher_id", "p_raw", "p_cal", "finish_place", "finish_time_seconds", "status"]
    show_cols = [c for c in show_cols if c in d.columns]

    print(f"\nTop {n} predicted ({scope}, year={test_year}, sort={p_col}):")
    print(d.sort_values(p_col, ascending=False).head(n)[show_cols].to_string(index=False))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_start", type=int, default=DEFAULT_TRAIN_START)
    ap.add_argument("--train_end", type=int, default=DEFAULT_TRAIN_END)
    ap.add_argument("--test_year", type=int, default=DEFAULT_TEST_YEAR)
    ap.add_argument("--target", choices=["won", "top10", "top5"], default="top10")
    ap.add_argument("--class_weight", default="balanced", help="balanced or None")
    ap.add_argument("--topn", type=int, default=10)

    ap.add_argument(
        "--feature_set",
        choices=["raw", "buckets", "slim"],
        default="slim",
        help="Use all musher_strength columns (raw), explicit conceptual buckets (buckets), or curated slim set (slim).",
    )

    ap.add_argument(
        "--calibrate",
        choices=["none", "sigmoid", "isotonic"],
        default="sigmoid",
        help="Probability calibration on the TRAIN set via CV",
    )
    ap.add_argument("--calib_cv", type=int, default=5)

    # Output control (match backtest)
    ap.add_argument(
        "--show_topn",
        choices=["none", "all", "labeled", "both"],
        default="all",
        help="Print top-N predictions: none|all|labeled|both",
    )

    args = ap.parse_args()
    con = connect()

    year_min = min(args.train_start, args.test_year)
    year_max = max(args.train_end, args.test_year)

    df, MODEL_COLS, FEATURE_COLS = build_dataset(con, year_min, year_max, feature_set=args.feature_set)

    d_train = df[(df["year"] >= args.train_start) & (df["year"] <= args.train_end)].copy()
    d_test = df[df["year"] == args.test_year].copy()

    print(
        f"Feature set: {args.feature_set} | base features: {len(FEATURE_COLS)} | "
        f"model cols: {len(MODEL_COLS)} (incl. missing flags)"
    )
    print(f"Train years: {args.train_start}..{args.train_end} | Test year: {args.test_year}")
    print("Train rows:", len(d_train), "| Test rows:", len(d_test))
    print("Train unique mushers:", d_train["musher_id"].nunique(), "| Test unique mushers:", d_test["musher_id"].nunique())

    if args.target not in d_train.columns:
        raise RuntimeError(f"Target column {args.target} not found in dataset.")

    # Drop unknown labels from training
    y_train = d_train[args.target].astype("Int64")
    train_mask = y_train.notna()
    d_train = d_train[train_mask].copy()
    y_train = d_train[args.target].astype(int)

    if y_train.nunique() < 2:
        raise RuntimeError(
            f"Not enough class variation for target={args.target} in training set "
            f"({y_train.value_counts().to_dict()})."
        )

    X_train = d_train[MODEL_COLS]

    # Train base model
    cw = _parse_class_weight(args.class_weight)
    base_model = make_model(class_weight=cw)
    base_model.fit(X_train, y_train)

    if d_test.empty:
        raise RuntimeError(f"No rows for test_year={args.test_year}. Did musher_strength build for that year?")

    X_test = d_test[MODEL_COLS]

    # Raw probabilities
    p_test_raw = base_model.predict_proba(X_test)[:, 1]

    # Calibration (train-set CV only)
    if args.calibrate == "none":
        p_test = p_test_raw
        p_label = "p_raw"
        print("\nCalibration: none (using raw probabilities)")
    else:
        calibrator = _wrap_with_calibration(base_model, method=args.calibrate, cv=args.calib_cv)
        calibrator.fit(X_train, y_train)
        p_test = calibrator.predict_proba(X_test)[:, 1]
        p_label = "p_cal"
        print(f"\nCalibration: {args.calibrate} (cv={args.calib_cv})")

    # Store both raw + calibrated
    d_test = d_test.assign(p_raw=p_test_raw, p_cal=p_test if p_label == "p_cal" else p_test_raw)

    print("\nSanity checks:")
    print("  Avg predicted p:", float(d_test[p_label].mean()))
    if d_test[args.target].notna().any():
        print("  Actual base rate:", float((d_test[args.target] == 1).mean()))

    # Evaluate on test year (only where label exists)
    y_test = d_test[args.target]
    test_mask = y_test.notna()

    if test_mask.sum() > 0 and y_test[test_mask].nunique() >= 2:
        ll, auc, brier = eval_binary(y_test[test_mask].astype(int), d_test.loc[test_mask, p_label])
        auc_str = "None" if auc is None else f"{auc:.4f}"
        print(f"\nTest metrics for target={args.target}:")
        print(f"  logloss: {ll:.4f}")
        print(f"  AUC:     {auc_str}")
        print(f"  brier:   {brier:.4f}")

        try:
            print(f"\nReliability table ({p_label}, target={args.target}):")
            print(reliability_table(y_test[test_mask].astype(int), d_test.loc[test_mask, p_label], n_bins=5))
        except Exception as e:
            print(f"\nReliability table skipped (error): {e}")

        try:
            print(f"\nReliability table (RAW probs, target={args.target}):")
            print(reliability_table(y_test[test_mask].astype(int), d_test.loc[test_mask, "p_raw"], n_bins=5))
        except Exception as e:
            print(f"\nRaw reliability table skipped (error): {e}")
    else:
        print("\nTest metrics skipped (test labels missing or only one class present).")

    # Precision@N on test year (if labels exist)
    topn = min(args.topn, len(d_test))
    if test_mask.sum() > 0:
        true_set = set(d_test.loc[(d_test[args.target] == 1) & test_mask, "musher_id"])
        pred_set = set(d_test.sort_values(p_label, ascending=False).head(topn)["musher_id"])
        denom = topn if topn > 0 else 1
        p_at_n = len(true_set & pred_set) / denom
        print(f"\nPrecision@{topn}: {p_at_n:.2f} (target={args.target})")

    # --- Show top predictions (match backtest options) ---
    if args.show_topn != "none":
        if args.show_topn in {"all", "both"}:
            _print_topn(d_test, args.target, p_label, topn, label_only=False, test_year=args.test_year)
        if args.show_topn in {"labeled", "both"}:
            _print_topn(d_test, args.target, p_label, topn, label_only=True, test_year=args.test_year)

    # Coeff magnitudes (base logistic regression only)
    try:
        clf = base_model.named_steps["clf"]
        coefs = pd.Series(clf.coef_[0], index=MODEL_COLS).sort_values(key=lambda s: s.abs(), ascending=False)
        print("\nTop coefficient magnitudes (not causal, just directional):")
        print(coefs.head(20).to_string())
    except Exception:
        pass


if __name__ == "__main__":
    main()

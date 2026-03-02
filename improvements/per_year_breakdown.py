"""
per_year_breakdown.py
=====================
Compare ORIG (10/25/40/25) vs top10 ridge (28/26/29/17) weights
year-by-year to see where the improvement comes from.

Usage:
    python improvements/per_year_breakdown.py
    python improvements/per_year_breakdown.py --start_year 2010
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

from src.db import connect


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


def build_data(con, year_min, year_max):
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

        mask = tr["top5"].notna()
        model_win = _make_calibrated_model(class_weight="balanced")
        model_win.fit(tr.loc[mask, win_mcols], tr.loc[mask, "top5"].astype(int))
        p_top5_for_win = model_win.predict_proba(te[win_mcols])[:, 1]
        year_preds["p_won"] = derive_win_probability(p_top5_for_win)

        for target, col_name in [("top5", "p_top5"), ("top10", "p_top10"), ("finished", "p_finished")]:
            mask = tr[target].notna()
            cw = "balanced" if target != "finished" else None
            model = _make_calibrated_model(class_weight=cw)
            model.fit(tr.loc[mask, rank_mcols], tr.loc[mask, target].astype(int))
            year_preds[col_name] = model.predict_proba(te[rank_mcols])[:, 1]

        for target in ["won", "top5", "top10", "finished"]:
            year_preds[f"rank_{target}"] = year_preds[f"p_{target}"].rank(ascending=False, method="min")

        all_preds.append(year_preds)

    return pd.concat(all_preds, ignore_index=True)


def per_year_metrics(grp, col):
    if len(grp) < 5:
        return None
    rho, _ = spearmanr(grp[col], grp["finish_place"])

    pred_t5 = set(grp.nsmallest(5, col)["musher_id"])
    actual_t5 = set(grp.nsmallest(5, "finish_place")["musher_id"])
    p5 = len(pred_t5 & actual_t5) / 5

    pred_t10 = set(grp.nsmallest(10, col)["musher_id"])
    actual_t10 = set(grp.nsmallest(10, "finish_place")["musher_id"])
    p10 = len(pred_t10 & actual_t10) / 10

    winner = grp[grp["finish_place"] == 1]
    if not winner.empty:
        all_ranks = grp[col].rank(ascending=True, method="min")
        w_rank = int(all_ranks.loc[winner.index[0]])
    else:
        w_rank = None

    return {"spearman": rho, "P@5": p5, "P@10": p10, "winner_rank": w_rank}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_start", type=int, default=2006)
    ap.add_argument("--train_end", type=int, default=2025)
    ap.add_argument("--start_year", type=int, default=2010)
    args = ap.parse_args()

    con = connect()
    df = build_data(con, args.train_start, args.train_end)
    print(f"Loaded {len(df)} musher-years ({args.train_start}-{args.train_end})")

    print(f"\nRunning LOOCV from {args.start_year} to {args.train_end}...")
    preds = run_loocv(df, args.start_year, args.train_end)
    print(f"Collected {len(preds)} predictions across {preds['year'].nunique()} years")

    valid = preds[preds["finish_place"].notna()].copy()
    valid["finish_place"] = pd.to_numeric(valid["finish_place"], errors="coerce")
    valid = valid[valid["finish_place"].notna()].copy()

    # ORIG weights
    valid["comp_orig"] = (
        0.10 * valid["rank_won"]
        + 0.25 * valid["rank_top5"]
        + 0.40 * valid["rank_top10"]
        + 0.25 * valid["rank_finished"]
    )

    # top10 ridge weights
    valid["comp_ridge"] = (
        0.282 * valid["rank_won"]
        + 0.263 * valid["rank_top5"]
        + 0.286 * valid["rank_top10"]
        + 0.169 * valid["rank_finished"]
    )

    print(f"\n{'='*105}")
    print(f"PER-YEAR BREAKDOWN: ORIG (10/25/40/25) vs RIDGE-top10 (28/26/29/17)")
    print(f"{'='*105}")
    print(f"{'Year':>6} | {'Sp_O':>6} {'Sp_R':>6} {'D':>6} | "
          f"{'P5_O':>5} {'P5_R':>5} {'D':>5} | "
          f"{'P10_O':>5} {'P10_R':>5} {'D':>5} | "
          f"{'WinR_O':>6} {'WinR_R':>6} | {'Better':>7}")
    print("-" * 105)

    years = sorted(valid["year"].unique())
    orig_wins = 0
    ridge_wins = 0
    ties = 0

    for yr in years:
        grp = valid[valid["year"] == yr]
        o = per_year_metrics(grp, "comp_orig")
        r = per_year_metrics(grp, "comp_ridge")
        if o is None or r is None:
            continue

        d_sp = r["spearman"] - o["spearman"]
        d_p5 = r["P@5"] - o["P@5"]
        d_p10 = r["P@10"] - o["P@10"]

        wr_o = f"#{o['winner_rank']}" if o["winner_rank"] else "N/A"
        wr_r = f"#{r['winner_rank']}" if r["winner_rank"] else "N/A"

        ridge_better = sum([d_sp > 0.001, d_p5 > 0.001, d_p10 > 0.001])
        orig_better = sum([d_sp < -0.001, d_p5 < -0.001, d_p10 < -0.001])
        if ridge_better > orig_better:
            winner = "RIDGE"
            ridge_wins += 1
        elif orig_better > ridge_better:
            winner = "ORIG"
            orig_wins += 1
        else:
            winner = "tie"
            ties += 1

        d_sp_s = f"{'+' if d_sp > 0 else ''}{d_sp:.3f}" if abs(d_sp) > 0.001 else "    = "
        d_p5_s = f"{'+' if d_p5 > 0 else ''}{d_p5:.2f}" if abs(d_p5) > 0.001 else "   = "
        d_p10_s = f"{'+' if d_p10 > 0 else ''}{d_p10:.2f}" if abs(d_p10) > 0.001 else "   = "

        print(f"{yr:>6} | {o['spearman']:>6.3f} {r['spearman']:>6.3f} {d_sp_s:>6} | "
              f"{o['P@5']:>5.2f} {r['P@5']:>5.2f} {d_p5_s:>5} | "
              f"{o['P@10']:>5.2f} {r['P@10']:>5.2f} {d_p10_s:>5} | "
              f"{wr_o:>6} {wr_r:>6} | {winner:>7}")

    print("-" * 105)
    print(f"Ridge better: {ridge_wins} yrs | ORIG better: {orig_wins} yrs | Tied: {ties} yrs")

    # Averages by era
    print(f"\n{'='*75}")
    print("AVERAGES BY ERA")
    print(f"{'='*75}")
    for label, yr_min, yr_max in [("Early (2010-2014)", 2010, 2014),
                                   ("Mid   (2015-2019)", 2015, 2019),
                                   ("Recent(2020-2025)", 2020, 2025)]:
        era = valid[(valid["year"] >= yr_min) & (valid["year"] <= yr_max)]
        if era.empty:
            continue

        o_sp, r_sp, o_p5, r_p5, o_p10, r_p10 = [], [], [], [], [], []
        for yr in sorted(era["year"].unique()):
            grp = era[era["year"] == yr]
            o = per_year_metrics(grp, "comp_orig")
            r = per_year_metrics(grp, "comp_ridge")
            if o and r:
                o_sp.append(o["spearman"]); r_sp.append(r["spearman"])
                o_p5.append(o["P@5"]); r_p5.append(r["P@5"])
                o_p10.append(o["P@10"]); r_p10.append(r["P@10"])

        n = len(o_sp)
        if n == 0:
            continue
        print(f"\n{label} ({n} years):")
        print(f"  Spearman:  ORIG={np.mean(o_sp):.3f}  RIDGE={np.mean(r_sp):.3f}  delta={np.mean(r_sp)-np.mean(o_sp):+.3f}")
        print(f"  P@5:       ORIG={np.mean(o_p5):.3f}  RIDGE={np.mean(r_p5):.3f}  delta={np.mean(r_p5)-np.mean(o_p5):+.3f}")
        print(f"  P@10:      ORIG={np.mean(o_p10):.3f}  RIDGE={np.mean(r_p10):.3f}  delta={np.mean(r_p10)-np.mean(o_p10):+.3f}")


if __name__ == "__main__":
    main()
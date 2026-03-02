"""
backtest_win_pct_vs_composite.py
================================
For each backtest year, shows:
  - Who the composite ranked #1
  - Who had the highest win%
  - What each actually finished
  - Who actually won

Usage:
    python backtest_win_pct_vs_composite.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.calibration import CalibratedClassifierCV

from src.db import connect


# ---- Copy key functions from predict_prerace_2026.py ----

rank_features = [
    "w_avg_finish_place", "w_pct_top10", "w_pct_finished", "pct_top5",
    "w_avg_time_behind_winner_seconds", "n_finishes", "w_n_entries",
    "years_since_last_entry", "is_rookie", "last_year_finish_place",
]

win_features = [
    "w_avg_finish_place", "w_pct_top10", "w_pct_finished", "pct_top5",
    "w_avg_time_behind_winner_seconds", "n_finishes", "w_n_entries",
    "years_since_last_entry", "is_rookie",
]

all_features = list(dict.fromkeys(rank_features + win_features))


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


def main():
    con = connect()

    # Build full training data
    ms = con.execute(
        f"SELECT year, musher_id, {', '.join(all_features)} "
        f"FROM musher_strength WHERE year BETWEEN 2006 AND 2025"
    ).df()
    hr = con.execute(
        "SELECT year, musher_id, finish_place, status "
        "FROM historical_results WHERE year BETWEEN 2006 AND 2025"
    ).df()
    names = con.execute(
        "SELECT musher_id, name_canonical AS name FROM mushers"
    ).df()

    df_all = ms.merge(hr, on=["year", "musher_id"], how="inner")
    df_all = df_all.merge(names, on="musher_id", how="left")

    df_all["won"] = (df_all["finish_place"] == 1).astype("Int64")
    df_all["top5"] = (df_all["finish_place"].notna() & (df_all["finish_place"] <= 5)).astype("Int64")
    df_all["top10"] = (df_all["finish_place"].notna() & (df_all["finish_place"] <= 10)).astype("Int64")
    df_all["finished"] = (df_all["finish_place"].notna()).astype("Int64")

    # Clean features
    df_all[all_features] = df_all[all_features].apply(pd.to_numeric, errors="coerce")
    for f in all_features:
        df_all[f"{f}_missing"] = df_all[f].isna().astype("int64")

    win_model_cols = _model_cols(win_features)
    rank_model_cols = _model_cols(rank_features)

    bt_years = list(range(2015, 2026))

    print("=" * 110)
    print("COMPOSITE #1 vs WIN% LEADER — Backtest 2015–2025")
    print("=" * 110)
    print(f"{'Year':>4} | {'Composite #1':<22} {'Fin':>4} | {'Win% Leader':<22} {'Win%':>6} {'Fin':>4} | {'Actual Winner':<22} {'CR':>3} {'WR':>3}")
    print("-" * 110)

    comp_correct = 0
    winpct_correct = 0
    comp_winner_ranks = []
    winpct_winner_ranks = []

    for test_yr in bt_years:
        bt_tr = df_all[df_all["year"] < test_yr].copy()
        bt_te = df_all[df_all["year"] == test_yr].copy()
        if bt_tr.empty or bt_te.empty:
            continue

        # Train models
        yr_probs = {}
        for target, mcols, key in [
            ("top5", win_model_cols, "top5_win"),
            ("top5", rank_model_cols, "top5"),
            ("top10", rank_model_cols, "top10"),
            ("finished", rank_model_cols, "finished"),
        ]:
            mask_tr = bt_tr[target].notna()
            y_tr = bt_tr.loc[mask_tr, target].astype(int)
            if y_tr.nunique() < 2:
                continue
            cw = "balanced" if target != "finished" else None
            model = _make_calibrated_model(class_weight=cw)
            model.fit(bt_tr.loc[mask_tr, mcols], y_tr)
            mask_te = bt_te[target].notna()
            if mask_te.sum() == 0:
                continue
            p = model.predict_proba(bt_te.loc[mask_te, mcols])[:, 1]
            yr_probs[key] = p

        if not all(k in yr_probs for k in ["top5_win", "top5", "top10", "finished"]):
            continue

        te = bt_te.copy()
        te["p_won"] = derive_win_probability(yr_probs["top5_win"])
        te["p_top5"] = yr_probs["top5"]
        te["p_top10"] = yr_probs["top10"]
        te["p_finished"] = yr_probs["finished"]

        # Composite ranking
        for t in ["won", "top5", "top10", "finished"]:
            te[f"rank_{t}"] = te[f"p_{t}"].rank(ascending=False, method="min")
        te["composite_rank"] = (
            0.10 * te["rank_won"]
            + 0.25 * te["rank_top5"]
            + 0.40 * te["rank_top10"]
            + 0.25 * te["rank_finished"]
        )
        te = te.sort_values("composite_rank").reset_index(drop=True)
        te["predicted_rank"] = range(1, len(te) + 1)

        # Win% ranking
        te["winpct_rank"] = te["p_won"].rank(ascending=False, method="min").astype(int)

        # Find key mushers
        comp_1 = te.iloc[0]
        winpct_1 = te.loc[te["winpct_rank"] == 1].iloc[0]

        actual_winner = te.loc[te["finish_place"] == 1]
        if actual_winner.empty:
            continue
        actual_winner = actual_winner.iloc[0]

        comp_1_finish = int(comp_1["finish_place"]) if pd.notna(comp_1["finish_place"]) else "DNF"
        winpct_1_finish = int(winpct_1["finish_place"]) if pd.notna(winpct_1["finish_place"]) else "DNF"

        winner_comp_rank = int(actual_winner["predicted_rank"])
        winner_winpct_rank = int(actual_winner["winpct_rank"])

        comp_winner_ranks.append(winner_comp_rank)
        winpct_winner_ranks.append(winner_winpct_rank)

        if comp_1_finish == 1:
            comp_correct += 1
        if winpct_1_finish == 1:
            winpct_correct += 1

        # Check if they're the same musher
        same = " ← same" if comp_1["musher_id"] == winpct_1["musher_id"] else ""

        print(f"{test_yr:>4} | {comp_1['name']:<22} {str(comp_1_finish):>4} | "
              f"{winpct_1['name']:<22} {winpct_1['p_won']*100:>5.1f}% {str(winpct_1_finish):>4} | "
              f"{actual_winner['name']:<22} {winner_comp_rank:>3} {winner_winpct_rank:>3}{same}")

    print("-" * 110)
    n = len(comp_winner_ranks)
    print(f"\nComposite #1 correct:  {comp_correct}/{n} ({comp_correct/n*100:.0f}%)")
    print(f"Win% #1 correct:       {winpct_correct}/{n} ({winpct_correct/n*100:.0f}%)")
    print(f"\nWinner's composite rank:  {comp_winner_ranks}")
    print(f"Winner's win% rank:       {winpct_winner_ranks}")
    print(f"\nComposite — winner in top 3: {sum(1 for r in comp_winner_ranks if r <= 3)}/{n}")
    print(f"Win%      — winner in top 3: {sum(1 for r in winpct_winner_ranks if r <= 3)}/{n}")
    print(f"Composite — winner in top 5: {sum(1 for r in comp_winner_ranks if r <= 5)}/{n}")
    print(f"Win%      — winner in top 5: {sum(1 for r in winpct_winner_ranks if r <= 5)}/{n}")

    # Also show top 5 by win% for each year
    print(f"\n\n{'='*80}")
    print("TOP 5 BY WIN% EACH YEAR")
    print(f"{'='*80}")

    for test_yr in bt_years:
        bt_tr = df_all[df_all["year"] < test_yr].copy()
        bt_te = df_all[df_all["year"] == test_yr].copy()
        if bt_tr.empty or bt_te.empty:
            continue

        yr_probs = {}
        for target, mcols, key in [
            ("top5", win_model_cols, "top5_win"),
            ("top5", rank_model_cols, "top5"),
            ("top10", rank_model_cols, "top10"),
            ("finished", rank_model_cols, "finished"),
        ]:
            mask_tr = bt_tr[target].notna()
            y_tr = bt_tr.loc[mask_tr, target].astype(int)
            if y_tr.nunique() < 2:
                continue
            cw = "balanced" if target != "finished" else None
            model = _make_calibrated_model(class_weight=cw)
            model.fit(bt_tr.loc[mask_tr, mcols], y_tr)
            mask_te = bt_te[target].notna()
            if mask_te.sum() == 0:
                continue
            p = model.predict_proba(bt_te.loc[mask_te, mcols])[:, 1]
            yr_probs[key] = p

        if not all(k in yr_probs for k in ["top5_win", "top5", "top10", "finished"]):
            continue

        te = bt_te.copy()
        te["p_won"] = derive_win_probability(yr_probs["top5_win"])
        te = te.sort_values("p_won", ascending=False).reset_index(drop=True)

        actual_winner_name = te.loc[te["finish_place"] == 1, "name"].iloc[0] if (te["finish_place"] == 1).any() else "?"

        print(f"\n{test_yr} (winner: {actual_winner_name}):")
        for i, (_, row) in enumerate(te.head(5).iterrows(), 1):
            fp = int(row["finish_place"]) if pd.notna(row["finish_place"]) else "DNF"
            won_tag = " ★" if fp == 1 else ""
            print(f"  {i}. {row['name']:<22} Win%: {row['p_won']*100:>5.1f}%  Finished: {fp}{won_tag}")


if __name__ == "__main__":
    main()
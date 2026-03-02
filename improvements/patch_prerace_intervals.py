"""
patch_prerace_intervals.py
===========================
Adds Monte Carlo prediction intervals to the pre-race rankings.

Instead of just "Rank #8, Vol: 61.3", mushers now get:
  "Rank #8, 80% CI [#2, #18], σ=1.80"

This makes volatility go from a passive display metric to something
backed by actual simulation. Uses the same structural uncertainty
multipliers as the in-race model (#5).

APPROACH:
  For each of 10,000 simulations:
    1. Perturb each musher's probabilities on the logit scale
       (noise scaled by per-musher uncertainty multiplier)
    2. Re-rank and re-compute composite
    3. Record simulated finishing rank

  This produces a distribution of ranks for each musher.
  Veterans get tight intervals, rookies get wide ones.

Usage:
    python improvements/patch_prerace_intervals.py          # show
    python improvements/patch_prerace_intervals.py --apply   # auto-apply
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
import shutil


def try_replace(text, old, new):
    if old in text:
        return text.replace(old, new), True
    old_crlf = old.replace("\n", "\r\n")
    if old_crlf in text:
        new_crlf = new.replace("\n", "\r\n")
        return text.replace(old_crlf, new_crlf), True
    return text, False


PREDICT_FILE = Path("predict_prerace_2026.py")

# ---- PATCH 1: Add MC simulation after volatility, before composite ranking ----
# Insert the MC block AFTER the composite ranking and predicted_rank assignment

MC_INSERT_AFTER = '''\
    results["rookie"] = results["musher_id"].isin(rookie_ids).map({True: "R", False: ""})'''

MC_BLOCK = '''

    # ---- Pre-race Monte Carlo: prediction intervals on rank ----
    # Perturbs probabilities on logit scale with per-musher noise,
    # re-ranks across 10k sims to produce rank distributions.
    from scipy.stats import rankdata as _rankdata

    n_sims_prerace = 10000
    rng = np.random.default_rng(42)

    # Per-musher uncertainty multiplier (same as in-race #5)
    _n_fin = pd.to_numeric(df_2026.get("n_finishes", 0), errors="coerce").fillna(0).values
    _is_rook = pd.to_numeric(df_2026.get("is_rookie", 0), errors="coerce").fillna(0).values
    _ysl = pd.to_numeric(df_2026.get("years_since_last_entry", 0), errors="coerce").fillna(0).values

    unc_mult = np.sqrt(8.0 / np.maximum(_n_fin, 1.0))
    unc_mult = np.where(_is_rook == 1, 1.8, unc_mult)
    _absence_add = np.minimum(np.maximum(_ysl - 1, 0) * 0.05, 0.4)
    unc_mult = np.clip(unc_mult + _absence_add, 0.85, 2.0)
    results["unc_mult"] = unc_mult

    def _logit(p):
        p = np.clip(p, 1e-6, 1 - 1e-6)
        return np.log(p / (1 - p))

    def _sigmoid(x):
        return 1.0 / (1.0 + np.exp(-x))

    _logit_won = _logit(results["p_won"].values)
    _logit_top5 = _logit(results["p_top5"].values)
    _logit_top10 = _logit(results["p_top10"].values)
    _logit_fin = _logit(results["p_finished"].values)

    n_mushers = len(results)
    _logit_noise_scale = 0.5 * unc_mult
    _sim_ranks = np.zeros((n_sims_prerace, n_mushers), dtype=int)

    for _sim in range(n_sims_prerace):
        _noise = rng.normal(0, 1, (4, n_mushers)) * _logit_noise_scale[None, :]
        _sp_won = _sigmoid(_logit_won + _noise[0])
        _sp_top5 = _sigmoid(_logit_top5 + _noise[1])
        _sp_top10 = _sigmoid(_logit_top10 + _noise[2])
        _sp_fin = _sigmoid(_logit_fin + _noise[3])

        _r_won = _rankdata(-_sp_won, method="min")
        _r_top5 = _rankdata(-_sp_top5, method="min")
        _r_top10 = _rankdata(-_sp_top10, method="min")
        _r_fin = _rankdata(-_sp_fin, method="min")

        _comp = 0.10 * _r_won + 0.25 * _r_top5 + 0.40 * _r_top10 + 0.25 * _r_fin
        _sim_ranks[_sim] = _rankdata(_comp, method="min")

    results["rank_p10"] = np.percentile(_sim_ranks, 10, axis=0).astype(int)
    results["rank_p90"] = np.percentile(_sim_ranks, 90, axis=0).astype(int)
    results["rank_ci"] = [
        f"[{lo},{hi}]" for lo, hi in zip(results["rank_p10"], results["rank_p90"])
    ]

    print(f"\\n  Pre-race Monte Carlo: {n_sims_prerace} sims, "
          f"unc_mult range [{unc_mult.min():.2f}, {unc_mult.max():.2f}]")'''


# ---- PATCH 2: Update display table to include 80% CI ----

DISPLAY_OLD = '''\
    display = results[[
        "predicted_rank", "name", "rookie",
        "p_won", "p_top5", "p_top10", "p_finished", "volatility",
    ]].copy()
    display.columns = ["Rank", "Musher", "R", "Win%", "Top5%", "Top10%", "Finish%", "Vol"]
    for col in ["Win%", "Top5%", "Top10%", "Finish%"]:
        display[col] = (display[col] * 100).round(1)'''

DISPLAY_NEW = '''\
    display = results[[
        "predicted_rank", "name", "rookie",
        "p_won", "p_top5", "p_top10", "p_finished", "rank_ci", "unc_mult",
    ]].copy()
    display.columns = ["Rank", "Musher", "R", "Win%", "Top5%", "Top10%", "Finish%", "80% CI", "Unc"]
    for col in ["Win%", "Top5%", "Top10%", "Finish%"]:
        display[col] = (display[col] * 100).round(1)
    display["Unc"] = display["Unc"].round(2)'''


# ---- PATCH 3: Update header description ----

HEADER_OLD = '''\
    print(f"  Volatility:  composite uncertainty score (0-100, higher = wider outcome range)")'''

HEADER_NEW = '''\
    print(f"  80% CI:      prediction interval on rank from {n_sims_prerace}-sim Monte Carlo")
    print(f"  Unc:         per-musher uncertainty multiplier (1.0 = baseline, 2.0 = max)")'''


# ---- PATCH 4: Update top-5 display to show CI ----

TOP5_OLD = '''\
              f"Vol: {row[\'volatility\']}  [field rank: #{field_rank}]")'''

TOP5_NEW = '''\
              f"80% CI: {row[\'rank_ci\']}  [field rank: #{field_rank}]")'''


# ---- PATCH 5: Update volatility wildcards to uncertainty wildcards ----

WILD_OLD = '''\
    # Highest volatility mushers
    vol_top = results.nlargest(5, "volatility")
    print(f"\\nHighest Volatility (biggest wildcards):")
    for _, row in vol_top.iterrows():
        tag = " (R)" if row["rookie"] == "R" else ""
        print(f"  #{int(row[\'predicted_rank\'])} {row[\'name\']}{tag} \\u2014 "
              f"Vol: {row[\'volatility\']}, Win: {row[\'p_won\']*100:.1f}%, "
              f"Top10: {row[\'p_top10\']*100:.1f}%")'''

WILD_NEW = '''\
    # Highest uncertainty mushers (widest prediction intervals)
    wild_top = results.nlargest(5, "unc_mult")
    print(f"\\nBiggest Wildcards (widest prediction intervals):")
    for _, row in wild_top.iterrows():
        tag = " (R)" if row["rookie"] == "R" else ""
        print(f"  #{int(row[\'predicted_rank\'])} {row[\'name\']}{tag} \\u2014 "
              f"80% CI: {row[\'rank_ci\']}, Unc: {row[\'unc_mult\']:.2f}, "
              f"Win: {row[\'p_won\']*100:.1f}%, Top10: {row[\'p_top10\']*100:.1f}%")'''


# ---- PATCH 6: Update rookie display ----

ROOKIE_OLD = '''\
              f"Finish: {r[\'p_finished\']*100:.1f}%, Vol: {r[\'volatility\']}")'''

ROOKIE_NEW = '''\
              f"Finish: {r[\'p_finished\']*100:.1f}%, 80% CI: {r[\'rank_ci\']}")'''


# ---- PATCH 7: Update markdown table ----

MD_OLD = '''\
    print("| Rank | Musher | Win% | Top 5% | Top 10% | Finish% | Volatility |")
    print("|------|--------|------|--------|---------|---------|------------|")
    for _, row in display.iterrows():
        r_tag = " \\U0001f539" if row["R"] == "R" else ""
        print(f"| {int(row[\'Rank\'])} | {row[\'Musher\']}{r_tag} | "
              f"{row[\'Win%\']}% | {row[\'Top5%\']}% | {row[\'Top10%\']}% | "
              f"{row[\'Finish%\']}% | {row[\'Vol\']} |")'''

MD_NEW = '''\
    print("| Rank | Musher | Win% | Top 5% | Top 10% | Finish% | 80% CI | Unc |")
    print("|------|--------|------|--------|---------|---------|--------|-----|")
    for _, row in display.iterrows():
        r_tag = " \\U0001f539" if row["R"] == "R" else ""
        print(f"| {int(row[\'Rank\'])} | {row[\'Musher\']}{r_tag} | "
              f"{row[\'Win%\']}% | {row[\'Top5%\']}% | {row[\'Top10%\']}% | "
              f"{row[\'Finish%\']}% | {row[\'80% CI\']} | {row[\'Unc\']} |")'''


def apply_patches():
    if not PREDICT_FILE.exists():
        print(f"ERROR: {PREDICT_FILE} not found. Run from project root.")
        return False

    backup = PREDICT_FILE.with_suffix(".py.bak_intervals")
    shutil.copy2(PREDICT_FILE, backup)
    print(f"Backed up to {backup}")

    text = PREDICT_FILE.read_text(encoding="utf-8")

    patches = [
        ("MC simulation block", MC_INSERT_AFTER, MC_BLOCK, "insert"),
        ("Display table columns", DISPLAY_OLD, DISPLAY_NEW, "replace"),
        ("Header description", HEADER_OLD, HEADER_NEW, "replace"),
        ("Top-5 display", TOP5_OLD, TOP5_NEW, "replace"),
        ("Wildcards section", WILD_OLD, WILD_NEW, "replace"),
        ("Rookie display", ROOKIE_OLD, ROOKIE_NEW, "replace"),
        ("Markdown table", MD_OLD, MD_NEW, "replace"),
    ]

    applied = 0
    for label, old, new, mode in patches:
        if mode == "insert":
            text, ok = try_replace(text, old, old + new)
        else:
            text, ok = try_replace(text, old, new)

        if ok:
            print(f"  Applied: {label}")
            applied += 1
        else:
            print(f"  WARNING: Could not match: {label}")

    PREDICT_FILE.write_text(text, encoding="utf-8")
    print(f"\n{applied}/{len(patches)} patches applied to {PREDICT_FILE}")
    print(f"Backup at {backup}")

    return applied == len(patches)


def show_patches():
    print("=" * 70)
    print("PATCHES: Pre-race Monte Carlo prediction intervals")
    print("=" * 70)
    print(f"\n7 patches to {PREDICT_FILE}:")
    print(f"  1. Add MC simulation (10k sims, logit-scale perturbation)")
    print(f"  2. Update display table (Vol -> 80% CI + Unc)")
    print(f"  3. Update header description")
    print(f"  4. Update top-5 display (Vol -> CI)")
    print(f"  5. Update wildcards section (volatility -> uncertainty)")
    print(f"  6. Update rookie display")
    print(f"  7. Update markdown table")
    print(f"\nRun with --apply to auto-patch.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true")
    args = ap.parse_args()

    if args.apply:
        apply_patches()
    else:
        show_patches()


if __name__ == "__main__":
    main()
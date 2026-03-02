"""
patch_dual_composite.py
========================
Patches predict_prerace_2026.py to use dual composite rankings:

  1. CONTENDER composite (28/26/29/17) — optimized for top-of-field separation.
     Used for: "Predicted Top 5", winner identification, headline picks.

  2. FIELD composite (10/25/40/25) — optimized for full-field ranking accuracy.
     Used for: overall ranking table, mid/back-of-pack ordering.

The insight: different weight sets are optimal for different parts of the field.
Among top-10 contenders, win/top5/top10 probs all discriminate roughly equally.
Among mid-pack mushers, finish probability is the only meaningful signal.

Usage:
    python improvements/patch_dual_composite.py

    This prints the patched sections. Apply them manually to predict_prerace_2026.py,
    or use the --apply flag to auto-patch (creates a backup first):

    python improvements/patch_dual_composite.py --apply
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
import shutil

PREDICT_FILE = Path("predict_prerace_2026.py")

# The old composite block (lines ~405-417 in original)
OLD_COMPOSITE = '''    results["composite_rank"] = (
        0.10 * results["rank_won"]
        + 0.25 * results["rank_top5"]
        + 0.40 * results["rank_top10"]
        + 0.25 * results["rank_finished"]
    )

    results = results.sort_values("composite_rank").reset_index(drop=True)
    results["predicted_rank"] = range(1, len(results) + 1)'''

NEW_COMPOSITE = '''    # ---- Dual composite ranking ----
    # CONTENDER weights (ridge-learned on top-10 finishers, 16yr backtest):
    #   Optimized for separating contenders from each other.
    #   Win/Top5/Top10 carry roughly equal weight; Finish matters less
    #   because top-10 mushers nearly all finish.
    results["contender_rank"] = (
        0.282 * results["rank_won"]
        + 0.263 * results["rank_top5"]
        + 0.286 * results["rank_top10"]
        + 0.169 * results["rank_finished"]
    )

    # FIELD weights (original hand-tuned):
    #   Optimized for full-field ranking accuracy.
    #   Heavier on Top10 and Finish because those are the only signals
    #   that meaningfully separate mid-pack from back-of-pack.
    results["field_rank"] = (
        0.10 * results["rank_won"]
        + 0.25 * results["rank_top5"]
        + 0.40 * results["rank_top10"]
        + 0.25 * results["rank_finished"]
    )

    # Primary sort: use field composite for overall ranking table
    # (contender composite used separately for top-5 picks below)
    results["composite_rank"] = results["field_rank"]

    results = results.sort_values("composite_rank").reset_index(drop=True)
    results["predicted_rank"] = range(1, len(results) + 1)

    # Contender ranking: separate sort for top-of-field identification
    results["contender_predicted_rank"] = results["contender_rank"].rank(ascending=True, method="min").astype(int)'''

# Old top-5 display block
OLD_TOP5 = '''    print(f"\\nPredicted Top 5:")
    for _, row in results.head(5).iterrows():
        tag = " (R)" if row["rookie"] == "R" else ""
        print(f"  {int(row[\'predicted_rank\'])}. {row[\'name\']}{tag} \\u2014 "
              f"Win: {row[\'p_won\']*100:.1f}%, Top10: {row[\'p_top10\']*100:.1f}%, "
              f"Vol: {row[\'volatility\']}")'''

NEW_TOP5 = '''    # Top 5 using CONTENDER composite (optimized for top-of-field)
    contender_top5 = results.nsmallest(5, "contender_rank")
    print(f"\\nPredicted Top 5 (contender composite: 28/26/29/17):")
    for i, (_, row) in enumerate(contender_top5.iterrows(), 1):
        tag = " (R)" if row["rookie"] == "R" else ""
        field_rank = int(row["predicted_rank"])
        print(f"  {i}. {row[\'name\']}{tag} \\u2014 "
              f"Win: {row[\'p_won\']*100:.1f}%, Top10: {row[\'p_top10\']*100:.1f}%, "
              f"Vol: {row[\'volatility\']}  [field rank: #{field_rank}]")

    # Show where contender and field rankings disagree
    field_top5 = set(results.head(5)["musher_id"])
    contender_top5_ids = set(contender_top5["musher_id"])
    if field_top5 != contender_top5_ids:
        only_contender = contender_top5_ids - field_top5
        only_field = field_top5 - contender_top5_ids
        if only_contender or only_field:
            print(f"\\n  Ranking disagreements (contender vs field composite):")
            for mid in only_contender:
                r = results[results["musher_id"] == mid].iloc[0]
                print(f"    {r[\'name\']}: contender top-5, field #{int(r[\'predicted_rank\'])}")
            for mid in only_field:
                r = results[results["musher_id"] == mid].iloc[0]
                c_rank = int(r["contender_predicted_rank"])
                print(f"    {r[\'name\']}: field top-5, contender #{c_rank}")'''

# Old backtest composite block
OLD_BT_COMPOSITE = '''            te["composite_rank"] = (
                0.10 * te["rank_won"]
                + 0.25 * te["rank_top5"]
                + 0.40 * te["rank_top10"]
                + 0.25 * te["rank_finished"]
            )'''

NEW_BT_COMPOSITE = '''            te["composite_rank"] = (
                0.10 * te["rank_won"]
                + 0.25 * te["rank_top5"]
                + 0.40 * te["rank_top10"]
                + 0.25 * te["rank_finished"]
            )
            te["contender_rank"] = (
                0.282 * te["rank_won"]
                + 0.263 * te["rank_top5"]
                + 0.286 * te["rank_top10"]
                + 0.169 * te["rank_finished"]
            )'''


def show_patches():
    print("=" * 70)
    print("PATCH 1: Replace composite ranking block (~line 409)")
    print("=" * 70)
    print("\nOLD:")
    print(OLD_COMPOSITE)
    print("\nNEW:")
    print(NEW_COMPOSITE)

    print("\n" + "=" * 70)
    print("PATCH 2: Replace 'Predicted Top 5' display (~line 451)")
    print("=" * 70)
    print("\nReplace the 'Predicted Top 5' for loop with the contender-composite")
    print("version that also shows field rank and disagreements.")
    print("(See NEW_TOP5 in this script)")

    print("\n" + "=" * 70)
    print("PATCH 3: Add contender_rank to backtest composite (~line 309)")
    print("=" * 70)
    print("\nAfter the existing composite_rank in the backtest loop,")
    print("add contender_rank so you can evaluate both in backtest too.")

    print("\n" + "=" * 70)
    print("PATCH 4: Add contender_predicted_rank to CSV output (~line 480)")
    print("=" * 70)
    print('\nAdd "contender_predicted_rank" to the out_df column list.')


def apply_patches():
    if not PREDICT_FILE.exists():
        print(f"ERROR: {PREDICT_FILE} not found. Run from project root.")
        return False

    # Backup
    backup = PREDICT_FILE.with_suffix(".py.bak")
    shutil.copy2(PREDICT_FILE, backup)
    print(f"Backed up to {backup}")

    text = PREDICT_FILE.read_text(encoding="utf-8")

    # Patch 1: composite ranking
    old_comp_needle = '0.10 * results["rank_won"]'
    if old_comp_needle not in text:
        print("WARNING: Could not find composite ranking block. Skipping patch 1.")
    else:
        # Find the full block
        text = text.replace(
            '''    results["composite_rank"] = (
        0.10 * results["rank_won"]
        + 0.25 * results["rank_top5"]
        + 0.40 * results["rank_top10"]
        + 0.25 * results["rank_finished"]
    )

    results = results.sort_values("composite_rank").reset_index(drop=True)
    results["predicted_rank"] = range(1, len(results) + 1)''',
            NEW_COMPOSITE
        )
        print("Applied patch 1: dual composite ranking")

    # Patch 2: top-5 display
    old_top5_needle = 'print(f"\\nPredicted Top 5:")'
    if old_top5_needle not in text:
        print("WARNING: Could not find top-5 display block. Skipping patch 2.")
    else:
        # Replace the top-5 block (5 lines)
        old_block = '''    print(f"\\nPredicted Top 5:")
    for _, row in results.head(5).iterrows():
        tag = " (R)" if row["rookie"] == "R" else ""
        print(f"  {int(row[\'predicted_rank\'])}. {row[\'name\']}{tag} \\u2014 "
              f"Win: {row[\'p_won\']*100:.1f}%, Top10: {row[\'p_top10\']*100:.1f}%, "
              f"Vol: {row[\'volatility\']}")'''
        new_block = '''    # Top 5 using CONTENDER composite (optimized for top-of-field)
    contender_top5 = results.nsmallest(5, "contender_rank")
    print(f"\\nPredicted Top 5 (contender composite: 28/26/29/17):")
    for i, (_, row) in enumerate(contender_top5.iterrows(), 1):
        tag = " (R)" if row["rookie"] == "R" else ""
        field_rank = int(row["predicted_rank"])
        print(f"  {i}. {row[\'name\']}{tag} \\u2014 "
              f"Win: {row[\'p_won\']*100:.1f}%, Top10: {row[\'p_top10\']*100:.1f}%, "
              f"Vol: {row[\'volatility\']}  [field rank: #{field_rank}]")

    # Show where contender and field rankings disagree
    field_top5 = set(results.head(5)["musher_id"])
    contender_top5_ids = set(contender_top5["musher_id"])
    if field_top5 != contender_top5_ids:
        only_contender = contender_top5_ids - field_top5
        only_field = field_top5 - contender_top5_ids
        if only_contender or only_field:
            print(f"\\n  Ranking disagreements (contender vs field composite):")
            for mid in only_contender:
                r = results[results["musher_id"] == mid].iloc[0]
                print(f"    {r[\'name\']}: contender top-5, field #{int(r[\'predicted_rank\'])}")
            for mid in only_field:
                r = results[results["musher_id"] == mid].iloc[0]
                c_rank = int(r["contender_predicted_rank"])
                print(f"    {r[\'name\']}: field top-5, contender #{c_rank}")'''

        if old_block in text:
            text = text.replace(old_block, new_block)
            print("Applied patch 2: contender top-5 display")
        else:
            print("WARNING: Exact top-5 block not matched. Apply manually.")

    # Patch 3: backtest contender_rank
    bt_needle = '0.10 * te["rank_won"]'
    if bt_needle in text:
        text = text.replace(
            '''            te["composite_rank"] = (
                0.10 * te["rank_won"]
                + 0.25 * te["rank_top5"]
                + 0.40 * te["rank_top10"]
                + 0.25 * te["rank_finished"]
            )''',
            '''            te["composite_rank"] = (
                0.10 * te["rank_won"]
                + 0.25 * te["rank_top5"]
                + 0.40 * te["rank_top10"]
                + 0.25 * te["rank_finished"]
            )
            te["contender_rank"] = (
                0.282 * te["rank_won"]
                + 0.263 * te["rank_top5"]
                + 0.286 * te["rank_top10"]
                + 0.169 * te["rank_finished"]
            )'''
        )
        print("Applied patch 3: backtest contender_rank")

    # Patch 4: add to CSV output
    old_csv = '"p_won", "p_top5", "p_top10", "p_finished", "volatility", "composite_rank",'
    new_csv = '"p_won", "p_top5", "p_top10", "p_finished", "volatility", "composite_rank", "contender_predicted_rank",'
    if old_csv in text:
        text = text.replace(old_csv, new_csv)
        print("Applied patch 4: contender_predicted_rank in CSV output")

    PREDICT_FILE.write_text(text, encoding="utf-8")
    print(f"\nDone. Patched {PREDICT_FILE}")
    print(f"Backup at {backup}")
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true",
                    help="Auto-apply patches to predict_prerace_2026.py (creates backup)")
    args = ap.parse_args()

    if args.apply:
        apply_patches()
    else:
        show_patches()
        print("\n\nRun with --apply to auto-patch, or apply manually.")


if __name__ == "__main__":
    main()
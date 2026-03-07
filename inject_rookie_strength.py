"""
inject_rookie_strength.py
=========================
Overwrite musher_strength rows for 2026 rookies with research-informed
baseline values instead of leaving them all as NaN / identical.

Approach:
  1. Start with the **median historical rookie** profile (2012-2025):
       - avg_finish_place ~35, best_finish_place ~28, etc.
  2. Adjust each rookie up/down based on their qualifying-race strength score (1-5).
  3. Write the updated values back into the musher_strength table.

The rookie strength scores are derived from manual research of each rookie's
pre-Iditarod qualifying race results. See:
  - 2026_rookie_research.md
  - historical_rookie_qualifying_profiles.md

Usage:
    python inject_rookie_strength.py          # dry-run (shows changes, doesn't write)
    python inject_rookie_strength.py --write  # actually update the database
"""

import argparse
import sys
from src.db import connect

# ─────────────────────────────────────────────────────────
# 2026 ROOKIE STRENGTH SCORES (from research)
# ─────────────────────────────────────────────────────────
# Score 5: Won/podiumed major 300+ mile qualifier
# Score 4: Top-5 in major 300+ mile qualifier
# Score 3: Top-10 in 300+ mile qualifier OR podium in 200-mile race
# Score 2: Completed qualifying races, no standout results
# Score 1: Prior Iditarod scratch, limited results, or unknown

ROOKIE_SCORES = {
    "1166": 5,   # Jesse Terry — 3x Canadian Challenge champion, 3rd Beargrease, 3rd YQ450
    "1167": 4,   # Kevin Hansen — 2nd Kobuk 440 (behind Iditarod champ Holmes)
    "1168": 2,   # Jody Potts-Joseph — 7th Kobuk 440, heart-and-finish musher, not pushing for top
    "1164": 3,   # Adam Lindenmuth — CB300 + Kobuk + YQ300, won several qualifiers, strong prep
    "1069": 3,   # Richie Beattie — Made it to Nome in 2019 (WD on technicality), real trail experience
    "1163": 2,   # Joseph Sabin — Finished YQA 550, young team
    "1115": 2,   # Jaye Foucher — Mixed Siberian/Alaskan team, not racing aggressively
    "1103": 2,   # Brenda Mackey — Prior exits were dog illness/emergency not ability, 3rd attempt, capable
    "1155": 1,   # Sydnie Bahl — Withdrawn from 2025 Iditarod (Rule 36), second attempt
    "1169": 1,   # Sam Martin — Failor kennel, conservative pacer
    "1170": 1,   # Sam Paperman — Running Turning Heads Kennel puppy team, not racing competitively
    "1171": 1,   # Sadie Lindquist — Running Mitch Seavey puppy team, not racing competitively
}

# ─────────────────────────────────────────────────────────
# HISTORICAL ROOKIE BASELINE (median values from 2012-2025)
# From analyze_rookie_performance.py Part 1 results:
#   - Median finish place: 35
#   - Mean finish place: 37.2
#   - Finish rate: 73.8%
#   - Mean finish percentile: 58.5%
# ─────────────────────────────────────────────────────────

# Baseline: what a "median" rookie (score 3) looks like
BASELINE = {
    "avg_finish_place":        35.0,
    "best_finish_place":       35,    # rookies have only 1 race, so best = avg
    "median_finish_place":     35.0,
    "last3_avg_finish_place":  35.0,
    "last5_avg_finish_place":  35.0,
    "w_avg_finish_place":      35.0,
    "form_last3_avg_finish_place": 35.0,
    "form_last5_avg_finish_place": 35.0,
    "form_w_avg_finish_place":     35.0,
    "n_finishes":              1,     # treat as if they have ~1 prior finish equivalent
    "n_entries":               1,
    "n_years":                 1,
    "pct_finished":            0.74,  # historical rookie finish rate
    "pct_top10":               0.0,
    "pct_top5":                0.0,
    "pct_win":                 0.0,
    "w_pct_top10":             0.0,
    "w_pct_finished":          0.74,
    "last3_pct_top10":         0.0,
    "last5_pct_top10":         0.0,
    "form_last3_pct_top10":    0.0,
    "form_last5_pct_top10":    0.0,
    "form_w_pct_top10":        0.0,
    "cons_pct_finished":       0.74,
    "cons_last5_pct_finished": 0.74,
    "cons_w_pct_finished":     0.74,
    "exp_n_entries":           1.0,
    "exp_n_finishes":          1.0,
    "exp_n_years":             1.0,
    "exp_w_entries":           1.0,
    "peak_best_finish_place":  35.0,
    "peak_pct_top10":          0.0,
    "peak_pct_top5":           0.0,
    "peak_pct_win":            0.0,
    "w_avg_time_behind_winner_seconds": 110000.0,  # ~30.5 hours behind winner (median rookie)
    "w_n_entries":             0.7,   # roughly one entry with decay
    # trajectory features — keep NaN since there's no "prior year" to compare
    # career_race_number stays at 1 (first Iditarod)
}

# ─────────────────────────────────────────────────────────
# SCORE → FINISH PLACE ADJUSTMENT
# Based on historical calibration (see historical_rookie_qualifying_profiles.md):
#   Score 5: ~25th place (elite qualifiers like Ulsom, Holmes, Burke)
#   Score 4: ~28th place (strong qualifiers like Hansen)
#   Score 3: ~33rd place (solid qualifiers — baseline shifted slightly up)
#   Score 2: ~38th place (completed qualifiers, nothing standout)
#   Score 1: ~42nd place (weak/unknown, high scratch risk)
# ─────────────────────────────────────────────────────────

SCORE_TO_FINISH = {
    5: 25,
    4: 28,
    3: 33,
    2: 38,
    1: 42,
}

# Derived adjustments for top-10 probability
SCORE_TO_TOP10_PCT = {
    5: 0.15,   # ~15% chance (strong qualifiers occasionally crack top 10)
    4: 0.08,
    3: 0.03,
    2: 0.00,
    1: 0.00,
}

SCORE_TO_FINISH_PCT = {
    5: 0.90,   # strong qualifiers almost always finish
    4: 0.85,
    3: 0.75,   # near baseline
    2: 0.70,
    1: 0.55,   # high scratch risk
}

# Time behind winner in seconds (calibrated to finish place)
SCORE_TO_TIME_BEHIND = {
    5: 85000,   # ~23.6 hrs — competitive rookie
    4: 95000,   # ~26.4 hrs
    3: 110000,  # ~30.5 hrs — median rookie
    2: 120000,  # ~33.3 hrs
    1: 135000,  # ~37.5 hrs — back of pack
}


def build_rookie_row(musher_id: str, score: int) -> dict:
    """Build an adjusted musher_strength profile for a rookie."""
    finish_place = SCORE_TO_FINISH[score]
    top10_pct = SCORE_TO_TOP10_PCT[score]
    finish_pct = SCORE_TO_FINISH_PCT[score]

    row = dict(BASELINE)  # copy baseline
    row["musher_id"] = musher_id
    row["year"] = 2026

    # Adjust finish place features
    for key in [
        "avg_finish_place", "median_finish_place",
        "last3_avg_finish_place", "last5_avg_finish_place",
        "w_avg_finish_place",
        "form_last3_avg_finish_place", "form_last5_avg_finish_place",
        "form_w_avg_finish_place",
        "peak_best_finish_place",
    ]:
        row[key] = float(finish_place)

    row["best_finish_place"] = finish_place

    # Adjust top-10 / finish probabilities
    for key in [
        "pct_top10", "w_pct_top10",
        "last3_pct_top10", "last5_pct_top10",
        "form_last3_pct_top10", "form_last5_pct_top10",
        "form_w_pct_top10",
        "peak_pct_top10",
    ]:
        row[key] = top10_pct

    for key in [
        "pct_finished", "w_pct_finished",
        "cons_pct_finished", "cons_last5_pct_finished", "cons_w_pct_finished",
    ]:
        row[key] = finish_pct

    # Adjust time behind winner
    row["w_avg_time_behind_winner_seconds"] = float(SCORE_TO_TIME_BEHIND[score])

    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true", help="Actually write to database (default: dry-run)")
    args = ap.parse_args()

    con = connect()

    # Verify these mushers exist in the 2026 entries
    entries_2026 = con.execute(
        "SELECT musher_id FROM entries WHERE year = 2026"
    ).df()

    if entries_2026.empty:
        print("ERROR: No 2026 entries found. Run scrape/build_entries first.")
        sys.exit(1)

    entry_ids = set(entries_2026.musher_id.tolist())

    # Check which rookies are actually in the field
    rookies_in_field = {mid: score for mid, score in ROOKIE_SCORES.items() if mid in entry_ids}
    rookies_missing = {mid: score for mid, score in ROOKIE_SCORES.items() if mid not in entry_ids}

    print(f"2026 field size: {len(entry_ids)}")
    print(f"Rookies in field: {len(rookies_in_field)}")
    if rookies_missing:
        print(f"Rookies NOT in field (skipped): {list(rookies_missing.keys())}")

    # Check current state
    current = con.execute("""
        SELECT musher_id, n_finishes, avg_finish_place, best_finish_place,
               career_race_number, is_rookie
        FROM musher_strength
        WHERE year = 2026 AND musher_id IN ({})
    """.format(",".join(f"'{m}'" for m in rookies_in_field))).df()

    print(f"\nCurrent musher_strength rows for 2026 rookies:")
    print(current.to_string(index=False))

    # Build updated rows
    print(f"\n{'='*70}")
    print("PROPOSED CHANGES")
    print(f"{'='*70}")

    rows = []
    for musher_id, score in sorted(rookies_in_field.items(), key=lambda x: -x[1]):
        row = build_rookie_row(musher_id, score)
        rows.append(row)
        fp = row["avg_finish_place"]
        print(f"  {musher_id}: score={score} → avg_finish_place={fp:.0f}, "
              f"pct_top10={row['pct_top10']:.0%}, pct_finished={row['pct_finished']:.0%}")

    if not args.write:
        print(f"\n*** DRY RUN — no changes written. Use --write to apply. ***")
        return

    # Write the updates
    print(f"\nWriting updates to musher_strength...")

    # We update only the columns that were NaN for rookies.
    # We keep is_rookie=1 and career_race_number=1 (unchanged)
    # We do NOT touch trajectory or last_year_improvement (keep NaN — no prior Iditarod)
    update_cols = [
        "n_finishes", "n_entries", "n_years",
        "avg_finish_place", "median_finish_place", "best_finish_place",
        "pct_finished", "pct_top10", "pct_top5", "pct_win",
        "last3_avg_finish_place", "last5_avg_finish_place",
        "last3_pct_top10", "last5_pct_top10",
        "w_avg_finish_place", "w_pct_top10", "w_pct_finished",
        "form_last3_avg_finish_place", "form_last5_avg_finish_place",
        "form_last3_pct_top10", "form_last5_pct_top10",
        "form_w_avg_finish_place", "form_w_pct_top10",
        "exp_n_entries", "exp_n_finishes", "exp_n_years", "exp_w_entries",
        "peak_best_finish_place", "peak_pct_top10", "peak_pct_top5", "peak_pct_win",
        "cons_pct_finished", "cons_last5_pct_finished", "cons_w_pct_finished",
        "w_avg_time_behind_winner_seconds", "w_n_entries",
    ]

    for row in rows:
        set_clauses = ", ".join(f"{col} = {row[col]}" for col in update_cols)
        sql = f"""
            UPDATE musher_strength
            SET {set_clauses}
            WHERE year = 2026 AND musher_id = '{row['musher_id']}'
        """
        con.execute(sql)

    # Verify
    updated = con.execute("""
        SELECT musher_id, n_finishes, avg_finish_place, best_finish_place,
               pct_top10, pct_finished, career_race_number, is_rookie
        FROM musher_strength
        WHERE year = 2026 AND musher_id IN ({})
        ORDER BY avg_finish_place
    """.format(",".join(f"'{m}'" for m in rookies_in_field))).df()

    print(f"\nUpdated musher_strength rows:")
    print(updated.to_string(index=False))
    print(f"\n✅ Rookie strength injection complete for {len(rows)} mushers.")
    print("Next steps: rebuild snapshots for 2026 and retrain the model.")


if __name__ == "__main__":
    main()
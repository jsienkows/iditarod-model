"""
patch_structural_uncertainty.py
================================
Patches src/model/predict_inrace.py to use per-musher noise scaling
in the Monte Carlo simulation.

Currently all mushers get the same sigma. But prediction confidence
varies with career history: a 12-finish veteran is much more predictable
than a rookie or a returning musher with a 6-year gap.

This patch:
  1. Extracts n_finishes, is_rookie, years_since_last from the snapshot data
  2. Computes a per-musher uncertainty multiplier (sqrt(8/n_finishes), capped)
  3. Scales the individual noise component by that multiplier
  4. Adds uncertainty_mult and prediction intervals to output

Works with BOTH log-normal (improvement #2) and the prior-decay blend (#4).
The musher history fields come from the musher_strength table which is
already joined in _load_snapshots_for_cp().

Usage:
    python improvements/patch_structural_uncertainty.py          # show
    python improvements/patch_structural_uncertainty.py --apply   # auto-apply
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
import shutil


def try_replace(text, old, new):
    """Try exact match, then CRLF variant."""
    if old in text:
        return text.replace(old, new), True
    old_crlf = old.replace("\n", "\r\n")
    if old_crlf in text:
        new_crlf = new.replace("\n", "\r\n")
        return text.replace(old_crlf, new_crlf), True
    return text, False


PREDICT_FILE = Path("src/model/predict_inrace.py")

# ---- ADDITION 1: uncertainty multiplier function (after imports) ----
FUNC_INSERT_AFTER = "from src.db import connect"

UNCERTAINTY_FUNC = '''


def _compute_uncertainty_multiplier(
    n_finishes, is_rookie, years_since_last=None,
    base_n=8.0, max_mult=2.0, min_mult=0.85, rookie_mult=1.8,
    absence_bonus_per_year=0.05, max_absence_bonus=0.4,
):
    """
    Per-musher uncertainty multiplier for Monte Carlo noise scaling.

    More races = more certainty = less noise. Rookies and returning
    mushers get wider distributions.

    Returns array of multipliers (>1 = wider, <1 = narrower than baseline).
    """
    n_fin = np.asarray(n_finishes, dtype=float)
    rookie = np.asarray(is_rookie, dtype=float)

    mult = np.sqrt(base_n / np.maximum(n_fin, 1.0))
    mult = np.where(rookie == 1, rookie_mult, mult)

    if years_since_last is not None:
        ysl = np.asarray(years_since_last, dtype=float)
        absence_years = np.maximum(ysl - 1, 0)
        absence_add = np.minimum(absence_years * absence_bonus_per_year, max_absence_bonus)
        mult = mult + absence_add

    return np.clip(mult, min_mult, max_mult)

'''

# ---- PATCH: Replace log-normal simulation block with per-musher version ----
# This matches the PATCHED (log-normal) version of the noise block.

LOGNORMAL_OLD = """\
    # Correlated log-normal noise: shared + individual
    # Log-normal is multiplicative: sim_time = T * exp(noise)
    # This ensures: always positive, right-skewed (slowdowns more likely than speedups)
    #
    # Convert sigma from hours to log-space: sigma_log ~ sigma_hours / T_mean
    # Bias correction -sigma^2/2 ensures E[exp(noise)] = 1, so E[sim_time] = T
    T_mean = np.nanmean(pred_finish_time_hours)
    sigma_log = sigma_hours / T_mean if T_mean > 0 else 0.1

    shared_sigma_log = sigma_log * np.sqrt(shared_frac)
    indiv_sigma_log = sigma_log * np.sqrt(1.0 - shared_frac)

    shared_noise = rng.normal(
        loc=-shared_sigma_log**2 / 2, scale=shared_sigma_log, size=(args.n_sims, 1)
    )
    indiv_noise = rng.normal(
        loc=-indiv_sigma_log**2 / 2, scale=indiv_sigma_log, size=(args.n_sims, len(df))
    )

    sim_finish = pred_finish_time_hours[None, :] * np.exp(shared_noise + indiv_noise)

    # Apply finish probabilities \u2192 some sims convert to DNF penalty
    u = rng.random(size=(args.n_sims, len(df)))
    sim_is_finish = u < p_finish[None, :]
    sim_finish = np.where(sim_is_finish, sim_finish, dnf_penalty_hours)

    # Places: argsort per sim
    # place 1 = smallest time
    order = np.argsort(sim_finish, axis=1)
    ranks = np.empty_like(order)
    rows = np.arange(args.n_sims)[:, None]
    ranks[rows, order] = np.arange(len(df))[None, :]
    place = ranks + 1

    p_win = (place == 1).mean(axis=0)
    p_top10 = (place <= 10).mean(axis=0)
    exp_place = place.mean(axis=0)

    out = df[["year", "checkpoint_order", "musher_id", "asof_time_utc", "rank_at_checkpoint"]].copy()
    out["p_finish"] = p_finish
    out["pred_remaining_hours"] = pred_remaining_hours
    out["pred_finish_time_hours"] = pred_finish_time_hours
    out["p_win"] = p_win
    out["p_top10"] = p_top10
    out["exp_place"] = exp_place

    # Readability columns (so tiny probs aren't printed as 0.00000)
    out["p_win_pct"] = 100.0 * out["p_win"]
    out["p_top10_pct"] = 100.0 * out["p_top10"]"""

LOGNORMAL_NEW = """\
    # ---- Per-musher structural uncertainty ----
    # Extract musher history from snapshot data (joined from musher_strength)
    n_finishes = pd.to_numeric(df.get("n_finishes", 0), errors="coerce").fillna(0).values
    is_rookie = pd.to_numeric(df.get("is_rookie", 0), errors="coerce").fillna(0).values
    ysl = pd.to_numeric(df.get("years_since_last_entry", 0), errors="coerce").fillna(0).values

    unc_mult = _compute_uncertainty_multiplier(n_finishes, is_rookie, ysl)

    # Log-normal noise with per-musher scaling
    # Shared noise (weather/trail, same for all mushers)
    # Individual noise scaled by uncertainty multiplier per musher
    T_mean = np.nanmean(pred_finish_time_hours)
    if T_mean <= 0 or not np.isfinite(T_mean):
        T_mean = 240.0

    shared_sigma_log = np.clip(sigma_hours * np.sqrt(shared_frac) / T_mean, 0.01, 0.3)

    # Per-musher individual sigma in log-space
    indiv_sigma_base = sigma_hours * np.sqrt(1.0 - shared_frac)
    indiv_sigma_per_musher = indiv_sigma_base * unc_mult  # (n_mushers,)
    indiv_sigma_log = np.clip(indiv_sigma_per_musher / T_mean, 0.01, 0.5)

    # Shared noise (bias-corrected for log-normal)
    shared_noise = rng.normal(
        -shared_sigma_log**2 / 2, shared_sigma_log, (args.n_sims, 1)
    )

    # Individual noise: per-musher sigma (bias-corrected)
    indiv_noise = rng.normal(
        -indiv_sigma_log**2 / 2,   # per-musher bias correction
        indiv_sigma_log,             # per-musher sigma
        size=(args.n_sims, len(df)),
    )

    sim_finish = pred_finish_time_hours[None, :] * np.exp(shared_noise + indiv_noise)

    # Apply finish probabilities -> some sims convert to DNF penalty
    u = rng.random(size=(args.n_sims, len(df)))
    sim_is_finish = u < p_finish[None, :]
    sim_finish = np.where(sim_is_finish, sim_finish, dnf_penalty_hours)

    # Places: argsort per sim
    order = np.argsort(sim_finish, axis=1)
    ranks = np.empty_like(order)
    rows = np.arange(args.n_sims)[:, None]
    ranks[rows, order] = np.arange(len(df))[None, :]
    place = ranks + 1

    p_win = (place == 1).mean(axis=0)
    p_top10 = (place <= 10).mean(axis=0)
    exp_place = place.mean(axis=0).astype(float)

    # Prediction intervals (80% CI on finishing position)
    place_p10 = np.percentile(place, 10, axis=0).astype(int)
    place_p90 = np.percentile(place, 90, axis=0).astype(int)

    out = df[["year", "checkpoint_order", "musher_id", "asof_time_utc", "rank_at_checkpoint"]].copy()
    out["p_finish"] = p_finish
    out["pred_remaining_hours"] = pred_remaining_hours
    out["pred_finish_time_hours"] = pred_finish_time_hours
    out["p_win"] = p_win
    out["p_top10"] = p_top10
    out["exp_place"] = exp_place
    out["place_p10"] = place_p10
    out["place_p90"] = place_p90
    out["unc_mult"] = unc_mult

    # Readability columns
    out["p_win_pct"] = 100.0 * out["p_win"]
    out["p_top10_pct"] = 100.0 * out["p_top10"]
    out["place_ci"] = [f"[{lo},{hi}]" for lo, hi in zip(place_p10, place_p90)]"""


# Also need to update the print format to show uncertainty info
PRINT_OLD = """\
    print(
        f"\\nIn-race predictions | year={args.year} cp={args.checkpoint_order} | "
        f"n_mushers={len(out)} | n_sims={args.n_sims} | reg_model_used={reg_used} | "
        f"sigma_hours\\u2248{sigma_hours:.2f} (raw\\u2248{sigma_raw:.2f}, floor={sigma_floor:.2f}, mult={args.sigma_mult:.2f}) | "
        f"shared_noise_frac={shared_frac:.2f}"
    )"""

PRINT_NEW = """\
    print(
        f"\\nIn-race predictions | year={args.year} cp={args.checkpoint_order} | "
        f"n_mushers={len(out)} | n_sims={args.n_sims} | reg_model_used={reg_used} | "
        f"sigma_hours\\u2248{sigma_hours:.2f} (raw\\u2248{sigma_raw:.2f}, floor={sigma_floor:.2f}, mult={args.sigma_mult:.2f}) | "
        f"shared_noise_frac={shared_frac:.2f} | "
        f"unc_mult: [{unc_mult.min():.2f}, {unc_mult.max():.2f}]"
    )"""


def apply_patch():
    if not PREDICT_FILE.exists():
        print(f"ERROR: {PREDICT_FILE} not found. Run from project root.")
        return False

    backup = PREDICT_FILE.with_suffix(".py.bak_struct")
    shutil.copy2(PREDICT_FILE, backup)
    print(f"Backed up to {backup}")

    text = PREDICT_FILE.read_text(encoding="utf-8")
    success = True

    # 1. Add uncertainty function
    if FUNC_INSERT_AFTER in text:
        if "_compute_uncertainty_multiplier" not in text:
            text = text.replace(FUNC_INSERT_AFTER, FUNC_INSERT_AFTER + UNCERTAINTY_FUNC, 1)
            print("  Added _compute_uncertainty_multiplier() function")
        else:
            print("  _compute_uncertainty_multiplier() already present, skipping")
    else:
        print("  WARNING: Could not find import anchor")
        success = False

    # 2. Replace simulation block (log-normal version)
    text, ok = try_replace(text, LOGNORMAL_OLD, LOGNORMAL_NEW)
    if ok:
        print("  Replaced simulation block with per-musher uncertainty version")
    else:
        print("  WARNING: Could not match log-normal simulation block.")
        print("  This patch expects the log-normal noise patch (#2) to be applied first.")
        print("  If you haven't applied it, run patch_lognormal_noise.py --apply first.")
        success = False

    # 3. Update print line
    text, ok2 = try_replace(text, PRINT_OLD, PRINT_NEW)
    if ok2:
        print("  Updated print line with uncertainty multiplier range")
    else:
        print("  WARNING: Could not match print line (non-critical)")

    PREDICT_FILE.write_text(text, encoding="utf-8")
    print(f"\n{'Patched' if success else 'Partially patched'} {PREDICT_FILE}")

    if success:
        print(f"\nOutput now includes:")
        print(f"  unc_mult   - per-musher uncertainty multiplier (>1 = wider)")
        print(f"  place_p10  - 10th percentile finishing position (optimistic)")
        print(f"  place_p90  - 90th percentile finishing position (pessimistic)")
        print(f"  place_ci   - 80% prediction interval as [lo, hi]")

    return success


def show_patch():
    print("=" * 70)
    print("PATCH: Structural uncertainty for predict_inrace.py")
    print("=" * 70)
    print(f"\n1. Add _compute_uncertainty_multiplier() function")
    print(f"\n2. Replace simulation block (expects log-normal already applied):")
    print(f"\n   Key change: indiv_sigma becomes per-musher vector:")
    print(f"     indiv_sigma_per_musher = indiv_sigma_base * unc_mult")
    print(f"\n3. Add to output: unc_mult, place_p10, place_p90, place_ci")
    print(f"\nRun with --apply to auto-patch.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true")
    args = ap.parse_args()

    if args.apply:
        apply_patch()
    else:
        show_patch()


if __name__ == "__main__":
    main()
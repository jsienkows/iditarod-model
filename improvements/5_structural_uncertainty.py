"""
5_structural_uncertainty.py
============================
Add wider prediction intervals for thin-history mushers in the Monte
Carlo simulation, so mushers like Waerner (2 career races) get more
noise than Jessie Holmes (10+ races).

PROBLEM:
  Currently all mushers get the same sigma in the simulation. But a
  prediction for a musher with 10 finishes is inherently more reliable
  than one for a musher with 1-2 finishes. The model outputs a confident
  number either way because it has no mechanism to express "I'm guessing."

SOLUTION:
  Scale sigma per-musher based on their career history:
    sigma_i = sigma_base * uncertainty_multiplier(n_finishes_i, is_rookie_i)

  The multiplier is >1 for thin-history mushers and ~1 for well-known ones.
  This makes the simulation naturally produce wider outcome distributions
  for uncertain mushers without changing the point predictions.

INTEGRATION:
  This modifies the noise generation in both predict_inrace.py and
  backtest_inrace.py. The key change is that `indiv_sigma` becomes a
  per-musher vector instead of a scalar.

Usage:
    # This file provides functions and integration instructions.
    # Not a standalone runner.
"""

import numpy as np
import pandas as pd


def compute_uncertainty_multiplier(
    n_finishes: np.ndarray,
    is_rookie: np.ndarray,
    years_since_last: np.ndarray = None,
    base_n: float = 8.0,
    max_multiplier: float = 2.0,
    min_multiplier: float = 0.85,
    rookie_multiplier: float = 1.8,
    absence_bonus_per_year: float = 0.05,
    max_absence_bonus: float = 0.4,
) -> np.ndarray:
    """
    Compute per-musher uncertainty multiplier for Monte Carlo noise.

    The idea: simulation noise should scale with how uncertain we are
    about each musher's true ability. More races = more certainty = less noise.

    Parameters
    ----------
    n_finishes : array (n_mushers,)
        Career Iditarod finishes for each musher.
    is_rookie : array (n_mushers,)
        1 if rookie, 0 otherwise.
    years_since_last : array (n_mushers,), optional
        Years since last race entry. Long absences add uncertainty.
    base_n : float
        Number of finishes at which multiplier = 1.0.
        Mushers with fewer finishes get multiplied up.
    max_multiplier : float
        Cap on uncertainty multiplier.
    min_multiplier : float
        Floor (even experienced mushers have some irreducible uncertainty).
    rookie_multiplier : float
        Fixed multiplier for rookies (overrides the n_finishes calculation).
    absence_bonus_per_year : float
        Additional multiplier per year of absence beyond 1.
    max_absence_bonus : float
        Cap on absence-related multiplier increase.

    Returns
    -------
    multiplier : array (n_mushers,)
        Per-musher sigma multiplier. Values > 1 = wider distribution.
    """
    n_fin = np.asarray(n_finishes, dtype=float)
    rookie = np.asarray(is_rookie, dtype=float)

    # Base multiplier: sqrt(base_n / max(n_finishes, 1))
    # This gives a natural shrinkage: 1 finish → sqrt(8) ≈ 2.8x, 4 finishes → sqrt(2) ≈ 1.4x
    mult = np.sqrt(base_n / np.maximum(n_fin, 1.0))

    # Rookies get a fixed multiplier
    mult = np.where(rookie == 1, rookie_multiplier, mult)

    # Absence bonus: long time away adds uncertainty
    if years_since_last is not None:
        ysl = np.asarray(years_since_last, dtype=float)
        absence_years = np.maximum(ysl - 1, 0)  # 0-1 years = no bonus
        absence_add = np.minimum(absence_years * absence_bonus_per_year, max_absence_bonus)
        mult = mult + absence_add

    # Clip to bounds
    mult = np.clip(mult, min_multiplier, max_multiplier)

    return mult


def apply_structural_uncertainty(
    pred_finish_time_hours: np.ndarray,
    p_finish: np.ndarray,
    n_sims: int,
    sigma_hours: float,
    shared_frac: float,
    rng: np.random.Generator,
    n_finishes: np.ndarray,
    is_rookie: np.ndarray,
    years_since_last: np.ndarray = None,
    use_lognormal: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Monte Carlo simulation with per-musher uncertainty scaling.

    Combines improvements #2 (log-normal) and #5 (structural uncertainty).

    Returns
    -------
    p_win, p_top10, exp_place, uncertainty_mult : arrays (n_mushers,)
    """
    n_mushers = len(pred_finish_time_hours)

    # Compute per-musher uncertainty multiplier
    mult = compute_uncertainty_multiplier(
        n_finishes, is_rookie, years_since_last
    )

    # Shared noise (same for all mushers — weather, trail conditions)
    shared_sigma = sigma_hours * np.sqrt(shared_frac)

    # Individual noise: base sigma * per-musher multiplier
    indiv_sigma_base = sigma_hours * np.sqrt(1.0 - shared_frac)
    indiv_sigma_per_musher = indiv_sigma_base * mult  # shape: (n_mushers,)

    if use_lognormal:
        # Log-normal approach
        T_mean = np.nanmean(pred_finish_time_hours)
        if T_mean <= 0 or not np.isfinite(T_mean):
            T_mean = 240.0

        shared_sigma_log = shared_sigma / T_mean
        shared_sigma_log = np.clip(shared_sigma_log, 0.01, 0.3)

        indiv_sigma_log = indiv_sigma_per_musher / T_mean  # per-musher log-space sigma
        indiv_sigma_log = np.clip(indiv_sigma_log, 0.01, 0.5)

        # Shared noise (bias-corrected)
        shared_noise = rng.normal(
            -shared_sigma_log**2 / 2, shared_sigma_log, (n_sims, 1)
        )

        # Individual noise: each musher gets their own sigma
        # rng.normal can broadcast if scale is (1, n_mushers) shaped
        indiv_noise = rng.normal(
            -indiv_sigma_log**2 / 2,  # bias correction per-musher
            indiv_sigma_log,           # per-musher sigma
            size=(n_sims, n_mushers),
        )

        sim_finish = pred_finish_time_hours[None, :] * np.exp(shared_noise + indiv_noise)

    else:
        # Gaussian approach (with per-musher scaling)
        shared = rng.normal(0, shared_sigma, (n_sims, 1))
        indiv = rng.normal(0, 1.0, (n_sims, n_mushers)) * indiv_sigma_per_musher[None, :]
        sim_finish = pred_finish_time_hours[None, :] + shared + indiv

    # DNF penalty
    max_ft = np.nanmax(pred_finish_time_hours)
    if not np.isfinite(max_ft):
        max_ft = 0.0
    dnf_penalty = max_ft + 500.0

    u = rng.random((n_sims, n_mushers))
    sim_is_finish = u < p_finish[None, :]
    sim_finish = np.where(sim_is_finish, sim_finish, dnf_penalty)

    # Rank
    order = np.argsort(sim_finish, axis=1)
    ranks = np.empty_like(order)
    rows = np.arange(n_sims)[:, None]
    ranks[rows, order] = np.arange(n_mushers)[None, :]
    place = ranks + 1

    p_win = (place == 1).mean(axis=0)
    p_top10 = (place <= 10).mean(axis=0)
    exp_place = place.mean(axis=0).astype(float)

    return p_win, p_top10, exp_place, mult


def compute_prediction_intervals(
    pred_finish_time_hours: np.ndarray,
    p_finish: np.ndarray,
    n_sims: int,
    sigma_hours: float,
    shared_frac: float,
    rng: np.random.Generator,
    n_finishes: np.ndarray,
    is_rookie: np.ndarray,
    years_since_last: np.ndarray = None,
    intervals: tuple = (0.10, 0.90),
) -> pd.DataFrame:
    """
    Compute prediction intervals (e.g., 80% CI) for each musher's
    finishing position from the simulation.

    Returns DataFrame with columns:
        musher_idx, p_win, p_top10, exp_place, place_low, place_high,
        uncertainty_mult
    """
    n_mushers = len(pred_finish_time_hours)
    mult = compute_uncertainty_multiplier(n_finishes, is_rookie, years_since_last)

    # Run full simulation
    shared_sigma = sigma_hours * np.sqrt(shared_frac)
    indiv_sigma_base = sigma_hours * np.sqrt(1.0 - shared_frac)
    indiv_sigma_per_musher = indiv_sigma_base * mult

    T_mean = np.nanmean(pred_finish_time_hours)
    if T_mean <= 0 or not np.isfinite(T_mean):
        T_mean = 240.0

    shared_sigma_log = np.clip(shared_sigma / T_mean, 0.01, 0.3)
    indiv_sigma_log = np.clip(indiv_sigma_per_musher / T_mean, 0.01, 0.5)

    shared_noise = rng.normal(-shared_sigma_log**2 / 2, shared_sigma_log, (n_sims, 1))
    indiv_noise = rng.normal(-indiv_sigma_log**2 / 2, indiv_sigma_log, (n_sims, n_mushers))
    sim_finish = pred_finish_time_hours[None, :] * np.exp(shared_noise + indiv_noise)

    max_ft = np.nanmax(pred_finish_time_hours)
    dnf_penalty = (max_ft if np.isfinite(max_ft) else 0) + 500.0

    u = rng.random((n_sims, n_mushers))
    sim_finish = np.where(u < p_finish[None, :], sim_finish, dnf_penalty)

    order = np.argsort(sim_finish, axis=1)
    ranks = np.empty_like(order)
    rows = np.arange(n_sims)[:, None]
    ranks[rows, order] = np.arange(n_mushers)[None, :]
    place = ranks + 1

    # Compute intervals per musher
    results = []
    for i in range(n_mushers):
        places_i = place[:, i]
        results.append({
            "musher_idx": i,
            "p_win": float((places_i == 1).mean()),
            "p_top10": float((places_i <= 10).mean()),
            "exp_place": float(places_i.mean()),
            f"place_p{int(intervals[0]*100)}": int(np.percentile(places_i, intervals[0] * 100)),
            f"place_p{int(intervals[1]*100)}": int(np.percentile(places_i, intervals[1] * 100)),
            "place_std": float(places_i.std()),
            "uncertainty_mult": float(mult[i]),
        })

    return pd.DataFrame(results)


# ==============================================================================
# INTEGRATION GUIDE
# ==============================================================================
#
# STEP 1: predict_inrace.py
# -------------------------
# After loading snapshots and before the simulation, extract musher history:
#
#   n_finishes = pd.to_numeric(df.get("n_finishes", 0), errors="coerce").fillna(0).values
#   is_rookie = pd.to_numeric(df.get("is_rookie", 0), errors="coerce").fillna(0).values
#   ysl = pd.to_numeric(df.get("years_since_last_entry", 0), errors="coerce").fillna(0).values
#
# Then replace the simulation block with:
#
#   from improvements.structural_uncertainty import apply_structural_uncertainty
#   p_win, p_top10, exp_place, unc_mult = apply_structural_uncertainty(
#       pred_finish_time_hours, p_finish, args.n_sims,
#       sigma_hours, shared_frac, rng,
#       n_finishes, is_rookie, ysl,
#       use_lognormal=True,
#   )
#
# Add unc_mult to the output DataFrame for reporting.
#
#
# STEP 2: Add prediction intervals to output
# -------------------------------------------
# For each musher, report not just "expected place = 3.2" but also
# "80% CI: [1, 7]" so thin-history mushers show wider ranges.
#
# Use compute_prediction_intervals() and add place_p10, place_p90
# columns to the output CSV.
#
#
# STEP 3: predict_prerace_2026.py (optional)
# ------------------------------------------
# For pre-race predictions, the structural uncertainty is already
# partially captured by the volatility score. But you could additionally
# report prediction intervals on the composite rank:
#
#   "Waerner: predicted #8, 80% CI [#2, #18]"
#
# This requires running the pre-race Monte Carlo with per-musher sigma
# scaling (same approach as in-race, but using pre-race model outputs).


# ==============================================================================
# EXAMPLE: Show what uncertainty multipliers look like for 2026 field
# ==============================================================================

def demo_2026_multipliers():
    """Print example multipliers for a typical Iditarod field."""
    print("Example uncertainty multipliers for typical musher profiles:")
    print(f"{'Profile':>35} | {'n_fin':>5} {'rook':>4} {'ysl':>4} | {'mult':>6}")
    print("-" * 65)

    profiles = [
        ("Veteran champion (10+ finishes)", 12, 0, 1),
        ("Strong contender (6 finishes)", 6, 0, 1),
        ("Mid-pack regular (4 finishes)", 4, 0, 1),
        ("Waerner-type (2 fin, 6yr gap)", 2, 0, 6),
        ("2nd-year musher (1 finish)", 1, 0, 1),
        ("Rookie (strong qualifier)", 0, 1, 0),
    ]

    for label, n_fin, rookie, ysl in profiles:
        mult = compute_uncertainty_multiplier(
            np.array([n_fin]), np.array([rookie]), np.array([ysl])
        )[0]
        print(f"{label:>35} | {n_fin:>5} {rookie:>4} {ysl:>4} | {mult:>6.2f}")

    print(f"\nInterpretation: mult=1.0 is baseline noise, mult=2.0 means 2x wider distribution.")
    print(f"A musher with mult=1.8 will have ~80% wider prediction intervals than mult=1.0.")


if __name__ == "__main__":
    demo_2026_multipliers()
"""
patch_lognormal_noise.py
========================
Patches BOTH src/model/predict_inrace.py and src/model/backtest_inrace.py
to use log-normal noise instead of Gaussian.

Usage:
    python improvements/patch_lognormal_noise.py          # show patches
    python improvements/patch_lognormal_noise.py --apply   # auto-apply

Why log-normal:
  - Race times are right-skewed (slowdowns more likely than speedups)
  - Multiplicative noise can't produce negative remaining times
  - Modest improvement in winner probability calibration (+0.021 avg)
  - P@10 is unchanged (no downside)
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


# ============================================================
# PATCH 1: src/model/predict_inrace.py
# ============================================================

PREDICT_FILE = Path("src/model/predict_inrace.py")

PREDICT_OLD = """\
    # Correlated noise: shared + individual
    # Total variance = sigma_hours^2
    # shared contributes shared_frac of variance, individual contributes (1-shared_frac)
    shared_sigma = sigma_hours * np.sqrt(shared_frac)
    indiv_sigma = sigma_hours * np.sqrt(1.0 - shared_frac)

    shared = rng.normal(loc=0.0, scale=shared_sigma, size=(args.n_sims, 1))
    indiv = rng.normal(loc=0.0, scale=indiv_sigma, size=(args.n_sims, len(df)))

    sim_finish = pred_finish_time_hours[None, :] + shared + indiv"""

PREDICT_NEW = """\
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

    sim_finish = pred_finish_time_hours[None, :] * np.exp(shared_noise + indiv_noise)"""


# ============================================================
# PATCH 2: src/model/backtest_inrace.py
# ============================================================

BACKTEST_FILE = Path("src/model/backtest_inrace.py")

BACKTEST_OLD = """\
    shared_sigma = sigma_hours * np.sqrt(shared_frac)
    indiv_sigma = sigma_hours * np.sqrt(1.0 - shared_frac)

    shared = rng.normal(0, shared_sigma, (n_sims, 1))
    indiv = rng.normal(0, indiv_sigma, (n_sims, len(X)))

    sim_finish = pred_finish_hours[None, :] + shared + indiv"""

BACKTEST_NEW = """\
    # Log-normal noise (multiplicative, right-skewed)
    T_mean = np.nanmean(pred_finish_hours)
    sigma_log = sigma_hours / T_mean if T_mean > 0 else 0.1

    shared_sigma_log = sigma_log * np.sqrt(shared_frac)
    indiv_sigma_log = sigma_log * np.sqrt(1.0 - shared_frac)

    shared_noise = rng.normal(-shared_sigma_log**2 / 2, shared_sigma_log, (n_sims, 1))
    indiv_noise = rng.normal(-indiv_sigma_log**2 / 2, indiv_sigma_log, (n_sims, len(X)))

    sim_finish = pred_finish_hours[None, :] * np.exp(shared_noise + indiv_noise)"""


def apply_patches():
    results = []

    for label, fpath, old, new in [
        ("predict_inrace.py", PREDICT_FILE, PREDICT_OLD, PREDICT_NEW),
        ("backtest_inrace.py", BACKTEST_FILE, BACKTEST_OLD, BACKTEST_NEW),
    ]:
        if not fpath.exists():
            print(f"WARNING: {fpath} not found. Skipping {label}.")
            results.append((label, False))
            continue

        backup = fpath.with_suffix(".py.bak")
        shutil.copy2(fpath, backup)

        text = fpath.read_text(encoding="utf-8")
        text, success = try_replace(text, old, new)

        if success:
            fpath.write_text(text, encoding="utf-8")
            print(f"Patched {label} (backup: {backup})")
            results.append((label, True))
        else:
            print(f"WARNING: Could not match noise block in {label}. Apply manually.")
            results.append((label, False))

    print(f"\nResults: {sum(1 for _, s in results if s)}/{len(results)} files patched.")


def show_patches():
    print("=" * 70)
    print("PATCH 1: predict_inrace.py (~lines 356-365)")
    print("=" * 70)
    print("\nOLD:")
    print(PREDICT_OLD)
    print("\nNEW:")
    print(PREDICT_NEW)

    print("\n" + "=" * 70)
    print("PATCH 2: backtest_inrace.py (~lines 154-160)")
    print("=" * 70)
    print("\nOLD:")
    print(BACKTEST_OLD)
    print("\nNEW:")
    print(BACKTEST_NEW)

    print("\nRun with --apply to auto-patch both files.")


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
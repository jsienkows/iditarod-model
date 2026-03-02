"""
patch_prior_decay.py
=====================
Patches src/model/predict_inrace.py to blend full (with priors) and
snapshot-only (no priors) model predictions with a checkpoint-dependent
weight that fades priors as in-race data accumulates.

Based on the diagnostic (4_prior_decay_calibration.py):
  - Priors are essential at CP 3-5 (snapshot-only can't rank anyone)
  - Priors are neutral at CP 8
  - Priors actively hurt at CP 10-18
  - Both converge by CP 20+

The blend weight follows a hyperbolic decay:
  w = 1 / (1 + checkpoint_order)

Prerequisites:
  Run train_snapshot_models.py first to create the snapshot-only models.

Usage:
    python improvements/patch_prior_decay.py          # show patches
    python improvements/patch_prior_decay.py --apply   # auto-apply
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

# ---- ADDITION 1: decay weight function (after imports) ----
FUNC_INSERT_AFTER = "from src.db import connect"

DECAY_FUNC = '''


def _prior_decay_weight(checkpoint_order: int) -> float:
    """
    Hyperbolic decay: weight on the full (prior-including) model.
    w = 1 / (1 + checkpoint_order)

    CP  1 -> w = 0.500 (priors get half weight)
    CP  5 -> w = 0.167
    CP 10 -> w = 0.091
    CP 15 -> w = 0.062
    CP 20 -> w = 0.048

    Returns the weight for the FULL model. Snapshot-only gets (1 - w).
    """
    return 1.0 / (1.0 + checkpoint_order)

'''

# ---- ADDITION 2: replace prediction block with blended version ----

PREDICT_OLD = """\
    # Predict finish probability
    p_finish = finish_model.predict_proba(Xf)[:, 1]

    # Predict remaining seconds (log1p model)
    pred_remaining_sec, reg_used = _predict_remaining_seconds(reg_model, Xf, args.checkpoint_order)
    pred_remaining_hours = pred_remaining_sec / 3600.0"""

PREDICT_NEW = """\
    # ---- Prior decay: blend full + snapshot-only predictions ----
    snap_finish_path = model_dir / "inrace_finish_model_snapshot.joblib"
    snap_reg_path = model_dir / "inrace_remaining_time_model_snapshot.joblib"
    snapshot_only_cols = meta.get("snapshot_only_cols")

    use_prior_decay = (
        snap_finish_path.exists()
        and snap_reg_path.exists()
        and snapshot_only_cols is not None
    )

    # Full model predictions (with priors)
    p_finish_full = finish_model.predict_proba(Xf)[:, 1]
    pred_remaining_sec_full, reg_used = _predict_remaining_seconds(reg_model, Xf, args.checkpoint_order)

    if use_prior_decay:
        # Snapshot-only predictions (no priors)
        snap_finish_model = joblib.load(snap_finish_path)
        snap_reg_model = joblib.load(snap_reg_path)
        Xf_snap = X[snapshot_only_cols].copy()

        p_finish_snap = snap_finish_model.predict_proba(Xf_snap)[:, 1]
        # Snapshot regressor is always global (not per-checkpoint bundle)
        snap_pred_log = snap_reg_model.predict(Xf_snap)
        pred_remaining_sec_snap = np.clip(np.expm1(snap_pred_log), 0, None)

        w = _prior_decay_weight(args.checkpoint_order)
        p_finish = w * p_finish_full + (1 - w) * p_finish_snap
        pred_remaining_sec = w * pred_remaining_sec_full + (1 - w) * pred_remaining_sec_snap
        print(f"  Prior decay: w_full={w:.3f}, w_snap={1-w:.3f} at CP {args.checkpoint_order}")
    else:
        p_finish = p_finish_full
        pred_remaining_sec = pred_remaining_sec_full
        print("  Prior decay: snapshot models not found, using full model only")

    pred_remaining_hours = pred_remaining_sec / 3600.0"""


def apply_patch():
    if not PREDICT_FILE.exists():
        print(f"ERROR: {PREDICT_FILE} not found. Run from project root.")
        return False

    backup = PREDICT_FILE.with_suffix(".py.bak_decay")
    shutil.copy2(PREDICT_FILE, backup)
    print(f"Backed up to {backup}")

    text = PREDICT_FILE.read_text(encoding="utf-8")
    success = True

    # 1. Add decay function
    if FUNC_INSERT_AFTER in text:
        # Check not already patched
        if "_prior_decay_weight" not in text:
            text = text.replace(FUNC_INSERT_AFTER, FUNC_INSERT_AFTER + DECAY_FUNC, 1)
            print("  Added _prior_decay_weight() function")
        else:
            print("  _prior_decay_weight() already present, skipping")
    else:
        print("  WARNING: Could not find import anchor")
        success = False

    # 2. Replace prediction block with blended version
    text, ok = try_replace(text, PREDICT_OLD, PREDICT_NEW)
    if ok:
        print("  Replaced prediction block with blended version")
    else:
        print("  WARNING: Could not match prediction block. Apply manually.")
        success = False

    PREDICT_FILE.write_text(text, encoding="utf-8")
    print(f"\n{'Patched' if success else 'Partially patched'} {PREDICT_FILE}")

    if success:
        print(f"\nSetup steps:")
        print(f"  1. Train snapshot-only models:")
        print(f"     python improvements/train_snapshot_models.py")
        print(f"  2. Run predictions as usual:")
        print(f"     python -m src.model.predict_inrace --year 2025 --checkpoint_order 10")
        print(f"  3. Blending happens automatically when snapshot models exist.")

    return success


def show_patch():
    print("=" * 70)
    print("PATCH: Prior decay blending for predict_inrace.py")
    print("=" * 70)

    print(f"\n1. Add function after '{FUNC_INSERT_AFTER}':")
    print(DECAY_FUNC)

    print(f"\n2. Replace prediction block:")
    print(f"\nOLD:\n{PREDICT_OLD}")
    print(f"\nNEW:\n{PREDICT_NEW}")

    print("\nRun with --apply to auto-patch.")


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
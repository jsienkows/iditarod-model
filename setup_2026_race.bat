@echo off
REM setup_2026_race.bat — One-time setup before the 2026 race starts.
REM Run this after the musher list is finalized but before the race begins.

setlocal

echo.
echo ============================================================
echo  Iditarod 2026 Pre-Race Setup
echo ============================================================
echo.

echo [1/5] Race context (route regime)...
python -m src.features.race_context --year_min 2026 --year_max 2026
echo.

echo [2/5] Checkpoint distances...
python -m src.features.checkpoint_distances --year 2026
echo.

echo [3/5] Musher strength features...
python -m src.features.build_musher_strength --year 2026
echo.

echo [4/5] Rookie strength injection...
python inject_rookie_strength.py --write
echo.

echo [5/5] Training in-race models (full + snapshot)...
python -m src.model.train_inrace_model --train_start 2016 --train_end 2025
python improvements/train_snapshot_models.py
echo.

echo ============================================================
echo  Setup complete. Ready for race day.
echo  Usage: update_inrace.bat CHECKPOINT_NUMBER
echo ============================================================
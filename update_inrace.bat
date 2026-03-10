@echo off
:: =============================================================
::  Iditarod 2026 — In-Race Update Pipeline
::  Usage: update_inrace.bat CHECKPOINT_NUMBER
::  Example: update_inrace.bat 5
:: =============================================================

if "%~1"=="" (
    echo Usage: update_inrace.bat CHECKPOINT_NUMBER
    exit /b 1
)

set CP=%~1

echo.
echo ============================================================
echo  Updating predictions through checkpoint %CP%
echo ============================================================

echo.
echo [1/5] Scraping latest checkpoint data...
python -m src.scrape.scrape_all_checkpoints --year 2026

echo.
echo [2/5] Parsing HTML into splits table...
python -m src.scrape.parse_all_checkpoints --year 2026

echo.
echo [3/5] Updating checkpoint distances...
python -m src.features.checkpoint_distances --year 2026

echo.
echo [4/5] Rebuilding snapshots...
python -m src.features.build_snapshots --year 2026

echo.
echo [5/5] Running predictions for checkpoint %CP%...
python -m src.model.predict_inrace --year 2026 --checkpoint_order %CP%

echo.
echo ============================================================
echo  Done. Predictions updated through checkpoint %CP%.
echo ============================================================
@echo off
REM update_inrace.bat — Scrape, parse, build features, and predict for a given checkpoint.
REM
REM Usage:
REM   update_inrace.bat 10        (predict at checkpoint 10)
REM   update_inrace.bat 15        (predict at checkpoint 15)

setlocal

set YEAR=2026
set CP=%1

if "%CP%"=="" (
    echo Usage: update_inrace.bat CHECKPOINT_ORDER
    echo Example: update_inrace.bat 10
    exit /b 1
)

echo.
echo ============================================================
echo  Iditarod %YEAR% In-Race Update — Checkpoint %CP%
echo ============================================================
echo.

echo [1/4] Scraping checkpoint data...
python -m src.scrape.scrape_all_checkpoints --year_min %YEAR% --year_max %YEAR%
if errorlevel 1 goto :error
echo.

echo [2/4] Parsing HTML to splits...
python -m src.scrape.parse_all_checkpoints --year_min %YEAR% --year_max %YEAR%
if errorlevel 1 goto :error
echo.

echo [3/4] Building snapshots...
python -m src.features.build_snapshots --year %YEAR%
if errorlevel 1 goto :error
echo.

echo [4/4] Running predictions for checkpoint %CP%...
python -m src.model.predict_inrace --year %YEAR% --checkpoint_order %CP%
if errorlevel 1 goto :error
echo.

echo ============================================================
echo  Done. Predictions saved to models\pred_inrace_%YEAR%_cp%CP%.csv
echo ============================================================
exit /b 0

:error
echo.
echo ERROR: Pipeline failed at the step above.
exit /b 1
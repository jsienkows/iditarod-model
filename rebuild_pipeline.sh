#!/usr/bin/env bash
# rebuild_pipeline.sh — Full pipeline rebuild with correct execution order.
#
# Usage:
#   ./rebuild_pipeline.sh           # rebuild all years 2016-2025
#   ./rebuild_pipeline.sh 2025      # rebuild only 2025
#
# Prerequisites:
#   - Raw HTML pages already scraped (scrape_all_checkpoints, build_entries, etc.)
#   - DuckDB database at data/db/iditarod.duckdb with splits/entries tables populated
#
# This script applies migrations, builds features, then trains models.

set -euo pipefail

YEAR_MIN=${1:-2016}
YEAR_MAX=${2:-2025}

echo "=== Iditarod Model Pipeline Rebuild ==="
echo "Years: ${YEAR_MIN} to ${YEAR_MAX}"
echo ""

# ────────────────────────────────────────────────────────────────
# Step 0: Database migrations (safe to re-run)
# ────────────────────────────────────────────────────────────────
echo "Step 0: Running database migrations..."
python -m src.db_migrations.add_new_snapshot_columns
echo ""

# ────────────────────────────────────────────────────────────────
# Step 1: Build race context (route_regime, year difficulty proxies)
# ────────────────────────────────────────────────────────────────
echo "Step 1: Building race context metadata..."
python -m src.features.race_context --year_min "$YEAR_MIN" --year_max "$YEAR_MAX"
echo ""

# ────────────────────────────────────────────────────────────────
# Step 2: Build checkpoint distances (for checkpoint_pct normalization)
# ────────────────────────────────────────────────────────────────
echo "Step 2: Building checkpoint distances..."
for year in $(seq "$YEAR_MIN" "$YEAR_MAX"); do
    python -m src.features.checkpoint_distances --year "$year" 2>&1 || echo "  (skipped year=$year — no checkpoint data)"
done
echo ""

# ────────────────────────────────────────────────────────────────
# Step 3: Build historical results table
# ────────────────────────────────────────────────────────────────
echo "Step 3: Building historical results..."
python -m src.features.build_historical_results
echo ""

# ────────────────────────────────────────────────────────────────
# Step 4: Build musher strength features (pre-race priors)
# ────────────────────────────────────────────────────────────────
echo "Step 4: Building musher strength features..."
for year in $(seq "$YEAR_MIN" "$YEAR_MAX"); do
    python -m src.features.build_musher_strength --year "$year" 2>&1 || echo "  (skipped year=$year)"
done
echo ""

# ────────────────────────────────────────────────────────────────
# Step 5: Build snapshots (in-race features, now includes checkpoint_pct + dogs)
# ────────────────────────────────────────────────────────────────
echo "Step 5: Building snapshots..."
for year in $(seq "$YEAR_MIN" "$YEAR_MAX"); do
    python -m src.features.build_snapshots --year "$year" 2>&1 || echo "  (skipped year=$year — no splits data)"
done
echo ""

# ────────────────────────────────────────────────────────────────
# Step 6: Train models
# ────────────────────────────────────────────────────────────────
echo "Step 6: Training in-race model..."
python -m src.model.train_inrace_model \
    --train_start "$YEAR_MIN" \
    --train_end $((YEAR_MAX - 1)) \
    --test_year "$YEAR_MAX" \
    --reg_mode global
echo ""

echo "Step 6b: Fitting sigma from residuals..."
python -m src.model.fit_inrace_sigma \
    --train_start "$YEAR_MIN" \
    --train_end $((YEAR_MAX - 1)) \
    --sigma_method mad \
    --write_top_level
echo ""

echo "Step 6c: Training pre-race baseline (slim features)..."
python -m src.model.train_prerace_baseline \
    --train_start "$YEAR_MIN" \
    --train_end $((YEAR_MAX - 1)) \
    --test_year "$YEAR_MAX" \
    --feature_set slim \
    --target top10
echo ""

# ────────────────────────────────────────────────────────────────
# Step 7 (optional): Run backtests
# ────────────────────────────────────────────────────────────────
echo "Step 7: Running in-race backtest..."
python -m src.model.backtest_inrace \
    --start_year $((YEAR_MIN + 3)) \
    --end_year $((YEAR_MAX - 1)) \
    --checkpoints "5,10,15,20" \
    --n_sims 5000
echo ""

echo "Step 7b: Running pre-race backtest..."
python -m src.model.backtest_prerace_baseline \
    --start_year "$YEAR_MIN" \
    --end_year "$YEAR_MAX" \
    --feature_set slim \
    --target top10
echo ""

echo "=== Pipeline rebuild complete ==="

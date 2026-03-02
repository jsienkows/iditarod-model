# Iditarod Prediction Model

A machine learning system for predicting outcomes of the Iditarod Trail Sled Dog Race, combining **pre-race rankings** based on historical musher performance with **in-race predictions** that update dynamically as checkpoint data comes in.

## 2026 Pre-Race Rankings

| Rank | Musher | Win% | Top 5% | Top 10% | Finish% | 80% CI | Unc |
|------|--------|------|--------|---------|---------|--------|-----|
| 1 | Jessie Holmes | 11.9% | 50.3% | 73.0% | 90.0% | [1,5] | 1.0 |
| 2 | Matt Hall | 8.7% | 46.3% | 70.4% | 89.7% | [1,6] | 1.07 |
| 3 | Paige Drobny | 7.0% | 42.8% | 68.1% | 88.0% | [1,8] | 1.07 |
| 4 | Michelle Phillips | 5.7% | 44.2% | 68.0% | 86.0% | [2,9] | 1.07 |
| 5 | Travis Beals | 6.9% | 35.8% | 63.2% | 86.0% | [2,10] | 1.0 |
| 6 | Ryan Redington | 5.9% | 29.2% | 57.8% | 86.8% | [3,11] | 1.07 |
| 7 | Mille Porsild | 5.7% | 26.7% | 55.3% | 87.4% | [3,12] | 1.15 |
| 8 | Thomas Waerner 🔸 | 28.3% | 35.6% | 29.7% | 89.5% | [3,16] | 2.0 |
| 9 | Bailey Vitello | 0.3% | 20.4% | 49.2% | 86.7% | [5,16] | 1.63 |
| 10 | Peter Kaiser | 5.0% | 21.1% | 41.8% | 85.9% | [6,15] | 1.05 |
| 11 | Wade Marrs | 6.9% | 23.9% | 38.5% | 83.4% | [6,16] | 1.25 |
| 12 | Jessie Royer | 2.1% | 19.3% | 38.6% | 85.1% | [7,16] | 1.05 |
| 13 | Josi (Thyr) Shelley | 0.6% | 17.8% | 38.2% | 82.7% | [6,20] | 2.0 |
| 14 | Chad Stoddard | 4.0% | 19.4% | 30.1% | 81.1% | [7,20] | 2.0 |
| 15 | Jeff Deeter | 0.3% | 11.5% | 36.2% | 79.0% | [11,20] | 1.26 |
| 16 | Jesse Terry 🔹 | 0.0% | 5.1% | 10.7% | 84.9% | [15,28] | 1.8 |
| 17 | Jason Mackey | 0.0% | 7.1% | 21.4% | 82.8% | [14,23] | 1.26 |
| 18 | Lauro Eklund | 0.0% | 7.7% | 24.6% | 81.2% | [11,25] | 2.0 |
| 19 | Riley Dyche | 0.1% | 9.5% | 31.4% | 78.4% | [12,22] | 1.41 |
| 20 | Kevin Hansen 🔹 | 0.0% | 3.8% | 8.2% | 84.3% | [17,29] | 1.8 |
| 21 | Hanna Lyrek | 0.2% | 11.4% | 19.7% | 80.5% | [11,24] | 2.0 |
| 22 | Keaton Loebrich | 0.0% | 7.6% | 24.4% | 79.1% | [12,25] | 2.0 |
| 23 | Jody Potts-Joseph 🔹 | 0.0% | 2.4% | 5.4% | 82.8% | [20,32] | 1.8 |
| 24 | Jaye Foucher 🔹 | 0.0% | 2.4% | 5.4% | 82.8% | [20,33] | 1.8 |
| 25 | Gabe Dunham | 0.0% | 5.5% | 21.7% | 71.0% | [15,27] | 2.0 |
| 26 | Adam Lindenmuth 🔹 | 0.0% | 1.6% | 3.7% | 82.0% | [22,34] | 1.8 |
| 27 | Sam Paperman 🔹 | 0.0% | 1.6% | 3.7% | 82.0% | [22,34] | 1.8 |
| 28 | Sadie Lindquist 🔹 | 0.0% | 1.6% | 3.7% | 82.0% | [22,34] | 1.8 |
| 29 | Joseph Sabin 🔹 | 0.0% | 1.6% | 3.7% | 82.0% | [22,34] | 1.8 |
| 30 | Grayson Bruton | 0.0% | 4.2% | 6.4% | 60.2% | [21,33] | 2.0 |
| 31 | Sydnie Bahl 🔹 | 0.0% | 0.8% | 1.1% | 82.9% | [26,36] | 2.0 |
| 32 | Rohn Buser | 0.0% | 2.4% | 0.4% | 81.0% | [24,35] | 2.0 |
| 33 | Brenda Mackey 🔹 | 0.0% | 0.8% | 1.0% | 83.5% | [26,36] | 2.0 |
| 34 | Sam Martin 🔹 | 0.0% | 1.0% | 2.5% | 79.1% | [25,36] | 1.8 |
| 35 | Kjell Rokke 🔹 | 0.0% | 1.0% | 2.5% | 79.1% | [25,35] | 1.8 |
| 36 | Richie Beattie 🔹 | 0.0% | 1.2% | 0.4% | 81.7% | [27,36] | 2.0 |

🔹 Rookie &nbsp;&nbsp; 🔸 High-uncertainty contender

**80% CI** is the prediction interval on finishing rank from a 10,000-draw Monte Carlo simulation that perturbs each musher's probabilities with noise scaled by their uncertainty multiplier. **Unc** (uncertainty multiplier) ranges from 0.85 for veterans with 8+ finishes to 2.0 for mushers with thin race history or long absences. Mushers with wider intervals are more likely to significantly over- or under-perform their predicted rank.

Thomas Waerner is the most polarizing musher in the field: he has the highest raw win probability (28.3%) but only 29.7% top-10 rate, reflecting a boom-or-bust profile. His Unc of 2.0 (only 1 Iditarod finish plus a 6-year absence) gives him the widest prediction interval [3,16] of any top-10 musher.

## How It Works

### Pre-Race Model

Predicts each musher's probability of winning, finishing top 5, top 10, and finishing the race based on career history.

**Architecture:**
- Calibrated Logistic Regression with target-specific feature sets
- **Win probability**: derived from P(top 5) using a 9-feature model (v7), then sharpened via power function and normalized to sum to 100%
- **Top 5 / Top 10 / Finish probability**: 10-feature model (v8) adding last year's finish position
- **Composite ranking**: two weighted blends of per-target probability ranks:
  - *Field composite* (10/25/40/25): optimized for full-field ranking accuracy
  - *Contender composite* (28/26/29/17): ridge-optimized for top-10 separation, used for headline top-5 picks
- **Prediction intervals**: 10,000-draw Monte Carlo with per-musher uncertainty scaling based on career history depth

**Features** (exponentially decay-weighted over career, λ=0.70):

| Feature | Description |
|---------|-------------|
| `w_avg_finish_place` | Weighted average finish position |
| `w_pct_top10` | Weighted top-10 rate |
| `w_pct_finished` | Weighted finish rate |
| `pct_top5` | Career top-5 rate |
| `w_avg_time_behind_winner_seconds` | Weighted average time gap to winner |
| `n_finishes` | Total career finishes |
| `w_n_entries` | Weighted entry count |
| `years_since_last_entry` | Years since last race |
| `is_rookie` | First-time Iditarod indicator |
| `last_year_finish_place` | Previous year result *(v8 models only)* |

**Backtest performance** (11-year leave-one-year-out, 2015–2025):

| Metric | Value | Description |
|--------|-------|-------------|
| AUC (top-10) | 0.891 | Discrimination for top-10 identification |
| P@5 | 0.545 | Of predicted top 5, how many actually finished top 5 |
| P@10 | 0.618 | Of predicted top 10, how many actually finished top 10 |
| Spearman | 0.668 | Rank correlation: predicted vs actual finish order |
| Winner in top 3 | 54.5% (6/11) | Actual winner ranked in composite top 3 |
| Winner in top 5 | 90.9% (10/11) | Actual winner ranked in composite top 5 |

All metrics are evaluated on the **composite ranking** (the same blended ranking used for final predictions).

### In-Race Model

Updates predictions at each checkpoint using live race data. As the race progresses, the model shifts from relying on pre-race priors to current race performance.

**Architecture:**
- HistGradientBoosting regressor for remaining time prediction
- HistGradientBoosting classifier for finish probability
- Log-normal Monte Carlo simulation (20,000 draws) with per-musher uncertainty scaling
- **Prior decay blending**: checkpoint-dependent mix of full model (with career priors) and snapshot-only model (race data only), following hyperbolic decay w = 1/(1 + checkpoint). Priors dominate early (50% at CP1) and fade to near-zero by mid-race (9% at CP10), matching the empirical crossover where race data becomes more predictive than career history
- **Structural uncertainty**: per-musher noise scaling in the simulation — veterans with 8+ finishes get tighter distributions, rookies and thin-history mushers get wider ones, producing honest prediction intervals

**Key in-race features:** current position, pace vs. field median, gap to leader/10th place, cumulative rest, dogs remaining, leg-over-leg speed trends, pre-race musher strength, and race progress percentage.

## Project Structure

```
project/
├── predict_prerace_2026.py            # Generate 2026 pre-race rankings
├── inject_rookie_strength.py          # Assign synthetic strength priors to rookies
├── match_2026_mushers.py              # Match 2026 entrants to musher database IDs
├── rebuild_pipeline.sh                # Full pipeline rebuild (features → models → backtest)
├── rankings_2026.csv                  # Final 2026 pre-race rankings output
├── requirements.txt
├── .gitignore
│
├── improvements/                      # Model improvement analyses & patches
│   ├── train_snapshot_models.py       # Train snapshot-only models for prior decay
│   ├── 1_ridge_stacking.py            # Ridge meta-learner for composite weights
│   ├── 2_lognormal_noise.py           # Log-normal vs Gaussian noise comparison
│   ├── 3_bootstrap_ci.py              # Bootstrap confidence intervals on metrics
│   ├── 4_prior_decay_calibration.py   # Prior value diagnostic by checkpoint
│   ├── 5_structural_uncertainty.py    # Per-musher uncertainty multiplier library
│   ├── SUMMARY.md                     # Full writeup of all improvements
│   └── patch_*.py                     # Auto-patchers for production code
│
├── src/
│   ├── db.py                          # DuckDB connection & schema definition
│   │
│   ├── scrape/                        # Data collection from iditarod.com
│   │   ├── scrape_all_checkpoints.py  # Scrape checkpoint split times
│   │   ├── scrape_final_standings.py  # Scrape final race results
│   │   ├── build_entries.py           # Build entries table from standings
│   │   ├── build_entries_from_splits.py
│   │   ├── scrape_musher_dogs.py      # Dog team information
│   │   ├── parse_all_checkpoints.py   # Parse scraped HTML → splits table
│   │   ├── parse_one_checkpoint.py
│   │   ├── parse_helpers.py
│   │   ├── fetch.py                   # HTTP fetch with caching
│   │   └── download_one_page.py
│   │
│   ├── features/                      # Feature engineering
│   │   ├── build_musher_strength.py   # Pre-race musher features (60+ columns incl. Bayesian shrinkage)
│   │   ├── build_snapshots.py         # In-race checkpoint-level features
│   │   ├── build_historical_results.py# Unified historical results table
│   │   ├── checkpoint_distances.py    # Checkpoint → race progress % mapping
│   │   ├── race_context.py            # Route regime & year metadata
│   │   ├── build_weather_features.py  # NOAA weather data integration
│   │   └── reset_snapshots_table.py   # Utility to rebuild snapshots from scratch
│   │
│   ├── model/                         # Model training & evaluation
│   │   ├── train_inrace_model.py      # Train in-race HistGBT models
│   │   ├── predict_inrace.py          # Generate live race predictions
│   │   ├── backtest_inrace.py         # Rolling backtest for in-race model
│   │   ├── train_prerace_baseline.py  # Train pre-race logistic regression
│   │   ├── backtest_prerace_baseline.py
│   │   ├── fit_inrace_sigma.py        # Calibrate prediction uncertainty (MAD)
│   │   └── calibration_utils.py       # Shared calibration helpers
│   │
│   ├── eval/
│   │   └── diagnose_regression.py     # Model diagnostic utilities
│   │
│   └── db_migrations/                 # Schema evolution scripts
│       ├── add_new_snapshot_columns.py
│       └── add_gap_leg_delta_to_snapshots.py
│
├── data/
│   ├── noaa_iditarod_weather.csv      # Historical race-week weather data
│   └── db/                            # DuckDB database (gitignored)
│
└── models/                            # Trained model artifacts (gitignored)
    ├── inrace_*_model.joblib          # Full in-race models (with priors)
    └── inrace_*_snapshot.joblib       # Snapshot-only models (for prior decay)
```

## Setup

```bash
# Clone and install dependencies
git clone https://github.com/jsienkows/iditarod-model.git
cd iditarod-model/project
pip install -r requirements.txt

# Initialize database and scrape data (2006–2025)
python -m src.init_db
python -m src.scrape.scrape_all_checkpoints --year_min 2006 --year_max 2025
python -m src.scrape.scrape_final_standings --year_min 2006 --year_max 2025

# Build all features and train models
./rebuild_pipeline.sh 2006 2025

# Generate 2026 pre-race rankings
python -m src.features.build_musher_strength --year 2026
python inject_rookie_strength.py --write
python predict_prerace_2026.py --output rankings_2026.csv
```

## Data

All data is scraped from the [official Iditarod website](https://iditarod.com). The DuckDB database is not included in the repository but can be fully reconstructed by running the scraping and pipeline steps above.

Historical coverage: **2006–2025** (20 years, ~1,200 musher-year observations).

## Key Design Decisions

**Exponential decay weighting** (λ=0.70): Recent races count more than old ones. A race from 5 years ago gets ~17% the weight of last year's race. This balances recency with career body of work.

**Target-specific feature sets**: The win model uses 9 features (excluding `last_year_finish_place`) because it yields better winner identification in backtesting (82% in top 3 vs 64%). The ranking models add `last_year_finish_place` for better group identification (P@5 = 0.545 vs 0.436).

**Calibrated probabilities**: All models use Platt scaling (sigmoid calibration with 5-fold CV) so predicted probabilities are well-calibrated — a musher predicted at 30% P(top 10) should finish top 10 roughly 30% of the time.

**Bayesian shrinkage features**: `musher_strength` includes shrinkage estimates that pull small-sample statistics toward population base rates (k=5 pseudo-observations). These don't improve the ML model (which already learns sample-size adjustments via `n_finishes`) but are useful for direct display and volatility scoring.

**Rookie handling**: Rookies have no historical data, so `inject_rookie_strength.py` assigns synthetic priors based on qualification scores (1–5 scale mapped to estimated finish distributions).

## License

MIT
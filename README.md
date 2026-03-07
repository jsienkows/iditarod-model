# Iditarod Prediction Model

A machine learning system for predicting outcomes of the Iditarod Trail Sled Dog Race, combining **pre-race rankings** based on historical musher performance with **in-race predictions** that update dynamically as checkpoint data comes in.

## 2026 Pre-Race Rankings

Rankings use the **contender composite** (28% win + 26% top 5 + 29% top 10 + 17% finish), which outperforms the field composite at identifying the actual winner in backtesting (winner in top 3: 82% of years by win%, vs 55% by field composite).

| Rank | Musher | Win% | Top 5% | Top 10% | Finish% | 80% CI | Unc |
|------|--------|------|--------|---------|---------|--------|-----|
| 1 | Jessie Holmes | 15.6% | 50.3% | 73.0% | 90.0% | [1,4] | 1.0 |
| 2 | Matt Hall | 11.4% | 46.3% | 70.4% | 89.7% | [1,5] | 1.07 |
| 3 | Paige Drobny | 9.9% | 44.7% | 68.5% | 86.4% | [1,7] | 0.89 |
| 4 | Travis Beals | 8.5% | 36.7% | 62.6% | 84.4% | [3,9] | 0.85 |
| 5 | Michelle Phillips | 7.3% | 47.0% | 67.3% | 82.6% | [2,8] | 0.85 |
| 6 | Ryan Redington | 8.1% | 30.3% | 58.1% | 85.7% | [3,10] | 0.94 |
| 7 | Mille Porsild | 7.5% | 26.7% | 55.3% | 87.4% | [3,11] | 1.15 |
| 8 | Peter Kaiser | 9.3% | 25.3% | 44.4% | 82.1% | [6,14] | 0.85 |
| 9 | Wade Marrs | 10.3% | 26.1% | 39.7% | 80.3% | [7,15] | 0.99 |
| 10 | Bailey Vitello | 0.4% | 20.4% | 49.2% | 86.7% | [5,16] | 1.63 |
| 11 | Jessie Royer | 4.2% | 24.1% | 41.5% | 80.1% | [8,16] | 0.85 |
| 12 | Josi (Thyr) Shelley | 0.9% | 17.8% | 38.2% | 82.7% | [6,20] | 2.0 |
| 13 | Chad Stoddard | 5.3% | 19.4% | 30.1% | 81.1% | [7,21] | 2.0 |
| 14 | Jesse Terry 🔹 | 0.2% | 11.5% | 40.8% | 78.5% | [8,21] | 1.8 |
| 15 | Jeff Deeter | 0.5% | 11.8% | 36.3% | 78.3% | [11,20] | 1.15 |
| 16 | Riley Dyche | 0.1% | 9.5% | 31.4% | 78.4% | [12,23] | 1.41 |
| 17 | Hanna Lyrek | 0.2% | 11.4% | 19.7% | 80.5% | [11,25] | 2.0 |
| 18 | Lauro Eklund | 0.0% | 7.7% | 24.6% | 81.2% | [11,26] | 2.0 |
| 19 | Kevin Hansen 🔹 | 0.0% | 8.0% | 33.0% | 77.0% | [11,24] | 1.8 |
| 20 | Keaton Loebrich | 0.1% | 7.6% | 24.4% | 79.1% | [12,26] | 2.0 |
| 21 | Jason Mackey | 0.0% | 7.2% | 20.8% | 81.2% | [15,24] | 1.0 |
| 22 | Gabe Dunham | 0.0% | 5.5% | 21.7% | 71.0% | [15,29] | 2.0 |
| 23 | Adam Lindenmuth 🔹 | 0.0% | 4.4% | 22.0% | 73.8% | [16,29] | 1.8 |
| 24 | Richie Beattie 🔹 | 0.0% | 3.9% | 3.5% | 79.6% | [21,33] | 2.0 |
| 25 | Joseph Sabin 🔹 | 0.0% | 2.7% | 14.9% | 71.9% | [20,32] | 1.8 |
| 26 | Jaye Foucher 🔹 | 0.0% | 2.7% | 14.9% | 71.9% | [20,32] | 1.8 |
| 27 | Jody Potts-Joseph 🔹 | 0.0% | 2.7% | 14.9% | 71.9% | [19,32] | 1.8 |
| 28 | Rohn Buser | 0.0% | 4.1% | 1.6% | 73.0% | [24,33] | 2.0 |
| 29 | Grayson Bruton | 0.0% | 4.2% | 6.4% | 60.2% | [23,33] | 2.0 |
| 30 | Brenda Mackey 🔹 | 0.0% | 2.4% | 8.4% | 76.5% | [21,33] | 2.0 |
| 31 | Sadie Lindquist 🔹 | 0.0% | 1.6% | 9.7% | 66.7% | [24,34] | 1.8 |
| 32 | Sam Paperman 🔹 | 0.0% | 1.6% | 9.7% | 66.7% | [24,34] | 1.8 |
| 33 | Sam Martin 🔹 | 0.0% | 1.6% | 9.7% | 66.7% | [24,34] | 1.8 |
| 34 | Sydnie Bahl 🔹 | 0.0% | 1.5% | 5.5% | 71.7% | [24,34] | 2.0 |

🔹 Rookie

**80% CI** is the prediction interval from 10,000-sim Monte Carlo — the range where the model expects each musher to finish 80% of the time. **Unc** (uncertainty multiplier) ranges from 0.85 to 2.0 based on career history depth; higher values mean wider prediction intervals.

**Note on expedition mushers:** Thomas Waerner, Kjell Rokke, and Steve Curtis are competing as non-competitive expedition class mushers and have been excluded from rankings. Waerner was initially ranked with a field-leading 28.3% win probability based on his 2020 championship, but has confirmed he will be traveling at a non-competitive pace alongside expedition musher Rokke.

## How It Works

### Pre-Race Model

Predicts each musher's probability of winning, finishing top 5, top 10, and finishing the race based on career history.

**Architecture:**
- Calibrated Logistic Regression with target-specific feature sets
- **Win probability**: derived from P(top 5) using a 9-feature model (v7), then sharpened via power function and normalized to sum to 100%
- **Top 5 / Top 10 / Finish probability**: 10-feature model (v8) adding last year's finish position
- **Contender ranking**: weighted blend of per-target probability ranks (28% win + 26% top 5 + 29% top 10 + 17% finish) — optimized for identifying the winner
- **Field ranking**: alternative composite (10% win + 25% top 5 + 40% top 10 + 25% finish) — optimized for overall field accuracy
- **Prediction intervals**: 10,000-draw Monte Carlo simulation perturbing probabilities on logit scale with per-musher noise scaled by uncertainty multiplier

**Features** (exponentially decay-weighted over career, λ=0.70, 20-year lookback):

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
| Winner in top 3 (composite) | 54.5% (6/11) | Actual winner ranked in composite top 3 |
| Winner in top 5 (composite) | 90.9% (10/11) | Actual winner ranked in composite top 5 |
| Winner in top 3 (win%) | 81.8% (9/11) | Actual winner in top 3 by raw win probability |

### In-Race Model

Updates predictions at each checkpoint using live race data. As the race progresses, the model shifts from relying on pre-race priors to current race performance.

**Architecture:**
- HistGradientBoosting regressor for remaining time prediction
- HistGradientBoosting classifier for finish probability
- Dual-model blending: full model (with priors) and snapshot-only model (no priors), blended via prior decay weight that decreases as race progresses
- Log-normal simulation (20,000 draws) with per-musher structural uncertainty for position distributions
- Pre-race priors blended in early checkpoints, fading as race data accumulates

**Key in-race features:** current position, pace vs. field median, gap to leader/10th place, cumulative rest, dogs remaining, leg-over-leg speed trends, pre-race musher strength, and race progress percentage.

## Project Structure

```
project/
├── predict_prerace_2026.py            # Generate 2026 pre-race rankings
├── inject_rookie_strength.py          # Assign synthetic strength priors to rookies
├── match_2026_mushers.py              # Match 2026 entrants to musher database IDs
├── rebuild_pipeline.sh                # Full pipeline rebuild (features → models → backtest)
├── setup_2026_race.bat                # One-time pre-race setup (Windows)
├── update_inrace.bat                  # Live race update pipeline (Windows)
├── rankings_2026.csv                  # Final 2026 pre-race rankings output
├── requirements.txt
├── .gitignore
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
│   │   ├── build_musher_strength.py   # Pre-race musher features (60+ columns, 20yr lookback, λ=0.70 decay)
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
├── improvements/                      # Model improvement patches & analysis
│   ├── SUMMARY.md                     # Changelog of all improvements
│   ├── train_snapshot_models.py       # Train snapshot-only models for blending
│   └── patch_*.py                     # Various improvement patches
│
├── data/
│   ├── noaa_iditarod_weather.csv      # Historical race-week weather data
│   └── db/                            # DuckDB database (gitignored)
│
└── models/                            # Trained model artifacts
    ├── inrace_finish_model.joblib
    ├── inrace_remaining_time_model.joblib
    ├── inrace_finish_model_snapshot.joblib
    ├── inrace_remaining_time_model_snapshot.joblib
    └── inrace_metadata.json
```

## Setup

```bash
# Clone and install dependencies
git clone https://github.com/yourusername/iditarod-model.git
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

## Race Day Usage

```bash
# One-time pre-race setup
setup_2026_race.bat

# After each checkpoint (wait for leaders to depart):
update_inrace.bat CHECKPOINT_NUMBER
```

## Data

All data is scraped from the [official Iditarod website](https://iditarod.com). The DuckDB database is not included in the repository but can be fully reconstructed by running the scraping and pipeline steps above.

Historical coverage: **2006–2025** (20 years, ~1,200 musher-year observations).

## Key Design Decisions

**Exponential decay weighting** (λ=0.70): Recent races count more than old ones. A race from 5 years ago gets ~17% the weight of last year's race. This balances recency with career body of work.

**20-year lookback window**: Musher strength features are computed over a 20-year window (effectively the full dataset back to 2006). This ensures returning mushers with long absences still have their full history available, while exponential decay handles recency.

**Dual composite rankings**: The contender composite (28/26/29/17) weights win probability more heavily and is better at identifying the actual winner (top-3 hit rate: 82% by win% vs 55% by field composite). The field composite (10/25/40/25) produces better overall rank ordering across the full field.

**Target-specific feature sets**: The win model uses 9 features (excluding `last_year_finish_place`) because it yields better winner identification in backtesting. The ranking models add `last_year_finish_place` for better group identification.

**Calibrated probabilities**: All models use Platt scaling (sigmoid calibration with 5-fold CV) so predicted probabilities are well-calibrated.

**Prediction intervals**: Monte Carlo simulation (10,000 draws) perturbs each musher's probabilities on a logit scale with noise proportional to their uncertainty multiplier. Mushers with thin career histories get wider intervals. The 80% CI shows the 10th–90th percentile of simulated finishing positions.

**Rookie handling**: Rookies have no Iditarod history, so `inject_rookie_strength.py` assigns synthetic priors based on qualification scores (1–5 scale) mapped to estimated finish distributions including `w_avg_time_behind_winner_seconds`, the model's most predictive feature. Scores are informed by qualifying race results, kennel affiliations, and pre-race reporting.

**In-race prior blending**: The in-race model uses dual-model blending — a full model incorporating pre-race musher strength priors and a snapshot-only model using purely in-race features. A prior decay weight smoothly transitions from prior-heavy early in the race to snapshot-dominant as checkpoint data accumulates. Log-normal noise and per-musher structural uncertainty produce realistic prediction intervals.

## License

MIT
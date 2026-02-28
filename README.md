# Iditarod Prediction Model

A machine learning system for predicting outcomes of the Iditarod Trail Sled Dog Race, combining **pre-race rankings** based on historical musher performance with **in-race predictions** that update dynamically as checkpoint data comes in.

## 2026 Pre-Race Rankings

| Rank | Musher | Win% | Top 5% | Top 10% | Finish% | Volatility |
|------|--------|------|--------|---------|---------|------------|
| 1 | Jessie Holmes | 11.9% | 50.3% | 73.0% | 90.0% | 16.7 |
| 2 | Matt Hall | 8.7% | 46.3% | 70.4% | 89.7% | 19.4 |
| 3 | Paige Drobny | 7.0% | 42.8% | 68.1% | 88.0% | 20.5 |
| 4 | Michelle Phillips | 5.7% | 44.2% | 68.0% | 86.0% | 20.2 |
| 5 | Travis Beals | 6.9% | 35.8% | 63.2% | 86.0% | 15.9 |
| 6 | Ryan Redington | 5.9% | 29.2% | 57.8% | 86.8% | 18.6 |
| 7 | Mille Porsild | 5.7% | 26.7% | 55.3% | 87.4% | 22.0 |
| 8 | Thomas Waerner 🔸 | 28.3% | 35.6% | 29.7% | 89.5% | 61.3 |
| 9 | Bailey Vitello | 0.3% | 20.4% | 49.2% | 86.7% | 34.5 |
| 10 | Peter Kaiser | 5.0% | 21.1% | 41.8% | 85.9% | 19.2 |
| 11 | Wade Marrs | 6.9% | 23.9% | 38.5% | 83.4% | 29.4 |
| 12 | Jessie Royer | 2.1% | 19.3% | 38.6% | 85.1% | 22.9 |
| 13 | Josi (Thyr) Shelley | 0.6% | 17.8% | 38.2% | 82.7% | 33.9 |
| 14 | Chad Stoddard | 4.0% | 19.4% | 30.1% | 81.1% | 47.6 |
| 15 | Jeff Deeter | 0.3% | 11.5% | 36.2% | 79.0% | 23.4 |
| 16 | Jesse Terry 🔹 | 0.0% | 5.1% | 10.7% | 84.9% | 52.0 |
| 17 | Jason Mackey | 0.0% | 7.1% | 21.4% | 82.8% | 22.4 |
| 18 | Lauro Eklund | 0.0% | 7.7% | 24.6% | 81.2% | 33.5 |
| 19 | Riley Dyche | 0.1% | 9.5% | 31.4% | 78.4% | 29.3 |
| 20 | Kevin Hansen 🔹 | 0.0% | 3.8% | 8.2% | 84.3% | 52.0 |
| 21 | Hanna Lyrek | 0.2% | 11.4% | 19.7% | 80.5% | 40.5 |
| 22 | Keaton Loebrich | 0.0% | 7.6% | 24.4% | 79.1% | 30.4 |
| 23 | Jody Potts-Joseph 🔹 | 0.0% | 2.4% | 5.4% | 82.8% | 52.0 |
| 24 | Jaye Foucher 🔹 | 0.0% | 2.4% | 5.4% | 82.8% | 52.0 |
| 25 | Gabe Dunham | 0.0% | 5.5% | 21.7% | 71.0% | 31.0 |
| 26 | Adam Lindenmuth 🔹 | 0.0% | 1.6% | 3.7% | 82.0% | 52.0 |
| 27 | Sam Paperman 🔹 | 0.0% | 1.6% | 3.7% | 82.0% | 52.0 |
| 28 | Sadie Lindquist 🔹 | 0.0% | 1.6% | 3.7% | 82.0% | 52.0 |
| 29 | Joseph Sabin 🔹 | 0.0% | 1.6% | 3.7% | 82.0% | 52.0 |
| 30 | Grayson Bruton | 0.0% | 4.2% | 6.4% | 60.2% | 47.0 |
| 31 | Sydnie Bahl 🔹 | 0.0% | 0.8% | 1.1% | 82.9% | 34.8 |
| 32 | Rohn Buser | 0.0% | 2.4% | 0.4% | 81.0% | 60.1 |
| 33 | Brenda Mackey 🔹 | 0.0% | 0.8% | 1.0% | 83.5% | 34.8 |
| 34 | Sam Martin 🔹 | 0.0% | 1.0% | 2.5% | 79.1% | 52.0 |
| 35 | Kjell Rokke 🔹 | 0.0% | 1.0% | 2.5% | 79.1% | 52.0 |
| 36 | Richie Beattie 🔹 | 0.0% | 1.2% | 0.4% | 81.7% | 51.5 |

🔹 Rookie &nbsp;&nbsp; 🔸 High-volatility contender

**Volatility** measures outcome uncertainty (0–100). Higher values indicate mushers with wider ranges of possible outcomes — driven by small sample sizes, long absences, or inconsistent career results. Thomas Waerner has the highest volatility among contenders (61.3) due to only 2 career Iditarod races and a 6-year absence, despite being a former champion with the highest raw win probability in the field.

## How It Works

### Pre-Race Model

Predicts each musher's probability of winning, finishing top 5, top 10, and finishing the race based on career history.

**Architecture:**
- Calibrated Logistic Regression with target-specific feature sets
- **Win probability**: derived from P(top 5) using a 9-feature model (v7), then sharpened via power function and normalized to sum to 100%
- **Top 5 / Top 10 / Finish probability**: 10-feature model (v8) adding last year's finish position
- **Composite ranking**: weighted blend of per-target probability ranks (10% win + 25% top 5 + 40% top 10 + 25% finish)

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
- Gaussian simulation (5,000 draws) for position distributions
- Pre-race priors blended in early checkpoints, fading as race data accumulates

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

"""
Build race-year-level weather features from NOAA daily summaries.

Reads data/noaa_iditarod_weather.csv and computes per-year averages
across stations for the race window (March 1-20).

Features created:
  - weather_avg_temp_c:   Average of (TMAX+TMIN)/2 across stations, °C
  - weather_avg_tmax_c:   Average daily high across stations, °C
  - weather_avg_tmin_c:   Average daily low across stations, °C
  - weather_total_prcp_mm: Total precipitation across all station-days, mm
  - weather_total_snow_mm: Total snowfall across all station-days, mm
  - weather_temp_anomaly:  Deviation from the grand mean (z-score style)

Usage:
    python -m src.features.build_weather_features
"""

import os
import csv
import statistics
import argparse
from src.db import connect


WEATHER_CSV = os.path.join("data", "noaa_iditarod_weather.csv")

# Race window: March 1-20 covers most race start-to-finish dates
RACE_DAY_START = 1
RACE_DAY_END = 20


def load_noaa_csv(path: str) -> list[dict]:
    """Load and parse the NOAA weather CSV."""
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def to_float(v: str):
    """Convert NOAA tenths-of-unit string to float. Returns None for blanks."""
    v = v.strip()
    if v == "" or v == "null":
        return None
    return int(v) / 10.0


def compute_yearly_weather(rows: list[dict]) -> dict[int, dict]:
    """
    Compute per-year weather summaries from daily station data.
    Returns {year: {feature_name: value, ...}, ...}
    """
    # Group rows by year, filter to race window
    years = sorted(set(r["DATE"][:4] for r in rows))
    yearly = {}

    for year_str in years:
        year = int(year_str)
        yr_rows = [
            r for r in rows
            if r["DATE"][:4] == year_str
            and RACE_DAY_START <= int(r["DATE"][8:10]) <= RACE_DAY_END
        ]

        tmaxs = [to_float(r["TMAX"]) for r in yr_rows]
        tmins = [to_float(r["TMIN"]) for r in yr_rows]
        prcps = [to_float(r["PRCP"]) for r in yr_rows]
        snows = [to_float(r["SNOW"]) for r in yr_rows]

        # Filter None values
        tmaxs = [v for v in tmaxs if v is not None]
        tmins = [v for v in tmins if v is not None]
        prcps = [v for v in prcps if v is not None]
        snows = [v for v in snows if v is not None]

        avg_tmax = statistics.mean(tmaxs) if tmaxs else None
        avg_tmin = statistics.mean(tmins) if tmins else None
        avg_temp = (
            statistics.mean([(tx + tn) / 2 for tx, tn in zip(tmaxs, tmins)])
            if tmaxs and tmins
            else None
        )
        total_prcp = sum(prcps) if prcps else 0.0
        total_snow = sum(snows) if snows else 0.0
        n_stations = len(set(r["STATION"] for r in yr_rows))

        yearly[year] = {
            "weather_avg_temp_c": round(avg_temp, 2) if avg_temp else None,
            "weather_avg_tmax_c": round(avg_tmax, 2) if avg_tmax else None,
            "weather_avg_tmin_c": round(avg_tmin, 2) if avg_tmin else None,
            "weather_total_prcp_mm": round(total_prcp, 1),
            "weather_total_snow_mm": round(total_snow, 1),
            "weather_n_stations": n_stations,
        }

    # Compute temperature anomaly (how many SDs from grand mean)
    temps = [v["weather_avg_temp_c"] for v in yearly.values() if v["weather_avg_temp_c"] is not None]
    if len(temps) >= 3:
        grand_mean = statistics.mean(temps)
        grand_std = statistics.stdev(temps)
        for year, feats in yearly.items():
            if feats["weather_avg_temp_c"] is not None and grand_std > 0:
                feats["weather_temp_anomaly"] = round(
                    (feats["weather_avg_temp_c"] - grand_mean) / grand_std, 3
                )
            else:
                feats["weather_temp_anomaly"] = None
    else:
        for feats in yearly.values():
            feats["weather_temp_anomaly"] = None

    return yearly


def build_weather_table(con, yearly: dict[int, dict]):
    """Create and populate the weather_features table."""
    con.execute("DROP TABLE IF EXISTS weather_features")
    con.execute("""
        CREATE TABLE weather_features (
            year INTEGER PRIMARY KEY,
            weather_avg_temp_c DOUBLE,
            weather_avg_tmax_c DOUBLE,
            weather_avg_tmin_c DOUBLE,
            weather_total_prcp_mm DOUBLE,
            weather_total_snow_mm DOUBLE,
            weather_temp_anomaly DOUBLE,
            weather_n_stations INTEGER
        )
    """)

    for year, feats in sorted(yearly.items()):
        con.execute(
            """
            INSERT INTO weather_features
                (year, weather_avg_temp_c, weather_avg_tmax_c, weather_avg_tmin_c,
                 weather_total_prcp_mm, weather_total_snow_mm,
                 weather_temp_anomaly, weather_n_stations)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                year,
                feats["weather_avg_temp_c"],
                feats["weather_avg_tmax_c"],
                feats["weather_avg_tmin_c"],
                feats["weather_total_prcp_mm"],
                feats["weather_total_snow_mm"],
                feats["weather_temp_anomaly"],
                feats["weather_n_stations"],
            ],
        )

    print(f"weather_features table: {len(yearly)} years inserted ✅")

    # Print summary
    df = con.execute(
        "SELECT year, weather_avg_temp_c, weather_temp_anomaly, weather_total_snow_mm FROM weather_features ORDER BY year"
    ).df()
    print(df.to_string(index=False))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=WEATHER_CSV, help="Path to NOAA weather CSV")
    args = ap.parse_args()

    if not os.path.exists(args.csv):
        print(f"ERROR: {args.csv} not found. Run fetch_noaa_weather.py first.")
        return

    rows = load_noaa_csv(args.csv)
    print(f"Loaded {len(rows)} daily observations from {args.csv}")

    yearly = compute_yearly_weather(rows)
    print(f"Computed weather for {len(yearly)} years")

    con = connect()
    build_weather_table(con, yearly)


if __name__ == "__main__":
    main()
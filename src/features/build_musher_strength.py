import argparse
import pandas as pd
import numpy as np
from src.db import connect


def add_window_aggs(hist: pd.DataFrame, window: int, prefix: str) -> pd.Series:
    """
    hist is prior-years history for a given (target_year, musher_id)
    Must include: finished (0/1), top10 (0/1), finish_place, time_behind_winner_seconds, year
    """
    if hist.empty:
        return pd.Series(dtype="float64")

    # last N years only (relative to most recent year present in hist)
    y_max = int(hist["year"].max())
    y_min = y_max - (window - 1)
    d = hist[hist["year"].between(y_min, y_max)].copy()

    if d.empty:
        return pd.Series(dtype="float64")

    n_entries = len(d)
    n_finishes = int(d["finished"].sum())

    out = {
        f"{prefix}n_entries": float(n_entries),
        f"{prefix}n_finishes": float(n_finishes),
        f"{prefix}pct_finished": (n_finishes / n_entries) if n_entries else np.nan,
        f"{prefix}pct_top10": float(d["top10"].mean()) if n_entries else np.nan,
    }

    # Only compute place/time stats on finishers (avoid DNFs / missing)
    fin = d[d["finished"] == 1].copy()
    if not fin.empty:
        out.update({
            f"{prefix}avg_finish_place": float(fin["finish_place"].mean()),
            f"{prefix}best_finish_place": float(fin["finish_place"].min()),
            f"{prefix}avg_time_behind_winner_seconds": float(fin["time_behind_winner_seconds"].mean()),
        })
    else:
        out.update({
            f"{prefix}avg_finish_place": np.nan,
            f"{prefix}best_finish_place": np.nan,
            f"{prefix}avg_time_behind_winner_seconds": np.nan,
        })

    return pd.Series(out)


def add_weighted_aggs(hist: pd.DataFrame, target_year: int, decay: float = 0.70) -> pd.Series:
    """
    Exponentially decayed weights: weight = decay ** (target_year - year)
    """
    if hist.empty:
        return pd.Series(dtype="float64")

    d = hist.copy()
    d["age"] = target_year - d["year"]
    d["w"] = (decay ** d["age"]).astype(float)

    w_sum = float(d["w"].sum())
    if w_sum <= 0:
        return pd.Series(dtype="float64")

    out = {
        "w_n_entries": w_sum,
        "w_pct_finished": float(np.average(d["finished"], weights=d["w"])),
        "w_pct_top10": float(np.average(d["top10"], weights=d["w"])),
    }

    fin = d[d["finished"] == 1].copy()
    if not fin.empty:
        w_fin = float(fin["w"].sum())
        out["w_avg_finish_place"] = float(np.average(fin["finish_place"], weights=fin["w"]))
        out["w_avg_time_behind_winner_seconds"] = float(np.average(fin["time_behind_winner_seconds"], weights=fin["w"]))
        out["w_finishes_weight"] = w_fin
    else:
        out["w_avg_finish_place"] = np.nan
        out["w_avg_time_behind_winner_seconds"] = np.nan
        out["w_finishes_weight"] = 0.0

    return pd.Series(out)


def _ensure_column(con, table: str, col: str, col_type: str):
    """
    DuckDB supports ADD COLUMN IF NOT EXISTS in newer versions, but to be safe we
    try with IF NOT EXISTS and fall back to plain ADD COLUMN inside try/except.
    """
    try:
        con.execute(f"ALTER TABLE {table} ADD COLUMN IF NOT EXISTS {col} {col_type}")
        return
    except Exception:
        pass
    try:
        con.execute(f"ALTER TABLE {table} ADD COLUMN {col} {col_type}")
    except Exception:
        pass


def _ensure_out_has_columns(out: pd.DataFrame, cols_defaults: dict) -> pd.DataFrame:
    """
    Ensure out has every key in cols_defaults. If missing, create with default.
    """
    for c, default in cols_defaults.items():
        if c not in out.columns:
            out[c] = default
    return out


# Option A: interpretable summary stats (no learned rating yet)
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--year", type=int, required=True, help="Target race year to build pre-race strength features for")
    ap.add_argument("--lookback", type=int, default=20, help="How many prior years to use (default: 20)")
    args = ap.parse_args()

    year = args.year
    lookback = args.lookback

    con = connect()

    # --- Create output table if missing ---
    con.execute("""
    CREATE TABLE IF NOT EXISTS musher_strength (
        year INTEGER,
        musher_id TEXT,

        n_years INTEGER,
        n_entries INTEGER,
        n_finishes INTEGER,

        pct_finished DOUBLE,
        pct_top10 DOUBLE,
        pct_top5 DOUBLE,
        pct_win DOUBLE,

        avg_finish_place DOUBLE,
        median_finish_place DOUBLE,
        best_finish_place INTEGER,

        avg_finish_time_seconds DOUBLE,
        median_finish_time_seconds DOUBLE,

        avg_time_behind_winner_seconds DOUBLE,
        median_time_behind_winner_seconds DOUBLE,

        last_year_finish_place INTEGER,
        last_year_finished INTEGER,
        last_year_time_behind_winner_seconds BIGINT,

        years_since_last_entry INTEGER,
        is_rookie INTEGER,

        PRIMARY KEY (year, musher_id)
    );
    """)

    # --- Ensure base columns exist (safe to rerun) ---
    _ensure_column(con, "musher_strength", "years_since_last_entry", "INTEGER")
    _ensure_column(con, "musher_strength", "is_rookie", "INTEGER")

    # --- Ensure recency-weighted columns exist (safe to rerun) ---
    _ensure_column(con, "musher_strength", "last3_n_entries", "DOUBLE")
    _ensure_column(con, "musher_strength", "last3_n_finishes", "DOUBLE")
    _ensure_column(con, "musher_strength", "last3_pct_finished", "DOUBLE")
    _ensure_column(con, "musher_strength", "last3_pct_top10", "DOUBLE")
    _ensure_column(con, "musher_strength", "last3_avg_finish_place", "DOUBLE")
    _ensure_column(con, "musher_strength", "last3_best_finish_place", "DOUBLE")
    _ensure_column(con, "musher_strength", "last3_avg_time_behind_winner_seconds", "DOUBLE")

    _ensure_column(con, "musher_strength", "last5_n_entries", "DOUBLE")
    _ensure_column(con, "musher_strength", "last5_n_finishes", "DOUBLE")
    _ensure_column(con, "musher_strength", "last5_pct_finished", "DOUBLE")
    _ensure_column(con, "musher_strength", "last5_pct_top10", "DOUBLE")
    _ensure_column(con, "musher_strength", "last5_avg_finish_place", "DOUBLE")
    _ensure_column(con, "musher_strength", "last5_best_finish_place", "DOUBLE")
    _ensure_column(con, "musher_strength", "last5_avg_time_behind_winner_seconds", "DOUBLE")

    _ensure_column(con, "musher_strength", "w_n_entries", "DOUBLE")
    _ensure_column(con, "musher_strength", "w_pct_finished", "DOUBLE")
    _ensure_column(con, "musher_strength", "w_pct_top10", "DOUBLE")
    _ensure_column(con, "musher_strength", "w_avg_finish_place", "DOUBLE")
    _ensure_column(con, "musher_strength", "w_avg_time_behind_winner_seconds", "DOUBLE")
    _ensure_column(con, "musher_strength", "w_finishes_weight", "DOUBLE")

    # --- Ensure bucket columns exist (safe to rerun) ---
    # Peak ability
    _ensure_column(con, "musher_strength", "peak_best_finish_place", "DOUBLE")
    _ensure_column(con, "musher_strength", "peak_pct_top10", "DOUBLE")
    _ensure_column(con, "musher_strength", "peak_pct_top5", "DOUBLE")
    _ensure_column(con, "musher_strength", "peak_pct_win", "DOUBLE")

    # Current form
    _ensure_column(con, "musher_strength", "form_last3_avg_finish_place", "DOUBLE")
    _ensure_column(con, "musher_strength", "form_last5_avg_finish_place", "DOUBLE")
    _ensure_column(con, "musher_strength", "form_last3_pct_top10", "DOUBLE")
    _ensure_column(con, "musher_strength", "form_last5_pct_top10", "DOUBLE")
    _ensure_column(con, "musher_strength", "form_w_avg_finish_place", "DOUBLE")
    _ensure_column(con, "musher_strength", "form_w_pct_top10", "DOUBLE")

    # Experience / volume
    _ensure_column(con, "musher_strength", "exp_n_entries", "DOUBLE")
    _ensure_column(con, "musher_strength", "exp_n_finishes", "DOUBLE")
    _ensure_column(con, "musher_strength", "exp_n_years", "DOUBLE")
    _ensure_column(con, "musher_strength", "exp_w_entries", "DOUBLE")

    # Rust / comeback
    _ensure_column(con, "musher_strength", "rust_years_since_last_entry", "INTEGER")
    _ensure_column(con, "musher_strength", "rust_is_rookie", "INTEGER")

    # Consistency / reliability
    _ensure_column(con, "musher_strength", "cons_pct_finished", "DOUBLE")
    _ensure_column(con, "musher_strength", "cons_last5_pct_finished", "DOUBLE")
    _ensure_column(con, "musher_strength", "cons_w_pct_finished", "DOUBLE")

    # Career stage / trajectory
    _ensure_column(con, "musher_strength", "career_race_number", "INTEGER")
    _ensure_column(con, "musher_strength", "trajectory", "DOUBLE")
    _ensure_column(con, "musher_strength", "last_year_improvement", "DOUBLE")

    # Bayesian shrinkage features (small-sample correction)
    _ensure_column(con, "musher_strength", "shrunk_pct_top10", "DOUBLE")
    _ensure_column(con, "musher_strength", "shrunk_pct_top5", "DOUBLE")
    _ensure_column(con, "musher_strength", "shrunk_pct_win", "DOUBLE")
    _ensure_column(con, "musher_strength", "shrunk_pct_finished", "DOUBLE")
    _ensure_column(con, "musher_strength", "shrunk_w_avg_finish_place", "DOUBLE")
    _ensure_column(con, "musher_strength", "shrunk_w_avg_time_behind_winner", "DOUBLE")

    # Variance / spread features
    _ensure_column(con, "musher_strength", "std_finish_place", "DOUBLE")
    _ensure_column(con, "musher_strength", "finish_place_range", "DOUBLE")

    # Confidence weight
    _ensure_column(con, "musher_strength", "confidence_weight", "DOUBLE")

    # --- All mushers entered in the target year ---
    entrants = con.execute("""
        SELECT musher_id
        FROM entries
        WHERE year = ?
    """, [year]).df()

    if entrants.empty:
        raise RuntimeError(f"No entries found for year={year}. Did you scrape standings/entries for that year?")

    # --- Years since last entry (ALL prior entries in DB; rookies remain NULL) ---
    last_entry = con.execute("""
        SELECT musher_id, MAX(year) AS last_entry_year
        FROM entries
        WHERE year < ?
        GROUP BY 1
    """, [year]).df()

    # --- Pull prior-year results for lookback window (for feature aggs only) ---
    hist = con.execute("""
        WITH base AS (
            SELECT
                e.year,
                e.musher_id,
                e.finish_place,
                e.finish_time_seconds,
                CASE WHEN e.finish_place IS NOT NULL THEN 1 ELSE 0 END AS finished,
                CASE WHEN e.finish_place IS NOT NULL AND e.finish_place <= 10 THEN 1 ELSE 0 END AS top10,
                CASE WHEN e.finish_place IS NOT NULL AND e.finish_place <= 5 THEN 1 ELSE 0 END AS top5,
                CASE WHEN e.finish_place = 1 THEN 1 ELSE 0 END AS won
            FROM entries e
            WHERE e.year >= ? AND e.year < ?
        ),
        winners AS (
            SELECT
                year,
                MIN(finish_time_seconds) AS winner_time_seconds
            FROM base
            WHERE finished = 1 AND finish_time_seconds IS NOT NULL
            GROUP BY year
        )
        SELECT
            b.*,
            CASE
                WHEN b.finished = 1 AND b.finish_time_seconds IS NOT NULL AND w.winner_time_seconds IS NOT NULL
                THEN (b.finish_time_seconds - w.winner_time_seconds)
                ELSE NULL
            END AS time_behind_winner_seconds
        FROM base b
        LEFT JOIN winners w USING(year)
    """, [year - lookback, year]).df()

    if hist.empty:
        # No lookback-window history at all; still output rows per entrant
        out = entrants.copy()
        out["year"] = year

        for c in [
            "n_years", "n_entries", "n_finishes",
            "pct_finished", "pct_top10", "pct_top5", "pct_win",
            "avg_finish_place", "median_finish_place", "best_finish_place",
            "avg_finish_time_seconds", "median_finish_time_seconds",
            "avg_time_behind_winner_seconds", "median_time_behind_winner_seconds",
            "last_year_finish_place", "last_year_finished", "last_year_time_behind_winner_seconds",
        ]:
            out[c] = None

        out[["n_years", "n_entries", "n_finishes"]] = 0
        out[["pct_finished", "pct_top10", "pct_top5", "pct_win"]] = 0.0

        for c in [
            "last3_n_entries", "last3_n_finishes", "last3_pct_finished", "last3_pct_top10",
            "last3_avg_finish_place", "last3_best_finish_place", "last3_avg_time_behind_winner_seconds",
            "last5_n_entries", "last5_n_finishes", "last5_pct_finished", "last5_pct_top10",
            "last5_avg_finish_place", "last5_best_finish_place", "last5_avg_time_behind_winner_seconds",
            "w_n_entries", "w_pct_finished", "w_pct_top10", "w_avg_finish_place",
            "w_avg_time_behind_winner_seconds", "w_finishes_weight",
        ]:
            out[c] = np.nan

        # New feature defaults (empty history)
        for c in [
            "shrunk_pct_top10", "shrunk_pct_top5", "shrunk_pct_win", "shrunk_pct_finished",
            "shrunk_w_avg_finish_place", "shrunk_w_avg_time_behind_winner",
            "std_finish_place", "finish_place_range",
        ]:
            out[c] = np.nan
        out["confidence_weight"] = 0.0

    else:
        # Aggregate summary stats
        g = hist.groupby("musher_id", dropna=False)

        agg = pd.DataFrame({
            "n_years": g["year"].nunique(),
            "n_entries": g.size(),
            "n_finishes": g["finished"].sum(),

            "pct_finished": g["finished"].mean(),
            "pct_top10": g["top10"].mean(),
            "pct_top5": g["top5"].mean(),
            "pct_win": g["won"].mean(),

            "avg_finish_place": g["finish_place"].mean(),
            "median_finish_place": g["finish_place"].median(),
            "best_finish_place": g["finish_place"].min(),

            "avg_finish_time_seconds": g["finish_time_seconds"].mean(),
            "median_finish_time_seconds": g["finish_time_seconds"].median(),

            "avg_time_behind_winner_seconds": g["time_behind_winner_seconds"].mean(),
            "median_time_behind_winner_seconds": g["time_behind_winner_seconds"].median(),

            "std_finish_place": g["finish_place"].std(),
            "finish_place_range": g["finish_place"].max() - g["finish_place"].min(),
        }).reset_index()

        # Last-year features (year-1 only)
        last = hist[hist["year"] == (year - 1)].copy()
        last_g = last.groupby("musher_id", dropna=False)
        last_agg = pd.DataFrame({
            "last_year_finish_place": last_g["finish_place"].min(),
            "last_year_finished": last_g["finished"].max(),
            "last_year_time_behind_winner_seconds": last_g["time_behind_winner_seconds"].min(),
        }).reset_index()

        # Apply recency-weighted helpers (computed from lookback-limited hist)
        rec_rows = []
        for mid in entrants["musher_id"].tolist():
            h = hist[hist["musher_id"] == mid].copy()

            s3 = add_window_aggs(h, window=3, prefix="last3_")
            s5 = add_window_aggs(h, window=5, prefix="last5_")
            sw = add_weighted_aggs(h, target_year=year, decay=0.70)

            row = {"musher_id": mid}
            if not s3.empty:
                row.update(s3.to_dict())
            if not s5.empty:
                row.update(s5.to_dict())
            if not sw.empty:
                row.update(sw.to_dict())

            rec_rows.append(row)

        recent_df = pd.DataFrame(rec_rows)

        out = (
            entrants
            .merge(agg, on="musher_id", how="left")
            .merge(last_agg, on="musher_id", how="left")
            .merge(recent_df, on="musher_id", how="left")
        )
        out["year"] = year

        # Fill “no history” defaults for base cols
        out["n_years"] = out["n_years"].fillna(0).astype(int)
        out["n_entries"] = out["n_entries"].fillna(0).astype(int)
        out["n_finishes"] = out["n_finishes"].fillna(0).astype(int)
        for c in ["pct_finished", "pct_top10", "pct_top5", "pct_win"]:
            out[c] = out[c].fillna(0.0)

    # --- Merge last_entry for ALL entrants and compute years_since_last_entry ---
    out = out.merge(last_entry, on="musher_id", how="left")
    out["years_since_last_entry"] = (out["year"] - out["last_entry_year"]).astype("Int64")

    # rookies explicitly flagged; years_since_last_entry stays NULL/<NA> for rookies
    out["is_rookie"] = out["years_since_last_entry"].isna().astype("int64")
    out = out.drop(columns=["last_entry_year"])

    # --- Ensure all bucket columns exist in out even if hist.empty ---
    # (This prevents missing columns -> NULL inserts.)
    out = _ensure_out_has_columns(out, {
        "peak_best_finish_place": np.nan,
        "peak_pct_top10": np.nan,
        "peak_pct_top5": np.nan,
        "peak_pct_win": np.nan,

        "form_last3_avg_finish_place": np.nan,
        "form_last5_avg_finish_place": np.nan,
        "form_last3_pct_top10": np.nan,
        "form_last5_pct_top10": np.nan,
        "form_w_avg_finish_place": np.nan,
        "form_w_pct_top10": np.nan,

        "exp_n_entries": 0.0,
        "exp_n_finishes": 0.0,
        "exp_n_years": 0.0,
        "exp_w_entries": np.nan,

        "rust_years_since_last_entry": pd.Series([pd.NA] * len(out), dtype="Int64"),
        "rust_is_rookie": 0,

        "cons_pct_finished": np.nan,
        "cons_last5_pct_finished": np.nan,
        "cons_w_pct_finished": np.nan,

        "career_race_number": 1,
        "trajectory": np.nan,
        "last_year_improvement": np.nan,

        "shrunk_pct_top10": np.nan,
        "shrunk_pct_top5": np.nan,
        "shrunk_pct_win": np.nan,
        "shrunk_pct_finished": np.nan,
        "shrunk_w_avg_finish_place": np.nan,
        "shrunk_w_avg_time_behind_winner": np.nan,
        "std_finish_place": np.nan,
        "finish_place_range": np.nan,
        "confidence_weight": 0.0,
    })

    # --- Derive conceptual bucket features (computed from existing raw stats) ---
    # Peak ability
    out["peak_best_finish_place"] = pd.to_numeric(out.get("best_finish_place"), errors="coerce")
    out["peak_pct_top10"] = pd.to_numeric(out.get("pct_top10"), errors="coerce")
    out["peak_pct_top5"] = pd.to_numeric(out.get("pct_top5"), errors="coerce")
    out["peak_pct_win"] = pd.to_numeric(out.get("pct_win"), errors="coerce")

    # Current form
    out["form_last3_avg_finish_place"] = pd.to_numeric(out.get("last3_avg_finish_place"), errors="coerce")
    out["form_last5_avg_finish_place"] = pd.to_numeric(out.get("last5_avg_finish_place"), errors="coerce")
    out["form_last3_pct_top10"] = pd.to_numeric(out.get("last3_pct_top10"), errors="coerce")
    out["form_last5_pct_top10"] = pd.to_numeric(out.get("last5_pct_top10"), errors="coerce")
    out["form_w_avg_finish_place"] = pd.to_numeric(out.get("w_avg_finish_place"), errors="coerce")
    out["form_w_pct_top10"] = pd.to_numeric(out.get("w_pct_top10"), errors="coerce")

    # Experience / volume
    out["exp_n_entries"] = pd.to_numeric(out.get("n_entries"), errors="coerce")
    out["exp_n_finishes"] = pd.to_numeric(out.get("n_finishes"), errors="coerce")
    out["exp_n_years"] = pd.to_numeric(out.get("n_years"), errors="coerce")
    out["exp_w_entries"] = pd.to_numeric(out.get("w_n_entries"), errors="coerce")

    # Rust / comeback (mirror existing base fields)
    out["rust_years_since_last_entry"] = out["years_since_last_entry"].astype("Int64")
    out["rust_is_rookie"] = out["is_rookie"].astype("int64")

    # Consistency / reliability
    out["cons_pct_finished"] = pd.to_numeric(out.get("pct_finished"), errors="coerce")
    out["cons_last5_pct_finished"] = pd.to_numeric(out.get("last5_pct_finished"), errors="coerce")
    out["cons_w_pct_finished"] = pd.to_numeric(out.get("w_pct_finished"), errors="coerce")

    # Career stage / trajectory
    # career_race_number: how many Iditarods has this musher entered (including this one)
    out["career_race_number"] = (pd.to_numeric(out.get("n_entries"), errors="coerce").fillna(0) + 1).astype(int)

    # trajectory: recent form vs career average (negative = improving)
    # last3_avg_finish_place - avg_finish_place
    l3 = pd.to_numeric(out.get("last3_avg_finish_place"), errors="coerce")
    avg = pd.to_numeric(out.get("avg_finish_place"), errors="coerce")
    out["trajectory"] = l3 - avg  # negative means recent results better than career avg

    # last_year_improvement: last year's finish vs career average (negative = better than usual)
    lyp = pd.to_numeric(out.get("last_year_finish_place"), errors="coerce")
    out["last_year_improvement"] = lyp - avg  # negative means last year was better than average

    # --- Bayesian shrinkage ---
    # Shrinks small-sample rate estimates toward field-wide priors.
    # Formula: shrunk = (n * raw + k * prior) / (n + k)
    # k=5 means "trust the prior as much as 5 observations"
    SHRINKAGE_K = 5.0

    # Priors: approximate historical base rates across all musher-years
    PRIOR_PCT_TOP10 = 0.16
    PRIOR_PCT_TOP5 = 0.08
    PRIOR_PCT_WIN = 0.02
    PRIOR_PCT_FINISHED = 0.78
    PRIOR_AVG_FINISH = 22.0
    PRIOR_AVG_TIME_BEHIND = 90000.0  # ~25 hours behind winner

    n_ent = pd.to_numeric(out.get("n_entries"), errors="coerce").fillna(0)
    n_fin = pd.to_numeric(out.get("n_finishes"), errors="coerce").fillna(0)

    def _shrink(raw, n, prior):
        raw = pd.to_numeric(raw, errors="coerce").fillna(prior)
        return (n * raw + SHRINKAGE_K * prior) / (n + SHRINKAGE_K)

    out["shrunk_pct_top10"] = _shrink(out.get("pct_top10"), n_ent, PRIOR_PCT_TOP10)
    out["shrunk_pct_top5"] = _shrink(out.get("pct_top5"), n_ent, PRIOR_PCT_TOP5)
    out["shrunk_pct_win"] = _shrink(out.get("pct_win"), n_ent, PRIOR_PCT_WIN)
    out["shrunk_pct_finished"] = _shrink(out.get("pct_finished"), n_ent, PRIOR_PCT_FINISHED)
    out["shrunk_w_avg_finish_place"] = _shrink(out.get("w_avg_finish_place"), n_fin, PRIOR_AVG_FINISH)
    out["shrunk_w_avg_time_behind_winner"] = _shrink(
        out.get("w_avg_time_behind_winner_seconds"), n_fin, PRIOR_AVG_TIME_BEHIND
    )

    # --- Confidence weight ---
    out["confidence_weight"] = n_fin / (n_fin + SHRINKAGE_K)

    # --- Upsert into table ---
    con.execute("DELETE FROM musher_strength WHERE year = ?", [year])
    con.register("out_df", out)

    con.execute("""
        INSERT INTO musher_strength (
            year, musher_id,
            n_years, n_entries, n_finishes,
            pct_finished, pct_top10, pct_top5, pct_win,
            avg_finish_place, median_finish_place, best_finish_place,
            avg_finish_time_seconds, median_finish_time_seconds,
            avg_time_behind_winner_seconds, median_time_behind_winner_seconds,
            last_year_finish_place, last_year_finished, last_year_time_behind_winner_seconds,
            years_since_last_entry, is_rookie,

            last3_n_entries, last3_n_finishes, last3_pct_finished, last3_pct_top10,
            last3_avg_finish_place, last3_best_finish_place, last3_avg_time_behind_winner_seconds,

            last5_n_entries, last5_n_finishes, last5_pct_finished, last5_pct_top10,
            last5_avg_finish_place, last5_best_finish_place, last5_avg_time_behind_winner_seconds,

            w_n_entries, w_pct_finished, w_pct_top10,
            w_avg_finish_place, w_avg_time_behind_winner_seconds, w_finishes_weight,

            peak_best_finish_place, peak_pct_top10, peak_pct_top5, peak_pct_win,
            form_last3_avg_finish_place, form_last5_avg_finish_place,
            form_last3_pct_top10, form_last5_pct_top10,
            form_w_avg_finish_place, form_w_pct_top10,
            exp_n_entries, exp_n_finishes, exp_n_years, exp_w_entries,
            rust_years_since_last_entry, rust_is_rookie,
            cons_pct_finished, cons_last5_pct_finished, cons_w_pct_finished,
            career_race_number, trajectory, last_year_improvement,
            shrunk_pct_top10, shrunk_pct_top5, shrunk_pct_win, shrunk_pct_finished,
            shrunk_w_avg_finish_place, shrunk_w_avg_time_behind_winner,
            std_finish_place, finish_place_range,
            confidence_weight
        )
        SELECT
            year, musher_id,
            n_years, n_entries, n_finishes,
            pct_finished, pct_top10, pct_top5, pct_win,
            avg_finish_place, median_finish_place, best_finish_place,
            avg_finish_time_seconds, median_finish_time_seconds,
            avg_time_behind_winner_seconds, median_time_behind_winner_seconds,
            last_year_finish_place, last_year_finished, last_year_time_behind_winner_seconds,
            years_since_last_entry, is_rookie,

            last3_n_entries, last3_n_finishes, last3_pct_finished, last3_pct_top10,
            last3_avg_finish_place, last3_best_finish_place, last3_avg_time_behind_winner_seconds,

            last5_n_entries, last5_n_finishes, last5_pct_finished, last5_pct_top10,
            last5_avg_finish_place, last5_best_finish_place, last5_avg_time_behind_winner_seconds,

            w_n_entries, w_pct_finished, w_pct_top10,
            w_avg_finish_place, w_avg_time_behind_winner_seconds, w_finishes_weight,

            peak_best_finish_place, peak_pct_top10, peak_pct_top5, peak_pct_win,
            form_last3_avg_finish_place, form_last5_avg_finish_place,
            form_last3_pct_top10, form_last5_pct_top10,
            form_w_avg_finish_place, form_w_pct_top10,
            exp_n_entries, exp_n_finishes, exp_n_years, exp_w_entries,
            rust_years_since_last_entry, rust_is_rookie,
            cons_pct_finished, cons_last5_pct_finished, cons_w_pct_finished,
            career_race_number, trajectory, last_year_improvement,
            shrunk_pct_top10, shrunk_pct_top5, shrunk_pct_win, shrunk_pct_finished,
            shrunk_w_avg_finish_place, shrunk_w_avg_time_behind_winner,
            std_finish_place, finish_place_range,
            confidence_weight
        FROM out_df
    """)

    print(f"Built musher_strength for year={year} (lookback={lookback}) ✅")
    print(out.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
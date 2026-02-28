# src/features/checkpoint_distances.py
"""
Reference table: checkpoint name → cumulative miles from start.

The Iditarod has multiple route configurations:
  - Northern route (traditional even years)
  - Southern route (traditional odd years)
  - Fairbanks restart (2017, 2025, others) — different first ~300mi
  - 2021 COVID out-and-back (Deshka Landing → Iditarod → back)
  - 2018 exception: even year but ran southern route

Sources: official Iditarod checkpoint mileage charts, race archives.
Distances are approximate (trail conditions shift ±5 mi).

Usage:
    python -m src.features.checkpoint_distances --year 2025
"""

import argparse
from src.db import connect


# ──────────────────────────────────────────────────────────────────────
# Route definitions: checkpoint name → cumulative miles from start
# ──────────────────────────────────────────────────────────────────────

# Standard Northern route (Willow restart, even years: 2016, 2020, 2022, 2024)
NORTHERN_CHECKPOINTS = {
    "Willow":          0,
    "Yentna Station":  42,
    "Skwentna":        83,
    "Finger Lake":    107,
    "Rainy Pass":     137,
    "Rohn":           170,
    "Nikolai":        241,
    "McGrath":        293,
    "Takotna":        310,
    "Ophir":          330,
    "Cripple":        395,
    "Ruby":           462,
    "Galena":         484,
    "Nulato":         531,
    "Kaltag":         567,
    "Unalakleet":     624,
    "Shaktoolik":     661,
    "Koyuk":          700,
    "Elim":           736,
    "White Mountain":  760,
    "Safety":         798,
    "Nome":           811,
}
NORTHERN_TOTAL = 811

# Standard Southern route (Willow restart, odd years: 2019, 2023)
SOUTHERN_CHECKPOINTS = {
    "Willow":          0,
    "Yentna Station":  42,
    "Skwentna":        83,
    "Finger Lake":    107,
    "Rainy Pass":     137,
    "Rohn":           170,
    "Nikolai":        241,
    "McGrath":        293,
    "Takotna":        310,
    "Ophir":          330,
    "Iditarod":       397,
    "Shageluk":       454,
    "Anvik":          480,
    "Grayling":       504,
    "Eagle Island":   553,
    "Kaltag":         590,
    "Unalakleet":     647,
    "Shaktoolik":     684,
    "Koyuk":          723,
    "Elim":           759,
    "White Mountain":  783,
    "Safety":         821,
    "Nome":           834,
}
SOUTHERN_TOTAL = 834

# Fairbanks restart route (2017): Fairbanks → interior → Nome
# 2017 went through Huslia and Koyukuk (unique to that year)
FAIRBANKS_2017_CHECKPOINTS = {
    "Fairbanks":        0,
    "Nenana":          60,
    "Manley":          90,
    "Tanana":         150,
    "Ruby":           230,
    "Galena":         280,
    "Huslia":         315,
    "Koyukuk":        360,
    "Nulato":         390,
    "Kaltag":         420,
    "Unalakleet":     480,
    "Shaktoolik":     517,
    "Koyuk":          556,
    "Elim":           592,
    "White Mountain":  616,
    "Safety":         654,
    "Nome":           667,
}
FAIRBANKS_2017_TOTAL = 667

# Fairbanks restart route (2025): Fairbanks → interior → southern loop → Nome
# Visits some checkpoints twice (outbound "1" and return "2")
FAIRBANKS_2025_CHECKPOINTS = {
    "fairbanks":        0,
    "nenana":          60,
    "manley":          90,
    "tanana":         150,
    "ruby":           230,
    "galena":         280,
    "nulato":         330,
    "kaltag 1":       365,    # outbound
    "eagle island 1": 415,    # outbound into southern loop
    "grayling 1":     440,
    "anvik":          465,
    "shageluk":       490,
    "grayling 2":     515,    # return from southern loop
    "eagle island 2": 540,    # return
    "kaltag 2":       590,    # return — back at Kaltag heading to coast
    "unalakleet":     640,
    "shaktoolik":     677,
    "koyuk":          716,
    "elim":           752,
    "white mountain":  776,
    "safety":         814,
    "nome":           827,
}
FAIRBANKS_2025_TOTAL = 827

# 2018: Even year but ran SOUTHERN route (confirmed by Iditarod/Shageluk/Anvik/Grayling)
SOUTHERN_2018_CHECKPOINTS = SOUTHERN_CHECKPOINTS.copy()
SOUTHERN_2018_TOTAL = SOUTHERN_TOTAL

# 2021 COVID: Out-and-back, Deshka Landing → Iditarod → back (~860 mi)
# Checkpoints have "N" (northbound) and "S" (southbound) suffixes
# We compute cumulative distance for the full out-and-back
_2021_OUTBOUND = {
    "Deshka Landing N":   0,
    "Skwentna N":        50,
    "Finger Lake N":     74,
    "Rainy Pass N":     104,
    "Rohn N":           137,
    "Nikolai N":        208,
    "McGrath N":        260,
    "Ophir N":          297,
    "Iditarod N":       364,  # turnaround
}
_2021_RETURN = {
    "Ophir S":          431,
    "McGrath S":        468,
    "Nikolai S":        520,
    "Rohn S":           591,
    "Rainy Pass S":     624,
    "Finger Lake S":    654,
    "Skwentna S":       678,
    "Deshka Landing S": 728,
}
COVID_2021_CHECKPOINTS = {**_2021_OUTBOUND, **_2021_RETURN}
COVID_2021_TOTAL = 728


# ──────────────────────────────────────────────────────────────────────
# Year → route mapping
# ──────────────────────────────────────────────────────────────────────

YEAR_ROUTE_CONFIG = {
    # Pre-2016 standard northern (even years)
    2006: ("northern",   NORTHERN_CHECKPOINTS,          NORTHERN_TOTAL),
    2008: ("northern",   NORTHERN_CHECKPOINTS,          NORTHERN_TOTAL),
    2010: ("northern",   NORTHERN_CHECKPOINTS,          NORTHERN_TOTAL),
    2012: ("northern",   NORTHERN_CHECKPOINTS,          NORTHERN_TOTAL),
    2014: ("northern",   NORTHERN_CHECKPOINTS,          NORTHERN_TOTAL),
    # Pre-2016 standard southern (odd years)
    2009: ("southern",   SOUTHERN_CHECKPOINTS,           SOUTHERN_TOTAL),
    2011: ("southern",   SOUTHERN_CHECKPOINTS,           SOUTHERN_TOTAL),
    2013: ("southern",   SOUTHERN_CHECKPOINTS,           SOUTHERN_TOTAL),
    # 2015 Fairbanks restart (same route as 2017)
    2015: ("fairbanks",  FAIRBANKS_2017_CHECKPOINTS,     FAIRBANKS_2017_TOTAL),
    # Standard northern
    2016: ("northern",   NORTHERN_CHECKPOINTS,          NORTHERN_TOTAL),
    2020: ("northern",   NORTHERN_CHECKPOINTS,          NORTHERN_TOTAL),
    2022: ("northern",   NORTHERN_CHECKPOINTS,          NORTHERN_TOTAL),
    2024: ("northern",   NORTHERN_CHECKPOINTS,          NORTHERN_TOTAL),
    # Standard southern
    2019: ("southern",   SOUTHERN_CHECKPOINTS,           SOUTHERN_TOTAL),
    2023: ("southern",   SOUTHERN_CHECKPOINTS,           SOUTHERN_TOTAL),
    # Exceptions
    2017: ("fairbanks",  FAIRBANKS_2017_CHECKPOINTS,     FAIRBANKS_2017_TOTAL),
    2018: ("southern",   SOUTHERN_2018_CHECKPOINTS,       SOUTHERN_2018_TOTAL),
    2021: ("covid",      COVID_2021_CHECKPOINTS,          COVID_2021_TOTAL),
    2025: ("fairbanks",  FAIRBANKS_2025_CHECKPOINTS,      FAIRBANKS_2025_TOTAL),
}


def _normalize_name(name: str) -> str:
    """Normalize a checkpoint name for fuzzy matching."""
    s = name.strip().lower()

    # Preserve meaningful suffixes like "Eagle Island 1" vs "Eagle Island 2"
    # and "Grayling 1" vs "Grayling 2" (out-and-back checkpoints).
    # Only strip trailing numbers if they look like scraping artifacts
    # (e.g., "Kaltag 1" where there's no "Kaltag 2" — a single visit).
    # We keep the number and let the lookup handle it; if no match,
    # we'll try stripping below in resolve_checkpoint_miles.

    # Common aliases
    aliases = {
        "rainy pass lodge": "rainy pass",
        "yentna": "yentna station",
        "deshka landing": "deshka landing",
        "manley hot springs": "manley",
    }
    return aliases.get(s, s)


def get_route_for_year(year: int) -> str:
    if year in YEAR_ROUTE_CONFIG:
        return YEAR_ROUTE_CONFIG[year][0]
    # Fallback for unknown years
    return "northern" if year % 2 == 0 else "southern"


def get_distance_lookup(year: int) -> tuple[dict[str, float], float]:
    """
    Returns (normalized_name → miles dict, total_miles) for the given year.
    """
    if year in YEAR_ROUTE_CONFIG:
        _, raw, total = YEAR_ROUTE_CONFIG[year]
    elif year % 2 == 0:
        raw, total = NORTHERN_CHECKPOINTS, NORTHERN_TOTAL
    else:
        raw, total = SOUTHERN_CHECKPOINTS, SOUTHERN_TOTAL

    lookup = {}
    for name, miles in raw.items():
        lookup[_normalize_name(name)] = miles
    return lookup, total


def resolve_checkpoint_miles(checkpoint_name: str, year: int) -> tuple[float | None, float | None]:
    """
    Given a checkpoint name (as scraped) and year, return:
      (cumulative_miles, checkpoint_pct)
    where checkpoint_pct = cumulative_miles / total_race_miles.

    Tries multiple matching strategies:
      1. Exact normalized match (e.g., "eagle island 1")
      2. With trailing number stripped (e.g., "kaltag 1" → "kaltag")
      3. With N/S suffix stripped (e.g., "skwentna n" → "skwentna")

    Returns (None, None) if the name can't be matched.
    """
    lookup, total = get_distance_lookup(year)
    norm = _normalize_name(checkpoint_name)

    # 1. Exact match
    miles = lookup.get(norm)

    # 2. Strip trailing number (for single-visit checkpoints like "Kaltag 1")
    if miles is None:
        stripped = norm.rstrip("0123456789").strip()
        if stripped != norm:
            miles = lookup.get(stripped)

    # 3. Strip N/S suffix (for 2021-style names)
    if miles is None and norm.endswith((" n", " s")):
        miles = lookup.get(norm[:-2].strip())

    # Special: "Anchorage" is ceremonial start → treat as mile 0
    if miles is None and norm == "anchorage":
        return 0.0, 0.0

    if miles is None:
        return None, None

    pct = miles / total if total > 0 else None
    return float(miles), float(pct)


def build_checkpoint_distances(con, year: int):
    """
    For a given year, look up every checkpoint in the splits table
    and populate checkpoint_distances with cumulative miles and pct.
    """
    con.execute("""
        CREATE TABLE IF NOT EXISTS checkpoint_distances (
            year INTEGER,
            checkpoint_order INTEGER,
            checkpoint_name TEXT,
            cumulative_miles DOUBLE,
            checkpoint_pct DOUBLE,
            route TEXT,
            PRIMARY KEY (year, checkpoint_order)
        )
    """)

    cp_names = con.execute("""
        SELECT DISTINCT checkpoint_order, checkpoint_name
        FROM splits
        WHERE year = ? AND checkpoint_name IS NOT NULL
        ORDER BY checkpoint_order
    """, [year]).fetchall()

    if not cp_names:
        print(f"WARNING: No checkpoint names found in splits for year={year}")
        return

    route = get_route_for_year(year)

    con.execute("DELETE FROM checkpoint_distances WHERE year = ?", [year])

    inserted = 0
    unmatched = []
    for cp_order, cp_name in cp_names:
        miles, pct = resolve_checkpoint_miles(cp_name, year)
        if miles is None:
            unmatched.append((cp_order, cp_name))
            continue

        con.execute("""
            INSERT OR REPLACE INTO checkpoint_distances
                (year, checkpoint_order, checkpoint_name, cumulative_miles, checkpoint_pct, route)
            VALUES (?, ?, ?, ?, ?, ?)
        """, [year, cp_order, cp_name, miles, pct, route])
        inserted += 1

    print(f"checkpoint_distances year={year} route={route}: {inserted} matched, {len(unmatched)} unmatched")
    if unmatched:
        for o, n in unmatched:
            print(f"  UNMATCHED: cp_order={o} name='{n}'")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--year", type=int, required=True)
    args = ap.parse_args()

    con = connect()
    build_checkpoint_distances(con, args.year)


if __name__ == "__main__":
    main()
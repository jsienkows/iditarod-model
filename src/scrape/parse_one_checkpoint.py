"""Parse a single checkpoint HTML page into structured split data. Used for debugging."""

import re
from bs4 import BeautifulSoup
from dateutil import parser, tz
from src.db import connect

# ✅ Use the SAME musher_id extraction as final standings (stable across pages/years)
from src.scrape.parse_helpers import extract_musher_id_from_row

AK = tz.gettz("America/Anchorage")
UTC = tz.gettz("UTC")


def norm_header(s: str) -> str:
    """Normalize header text for fuzzy matching."""
    s = (s or "").strip().lower()
    s = re.sub(r"[^\w\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def parse_time_to_utc(s: str, default_dt, year: int):
    """
    Parse a time string into UTC.
    - If s includes a full date, dateutil will use it.
    - If s is time-only (e.g., '4:00 AM'), we use default_dt's date.
    """
    if not s or not str(s).strip():
        return None

    s = str(s).strip()

    try:
        if default_dt is not None:
            dt = parser.parse(s, default=default_dt)
        else:
            # Fallback default: inject a plausible race month-day (better than Jan 1)
            dt = parser.parse(s, default=parser.parse(f"{year}-03-01"))

        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=AK)

        return dt.astimezone(UTC)
    except Exception:
        return None


def parse_duration_to_seconds(s: str):
    # Handles "HH:MM" or "HH:MM:SS"
    if not s or not str(s).strip():
        return None
    s = str(s).strip()
    if ":" not in s:
        return None
    parts = s.split(":")
    try:
        if len(parts) == 2:
            h = int(parts[0])
            m = int(parts[1])
            return h * 3600 + m * 60
        if len(parts) == 3:
            h = int(parts[0])
            m = int(parts[1])
            sec = int(parts[2])
            return h * 3600 + m * 60 + sec
    except ValueError:
        return None
    return None


def main():
    YEAR = 2025
    CHECKPOINT_ORDER = 8
    CHECKPOINT_NAME = "Kaltag 1"
    URL = "https://iditarod.com/race/2025/checkpoints/8-Kaltag-1/"

    con = connect()

    row = con.execute("SELECT html FROM raw_pages WHERE url = ?", [URL]).fetchone()
    if not row:
        raise RuntimeError("HTML not found in raw_pages. Run the downloader again first.")
    html = row[0]

    soup = BeautifulSoup(html, "lxml")
    table = soup.find("table")
    if table is None:
        raise RuntimeError("Could not find a <table> on the page.")

    # ✅ Prefer THEAD headers to avoid capturing random <th> cells from the whole table
    thead = table.find("thead")
    if thead:
        header_cells = thead.find_all("th")
    else:
        # fallback: first row
        first_tr = table.find("tr")
        header_cells = first_tr.find_all(["th", "td"]) if first_tr else []

    headers_raw = [h.get_text(" ", strip=True) for h in header_cells]
    headers_norm = [norm_header(h) for h in headers_raw]

    tbody = table.find("tbody") or table
    rows = tbody.find_all("tr")

    def get(cells, wanted_norm_names, occurrence=1):
        """Find a cell value by fuzzy normalized header.
        
        occurrence: which match to return (1=first, 2=second, etc.)
        Useful for duplicate column names like 'Dogs', 'Dogs'.
        """
        count = 0
        for wanted in wanted_norm_names:
            for idx, h in enumerate(headers_norm):
                if h == wanted and idx < len(cells):
                    count += 1
                    if count == occurrence:
                        return cells[idx]
        return ""

    inserted = 0
    skipped_no_id = 0

    for tr in rows:
        cells = [td.get_text(" ", strip=True) for td in tr.find_all(["td", "th"])]
        if len(cells) < 3:
            continue

        # ✅ Use stable musher_id extracted from the row (links / attributes on the site)
        musher_id = extract_musher_id_from_row(tr)
        if musher_id is None:
            skipped_no_id += 1
            continue

        place = get(cells, ["place", "rank"])
        in_t = get(cells, ["in"])
        out_t = get(cells, ["out"])
        rest = get(cells, ["rest", "rest time"])
        enrt = get(cells, ["time en route", "en route", "time enroute"])
       # Handle 2026+ format: two "Dogs" columns (prev checkpoint, current)
        n_dogs_cols = sum(1 for h in headers_norm if h == "dogs")
        if n_dogs_cols >= 2:
            dogs_in = get(cells, ["dogs"], occurrence=2)
            dogs_out = get(cells, ["dogs"], occurrence=2)
        else:
            dogs_in = get(cells, ["dogs in", "dogs"])
            dogs_out = get(cells, ["dogs out", "dogs"])

        rank = int(place) if str(place).isdigit() else None

        # Parse IN first (it usually contains the date)
        in_utc = parse_time_to_utc(in_t, default_dt=None, year=YEAR)

        # Use IN's date as the default for OUT if OUT is time-only
        default_out_dt = None
        if in_utc is not None:
            in_local = in_utc.astimezone(AK)
            default_out_dt = in_local.replace(tzinfo=AK)

        out_utc = parse_time_to_utc(out_t, default_dt=default_out_dt, year=YEAR)

        rest_s = parse_duration_to_seconds(rest)
        enrt_s = parse_duration_to_seconds(enrt)

        din = int(dogs_in) if str(dogs_in).isdigit() else None
        dout = int(dogs_out) if str(dogs_out).isdigit() else None

        con.execute(
            """
            INSERT OR REPLACE INTO splits (
                year, musher_id, checkpoint_order, checkpoint_name,
                in_time_utc, out_time_utc, rest_seconds, time_en_route_seconds,
                dogs_in, dogs_out, rank_at_checkpoint
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                YEAR,
                musher_id,
                CHECKPOINT_ORDER,
                CHECKPOINT_NAME,
                in_utc,
                out_utc,
                rest_s,
                enrt_s,
                din,
                dout,
                rank,
            ],
        )

        inserted += 1

    print(f"Inserted/updated {inserted} rows into splits ✅")
    if skipped_no_id:
        print(f"Skipped rows with no extractable musher_id: {skipped_no_id}")


if __name__ == "__main__":
    main()

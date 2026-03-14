"""
Parse scraped checkpoint HTML pages into the splits table.

Reads raw HTML from the raw_pages table, extracts musher checkpoint
times (in/out), dog counts, and rest durations, then populates the
splits and checkpoints tables.

Usage:
    python -m src.scrape.parse_all_checkpoints --year_min 2006 --year_max 2025
"""

import argparse
import re
from bs4 import BeautifulSoup
from dateutil import parser as dtparser, tz

from src.db import connect
from src.scrape.parse_helpers import extract_musher_id_from_row, clean_text

AK = tz.gettz("America/Anchorage")
UTC = tz.gettz("UTC")


def norm_header(s: str) -> str:
    s = (s or "").strip().lower()
    s = re.sub(r"[^\w\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def parse_time_to_utc(s: str, year: int, default_dt=None):
    """
    Parse a timestamp string into UTC.
    - If the string contains a full date, dateutil uses it.
    - If the string is time-only, dateutil uses default_dt's date.
    - Naive timestamps are assumed AK time.
    """
    s = (s or "").strip()
    if not s:
        return None

    try:
        if default_dt is not None:
            dt = dtparser.parse(s, default=default_dt)
        else:
            # Better fallback than Jan 1: race is in March
            dt = dtparser.parse(s, default=dtparser.parse(f"{year}-03-01"))

        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=AK)

        return dt.astimezone(UTC)
    except Exception:
        return None


def parse_duration_to_seconds(s: str):
    """
    Handles "HH:MM", "HH:MM:SS", and "Xh Ym" formats.
    """
    s = (s or "").strip()
    if not s:
        return None
    
    # Handle "Xh Ym" format (2026+)
    m_hm = re.match(r"(\d+)h\s*(\d+)m", s)
    if m_hm:
        return int(m_hm.group(1)) * 3600 + int(m_hm.group(2)) * 60
    
    # Handle "HH:MM" or "HH:MM:SS"
    if ":" not in s:
        return None
    parts = s.split(":")
    try:
        if len(parts) == 2:
            h = int(parts[0]); m = int(parts[1])
            return h * 3600 + m * 60
        if len(parts) == 3:
            h = int(parts[0]); m = int(parts[1]); sec = int(parts[2])
            return h * 3600 + m * 60 + sec
    except ValueError:
        return None
    return None


def parse_checkpoint_meta_from_url(url: str):
    # Example slug: .../checkpoints/8-Kaltag-1/
    slug = url.rstrip("/").split("/")[-1]
    m = re.match(r"^(\d+)-(.+)$", slug)
    if not m:
        return None, None
    order = int(m.group(1))
    name_part = m.group(2).replace("-", " ").strip()
    return order, name_part


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--year", type=int, required=True)
    args = ap.parse_args()
    year = args.year

    con = connect()

    pages = con.execute(
        """
        SELECT url, html
        FROM raw_pages
        WHERE year = ? AND page_type = 'checkpoint'
        ORDER BY url
        """,
        [year],
    ).fetchall()

    if not pages:
        raise RuntimeError(f"No checkpoint pages in raw_pages for {year}. Run scrape_all_checkpoints first.")

    total_inserted = 0
    failed = 0

    for url, html in pages:
        checkpoint_order, checkpoint_name = parse_checkpoint_meta_from_url(url)
        if checkpoint_order is None:
            print(f"SKIP (could not parse checkpoint order): {url}")
            continue

        soup = BeautifulSoup(html, "lxml")
        table = soup.find("table")
        if table is None:
            print(f"FAIL (no table): {url}")
            failed += 1
            continue

        # Prefer THEAD headers; fallback to first row
        thead = table.find("thead")
        if thead:
            headers_raw = [th.get_text(" ", strip=True) for th in thead.find_all("th")]
        else:
            first_tr = table.find("tr")
            headers_raw = [c.get_text(" ", strip=True) for c in first_tr.find_all(["th", "td"])] if first_tr else []

        headers_norm = [norm_header(h) for h in headers_raw]

        tbody = table.find("tbody") or table
        rows = tbody.find_all("tr")

        def get_cell(cells, wanted_norm_names, occurrence=1):
            count = 0
            for wanted in wanted_norm_names:
                for idx, h in enumerate(headers_norm):
                    if h == wanted and idx < len(cells):
                        count += 1
                        if count == occurrence:
                            return cells[idx]
            return ""

        inserted_here = 0
        skipped_no_id = 0

        for tr in rows:
            cells = [td.get_text(" ", strip=True) for td in tr.find_all(["td", "th"])]
            if len(cells) < 3:
                continue

            musher_id = extract_musher_id_from_row(tr)
            if musher_id is None:
                skipped_no_id += 1
                continue

            place = clean_text(get_cell(cells, ["place", "rank"]))

            # Detect 2026+ multi-row header format (13+ data cells, 9 headers)
            if len(cells) >= 13 and len(headers_norm) > len(cells):
                in_t = clean_text(cells[2]) if len(cells) > 2 else ""
                dogs_in = clean_text(cells[3]) if len(cells) > 3 else ""
                out_t = clean_text(cells[4]) if len(cells) > 4 else ""
                dogs_out = clean_text(cells[5]) if len(cells) > 5 else ""
                rest = clean_text(cells[6]) if len(cells) > 6 else ""
                enrt = clean_text(cells[7]) if len(cells) > 7 else ""
            else:
                in_t = clean_text(get_cell(cells, ["in"]))
                out_t = clean_text(get_cell(cells, ["out"]))
                rest = clean_text(get_cell(cells, ["rest", "rest time"]))
                enrt = clean_text(get_cell(cells, ["time en route", "en route", "time enroute"]))
                dogs_in = clean_text(get_cell(cells, ["dogs in", "dogs"]))
                dogs_out = clean_text(get_cell(cells, ["dogs out", "dogs"]))

            rank = int(place) if (place and place.isdigit()) else None

            # ✅ Parse IN first (usually has date)
            in_utc = parse_time_to_utc(in_t, year, default_dt=None)

            # ✅ Use IN date as default for OUT if OUT is time-only
            default_out_dt = None
            if in_utc is not None:
                in_local = in_utc.astimezone(AK)
                default_out_dt = in_local.replace(tzinfo=AK)

            out_utc = parse_time_to_utc(out_t, year, default_dt=default_out_dt)

            # If OUT winds up earlier than IN, treat it as unknown rather than corrupting timeline
            if in_utc is not None and out_utc is not None and out_utc < in_utc:
                out_utc = None

            rest_s = parse_duration_to_seconds(rest)
            enrt_s = parse_duration_to_seconds(enrt)
            din = int(dogs_in) if (dogs_in and dogs_in.isdigit()) else None
            dout = int(dogs_out) if (dogs_out and dogs_out.isdigit()) else None

            con.execute(
                """
                INSERT OR REPLACE INTO splits (
                    year, musher_id, checkpoint_order, checkpoint_name,
                    in_time_utc, out_time_utc, rest_seconds, time_en_route_seconds,
                    dogs_in, dogs_out, rank_at_checkpoint
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    year, musher_id, checkpoint_order, checkpoint_name,
                    in_utc, out_utc, rest_s, enrt_s, din, dout, rank
                ],
            )

            inserted_here += 1

        total_inserted += inserted_here
        print(f"OK {year} cp{checkpoint_order} {checkpoint_name}: {inserted_here} rows (skipped_no_id={skipped_no_id})")

    print(f"\nDONE ✅ Total inserted/updated rows: {total_inserted}")
    print(f"Failed pages: {failed}")


if __name__ == "__main__":
    main()

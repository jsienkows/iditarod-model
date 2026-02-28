"""
Scrape final race standings from iditarod.com for a range of years.

Fetches each year's results page, parses finish positions, times, and
statuses, and populates the entries and mushers tables.

Usage:
    python -m src.scrape.scrape_final_standings --year_min 2006 --year_max 2025
"""

import argparse
import re
from bs4 import BeautifulSoup

from src.db import connect
from src.scrape.fetch import fetch_html, utc_now
from src.scrape.parse_helpers import extract_musher_id_from_row, clean_text

def parse_elapsed_to_seconds(s: str):
    if not s or not s.strip():
        return None
    s = s.strip().lower()

    d = re.search(r"(\d+)\s*d", s)
    h = re.search(r"(\d+)\s*h", s)
    m = re.search(r"(\d+)\s*m", s)
    sec = re.search(r"(\d+)\s*s", s)

    if d or h or m or sec:
        days = int(d.group(1)) if d else 0
        hrs  = int(h.group(1)) if h else 0
        mins = int(m.group(1)) if m else 0
        secs = int(sec.group(1)) if sec else 0
        return days*86400 + hrs*3600 + mins*60 + secs

    if ":" in s:
        parts = s.split(":")
        try:
            if len(parts) == 3:
                hh, mm, ss = int(parts[0]), int(parts[1]), int(parts[2])
                return hh*3600 + mm*60 + ss
            if len(parts) == 2:
                hh, mm = int(parts[0]), int(parts[1])
                return hh*3600 + mm*60
        except ValueError:
            return None

    return None

def upsert_raw_page(con, url, page_type, year, html):
    con.execute("""
        INSERT OR REPLACE INTO raw_pages (url, fetched_at, page_type, year, checkpoint_name, html)
        VALUES (?, ?, ?, ?, ?, ?)
    """, [url, utc_now(), page_type, year, None, html])

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--year", type=int, required=True)
    args = ap.parse_args()
    year = args.year

    con = connect()
    url = f"https://iditarod.com/race/{year}/"

    html = fetch_html(url)
    upsert_raw_page(con, url, "standings", year, html)
    print(f"Downloaded final standings page for {year} ✅")

    soup = BeautifulSoup(html, "lxml")

    # Find best results table
    tables = soup.find_all("table")
    best = None
    best_rows = -1
    best_headers = None

    for t in tables:
        head = t.find("thead")
        if not head:
            continue
        headers = [th.get_text(" ", strip=True) for th in head.find_all("th")]
        header_set = set(h.lower() for h in headers)

        looks_like_results = ("place" in header_set) and (("name" in header_set) or ("musher" in header_set))
        if not looks_like_results:
            continue

        tbody = t.find("tbody")
        row_count = len(tbody.find_all("tr")) if tbody else len(t.find_all("tr"))
        if row_count > best_rows:
            best = t
            best_rows = row_count
            best_headers = headers

    if best is None:
        raise RuntimeError("Could not find a results table with Place + Name/Musher.")

    print(f"Using table with {best_rows} rows and headers: {best_headers}")

    headers = best_headers
    tbody = best.find("tbody") or best
    rows = tbody.find_all("tr")

    def get_cell(cells, col_names):
        for col in col_names:
            if col in headers:
                idx = headers.index(col)
                if idx < len(cells):
                    return cells[idx]
        return ""

    inserted = 0
    for tr in rows:
        cells = [td.get_text(" ", strip=True) for td in tr.find_all(["td","th"])]
        if len(cells) < 3:
            continue

        place_s = clean_text(get_cell(cells, ["Place", "Rank"]))
        name    = clean_text(get_cell(cells, ["Musher", "Name"]))
        bib_s   = clean_text(get_cell(cells, ["Bib"]))
        status  = clean_text(get_cell(cells, ["Status"]))
        elapsed = clean_text(get_cell(cells, ["Time", "Elapsed Time", "Final Time"]))

        if not name:
            continue

        musher_id = extract_musher_id_from_row(tr)
        if musher_id is None:
            # fall back: skip row (better than creating bad IDs)
            continue

        m = re.search(r"\d+", place_s or "")
        place = int(m.group(0)) if m else None

        m = re.search(r"\d+", bib_s or "")
        bib = int(m.group(0)) if m else None

        status_norm = (status or "").strip().upper()
        if not status_norm and place is not None:
            status_norm = "FINISHED"

        finish_seconds = parse_elapsed_to_seconds(elapsed)

        con.execute("""
            INSERT OR REPLACE INTO entries
              (year, musher_id, bib, finish_place, finish_time_seconds, status)
            VALUES (?, ?, ?, ?, ?, ?)
        """, [year, musher_id, bib, place, finish_seconds, status_norm])

        inserted += 1

    print(f"Inserted/updated {inserted} entries rows ✅")

    n_place = con.execute("""
      SELECT COUNT(*) FROM entries WHERE year = ? AND finish_place IS NOT NULL
    """, [year]).fetchone()[0]
    print(f"Entries with finish_place: {n_place}")

if __name__ == "__main__":
    main()

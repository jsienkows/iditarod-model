"""
Build the entries table from scraped final standings HTML pages.

Parses finish place, time, and status for each musher-year and upserts
into the entries table. Also creates/updates musher records.

Usage:
    python -m src.scrape.build_entries --year_min 2006 --year_max 2025
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

def parse_table_by_headers(table, required_headers_lower: set[str]):
    head = table.find("thead")
    if not head:
        return None, None, None
    headers = [th.get_text(" ", strip=True) for th in head.find_all("th")]
    header_set = set(h.lower() for h in headers)
    if not required_headers_lower.issubset(header_set):
        return None, None, None
    tbody = table.find("tbody") or table
    rows = tbody.find_all("tr")
    return headers, header_set, rows

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--year", type=int, required=True)
    args = ap.parse_args()
    year = args.year

    con = connect()

    roster_url = f"https://iditarod.com/race/{year}/mushers/list/"
    finishers_url = f"https://iditarod.com/race/{year}/"

    # 1) Roster
    roster_html = fetch_html(roster_url)
    upsert_raw_page(con, roster_url, "musher_roster", year, roster_html)

    soup = BeautifulSoup(roster_html, "lxml")
    tables = soup.find_all("table")

    starters_table = None
    withdrawn_table = None

    for t in tables:
        headers, header_set, rows = parse_table_by_headers(
            t, required_headers_lower={"bib #", "musher name"}
        )
        if headers:
            starters_table = (headers, rows)
            break

    for t in tables:
        headers, header_set, rows = parse_table_by_headers(
            t, required_headers_lower={"musher name", "status"}
        )
        if headers and ("bib #" not in header_set):
            withdrawn_table = (headers, rows)
            break

    if starters_table is None:
        raise RuntimeError("Could not find starters roster table (Bib # / Musher Name).")

    starters_headers, starters_rows = starters_table

    def get_cell(headers, cells, col_name):
        if col_name in headers:
            idx = headers.index(col_name)
            return cells[idx] if idx < len(cells) else ""
        return ""

    starters_inserted = 0
    for tr in starters_rows:
        cells = [td.get_text(" ", strip=True) for td in tr.find_all(["td", "th"])]
        if len(cells) < 2:
            continue

        bib_s = clean_text(get_cell(starters_headers, cells, "Bib #"))
        name  = clean_text(get_cell(starters_headers, cells, "Musher Name"))
        status = clean_text(get_cell(starters_headers, cells, "Status"))

        if not name:
            continue

        musher_id = extract_musher_id_from_row(tr)
        if musher_id is None:
            continue

        m = re.search(r"\d+", bib_s or "")
        bib = int(m.group(0)) if m else None

        status_norm = (status or "").strip().upper() or "STARTER"

        con.execute("""
            INSERT OR REPLACE INTO entries
              (year, musher_id, bib, finish_place, finish_time_seconds, status)
            VALUES (?, ?, ?, NULL, NULL, ?)
        """, [year, musher_id, bib, status_norm])

        starters_inserted += 1

    withdrawn_inserted = 0
    if withdrawn_table:
        w_headers, w_rows = withdrawn_table
        for tr in w_rows:
            cells = [td.get_text(" ", strip=True) for td in tr.find_all(["td", "th"])]
            if len(cells) < 1:
                continue

            name = clean_text(get_cell(w_headers, cells, "Musher Name"))
            status = clean_text(get_cell(w_headers, cells, "Status"))
            if not name:
                continue

            musher_id = extract_musher_id_from_row(tr)
            if musher_id is None:
                continue

            status_norm = (status or "").strip().upper() or "WITHDRAWN"

            con.execute("""
                INSERT OR REPLACE INTO entries
                  (year, musher_id, bib, finish_place, finish_time_seconds, status)
                VALUES (
                  ?, ?, (SELECT bib FROM entries WHERE year=? AND musher_id=?),
                  NULL, NULL, ?
                )
            """, [year, musher_id, year, musher_id, status_norm])

            withdrawn_inserted += 1

    print(f"Roster loaded ✅ starters: {starters_inserted}, withdrawn: {withdrawn_inserted}")

    # 2) Finishers / standings page update (place/time)
    finish_html = fetch_html(finishers_url)
    upsert_raw_page(con, finishers_url, "final_results_finishers", year, finish_html)

    soup = BeautifulSoup(finish_html, "lxml")
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

        looks = ("place" in header_set) and (("name" in header_set) or ("musher" in header_set)) and ("time" in header_set)
        if not looks:
            continue

        tbody = t.find("tbody")
        row_count = len(tbody.find_all("tr")) if tbody else len(t.find_all("tr"))
        if row_count > best_rows:
            best = t
            best_rows = row_count
            best_headers = headers

    if best is None:
        raise RuntimeError("Could not find finishers results table (Place/Name/Time).")

    tbody = best.find("tbody") or best
    rows = tbody.find_all("tr")

    def get_finish_cell(cells, col):
        if col in best_headers:
            idx = best_headers.index(col)
            return cells[idx] if idx < len(cells) else ""
        return ""

    finishers_updated = 0
    for tr in rows:
        cells = [td.get_text(" ", strip=True) for td in tr.find_all(["td","th"])]
        if len(cells) < 3:
            continue

        place_s = clean_text(get_finish_cell(cells, "Place"))
        elapsed = clean_text(get_finish_cell(cells, "Time"))

        musher_id = extract_musher_id_from_row(tr)
        if musher_id is None:
            continue

        m = re.search(r"\d+", place_s or "")
        place = int(m.group(0)) if m else None
        finish_seconds = parse_elapsed_to_seconds(elapsed)

        if place is None:
            continue

        con.execute("""
            UPDATE entries
            SET finish_place = ?, finish_time_seconds = ?, status = 'FINISHED'
            WHERE year = ? AND musher_id = ?
        """, [place, finish_seconds, year, musher_id])

        finishers_updated += 1

    print(f"Finishers updated ✅ rows: {finishers_updated}")

    total_entries = con.execute("SELECT COUNT(*) FROM entries WHERE year=?", [year]).fetchone()[0]
    with_place = con.execute("SELECT COUNT(*) FROM entries WHERE year=? AND finish_place IS NOT NULL", [year]).fetchone()[0]
    print(f"Entries total rows: {total_entries}, with finish_place: {with_place}")

if __name__ == "__main__":
    main()

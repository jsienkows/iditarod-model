# src/scrape/build_entries_from_splits.py
"""
Build entries table from checkpoint splits data + standings page.

This is an alternative to build_entries.py that works for older years
where the roster page format differs. It extracts musher IDs from
the splits table (which was populated by parse_all_checkpoints) and
then fetches finish place/time from the standings page.

Usage:
    python -m src.scrape.build_entries_from_splits --year 2010
"""

import argparse
import re

from bs4 import BeautifulSoup

from src.db import connect
from src.scrape.fetch import fetch_html, utc_now
from src.scrape.parse_helpers import extract_musher_id_from_row, clean_text
from src.scrape.build_entries import parse_elapsed_to_seconds, upsert_raw_page


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--year", type=int, required=True)
    args = ap.parse_args()
    year = args.year

    con = connect()

    # 1) Extract unique musher IDs from splits (already parsed from checkpoint pages)
    splits_mushers = con.execute(
        "SELECT DISTINCT musher_id FROM splits WHERE year = ?", [year]
    ).fetchall()

    if not splits_mushers:
        print(f"No splits data for {year}. Run scrape_all_checkpoints + parse_all_checkpoints first.")
        return

    musher_ids = [row[0] for row in splits_mushers]
    print(f"Found {len(musher_ids)} unique mushers in splits for {year}")

    # Insert all as starters (we'll update finish info below)
    for mid in musher_ids:
        con.execute("""
            INSERT OR IGNORE INTO entries (year, musher_id, bib, finish_place, finish_time_seconds, status)
            VALUES (?, ?, NULL, NULL, NULL, 'STARTER')
        """, [year, mid])

    # 2) Try to get finish place/time from standings page
    finishers_url = f"https://iditarod.com/race/{year}/"
    try:
        finish_html = fetch_html(finishers_url)
        upsert_raw_page(con, finishers_url, "final_results_finishers", year, finish_html)
    except Exception as e:
        print(f"WARNING: Could not fetch standings page: {e}")
        print(f"Entries created from splits data only (no finish places).")
        total = con.execute("SELECT COUNT(*) FROM entries WHERE year=?", [year]).fetchone()[0]
        print(f"Entries total: {total}")
        return

    soup = BeautifulSoup(finish_html, "lxml")
    tables = soup.find_all("table")

    # Find the results table (has Place + Name/Musher + Time columns)
    best = None
    best_rows = -1
    best_headers = None

    for t in tables:
        head = t.find("thead")
        if not head:
            continue
        headers = [th.get_text(" ", strip=True) for th in head.find_all("th")]
        header_set = set(h.lower() for h in headers)

        looks = ("place" in header_set) and \
                (("name" in header_set) or ("musher" in header_set) or ("musher name" in header_set)) and \
                ("time" in header_set)
        if not looks:
            continue

        tbody = t.find("tbody")
        row_count = len(tbody.find_all("tr")) if tbody else len(t.find_all("tr"))
        if row_count > best_rows:
            best = t
            best_rows = row_count
            best_headers = headers

    if best is None:
        # Try alternative: some years have different header names
        for t in tables:
            head = t.find("thead")
            if not head:
                continue
            headers = [th.get_text(" ", strip=True) for th in head.find_all("th")]
            header_set = set(h.lower() for h in headers)

            # More flexible matching
            has_place = any("place" in h or "pos" in h or "#" in h for h in header_set)
            has_name = any("name" in h or "musher" in h for h in header_set)
            has_time = any("time" in h or "elapsed" in h for h in header_set)

            if has_place and has_name:
                tbody = t.find("tbody")
                row_count = len(tbody.find_all("tr")) if tbody else len(t.find_all("tr"))
                if row_count > best_rows:
                    best = t
                    best_rows = row_count
                    best_headers = headers

    if best is None:
        print(f"WARNING: Could not find results table on standings page for {year}.")
        print(f"Entries created from splits only (no finish places).")
        total = con.execute("SELECT COUNT(*) FROM entries WHERE year=?", [year]).fetchone()[0]
        print(f"Entries total: {total}")
        return

    tbody = best.find("tbody") or best
    rows = tbody.find_all("tr")

    def get_cell(cells, col):
        if col in best_headers:
            idx = best_headers.index(col)
            return cells[idx] if idx < len(cells) else ""
        return ""

    finishers_updated = 0
    for tr in rows:
        cells = [td.get_text(" ", strip=True) for td in tr.find_all(["td", "th"])]
        if len(cells) < 3:
            continue

        # Try multiple header names for place
        place_s = clean_text(get_cell(cells, "Place"))
        if not place_s:
            place_s = clean_text(get_cell(cells, "Pos"))

        # Try multiple header names for time
        elapsed = clean_text(get_cell(cells, "Time"))
        if not elapsed:
            elapsed = clean_text(get_cell(cells, "Elapsed"))

        musher_id = extract_musher_id_from_row(tr)
        if musher_id is None:
            continue

        m = re.search(r"\d+", place_s or "")
        place = int(m.group(0)) if m else None
        finish_seconds = parse_elapsed_to_seconds(elapsed)

        if place is None:
            continue

        con.execute("""
            INSERT OR REPLACE INTO entries
              (year, musher_id, bib, finish_place, finish_time_seconds, status)
            VALUES (
              ?,
              ?,
              (SELECT bib FROM entries WHERE year=? AND musher_id=?),
              ?,
              ?,
              'FINISHED'
            )
        """, [year, musher_id, year, musher_id, place, finish_seconds])

        finishers_updated += 1

    # Mark non-finishers
    con.execute("""
        UPDATE entries
        SET status = 'SCRATCHED'
        WHERE year = ? AND finish_place IS NULL AND status = 'STARTER'
    """, [year])

    total = con.execute("SELECT COUNT(*) FROM entries WHERE year=?", [year]).fetchone()[0]
    finished = con.execute(
        "SELECT COUNT(*) FROM entries WHERE year=? AND finish_place IS NOT NULL", [year]
    ).fetchone()[0]
    scratched = con.execute(
        "SELECT COUNT(*) FROM entries WHERE year=? AND status='SCRATCHED'", [year]
    ).fetchone()[0]

    print(f"Entries built ✅  total: {total}, finished: {finished}, scratched: {scratched}")


if __name__ == "__main__":
    main()
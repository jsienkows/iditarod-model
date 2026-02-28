# src/scrape/scrape_musher_dogs.py
"""
Scrape dogs_in / dogs_out data from musher profile pages on iditarod.com.

The checkpoint pages do NOT include dog counts. The data is only available
on individual musher profile pages:
    https://iditarod.com/race/{year}/mushers/{musher_id}-{slug}/

Each profile has a "Standings" table with columns:
    Checkpoint | Time In | Dogs In | Time Out | Dogs Out | Rest Time | ...

This script:
  1. Reads musher_id + slug from the entries table (or discovers from roster page)
  2. Fetches each profile page
  3. Parses the standings table for dogs_in / dogs_out at each checkpoint
  4. Updates the splits table with the dog counts

Usage:
    python -m src.scrape.scrape_musher_dogs --year 2025
    python -m src.scrape.scrape_musher_dogs --year 2025 --musher_id 1075  # single musher
"""

import argparse
import re
import time

from bs4 import BeautifulSoup

from src.db import connect
from src.scrape.fetch import fetch_html, utc_now
from src.scrape.parse_helpers import clean_text


def norm_header(s: str) -> str:
    """Normalize header text for fuzzy matching."""
    s = (s or "").strip().lower()
    s = re.sub(r"[^\w\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _discover_musher_urls(con, year: int) -> list[tuple[str, str]]:
    """
    Discover musher profile URLs from available HTML pages.
    Tries: 1) roster page, 2) checkpoint pages, 3) musher gallery page.
    Returns list of (musher_id, profile_url).
    """
    results = []

    # Source 1: roster page HTML
    row = con.execute(
        "SELECT html FROM raw_pages WHERE year = ? AND page_type = 'musher_roster' LIMIT 1",
        [year],
    ).fetchone()

    if row:
        soup = BeautifulSoup(row[0], "lxml")
        for a in soup.find_all("a", href=True):
            href = a["href"]
            m = re.search(r"/race/\d+/mushers/(\d+)-([^/]+)", href)
            if m:
                mid = m.group(1)
                slug = m.group(2)
                url = f"https://iditarod.com/race/{year}/mushers/{mid}-{slug}/"
                results.append((mid, url))

    # Source 2: checkpoint page HTML (always available if we scraped checkpoints)
    if not results:
        cp_pages = con.execute(
            "SELECT html FROM raw_pages WHERE year = ? AND page_type = 'checkpoint'",
            [year],
        ).fetchall()

        for (html,) in cp_pages:
            soup = BeautifulSoup(html, "lxml")
            for a in soup.find_all("a", href=True):
                href = a["href"]
                m = re.search(r"/race/\d+/mushers/(\d+)-([^/]+)", href)
                if m:
                    mid = m.group(1)
                    slug = m.group(2)
                    url = f"https://iditarod.com/race/{year}/mushers/{mid}-{slug}/"
                    results.append((mid, url))

    # Deduplicate
    if results:
        seen = set()
        deduped = []
        for mid, url in results:
            if mid not in seen:
                seen.add(mid)
                deduped.append((mid, url))
        return deduped

    return []

    # We need slugs to construct URLs. Try the roster page to find them.
    print(f"  WARNING: Could not find musher URLs in roster HTML. "
          f"Have {len(entries)} entries but need profile URLs.")
    print(f"  Try re-running: python -m src.scrape.build_entries --year {year}")
    return []


def _parse_profile_dogs(html: str, year: int) -> list[dict]:
    """
    Parse the standings table from a musher profile page.
    Returns list of dicts with keys: checkpoint_name, dogs_in, dogs_out
    """
    soup = BeautifulSoup(html, "lxml")

    # Find the standings table — look for a table with "Checkpoint" and "Dogs In" headers
    tables = soup.find_all("table")

    for table in tables:
        thead = table.find("thead")
        if not thead:
            continue

        # The table has nested header rows. Collect all header text.
        all_ths = thead.find_all("th")
        headers_raw = [th.get_text(" ", strip=True) for th in all_ths]
        headers_norm = [norm_header(h) for h in headers_raw]

        # Check if this looks like the standings table
        has_checkpoint = any("checkpoint" in h for h in headers_norm)
        has_dogs = any("dogs in" in h for h in headers_norm)

        if not has_checkpoint or not has_dogs:
            continue

        # Found it — parse rows
        tbody = table.find("tbody") or table
        rows = tbody.find_all("tr")

        def get_cell(cells, wanted_names):
            for wanted in wanted_names:
                if wanted in headers_norm:
                    idx = headers_norm.index(wanted)
                    if idx < len(cells):
                        return cells[idx]
            return ""

        results = []
        for tr in rows:
            cells = [td.get_text(" ", strip=True) for td in tr.find_all(["td", "th"])]
            if len(cells) < 3:
                continue

            cp_name = clean_text(get_cell(cells, ["checkpoint"]))
            dogs_in_s = clean_text(get_cell(cells, ["dogs in"]))
            dogs_out_s = clean_text(get_cell(cells, ["dogs out"]))

            if not cp_name:
                continue

            din = int(dogs_in_s) if (dogs_in_s and dogs_in_s.isdigit()) else None
            dout = int(dogs_out_s) if (dogs_out_s and dogs_out_s.isdigit()) else None

            results.append({
                "checkpoint_name": cp_name,
                "dogs_in": din,
                "dogs_out": dout,
            })

        return results

    return []


def _checkpoint_name_to_order(con, year: int) -> dict[str, int]:
    """Build a mapping from checkpoint_name → checkpoint_order for the year."""
    rows = con.execute(
        "SELECT DISTINCT checkpoint_order, checkpoint_name FROM splits WHERE year = ?",
        [year],
    ).fetchall()

    mapping = {}
    for order, name in rows:
        mapping[name.strip().lower()] = order
    return mapping


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--year", type=int, required=True)
    ap.add_argument("--musher_id", type=str, default=None, help="Scrape a single musher")
    ap.add_argument("--delay", type=float, default=1.0, help="Delay between requests (seconds)")
    ap.add_argument("--dry_run", action="store_true", help="Fetch and parse but don't update DB")
    args = ap.parse_args()

    year = args.year
    con = connect()

    # Discover musher profile URLs
    all_mushers = _discover_musher_urls(con, year)

    if not all_mushers:
        print(f"No musher profile URLs found for {year}.")
        return

    if args.musher_id:
        all_mushers = [(mid, url) for mid, url in all_mushers if mid == args.musher_id]
        if not all_mushers:
            print(f"Musher {args.musher_id} not found in roster for {year}.")
            return

    # Build checkpoint name → order mapping
    cp_map = _checkpoint_name_to_order(con, year)
    if not cp_map:
        print(f"WARNING: No splits data found for {year}. Run scrape + parse checkpoints first.")

    print(f"Scraping dog counts from {len(all_mushers)} musher profiles for {year}...")
    print(f"Checkpoint mapping: {len(cp_map)} checkpoints")

    total_updated = 0
    mushers_with_data = 0
    errors = 0

    for i, (musher_id, url) in enumerate(all_mushers):
        try:
            html = fetch_html(url)

            # Save to raw_pages
            con.execute("""
                INSERT OR REPLACE INTO raw_pages (url, fetched_at, page_type, year, checkpoint_name, html)
                VALUES (?, ?, ?, ?, ?, ?)
            """, [url, utc_now(), "musher_profile", year, None, html])

            # Parse dog data
            dog_rows = _parse_profile_dogs(html, year)

            if not dog_rows:
                if (i + 1) % 10 == 0 or i == 0:
                    print(f"  [{i+1}/{len(all_mushers)}] musher={musher_id}: no dog data found")
                if args.delay > 0:
                    time.sleep(args.delay)
                continue

            musher_updates = 0
            for row in dog_rows:
                cp_name_lower = row["checkpoint_name"].strip().lower()

                # Try exact match first
                cp_order = cp_map.get(cp_name_lower)

                # Try with link text cleaned (profile may say "Fairbanks" while splits has "Fairbanks")
                if cp_order is None:
                    for key, val in cp_map.items():
                        if key in cp_name_lower or cp_name_lower in key:
                            cp_order = val
                            break

                if cp_order is None:
                    continue

                if row["dogs_in"] is None and row["dogs_out"] is None:
                    continue

                if not args.dry_run:
                    # Update existing splits row with dog data
                    con.execute("""
                        UPDATE splits
                        SET dogs_in = COALESCE(?, dogs_in),
                            dogs_out = COALESCE(?, dogs_out)
                        WHERE year = ? AND musher_id = ? AND checkpoint_order = ?
                    """, [row["dogs_in"], row["dogs_out"], year, musher_id, cp_order])

                musher_updates += 1

            if musher_updates > 0:
                mushers_with_data += 1
                total_updated += musher_updates

            if (i + 1) % 10 == 0 or i == 0:
                print(f"  [{i+1}/{len(all_mushers)}] musher={musher_id}: {musher_updates} checkpoints updated")

        except Exception as e:
            errors += 1
            print(f"  ERROR musher={musher_id}: {e}")

        if args.delay > 0 and i < len(all_mushers) - 1:
            time.sleep(args.delay)

    print(f"\nDONE ✅")
    print(f"  Mushers with dog data: {mushers_with_data}/{len(all_mushers)}")
    print(f"  Total checkpoint rows updated: {total_updated}")
    print(f"  Errors: {errors}")

    if not args.dry_run:
        # Verify
        result = con.execute("""
            SELECT COUNT(*) as total,
                   COUNT(dogs_in) as has_dogs_in,
                   COUNT(dogs_out) as has_dogs_out
            FROM splits
            WHERE year = ?
        """, [year]).fetchone()
        print(f"\n  Splits for {year}: {result[0]} total, "
              f"{result[1]} with dogs_in ({result[1]*100//max(result[0],1)}%), "
              f"{result[2]} with dogs_out ({result[2]*100//max(result[0],1)}%)")


if __name__ == "__main__":
    main()
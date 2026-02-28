"""
Scrape checkpoint split-time pages from iditarod.com for a range of years.

Discovers checkpoint URLs from each year's index page, fetches the HTML,
and stores raw pages in the database for later parsing.

Usage:
    python -m src.scrape.scrape_all_checkpoints --year_min 2006 --year_max 2025
"""

import argparse
import re
from urllib.parse import urljoin

from bs4 import BeautifulSoup
from src.db import connect
from src.scrape.fetch import fetch_html, utc_now


def get_checkpoint_links(index_html: str, year: int) -> list[str]:
    soup = BeautifulSoup(index_html, "lxml")

    # Matches URLs like:
    # https://iditarod.com/race/2025/checkpoints/8-Kaltag-1/
    pat = re.compile(rf"/race/{year}/checkpoints/\d+-[^/]+/?$")

    links = []
    for a in soup.find_all("a", href=True):
        href = a["href"].strip()
        url = urljoin("https://iditarod.com", href)

        if pat.search(url):
            if not url.endswith("/"):
                url += "/"
            links.append(url)

    # de-dupe while preserving order
    seen = set()
    unique = []
    for u in links:
        if u not in seen:
            seen.add(u)
            unique.append(u)

    return unique


def parse_checkpoint_meta_from_url(url: str):
    """
    Example: .../checkpoints/8-Kaltag-1/
    Returns (checkpoint_order:int, checkpoint_name:str)
    """
    slug = url.rstrip("/").split("/")[-1]
    m = re.match(r"^(\d+)-(.+)$", slug)
    if not m:
        return None, None
    order = int(m.group(1))
    name = m.group(2).replace("-", " ").strip()
    return order, name


def upsert_raw_page(con, url, page_type, year, checkpoint_name, html):
    con.execute(
        """
        INSERT OR REPLACE INTO raw_pages (url, fetched_at, page_type, year, checkpoint_name, html)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        [url, utc_now(), page_type, year, checkpoint_name, html],
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--year", type=int, required=True)
    args = ap.parse_args()
    year = args.year

    index_url = f"https://iditarod.com/race/{year}/checkpoints/"

    con = connect()

    # 1) Download checkpoint index
    idx_html = fetch_html(index_url)
    upsert_raw_page(con, index_url, "checkpoint_index", year, None, idx_html)
    print(f"Downloaded checkpoint index for {year} ✅")

    # 2) Extract checkpoint page links
    links = get_checkpoint_links(idx_html, year)
    print(f"Found {len(links)} checkpoint links")

    # 3) Download each checkpoint page into raw_pages
    for url in links:
        html = fetch_html(url)

        checkpoint_order, checkpoint_name = parse_checkpoint_meta_from_url(url)
        # Store a readable name; if parsing fails, keep slug-ish fallback
        if checkpoint_name is None:
            slug = url.rstrip("/").split("/")[-1]
            checkpoint_name = slug.replace("-", " ")

        upsert_raw_page(con, url, "checkpoint", year, checkpoint_name, html)
        print(f"Saved checkpoint page: {checkpoint_order} {checkpoint_name}")

    print("All checkpoint pages saved into raw_pages ✅")


if __name__ == "__main__":
    main()

"""Fetch and store a single checkpoint page. Utility for debugging scrapers."""

from pathlib import Path
from src.db import connect
from src.scrape.fetch import fetch_html, utc_now

# Pick ONE checkpoint page to start.
# We'll use a known checkpoint example.
URL = "https://iditarod.com/race/2025/checkpoints/8-Kaltag-1/"

def main():
    # 1) Download HTML
    html = fetch_html(URL)
    print(f"Downloaded {len(html)} characters")

    # 2) Save HTML to disk (so you can open it in a browser)
    out_dir = Path("data/raw_html")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "checkpoint_kaltag1_2025.html"
    out_path.write_text(html, encoding="utf-8")
    print(f"Saved to {out_path}")

    # 3) Save HTML to DuckDB
    con = connect()
    con.execute("""
        CREATE TABLE IF NOT EXISTS raw_pages (
            url TEXT PRIMARY KEY,
            fetched_at TIMESTAMP,
            page_type TEXT,
            year INTEGER,
            checkpoint_name TEXT,
            html TEXT
        )
    """)

    con.execute("""
        INSERT OR REPLACE INTO raw_pages (url, fetched_at, page_type, year, checkpoint_name, html)
        VALUES (?, ?, ?, ?, ?, ?)
    """, [URL, utc_now(), "checkpoint", 2025, "Kaltag 1", html])

    print("Stored in DuckDB table raw_pages ✅")

if __name__ == "__main__":
    main()

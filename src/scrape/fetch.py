"""HTTP fetch utilities with browser-like headers and UTC timestamp helper."""

import requests
from datetime import datetime, timezone

def fetch_html(url: str) -> str:
    headers = {
        "User-Agent": "iditarod-model beginner project (learning scraper)"
    }
    resp = requests.get(url, headers=headers, timeout=30)
    resp.raise_for_status()
    return resp.text

def utc_now():
    return datetime.now(timezone.utc)

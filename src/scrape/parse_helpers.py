import re

def extract_musher_id_from_row(tr):
    """
    Prefer stable musher id from the musher profile link.
    Typical href: /race/2025/mushers/23-Dallas-Seavey/
    We'll use the numeric id '23' as musher_id (string).
    Returns None if not found.
    """
    a = tr.find("a", href=True)
    if not a:
        return None

    href = a["href"]
    # Look for ".../mushers/<id>-<slug>/"
    m = re.search(r"/mushers/(\d+)-", href)
    if m:
        return m.group(1)

    # Sometimes might be ".../mushers/<id>/"
    m = re.search(r"/mushers/(\d+)/", href)
    if m:
        return m.group(1)

    return None


def clean_text(s: str) -> str:
    return (s or "").strip()

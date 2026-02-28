"""
Match 2026 Iditarod mushers (by numeric ID from iditarod.com) to existing
database entries. Shows career history for veterans and flags rookies.

Usage:
    python match_2026_mushers.py
"""

import pandas as pd
from src.db import connect

# 2026 musher list: (musher_id from URL, name, rookie/veteran)
MUSHERS_2026 = [
    (1169, "Sam Martin", "Rookie"),
    (83, "Ryan Redington", "Veteran"),
    (1166, "Jesse Terry", "Rookie"),
    (1115, "Jaye Foucher", "Rookie"),
    (1167, "Kevin Hansen", "Rookie"),
    (1131, "Josi (Thyr) Shelley", "Veteran"),
    (1116, "Hanna Lyrek", "Veteran"),
    (1168, "Jody Potts-Joseph", "Rookie"),
    (344, "Michelle Phillips", "Veteran"),
    (1077, "Mille Porsild", "Veteran"),
    (1170, "Sam Paperman", "Rookie"),
    (290, "Jeff Deeter", "Veteran"),
    (1171, "Sadie Lindquist", "Rookie"),
    (1085, "Grayson Bruton", "Veteran"),
    (1103, "Brenda Mackey", "Rookie"),
    (328, "Wade Marrs", "Veteran"),
    (359, "Peter Kaiser", "Veteran"),
    (1080, "Riley Dyche", "Veteran"),
    (80, "Jason Mackey", "Veteran"),
    (1163, "Joseph Sabin", "Rookie"),
    (1075, "Gabe Dunham", "Veteran"),
    (1152, "Keaton Loebrich", "Veteran"),
    (394, "Travis Beals", "Veteran"),
    (284, "Rohn Buser", "Veteran"),
    (1052, "Jessie Holmes", "Veteran"),
    (1096, "Chad Stoddard", "Veteran"),
    (1155, "Sydnie Bahl", "Rookie"),
    (1127, "Bailey Vitello", "Veteran"),
    (1044, "Matt Hall", "Veteran"),
    (1164, "Adam Lindenmuth", "Rookie"),
    (1134, "Lauro Eklund", "Veteran"),
    (1069, "Richie Beattie", "Rookie"),
    (1165, "Kjell Rokke", "Rookie"),
    (994, "Thomas Waerner", "Veteran"),
    (400, "Paige Drobny", "Veteran"),
    (45, "Jessie Royer", "Veteran"),
]


def main():
    con = connect()

    hist = con.execute("""
        SELECT musher_id,
               MIN(year) as first_year,
               MAX(year) as last_year,
               COUNT(DISTINCT year) as n_years,
               MIN(finish_place) as best_finish
        FROM entries
        GROUP BY musher_id
    """).df()

    hist["musher_id"] = hist["musher_id"].astype(str)

    matched = []
    no_history = []

    for mid, name, status in MUSHERS_2026:
        mid_str = str(mid)
        row = hist[hist["musher_id"] == mid_str]

        if len(row) > 0:
            r = row.iloc[0]
            bf = r["best_finish"]
            best = int(bf) if pd.notna(bf) else None
            matched.append({
                "id": mid,
                "name": name,
                "status": status,
                "first_year": int(r["first_year"]),
                "last_year": int(r["last_year"]),
                "n_years": int(r["n_years"]),
                "best_finish": best,
            })
        else:
            no_history.append({"id": mid, "name": name, "status": status})

    print("=" * 90)
    print("MUSHERS WITH HISTORY IN DATABASE")
    print("=" * 90)
    for m in sorted(matched, key=lambda x: (x["best_finish"] or 999)):
        bf = m["best_finish"] or "N/A"
        print(f"  ID {m['id']:>5}  {m['name']:<28} "
              f"best: {bf:<4} races: {m['n_years']:<3} "
              f"({m['first_year']}-{m['last_year']})")

    print(f"\n  Found in DB: {len(matched)}/{len(MUSHERS_2026)}")

    print()
    print("=" * 90)
    print("MUSHERS WITH NO HISTORY (rookies or new to our dataset)")
    print("=" * 90)
    for m in no_history:
        print(f"  ID {m['id']:>5}  {m['name']:<28} ({m['status']})")

    print(f"\n  No history: {len(no_history)}/{len(MUSHERS_2026)}")


if __name__ == "__main__":
    main()
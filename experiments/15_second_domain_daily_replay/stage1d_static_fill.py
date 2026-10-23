#!/usr/bin/env python3
"""Stage 1d: fill pages that had zero revisions in the 2026-07-01..21
window with their exact t0 boundary revision (constant series).
The revision log certifies the page did not change, so the constant
series is exact; pages are flagged static_in_window=true."""
import hashlib
import json
import os

from stage1c_fetch_expanded import (TICKS, api, fetch_contents,
                                    extract_infobox, strip_wiki, lead_text)

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "series21.json")
T0 = "2026-07-01T00:00:00Z"


def boundary_rev_before(title, ts):
    d = api(dict(action="query", prop="revisions", titles=title,
                 rvprop="ids|timestamp", rvlimit="1", rvdir="older",
                 rvstart=ts))
    pages = d.get("query", {}).get("pages", [])
    revs = pages[0].get("revisions", []) if pages else []
    return revs[0]["revid"] if revs else None


def main():
    data = json.load(open(OUT))
    for title, p in list(data["pages"].items()):
        if "error" not in p or p["error"] != "no revisions":
            continue
        rid = boundary_rev_before(title, T0)
        if rid is None:
            print(f"[fail] {title}: no revision before t0"); continue
        content = fetch_contents([rid]).get(rid, "")
        kv, rest = extract_infobox(content)
        snap = {"infobox": {k: strip_wiki(v) for k, v in kv.items()},
                "lead": lead_text(rest),
                "hash": hashlib.md5(content.encode()).hexdigest()[:12]}
        data["pages"][title] = {
            "group": p["group"], "n_revisions": 0,
            "static_in_window": True,
            "series": {day: dict(snap) for day in TICKS}}
        json.dump(data, open(OUT, "w"))
        print(f"[fill] {title} <- rev {rid}")
    print("done")


if __name__ == "__main__":
    main()

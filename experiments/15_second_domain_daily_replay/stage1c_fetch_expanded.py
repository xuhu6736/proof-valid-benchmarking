#!/usr/bin/env python3
"""Stage 1: fetch daily (infobox KV + lead text) series for the candidate
page pool over t0..t7 (2026-07-01 .. 2026-07-08)."""
import hashlib
import json
import os
import re
import time
import urllib.error
import urllib.parse
import urllib.request

PROXY = os.environ.get("HTTP_PROXY", "http://127.0.0.1:7890")
API = "https://en.wikipedia.org/w/api.php"
OPENER = urllib.request.build_opener(
    urllib.request.ProxyHandler({"http": PROXY, "https": PROXY}))
OPENER.addheaders = [("User-Agent", "PVCacheServe-second-scenario/0.3 (research)")]

TICKS = [f"2026-07-{d:02d}" for d in range(1, 22)]   # 07-01 .. 07-08
REV_START = "2026-06-30T00:00:00Z"
REV_END = "2026-07-22T00:00:00Z"

CANDIDATES = {
    "wc_player": [
        "Kylian Mbappé", "Erling Haaland", "Jude Bellingham",
        "Vinícius Júnior", "Lamine Yamal", "Harry Kane", "Lionel Messi",
        "Cristiano Ronaldo", "Mohamed Salah", "Kevin De Bruyne",
        "Robert Lewandowski", "Son Heung-min", "Lautaro Martínez",
        "Bukayo Saka", "Phil Foden", "Pedri", "Jamal Musiala",
        "Florian Wirtz", "Antoine Griezmann", "Rodri", "Luka Modrić",
        "Neymar", "Virgil van Dijk", "Alisson Becker",
        "Khvicha Kvaratskhelia", "Rafael Leão", "Bruno Fernandes",
        "Bernardo Silva", "Declan Rice", "Cole Palmer", "Ollie Watkins",
        "Federico Valverde", "Darwin Núñez", "Julián Álvarez",
        "Alexis Mac Allister", "Enzo Fernández", "Aurélien Tchouaméni",
        "Eduardo Camavinga", "William Saliba", "Rúben Dias",
        "Bruno Guimarães", "Rodrygo", "Endrick",
        "Trent Alexander-Arnold", "Andrew Robertson", "Dominik Szoboszlai",
        "Ryan Gravenberch", "Cody Gakpo", "Xavi Simons", "Joshua Kimmich",
        "Leroy Sané", "Serge Gnabry", "Kai Havertz", "Nicolò Barella",
    ],
    "wc_team": [
        "Argentina national football team", "France national football team",
        "England national football team", "Spain national football team",
        "Germany national football team", "Brazil national football team",
        "Portugal national football team", "Netherlands national football team",
        "Croatia national football team", "Belgium national football team",
        "Morocco national football team", "Japan national football team",
        "United States men's national soccer team",
        "Mexico national football team", "Uruguay national football team",
        "Colombia national football team",
    ],
    "event": [
        "2026 FIFA World Cup", "2026 FIFA World Cup final",
        "2026 FIFA World Cup knockout stage",
        "2026 Wimbledon Championships", "2026 Tour de France",
        "2026 Formula One World Championship", "2026 NBA Finals",
        "2026 Stanley Cup Finals",
        "2026 FIFA World Cup group stage", "2026 FIFA World Cup squads",
        "2026 Wimbledon Championships – Men's singles",
        "2026 Wimbledon Championships – Women's singles",
        "2026 British Grand Prix", "2026 Austrian Grand Prix",
    ],
    "weather": [
        "2026 Atlantic hurricane season", "2026 Pacific typhoon season",
        "2026 Pacific hurricane season", "2026 North Indian Ocean cyclone season",
        "2026–27 South-West Indian Ocean cyclone season",
        "Tropical cyclones in 2026",
    ],
    "control": [
        "Claude Monet", "Mount Everest", "Photosynthesis", "Plate tectonics",
        "Marie Curie", "Theory of relativity", "Amazon River",
        "Periodic table", "William Shakespeare", "Great Wall of China",
    ],
}


def api(params):
    params = dict(params, format="json", formatversion="2")
    url = API + "?" + urllib.parse.urlencode(params)
    for attempt in range(8):
        try:
            with OPENER.open(url, timeout=40) as r:
                return json.loads(r.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            if e.code == 429 and attempt < 7:
                time.sleep(min(60, 15 * (attempt + 1))); continue
            if attempt == 7: raise
            time.sleep(2 * (attempt + 1))
        except Exception:
            if attempt == 7: raise
            time.sleep(2 * (attempt + 1))


def list_revisions(title):
    out, cont = [], {}
    while True:
        d = api(dict(action="query", prop="revisions", titles=title,
                     rvprop="ids|timestamp", rvlimit="max", rvdir="newer",
                     rvstart=REV_START, rvend=REV_END, **cont))
        pages = d.get("query", {}).get("pages", [])
        if pages and "revisions" in pages[0]:
            out += [(r["revid"], r["timestamp"]) for r in pages[0]["revisions"]]
        if "continue" in d:
            cont = d["continue"]
        else:
            break
    return out


def boundary_revisions(revs):
    import datetime as dt
    ts = [(rid, dt.datetime.strptime(t, "%Y-%m-%dT%H:%M:%SZ")) for rid, t in revs]
    res = {}
    for day in TICKS:
        end = dt.datetime.strptime(day, "%Y-%m-%d") + dt.timedelta(days=1)
        elig = [rid for rid, t in ts if t <= end]
        if elig:
            res[day] = elig[-1]
    return res


def fetch_contents(revids):
    out = {}
    revids = list(dict.fromkeys(revids))
    for i in range(0, len(revids), 50):
        d = api(dict(action="query", prop="revisions",
                     revids="|".join(map(str, revids[i:i + 50])),
                     rvprop="ids|content", rvslots="main"))
        for p in d.get("query", {}).get("pages", []):
            for r in p.get("revisions", []):
                out[r["revid"]] = r.get("slots", {}).get("main", {}).get("content", "")
    return out


INFOBOX_RE = re.compile(r"\{\{\s*[Ii]nfobox")
KV_RE = re.compile(r"^\s*\|\s*([^=|{}]+?)\s*=\s*(.*?)\s*$")


def extract_infobox(text):
    m = INFOBOX_RE.search(text)
    if not m:
        return {}, text
    depth, i = 0, m.start()
    while i < len(text) - 1:
        if text[i:i + 2] == "{{":
            depth += 1; i += 2; continue
        if text[i:i + 2] == "}}":
            depth -= 1; i += 2
            if depth == 0:
                break
            continue
        i += 1
    block, kv = text[m.start():i], {}
    for line in block.splitlines():
        mm = KV_RE.match(line)
        if mm:
            k = re.sub(r"\s+", " ", mm.group(1)).strip().lower()
            v = mm.group(2).strip()
            if k and v and len(k) < 60:
                kv[k] = v
    return kv, text[i:]   # remainder starts after the infobox


def strip_wiki(s):
    s = re.sub(r"<ref[^>]*>.*?</ref>|<ref[^/]*/>", "", s, flags=re.S)
    s = re.sub(r"<!--.*?-->", "", s, flags=re.S)
    prev = None
    while prev != s:                       # peel nested templates
        prev = s
        s = re.sub(r"\{\{[^{}]*\}\}", "", s)
    s = re.sub(r"\[\[([^]|]*)\|([^]]*)\]\]", r"\2", s)
    s = re.sub(r"\[\[([^]]*)\]\]", r"\1", s)
    s = re.sub(r"\[https?://\S+\s*([^]]*)\]", r"\1", s)
    s = re.sub(r"'{2,5}", "", s)
    s = re.sub(r"<[^>]+>", "", s)
    return re.sub(r"\s+", " ", s).strip()


def lead_text(after_infobox):
    # first ~2 paragraphs of body text after the infobox
    paras = [p for p in after_infobox.split("\n\n") if p.strip()
             and not p.strip().startswith(("|", "!", "{|", "=="))]
    return strip_wiki("\n\n".join(paras[:2]))[:4000]


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    out = os.path.join(here, "series21.json")
    data = {"ticks": TICKS, "pages": {}}
    if os.path.exists(out):
        data = json.load(open(out))
    for group, titles in CANDIDATES.items():
        for title in titles:
            if title in data["pages"] and "error" not in data["pages"][title]:
                print(f"[skip] {title}", flush=True); continue
            print(f"[fetch] {title}", flush=True)
            try:
                revs = list_revisions(title)
                bmap = boundary_revisions(revs)
                contents = fetch_contents(bmap.values())
            except Exception as e:
                data["pages"][title] = {"group": group, "error": str(e)}
                json.dump(data, open(out, "w")); continue
            if not bmap:
                data["pages"][title] = {"group": group, "error": "no revisions"}
                json.dump(data, open(out, "w")); continue
            series = {}
            for day in TICKS:
                rid = bmap.get(day)
                if rid is None or rid not in contents:
                    continue
                kv, rest = extract_infobox(contents[rid])
                series[day] = {
                    "infobox": {k: strip_wiki(v) for k, v in kv.items()},
                    "lead": lead_text(rest),
                    "hash": hashlib.md5(contents[rid].encode()).hexdigest()[:12],
                }
            data["pages"][title] = {"group": group, "n_revisions": len(revs),
                                    "series": series}
            json.dump(data, open(out, "w"))
            print(f"        ticks={len(series)}", flush=True)
    print("saved", out)


if __name__ == "__main__":
    main()

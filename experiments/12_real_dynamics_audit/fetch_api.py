"""轻量 API 抓取：面板页面两期 wikitext + pageprops QID 映射 + 基期 truthy 声明。
幂等可重入：每批一个 checkpoint 文件。用法: python3 fetch_api.py {pages|claims|merge}
"""
import json, os, re, sys, time
import urllib.parse, urllib.request

OUT = "/mnt/agents/output/pvb_pilot/data/api"
EN = "https://en.wikipedia.org/w/api.php"
WD = "https://www.wikidata.org/w/api.php"
UA = {"User-Agent": "pvb-constructor-study/1.0 (research; contact: anonymous)"}
DATES = {"20260401": "2026-04-01T00:00:00Z", "20260801": "2026-08-01T00:00:00Z"}


def api_get(base, params, retries=8):
    params = dict(params, format="json", formatversion="2")
    url = base + "?" + urllib.parse.urlencode(params)
    for a in range(retries):
        try:
            req = urllib.request.Request(url, headers=UA)
            with urllib.request.urlopen(req, timeout=120) as r:
                return json.loads(r.read().decode("utf-8", "replace"))
        except Exception as e:
            print(f"retry{a}: {repr(e)[:80]}", flush=True)
            time.sleep(3 + a * 3)
    raise RuntimeError("FAILED " + url[:200])


def resolve_maps(resp, requested):
    """requested title -> final page title (after normalization/redirects)"""
    m = {}
    for x in resp.get("query", {}).get("normalized", []):
        m[x["from"]] = x["to"]
    red = {}
    for x in resp.get("query", {}).get("redirects", []):
        red[x["from"]] = x["to"]
    out = {}
    for t in requested:
        t2 = m.get(t, t)
        seen = set()
        while t2 in red and t2 not in seen:
            seen.add(t2)
            t2 = red[t2]
        out[t] = t2
    return out


def stage_pages():
    titles = open(f"{OUT}/target_titles.txt").read().split("\n")
    B = 50
    for date, ts in DATES.items():
        batches = [titles[i:i + B] for i in range(0, len(titles), B)]
        for bi, batch in enumerate(batches):
            ck = f"{OUT}/pages_{date}_{bi:03d}.json"
            if os.path.exists(ck):
                continue
            resp = api_get(EN, {
                "action": "query", "prop": "revisions|pageprops",
                "rvstart": ts, "rvdir": "older", "rvlimit": "1",
                "rvslots": "main", "rvprop": "timestamp|content",
                "redirects": "1", "maxlag": "5",
                "titles": "|".join(batch)})
            fmap = resolve_maps(resp, batch)
            pages = {p["title"]: p for p in resp.get("query", {}).get("pages", [])}
            recs = {}
            for req_t in batch:
                ft = fmap.get(req_t, req_t)
                p = pages.get(ft)
                if not p or "missing" in p or "revisions" not in p:
                    recs[req_t] = None
                    continue
                rev = p["revisions"][0]
                recs[req_t] = {
                    "final_title": ft,
                    "qid": p.get("pageprops", {}).get("wikibase_item"),
                    "rev_ts": rev.get("timestamp"),
                    "text": rev.get("slots", {}).get("main", {}).get("content", ""),
                }
            json.dump(recs, open(ck, "w"))
            print(f"pages {date} batch {bi}/{len(batches)} ok={sum(v is not None for v in recs.values())}", flush=True)
            time.sleep(0.4)


def truthy_slim(claims):
    """claims -> {prop: [[rank, snaktype, value_norm], ...]} truthy only"""
    out = {}
    for prop, stmts in claims.items():
        prefs = [s for s in stmts if s.get("rank") == "preferred"]
        use = prefs if prefs else [s for s in stmts if s.get("rank") == "normal"]
        sl = []
        for s in use:
            ms = s.get("mainsnak", {})
            st = ms.get("snaktype")
            dv = ms.get("datavalue", {}).get("value") if st == "value" else None
            v = None
            if isinstance(dv, dict):
                v = dv.get("id") or dv.get("text") or dv.get("time") or dv.get("amount")
            elif dv is not None:
                v = str(dv)
            if v is not None:
                sl.append([s.get("rank"), st, str(v)])
        if sl:
            out[prop] = sl
    return out


def stage_claims():
    # 从 pages_20260401 的 pageprops 汇总 QID
    qids = set()
    for fn in sorted(os.listdir(OUT)):
        if fn.startswith("pages_20260401_") and fn.endswith(".json"):
            for v in json.load(open(f"{OUT}/{fn}")).values():
                if v and v.get("qid"):
                    qids.add(v["qid"])
    qids = sorted(qids)
    print("qids:", len(qids), flush=True)
    B = 20
    batches = [qids[i:i + B] for i in range(0, len(qids), B)]
    for bi, batch in enumerate(batches):
        ck = f"{OUT}/claims_20260401_{bi:03d}.json"
        if os.path.exists(ck):
            continue
        resp = api_get(WD, {
            "action": "query", "prop": "revisions",
            "rvstart": DATES["20260401"], "rvdir": "older", "rvlimit": "1",
            "rvslots": "main", "rvprop": "timestamp|content",
            "maxlag": "5", "titles": "|".join(batch)})
        pages = {p["title"]: p for p in resp.get("query", {}).get("pages", [])}
        recs = {}
        for q in batch:
            p = pages.get(q)
            if not p or "missing" in p or "revisions" not in p:
                recs[q] = None
                continue
            rev = p["revisions"][0]
            txt = rev.get("slots", {}).get("main", {}).get("content", "")
            try:
                ent = json.loads(txt)
                if "entities" in ent:
                    ent = list(ent["entities"].values())[0]
                recs[q] = {"rev_ts": rev.get("timestamp"),
                           "claims": truthy_slim(ent.get("claims", {})),
                           "label": (ent.get("labels", {}).get("en") or {}).get("value")}
            except Exception as e:
                recs[q] = None
                print(f"claims parse fail {q}: {repr(e)[:60]}", flush=True)
        json.dump(recs, open(ck, "w"))
        print(f"claims batch {bi}/{len(batches)} ok={sum(v is not None for v in recs.values())}", flush=True)
        time.sleep(0.4)


def merge():
    for prefix, out_name in [("pages_20260401", "pages_20260401.json"),
                             ("pages_20260801", "pages_20260801.json"),
                             ("claims_20260401", "claims_20260401.json")]:
        merged = {}
        for fn in sorted(os.listdir(OUT)):
            if re.fullmatch(prefix + r"_\d{3}\.json", fn):
                merged.update(json.load(open(f"{OUT}/{fn}")))
        json.dump(merged, open(f"{OUT}/{out_name}", "w"))
        n_ok = sum(v is not None for v in merged.values())
        sz = os.path.getsize(f"{OUT}/{out_name}") / 1e6
        print(f"{out_name}: {len(merged)} keys, {n_ok} ok, {sz:.1f}MB", flush=True)


if __name__ == "__main__":
    {"pages": stage_pages, "claims": stage_claims, "merge": merge}[sys.argv[1]]()

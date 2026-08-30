"""Wikidata 管线 v2（修复 pageid≠QID 的匹配错误）。
v1 的 scan_wd 用索引第二列(pageid)匹配 QID 号码；实际上索引第三列(标题 "Q15")
才是 QID。v2 按标题匹配，并以实体 JSON 自身的 id 键存储。
用法: python3 wd_pipeline_v2.py {listing|dlidx|scan|fetch|all} [date ...]
"""
import bz2, bisect, json, os, re, sys, time, urllib.request
from concurrent.futures import ThreadPoolExecutor

WORK = "/tmp/pvb_data"
WD = "https://ftp.acc.umu.se/mirror/wikimedia.org/dumps/wikidatawiki"
WD_DATES = ["20260401", "20260501", "20260701", "20260801"]

def http_get(url, timeout=90, retries=8):
    for a in range(retries):
        try:
            return urllib.request.urlopen(url, timeout=timeout).read()
        except Exception as e:
            print(f"retry{a} {url[-60:]}: {repr(e)[:60]}", flush=True); time.sleep(2 + a * 2)
    raise RuntimeError(f"FAILED {url}")

def stage_listing(date):
    out = f"{WORK}/wd_shards_{date}.json"
    if os.path.exists(out):
        print(date, "listing cached", flush=True); return
    html = http_get(f"{WD}/{date}/").decode("utf-8", "replace")
    shards = {}
    for m in re.finditer(r'href="(wikidatawiki-' + date + r'-pages-articles-multistream(\d+)\.xml-p(\d+)p(\d+)\.bz2)"', html):
        name, n, pa, pb = m.group(1), int(m.group(2)), int(m.group(3)), int(m.group(4))
        pos = html.find(f">{name}<")
        sm = re.search(r'indexcolsize">\s*([\d.]+)([KMG]?)\s*</td>', html[pos:pos + 500])
        size = None
        if sm:
            mult = {"": 1, "K": 1024, "M": 1024**2, "G": 1024**3}[sm.group(2)]
            size = int(float(sm.group(1)) * mult)
        shards[name] = {"shard": n, "p_start": pa, "p_end": pb, "size": size}
    json.dump(shards, open(out, "w"))
    print(date, "shards:", len(shards), flush=True)

def stage_dlidx(date):
    """下载该期全部分片索引（标题匹配需要全量扫描）"""
    d = f"{WORK}/wdidx2_{date}"
    os.makedirs(d, exist_ok=True)
    shards = json.load(open(f"{WORK}/wd_shards_{date}.json"))
    def dl(name):
        idx = re.sub(r"multistream(\d+)\.xml-", r"multistream-index\1.txt-", name)
        out = f"{d}/{idx}"
        if os.path.exists(out):
            return
        try:
            open(out, "wb").write(http_get(f"{WD}/{date}/{idx}"))
        except Exception as e:
            print("idx FAIL", idx, repr(e)[:60], flush=True)
    with ThreadPoolExecutor(max_workers=4) as ex:
        list(ex.map(dl, sorted(shards)))
    print(date, "indexes:", len(os.listdir(d)), "/", len(shards), flush=True)

def stage_scan(date):
    """按索引标题列 Q{qid} 匹配（修复点）；记录 pageid 供块内定位"""
    qids = set(str(q) for q in json.load(open(f"{WORK}/title_qid_map.json")).values())
    hits, offs = {}, {}
    d = f"{WORK}/wdidx2_{date}"
    for fn in sorted(os.listdir(d)):
        shard_offs = set()
        with bz2.open(f"{d}/{fn}", "rt", errors="replace") as f:
            for line in f:
                p = line.rstrip("\n").split(":", 2)
                if len(p) < 3:
                    continue
                try:
                    off, pid = int(p[0]), int(p[1])
                except ValueError:
                    continue
                shard_offs.add(off)
                t = p[2]
                if t.startswith("Q") and t[1:] in qids:
                    hits[t[1:]] = (fn, off, pid)
        offs[fn] = sorted(shard_offs)
    json.dump({"hits": hits, "offsets": offs}, open(f"{WORK}/wd_qid_offsets2_{date}.json", "w"))
    print(date, "qid hits:", len(hits), "/", len(qids), flush=True)

def stage_fetch(date):
    out_path = f"{WORK}/entities_v2_{date}.json"
    if os.path.exists(out_path) and os.path.getsize(out_path) > 1000:
        print(date, "entities cached, skip", flush=True); return
    data = json.load(open(f"{WORK}/wd_qid_offsets2_{date}.json"))
    hits, offsets = data["hits"], data["offsets"]
    shards = json.load(open(f"{WORK}/wd_shards_{date}.json"))
    idx2xml = lambda fn: re.sub(r"multistream-index(\d+)\.txt-", r"multistream\1.xml-", fn)
    by_block = {}
    for qid, (idx_fn, off, pid) in hits.items():
        by_block.setdefault((idx2xml(idx_fn), off), {})[pid] = qid
    def bend(idx_fn, off):
        offs = offsets[idx_fn]
        i = bisect.bisect_right(offs, off)
        if i < len(offs):
            return offs[i]
        sz = shards.get(idx2xml(idx_fn), {}).get("size")
        return sz if sz else off + 3 * 1024 * 1024
    def fetch(key):
        xml_fn, off = key
        idx_fn = re.sub(r"multistream(\d+)\.xml-", r"multistream-index\1.txt-", xml_fn)
        end = bend(idx_fn, off) - 1
        url = f"{WD}/{date}/{xml_fn}"
        for a in range(4):
            try:
                r = urllib.request.urlopen(urllib.request.Request(
                    url, headers={"Range": f"bytes={off}-{end}"}), timeout=90)
                raw = r.read(); assert len(raw) == end - off + 1
                return key, bz2.decompress(raw)
            except Exception:
                time.sleep(2 + a * 3)
        try:
            r = urllib.request.urlopen(urllib.request.Request(
                url, headers={"Range": f"bytes={off}-{off + 3*1024*1024}"}), timeout=90)
            d = bz2.BZ2Decompressor()
            return key, d.decompress(r.read())
        except Exception:
            print("FAIL", date, xml_fn, off, flush=True); return key, None
    import html as _html
    entities = {}
    with ThreadPoolExecutor(max_workers=4) as ex:
        for key, xml in ex.map(fetch, sorted(by_block)):
            if xml is None:
                continue
            wanted = by_block[key]  # pageid -> qid(按索引标题)
            for m in re.finditer(rb"<page>(.*?)</page>", xml, re.S):
                c = m.group(1)
                mid = re.search(rb"<id>(\d+)</id>", c)
                if not mid or int(mid.group(1)) not in wanted:
                    continue
                t = re.search(rb"<text[^>]*>(.*?)</text>", c, re.S)
                if not t:
                    continue
                try:
                    ent = json.loads(_html.unescape(t.group(1).decode("utf-8", "replace")))
                    eid = str(ent.get("id", ""))  # 以实体自身 id 为准
                    if eid.startswith("Q"):
                        eid = eid[1:]
                    claims = ent.get("claims", {})
                    slim = {}
                    for prop, stmts in claims.items():
                        sl = []
                        for s in stmts:
                            ms = s.get("mainsnak", {})
                            dv = ms.get("datavalue", {}).get("value")
                            if isinstance(dv, dict):
                                dv = {k: dv.get(k) for k in ("id", "text", "time", "amount") if k in dv}
                            sl.append([s.get("rank"), ms.get("snaktype"), dv])
                        slim[prop] = sl
                    lab = (ent.get("labels", {}).get("en") or {}).get("value")
                    entities[eid] = {"claims": slim, "label": lab,
                                     "qid_by_index": wanted[int(mid.group(1))]}
                except Exception:
                    pass
    json.dump(entities, open(out_path, "w"))
    print(date, "entities DONE", len(entities), flush=True)

if __name__ == "__main__":
    s = sys.argv[1]
    dates = sys.argv[2:] or WD_DATES
    if s == "all":
        for d in dates:
            stage_listing(d); stage_dlidx(d); stage_scan(d); stage_fetch(d)
    else:
        fn = {"listing": stage_listing, "dlidx": stage_dlidx,
              "scan": stage_scan, "fetch": stage_fetch}[s]
        for d in dates:
            fn(d)

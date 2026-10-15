"""Wikidata 四期声明抓取管线：listing → qidmap → 分片索引 → 实体块 Range 抓取 → claims 提取。
阶段化、幂等可重入。用法: python3 wd_pipeline.py {listing|qidmap|dl_wdidx|scan_wd|fetch_wd} [date]
"""
import bz2, bisect, json, os, re, sys, time, urllib.request, gzip
from concurrent.futures import ThreadPoolExecutor

WORK = "/tmp/pvb_data"
WD = "https://ftp.acc.umu.se/mirror/wikimedia.org/dumps/wikidatawiki"
WD_DATES = ["20260401", "20260501", "20260701", "20260801"]  # 20260601 不存在

def http_get(url, timeout=60, retries=8):
    for a in range(retries):
        try:
            return urllib.request.urlopen(url, timeout=timeout).read()
        except Exception as e:
            print(f"retry{a} {url}: {repr(e)[:60]}", flush=True); time.sleep(2 + a * 2)
    raise RuntimeError(f"FAILED {url}")

def stage_listing(date):
    html = http_get(f"{WD}/{date}/").decode("utf-8", "replace")
    open(f"{WORK}/listing_wd_{date}.html", "w").write(html)
    # multistream 分片: wikidatawiki-{date}-pages-articles-multistream{N}.xml-p{A}p{B}.bz2
    shards = {}
    for m in re.finditer(r'href="(wikidatawiki-' + date + r'-pages-articles-multistream(\d+)\.xml-p(\d+)p(\d+)\.bz2)"', html):
        name, n, pa, pb = m.group(1), int(m.group(2)), int(m.group(3)), int(m.group(4))
        # 文件大小在链接后 indexcolsize 列, 形如 2.1G / 512M
        pos = html.find(f">{name}<")
        tail = html[pos:pos + 500]
        sm = re.search(r'indexcolsize">\s*([\d.]+)([KMG]?)\s*</td>', tail)
        size = None
        if sm:
            mult = {"": 1, "K": 1024, "M": 1024**2, "G": 1024**3}[sm.group(2)]
            size = int(float(sm.group(1)) * mult)
        shards[name] = {"shard": n, "p_start": pa, "p_end": pb, "size": size}
    json.dump(shards, open(f"{WORK}/wd_shards_{date}.json", "w"))
    print(date, "shards:", len(shards), flush=True)

def stage_qidmap():
    """page_props.gz: (pageid,'wikibase_item',QID) → 与 scan_20260401 的 title→pageid 连接"""
    scan = json.load(open(f"{WORK}/scan_20260401.json"))
    title_pid = {t: pid for t, (off, pid) in scan["hits"].items()}
    pid_title = {pid: t for t, pid in title_pid.items()}
    wanted = set(pid_title)
    title_qid = {}
    pat = re.compile(r"\((\d+),'wikibase_item','Q?(\d+)'")
    with gzip.open(f"{WORK}/page_props_20260401.sql.gz", "rt", errors="replace") as f:
        for line in f:
            if "wikibase_item" not in line:
                continue
            for m in pat.finditer(line):
                pid = int(m.group(1))
                if pid in wanted:
                    title_qid[pid_title[pid]] = int(m.group(2))
            if len(title_qid) == len(wanted):
                break
    json.dump(title_qid, open(f"{WORK}/title_qid_map.json", "w"))
    print("qidmap:", len(title_qid), "/", len(wanted), flush=True)

def needed_shards(date):
    """目标 QID 落在哪些分片页码范围内"""
    title_qid = json.load(open(f"{WORK}/title_qid_map.json"))
    qids = sorted(title_qid.values())
    shards = json.load(open(f"{WORK}/wd_shards_{date}.json"))
    out = {}
    for name, s in shards.items():
        # 范围 [p_start, p_end] 与 qids 有交集
        i = bisect.bisect_left(qids, s["p_start"])
        if i < len(qids) and qids[i] <= s["p_end"]:
            out[name] = s
    return out

def stage_dl_wdidx(date):
    os.makedirs(f"{WORK}/wdidx_{date}", exist_ok=True)
    shards = needed_shards(date)
    def dl(item):
        name, s = item
        # 实际索引文件名: ...multistream-index{N}.txt-p{A}p{B}.bz2
        idx_name = re.sub(r"multistream(\d+)\.xml-", r"multistream-index\1.txt-", name)
        out = f"{WORK}/wdidx_{date}/{idx_name}"
        if os.path.exists(out):
            return
        try:
            data = http_get(f"{WD}/{date}/{idx_name}")
            open(out, "wb").write(data)
        except Exception as e:
            print("idx FAIL", idx_name, repr(e)[:60], flush=True)
    with ThreadPoolExecutor(max_workers=4) as ex:
        list(ex.map(dl, sorted(shards.items())))
    n = len(os.listdir(f"{WORK}/wdidx_{date}"))
    print(date, "shard indexes:", n, "/", len(shards), flush=True)

def stage_scan_wd(date):
    title_qid = json.load(open(f"{WORK}/title_qid_map.json"))
    qid_title = {q: t for t, q in title_qid.items()}
    hits, offs = {}, {}  # shard_name -> sorted offsets
    d = f"{WORK}/wdidx_{date}"
    for fn in sorted(os.listdir(d)):
        shard_offs = set()
        with bz2.open(f"{d}/{fn}", "rt", errors="replace") as f:
            for line in f:
                p = line.rstrip("\n").split(":", 2)
                if len(p) < 2:
                    continue
                try:
                    off, pid = int(p[0]), int(p[1])
                except ValueError:
                    continue
                shard_offs.add(off)
                if pid in qid_title:
                    hits[pid] = (fn, off)
        offs[fn] = sorted(shard_offs)
    json.dump({"hits": {str(k): v for k, v in hits.items()}, "offsets": offs},
              open(f"{WORK}/wd_qid_offsets_{date}.json", "w"))
    print(date, "qid hits:", len(hits), flush=True)

def stage_fetch_wd(date):
    out_path = f"{WORK}/entities_{date}.json"
    if os.path.exists(out_path) and os.path.getsize(out_path) > 1000:
        print(date, "entities cached, skip", flush=True); return
    data = json.load(open(f"{WORK}/wd_qid_offsets_{date}.json"))
    hits, offsets = data["hits"], data["offsets"]
    shards = json.load(open(f"{WORK}/wd_shards_{date}.json"))
    # 按 (shard_file, offset) 分组
    idx2xml = lambda fn: re.sub(r"multistream-index(\d+)\.txt-", r"multistream\1.xml-", fn)
    by_block = {}
    for pid, (idx_fn, off) in hits.items():
        by_block.setdefault((idx2xml(idx_fn), off), []).append(int(pid))
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
                    url, headers={"Range": f"bytes={off}-{end}"}), timeout=60)
                raw = r.read(); assert len(raw) == end - off + 1
                return key, bz2.decompress(raw)
            except Exception:
                time.sleep(2 + a * 3)
        # 末尾分片: 大小未知时前缀解压
        try:
            r = urllib.request.urlopen(urllib.request.Request(
                url, headers={"Range": f"bytes={off}-{off + 3*1024*1024}"}), timeout=60)
            d = bz2.BZ2Decompressor()
            return key, d.decompress(r.read())
        except Exception:
            print("FAIL", date, xml_fn, off, flush=True); return key, None
    entities = {}
    with ThreadPoolExecutor(max_workers=4) as ex:
        for key, xml in ex.map(fetch, sorted(by_block)):
            if xml is None:
                continue
            for m in re.finditer(rb"<page>(.*?)</page>", xml, re.S):
                c = m.group(1)
                mid = re.search(rb"<id>(\d+)</id>", c)
                if not mid or int(mid.group(1)) not in by_block[key]:
                    continue
                t = re.search(rb"<text[^>]*>(.*?)</text>", c, re.S)
                if not t:
                    continue
                try:
                    import html as _html
                    ent = json.loads(_html.unescape(t.group(1).decode("utf-8", "replace")))
                    claims = ent.get("claims", {})
                    # 只保留 mainsnak 的 (prop, rank, datavalue) 以减小体积
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
                    entities[mid.group(1).decode()] = slim
                except Exception:
                    pass
    json.dump(entities, open(f"{WORK}/entities_{date}.json", "w"))
    print(date, "entities DONE", len(entities), flush=True)

if __name__ == "__main__":
    s = sys.argv[1]
    if s == "listing":
        for d in (sys.argv[2:] or WD_DATES):
            stage_listing(d)
    elif s == "qidmap":
        stage_qidmap()
    elif s in ("dl_wdidx", "scan_wd", "fetch_wd"):
        dates = sys.argv[2:] or WD_DATES
        fn = {"dl_wdidx": stage_dl_wdidx, "scan_wd": stage_scan_wd, "fetch_wd": stage_fetch_wd}[s]
        for d in dates:
            fn(d)

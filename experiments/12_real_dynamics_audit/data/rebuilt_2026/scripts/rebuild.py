"""PVSB 真实动态实验 Stage 0 重建驱动（2026-10-14）。
逐阶段幂等可重入，逻辑逐行复刻原始管线：
  full_orchestrator.py / wd_pipeline.py(v1 qidmap) / wd_pipeline_v2.py(实体抓取)
路径与镜像适配：WORK = 本目录；20260401 期走官方 dumps.wikimedia.org，
其余各期优先 ftp.acc.umu.se 镜像，失败回退官方。
用法: python3 rebuild.py {idx|scan|pages|validate_base|pageprops|qidmap|
                          wdlisting|wdidx|wdscan|wdfetch|wfd} [date ...]
"""
import bz2, bisect, gzip, json, os, re, sys, time, urllib.request
from concurrent.futures import ThreadPoolExecutor

UA = {"User-Agent": "pvb-rebuild/1.0 (research reproduction; contact: anonymous)"}
HERE = os.path.dirname(os.path.abspath(__file__))
WORK = HERE

EN_MIRROR = "https://ftp.acc.umu.se/mirror/wikimedia.org/dumps/enwiki"
EN_OFFICIAL = "https://dumps.wikimedia.org/enwiki"
WD_MIRROR = "https://ftp.acc.umu.se/mirror/wikimedia.org/dumps/wikidatawiki"
WD_OFFICIAL = "https://dumps.wikimedia.org/wikidatawiki"

DATES = ["20260401", "20260501", "20260601", "20260701", "20260801"]
IDX_SIZE = {"20260401": 280349081, "20260501": 281216902, "20260601": 281979710,
            "20260701": 282804233, "20260801": 283588691}
MAIN_SIZE = {"20260401": 26207949994, "20260501": 26319713415, "20260601": 26437250146,
             "20260701": 26564488717, "20260801": 26668484995}
PAGEPROPS_SIZE = 450210296
WFD_URL = "https://hf-mirror.com/datasets/Orange/WikiFactDiff/resolve/main/20210104-20230227_legacy/train.parquet"
WFD_SIZE = 177696222
WD_DATES = ["20260401", "20260501", "20260701", "20260801"]  # 20260601 不存在

def en_base(date):
    return EN_MIRROR if date != "20260401" else EN_OFFICIAL

def wd_base(date):
    return WD_MIRROR if date != "20260401" else WD_OFFICIAL

def chunked(urls, out, total, chunk=16*1024*1024, tag=""):
    """多镜像顺序尝试的断点续传；完成校验总大小。"""
    if os.path.exists(out) and os.path.getsize(out) == total:
        print(f"[{tag}] cached", flush=True); return True
    done = os.path.getsize(out) if os.path.exists(out) else 0
    if done % chunk:
        done = (done // chunk) * chunk
        with open(out, "r+b") as f: f.truncate(done)
    cur = 0
    with open(out, "r+b" if os.path.exists(out) else "wb") as f:
        while done < total:
            end = min(done + chunk - 1, total - 1)
            ok = False
            for url in urls:
                for a in range(6):
                    try:
                        r = urllib.request.urlopen(urllib.request.Request(
                            url, headers={**UA, "Range": f"bytes={done}-{end}"}), timeout=150)
                        data = r.read()
                        assert len(data) == end - done + 1, f"short {len(data)}"
                        f.seek(done); f.write(data); done = end + 1
                        ok = True; break
                    except Exception as e:
                        print(f"[{tag}] retry{a}@{done}: {repr(e)[:70]}", flush=True)
                        time.sleep(2 + a*2)
                if ok: break
            if not ok:
                print(f"[{tag}] FAILED at {done}", flush=True); return False
            cur += 1
            if cur % 8 == 0:
                print(f"[{tag}] {done/1e6:.0f}/{total/1e6:.0f}MB", flush=True)
    print(f"[{tag}] DONE", flush=True); return True

# ---------------- enwiki ----------------

def stage_idx(dates):
    def one(d):
        chunked([f"{en_base(d)}/{d}/enwiki-{d}-pages-articles-multistream-index.txt.bz2",
                 f"{EN_OFFICIAL}/{d}/enwiki-{d}-pages-articles-multistream-index.txt.bz2"],
                f"{WORK}/idx_{d}.bz2", IDX_SIZE[d], tag=f"idx{d}")
    with ThreadPoolExecutor(max_workers=4) as ex:
        list(ex.map(one, dates))

def stage_scan(date):
    target = set(open(f'{WORK}/target_titles.txt').read().split("\n"))
    hits, offs = {}, set()
    with bz2.open(f'{WORK}/idx_{date}.bz2', 'rt', errors='replace') as f:
        for line in f:
            p = line.rstrip("\n").split(":", 2)
            offs.add(int(p[0]))
            if len(p) == 3 and p[2] in target:
                hits[p[2]] = (int(p[0]), int(p[1]))
    json.dump({"hits": hits, "offsets": sorted(offs)}, open(f'{WORK}/scan_{date}.json', 'w'))
    print(date, "hits", len(hits), flush=True)

def fetch_pages(date):
    out_path = f'{WORK}/pages_{date}.json'
    if os.path.exists(out_path) and os.path.getsize(out_path) > 1000:
        print(date, "pages cached, skip", flush=True); return
    part_path = f'{WORK}/pages_{date}.partial.json'
    pages = {}
    if os.path.exists(part_path):
        pages = json.load(open(part_path))
        print(date, "resume with", len(pages), "pages", flush=True)
    scan = json.load(open(f'{WORK}/scan_{date}.json'))
    hits, all_offs = scan['hits'], scan['offsets']
    by_block = {}
    for title, (off, pid) in hits.items():
        by_block.setdefault(off, {})[pid] = title
    todo = {off: pids for off, pids in by_block.items()
            if not all(t in pages for t in pids.values())}
    print(date, "blocks todo:", len(todo), "/", len(by_block), flush=True)
    def bend(off):
        i = bisect.bisect_right(all_offs, off)
        return all_offs[i] if i < len(all_offs) else MAIN_SIZE[date]
    urls = [f"{en_base(date)}/{date}/enwiki-{date}-pages-articles-multistream.xml.bz2",
            f"{EN_OFFICIAL}/{date}/enwiki-{date}-pages-articles-multistream.xml.bz2"]
    def fetch(off):
        end = bend(off) - 1
        for url in urls:
            for a in range(4):
                try:
                    r = urllib.request.urlopen(urllib.request.Request(
                        url, headers={**UA, "Range": f"bytes={off}-{end}"}), timeout=90)
                    data = r.read(); assert len(data) == end - off + 1
                    return off, bz2.decompress(data)
                except Exception:
                    time.sleep(2 + a*3)
        print("FAIL", date, off, flush=True); return off, None
    done_count = 0
    with ThreadPoolExecutor(max_workers=10) as ex:
        for off, xml in ex.map(fetch, sorted(todo)):
            if xml is None: continue
            for m in re.finditer(rb"<page>(.*?)</page>", xml, re.S):
                c = m.group(1)
                mid = re.search(rb"<id>(\d+)</id>", c)
                if mid and int(mid.group(1)) in by_block[off]:
                    t = re.search(rb"<text[^>]*>(.*?)</text>", c, re.S)
                    pages[by_block[off][int(mid.group(1))]] = \
                        t.group(1).decode("utf-8", "replace") if t else ""
            done_count += 1
            if done_count % 100 == 0:
                json.dump(pages, open(part_path, "w"))
                print(date, "checkpoint", len(pages), "pages", flush=True)
    json.dump(pages, open(out_path, "w"))
    if os.path.exists(part_path): os.remove(part_path)
    print(date, "pages DONE", len(pages), flush=True)

# ---------------- 基期文本对账（go/no-go） ----------------

def wikitext_to_plain(wt):  # 与 eps_t.py / module2_analyze.py 完全一致
    wt = re.sub(r"<ref[^>]*>.*?</ref>", " ", wt, flags=re.S)
    wt = re.sub(r"<ref[^>]*/>", " ", wt)
    wt = re.sub(r"\{\{[^{}]*\}\}", " ", wt)
    wt = re.sub(r"\{\{[^{}]*\}\}", " ", wt)
    wt = re.sub(r"\[\[([^|\]]*\|)?([^\]]*)\]\]", r"\2", wt)
    wt = re.sub(r"\[https?://[^\s\]]+\s*([^\]]*)\]", r"\1", wt)
    wt = re.sub(r"'{2,}", "", wt)
    wt = re.sub(r"<[^>]+>", " ", wt)
    wt = re.sub(r"^[=]+.*?[=]+$", " ", wt, flags=re.M)
    return wt

def sentences(text, min_len=40):
    parts = re.split(r"(?<=[.!?。！？])\s+|\n+", text)
    return [re.sub(r"\s+", " ", s).strip() for s in parts if len(s.strip()) >= min_len]

def stage_validate_base():
    pages = json.load(open(f"{WORK}/pages_20260401.json"))
    base_sents = {}
    for t, wt in pages.items():
        s = sentences(wikitext_to_plain(wt))
        if s: base_sents[t] = s
    n_sent = sum(len(s) for s in base_sents.values())
    print(f"pages fetched: {len(pages)}")
    print(f"base pages with sentences: {len(base_sents)}  (retained anchor: 1749)")
    print(f"base total sentences:      {n_sent}  (retained anchor: 86441)")
    ok = len(base_sents) == 1749 and n_sent == 86441
    print("VALIDATE_BASE:", "EXACT MATCH" if ok else "MISMATCH", flush=True)

# ---------------- page_props -> qidmap ----------------

PP_SIZE = {"20260401": 450210296, "20260501": 452699369}

def stage_pageprops(date="20260401"):
    chunked([f"{en_base(date)}/{date}/enwiki-{date}-page_props.sql.gz",
             f"{EN_OFFICIAL}/{date}/enwiki-{date}-page_props.sql.gz"],
            f"{WORK}/page_props_{date}.sql.gz", PP_SIZE[date], tag=f"pageprops{date}")

def stage_qidmap(date="20260401"):
    scan = json.load(open(f"{WORK}/scan_{date}.json"))
    title_pid = {t: pid for t, (off, pid) in scan["hits"].items()}
    pid_title = {pid: t for t, pid in title_pid.items()}
    wanted = set(pid_title)
    title_qid = {}
    pat = re.compile(r"\((\d+),'wikibase_item','Q?(\d+)'")
    with gzip.open(f"{WORK}/page_props_{date}.sql.gz", "rt", errors="replace") as f:
        for line in f:
            if "wikibase_item" not in line: continue
            for m in pat.finditer(line):
                pid = int(m.group(1))
                if pid in wanted:
                    title_qid[pid_title[pid]] = int(m.group(2))
            if len(title_qid) == len(wanted): break
    json.dump(title_qid, open(f"{WORK}/title_qid_map_{date}.json", "w"))
    print("qidmap", date, ":", len(title_qid), "/", len(wanted), flush=True)

# ---------------- wikidata（v1 分片裁剪 + v2 标题匹配） ----------------

def http_get(url, timeout=90, retries=8):
    for a in range(retries):
        try:
            return urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=timeout).read()
        except Exception as e:
            print(f"retry{a} {url[-60:]}: {repr(e)[:60]}", flush=True); time.sleep(2 + a*2)
    raise RuntimeError(f"FAILED {url}")

def stage_wdlisting(date):
    out = f"{WORK}/wd_shards_{date}.json"
    if os.path.exists(out):
        print(date, "listing cached", flush=True); return
    try:
        html = http_get(f"{wd_base(date)}/{date}/").decode("utf-8", "replace")
    except RuntimeError:
        html = http_get(f"{WD_OFFICIAL}/{date}/").decode("utf-8", "replace")
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
    if not shards:
        # 官方站为状态页格式: <a href="/wikidatawiki/{date}/{name}">name</a> 2.1 GB
        html2 = http_get(f"{WD_OFFICIAL}/wikidatawiki/{date}/".replace("/wikidatawiki/wikidatawiki", "/wikidatawiki")).decode("utf-8", "replace")
        for m in re.finditer(r'href="/wikidatawiki/' + date + r'/(wikidatawiki-' + date + r'-pages-articles-multistream(\d+)\.xml-p(\d+)p(\d+)\.bz2)">[^<]*</a>\s*([\d.]+)\s*([KMG]?)B', html2):
            name, n, pa, pb = m.group(1), int(m.group(2)), int(m.group(3)), int(m.group(4))
            mult = {"": 1, "K": 1024, "M": 1024**2, "G": 1024**3}[m.group(6)]
            size = int(float(m.group(5)) * mult)
            shards[name] = {"shard": n, "p_start": pa, "p_end": pb, "size": size}
    json.dump(shards, open(out, "w"))
    print(date, "shards:", len(shards), flush=True)

def needed_shards(date):
    # 改回 v2 行为：全量分片索引（pageid≠QID 的实体落在范围外分片）
    return json.load(open(f"{WORK}/wd_shards_{date}.json"))

def stage_wdidx(date):
    d = f"{WORK}/wdidx2_{date}"
    os.makedirs(d, exist_ok=True)
    shards = needed_shards(date)
    def dl(name):
        idx = re.sub(r"multistream(\d+)\.xml-", r"multistream-index\1.txt-", name)
        out = f"{d}/{idx}"
        if os.path.exists(out): return
        for base in (wd_base(date), WD_OFFICIAL):
            try:
                open(out, "wb").write(http_get(f"{base}/{date}/{idx}")); return
            except Exception as e:
                print("idx retry", idx, repr(e)[:60], flush=True)
        print("idx FAIL", idx, flush=True)
    with ThreadPoolExecutor(max_workers=4) as ex:
        list(ex.map(dl, sorted(shards)))
    print(date, "needed shard indexes:", len(os.listdir(d)), "/", len(shards), flush=True)

def stage_wdscan(date):
    """v2: 按索引标题列 Q{qid} 匹配"""
    qids = set(str(q) for q in json.load(open(f"{WORK}/title_qid_map.json")).values())
    hits, offs = {}, {}
    d = f"{WORK}/wdidx2_{date}"
    for fn in sorted(os.listdir(d)):
        shard_offs = set()
        with bz2.open(f"{d}/{fn}", "rt", errors="replace") as f:
            for line in f:
                p = line.rstrip("\n").split(":", 2)
                if len(p) < 3: continue
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

def stage_wdfetch(date):
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
        if i < len(offs): return offs[i]
        sz = shards.get(idx2xml(idx_fn), {}).get("size")
        return sz if sz else off + 3 * 1024 * 1024
    def fetch(key):
        xml_fn, off = key
        end = bend(re.sub(r"multistream(\d+)\.xml-", r"multistream-index\1.txt-", xml_fn), off) - 1
        for base in (wd_base(date), WD_OFFICIAL):
            url = f"{base}/{date}/{xml_fn}"
            for a in range(4):
                try:
                    r = urllib.request.urlopen(urllib.request.Request(
                        url, headers={**UA, "Range": f"bytes={off}-{end}"}), timeout=90)
                    raw = r.read(); assert len(raw) == end - off + 1
                    return key, bz2.decompress(raw)
                except Exception:
                    time.sleep(2 + a * 3)
        try:
            url = f"{wd_base(date)}/{date}/{xml_fn}"
            r = urllib.request.urlopen(urllib.request.Request(
                url, headers={**UA, "Range": f"bytes={off}-{off + 3*1024*1024}"}), timeout=90)
            d = bz2.BZ2Decompressor()
            return key, d.decompress(r.read())
        except Exception:
            print("FAIL", date, xml_fn, off, flush=True); return key, None
    import html as _html
    part_path = f"{WORK}/entities_v2_{date}.partial.json"
    entities = {}
    if os.path.exists(part_path):
        entities = json.load(open(part_path))
        print(date, "resume with", len(entities), "entities", flush=True)
    todo = {k: v for k, v in by_block.items()
            if not all(str(q) in entities for q in v.values())}
    print(date, "blocks todo:", len(todo), "/", len(by_block), flush=True)
    done_n = 0
    with ThreadPoolExecutor(max_workers=8) as ex:
        for key, xml in ex.map(fetch, sorted(todo)):
            if xml is None: continue
            wanted = by_block[key]
            for m in re.finditer(rb"<page>(.*?)</page>", xml, re.S):
                c = m.group(1)
                mid = re.search(rb"<id>(\d+)</id>", c)
                if not mid or int(mid.group(1)) not in wanted: continue
                t = re.search(rb"<text[^>]*>(.*?)</text>", c, re.S)
                if not t: continue
                try:
                    ent = json.loads(_html.unescape(t.group(1).decode("utf-8", "replace")))
                    eid = str(ent.get("id", ""))
                    if eid.startswith("Q"): eid = eid[1:]
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
            done_n += 1
            if done_n % 50 == 0:
                json.dump(entities, open(part_path, "w"))
                print(date, "checkpoint", len(entities), flush=True)
    json.dump(entities, open(out_path, "w"))
    if os.path.exists(part_path): os.remove(part_path)
    print(date, "entities DONE", len(entities), flush=True)

# ---------------- WikiFactDiff ----------------

def stage_wfd():
    chunked([WFD_URL], f"{WORK}/wfd_legacy.parquet", WFD_SIZE, tag="wfd")

if __name__ == "__main__":
    s = sys.argv[1]
    dates = sys.argv[2:] or None
    if s == "idx": stage_idx(dates or DATES)
    elif s == "scan":
        for d in (dates or DATES): stage_scan(d)
    elif s == "pages":
        for d in (dates or DATES): fetch_pages(d)
    elif s == "validate_base": stage_validate_base()
    elif s == "pageprops": stage_pageprops(dates[0] if dates else "20260401")
    elif s == "qidmap": stage_qidmap(dates[0] if dates else "20260401")
    elif s == "wdlisting":
        for d in (dates or WD_DATES): stage_wdlisting(d)
    elif s == "wdidx":
        for d in (dates or WD_DATES): stage_wdidx(d)
    elif s == "wdscan":
        for d in (dates or WD_DATES): stage_wdscan(d)
    elif s == "wdfetch":
        for d in (dates or WD_DATES): stage_wdfetch(d)
    elif s == "wfd": stage_wfd()

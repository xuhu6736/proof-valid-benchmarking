"""全量实验编排：重建 + ε(t) 扩展。阶段化执行，每阶段幂等可重入。"""
import bz2, glob, json, os, random, re, sys, time, urllib.request
from concurrent.futures import ThreadPoolExecutor

WORK = "/tmp/pvb_data"
os.makedirs(WORK, exist_ok=True)
ENWIKI = "https://ftp.acc.umu.se/mirror/wikimedia.org/dumps/enwiki"
WD = "https://ftp.acc.umu.se/mirror/wikimedia.org/dumps/wikidatawiki"
DATES = ["20260401", "20260501", "20260601", "20260701", "20260801"]
IDX_SIZE = {"20260401": 280349081, "20260501": 281216902, "20260601": 281979710,
            "20260701": 282804233, "20260801": 283588691}
MAIN_SIZE = {"20260401": 26207949994, "20260501": 26319713415, "20260601": 26437250146,
             "20260701": 26564488717, "20260801": 26668484995}
WD_DATES = ["20260401", "20260501", "20260701", "20260801"]  # 20260601 不存在

def chunked(url, out, total, chunk=16*1024*1024, tag=""):
    if os.path.exists(out) and os.path.getsize(out) == total:
        print(f"[{tag}] cached", flush=True); return True
    done = os.path.getsize(out) if os.path.exists(out) else 0
    if done % chunk:
        done = (done // chunk) * chunk
        with open(out, "r+b") as f: f.truncate(done)
    with open(out, "r+b" if os.path.exists(out) else "wb") as f:
        while done < total:
            end = min(done + chunk - 1, total - 1)
            for a in range(10):
                try:
                    r = urllib.request.urlopen(urllib.request.Request(
                        url, headers={"Range": f"bytes={done}-{end}"}), timeout=150)
                    data = r.read()
                    assert len(data) == end - done + 1, f"short {len(data)}"
                    f.seek(done); f.write(data); done = end + 1
                    break
                except Exception as e:
                    print(f"[{tag}] retry{a}@{done}: {repr(e)[:70]}", flush=True); time.sleep(2 + a*2)
            else:
                print(f"[{tag}] FAILED at {done}", flush=True); return False
    print(f"[{tag}] DONE", flush=True); return True

def stage_titles():
    import pandas as pd
    titles = set()
    for f in glob.glob('/mnt/agents/output/pvb_pilot/data/twiki_probes/*.csv'):
        titles.update(pd.read_csv(f)['subject'].tolist())
    random.seed(42)
    sample = random.sample(sorted(titles), 2000)
    open(f"{WORK}/target_titles.txt", "w").write("\n".join(sample))
    print("titles:", len(sample), flush=True)

def stage_download_enwiki_indexes():
    with ThreadPoolExecutor(max_workers=5) as ex:
        list(ex.map(lambda d: chunked(f"{ENWIKI}/{d}/enwiki-{d}-pages-articles-multistream-index.txt.bz2",
                                      f"{WORK}/idx_{d}.bz2", IDX_SIZE[d], tag=f"idx{d}"), DATES))

def stage_download_pageprops():
    chunked(f"{ENWIKI}/20260401/enwiki-20260401-page_props.sql.gz",
            f"{WORK}/page_props_20260401.sql.gz", 450210296, tag="pageprops")

def stage_download_wfd():
    chunked("https://hf-mirror.com/datasets/Orange/WikiFactDiff/resolve/main/20210104-20230227_legacy/train.parquet",
            f"{WORK}/wfd_legacy.parquet", 177696222, tag="wfd")

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

def stage_scan_all():
    with ThreadPoolExecutor(max_workers=5) as ex:
        list(ex.map(stage_scan, DATES))

def fetch_pages(date):
    scan = json.load(open(f'{WORK}/scan_{date}.json'))
    hits, all_offs = scan['hits'], scan['offsets']
    by_block = {}
    for title, (off, pid) in hits.items():
        by_block.setdefault(off, {})[pid] = title
    import bisect
    def bend(off):
        i = bisect.bisect_right(all_offs, off)
        return all_offs[i] if i < len(all_offs) else MAIN_SIZE[date]
    url = f"{ENWIKI}/{date}/enwiki-{date}-pages-articles-multistream.xml.bz2"
    def fetch(off):
        end = bend(off) - 1
        for a in range(4):
            try:
                r = urllib.request.urlopen(urllib.request.Request(url, headers={"Range": f"bytes={off}-{end}"}), timeout=60)
                data = r.read(); assert len(data) == end - off + 1
                return off, bz2.decompress(data)
            except Exception as e:
                time.sleep(2 + a*3)
        print("FAIL", date, off, flush=True); return off, None
    pages = {}
    with ThreadPoolExecutor(max_workers=4) as ex:
        for off, xml in ex.map(fetch, sorted(by_block)):
            if xml is None: continue
            for m in re.finditer(rb"<page>(.*?)</page>", xml, re.S):
                c = m.group(1)
                mid = re.search(rb"<id>(\d+)</id>", c)
                if mid and int(mid.group(1)) in by_block[off]:
                    t = re.search(rb"<text[^>]*>(.*?)</text>", c, re.S)
                    pages[by_block[off][int(mid.group(1))]] = t.group(1).decode("utf-8", "replace") if t else ""
    json.dump(pages, open(f"{WORK}/pages_{date}.json", "w"))
    print(date, "pages DONE", len(pages), flush=True)

def stage_pages_all():
    with ThreadPoolExecutor(max_workers=2) as ex:  # 两个日期并行，各自内部4线程
        list(ex.map(fetch_pages, DATES))

if __name__ == "__main__":
    s = sys.argv[1]
    {"titles": stage_titles,
     "dl_idx": stage_download_enwiki_indexes,
     "dl_pageprops": stage_download_pageprops,
     "dl_wfd": stage_download_wfd,
     "scan": stage_scan_all,
     "pages": stage_pages_all}[s]()

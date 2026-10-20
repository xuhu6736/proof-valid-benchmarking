#!/usr/bin/env python3
"""Step 2b: Range-fetch label entities for value qids (base & target).

Input : data/label_hits_{date}.json  {qid: [idx_fn, offset, pageid]}
Output: data/labels_{date}.json      {qid: "label"}
Blocks are fetched with HTTP Range from the multistream XML dump
(mirror first, official fallback), with checkpointing.
"""
import json, bz2, re, os, time, urllib.request, html as _html
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.normpath(os.path.join(HERE, '..', 'data'))
WORK = os.path.normpath(os.path.join(HERE, '..', '..', '..', '..', 'rebuild_work'))
UA = {"User-Agent": "PVSB-replication/1.0 (research; contact: author)"}
WD_MIRROR = "https://ftp.acc.umu.se/mirror/wikimedia.org/dumps/wikidatawiki"
WD_OFFICIAL = "https://dumps.wikimedia.org/wikidatawiki"
idx2xml = lambda fn: re.sub(r"multistream-index(\d+)\.txt-", r"multistream\1.xml-", fn)


def shard_offsets(date, idx_fn):
    """sorted list of all block offsets recorded in an index shard"""
    txt = bz2.decompress(open(f'{WORK}/wdidx2_{date}/{idx_fn}', 'rb').read()).decode('utf-8', 'replace')
    return sorted({int(l.split(':', 1)[0]) for l in txt.splitlines() if l.strip()})


def block_end(offs, off, size):
    i = bisect.bisect_right(offs, off)
    if i < len(offs):
        return offs[i] - 1
    return (size - 1) if size else off + 3 * 1024 * 1024


import bisect


def fetch_block(date, xml_fn, off, end, mirrors):
    for base in mirrors:
        url = f"{base}/{date}/{xml_fn}"
        for a in range(4):
            try:
                r = urllib.request.urlopen(urllib.request.Request(
                    url, headers={**UA, "Range": f"bytes={off}-{end}"}), timeout=90)
                raw = r.read()
                if len(raw) == end - off + 1:
                    return bz2.decompress(raw)
            except Exception:
                time.sleep(1.5 + a * 2)
    print('  FAIL', date, xml_fn, off, flush=True)
    return None


def run(date):
    hits = json.load(open(f'{DATA}/label_hits_{date}.json'))
    out_path = f'{DATA}/labels_{date}.json'
    part_path = out_path + '.partial'
    labels = json.load(open(part_path)) if os.path.exists(part_path) else {}
    shards_meta = json.load(open(f'{WORK}/wd_shards_{date}.json'))
    # group wanted pageids by (xml_fn, block offset)
    by_block = defaultdict(dict)
    for qid, (idx_fn, off, pid) in hits.items():
        if qid in labels:
            continue
        by_block[(idx_fn, off)][pid] = qid
    print(date, 'blocks to fetch:', len(by_block), 'labels cached:', len(labels), flush=True)
    offs_cache = {}
    mirrors = ([WD_MIRROR] if date >= '20260501' else []) + [WD_OFFICIAL]
    keys = sorted(by_block)
    done = 0

    def work(key):
        idx_fn, off = key
        if idx_fn not in offs_cache:
            offs_cache[idx_fn] = shard_offsets(date, idx_fn)
        size = shards_meta.get(idx2xml(idx_fn), {}).get('size')
        end = block_end(offs_cache[idx_fn], off, size)
        return key, fetch_block(date, idx2xml(idx_fn), off, end, mirrors)

    with ThreadPoolExecutor(max_workers=6) as ex:
        for key, xml in ex.map(work, keys):
            if xml is not None:
                wanted = by_block[key]
                for m in re.finditer(rb"<page>(.*?)</page>", xml, re.S):
                    c = m.group(1)
                    mid = re.search(rb"<id>(\d+)</id>", c)
                    if not mid or int(mid.group(1)) not in wanted:
                        continue
                    t = re.search(rb"<text[^>]*>(.*?)</text>", c, re.S)
                    if not t:
                        continue
                    try:
                        ent = json.loads(_html.unescape(t.group(1).decode('utf-8', 'replace')))
                        lab = (ent.get('labels', {}).get('en') or {}).get('value')
                        qid = wanted[int(mid.group(1))]
                        if lab:
                            labels[qid] = lab
                    except Exception:
                        pass
            done += 1
            if done % 40 == 0:
                json.dump(labels, open(part_path, 'w'), ensure_ascii=False)
                print(date, 'checkpoint', done, '/', len(keys), 'labels:', len(labels), flush=True)
    json.dump(labels, open(out_path, 'w'), ensure_ascii=False)
    if os.path.exists(part_path):
        os.remove(part_path)
    print(date, 'DONE labels:', len(labels), '/', len(hits), flush=True)


if __name__ == '__main__':
    run('20260401')
    run('20260801')

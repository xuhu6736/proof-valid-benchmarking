#!/usr/bin/env python3
"""Step 2: resolve EN labels for value qids at base and target periods.

1) scan the already-downloaded multistream INDEX shards (rebuild_work/wdidx2_*)
   for the needed qids -> (xml shard, byte offset, pageid)
2) report which index shards are still missing (qids not covered)
3) Range-fetch the containing bz2 blocks from the XML dump and extract labels.

Run mode 'scan' first; mode 'fetch' after missing index shards are present.
"""
import json, bz2, re, os, sys, bisect, time, urllib.request
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.normpath(os.path.join(HERE, '..', 'data'))
WORK = os.path.normpath(os.path.join(HERE, '..', '..', '..', '..', 'rebuild_work'))
UA = {"User-Agent": "PVSB-replication/1.0 (research; contact: author)"}
WD_MIRROR = "https://ftp.acc.umu.se/mirror/wikimedia.org/dumps/wikidatawiki"
WD_OFFICIAL = "https://dumps.wikimedia.org/wikidatawiki"

idx2xml = lambda fn: re.sub(r"multistream-index(\d+)\.txt-", r"multistream\1.xml-", fn)
xml2idx = lambda fn: re.sub(r"multistream(\d+)\.xml-", r"multistream-index\1.txt-", fn)


def scan_period(date, qids):
    """return {qid: (idx_fn, offset, pageid)} from downloaded index shards."""
    d = f'{WORK}/wdidx2_{date}'
    want = set(qids)
    hits = {}
    files = sorted(os.listdir(d))
    for fi, fn in enumerate(files):
        if not want:
            break
        try:
            txt = bz2.decompress(open(os.path.join(d, fn), 'rb').read()).decode('utf-8', 'replace')
        except Exception as e:
            print('  shard unreadable:', fn, e)
            continue
        for line in txt.splitlines():
            parts = line.split(':', 2)
            if len(parts) != 3:
                continue
            off, pid, title = parts
            if not title.startswith('Q'):
                continue
            q = title[1:]
            if q in want:
                hits[q] = (fn, int(off), int(pid))
                want.discard(q)
        if fi % 20 == 0:
            print(f'  {date} scanned {fi+1}/{len(files)} shards, hits={len(hits)}, remaining={len(want)}', flush=True)
    return hits, want


def main():
    qids = set(json.load(open(f'{DATA}/label_qids.json')))
    print('needed qids:', len(qids))
    all_hits = {}
    for date in ('20260401', '20260801'):
        print(f'--- scanning {date} index shards ---', flush=True)
        hits, missing = scan_period(date, qids)
        print(f'{date}: covered {len(hits)}/{len(qids)}, missing {len(missing)}')
        json.dump({q: list(v) for q, v in hits.items()},
                  open(f'{DATA}/label_hits_{date}.json', 'w'))
        json.dump(sorted(missing), open(f'{DATA}/label_missing_{date}.json', 'w'))
        all_hits[date] = hits


if __name__ == '__main__':
    main()

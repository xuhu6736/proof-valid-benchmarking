#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Supplementary E8 timing: gate-check cost on INTACT certificates (no
erasure, all leaves verified, no early exit) on the same Apple M3 Pro
platform and code path as remeasure_timing_m3pro.py.  Produces
remeasure_gate_intact_m3pro.json; does not modify existing artifacts."""
import hashlib, json, os, platform, subprocess, time
import duckdb
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
HP = os.path.join(HERE, '..', '..', 'data', 'hotpot_validation.parquet')
OUT_JSON = os.path.join(HERE, 'remeasure_gate_intact_m3pro.json')
SEED, K_MAIN = 42, 10


def build_records(titles, docs):
    index = set(titles)
    recs = []
    for t in titles:
        body = docs[t].encode()
        recs.append({'id': t, 'h': hashlib.sha256(body).hexdigest(),
                     'v': 1, 'I': t in index, 'pol': True, 'pi': True})
    return recs


def gate_check(recs, hashes):
    ns = time.perf_counter_ns()
    ok = True
    for r in recs:
        if not (r['I'] and r['pol'] and r['pi']):
            ok = False
            break
        if hashes[r['id']] != r['h']:
            ok = False
            break
    us = (time.perf_counter_ns() - ns) / 1e3
    return ok, us


def main():
    rng = np.random.default_rng(SEED)
    print('[intact] loading corpus ...')
    rows = duckdb.sql(
        "SELECT context['title'] AS titles, supporting_facts['title'] AS sup "
        "FROM read_parquet(?)", params=[HP]).fetchall()
    docs = {}
    per_q_titles, gold_titles = [], []
    for titles, sup in rows:
        titles = list(titles)
        per_q_titles.append(titles)
        gold_titles.append(set(sup))
    n_q = len(rows)
    # fetch doc bodies only for titles actually used (memory-light pass)
    rows2 = duckdb.sql(
        "SELECT context['title'] AS titles, context['sentences'] AS sents "
        "FROM read_parquet(?)", params=[HP]).fetchall()
    need = set()
    for titles, _ in rows2:
        need.update(list(dict.fromkeys(list(titles)))[:K_MAIN])
    for titles, sents in rows2:
        for t, ss in zip(titles, sents):
            if t in need and t not in docs:
                docs[t] = ' '.join(ss)
    print(f'[intact] {n_q} questions, {len(docs)} docs needed')
    hashes = {t: hashlib.sha256(d.encode()).hexdigest()
              for t, d in docs.items()}

    certs = []
    for i in range(n_q):
        rt = [t for t in dict.fromkeys(per_q_titles[i])][:K_MAIN]
        gt = [t for t in rt if t in gold_titles[i]]
        certs.append(build_records(gt, docs))
        certs.append(build_records(rt, docs))
    certs = [c for c in certs if len(c) > 0]
    print(f'[intact] {len(certs)} nonempty certificates')

    gate_us = []
    n_target = 20000
    while len(gate_us) < n_target:
        for r in certs:
            _, us = gate_check(r, hashes)
            gate_us.append(us)
            if len(gate_us) >= n_target:
                break
    gate_us = np.array(gate_us)

    brand = subprocess.run(['sysctl', '-n', 'machdep.cpu.brand_string'],
                           capture_output=True, text=True).stdout.strip()
    res = {
        'platform': {'chip': brand, 'arch': platform.machine(),
                     'python': platform.python_version()},
        'note': 'gate check on intact certificates: no erasure, every leaf '
                'verified (hash + flags), no early exit; same code path and '
                'platform as remeasure_timing_m3pro.json',
        'n_samples': int(len(gate_us)),
        'gate_us_intact': {'median': float(np.median(gate_us)),
                           'p95': float(np.percentile(gate_us, 95)),
                           'p99': float(np.percentile(gate_us, 99)),
                           'mean': float(np.mean(gate_us))},
    }
    json.dump(res, open(OUT_JSON, 'w'), indent=1)
    print(json.dumps(res['gate_us_intact'], indent=1))
    print('saved', OUT_JSON)


if __name__ == '__main__':
    main()

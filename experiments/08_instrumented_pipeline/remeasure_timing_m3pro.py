#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Re-measure E8 certificate construction / serving-gate latency on a
documented platform (Apple M3 Pro), replacing the original unlogged-platform
figures.  Faithfully replicates the code paths of run_pipeline_experiment.py
(build_records / gate_check, same per-question semantics
constr_us[i] = us_gold + us_fullcontext, gate sampled up to 20000 calls at
eps=0.1).  Retrieval top-10 is proxied by each question's supplied context
documents (same Wikipedia-intro byte distribution; the gold certificate is
the supporting subset), since the pipeline embedding cache is unavailable;
the timed operations (SHA-256 of full doc text, version tag, index probe,
five-factor gate) do not depend on which documents were retrieved.

Platform is recorded in the output JSON.  Managed runtime Python (no torch
needed).
"""
import hashlib, json, os, platform, subprocess, sys, time
import duckdb
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
HP = os.path.join(HERE, '..', '..', 'grounded_experiment_完整交付包_E1-E14',
                  'data', 'hotpot_validation.parquet')
OUT_JSON = os.path.join(HERE, 'remeasure_timing_m3pro.json')
OUT_NPZ = os.path.join(HERE, 'e8_timings_m3pro.npz')
SEED, N_REP, EPS_T = 42, 20, 0.1
K_MAIN = 10


def build_records(titles, docs):
    """Pi: build five-factor premise-leaf records; returns (records, total_us).
    Identical to run_pipeline_experiment.py."""
    index = set(titles)
    t0 = time.perf_counter_ns()
    recs = []
    for t in titles:
        body = docs[t].encode()
        recs.append({
            'id': t,
            'h': hashlib.sha256(body).hexdigest(),
            'v': 1,
            'I': t in index,
            'pol': True,
            'pi': True,
        })
    us = (time.perf_counter_ns() - t0) / 1e3
    return recs, us


def gate_check(recs, erased, hashes):
    """Proof-valid gate: all five factors must hold for every leaf.
    Identical to run_pipeline_experiment.py."""
    ns = time.perf_counter_ns()
    for j, r in enumerate(recs):
        if erased[j] or not (r['I'] and r['pol'] and r['pi']):
            ok = False
            break
        if hashes[r['id']] != r['h']:
            ok = False
            break
    else:
        ok = True
    us = (time.perf_counter_ns() - ns) / 1e3
    return ok, us


def main():
    rng = np.random.default_rng(SEED)
    print('[remeasure] loading corpus ...')
    rows = duckdb.sql(
        "SELECT context['title'] AS titles, context['sentences'] AS sents, "
        "supporting_facts['title'] AS sup FROM read_parquet(?)",
        params=[HP]).fetchall()
    docs = {}
    per_q_titles, gold_titles = [], []
    for titles, sents, sup in rows:
        titles = list(titles)
        per_q_titles.append(titles)
        gold_titles.append(set(sup))
        for t, ss in zip(titles, sents):
            if t not in docs:
                docs[t] = ' '.join(ss)
    n_q = len(rows)
    print(f'[remeasure] {n_q} questions, {len(docs)} unique docs')

    hashes = {t: hashlib.sha256(d.encode()).hexdigest()
              for t, d in docs.items()}

    gold_recs, full_recs = [], []
    constr_us = np.empty(n_q)
    cert_bytes = np.empty(n_q)
    for i in range(n_q):
        rt = [t for t in dict.fromkeys(per_q_titles[i])][:K_MAIN]
        gt = [t for t in rt if t in gold_titles[i]]
        recs_g, us_g = build_records(gt, docs)
        recs_f, us_f = build_records(rt, docs)
        gold_recs.append(recs_g)
        full_recs.append(recs_f)
        constr_us[i] = us_g + us_f
        cert_bytes[i] = 76 * (len(recs_g) + len(recs_f))

    gate_us_all = []
    for recs_all in (gold_recs, full_recs):
        idxs = [i for i, r in enumerate(recs_all) if len(r) > 0]
        for _ in range(N_REP):
            for i in idxs:
                r = recs_all[i]
                erased = rng.random(len(r)) < EPS_T
                _, us = gate_check(r, erased, hashes)
                if len(gate_us_all) < 20000:
                    gate_us_all.append(us)
            if len(gate_us_all) >= 20000:
                break
    gate_us = np.array(gate_us_all)

    def pct(a, q):
        return float(np.percentile(a, q))

    brand = subprocess.run(['sysctl', '-n', 'machdep.cpu.brand_string'],
                           capture_output=True, text=True).stdout.strip()
    mem = int(subprocess.run(['sysctl', '-n', 'hw.memsize'],
                             capture_output=True, text=True).stdout.strip())
    res = {
        'platform': {
            'chip': brand, 'arch': platform.machine(),
            'memory_GiB': round(mem / 2**30),
            'python': platform.python_version(),
            'note': 'timed operations exclude sentence embeddings '
                    '(batch-precomputed, memory-resident); policy and '
                    'presentation flags stubbed-allow',
        },
        'n_questions': int(n_q),
        'retrieval_proxy': 'question supplied context documents (<=10) stand in for '
                           'retrieved top-10 (same byte distribution); timed '
                           'operations are retrieval-independent',
        'construction_us': {'median': float(np.median(constr_us)),
                            'p95': pct(constr_us, 95),
                            'p99': pct(constr_us, 99),
                            'mean': float(np.mean(constr_us))},
        'gate_us': {'median': float(np.median(gate_us)),
                    'p95': pct(gate_us, 95),
                    'p99': pct(gate_us, 99),
                    'n_samples': int(len(gate_us))},
        'cert_metadata_bytes': {'median': float(np.median(cert_bytes)),
                                'p95': pct(cert_bytes, 95)},
    }
    json.dump(res, open(OUT_JSON, 'w'), indent=1)
    np.savez(OUT_NPZ, constr_us=constr_us, gate_us=gate_us)
    print(json.dumps(res, indent=1))
    print('[remeasure] saved', OUT_JSON, 'and', OUT_NPZ)


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Measure serving-time certificate-verification overhead and annotate the
generation-time construction cost of an instrumented-provenance cache on the
HotpotQA validation split (context = 10 full Wikipedia intro documents/question).

Outputs overhead_results.json plus a timing summary on stdout.
"""
import hashlib, json, os, time
import numpy as np
import pandas as pd

DATA = os.path.join(os.path.dirname(__file__), 'data', 'hotpot_validation.parquet')
OUT  = os.path.join(os.path.dirname(__file__), 'overhead_results.json')
N_TIMING_QUESTIONS = 50   # questions used for the stopwatch measurements

df = pd.read_parquet(DATA)
n = len(df)

tot_bytes, sup_bytes, tot_docs, tot_sents = [], [], [], []
contexts = []  # list of list-of-sentences for the first N_TIMING_QUESTIONS
for i, (_, r) in enumerate(df.iterrows()):
    ctx = r['context']; titles = list(ctx['title']); sents = ctx['sentences']
    sup_titles = set(r['supporting_facts']['title'].tolist())
    tb, sb, ns = 0, 0, 0
    q_sents = []
    for tname, slist in zip(titles, sents):
        sl = [str(s) for s in slist]
        b = sum(len(s.encode('utf-8')) for s in sl)
        tb += b; ns += len(sl)
        if tname in sup_titles:
            sb += b
        if i < N_TIMING_QUESTIONS:
            q_sents.append((tname in sup_titles, sl))
    tot_bytes.append(tb); sup_bytes.append(sb)
    tot_docs.append(len(titles)); tot_sents.append(ns)
    if i < N_TIMING_QUESTIONS:
        contexts.append(q_sents)

tot_bytes = np.array(tot_bytes); sup_bytes = np.array(sup_bytes)
tot_docs = np.array(tot_docs); tot_sents = np.array(tot_sents)

def stats(a, scale=1.0):
    return {k: float(v * scale) for k, v in zip(
        ['mean', 'p50', 'p95', 'p99'], [a.mean()] + list(np.percentile(a, [50, 95, 99])))}

# ---- stopwatch: SHA-256 verification + version-compare + index probe ----
REP = 30
def timed(fn, bytes_per_call, reps=REP):
    t0 = time.perf_counter()
    for _ in range(reps):
        fn()
    dt = (time.perf_counter() - t0) / reps
    return dt * 1e6, (bytes_per_call / dt) / 1e9   # us per call, GB/s

sup_sents_all = [s for qc in contexts for is_sup, sl in qc if is_sup for s in sl]
all_sents = [s for qc in contexts for _, sl in qc for s in sl]
sup_kb = sum(len(s.encode()) for s in sup_sents_all) / 1024
all_kb = sum(len(s.encode()) for s in all_sents) / 1024

def hash_sup():
    h = hashlib.sha256()
    for s in sup_sents_all: h.update(s.encode('utf-8'))
def hash_all():
    h = hashlib.sha256()
    for s in all_sents: h.update(s.encode('utf-8'))
def version_compare():   # lexicographic version-tag compare per sentence
    for s in all_sents: 'v3' >= s[:8]
def index_probe():       # id lookup + existence flag per sentence
    d = {hash(s): True for s in all_sents}
    for s in all_sents: d.get(hash(s), False)

t_hash_sup, thr_sup = timed(hash_sup, sup_kb * 1024)
t_hash_all, thr_all = timed(hash_all, all_kb * 1024)
t_ver, _  = timed(version_compare, all_kb * 1024)
t_idx, _  = timed(index_probe, all_kb * 1024)

n_q = N_TIMING_QUESTIONS
per_q = dict(
    sha256_support_us=t_hash_sup / n_q, sha256_context_us=t_hash_all / n_q,
    version_compare_us=t_ver / n_q, index_probe_us=t_idx / n_q)
per_q['certificate_gate_us'] = per_q['sha256_support_us'] + per_q['version_compare_us'] + per_q['index_probe_us']
per_q['full_audit_us'] = per_q['sha256_context_us'] + per_q['version_compare_us'] + per_q['index_probe_us']

res = {
    'n_questions': int(n),
    'docs_per_question_mean': float(tot_docs.mean()),
    'sentences_per_question_mean': float(tot_sents.mean()),
    'context_KB': stats(tot_bytes, 1 / 1024),
    'supporting_KB': stats(sup_bytes, 1 / 1024),
    'supporting_fraction_mean': float((sup_bytes / tot_bytes).mean()),
    'sha256_throughput_GBps': float(thr_sup),
    'serving_time_per_question_us': per_q,
    'metadata_bytes_per_leaf': {'id': 16, 'sha256': 32, 'version': 8, 'policy': 4, 'lineage': 16, 'total': 76},
}
json.dump(res, open(OUT, 'w'), indent=1)
print(json.dumps(res, indent=1))

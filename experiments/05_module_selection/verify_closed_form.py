#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Verify the decomposable closed form of the module-cost audit on both
corpora: OPT = |U| - sum_d max(k_d - rho, 0) per component, and check that
the density-greedy implementation coincides with it component-by-component.
Also recompute exact all-component totals (replacing the greedy-only
treatment of the 40 large MuSiQue components)."""
import json, os
import duckdb
import numpy as np
from collections import defaultdict
from itertools import combinations
import scipy.sparse as sp
import scipy.sparse.csgraph as cg

HERE = os.path.dirname(os.path.abspath(__file__))
RHO_LIST = [0.5, 1.0, 2.0, 4.0]
HP = os.path.join(HERE, '..', '..', 'grounded_experiment_完整交付包_E1-E14',
                  'data', 'hotpot_validation.parquet')
MQ = os.path.join(HERE, '..', '..', 'grounded_experiment_完整交付包_E1-E14',
                  'data', 'musique_full_dev.jsonl')


def components(q_docs):
    doc2q = defaultdict(set)
    for qi, ds in enumerate(q_docs):
        for d in ds:
            doc2q[d].add(qi)
    shared = {d: qs for d, qs in doc2q.items() if len(qs) >= 2}
    Q = len(q_docs)
    rows, cols = [], []
    for qs in shared.values():
        for a, b in combinations(sorted(qs), 2):
            rows.append(a); cols.append(b)
    A = sp.csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(Q, Q))
    ncomp, labels = cg.connected_components(A, directed=False)
    sizes = np.bincount(labels)
    comps = [[q for q in range(Q) if labels[q] == c]
             for c in range(ncomp) if sizes[c] >= 2]
    return shared, comps


def audit(q_docs, comps):
    """Closed-form totals and per-component greedy check."""
    out = {}
    max_k = 0
    for rho in RHO_LIST:
        L = O = G = 0
        n_mismatch = 0
        for qs in comps:
            qs_set = set(qs)
            cnt = defaultdict(int)
            for qi in qs:
                for d in q_docs[qi]:
                    cnt[d] += 1
            U = sum(cnt.values())
            cand = {d: k for d, k in cnt.items() if k >= 2}
            max_k_local = max(cand.values()) if cand else 0
            max_k = max(max_k, max_k_local)
            opt = U - sum(max(k - rho, 0) for k in cand.values())
            # density-greedy replication (gain = newly covered - rho)
            chosen, covered, cur = set(), 0, U
            while True:
                best_gain, best_d = 0, None
                for d, k in cand.items():
                    if d in chosen:
                        continue
                    gain = k - rho
                    if gain > best_gain:
                        best_gain, best_d = gain, d
                if best_d is None:
                    break
                chosen.add(best_d)
                covered += cand[best_d]
            greedy = rho * len(chosen) + (U - covered)
            if abs(greedy - opt) > 1e-9:
                n_mismatch += 1
            L += U; O += opt; G += greedy
        out[rho] = dict(leaf=L, closed_form=O, greedy=G,
                        saving=1 - O / L, mismatched_components=n_mismatch)
    return out, max_k


# HotpotQA: gold supporting documents
rows = duckdb.sql("SELECT supporting_facts['title'] AS g FROM read_parquet(?)",
                  params=[HP]).fetchall()
hp_qdocs = [set(r[0]) for r in rows]
shared_hp, comps_hp = components(hp_qdocs)
hp_res, hp_maxk = audit(hp_qdocs, comps_hp)

# MuSiQue: identical supporting paragraphs (stable text prefix as id)
import json as js
mq_qdocs = []
with open(MQ) as f:
    for line in f:
        q = js.loads(line)
        if not q.get('answerable', True):
            continue
        docs = set()
        for p in q['paragraphs']:
            if p.get('is_supporting'):
                docs.add(p['paragraph_text'][:200])
        if docs:
            mq_qdocs.append(docs)
shared_mq, comps_mq = components(mq_qdocs)
mq_res, mq_maxk = audit(mq_qdocs, comps_mq)

print('HotpotQA: components', len(comps_hp), 'questions',
      sum(len(c) for c in comps_hp), 'max k_d', hp_maxk)
for rho in RHO_LIST:
    r = hp_res[rho]
    print(f"  rho={rho}: leaf {r['leaf']} closed-form {r['closed_form']:.1f} "
          f"greedy {r['greedy']:.1f} saving {100*r['saving']:.1f}% "
          f"mismatch {r['mismatched_components']}")
print('MuSiQue: components', len(comps_mq), 'questions',
      sum(len(c) for c in comps_mq), 'max k_d', mq_maxk)
for rho in RHO_LIST:
    r = mq_res[rho]
    print(f"  rho={rho}: leaf {r['leaf']} closed-form {r['closed_form']:.1f} "
          f"greedy {r['greedy']:.1f} saving {100*r['saving']:.1f}% "
          f"mismatch {r['mismatched_components']}")

saved = {'hotpotqa': {'n_components': len(comps_hp),
                      'n_questions': sum(len(c) for c in comps_hp),
                      'max_k_d': hp_maxk,
                      'per_rho': {str(k): v for k, v in hp_res.items()}},
         'musique': {'n_components': len(comps_mq),
                     'n_questions': sum(len(c) for c in comps_mq),
                     'max_k_d': mq_maxk,
                     'per_rho': {str(k): v for k, v in mq_res.items()}}}
json.dump(saved, open(os.path.join(HERE, 'module_closed_form_verification.json'), 'w'), indent=1)
print('saved module_closed_form_verification.json')

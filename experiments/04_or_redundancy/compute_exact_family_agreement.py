#!/usr/bin/env python3
"""Exact-family agreement record for E11 (OR redundancy).

Rebuilds the strict and loose singleton-alternative families of
run_or_redundancy_strict_experiment.py from data/hotpot_validation.parquet
(no Monte-Carlo), evaluates the closed-form family availability

  R_q^family(eps) = 1 - [1 - (1-eps)^{e_q}] * eps^{a_q},   0 < eps < 1,

where e_q is the gold supporting-fact count and a_q the number of distinct
admitted non-gold singleton leaves (disjoint from the gold set by
construction), and compares the workload average against the retained
Monte-Carlo curves in e11_or_strict_results.json.

Also records: duplicate alternative-leaf incidence (expected 0), the
distinct-leaf definition, and the per-rate max absolute MC-minus-exact
difference cited in the text ("within 0.0012").

Run from the repository root:  python3 experiments/04_or_redundancy/compute_exact_family_agreement.py
Outputs: experiments/04_or_redundancy/e11_exact_family_agreement.json
"""
import json

import duckdb
import numpy as np

EPS_GRID = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5]
HP = 'data/hotpot_validation.parquet'
MC_JSON = 'experiments/04_or_redundancy/e11_or_strict_results.json'
OUT = 'experiments/04_or_redundancy/e11_exact_family_agreement.json'


def build_families():
    """Same construction as run_or_redundancy_strict_experiment.py.

    Uses duckdb because the managed runtime's pandas has no parquet engine;
    the per-row logic is identical to the pandas original.
    """
    rows = duckdb.connect().execute(
        "SELECT answer, supporting_facts.title, supporting_facts.sent_id, "
        "context.title, context.sentences "
        "FROM 'data/hotpot_validation.parquet'").fetchall()
    loose_a, strict_a, e_qs, dup_q = [], [], [], 0
    for answer, sf_titles, sf_sids, ctx_titles, ctx_sents in rows:
        a = str(answer).strip().lower()
        gold = sorted(set(zip(sf_titles, sf_sids)))
        gold_docs = set(t for t, _ in gold)
        alt_loose, alt_strict = [], []
        if a:
            for t, sl in zip(ctx_titles, ctx_sents):
                for si, s in enumerate(sl):
                    if a in str(s).lower():
                        leaf = (t, si)
                        if leaf not in gold:
                            alt_loose.append(leaf)
                            if t in gold_docs:
                                alt_strict.append(leaf)
        if len(set(alt_loose)) != len(alt_loose):
            dup_q += 1
        loose_a.append(len(alt_loose))
        strict_a.append(len(alt_strict))
        e_qs.append(len(gold))
    return np.array(loose_a), np.array(strict_a), np.array(e_qs), dup_q


def exact_family(e_q, a_q, eps):
    return 1.0 - (1.0 - (1.0 - eps) ** e_q) * eps ** a_q


def main():
    loose_a, strict_a, e_q, dup_q = build_families()
    mc = json.load(open(MC_JSON))
    assert mc['eps'] == EPS_GRID
    out = {
        'source': ('closed-form evaluation on the E11 strict/loose families '
                   'rebuilt from data/hotpot_validation.parquet; Monte-Carlo '
                   'curves from e11_or_strict_results.json (seed 42, 20 reps)'),
        'distinct_alternative_leaf': ('a (document title, sentence index) '
                                      'pair containing the answer string and '
                                      'not in the gold supporting-fact set'),
        'questions_with_duplicate_alternative_leaves': dup_q,
        'eps': EPS_GRID,
        'families': {},
    }
    for name, a_q in [('strict', strict_a), ('loose', loose_a)]:
        exact = [float(np.mean(exact_family(e_q, a_q, eps))) for eps in EPS_GRID]
        meas = mc[name]['meas']
        diff = [float(m - x) for m, x in zip(meas, exact)]
        out['families'][name] = {
            'exact_family_average': [round(v, 6) for v in exact],
            'mc_average': [round(v, 6) for v in meas],
            'mc_minus_exact': [round(v, 6) for v in diff],
            'max_abs_difference': round(max(abs(v) for v in diff), 6),
        }
        print(f"[{name}] max |MC-exact| = "
              f"{out['families'][name]['max_abs_difference']}")
    print('questions with duplicate alternative leaves:', dup_q)
    json.dump(out, open(OUT, 'w'), indent=1)
    print('saved', OUT)


if __name__ == '__main__':
    main()

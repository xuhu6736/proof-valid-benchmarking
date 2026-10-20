#!/usr/bin/env python3
"""Step 6 (package 1a): unified cache retrieval audit.

The main replay (s5) checks each paraphrase against its OWN cached question
(paired candidate).  Here every query is matched against ALL 154 cached
questions (unified retrieval): accept iff the top-1 match clears tau.

Metrics: genuine-sim vs best-impostor-sim margins, top-1 identity
accuracy, accepted/misrouted/missed counts at tau = 0.8364 (the
in-sample threshold from s5) and across a tau sweep.

Deterministic: uses cached data/embeddings.json only.

Output: data/unified_retrieval.json
"""
import json, os
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.normpath(os.path.join(HERE, '..', 'data'))
TAU = 0.8364  # display-rounded; s5 acceptance used >= with no ties

def cos(a, b):
    a = np.asarray(a); b = np.asarray(b)
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))

def main():
    panel = json.load(open(f'{DATA}/evidence.json'))
    gen = json.load(open(f'{DATA}/generation.json'))
    emb = json.load(open(f'{DATA}/embeddings.json'))
    usable = [x for x in panel if gen[str(x['pid'])]['answer_ok']
              and gen[str(x['pid'])]['paraphrase']]
    pids = [x['pid'] for x in usable]
    n = len(pids)
    assert n == 154, n

    Q = {p: np.asarray(emb[f'{p}:q']) for p in pids}   # cached questions
    P = {p: np.asarray(emb[f'{p}:p']) for p in pids}   # paraphrase queries

    rows = []
    for i in pids:
        sims = [(cos(P[i], Q[j]), j) for j in pids]
        sims.sort(reverse=True)
        genuine = cos(P[i], Q[i])
        impostor_sims = [s for s, j in sims if j != i]
        best_imp, best_imp_j = max(((s, j) for s, j in sims if j != i))
        rank = 1 + sum(1 for s, j in sims if s > genuine and j != i)
        rows.append({
            'pid': i, 'genuine': genuine, 'rank': rank,
            'top1_pid': sims[0][1], 'top1_sim': sims[0][0],
            'best_impostor': best_imp, 'best_impostor_pid': best_imp_j,
        })

    def decisions(tau):
        acc = mis = miss = 0
        for r in rows:
            if r['top1_sim'] >= tau:
                if r['top1_pid'] == r['pid']:
                    acc += 1
                else:
                    mis += 1
            else:
                miss += 1
        return acc, mis, miss

    acc, mis, miss = decisions(TAU)
    sweep = {f'{t:.2f}': decisions(t) for t in
             [0.50, 0.60, 0.70, 0.75, 0.80, 0.8364, 0.85, 0.90]}
    imp_over_tau = sum(1 for r in rows if r['best_impostor'] >= TAU)
    margins = [r['genuine'] - r['best_impostor'] for r in rows]

    out = {
        'n': n, 'tau': TAU,
        'paired_hits_at_tau': sum(1 for r in rows if r['genuine'] >= TAU),
        'unified_at_tau': {'accepted_correct': acc, 'misrouted': mis,
                           'missed': miss},
        'top1_identity_correct': sum(1 for r in rows if r['top1_pid'] == r['pid']),
        'impostor_ge_tau': imp_over_tau,
        'genuine_minus_impostor_margin': {
            'min': min(margins),
            'p5': float(np.percentile(margins, 5)),
            'median': float(np.median(margins)),
        },
        'sweep': {k: {'accepted_correct': v[0], 'misrouted': v[1],
                      'missed': v[2]} for k, v in sweep.items()},
        'rows': rows,
    }
    json.dump(out, open(f'{DATA}/unified_retrieval.json', 'w'), indent=1)
    print(f'n={n} tau={TAU}')
    print(f'paired hits:        {out["paired_hits_at_tau"]}')
    print(f'unified top1 correct identity: {out["top1_identity_correct"]}/154')
    print(f'unified @tau: accepted={acc} misrouted={mis} missed={miss}')
    print(f'impostor >= tau: {imp_over_tau}')
    print(f'margin min/p5/median: {min(margins):.4f} / '
          f'{out["genuine_minus_impostor_margin"]["p5"]:.4f} / '
          f'{out["genuine_minus_impostor_margin"]["median"]:.4f}')

if __name__ == '__main__':
    main()

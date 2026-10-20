#!/usr/bin/env python3
"""Step 7 (package 1b): out-of-sample threshold calibration.

The in-sample tau (5th percentile of the 154 genuine paraphrase
similarities) is replaced by entity-grouped 5-fold cross-validation:
for each fold, tau is calibrated on entries whose entities are outside
the fold, then applied to the held-out entries.  Entities are assigned
to folds by seeded shuffle, so all entries of one entity stay together.

Reports per-fold tau, paired hit rate, and unified-retrieval outcome on
held-out entries.

Output: data/tau_oos.json
"""
import json, os, random
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.normpath(os.path.join(HERE, '..', 'data'))

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
    qid = {x['pid']: x['qid'] for x in usable}

    entities = sorted(set(qid.values()))
    rng = random.Random(42)
    rng.shuffle(entities)
    K = 5
    folds = [set(entities[i::K]) for i in range(K)]

    Q = {p: np.asarray(emb[f'{p}:q']) for p in pids}
    P = {p: np.asarray(emb[f'{p}:p']) for p in pids}
    genuine = {p: cos(P[p], Q[p]) for p in pids}

    fold_rows = []
    all_test = []
    for k in range(K):
        test = [p for p in pids if qid[p] in folds[k]]
        train = [p for p in pids if qid[p] not in folds[k]]
        tau_k = float(np.percentile([genuine[p] for p in train], 5))
        hits = sum(1 for p in test if genuine[p] >= tau_k)
        # unified top-1 against the full cache
        uni_ok = uni_mis = uni_miss = 0
        for p in test:
            sims = sorted(((cos(P[p], Q[j]), j) for j in pids), reverse=True)
            if sims[0][0] >= tau_k:
                if sims[0][1] == p:
                    uni_ok += 1
                else:
                    uni_mis += 1
            else:
                uni_miss += 1
        fold_rows.append({'fold': k, 'tau': tau_k, 'n_test': len(test),
                          'paired_hits': hits,
                          'paired_hit_rate': hits / len(test),
                          'unified_correct': uni_ok, 'unified_misrouted': uni_mis,
                          'unified_missed': uni_miss})
        all_test += test

    taus = [f['tau'] for f in fold_rows]
    rates = [f['paired_hits'] / f['n_test'] for f in fold_rows]
    tot_ok = sum(f['unified_correct'] for f in fold_rows)
    tot_mis = sum(f['unified_misrouted'] for f in fold_rows)
    tot_miss = sum(f['unified_missed'] for f in fold_rows)
    out = {
        'scheme': 'entity-grouped 5-fold CV, seed 42, tau = 5th percentile of genuine train sims',
        'folds': fold_rows,
        'tau_min': min(taus), 'tau_max': max(taus),
        'oos_paired_hit_rate_mean': float(np.mean(rates)),
        'oos_paired_hit_rate_min': min(rates),
        'oos_paired_hits_total': sum(f['paired_hits'] for f in fold_rows),
        'oos_n_total': len(all_test),
        'oos_unified': {'correct': tot_ok, 'misrouted': tot_mis,
                        'missed': tot_miss},
    }
    json.dump(out, open(f'{DATA}/tau_oos.json', 'w'), indent=1)
    print(f'tau range across folds: {min(taus):.4f}..{max(taus):.4f}')
    print(f'OOS paired hits: {out["oos_paired_hits_total"]}/{len(all_test)} '
          f'= {out["oos_paired_hit_rate_mean"]*100:.1f}% (min fold {min(rates)*100:.1f}%)')
    print(f'OOS unified: correct {tot_ok}, misrouted {tot_mis}, missed {tot_miss}')

if __name__ == '__main__':
    main()

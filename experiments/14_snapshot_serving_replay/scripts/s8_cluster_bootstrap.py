#!/usr/bin/env python3
"""Step 8 (package 1c): entity-cluster bootstrap confidence intervals.

The replay entries share entities (154 entries over 143 entities), so
nominal Wilson intervals understate dependence.  Here we resample
ENTITIES with replacement (B = 10000, seed 42) and recompute the
headline rates over the 147 similarity hits:

  dependency level : stale-served share of similarity-only;
                     intact-decline rate of the strict gate
  answer level     : decayed-served share of similarity-only
                     (human labels, wrong + change-induced partial);
                     correct-decline rates of the three gated contracts

Indeterminate answer labels are excluded from answer-level denominators,
as in the paper.

Output: data/cluster_bootstrap.json
"""
import json, os, random
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.normpath(os.path.join(HERE, '..', 'data'))
B = 10000
DRIFT_PIDS = {8, 52, 95}   # incoming-query meaning drift (s18 labels)

def pct_ci(samples, lo=2.5, hi=97.5):
    s = sorted(samples)
    n = len(s)
    return s[int(lo / 100 * n)], s[min(n - 1, int(hi / 100 * n))]

def main():
    panel = json.load(open(f'{DATA}/evidence.json'))
    lab = json.load(open(os.path.normpath(
        os.path.join(DATA, '..', 'annotation', 'answer_labels.json'))))
    lab = {int(k): v for k, v in lab.items()}
    qid = {x['pid']: x['qid'] for x in panel}

    hits = [p for p, v in lab.items() if v['hit']]
    assert len(hits) == 147, len(hits)
    by_ent = defaultdict(list)
    for p in hits:
        by_ent[qid[p]].append(p)
    ents = sorted(by_ent)

    rng = random.Random(42)

    def rates(sample_ents):
        ps = [p for e in sample_ents for p in by_ent[e]]
        n = len(ps)
        out = {}
        # dependency level
        dep_stale = [p for p in ps if lab[p]['dep_stale']]
        out['dep_stale_share_simonly'] = len(dep_stale) / n
        intact = [p for p in ps if not lab[p]['dep_stale']]
        out['dep_intact_declined_strict'] = (
            sum(1 for p in intact if not lab[p]['strict']) / len(intact)
            if intact else None)
        # answer level
        dec = [p for p in ps if lab[p]['lab'] in ('wrong', 'partial')]
        out['ans_decayed_share_simonly'] = len(dec) / n
        drift = [p for p in ps if p in DRIFT_PIDS]
        out['ans_drifted_share_simonly'] = len(drift) / n
        out['ans_wrong_incoming_share_simonly'] = (len(dec) + len(drift)) / n
        corr = [p for p in ps if lab[p]['A4'] == '正确'
                and lab[p]['A5'] not in ('使其失效', '部分影响')
                and p not in DRIFT_PIDS]
        for g in ('strict', 'majority', 'claims'):
            out[f'ans_correct_declined_{g}'] = (
                sum(1 for p in corr if not lab[p][g]) / len(corr)
                if corr else None)
        return out

    keys = rates(ents).keys()
    boots = {k: [] for k in keys}
    for _ in range(B):
        sample = [ents[rng.randrange(len(ents))] for _ in range(len(ents))]
        r = rates(sample)
        for k in keys:
            if r[k] is not None:
                boots[k].append(r[k])

    point = rates(ents)
    out = {'B': B, 'cluster': 'entity (qid)', 'n_entities': len(ents),
           'n_hits': len(hits), 'point': point, 'ci95': {}}
    for k in keys:
        lo, hi = pct_ci(boots[k])
        out['ci95'][k] = [lo, hi]
        print(f'{k:32s} point={point[k]*100:5.1f}%  '
              f'cluster-boot95=[{lo*100:.1f}, {hi*100:.1f}]')
    json.dump(out, open(f'{DATA}/cluster_bootstrap.json', 'w'), indent=1)

if __name__ == '__main__':
    main()

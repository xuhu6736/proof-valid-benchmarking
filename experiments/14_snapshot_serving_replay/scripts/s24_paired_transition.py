#!/usr/bin/env python3
"""Step 24: paired per-request transition and paired-difference bootstrap.

Reviewer item: judge the closed-loop gain by PAIRED transitions, not by
comparing two marginal rates.  For every request we record

  cached answer category  ->  strict-policy final outcome
  (correct / incomplete / wrong / indet / miss) -> (same categories via
  cache or regen, or abstain)

and the paired indicator difference between the strict strategy and
similarity-only:
  Delta_i = 1{strict delivers correct}_i - 1{similarity delivers correct}_i
reported as Delta Y = mean(Delta_i) with an entity-cluster bootstrap CI
(B = 10,000, seed 42).  All policies share the same archived fallback
realization per request; regeneration prompts use the cached question,
which equals the incoming query's intent for the 151 faithful
paraphrases (the 3 drifted queries abstain).

Outputs: data/paired_transition.json + printed tables.
"""
import json, os, random
from collections import Counter, defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.normpath(os.path.join(HERE, '..', 'data'))
ANN = os.path.normpath(os.path.join(HERE, '..', 'annotation'))
REB = os.path.normpath(os.path.join(HERE, '..', '..', '12_real_dynamics_audit',
                                   'data', 'rebuilt_2026'))
DRIFT = {8, 52, 95}
B = 10000


def val_key(v):
    if isinstance(v, dict):
        if 'id' in v: return 'Q:' + str(v['id'])
        if 'amount' in v: return 'A:' + str(v['amount'])
        if 'time' in v: return 'T:' + str(v['time'])[:11]
        if 'text' in v: return 'S:' + str(v['text'])
    return 'S:' + str(v)


def truthy_keys(ent, prop):
    return {val_key(val) for r, s, val in ent['claims'].get(prop, [])
            if r in ('normal', 'preferred') and s == 'value'
            and val is not None}


def main():
    lab = {int(k): v for k, v in
           json.load(open(os.path.join(ANN, 'answer_labels.json'))).items()}
    rg = {int(k): v for k, v in
          json.load(open(f'{DATA}/regeneration.json')).items()}
    rl = {int(k): v for k, v in
          json.load(open(f'{DATA}/regen_labels.json')).items()}
    asserted = json.load(open(f'{DATA}/asserted_claims.json'))['asserted']
    panel = json.load(open(f'{DATA}/evidence.json'))
    entT = json.load(open(f'{REB}/entities_v2_20260801.json'))
    prop_of = {x['pid']: x['prop'] for x in panel}
    qid_of = {x['pid']: x['qid'] for x in panel}

    def asserted_pass(pid):
        ent = entT.get(qid_of[pid])
        tk = truthy_keys(ent, prop_of[pid]) if ent is not None else set()
        return all(k in tk for k in asserted.get(str(pid), []))

    def quality(pid):
        v = lab[pid]
        if pid in DRIFT or v['A4'] == '错误':
            return 'wrong'
        return {'正确': 'correct', '不完整': 'incomplete'}.get(v['A4'],
                                                              'indet')

    def regen_quality(pid):
        v = rl.get(pid)
        if v is None:
            return None
        return {'正确': 'correct', '不完整': 'incomplete',
                '错误': 'wrong'}.get(v['R1_adjudicated'], 'indet')

    def outcome(pid, serve_fn):
        if serve_fn(pid):
            return quality(pid)
        e = rg.get(pid, {})
        if e.get('regen_ok'):
            return regen_quality(pid)
        return 'abstain'

    policies = {
        'similarity-only': lambda p: lab[p]['hit'],
        'strict':   lambda p: lab[p]['hit'] and lab[p]['strict'],
        'majority': lambda p: lab[p]['hit'] and lab[p]['majority'],
        'claims':   lambda p: lab[p]['hit'] and lab[p]['claims'],
        'asserted': lambda p: lab[p]['hit'] and asserted_pass(p),
    }

    all_pids = sorted(lab)

    # ---- 1. transition table for the strict policy ----------------------
    def cached_cat(pid):
        return quality(pid) if lab[pid]['hit'] else 'miss'

    trans = Counter()
    for p in all_pids:
        trans[(cached_cat(p), outcome(p, policies['strict']))] += 1
    print('cached category -> strict-policy outcome:')
    for (a, b), n in sorted(trans.items()):
        print(f'  {a:11s} -> {b:11s} : {n}')

    # ---- 2. paired delta per policy vs similarity-only ------------------
    by_ent = defaultdict(list)
    for p in all_pids:
        by_ent[qid_of[p]].append(p)
    ents = sorted(by_ent)
    rng = random.Random(42)

    def delta_mean(sample_ents, serve_fn):
        ds = []
        for e in sample_ents:
            for p in by_ent[e]:
                y_pol = outcome(p, serve_fn) == 'correct'
                y_sim = outcome(p, policies['similarity-only']) == 'correct'
                ds.append(int(y_pol) - int(y_sim))
        return sum(ds) / len(ds) if ds else None

    out = {'transition_strict': {f'{a}->{b}': n for (a, b), n in
                                 sorted(trans.items())},
           'paired_vs_similarity': {}}
    for name in ('strict', 'majority', 'claims', 'asserted'):
        fn = policies[name]
        point = delta_mean(ents, fn)
        boots = []
        for _ in range(B):
            sample = [ents[rng.randrange(len(ents))]
                      for _ in range(len(ents))]
            v = delta_mean(sample, fn)
            if v is not None:
                boots.append(v)
        boots.sort()
        lo, hi = boots[int(0.025 * B)], boots[min(B - 1, int(0.975 * B))]
        n_up = sum(1 for p in all_pids
                   if (outcome(p, fn) == 'correct')
                   > (outcome(p, policies['similarity-only']) == 'correct'))
        n_dn = sum(1 for p in all_pids
                   if (outcome(p, fn) == 'correct')
                   < (outcome(p, policies['similarity-only']) == 'correct'))
        out['paired_vs_similarity'][name] = {
            'delta_correct_per_request': round(point, 4),
            'n_up': n_up, 'n_down': n_dn,
            'cluster_boot95': [round(lo, 4), round(hi, 4)]}
        print(f'{name:10s} DeltaY={point*100:+.2f} pts  '
              f'up={n_up} down={n_dn}  '
              f'cluster-boot95=[{lo*100:+.2f}, {hi*100:+.2f}]')

    json.dump(out, open(f'{DATA}/paired_transition.json', 'w'), indent=1)
    print('wrote data/paired_transition.json')


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""Step 22 (revised): unified strategy table, mutually exclusive quality
categories with decay/drift as a separate ATTRIBUTION axis.

Reviewer correction: under a quality axis that keeps wrong and incomplete
separate, wrong deliveries are 3 decayed-wrong + 3 drift-wrong = 6, not
11; the 5 partially invalidated answers are incomplete.  Each row sums to
154.  Decay/drift counts are reported as attribution tags, not mixed into
the quality count.

Quality categories per delivered answer:
  correct / incomplete / wrong / indeterminate;  abstain = no delivery.
  wrong    := A4=wrong (3) OR incoming-query drift (3)
  incomplete := A4=incomplete (12 non-decayed + 5 partially invalidated)
Attribution tags (separate axis): decayed (8 hits), drifted (3 hits).

Non-serve sets are nested (asserted <= claims <= majority <= strict) and
all policies SHARE THE SAME archived fallback realization per request, so
the cached strict closed loop covers every policy's fallback.

Outputs: data/unified_strategy_table.json + printed table.
"""
import json, os
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.normpath(os.path.join(HERE, '..', 'data'))
ANN = os.path.normpath(os.path.join(HERE, '..', 'annotation'))
REB = os.path.normpath(os.path.join(HERE, '..', '..', '12_real_dynamics_audit',
                                   'data', 'rebuilt_2026'))
DRIFT = {8, 52, 95}


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
        """Mutually exclusive delivery quality under incoming-query truth."""
        v = lab[pid]
        if pid in DRIFT or v['A4'] == '错误':
            return 'wrong'
        if v['A4'] == '正确':
            return 'correct'
        if v['A4'] == '不完整':
            return 'incomplete'
        return 'indet'

    def attribution(pid):
        tags = []
        if lab[pid]['A5'] in ('使其失效', '部分影响'):
            tags.append('decayed')
        if pid in DRIFT:
            tags.append('drifted')
        return tags

    def regen_quality(pid):
        v = rl.get(pid)
        if v is None:
            return None
        return {'正确': 'correct', '不完整': 'incomplete',
                '错误': 'wrong'}.get(v['R1_adjudicated'], 'indet')

    policies = {
        'similarity-only': lambda p: lab[p]['hit'],
        'strict':   lambda p: lab[p]['hit'] and lab[p]['strict'],
        'majority': lambda p: lab[p]['hit'] and lab[p]['majority'],
        'claims':   lambda p: lab[p]['hit'] and lab[p]['claims'],
        'asserted': lambda p: lab[p]['hit'] and asserted_pass(p),
    }

    all_pids = sorted(lab)
    out = {}
    for name, serve in policies.items():
        served = [p for p in all_pids if serve(p)]
        fallback = [p for p in all_pids if not serve(p)]
        regen = [p for p in fallback if rg.get(p, {}).get('regen_ok')]
        abstain = [p for p in fallback
                   if not rg.get(p, {}).get('regen_ok')]
        qc = Counter(quality(p) for p in served)
        rc = Counter(regen_quality(p) for p in regen)
        row = {
            'correct': qc['correct'] + rc['correct'],
            'incomplete': qc['incomplete'] + rc['incomplete'],
            'wrong': qc['wrong'] + rc['wrong'],
            'indet': qc['indet'] + rc['indet'],
            'abstain': len(abstain),
        }
        assert sum(row.values()) == 154, (name, row)
        # attribution axis (delivery-relevant tags)
        row['attr'] = {
            'decayed_served': sum(1 for p in served
                                  if 'decayed' in attribution(p)),
            'decayed_avoided': sum(1 for p in fallback
                                   if 'decayed' in attribution(p)),
            'drifted_served': sum(1 for p in served
                                  if 'drifted' in attribution(p)),
            'drifted_avoided': sum(1 for p in fallback
                                   if 'drifted' in attribution(p)),
            'served': len(served), 'fallback': len(fallback),
            'regenerated': len(regen),
        }
        out[name] = row

    print(f"{'policy':16s}{'correct':>8s}{'incompl':>8s}{'wrong':>6s}"
          f"{'indet':>6s}{'abstain':>8s}{'| dec srv/avoid':>16s}"
          f"{'drf srv/avoid':>15s}")
    for name, r in out.items():
        a = r['attr']
        print(f"{name:16s}{r['correct']:8d}{r['incomplete']:8d}"
              f"{r['wrong']:6d}{r['indet']:6d}{r['abstain']:8d}"
              f"{a['decayed_served']:9d}/{a['decayed_avoided']:<6d}"
              f"{a['drifted_served']:8d}/{a['drifted_avoided']:<6d}")

    json.dump({'n_requests': 154,
               'category_definitions': {
                   'wrong': 'A4=wrong (3) or incoming-query drift (3)',
                   'incomplete': 'A4=incomplete (12 non-decayed + 5 '
                                 'partially invalidated)',
                   'note': 'decayed/drifted are attribution tags on a '
                           'separate axis, not quality categories'},
               'policies': out},
              open(f'{DATA}/unified_strategy_table.json', 'w'), indent=1)
    print('\nwrote data/unified_strategy_table.json')


if __name__ == '__main__':
    main()

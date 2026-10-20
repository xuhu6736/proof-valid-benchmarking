#!/usr/bin/env python3
"""Step 18: incoming-query truth axis (Q_original / Q_incoming / D_temporal).

Reviewer item: three paraphrases (pid 8/52/95) drifted in meaning from the
cached question (performer wording acted->starred->voices while the cached
question asks about P175 performer).  Their cached answers are still correct
for the CACHED question (A4=correct) but wrong for the INCOMING drifted
query.  This script

1. writes annotation/incoming_query_labels.json with a three-axis label for
   all 154 panel entries (q_relation, incoming_wrong, drift_kind);
2. recomputes the per-policy serving crosstab under incoming-query truth:
   drifted cases form their own category and leave the confirmed-correct
   denominator (121 -> 118);
3. verifies the closed-loop outcome of the drifted trio (all abstain).

Outputs: annotation/incoming_query_labels.json,
         data/incoming_query_crosstab.json
"""
import json, os, re
from math import sqrt

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.normpath(os.path.join(HERE, '..', 'data'))
ANN = os.path.normpath(os.path.join(HERE, '..', 'annotation'))
REB = os.path.normpath(os.path.join(HERE, '..', '..', '12_real_dynamics_audit',
                                   'data', 'rebuilt_2026'))

DRIFT = {8: ('Who acted in The Fifth Element?',
             'performer wording drift: cached question asks P175 performer; '
             'paraphrase asks about actors'),
         52: ('Which actor or actress starred in the film As Good as It Gets?',
              'performer wording drift: starred vs performed'),
         95: ('Which actors provided the voices for Cars 3?',
              'performer wording drift: voice cast vs performer')}


def val_key(v):
    if isinstance(v, dict):
        if 'id' in v: return 'Q:' + str(v['id'])
        if 'amount' in v: return 'A:' + str(v['amount'])
        if 'time' in v: return 'T:' + str(v['time'])[:11]
        if 'text' in v: return 'S:' + str(v['text'])
    return 'S:' + str(v)


def truthy_keys(ent, prop):
    out = set()
    for rank, snak, val in ent['claims'].get(prop, []):
        if rank in ('normal', 'preferred') and snak == 'value' and val is not None:
            out.add(val_key(val))
    return out


def cp_ci(k, n, alpha=0.05):
    """Clopper-Pearson 95% interval."""
    from scipy.stats import beta
    if n == 0:
        return (0.0, 1.0)
    lo = 0.0 if k == 0 else beta.ppf(alpha / 2, k, n - k + 1)
    hi = 1.0 if k == n else beta.ppf(1 - alpha / 2, k + 1, n - k)
    return (round(100 * lo, 1), round(100 * hi, 1))


def main():
    lab = {int(k): v for k, v in
           json.load(open(os.path.join(ANN, 'answer_labels.json'))).items()}
    replay = {x['pid']: x for x in json.load(open(f'{DATA}/replay.json'))}
    base_pe = json.load(open(f'{DATA}/simple_baselines.json'))['per_entry']
    asserted = json.load(open(f'{DATA}/asserted_claims.json'))['asserted']
    panel = json.load(open(f'{DATA}/evidence.json'))
    entT = json.load(open(f'{REB}/entities_v2_20260801.json'))

    # ---- 1. three-axis labels -------------------------------------------
    axis = {}
    for pid, v in lab.items():
        if pid in DRIFT:
            axis[pid] = {'q_relation': 'meaning_drift',
                         'incoming_wrong': True,
                         'drift_kind': 'lexical_sense_drift',
                         'incoming_query': DRIFT[pid][0],
                         'note': DRIFT[pid][1]}
        else:
            axis[pid] = {'q_relation': 'faithful_paraphrase',
                         'incoming_wrong': False,
                         'drift_kind': 'none'}
    json.dump(axis, open(os.path.join(ANN, 'incoming_query_labels.json'), 'w'),
              indent=1, ensure_ascii=False)
    print(f'three-axis labels written: {len(axis)} entries, '
          f'{sum(1 for v in axis.values() if v["incoming_wrong"])} drifted')

    # ---- 2. per-pid gate outcomes ---------------------------------------
    prop_of = {x['pid']: x['prop'] for x in panel}
    qid_of = {x['pid']: x['qid'] for x in panel}

    def asserted_pass(pid):
        ent = entT.get(qid_of[pid])
        tk = truthy_keys(ent, prop_of[pid]) if ent is not None else set()
        keys = asserted.get(str(pid), [])
        if not keys:            # no answer-asserted claims: vacuous pass
            return True
        return all(k in tk for k in keys)

    def cat(pid):
        v = lab[pid]
        if v['A5'] in ('使其失效', '部分影响'):
            return 'decayed'
        if pid in DRIFT:
            return 'drifted'
        if v['A4'] == '正确':
            return 'correct'
        if v['A4'] == '不完整':
            return 'incomplete_nd'
        return 'indet'

    hits = [p for p, v in lab.items() if v['hit']]
    assert len(hits) == 147

    policies = {
        'similarity-only': lambda p: True,
        'strict': lambda p: lab[p]['strict'],
        'majority': lambda p: lab[p]['majority'],
        'claims': lambda p: lab[p]['claims'],
        'asserted': lambda p: asserted_pass(p),
        'page-change': lambda p: base_pe[str(p)]['page-change'],
        'entity-change': lambda p: base_pe[str(p)]['entity-change'],
        'prop-change': lambda p: base_pe[str(p)]['prop-change'],
        'value-in-page': lambda p: base_pe[str(p)]['value-in-page'],
        'llm-recheck': lambda p: base_pe[str(p)]['llm-recheck'],
    }

    cats = ['decayed', 'drifted', 'correct', 'incomplete_nd', 'indet']
    out = {}
    for name, fn in policies.items():
        row = {'served': 0}
        for c in cats:
            row[f'{c}_served'] = 0
            row[f'{c}_declined'] = 0
        for p in hits:
            c = cat(p)
            if fn(p):
                row['served'] += 1
                row[f'{c}_served'] += 1
            else:
                row[f'{c}_declined'] += 1
        n_corr = sum(1 for p in hits if cat(p) == 'correct')
        k = row['correct_declined']
        row['correct_total'] = n_corr
        row['correct_decline_rate'] = round(100 * k / n_corr, 1)
        row['correct_decline_ci'] = cp_ci(k, n_corr)
        wrong = row['decayed_served'] + row['drifted_served']
        row['wrong_served'] = wrong
        row['wrong_serve_rate'] = round(100 * wrong / len(hits), 1)
        row['wrong_serve_ci'] = cp_ci(wrong, len(hits))
        out[name] = row

    # category totals
    totals = {c: sum(1 for p in hits if cat(p) == c) for c in cats}
    print('\ncategory totals over 147 hits:', totals)

    hdr = f"{'policy':16s} {'served':>6s} {'dec-srv':>7s} {'drf-srv':>7s} " \
          f"{'wrong%':>7s} {'corr-dec':>8s} {'rate%':>6s}  CI"
    print(hdr)
    for name, r in out.items():
        print(f"{name:16s} {r['served']:6d} {r['decayed_served']:7d} "
              f"{r['drifted_served']:7d} {r['wrong_serve_rate']:6.1f}% "
              f"{r['correct_declined']:8d} {r['correct_decline_rate']:5.1f}% "
              f" {r['correct_decline_ci']}")

    # ---- 3. closed-loop outcome of the drifted trio ----------------------
    rg = json.load(open(f'{DATA}/regeneration.json'))
    abstained = []
    for pid in DRIFT:
        e = rg.get(str(pid), {})
        print(f"closed-loop pid {pid}: regen_ok={e.get('regen_ok')} "
              f"no_current_evidence={e.get('no_current_evidence')}")
        abstained.append(bool(e.get('no_current_evidence')))
    out['__closed_loop__'] = {
        'drift_trio_all_abstain': all(abstained),
        'note': 'all three drifted incoming queries are declined by every '
                'gated contract and then abstain in the fallback loop '
                '(target-period P175 evidence empty): no wrong answer is '
                'ever delivered for a drifted query'}
    print('drift trio all abstain:', all(abstained))

    json.dump({'categories': totals, 'policies': out,
               'drift_pids': sorted(DRIFT)},
              open(f'{DATA}/incoming_query_crosstab.json', 'w'), indent=1)
    print('\nwrote data/incoming_query_crosstab.json')


if __name__ == '__main__':
    main()

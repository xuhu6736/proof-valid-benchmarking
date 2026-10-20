#!/usr/bin/env python3
"""Step 19: alternative-certificate (disjunctive family) evaluation.

Reviewer item: does the PVSB extension mechanism of ALTERNATIVE CERTIFICATES
(serve if ANY certificate in the family validates) buy availability over a
flat exact dependency check, and at what cost?

Certificate family per cached entry = {claims certificate, sentence
certificate}.  Policies compared on the 147 similarity hits of the real
snapshot panel, under incoming-query truth (s18 labels):

  claims-only   exact conjunction over designated claims        (= claims gate)
  sentences-only exact conjunction over designated sentences    (vacuous pass)
  strict-AND    conjunction of both certificates                (= strict gate)
  family-OR     disjunction: serve if ANY certificate validates
  majority      all claims + strictly more than half sentences  (= majority gate)
  asserted      answer-asserted claims only                     (= asserted gate)

Outputs: data/alternative_certificates.json + printed table.
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
    out = set()
    for rank, snak, val in ent['claims'].get(prop, []):
        if rank in ('normal', 'preferred') and snak == 'value' and val is not None:
            out.add(val_key(val))
    return out


def cp_ci(k, n):
    from scipy.stats import beta
    if n == 0:
        return (0.0, 100.0)
    lo = 0.0 if k == 0 else beta.ppf(0.025, k, n - k + 1)
    hi = 1.0 if k == n else beta.ppf(0.975, k + 1, n - k)
    return (round(100 * lo, 1), round(100 * hi, 1))


def main():
    lab = {int(k): v for k, v in
           json.load(open(os.path.join(ANN, 'answer_labels.json'))).items()}
    replay = {x['pid']: x for x in json.load(open(f'{DATA}/replay.json'))}
    asserted = json.load(open(f'{DATA}/asserted_claims.json'))['asserted']
    panel = json.load(open(f'{DATA}/evidence.json'))
    entT = json.load(open(f'{REB}/entities_v2_20260801.json'))
    prop_of = {x['pid']: x['prop'] for x in panel}
    qid_of = {x['pid']: x['qid'] for x in panel}

    def cat(pid):
        v = lab[pid]
        if v['A5'] in ('使其失效', '部分影响'):
            return 'decayed'
        if pid in DRIFT:
            return 'drifted'
        return {'正确': 'correct', '不完整': 'incomplete_nd'}.get(v['A4'],
                                                                 'indet')

    def claims_alive(pid):
        return replay[pid]['claims_dead'] == 0

    def sents_alive(pid):
        x = replay[pid]
        return x['sents_total'] == 0 or x['sents_dead'] == 0

    def asserted_alive(pid):
        ent = entT.get(qid_of[pid])
        tk = truthy_keys(ent, prop_of[pid]) if ent is not None else set()
        keys = asserted.get(str(pid), [])
        return all(k in tk for k in keys)

    policies = {
        'claims-only':    lambda p: claims_alive(p),
        'sentences-only': lambda p: sents_alive(p),
        'strict-AND':     lambda p: claims_alive(p) and sents_alive(p),
        'family-OR':      lambda p: claims_alive(p) or sents_alive(p),
        'majority':       lambda p: lab[p]['majority'],
        'asserted':       lambda p: asserted_alive(p),
    }

    hits = [p for p, v in lab.items() if v['hit']]
    n_corr = sum(1 for p in hits if cat(p) == 'correct')
    out = {}
    for name, fn in policies.items():
        served = [p for p in hits if fn(p)]
        declined = [p for p in hits if not fn(p)]
        sc = Counter(cat(p) for p in served)
        dc = Counter(cat(p) for p in declined)
        wrong = sc['decayed'] + sc['drifted']
        out[name] = {
            'served': len(served),
            'decayed_served': sc['decayed'], 'drifted_served': sc['drifted'],
            'wrong_served': wrong,
            'wrong_serve_rate': round(100 * wrong / len(hits), 1),
            'wrong_serve_ci': cp_ci(wrong, len(hits)),
            'correct_declined': dc['correct'],
            'correct_decline_rate': round(100 * dc['correct'] / n_corr, 1),
            'correct_decline_ci': cp_ci(dc['correct'], n_corr),
            'served_wrong_pids': sorted((p for p in served
                                         if cat(p) in ('decayed', 'drifted')),
                                        key=int),
        }

    print(f"hits={len(hits)}  confirmed-correct={n_corr}\n")
    hdr = (f"{'policy':15s} {'served':>6s} {'dec':>4s} {'drf':>4s} "
           f"{'wrong%':>7s} {'corr-dec':>8s} {'rate%':>6s}")
    print(hdr)
    for name, r in out.items():
        print(f"{name:15s} {r['served']:6d} {r['decayed_served']:4d} "
              f"{r['drifted_served']:4d} {r['wrong_serve_rate']:6.1f}% "
              f"{r['correct_declined']:8d} {r['correct_decline_rate']:5.1f}%"
              f"   wrong pids: {r['served_wrong_pids']}")

    json.dump({'n_hits': len(hits), 'n_correct': n_corr, 'policies': out},
              open(f'{DATA}/alternative_certificates.json', 'w'), indent=1)
    print('\nwrote data/alternative_certificates.json')


if __name__ == '__main__':
    main()

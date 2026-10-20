#!/usr/bin/env python3
"""Step 28: QUALIFIED alternative certificates -- upgrading the negative
control (s19) into a positive mechanism validation.

The naive disjunction {claims certificate, sentence certificate} accepts a
sentence branch on string presence alone; s19 showed this launders the dead
claims branch's signal (8 decayed-or-drifted serves, 6 of them wrong).

The qualified sentence branch, stated a priori in contract terms:
an alternative branch must establish support for the CURRENT request-answer
pair from evidence whose currency is independently anchored.

  anchored(sentence)  := the sentence survives on the current page AND
                         mentions at least one currently-truthy value of the
                         property
  covered(value v)    := v is truthy in the current claim state OR v is
                         mentioned in at least one anchored surviving
                         certificate sentence
  qualified branch    := at least one certificate sentence exists AND all
                         certificate sentences survive AND every
                         answer-asserted value is covered

  * no vacuous pass: an empty sentence certificate cannot certify (fixes
    the naive rule's vacuous acceptance of sentence-free entries)
  * a surviving sentence that mentions only dead values is stale page lag
    and cannot anchor support (fixes the laundering)
  * claim-sentence agreement is required only through the anchor: a value
    the claim store dropped but the current text still carries alongside
    live values remains servable (the availability the family exists for)

Qualified family-OR := claims certificate OR qualified sentence certificate.

Validations:
  V1  naive rules reproduce the s19 archived numbers exactly
  V2  the qualified branch rejects every one of the 8 decayed-or-drifted
      serves the naive family accepted
  V3  five-class accounting over all 154 requests sums to 154

Outputs: data/qualified_alternatives.json + console summary.
"""
import json, os, re
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.normpath(os.path.join(HERE, '..', 'data'))
ANN = os.path.normpath(os.path.join(HERE, '..', 'annotation'))
REB = os.path.normpath(os.path.join(HERE, '..', '..', '12_real_dynamics_audit',
                                   'data', 'rebuilt_2026'))
DRIFT = {8, 52, 95}


def norm(s):
    return re.sub(r'[,\s]+', '', str(s).lower())


def val_key(v):
    if isinstance(v, dict):
        if 'id' in v: return 'Q:' + str(v['id'])
        if 'amount' in v: return 'A:' + str(v['amount'])
        if 'time' in v: return 'T:' + str(v['time'])[:11]
        if 'text' in v: return 'S:' + str(v['text'])
    return 'S:' + str(v)


def truthy_items(ent, prop):
    out = []
    for rank, snak, val in ent['claims'].get(prop, []):
        if rank in ('normal', 'preferred') and snak == 'value' and val is not None:
            out.append((val_key(val), val))
    return out


def mentions(text, vtext):
    nt, nv = norm(text), norm(vtext)
    if nv and nv in nt:
        return True
    m = re.match(r'(\d{4})', str(vtext))
    return bool(m and m.group(1) in str(text))


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
    by_pid = {int(x['pid']): x for x in panel}
    entT = json.load(open(f'{REB}/entities_v2_20260801.json'))
    qid2title = {int(v): k for k, v in
                 json.load(open(f'{REB}/title_qid_map.json')).items()}

    def value_text(v):
        if isinstance(v, dict):
            if 'id' in v:
                vid = v['id']
                if isinstance(vid, str) and vid.startswith('Q'):
                    vid = vid[1:]
                try:
                    return qid2title.get(int(vid), 'Q' + str(vid))
                except (TypeError, ValueError):
                    return str(v['id'])
            if 'amount' in v:
                return str(v['amount']).lstrip('+')
            if 'time' in v:
                return str(v['time'])[1:11]
            if 'text' in v:
                return str(v['text'])
        return str(v)

    def cat(pid):
        v = lab[pid]
        if v['A5'] in ('使其失效', '部分影响'):
            return 'decayed'
        if pid in DRIFT:
            return 'drifted'
        return {'正确': 'correct', '不完整': 'incomplete_nd'}.get(v['A4'],
                                                                 'indet')

    hits = sorted((p for p, v in lab.items() if v['hit']), key=int)

    # ---- per-entry evidence ----------------------------------------------
    info = {}
    for pid in hits:
        x = by_pid[pid]
        ent = entT.get(x['qid'])
        t_items = truthy_items(ent, x['prop']) if ent is not None else []
        t_keys = {k for k, _ in t_items}
        t_texts = [value_text(v) for _, v in t_items]
        # certificate claims with their display texts (aligned values_text)
        claim_key2text = {val_key(c['value']): vt
                          for c, vt in zip(x['certificate']['claims'],
                                           x['values_text'])}
        akeys = asserted.get(str(pid), [])
        sents = [(s['id'], s['text']) for s in x['certificate']['sentences']]
        r = replay[pid]
        sents_survive = (r['sents_total'] == 0 or r['sents_dead'] == 0)
        claims_alive = r['claims_dead'] == 0
        anchored = [sid for sid, txt in sents
                    if any(mentions(txt, tt) for tt in t_texts)]

        def covered(v):
            if v in t_keys:
                return True
            vt = claim_key2text.get(v, v)
            return any(sid in anchored and mentions(txt, vt)
                       for sid, txt in sents)

        qualified = (len(sents) > 0 and sents_survive
                     and len(anchored) > 0
                     and all(covered(v) for v in akeys))
        info[pid] = {
            'claims_alive': claims_alive,
            'asserted_alive': all(k in t_keys for k in akeys),
            'sents_alive_naive': sents_survive,      # incl. vacuous pass
            'n_sents': len(sents),
            'n_anchored': len(anchored),
            'asserted_dead': [k for k in akeys if k not in t_keys],
            'qualified_sents': qualified,
        }

    policies = {
        'claims-only':     lambda p: info[p]['claims_alive'],
        'sentences-naive': lambda p: info[p]['sents_alive_naive'],
        'family-OR-naive': lambda p: info[p]['claims_alive']
                                     or info[p]['sents_alive_naive'],
        'sents-qualified': lambda p: info[p]['qualified_sents'],
        'family-OR-qualified': lambda p: info[p]['claims_alive']
                                         or info[p]['qualified_sents'],
        'asserted-naive-sents': lambda p: info[p]['asserted_alive']
                                          or info[p]['sents_alive_naive'],
        'family-qualified-asserted': lambda p: info[p]['asserted_alive']
                                               or info[p]['qualified_sents'],
    }

    n_corr = sum(1 for p in hits if cat(p) == 'correct')
    out = {}
    for name, fn in policies.items():
        served = [p for p in hits if fn(p)]
        declined = [p for p in hits if not fn(p)]
        sc, dc = Counter(cat(p) for p in served), Counter(cat(p)
                                                          for p in declined)
        wrong = sc['decayed'] + sc['drifted']
        out[name] = {
            'served': len(served),
            'decayed_served': sc['decayed'], 'drifted_served': sc['drifted'],
            'wrong_served': wrong,
            'wrong_serve_rate': round(100 * wrong / len(hits), 1),
            'wrong_serve_ci': cp_ci(wrong, len(hits)),
            'correct_declined': dc['correct'],
            'incomplete_declined': dc['incomplete_nd'],
            'indet_declined': dc['indet'],
            'correct_decline_rate': round(100 * dc['correct'] / n_corr, 1),
            'correct_decline_ci': cp_ci(dc['correct'], n_corr),
            'served_wrong_pids': sorted((p for p in served
                                         if cat(p) in ('decayed', 'drifted')),
                                        key=int),
            'served_pids': served,
        }

    # ---- V1: naive reproduction of the s19 archive ------------------------
    s19 = json.load(open(f'{DATA}/alternative_certificates.json'))[
        'policies']
    for ours, theirs in (('sentences-naive', 'sentences-only'),
                         ('family-OR-naive', 'family-OR'),
                         ('claims-only', 'claims-only')):
        a, b = out[ours], s19[theirs]
        assert (a['served'], a['wrong_served'], a['correct_declined']) == \
               (b['served'], b['wrong_served'], b['correct_declined']), \
            f'V1 FAILED: {ours} {a} vs s19 {b}'
    print('V1 OK: naive rules reproduce the s19 archived numbers exactly.')

    # ---- V2: every laundered serve rejected by the qualified branch -------
    bad = s19['family-OR']['served_wrong_pids']
    still = [p for p in bad if out['family-OR-qualified']['served']
             and p in out['family-OR-qualified']['served_pids']]
    assert not still, f'V2 FAILED: {still} still served'
    print(f'V2 OK: all 8 laundered serves rejected {bad}.')

    # ---- five-class accounting over all 154 requests ----------------------
    regen = json.load(open(f'{DATA}/regeneration.json'))
    regen_lab = {int(k): v for k, v in
                 json.load(open(f'{DATA}/regen_labels.json')).items()}
    misses = sorted(set(lab) - set(hits))

    def five_class(policy):
        served_set = set(out[policy]['served_pids'])
        five = Counter()
        for pid in hits:
            if pid in served_set:
                five[{'correct': 'correct', 'incomplete_nd': 'incomplete',
                      'decayed': 'wrong', 'drifted': 'wrong',
                      'indet': 'indet'}[cat(pid)]] += 1
        for pid in [p for p in hits if p not in served_set] + misses:
            r = regen.get(str(pid))
            if r is None or not r.get('regen_ok') or pid in DRIFT:
                five['abstain'] += 1
            else:
                five['correct' if regen_lab[pid]['R1_adjudicated'] == '正确'
                     else 'incomplete'] += 1
        assert sum(five.values()) == 154, f'V3 FAILED: {policy}: {five}'
        return dict(five)

    five_classes = {p: five_class(p)
                    for p in ('family-OR-qualified',
                              'family-qualified-asserted')}
    print(f'V3 OK: five classes over 154 requests: {five_classes}')

    print(f"\n{'policy':20s} {'served':>6s} {'dec':>4s} {'drf':>4s} "
          f"{'wrong%':>7s} {'corr-dec':>8s} {'rate%':>6s}")
    for name, r in out.items():
        print(f"{name:20s} {r['served']:6d} {r['decayed_served']:4d} "
              f"{r['drifted_served']:4d} {r['wrong_serve_rate']:6.1f}% "
              f"{r['correct_declined']:8d} {r['correct_decline_rate']:5.1f}%"
              f"   wrong pids: {r['served_wrong_pids']}")

    # branch audit: what does the qualified sentence branch accept/reject?
    acc = [p for p in hits if info[p]['qualified_sents']]
    rej_naive = [p for p in hits if info[p]['sents_alive_naive']
                 and not info[p]['qualified_sents']]
    print(f'\nqualified sentence branch accepts {len(acc)} entries:',
          Counter(cat(p) for p in acc))
    print(f'naive-pass but qualified-reject: {len(rej_naive)} entries:',
          Counter(cat(p) for p in rej_naive))
    gained = [p for p in out['family-OR-qualified']['served_pids']
              if not info[p]['claims_alive']]
    print(f'\nfamily-OR-qualified serves beyond claims-only: {len(gained)}',
          sorted(gained, key=int))
    print('  their labels:', Counter(cat(p) for p in gained))
    gained_a = [p for p in out['family-qualified-asserted']['served_pids']
                if not info[p]['asserted_alive']]
    print(f'family-qualified-asserted serves beyond asserted-only: '
          f'{len(gained_a)}', sorted(gained_a, key=int))
    print('  their labels:', Counter(cat(p) for p in gained_a))
    naive_set = set(out['family-OR-naive']['served_pids'])
    qual_set = set(out['family-qualified-asserted']['served_pids'])
    naive_extra = sorted(naive_set - qual_set, key=int)
    qual_extra = sorted(qual_set - naive_set, key=int)
    print(f'\nset relation naive-vs-qualified: |N|={len(naive_set)} '
          f'|Q|={len(qual_set)} |N and Q|={len(naive_set & qual_set)} '
          f'|N\\Q|={len(naive_extra)} |Q\\N|={len(qual_extra)}')
    print('  N\\Q labels:', Counter(cat(p) for p in naive_extra))
    print('  Q\\N labels:', Counter(cat(p) for p in qual_extra))

    json.dump({
        'rule': {
            'anchored': 'sentence survives AND mentions >=1 currently-truthy '
                        'value of the property',
            'covered': 'value truthy OR mentioned in >=1 anchored surviving '
                       'certificate sentence',
            'qualified_branch': 'n_sents>0 AND all survive AND >=1 anchored '
                                'AND every answer-asserted value covered'},
        'policies': {k: dict(v) for k, v in out.items()},
        'five_class_over_154': five_classes,
        'branch_audit': {
            'accepts': len(acc),
            'accepts_by_label': dict(Counter(cat(p) for p in acc)),
            'naive_pass_qualified_reject': len(rej_naive),
            'reject_by_label': dict(Counter(cat(p) for p in rej_naive)),
            'served_beyond_claims': sorted(gained, key=int),
            'gained_by_label': dict(Counter(cat(p) for p in gained)),
            'served_beyond_asserted': sorted(gained_a, key=int),
            'gained_asserted_by_label': dict(Counter(cat(p)
                                                   for p in gained_a)),
            'naive_beyond_qualified': naive_extra,
            'naive_extra_by_label': dict(Counter(cat(p)
                                               for p in naive_extra)),
            'qualified_beyond_naive': qual_extra,
            'qualified_extra_by_label': dict(Counter(cat(p)
                                                   for p in qual_extra)),
            'intersection_size': len(naive_set & qual_set)},
        'validations': {'V1_naive_reproduces_s19': True,
                        'V2_all_8_laundered_rejected': True,
                        'V3_five_class_sums_154': True}},
        open(f'{DATA}/qualified_alternatives.json', 'w'),
        ensure_ascii=False, indent=1)
    print('\nwrote data/qualified_alternatives.json')


if __name__ == '__main__':
    main()

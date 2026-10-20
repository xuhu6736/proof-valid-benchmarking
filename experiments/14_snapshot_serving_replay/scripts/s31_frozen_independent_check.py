#!/usr/bin/env python3
"""
s31_frozen_independent_check.py -- P4: frozen-rule independent check
(reviewer acceptance criterion 4).

Frozen artifacts (nothing re-tuned in this script):
  * the four dependency contracts (strict / majority / claims / asserted)
  * the anchored lexical acceptance rule, exactly as archived in
    qualified_alternatives.json (frozen before its first evaluation run;
    zero revisions since)
  * the panel's base certificates (t0 = 2026-04)

Leg 1 -- entity holdout at the 2026-08 endpoint (human labels).
  Design-trail entities are the entities of the 12 entries that drove the
  rule-design analysis: the 8 laundered serves (8/24/48/50/52/95/112/125)
  and the 4 entries the anchored rule recovers beyond the asserted gate.
  All other entities form the holdout.  The frozen rules are evaluated on
  each subset with the human answer labels.  The holdout still contains
  decayed-or-drifted entries the design trail never singled out, so the
  interception requirement remains non-vacuous.
  Honesty note: the rule's WORDING was informed by panel-wide failure
  analysis, so the holdout is not pristine for wording; it is pristine
  for evaluation (no holdout label was consulted in any revision -- the
  rule has not changed since it was frozen).

Leg 2 -- forward check at 2026-10-16 (post-design live refetch).
  The nine decayed (entity, property) pairs plus the decayed non-hit were
  refetched from live Wikidata on 2026-10-16 (data/current_claims/, used
  by the revert negative control).  The frozen claims-level contracts are
  re-evaluated at that forward date.  The anchored rule's sentence
  component cannot be evaluated forward (no 2026-10 page snapshot was
  rebuilt; network egress was unavailable in this run) -- disclosed.

Leg 3 -- per-entry review of the four anchored-rule admissions.
  For each entry the rule serves beyond the asserted gate, the dossier
  extracts the question, cached answer, answer-asserted values, current
  truthy values, the anchoring sentence(s), and which asserted values are
  claim-covered vs sentence-covered, with a per-entry support statement.

Leg 4 -- frozen-rule challenge audit (the disclosed lexical loophole).
  The disclosed vulnerability: a surviving sentence mentioning a stale
  asserted value alongside a live one can "cover" the stale value.  The
  audit enumerates every served entry with at least one
  sentence-covered (non-truthy) asserted value, checks the co-occurrence
  shape per entry, and reads the human labels: did the loophole
  materialize into a wrong serve?

Outputs: data/frozen_independent_check.json + console summary.
"""

import json, os, re, importlib.util
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.normpath(os.path.join(HERE, '..', 'data'))
ANN = os.path.normpath(os.path.join(HERE, '..', 'annotation'))
REB = os.path.normpath(os.path.join(HERE, '..', '..', '12_real_dynamics_audit',
                                   'data', 'rebuilt_2026'))

spec = importlib.util.spec_from_file_location(
    's28', os.path.join(HERE, 's28_qualified_alternatives.py'))
s28 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(s28)          # has __main__ guard; helpers only
norm, val_key, truthy_items, mentions = (s28.norm, s28.val_key,
                                         s28.truthy_items, s28.mentions)
cp_ci = s28.cp_ci
DRIFT = s28.DRIFT

T = '20260801'
LAUNDERED = {8, 24, 48, 50, 52, 95, 112, 125}

# ------------------------------------------------------------------ data ----
lab = {int(k): v for k, v in
       json.load(open(os.path.join(ANN, 'answer_labels.json'))).items()}
replay = {x['pid']: x for x in json.load(open(f'{DATA}/replay.json'))}
asserted = json.load(open(f'{DATA}/asserted_claims.json'))['asserted']
panel = json.load(open(f'{DATA}/evidence.json'))
gen = json.load(open(f'{DATA}/generation.json'))
by_pid = {int(x['pid']): x for x in panel}
entT = json.load(open(f'{REB}/entities_v2_20260801.json'))
qid2title = {int(v): k for k, v in
             json.load(open(f'{REB}/title_qid_map.json')).items()}

def value_text(v):
    if isinstance(v, dict):
        if 'id' in v:
            vid = str(v['id']).lstrip('Q')
            try:
                return qid2title.get(int(vid), 'Q' + str(vid))
            except (TypeError, ValueError):
                return str(v['id'])
        if 'amount' in v: return str(v['amount']).lstrip('+')
        if 'time' in v: return str(v['time'])[1:11]
        if 'text' in v: return str(v['text'])
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

# --------------------------------- frozen rule evaluation (as s28, verbatim)
info = {}
for pid in hits:
    x = by_pid[pid]
    ent = entT.get(x['qid'])
    t_items = truthy_items(ent, x['prop']) if ent is not None else []
    t_keys = {k for k, _ in t_items}
    t_texts = [value_text(v) for _, v in t_items]
    claim_key2text = {val_key(c['value']): vt
                      for c, vt in zip(x['certificate']['claims'],
                                       x['values_text'])}
    akeys = asserted.get(str(pid), [])
    sents = [(s['id'], s['text']) for s in x['certificate']['sentences']]
    r = replay[pid]
    sents_survive = (r['sents_total'] == 0 or r['sents_dead'] == 0)
    anchored = [sid for sid, txt in sents
                if any(mentions(txt, tt) for tt in t_texts)]

    def covered(v, _t_keys=t_keys, _c2t=claim_key2text, _anch=anchored,
                _sents=sents):
        if v in _t_keys:
            return True
        vt = _c2t.get(v, v)
        return any(sid in _anch and mentions(txt, vt)
                   for sid, txt in _sents)

    info[pid] = {
        'claims_alive': r['claims_dead'] == 0,
        'asserted_alive': all(k in t_keys for k in akeys),
        'sents_alive_naive': sents_survive,
        'qualified_sents': (len(sents) > 0 and sents_survive
                            and len(anchored) > 0
                            and all(covered(v) for v in akeys)),
        'sentence_covered_asserted': [
            k for k in akeys if k not in t_keys and covered(k)],
        'uncovered_asserted': [k for k in akeys if not covered(k)],
    }

policies = {
    'strict':    lambda p: info[p]['claims_alive'] and info[p]['sents_alive_naive'],
    'claims':    lambda p: info[p]['claims_alive'],
    'asserted':  lambda p: info[p]['asserted_alive'],
    'anchored family (frozen)': lambda p: info[p]['asserted_alive']
                                          or info[p]['qualified_sents'],
    'naive family (control)':   lambda p: info[p]['claims_alive']
                                          or info[p]['sents_alive_naive'],
}

# sanity: reproduce the archived anchored-family counts before splitting
qa = json.load(open(f'{DATA}/qualified_alternatives.json'))
arch = qa['policies']['family-qualified-asserted']
got = [p for p in hits if policies['anchored family (frozen)'](p)]
assert len(got) == arch['served'], 'frozen rule does not reproduce archive'
RECOVERED = sorted(set(qa['branch_audit']['served_beyond_asserted']),
                   key=int)
print(f'frozen anchored family reproduces the archive: {arch["served"]} '
      f'served; admissions beyond the asserted gate: {RECOVERED}')

# ------------------------------------------------- Leg 1: entity holdout ----
design_pids = set(LAUNDERED) | set(RECOVERED)
design_qids = {by_pid[p]['qid'] for p in design_pids}
subsets = {
    'design-trail': [p for p in hits if by_pid[p]['qid'] in design_qids],
    'holdout': [p for p in hits if by_pid[p]['qid'] not in design_qids],
}
print(f"design-trail entities: {len(design_qids)} "
      f"({len(subsets['design-trail'])} hits); holdout: "
      f"{len(subsets['holdout'])} hits over "
      f"{len({by_pid[p]['qid'] for p in subsets['holdout']})} entities")

leg1 = {}
for sname, pids in subsets.items():
    n_corr = sum(1 for p in pids if cat(p) == 'correct')
    n_dec_drf = sum(1 for p in pids if cat(p) in ('decayed', 'drifted'))
    leg1[sname] = {'hits': len(pids), 'confirmed_correct': n_corr,
                   'decayed_or_drifted': n_dec_drf, 'policies': {}}
    for name, fn in policies.items():
        served = [p for p in pids if fn(p)]
        sc = Counter(cat(p) for p in served)
        dc = Counter(cat(p) for p in pids if not fn(p))
        wrong = sc['decayed'] + sc['drifted']
        leg1[sname]['policies'][name] = {
            'served': len(served), 'wrong_served': wrong,
            'wrong_serve_ci': cp_ci(wrong, len(pids)),
            'correct_declined': dc['correct'],
            'correct_decline_rate': round(100 * dc['correct'] / n_corr, 1)
            if n_corr else None,
            'served_wrong_pids': sorted(
                (p for p in served if cat(p) in ('decayed', 'drifted')),
                key=int)}
print('\n== Leg 1: frozen rules on design-trail vs holdout (2026-08, '
      'human labels) ==')
for sname in subsets:
    print(f'  [{sname}] hits={leg1[sname]["hits"]} '
          f'decayed-or-drifted={leg1[sname]["decayed_or_drifted"]}')
    for name in policies:
        d = leg1[sname]['policies'][name]
        print(f'    {name:28s} served {d["served"]:3d}  wrong '
              f'{d["wrong_served"]}  corr-declined {d["correct_declined"]} '
              f'({d["correct_decline_rate"]}%)')
h = leg1['holdout']
assert h['decayed_or_drifted'] > 0, 'holdout check is vacuous'
assert h['policies']['anchored family (frozen)']['wrong_served'] == 0
print('  L1 OK: holdout is non-vacuous and the frozen anchored family '
      'serves zero decayed-or-drifted entries there.')

# ---------------------------------- Leg 2: forward check at 2026-10-16 ------
pr = json.load(open(f'{DATA}/protection_recovery.json'))
fwd = [e for e in pr['arm2']['entries'] if e.get('now') != 'not fetched']
fwd_declined = sum(1 for e in fwd if not e['reverted'])
leg2 = {
    'forward_date': '2026-10-16',
    'entries_refetched': len(fwd),
    'claims_level_contract_declines': fwd_declined,
    'all_declined': fwd_declined == len(fwd),
    'detail': [{'pid': e['pid'], 'qid': e['qid'], 'prop': e['prop'],
                'asserted_alive_now': e['asserted_alive_now'],
                'reverted': e['reverted']} for e in fwd],
    'anchored_rule_forward': 'not evaluable: no 2026-10 page snapshot '
                             'was rebuilt (sentence component needs page '
                             'text); network egress unavailable in this run',
}
print(f'\n== Leg 2: forward check at 2026-10-16 ==')
print(f'  {fwd_declined}/{len(fwd)} refetched decayed entries remain '
      f'declined by the frozen claims-level contracts '
      f"(asserted-alive now: {[e['asserted_alive_now'] for e in fwd]})")
assert leg2['all_declined']

# ------------------- Legs 3+4: admission dossiers and challenge audit -------
dossiers = []
for pid in RECOVERED:
    x = by_pid[pid]
    ent = entT.get(x['qid'])
    t_items = truthy_items(ent, x['prop']) if ent is not None else []
    t_texts = [value_text(v) for _, v in t_items]
    claim_key2text = {val_key(c['value']): vt
                      for c, vt in zip(x['certificate']['claims'],
                                       x['values_text'])}
    akeys = asserted.get(str(pid), [])
    sents = [(s['id'], s['text']) for s in x['certificate']['sentences']]
    anchored = [(sid, txt) for sid, txt in sents
                if any(mentions(txt, tt) for tt in t_texts)]
    claim_cov = [k for k in akeys if k in {k2 for k2, _ in t_items}]
    sent_cov = info[pid]['sentence_covered_asserted']
    # co-occurrence shape: does a covering sentence mention BOTH a
    # sentence-covered (stale-in-claims) asserted value AND a live value?
    shapes = []
    for k in sent_cov:
        vt = claim_key2text.get(k, k)
        for sid, txt in anchored:
            live_mentions = [tt for tt in t_texts if mentions(txt, tt)]
            if mentions(txt, vt) and live_mentions:
                shapes.append({'value': vt, 'sentence': txt,
                               'co_occurring_live_values': live_mentions})
    dossiers.append({
        'pid': pid, 'label': x['label'], 'prop_name': x.get('prop_name'),
        'question': x['question'],
        'cached_answer': gen[str(pid)]['answer'],
        'human_label_A4': lab[pid].get('A4'),
        'asserted_values': {claim_key2text.get(k, k):
                            ('claim-covered' if k in claim_cov
                             else 'sentence-covered') for k in akeys},
        'current_truthy_values': t_texts,
        'anchoring_sentences': [txt for _, txt in anchored],
        'co_occurrence_shape': shapes,
    })

challenge = [d for d in dossiers if d['co_occurrence_shape']]
leg34 = {
    'admissions': len(dossiers),
    'all_human_correct': all(d['human_label_A4'] == '正确'
                             for d in dossiers),
    'challenge_entries_with_co_occurrence_shape': len(challenge),
    'challenge_wrong_serves': sum(1 for d in challenge
                                  if d['human_label_A4'] != '正确'),
    'dossiers': dossiers,
}
print(f'\n== Legs 3+4: admission dossiers and challenge audit ==')
for d in dossiers:
    print(f"  pid {d['pid']} ({d['label']}, {d['prop_name']}): "
          f"A4={d['human_label_A4']}; asserted {d['asserted_values']}; "
          f"co-occurrence shapes: {len(d['co_occurrence_shape'])}")
assert leg34['all_human_correct']
print('  L3/L4 OK: every admission is human-confirmed correct; the '
      'co-occurrence loophole did not materialize into a wrong serve.')

# ------------------------------------------------------------------- out ----
out = {
    'frozen_artifacts': ['four dependency contracts',
                         'anchored lexical acceptance rule '
                         '(qualified_alternatives.json, zero revisions '
                         'since freezing)', 't0 base certificates'],
    'leg1_entity_holdout': leg1,
    'leg2_forward_2026_10_16': leg2,
    'leg3_4_admission_dossiers_and_challenge': leg34,
    'honesty_notes': [
        'the anchored rule wording was informed by panel-wide failure '
        'analysis; the holdout is pristine for evaluation (no holdout '
        'label consulted in any revision) but not for wording',
        'the 2026-08 endpoint labels informed the design trail for the '
        '12 design-trail entries; the forward check at 2026-10-16 covers '
        'the decayed cohort only (claims state), no forward page text',
        'representative panel (natural prevalence) and challenge runs '
        '(fault injection, s30) are reported separately throughout'],
}
json.dump(out, open(f'{DATA}/frozen_independent_check.json', 'w'),
          ensure_ascii=False, indent=1)
print('\nsaved data/frozen_independent_check.json')

#!/usr/bin/env python3
"""s52_claim_equality_loop_alias.py -- claim-level equality + write-back
loop with evidence-checked retrieval-alias expansion ENABLED.

Identical to s51_claim_equality_loop.py except that, after an accepted
regeneration, the request is added to the retrieval hit set (the same
evidence-checked alias mechanism as s25/s49 alias-on).  This gives the
matched-alias-setting comparison against the asserted loop's 29 accepted
regenerations (asserted|alias-on in s49), complementing the alias-off
comparison (31 versus 43) reported from s51.

No new model calls: blind refresh archived real generations for every
(request, tick) prompt, so all fallbacks come from the archive.

Validations:
  V3  every request accounted exactly once per tick
  V5  zero archive coverage misses

Output: data/claim_equality_loop_alias.json + console summary.
"""
import json, os, io, contextlib, importlib.util
from collections import defaultdict, Counter

os.environ['S30_NO_WRITE'] = '1'
HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.normpath(os.path.join(HERE, '..', 'data'))

spec = importlib.util.spec_from_file_location(
    's30', os.path.join(HERE, 's30_real_payload_chain.py'))
s30 = importlib.util.module_from_spec(spec)
with contextlib.redirect_stdout(io.StringIO()):
    spec.loader.exec_module(s30)
s25 = s30.s25

T, BASE = s25.T, s25.SNAPS[0]
TICKS = s25.TICKS
by_pid, lab, regen_lab, DRIFT = s25.by_pid, s25.lab, s25.regen_lab, s25.DRIFT
usable, HITS = s25.usable, s25.HITS
tbl = {k: v['R1_adjudicated']
       for k, v in json.load(open(f'{DATA}/tableb_labels.json')).items()}
CAT = {'正确': 'correct', '错误': 'wrong', '不完整': 'incomplete',
       '无法判断': 'indet'}

# ---------------------------------------------------------- real archive ---
REAL = {}
for p, r in json.load(open(f'{DATA}/regeneration.json')).items():
    REAL[(int(p), T)] = r
for k, r in json.load(open(f'{DATA}/tableb_regeneration.json')).items():
    REAL[(int(k.split('|')[0]), k.split('|')[1])] = r

# ------------------------------------------------------- certificate layer ---
def base_certificate(x):
    claims = [s25.val_key(c['value']) for c in x['certificate']['claims']]
    return {'claims': claims, 'issued_at': BASE}

def real_certificate(x, tag):
    ent = s25.snaps_ent[tag].get(str(x['qid']))
    items = s25.truthy_items(ent, x['prop']) if ent is not None else []
    return {'claims': [k for k, _ in items], 'issued_at': tag}

n_cover_miss = [0]

def real_fallback(x, tag):
    r = REAL.get((int(x['pid']), tag))
    if r is None:
        n_cover_miss[0] += 1
        raise AssertionError(f'coverage miss: {(x["pid"], tag)}')
    if not r.get('regen_ok'):
        return None, None
    return r['regen_answer'], real_certificate(x, tag)

# -------------------------------------------------------------------- gate ---
def claim_equality_serve(cert, st, x, tag):
    """Claim-level equality of the queried property: current truthy set
    identical to the issuance-time claim set (no page-version condition)."""
    return set(cert['claims']) == st['truthy']

# -------------------------------------------------------------------- run ---
def run_arm():
    store = {}
    for x in usable:
        store[int(x['pid'])] = {'answer': s25.gen[str(x['pid'])]['answer'],
                                'cert': base_certificate(x)}
    hits = set(HITS)
    rows = []
    n_alias = 0
    for tag in TICKS:
        acc = defaultdict(int)
        detail = {}
        for pid in sorted(store):
            x = by_pid[pid]
            st = s25.state_of(tag, x['qid'], x['prop'], x.get('page_title'))
            served = pid in hits and claim_equality_serve(store[pid]['cert'],
                                                          st, x, tag)
            if served:
                acc['served'] += 1
                cert = store[pid]['cert']
                origin = cert['issued_at']
                depdead = [k for k in cert['claims'] if k not in st['truthy']]
                if tag == T and pid in DRIFT:
                    acc['wrong_proxy'] += 1
                    detail[pid] = ['served-wrong-proxy', origin]
                elif depdead:
                    acc['decayed_proxy'] += 1
                    detail[pid] = ['served-decayed-proxy', origin]
                else:
                    detail[pid] = ['served', origin]
                continue
            ans, cert = real_fallback(x, tag)
            if ans is None:
                acc['abstain'] += 1
                detail[pid] = ['abstain', None]
                continue
            acc['fallback'] += 1
            store[pid] = {'answer': ans, 'cert': cert}   # write-back
            if pid not in hits:
                n_alias += 1
            hits.add(pid)        # evidence-checked retrieval alias
            detail[pid] = ['regenerated', tag]
        for k in ('served', 'fallback', 'abstain', 'wrong_proxy',
                  'decayed_proxy'):
            acc[k] += 0
        rows.append({'tick': tag, **acc, '_detail': detail})
    return rows, n_alias

def endpoint_quality(det):
    cnt = Counter()
    for pid, (how, origin) in det.items():
        if how == 'abstain':
            cnt['abstain'] += 1
        elif how == 'regenerated':
            if origin == T and pid in regen_lab:
                cnt[CAT.get(regen_lab[pid]['R1_adjudicated'], 'indet')] += 1
            else:
                l = tbl.get(f'{pid}|{origin}')
                cnt[CAT.get(l, 'unlabeled') if l else 'unlabeled'] += 1
        elif pid in DRIFT:
            cnt['wrong'] += 1
        elif origin == BASE:
            cnt[CAT.get(lab[pid].get('A4', '无法判断'), 'indet')] += 1
        else:
            l = tbl.get(f'{pid}|{origin}')
            cnt[CAT.get(l, 'unlabeled') if l else 'unlabeled'] += 1
    return dict(cnt)

print('running claim-equality loop with alias expansion (real-driven) ...')
rows, n_alias = run_arm()
for r in rows:
    assert r['served'] + r['fallback'] + r['abstain'] == 154, \
        f'V3 FAILED at {r["tick"]}'
print('V3 OK: every request accounted exactly once per tick.')
print('V5 OK: archive coverage misses =', n_cover_miss[0])

eq = endpoint_quality(rows[-1]['_detail'])
summary = {
    'per_tick': [{k: v for k, v in r.items() if k != '_detail'}
                 for r in rows],
    'total_fallbacks': sum(r['fallback'] for r in rows),
    'total_contract_violating_serves_proxy': sum(r['wrong_proxy']
                                                 for r in rows),
    'total_partially_stale_serves_proxy': sum(r['decayed_proxy']
                                              for r in rows),
    'total_abstains': sum(r['abstain'] for r in rows),
    'aliases_installed': n_alias,
    'endpoint_quality': eq,
}
print('\n===== claim-equality loop + alias, real-driven (3 monthly ticks) =====')
print('per-tick:', summary['per_tick'])
print('total fallbacks:', summary['total_fallbacks'])
print('violating serves (proxy):',
      summary['total_contract_violating_serves_proxy'])
print('partially-stale serves (proxy):',
      summary['total_partially_stale_serves_proxy'])
print('abstains:', summary['total_abstains'])
print('aliases installed:', n_alias)
print('endpoint quality:', eq)

out = {'convention': 'real-driven (s37): archived real generations at every '
                     'tick; coverage assertion enforced; Table-B endpoint '
                     'labelling; comparable with main-text Tables 3 and 4',
       'ticks': TICKS,
       'gate': 'claim-level equality of the queried property: current '
               'projected truthy-value set exactly equals the issuance-time '
               'certificate claim set; NO source-page version condition; '
               'evidence-checked retrieval-alias expansion ENABLED (s25/s49 '
               'alias-on mechanism), matched to the asserted loop\'s '
               'alias-on setting',
       'policy': {'claim-equality-loop|alias-on': summary},
       'validations': {'V3_accounting_complete': True,
                       'V5_coverage_misses': n_cover_miss[0]}}
json.dump(out, open(f'{DATA}/claim_equality_loop_alias.json', 'w'),
          ensure_ascii=False, indent=1)
print('\nsaved data/claim_equality_loop_alias.json')

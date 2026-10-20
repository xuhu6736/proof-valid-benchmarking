#!/usr/bin/env python3
"""s48_mechanism_arms.py -- Stage A: mechanism-level baseline arms from
recent evidence/freshness-aware cache systems, run under the SAME
real-driven closed-loop convention as s37_real_driven_loop.py (archived
real generations at every tick, coverage assertion enforced, Table-B
endpoint labelling), so the counts are directly comparable with main-text
Tables 3 and 4.

Three arms, each a faithful mechanism-level mapping of the cited system:

  grounded-router   GroundedCache (Shah 2026, arXiv:2605.27494):
                    serve a hit only when ALL four gates hold:
                      G1 query similarity   -- the shared tau-calibrated
                                               retrieval hit (all arms share it)
                      G2 evidence overlap   -- >=1/2 of the certificate's
                                               claims still truthy
                      G3 source version     -- source page hash unchanged
                                               since certificate issuance
                      G4 lexical support    -- every asserted value is
                                               currently truthy OR still
                                               mentioned in the current
                                               source page text
                    gate failure -> archived real regeneration + write-back
                    (no evidence-checked alias expansion; that is PVSB-only)

  freshcache-risk   FreshCache (Yoon & Ahmad 2026, arXiv:2607.04281),
                    answer tier only: per-property monthly claim-survival
                    rates calibrated on the FULL rebuilt population across
                    all consecutive snapshot pairs; entry risk =
                    1 - s_prop^(age in ticks); serve iff risk <= 0.10 (the
                    paper's answer-tier budget).  Mechanism-level
                    simplification: properties play the role of freshness
                    classes, and the four monthly snapshots replace the
                    1h/12h/24h/7d re-crawl grid.

  finserve-fingerprint  FinCacheServe (Zeng 2026, arXiv:2607.26076):
                    dependency-consistent reuse bound to source version and
                    evidence fingerprint: serve iff the source page hash is
                    unchanged AND the current truthy claim set equals the
                    certificate claim set exactly (no loss, no addition).
                    Model identity and decoding configuration are constant
                    in this replay and therefore not re-checked.

Validations:
  V3  every request accounted exactly once per tick per policy
  V5  zero archive coverage misses (every reachable regeneration key has
      an archived record; regen_ok=False is an abstention)
  V6  diagnostic: grounded-router vs page-change-loop serve counts per tick
      (printed, not asserted: write-back histories can diverge)

Output: data/mechanism_arms.json + console summary.
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
TIDX = {t: i for i, t in enumerate(s25.SNAPS)}       # age in monthly ticks
by_pid, lab, regen_lab, DRIFT = s25.by_pid, s25.lab, s25.regen_lab, s25.DRIFT
usable, HITS = s25.usable, s25.HITS
tbl = {k: v['R1_adjudicated']
       for k, v in json.load(open(f'{DATA}/tableb_labels.json')).items()}
CAT = {'正确': 'correct', '错误': 'wrong', '不完整': 'incomplete',
       '无法判断': 'indet'}

RISK_BUDGET = 0.10       # FreshCache answer-tier error budget
OVERLAP_MIN = 0.5        # GroundedCache evidence-overlap gate

# ---------------------------------------------------------- real archive ---
REAL = {}
for p, r in json.load(open(f'{DATA}/regeneration.json')).items():
    REAL[(int(p), T)] = r
for k, r in json.load(open(f'{DATA}/tableb_regeneration.json')).items():
    REAL[(int(k.split('|')[0]), k.split('|')[1])] = r

# ------------------------------------------------------- certificate layer ---
def base_certificate(x):
    claims = [s25.val_key(c['value']) for c in x['certificate']['claims']]
    ans = s25.gen[str(x['pid'])]['answer']
    asserted, asserted_texts = [], {}
    for c, vt in zip(x['certificate']['claims'], x['values_text']):
        if s25.answer_asserts(ans, vt):
            k = s25.val_key(c['value'])
            asserted.append(k)
            asserted_texts[k] = vt
    return {'claims': claims, 'asserted': asserted,
            'asserted_texts': asserted_texts,
            'sentences': [s['id'] for s in x['certificate']['sentences']],
            'issued_at': BASE, 'origin': 'base'}

def real_certificate(x, tag, answer):
    """s37's real-answer certificate, augmented with asserted value texts
    (needed by the GroundedCache lexical-support gate)."""
    ent = s25.snaps_ent[tag].get(str(x['qid']))
    items = s25.truthy_items(ent, x['prop']) if ent is not None else []
    claims = [k for k, _ in items]
    asserted = [k for k, v in items
                if s25.answer_asserts(answer, s25.value_text(v))]
    asserted_texts = {k: s25.value_text(v) for k, v in items
                      if k in set(asserted)}
    texts = [s25.value_text(v) for _, v in items]
    sents = []
    wt = s25.pages[tag].get(x.get('page_title'), '')
    for s in s25.sentences(s25.wikitext_to_plain(wt)):
        if any(s25.norm(vt) and s25.norm(vt) in s25.norm(s)
               for vt in texts):
            sents.append(s25.sid(s))
        if len(sents) == 3:
            break
    return {'claims': claims, 'asserted': asserted,
            'asserted_texts': asserted_texts, 'sentences': sents,
            'issued_at': tag, 'origin': f'real@{tag}'}

n_cover_miss = [0]

def real_fallback(x, tag):
    r = REAL.get((int(x['pid']), tag))
    if r is None:
        n_cover_miss[0] += 1
        raise AssertionError(f'coverage miss: {(x["pid"], tag)}')
    if not r.get('regen_ok'):
        return None, None
    return r['regen_answer'], real_certificate(x, tag, r['regen_answer'])

# ------------------------------------------- FreshCache survival calibration ---
print('calibrating per-property monthly claim-survival rates '
      '(full rebuilt population, consecutive snapshot pairs) ...')
prop_surv_num, prop_surv_den = defaultdict(int), defaultdict(int)
for t0, t1 in zip(s25.SNAPS, s25.SNAPS[1:]):
    for qid, ent0 in s25.snaps_ent[t0].items():
        ent1 = s25.snaps_ent[t1].get(qid)
        if ent1 is None:
            continue
        for prop in ent0['claims']:
            k0 = s25.truthy_keys(ent0, prop)
            if not k0:
                continue
            k1 = s25.truthy_keys(ent1, prop)
            prop_surv_num[prop] += len(k0 & k1)
            prop_surv_den[prop] += len(k0)
prop_survival = {p: prop_surv_num[p] / prop_surv_den[p]
                 for p in prop_surv_den if prop_surv_den[p] > 0}
tot_num, tot_den = sum(prop_surv_num.values()), sum(prop_surv_den.values())
DEFAULT_SURVIVAL = tot_num / tot_den
print(f'  pooled monthly survival {DEFAULT_SURVIVAL:.4f} over '
      f'{tot_den:,} claim observations; {len(prop_survival)} properties')

# lazy normalized plain text of a page (GroundedCache G4 lexical support)
_plain_cache = {}
def page_plain_norm(tag, title):
    key = (tag, title)
    if key not in _plain_cache:
        w = s25.pages[tag].get(title)
        _plain_cache[key] = s25.norm(s25.wikitext_to_plain(w)) if w else ''
    return _plain_cache[key]

# ------------------------------------------------------------------- gates ---
def grounded_serve(cert, st, x, tag):
    """GroundedCache four-gate router (G1 = shared retrieval hit)."""
    claims = cert['claims']                                # G2 evidence overlap
    if claims:
        ov = sum(1 for k in claims if k in st['truthy']) / len(claims)
        if ov < OVERLAP_MIN:
            return False
    issued = cert['issued_at']                             # G3 source version
    if st['page_hash'] is None or \
       s25.page_hash[issued].get(x.get('page_title')) != st['page_hash']:
        return False
    ptxt = None                                            # G4 lexical support
    for k, vt in cert.get('asserted_texts', {}).items():
        if k in st['truthy']:
            continue
        if ptxt is None:
            ptxt = page_plain_norm(tag, x.get('page_title'))
        if not (s25.norm(vt) and s25.norm(vt) in ptxt):
            return False
    return True

def freshcache_serve(cert, st, x, tag):
    """FreshCache answer-tier risk gate: 1 - s_prop^age <= eps."""
    age = TIDX[tag] - TIDX[cert['issued_at']]
    s = prop_survival.get(x['prop'], DEFAULT_SURVIVAL)
    return (1.0 - s ** age) <= RISK_BUDGET

def finserve_serve(cert, st, x, tag):
    """FinCacheServe dependency consistency: source version + exact evidence
    fingerprint (current truthy set identical to the certificate claim set)."""
    if st['page_hash'] is None or \
       s25.page_hash[cert['issued_at']].get(x.get('page_title')) != \
       st['page_hash']:
        return False
    return set(cert['claims']) == st['truthy']

GATES = {'grounded-router': grounded_serve,
         'freshcache-risk': freshcache_serve,
         'finserve-fingerprint': finserve_serve}

# -------------------------------------------------------------------- run ---
def run_arm(name):
    store = {}
    for x in usable:
        store[int(x['pid'])] = {'answer': s25.gen[str(x['pid'])]['answer'],
                                'cert': base_certificate(x)}
    hits = set(HITS)        # fixed retrieval outcome; no PVSB alias expansion
    gate = GATES[name]
    rows = []
    for tag in TICKS:
        acc = defaultdict(int)
        detail = {}
        for pid in sorted(store):
            x = by_pid[pid]
            st = s25.state_of(tag, x['qid'], x['prop'], x.get('page_title'))
            served = pid in hits and gate(store[pid]['cert'], st, x, tag)
            if served:
                acc['served'] += 1
                cert = store[pid]['cert']
                origin = cert['issued_at']
                dead = [k for k in cert['asserted'] if k not in st['truthy']]
                depdead = [k for k in cert['claims'] if k not in st['truthy']]
                if dead or (tag == T and pid in DRIFT):
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
            store[pid] = {'answer': ans, 'cert': cert}   # write-back, no alias
            detail[pid] = ['regenerated', tag]
        for k in ('served', 'fallback', 'abstain', 'wrong_proxy',
                  'decayed_proxy'):
            acc[k] += 0
        rows.append({'tick': tag, **acc, '_detail': detail})
    return rows

def endpoint_quality(det):
    """s37's Table-B scoring: base serves -> A4 with drift override; real
    regenerations -> adjudicated labels; gap-fill endpoint keys without
    human labels counted separately as 'unlabeled', never scored."""
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

results, quality = {}, {}
for name in GATES:
    print(f'running {name} (real-driven) ...')
    rows = run_arm(name)
    for r in rows:
        assert r['served'] + r['fallback'] + r['abstain'] == 154, \
            f'V3 FAILED for {name} at {r["tick"]}'
    results[name] = rows
    quality[name] = endpoint_quality(rows[-1]['_detail'])
print('V3 OK: every request accounted exactly once per tick per policy.')
print('V5 OK: archive coverage misses =', n_cover_miss[0])

# ------------------------------------------------------- comparison print ---
ref = json.load(open(f'{DATA}/real_driven_loop.json'))
pc = {r['tick']: r['served'] for r in ref['erasure_free']['per_tick']['page-change-loop']}
print('\nV6 diagnostic (grounded-router serves vs page-change-loop serves):')
for r in results['grounded-router']:
    print(f'  {r["tick"]}: grounded {r["served"]}  page-change {pc[r["tick"]]}')

summary = {}
for name in GATES:
    rows = results[name]
    summary[name] = {
        'per_tick': [{k: v for k, v in r.items() if k != '_detail'}
                     for r in rows],
        'total_fallbacks': sum(r['fallback'] for r in rows),
        'total_contract_violating_serves_proxy': sum(r['wrong_proxy']
                                                     for r in rows),
        'total_partially_stale_serves_proxy': sum(r['decayed_proxy']
                                                  for r in rows),
        'total_abstains': sum(r['abstain'] for r in rows),
        'endpoint_quality': quality[name],
    }

print('\n===== mechanism arms, real-driven (3 monthly ticks) =====')
print(f'{"policy":22s} {"fallbacks":>9s} {"viol*":>7s} {"p-stale*":>8s} '
      f'{"abstains":>8s} {"correct@T":>9s} {"wrong@T":>8s} {"unlab@T":>8s}')
for name in list(GATES) + ['similarity-static', 'TTL-1mo', 'page-change-loop',
                           'pvsb-strict-loop', 'pvsb-asserted-loop']:
    if name in summary:
        s = summary[name]
        ep = s['endpoint_quality']
        corr, wrong, unl = ep.get('correct', 0), ep.get('wrong', 0), ep.get('unlabeled', 0)
        fb = s['total_fallbacks']
        vi = s['total_contract_violating_serves_proxy']
        ps = s['total_partially_stale_serves_proxy']
        ab = s['total_abstains']
    else:
        rows = ref['erasure_free']['per_tick'][name]
        ep = ref['erasure_free']['endpoint_quality'][name]
        fb = sum(r['fallback'] for r in rows)
        vi = sum(r['wrong_proxy'] for r in rows)
        ps = sum(r['decayed_proxy'] for r in rows)
        ab = sum(r['abstain'] for r in rows)
        corr, wrong, unl = ep.get('correct', 0), ep.get('wrong', 0), ep.get('unlabeled', 0)
    print(f'{name:22s} {fb:9d} {vi:7d} {ps:8d} {ab:8d} {corr:9d} '
          f'{wrong:8d} {unl:8d}')
for name in GATES:
    print(f'\n{name} endpoint quality:', summary[name]['endpoint_quality'])
    print(f'{name} per-tick:', summary[name]['per_tick'])

out = {'convention': 'real-driven (s37): archived real generations at every '
                     'tick; coverage assertion enforced; Table-B endpoint '
                     'labelling; comparable with main-text Tables 3 and 4',
       'ticks': TICKS,
       'calibration': {'risk_budget': RISK_BUDGET,
                       'overlap_min': OVERLAP_MIN,
                       'default_monthly_survival': DEFAULT_SURVIVAL,
                       'prop_survival': prop_survival},
       'policies': summary,
       'validations': {'V3_accounting_complete': True,
                       'V5_coverage_misses': n_cover_miss[0]},
       'notes': [
           'mechanism-level mappings of GroundedCache (arXiv:2605.27494), '
           'FreshCache (arXiv:2607.04281), and FinCacheServe '
           '(arXiv:2607.26076) into the same real-driven closed-loop replay '
           'as s37; not system-level reproductions of the cited '
           'implementations',
           'all three arms share the tau-calibrated retrieval stage, the '
           'archived fallback outputs, and write-back without '
           'evidence-checked alias expansion (a PVSB-specific mechanism)',
           'FreshCache arm: per-property monthly claim-survival rates '
           'calibrated on the full rebuilt population over all consecutive '
           'snapshot pairs; risk = 1 - s^age at the answer tier, eps = 0.10',
           'FinCacheServe arm: model identity and decoding configuration '
           'are constant in this replay; the implemented contract is '
           'source-page version plus exact truthy-claim-set fingerprint']}
json.dump(out, open(f'{DATA}/mechanism_arms.json', 'w'),
          ensure_ascii=False, indent=1)
print('\nsaved data/mechanism_arms.json')

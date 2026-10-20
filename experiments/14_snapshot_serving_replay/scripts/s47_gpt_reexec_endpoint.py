#!/usr/bin/env python3
"""s47_gpt_reexec_endpoint.py -- compare endpoint delivery routes from the
FULL GPT re-execution (s46 machinery: rebuilt certificates, per-tick gates,
write-back) against the fixed-map substitution used by s45, and score the
re-executed endpoints with the GPT human-label store.
"""
import json, os, io, contextlib, importlib.util
from collections import Counter

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
by_pid, lab, regen_lab, DRIFT = s25.by_pid, s25.lab, s25.regen_lab, s25.DRIFT

REAL = {}
for p, r in json.load(open(f'{DATA}/regeneration_gpt.json')).items():
    REAL[(int(p), T)] = r
for k, r in json.load(open(f'{DATA}/tableb_regeneration_gpt.json')).items():
    REAL[(int(k.split('|')[0]), k.split('|')[1])] = r

def real_certificate(x, tag, answer):
    ent = s25.snaps_ent[tag].get(str(x['qid']))
    items = s25.truthy_items(ent, x['prop']) if ent is not None else []
    claims = [k for k, _ in items]
    asserted = [k for k, v in items
                if s25.answer_asserts(answer, s25.value_text(v))]
    texts = [s25.value_text(v) for _, v in items]
    sents = []
    wt = s25.pages[tag].get(x.get('page_title'), '')
    for s in s25.sentences(s25.wikitext_to_plain(wt)):
        if any(s25.norm(vt) and s25.norm(vt) in s25.norm(s)
               for vt in texts):
            sents.append(s25.sid(s))
        if len(sents) == 3:
            break
    return {'claims': claims, 'asserted': asserted, 'sentences': sents,
            'issued_at': tag, 'origin': f'real@{tag}'}

def gpt_fallback(x, tag):
    r = REAL.get((int(x['pid']), tag))
    if r is None:
        raise AssertionError(f'coverage miss: {(x["pid"], tag)}')
    if not r.get('regen_ok'):
        return None, None
    return r['regen_answer'], real_certificate(x, tag, r['regen_answer'])

s25.fallback = gpt_fallback
ARMS = ['similarity-static', 'TTL-1mo', 'page-change-loop',
        'pvsb-strict-loop', 'pvsb-asserted-loop']

# ---- re-executed endpoint routes
reexec = {}
for name in ARMS:
    with contextlib.redirect_stdout(io.StringIO()):
        rows = s25.run_policy(name)
    reexec[name] = {str(pid): tuple(v) for pid, v in rows[-1]['_detail'].items()}

# ---- fixed-map routes with acceptance substitution (s45 semantics):
# start from kimi delivery map; flip the two disagreement keys
emap = json.load(open(f'{DATA}/tableb_endpoint_map.json'))
fixed = {}
for arm in ARMS:
    fixed[arm] = {}
    for pid, rec in emap[arm].items():
        fixed[arm][pid] = (rec['how'], rec['origin'])

# apply acceptance substitutions per arm: a delivered regenerated answer whose
# key is rejected by gpt becomes abstain; an abstain whose mapped key is accepted
# by gpt becomes a delivery.  Reproduce via delivery_diffs instead (s42 output):
cmp = json.load(open(f'{DATA}/gpt_generator_compare.json'))
for dd in cmp['delivery_diffs']:
    arm, pid = dd['arm'], dd['pid']
    if dd['gpt'][0] == 'abstain':
        fixed[arm][pid] = ('abstain', None)
    else:
        # kimi abstain -> gpt serve; keep kimi's mapped origin
        fixed[arm][pid] = ('regenerated', emap[arm][pid]['origin'])

print(f'{"arm":22s} {"route-diffs reexec vs fixed":>28s}')
tot_diff = 0
for arm in ARMS:
    diffs = []
    for pid in reexec[arm]:
        r1 = reexec[arm][pid]
        r2 = fixed[arm].get(pid, ('abstain', None))
        n1 = (r1[0], r1[1]); n2 = (r2[0], r2[1])
        if n1 != n2:
            diffs.append((pid, n1, n2))
    tot_diff += len(diffs)
    print(f'{arm:22s} {len(diffs):28d}')
    for d in diffs[:8]:
        print('   ', d)

# ---- score re-executed endpoints with the GPT label store
gtab = {k: v['R1_adjudicated']
        for k, v in json.load(open(f'{DATA}/gpt_labels.json')).items()}
CAT = {'正确': 'correct', '错误': 'wrong', '不完整': 'incomplete',
       '无法判断': 'indet'}
regen_lab_g = json.load(open(f'{DATA}/regeneration_gpt.json'))
q = {}
for arm in ARMS:
    cnt = Counter()
    for pid, (how, origin) in reexec[arm].items():
        ipid = int(pid)
        if how == 'abstain':
            cnt['abstain'] += 1
        elif how == 'regenerated':
            l = gtab.get(f'{pid}|{origin}')
            cnt[CAT.get(l) if l else 'UNLABELED'] += 1
        elif ipid in DRIFT or pid in DRIFT:
            cnt['wrong'] += 1
        elif origin == BASE:
            rec = lab.get(ipid) or lab.get(pid)
            cnt[CAT.get(rec.get('A4', '无法判断'), 'indet')] += 1
        else:
            l = gtab.get(f'{pid}|{origin}')
            cnt[CAT.get(l) if l else 'UNLABELED'] += 1
    q[arm] = dict(cnt)
print()
for arm in ARMS:
    print(f'{arm:22s}', q[arm])
json.dump({'routes_reexec': {a: {p: list(v) for p, v in reexec[a].items()}
                               for a in ARMS},
           'endpoint_quality_reexec': q,
           'route_diffs_vs_fixed_map': tot_diff},
          open(f'{DATA}/gpt_reexec_endpoint.json', 'w'), indent=1)
print('saved data/gpt_reexec_endpoint.json')

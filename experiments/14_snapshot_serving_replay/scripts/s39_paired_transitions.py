#!/usr/bin/env python3
"""s39_paired_transitions.py -- paired per-request attribution under
REAL-driven state (s37), asserted contract, p=0.10.

Recomputes the archived template-driven transition counts (17,948 ->
10,279 / 7,669) with fallback routing driven by archived real model
outputs: for each CRN seed, the none arm and each protection arm run the
same fault plan; per (tick, request) we record the none-arm category and
the protected-arm category and tally transitions.  Whole-trajectory
comparison: each arm maintains its own cache state (see main text).

Output: data/paired_transitions_real.json + console table.
"""
import json, os, io, contextlib, importlib.util, sqlite3
from collections import defaultdict

os.environ['S30_NO_WRITE'] = '1'
HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.normpath(os.path.join(HERE, '..', 'data'))

spec = importlib.util.spec_from_file_location(
    's30', os.path.join(HERE, 's30_real_payload_chain.py'))
s30 = importlib.util.module_from_spec(spec)
with contextlib.redirect_stdout(io.StringIO()):
    spec.loader.exec_module(s30)
s25 = s30.s25
T = s25.T
R = s30.R

# ---- real fallback (identical to s37/s38) -----------------------------------
REAL = {}
for p_, r in json.load(open(f'{DATA}/regeneration.json')).items():
    REAL[(int(p_), T)] = r
for k, r in json.load(open(f'{DATA}/tableb_regeneration.json')).items():
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

def real_fallback(x, tag):
    r = REAL.get((int(x['pid']), tag))
    assert r is not None, f'coverage miss {(x["pid"], tag)}'
    if not r.get('regen_ok'):
        return None, None
    return r['regen_answer'], real_certificate(x, tag, r['regen_answer'])

s30.fallback = real_fallback
s30.FB.clear()
s30.con = sqlite3.connect(s30.DB)
s30.cur = s30.con.cursor()

P = 0.10
ARMS = ('none', 'random', 'fixed', 'full')
details = {}
means = defaultdict(list)
for r in range(R):
    seed = 1000 + r
    for arm in ARMS:
        acc, det = s30.run_loop('asserted', P, arm, seed, keep_detail=True)
        details[(arm, seed)] = det
        means[arm].append(acc['fallback'])
    if (r + 1) % 50 == 0:
        print(f'{r + 1}/{R}', flush=True)

transitions = {}
for arm in ('random', 'fixed', 'full'):
    tr = defaultdict(int)
    for seed in range(1000, 1000 + R):
        dn = details[('none', seed)]
        da = details[(arm, seed)]
        for key, cat_n in dn.items():
            tr[f'{cat_n}->{da[key]}'] += 1
    transitions[f'none->{arm}'] = dict(tr)

fb_means = {arm: round(sum(v) / R, 1) for arm, v in means.items()}
print(f'\nreal-driven fallback means (asserted, p={P}): {fb_means}')
for k, v in transitions.items():
    conv = v.get('fallback->served', 0)
    rem = v.get('fallback->fallback', 0)
    tot = sum(vv for kk, vv in v.items() if kk.startswith('fallback->'))
    print(f'  {k}: total none-fallbacks {tot}  converted-to-serve {conv} '
          f'({conv / R:.1f}/run)  remain-fallback {rem} ({rem / R:.1f}/run)')
    print(f'     full: {v}')

out = {'cell': f'asserted/p={P}', 'realizations': R,
       'state': 'real-driven (s37 fallback routing)',
       'fallback_means': fb_means, 'transitions': transitions,
       'alignment': 'whole-trajectory; common random numbers keyed by '
                    '(seed, tick, request, premise, read route)'}
json.dump(out, open(f'{DATA}/paired_transitions_real.json', 'w'),
          ensure_ascii=False, indent=1)
print('saved data/paired_transitions_real.json')

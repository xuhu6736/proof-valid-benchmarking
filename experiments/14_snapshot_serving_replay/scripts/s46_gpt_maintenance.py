#!/usr/bin/env python3
"""s46_gpt_maintenance.py -- maintenance (accepted-regeneration) counts for
the five erasure-free arms when every archived answer is replaced by its
gpt-5-mini output.  Replicates s37's real-driven loop with the GPT archive.
"""
import json, os, io, contextlib, importlib.util

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
res = {}
for name in ARMS:
    with contextlib.redirect_stdout(io.StringIO()):
        rows = s25.run_policy(name)
    res[name] = [sum(r['fallback'] for r in rows),
                 [r['fallback'] for r in rows],
                 sum(r['abstain'] for r in rows)]
for n, (tot, pertick, ab) in res.items():
    print(f'{n:22s} gpt fallbacks={tot:4d} per-tick={pertick} abstains={ab}')
json.dump({n: {'fallbacks': v[0], 'per_tick': v[1], 'abstains': v[2]}
           for n, v in res.items()},
          open(f'{DATA}/gpt_maintenance_counts.json', 'w'), indent=1)
print('saved data/gpt_maintenance_counts.json')

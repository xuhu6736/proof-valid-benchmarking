#!/usr/bin/env python3
"""s37b_real_driven_sweep.py -- full-counter sweep under REAL-driven state.

s37 archived only fallback/serve/abstain/endpoint fields; the cost and
protection paragraphs also need restores, false declines, guard counters,
both-unreadable and p95 under real-driven fallback routing.  This script
re-runs the complete 2-contract x 4-arm x 3-loss-rate sweep (200 CRN seeds
per cell) with the same archived-real-output fallback as s37 and saves the
full aggregate set.

Output: data/real_driven_sweep.json + console table.
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

KEYS = ('fallback', 'false_decline', 'restores', 'guard_withdrawn',
        'guard_incompat', 'both_unreadable', 'served', 'abstain',
        'endpoint_correct')
results = {}
for contract in s30.CONTRACTS:
    for arm in ('none', 'random', 'fixed', 'full'):
        for p in s30.PS:
            cells = []
            for r in range(R):
                acc, _ = s30.run_loop(contract, p, arm, 1000 + r)
                cells.append(acc)
            agg = {f'{k}_mean': round(sum(c[k] for c in cells) / R, 1)
                   for k in KEYS}
            agg['fallback_p95'] = sorted(c['fallback'] for c in cells)[
                int(0.95 * R)]
            agg['wrong_max'] = max(c['wrong'] for c in cells)
            assert agg['wrong_max'] == 0
            results[f'{contract}/{arm}/p={p}'] = agg
    print(f'{contract} done', flush=True)

json.dump({'realizations': R, 'state': 'real-driven (archived real '
           'fallback outputs, coverage-asserted)', 'cells': results},
          open(f'{DATA}/real_driven_sweep.json', 'w'), indent=1)
print(f'{"cell":26s} {"fallb":>6s} {"p95":>5s} {"falseD":>6s} {"restr":>6s} '
      f'{"grdW":>5s} {"grdI":>5s} {"bothU":>5s} {"corr":>6s}')
for k, a in results.items():
    print(f'{k:26s} {a["fallback_mean"]:6.1f} {a["fallback_p95"]:5d} '
          f'{a["false_decline_mean"]:6.1f} {a["restores_mean"]:6.1f} '
          f'{a["guard_withdrawn_mean"]:5.1f} {a["guard_incompat_mean"]:5.1f} '
          f'{a["both_unreadable_mean"]:5.1f} {a["endpoint_correct_mean"]:6.1f}')
print('saved data/real_driven_sweep.json')

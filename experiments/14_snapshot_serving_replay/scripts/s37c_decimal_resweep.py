#!/usr/bin/env python3
"""s37c_decimal_resweep.py -- re-aggregate the real-driven fault sweep with
exact integer sums and decimal (half-up) one-decimal means.

Motivation: round(sum/200, 1) uses binary floating point, so the exact mean
34.15 was displayed as 34.1.  This rerun keeps the protocol identical to
s37b (same seeds, same real_fallback, same arms) and reports exact sums plus
decimal-rounded means, cross-checking against paired_transitions_real.json.

Output: data/real_driven_sweep.json is REPLACED (old file backed up to
data/real_driven_sweep_binary_rounding_backup.json); each cell gains
'<key>_sum' integer fields.
"""
import json, os, io, contextlib, importlib.util, sqlite3, shutil
from decimal import Decimal, ROUND_HALF_UP

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

def d1(num, den):
    """decimal half-up rounding of num/den to one decimal place"""
    return float((Decimal(num) / Decimal(den)).quantize(
        Decimal('0.1'), rounding=ROUND_HALF_UP))

results = {}
for contract in s30.CONTRACTS:
    for arm in ('none', 'random', 'fixed', 'full'):
        for p in s30.PS:
            cells = []
            for r in range(R):
                acc, _ = s30.run_loop(contract, p, arm, 1000 + r)
                cells.append(acc)
            agg = {}
            for k in KEYS:
                s = sum(c[k] for c in cells)
                agg[f'{k}_sum'] = s
                agg[f'{k}_mean'] = d1(s, R)
            agg['fallback_p95'] = sorted(c['fallback'] for c in cells)[
                int(0.95 * R)]
            agg['wrong_max'] = max(c['wrong'] for c in cells)
            assert agg['wrong_max'] == 0
            results[f'{contract}/{arm}/p={p}'] = agg
    print(f'{contract} done', flush=True)

# ---- cross-check against paired transitions (asserted, none vs fixed, p=.10)
pt = json.load(open(f'{DATA}/paired_transitions_real.json'))
nfu = results['asserted/none/p=0.1']['fallback_sum']
nfp = results['asserted/fixed/p=0.1']['fallback_sum']
print(f'paired totals check: none={nfu} fixed={nfp} '
      f'(paired_transitions_real.json expects 13711 / 6830)')
assert nfu == 13711 and nfp == 6830, 'paired totals mismatch'

# ---- diff against the archived binary-rounded aggregation
old = json.load(open(f'{DATA}/real_driven_sweep.json'))['cells']
print(f'{"cell":26s} {"old":>6s} {"new":>6s}  note')
diffs = []
for k, a in results.items():
    o = old[k]['fallback_mean']
    n = a['fallback_mean']
    mark = '' if o == n else '  <-- CHANGED'
    if o != n:
        diffs.append((k, o, n, a['fallback_sum']))
    print(f'{k:26s} {o:6.1f} {n:6.1f}{mark}')

out = {'realizations': R,
       'state': 'real-driven (archived real fallback outputs, '
                'coverage-asserted)',
       'rounding': 'decimal half-up from exact integer sums; '
                   'supersedes binary-float round() display',
       'cells': results}
shutil.copy(f'{DATA}/real_driven_sweep.json',
            f'{DATA}/real_driven_sweep_binary_rounding_backup.json')
json.dump(out, open(f'{DATA}/real_driven_sweep.json', 'w'), indent=1)
print('saved data/real_driven_sweep.json (old file backed up)')
print('changed cells:', diffs)

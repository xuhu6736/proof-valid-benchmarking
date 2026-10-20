#!/usr/bin/env python3
"""s38_residual_decomposition.py -- closed decomposition of the fixed-arm
fallback residual at asserted/p=0.10, under REAL-driven state (s37).

The archived text said the residual "decomposes into 9.3 false declines
and 6.3 both-unreadable losses", but 9.3 + 6.3 = 15.6 != 38.3 - 29.
This script instruments every fallback event of the asserted/fixed/p=0.10
cell (200 CRN realizations, real-output-driven state) and classifies it
in a mutually exclusive way:

  necessary        the request also falls back in the erasure-free loop
                   (evidence genuinely changed) -- the 29/run baseline
  both-unreadable  an extra fallback on a request whose run lost BOTH
                   the source read and the backup read of >=1 protected
                   certificate item this tick
  tail-loss        an extra fallback with no both-unreadable event: the
                   loss hit the unprotected byte tail or an unprotected
                   write-back item

"Extra" = fallback here while the same request is served in the SAME
run's erasure-free trajectory ... is not well-defined across runs, so we
instead replay the p=0 fixed arm per seed with the same code path and
align per request-tick: a fallback is extra iff the p=0 run serves that
request at that tick.  (State can diverge within a run between p=0 and
p=0.10; both use the same seeds and the same real generations.)

Output: data/residual_decomposition.json + console closed table.
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
T = s25.T

# ---- real fallback (identical to s37) --------------------------------------
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

import sqlite3
s30.con = sqlite3.connect(s30.DB)
s30.cur = s30.con.cursor()

# ---- instrumented single-cell loop -----------------------------------------
def run_cell(p, seed):
    """returns per-request-tick records for asserted/fixed at loss rate p."""
    S, _ = s30.arm_set('fixed', seed)
    store = {int(x['pid']): {'answer': s25.gen[str(x['pid'])]['answer'],
                             'cert': s30.base_cert_of(int(x['pid']))}
             for x in s25.usable}
    hits = set(s30.HITS)
    recs = {}
    for tag in s30.TICKS:
        for pid in sorted(store):
            x = s30.by_pid[pid]
            st = s30.st_of(pid, tag)
            served = False
            both_unread = 0
            if pid in hits:
                ctr = defaultdict(int)
                served = s30.chain_verify(x, store[pid]['cert'], tag, pid,
                                          'asserted', S, p, seed, ctr)
                both_unread = ctr.get('both_unreadable', 0)
            if served:
                recs[(tag, pid)] = ('served', 0)
                continue
            ans, cert = s30.fb_of(pid, tag)
            if ans is None:
                recs[(tag, pid)] = ('abstain', both_unread)
                continue
            store[pid] = {'answer': ans, 'cert': cert}
            hits.add(pid)
            recs[(tag, pid)] = ('fallback', both_unread)
    return recs

R = s30.R
agg = Counter()
for r in range(R):
    seed = 1000 + r
    rec_f = run_cell(0.10, seed)
    rec_0 = run_cell(0.0, seed)          # same seed, erasure-free trajectory
    for key, (cat, bu) in rec_f.items():
        if cat != 'fallback':
            continue
        base_cat = rec_0[key][0]
        if base_cat == 'fallback' or base_cat == 'abstain':
            agg['necessary'] += 1        # also falls back with no erasure
        elif bu > 0:
            agg['both_unreadable'] += 1
        else:
            agg['tail_loss'] += 1
    if (r + 1) % 50 == 0:
        print(f'{r + 1}/{R}', flush=True)

tot = sum(agg.values())
print('\nclosed decomposition of asserted/fixed/p=0.10 fallbacks '
      f'(real-driven state, {R} runs x 3 ticks):')
for k in ('necessary', 'both_unreadable', 'tail_loss'):
    print(f'  {k:16s} {agg[k] / R:6.2f} per run')
print(f'  {"TOTAL":16s} {tot / R:6.2f} per run')

out = {'cell': 'asserted/fixed/p=0.10', 'realizations': R,
       'state': 'real-driven (s37)', 'mutually_exclusive': True,
       'per_run_means': {k: round(v / R, 2) for k, v in agg.items()},
       'total_per_run': round(tot / R, 2),
       'alignment': 'a fallback is "necessary" iff the same request-tick '
                    'falls back or abstains in the same-seed p=0 run',
       'note': 'both-unreadable takes precedence over tail-loss when a '
               'request-tick shows both'}
json.dump(out, open(f'{DATA}/residual_decomposition.json', 'w'),
          ensure_ascii=False, indent=1)
print('saved data/residual_decomposition.json')

#!/usr/bin/env python3
"""s37_real_driven_loop.py -- re-drive the closed-loop state machine with
the archived REAL generations instead of template compositions.

Reviewer ask (acceptance round, criterion 1): the routing counts of the
closed-loop tables must be driven by real model outputs, because a real
answer may assert fewer claims than the template (which asserts every
truthy value), producing a different written-back certificate and
different verdicts at later ticks.

Method:
  * fallback at ANY tick returns the archived real regeneration for
    (pid, tick) -- regeneration.json (endpoint round, s9) merged with
    tableb_regeneration.json (s32 + s36 gap fill).  A missing key raises
    (coverage assertion: no silent template fallback).  regen_ok=False
    (accept-fail or no-current-evidence) is an abstention.
  * the written-back certificate is built from the REAL answer: claims
    = truthy keys at the issue tick, asserted = only the claims whose
    value text the answer actually asserts, sentences = the up-to-3
    evidence sentences (same construction as s25.current_certificate).
  * erasure-free arms: s25.run_policy with s25.fallback monkeypatched.
  * fault sweep: s30.run_loop with s30.fallback monkeypatched and the
    fallback cache cleared; same seeds/CRN fault plans as s30.
  * endpoint quality uses the human labels (s34 logic): base serves ->
    A4 with drift override; real regenerations -> adjudicated labels;
    gap-fill endpoint keys without human labels are counted separately
    as 'unlabeled' (never silently scored).

Outputs: data/real_driven_loop.json + console comparison against the
template-driven archives (closed_loop_system.json,
tableb_protected_loop.json cells).
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
by_pid, lab, regen_lab, DRIFT = s25.by_pid, s25.lab, s25.regen_lab, s25.DRIFT
tbl = {k: v['R1_adjudicated']
       for k, v in json.load(open(f'{DATA}/tableb_labels.json')).items()}
CAT = {'正确': 'correct', '错误': 'wrong', '不完整': 'incomplete',
       '无法判断': 'indet'}

REAL = {}
for p, r in json.load(open(f'{DATA}/regeneration.json')).items():
    REAL[(int(p), T)] = r
for k, r in json.load(open(f'{DATA}/tableb_regeneration.json')).items():
    REAL[(int(k.split('|')[0]), k.split('|')[1])] = r


def real_certificate(x, tag, answer):
    """certificate built from the REAL answer: asserted = only the claims
    whose value text the answer asserts."""
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


n_cover_miss = [0]

def real_fallback(x, tag):
    r = REAL.get((int(x['pid']), tag))
    if r is None:
        n_cover_miss[0] += 1
        raise AssertionError(f'coverage miss: {(x["pid"], tag)}')
    if not r.get('regen_ok'):
        return None, None                     # abstain (no evidence / fail)
    return r['regen_answer'], real_certificate(x, tag, r['regen_answer'])


def endpoint_quality(det, collect=None):
    """Table-B-style scoring of one endpoint detail map; unlabeled
    (pid|origin) keys are counted separately and, if `collect` is a set,
    recorded into it."""
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
                if l is None and collect is not None:
                    collect.add(f'{pid}|{origin}')
        elif pid in DRIFT:
            cnt['wrong'] += 1
        elif origin == BASE:
            cnt[CAT.get(lab[pid].get('A4', '无法判断'), 'indet')] += 1
        else:
            l = tbl.get(f'{pid}|{origin}')
            cnt[CAT.get(l, 'unlabeled') if l else 'unlabeled'] += 1
            if l is None and collect is not None:
                collect.add(f'{pid}|{origin}')
    return dict(cnt)


# ------------------------------------------- erasure-free arms, real-driven -
s25.fallback = real_fallback
ARMS = ['similarity-static', 'TTL-1mo', 'page-change-loop',
        'pvsb-strict-loop', 'pvsb-asserted-loop']
real_rows, real_quality = {}, {}
for name in ARMS:
    rows = s25.run_policy(name)
    real_rows[name] = [{k: v for k, v in r.items() if k != '_detail'}
                       for r in rows]
    real_quality[name] = endpoint_quality(rows[-1]['_detail'])

arch = json.load(open(f'{DATA}/closed_loop_system.json'))
print('=== erasure-free arms: template-driven -> real-driven ===')
print(f'{"policy":22s} {"fallbacks":>18s} {"viol":>10s} {"abstains":>14s}')
for name in ARMS:
    old = arch['policies'][name]
    new_fb = sum(r['fallback'] for r in real_rows[name])
    new_vi = sum(r['wrong_proxy'] for r in real_rows[name])
    new_ab = sum(r['abstain'] for r in real_rows[name])
    print(f'{name:22s} {old["total_fallbacks"]:>8d}->{new_fb:<8d} '
          f'{old["total_contract_violating_serves_proxy"]:>4d}->{new_vi:<4d} '
          f'{old["total_abstains"]:>6d}->{new_ab:<6d}')
    print(f'{"":22s} endpoint old(TabB) '
          f'{json.load(open(f"{DATA}/tableb_endpoint_quality.json"))["tableB_per_arm"][name]}')
    print(f'{"":22s} endpoint real-driven {real_quality[name]}')

# ------------------------------------------------------- fault sweep ---------
import sqlite3
s30.con = sqlite3.connect(s30.DB)          # s30 closed its handle at EOF
s30.cur = s30.con.cursor()
s30.fallback = real_fallback
s30.FB.clear()
s30.ENDPOINT_DETAIL.clear()
PS, R, CONTRACTS = s30.PS, s30.R, s30.CONTRACTS
cells = {}
UNL_KEYS = set()
for contract in CONTRACTS:
    for arm in ('none', 'random', 'fixed', 'full'):
        for p in PS:
            fb, sv, ab, corr, unl, wrong = [], [], [], [], [], []
            for r in range(R):
                seed = 1000 + r
                acc, _ = s30.run_loop(contract, p, arm, seed)
                fb.append(acc['fallback'])
                sv.append(acc['served'])
                ab.append(acc['abstain'])
                wrong.append(acc['wrong'])
                ep = endpoint_quality(
                    s30.ENDPOINT_DETAIL[(contract, p, arm, seed)],
                    collect=UNL_KEYS)
                corr.append(ep.get('correct', 0))
                unl.append(ep.get('unlabeled', 0))
            assert max(wrong) == 0
            cells[f'{contract}/{arm}/p={p}'] = {
                'fallbacks_mean': round(sum(fb) / R, 1),
                'serves_mean': round(sum(sv) / R, 1),
                'abstains_mean': round(sum(ab) / R, 1),
                'endpoint_correct_mean': round(sum(corr) / R, 1),
                'endpoint_unlabeled_mean': round(sum(unl) / R, 2)}

old_cells = json.load(open(f'{DATA}/real_payload_chain.json'))['cells']
old_q = json.load(open(f'{DATA}/tableb_protected_loop.json'))['cells']
print('\n=== fault sweep: fallbacks template -> real; endpoint correct '
      'TabB-rescored -> real-driven ===')
for k in cells:
    print(f'{k:26s} fb {old_cells[k]["fallbacks_mean"]:>6.1f} -> '
          f'{cells[k]["fallbacks_mean"]:<6.1f}  corrT '
          f'{old_q[k]["endpoint_correct_mean_tableB"]:>6.1f} -> '
          f'{cells[k]["endpoint_correct_mean"]:<6.1f}  (unlabeled '
          f'{cells[k]["endpoint_unlabeled_mean"]})')

out = {'accounting': 'state machine driven by archived real generations '
                     '(s9 + s32 + s36); write-back certificates asserted '
                     'sets reflect the real answer; coverage assertion '
                     'enforced (no silent template fallback)',
       'coverage_misses': n_cover_miss[0],
       'erasure_free': {'per_tick': real_rows, 'endpoint_quality':
                        real_quality},
       'fault_cells': cells,
       'unlabeled_endpoint_keys': sorted(UNL_KEYS),
       'notes': ['endpoint deliveries from gap-fill keys without human '
                 'labels are counted as unlabeled, never scored']}
json.dump(out, open(f'{DATA}/real_driven_loop.json', 'w'),
          ensure_ascii=False, indent=1)
print(f'\ndistinct unlabeled endpoint-delivered keys: {len(UNL_KEYS)}')
print('\nsaved data/real_driven_loop.json; coverage misses:',
      n_cover_miss[0])

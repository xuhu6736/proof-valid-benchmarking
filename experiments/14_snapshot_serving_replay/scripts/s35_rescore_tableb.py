#!/usr/bin/env python3
"""s35_rescore_tableb.py -- re-score the s30 real-payload-chain erasure
sweep under the Table-B matched accounting.

s30 archived endpoint_correct under the generous accounting (template
compositions count as correct).  After s32-s34 every template delivery has
a real regeneration with a human answer-level label, so each cell's
endpoint correctness is recomputed with the SAME label standard as
Table B (data/tableb_endpoint_quality.json):

  served, base origin        -> drift override, else human A4 label
  served, intermediate origin-> adjudicated Table-B label of (pid, origin)
  regenerated at 20260801    -> adjudicated s9 regeneration label where it
                                exists, else the Table-B label of (pid, T)
  regenerated at mid tick    -> adjudicated Table-B label of (pid, tick)
  the single accept-fail key -> 0 (counted as an abstention)

Method: import s30 with S30_NO_WRITE=1 (archive untouched), verify the
unpatched rescore reproduces the archived endpoint_correct_mean exactly
(CRN determinism + refactor sanity), then monkeypatch s30.score_endpoint
and re-run the identical sweep (same seeds, same fault plans).

Output: data/tableb_protected_loop.json
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

# ---- sanity: unpatched sweep reproduces the archived cells exactly --------
arch = json.load(open(f'{DATA}/real_payload_chain.json'))
mismatch = [k for k in arch['cells']
            if abs(arch['cells'][k]['endpoint_correct_mean']
                   - s30.results[k]['endpoint_correct_mean']) > 1e-9]
assert not mismatch, f'refactor changed scoring: {mismatch}'
print('sanity OK: unpatched re-run reproduces archived endpoint means.')

# ---- Table-B scorer ---------------------------------------------------------
tbl = {k: v['R1_adjudicated']
       for k, v in json.load(open(f'{DATA}/tableb_labels.json')).items()}
tbregen = json.load(open(f'{DATA}/tableb_regeneration.json'))
regen_lab, lab, DRIFT = s30.regen_lab, s30.lab, s30.DRIFT
T, BASE = s30.T, s30.SNAPS[0]
A4MAP = {'正确': 1, '错误': 0, '不完整': 0, '无法判断': 0}
n_accept_fail = 0

def tb_ok(pid, tick):
    global n_accept_fail
    r = tbregen.get(f'{pid}|{tick}')
    if r is None or not r['regen_ok']:
        n_accept_fail += 1
        return 0
    return int(tbl.get(f'{pid}|{tick}', '无法判断') == '正确')

def score_tableb(endpoint):
    correct = 0
    for pid, (how, origin) in endpoint.items():
        if how == 'abstain':
            continue
        if how == 'regenerated':
            if origin == T and pid in regen_lab:
                correct += regen_lab[pid]['R1_adjudicated'] == '正确'
            else:
                correct += tb_ok(pid, origin)
        elif pid in DRIFT:
            pass
        elif origin == BASE:
            correct += A4MAP.get(lab[pid].get('A4', '无法判断'), 0)
        else:
            correct += tb_ok(pid, origin)
    return correct

s30.score_endpoint = score_tableb

# s30 closed its sqlite handle at module end; reopen for the re-sweep
import sqlite3
s30.con = sqlite3.connect(s30.DB)
s30.cur = s30.con.cursor()

# ---- identical sweep, new scoring -------------------------------------------
PS, R, CONTRACTS, ARMS = s30.PS, s30.R, s30.CONTRACTS, ('none', 'random',
                                                        'fixed', 'full')
cells = {}
for contract in CONTRACTS:
    for arm in ARMS:
        for p in PS:
            vals = []
            for r in range(R):
                acc, _ = s30.run_loop(contract, p, arm, 1000 + r)
                vals.append(acc['endpoint_correct'])
            cells[f'{contract}/{arm}/p={p}'] = {
                'endpoint_correct_mean_tableB': round(sum(vals) / R, 1),
                'min': min(vals), 'max': max(vals)}

qual = json.load(open(f'{DATA}/tableb_endpoint_quality.json'))
ref = {'asserted': qual['tableB_per_arm']['pvsb-asserted-loop']['correct'],
       'strict': qual['tableB_per_arm']['pvsb-strict-loop']['correct']}
print('erasure-free Table-B references:', ref)
for contract in CONTRACTS:
    deltas = [round(v['endpoint_correct_mean_tableB'] - ref[contract], 1)
              for k, v in cells.items() if k.startswith(contract)]
    print(f'{contract}: cell means {[cells[k]["endpoint_correct_mean_tableB"] for k in cells if k.startswith(contract)]}')
    print(f'  deltas vs erasure-free {ref[contract]}: min {min(deltas)}, '
          f'max {max(deltas)}')
    cells[f'{contract}/_delta_vs_erasure_free'] = {'min': min(deltas),
                                                   'max': max(deltas)}

out = {'accounting': 'Table-B matched accounting (s32-s34 human labels for '
                     'every formerly template-composed delivery; the single '
                     'accept-fail key counts as an abstention)',
       'erasure_free_reference_tableB': ref,
       'accept_fail_scored_as_abstain_events': n_accept_fail,
       'cells': cells,
       'note': 'same seeds/fault plans as s30; mechanism columns (fallbacks, '
               'violating serves, restores, guards) are unchanged and remain '
               'archived in real_payload_chain.json'}
json.dump(out, open(f'{DATA}/tableb_protected_loop.json', 'w'),
          ensure_ascii=False, indent=1)
print('saved data/tableb_protected_loop.json')

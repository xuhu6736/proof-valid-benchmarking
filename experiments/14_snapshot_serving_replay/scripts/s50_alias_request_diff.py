#!/usr/bin/env python3
"""s50_alias_request_diff.py -- request-level decision diff for the crossed
gate-by-alias control (s49).

For each gate, compares per-(tick, request) routing decisions between the
alias-off and alias-on configurations.  Establishes whether the net count
changes (-2, 0, 0) reported by s49 correspond to few actual request-level
changes or mask offsetting routing flips.

Result: asserted gate -- exactly 2 changed cells, both
regenerated -> served, same request (pid 51) at ticks 20260701 and
20260801; overlap-proxy and equality-proxy -- zero changed cells.

Output: data/gate_alias_request_diff.json + console summary.
"""
import importlib.util, io, contextlib, json, os

os.environ['S30_NO_WRITE'] = '1'
HERE = os.path.dirname(os.path.abspath(__file__))

spec = importlib.util.spec_from_file_location(
    's49', os.path.join(HERE, 's49_gate_alias_cross.py'))
m = importlib.util.module_from_spec(spec)
with contextlib.redirect_stdout(io.StringIO()):
    spec.loader.exec_module(m)
res = m.results


def cells(rows):
    out = {}
    for r in rows:
        for pid, v in r['_detail'].items():
            out[(r['tick'], pid)] = v[0]
    return out


report = {}
for gate in ('asserted', 'overlap-proxy', 'equality-proxy'):
    off = cells(res[f'{gate}|alias-off'])
    on = cells(res[f'{gate}|alias-on'])
    assert off.keys() == on.keys()
    diff = {k: (off[k], on[k]) for k in off if off[k] != on[k]}
    report[gate] = {
        'n_changed_cells': len(diff),
        'changes': {f'{k[0]}|{k[1]}': list(v)
                    for k, v in sorted(diff.items())}}

print(json.dumps(report, indent=1))
out_path = os.path.join(HERE, '..', 'data', 'gate_alias_request_diff.json')
json.dump(report, open(out_path, 'w'), indent=1)
print('saved data/gate_alias_request_diff.json')

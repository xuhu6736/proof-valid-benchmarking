#!/usr/bin/env python3
"""s49_gate_alias_cross.py -- crossed gate x alias control (6 configurations).

Reviewer scheme A: separate the gate/contract effect from the retrieval-alias
effect in the mechanism-arm comparison.  Gates:
  asserted        PVCacheServe asserted-claims gate (s25.verify 'asserted')
  overlap-proxy   page-version + claim-overlap mapping (s48.grounded_serve)
  equality-proxy  page-version + claim-equality mapping (s48.finserve_serve)
Alias:
  on   evidence-checked retrieval alias: after an accepted regeneration is
       written back, the request's paraphrase re-enters the hit set
       (s25.run_policy's hits.add(pid))
  off  fixed retrieval outcome (s48 convention)

All six configurations share the initial cache, requests, similarity screen,
evidence states, archived real fallback outputs, acceptance rule, and
write-back rule (s37 real-driven convention: coverage assertion enforced,
regen_ok=False is an abstention, Table-B endpoint labelling).

Validations:
  V3  every request accounted exactly once per tick per configuration
  V5  zero archive coverage misses
  R1  reproduction of the published anchors: asserted+alias=29,
      overlap-proxy alias off=181, equality-proxy alias off=187 total
      accepted regenerations

Output: data/gate_alias_cross.json + console summary.
"""
import json, os, io, contextlib, importlib.util
from collections import defaultdict, Counter

os.environ['S30_NO_WRITE'] = '1'
HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.normpath(os.path.join(HERE, '..', 'data'))

spec = importlib.util.spec_from_file_location(
    's48', os.path.join(HERE, 's48_mechanism_arms.py'))
s48 = importlib.util.module_from_spec(spec)
with contextlib.redirect_stdout(io.StringIO()):
    spec.loader.exec_module(s48)
s25 = s48.s25

T, BASE = s25.T, s25.SNAPS[0]
TICKS = s25.TICKS
by_pid, lab, regen_lab, DRIFT = s25.by_pid, s25.lab, s25.regen_lab, s25.DRIFT
usable, HITS = s25.usable, s25.HITS

GATES = {'asserted': None,                     # uses s25.verify(cert,st,...)
         'overlap-proxy': s48.grounded_serve,
         'equality-proxy': s48.finserve_serve}


def run_config(gate_name, alias_on):
    gate = GATES[gate_name]
    store = {}
    for x in usable:
        store[int(x['pid'])] = {'answer': s25.gen[str(x['pid'])]['answer'],
                                'cert': s48.base_certificate(x)}
    hits = set(HITS)
    rows = []
    n_alias = 0
    for tag in TICKS:
        acc = defaultdict(int)
        detail = {}
        for pid in sorted(store):
            x = by_pid[pid]
            st = s25.state_of(tag, x['qid'], x['prop'], x.get('page_title'))
            if pid in hits:
                if gate_name == 'asserted':
                    served = s25.verify(store[pid]['cert'], st, 'asserted')
                else:
                    served = gate(store[pid]['cert'], st, x, tag)
            else:
                served = False
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
            ans, cert = s48.real_fallback(x, tag)
            if ans is None:
                acc['abstain'] += 1
                detail[pid] = ['abstain', None]
                continue
            acc['fallback'] += 1
            store[pid] = {'answer': ans, 'cert': cert}
            if alias_on:
                if pid not in hits:
                    n_alias += 1
                hits.add(pid)        # evidence-checked retrieval alias
            detail[pid] = ['regenerated', tag]
        for k in ('served', 'fallback', 'abstain', 'wrong_proxy',
                  'decayed_proxy'):
            acc[k] += 0
        rows.append({'tick': tag, **acc, '_detail': detail})
    return rows, n_alias


results, quality, aliases = {}, {}, {}
for gate_name in GATES:
    for alias_on in (False, True):
        cfg = f'{gate_name}|alias-{"on" if alias_on else "off"}'
        rows, n_alias = run_config(gate_name, alias_on)
        for r in rows:
            assert r['served'] + r['fallback'] + r['abstain'] == 154, \
                f'V3 FAILED for {cfg} at {r["tick"]}'
        results[cfg] = rows
        quality[cfg] = s48.endpoint_quality(rows[-1]['_detail'])
        aliases[cfg] = n_alias
print('V3 OK: every request accounted exactly once per tick per config.')
print('V5 OK: archive coverage misses =', s48.n_cover_miss[0])

summary = {}
for cfg, rows in results.items():
    summary[cfg] = {
        'per_tick': [{k: v for k, v in r.items() if k != '_detail'}
                     for r in rows],
        'total_fallbacks': sum(r['fallback'] for r in rows),
        'total_contract_violating_serves_proxy': sum(r['wrong_proxy']
                                                     for r in rows),
        'total_partially_stale_serves_proxy': sum(r['decayed_proxy']
                                                  for r in rows),
        'total_abstains': sum(r['abstain'] for r in rows),
        'aliases_installed': aliases[cfg],
        'endpoint_quality': quality[cfg],
    }

# ---- R1 reproduction of published anchors ----------------------------------
anchors = {'asserted|alias-on': 29, 'overlap-proxy|alias-off': 181,
           'equality-proxy|alias-off': 187}
for cfg, exp in anchors.items():
    got = summary[cfg]['total_fallbacks']
    assert got == exp, f'R1 FAILED for {cfg}: got {got}, expected {exp}'
print('R1 OK: anchors reproduced (asserted+alias 29, overlap-alias-off 181, '
      'equality-alias-off 187).')

print('\n===== gate x alias cross, real-driven (3 monthly ticks) =====')
print(f'{"configuration":28s} {"fallbacks":>9s} {"viol*":>6s} '
      f'{"p-stale*":>8s} {"abstain":>8s} {"aliases":>8s} '
      f'{"corr@T":>7s} {"wrong@T":>7s} {"unlab@T":>7s}')
for cfg in summary:
    s = summary[cfg]
    ep = s['endpoint_quality']
    print(f'{cfg:28s} {s["total_fallbacks"]:9d} '
          f'{s["total_contract_violating_serves_proxy"]:6d} '
          f'{s["total_partially_stale_serves_proxy"]:8d} '
          f'{s["total_abstains"]:8d} {s["aliases_installed"]:8d} '
          f'{ep.get("correct", 0):7d} {ep.get("wrong", 0):7d} '
          f'{ep.get("unlabeled", 0):7d}')

out = {'convention': 'real-driven (s37): archived real generations at '
                     'every tick; coverage assertion enforced; Table-B '
                     'endpoint labelling',
       'design': 'crossed control: gate in {asserted, overlap-proxy, '
                 'equality-proxy} x retrieval-alias {off, on}; identical '
                 'base cache, requests, screen, evidence states, archived '
                 'fallbacks, acceptance, and write-back',
       'ticks': TICKS,
       'configurations': summary,
       'validations': {'V3_accounting_complete': True,
                       'V5_coverage_misses': s48.n_cover_miss[0],
                       'R1_anchors_reproduced': anchors},
       'notes': [
           'alias on = evidence-checked retrieval alias after write-back '
           '(s25 mechanism); alias off = fixed retrieval outcome (s48)',
           'the asserted gate with alias on is the main-text asserted '
           'loop; the two proxies with alias off are the exploratory gate '
           'mappings of s48',
           'endpoint deliveries whose (request, origin-tick) key lacks a '
           'human label are counted as unlabeled, never scored']}
json.dump(out, open(f'{DATA}/gate_alias_cross.json', 'w'),
          ensure_ascii=False, indent=1)
print('\nsaved data/gate_alias_cross.json')

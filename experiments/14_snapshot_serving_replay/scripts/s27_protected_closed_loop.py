#!/usr/bin/env python3
"""
s27_protected_closed_loop.py -- P2 Experiment C: protection recovery inside
the runnable closed-loop prototype.

Extends s25 with a read-erasure observation layer: at each monthly tick, the
gate's read of every certificate item (claim key, sentence id) is lost
i.i.d. with probability p.  Audited restoration is transparent and
non-laundering: a lost read of an item held in the protection cache S is
restored to the item's TRUE current availability (a genuinely decayed item
is never resurrected); an unrestored loss reads as unavailable and the
conjunctive contract declines (fail-closed).  Declines trigger the same
fallback + write-back as in s25, so false declines cost real regeneration
executions and churn certificates into unprotected new items.

Variants per contract (asserted, strict) and p in {0.05, 0.10, 0.20}:
  none    -- no protection cache
  random  -- model R: each item protected i.i.d. with probability 0.90
  fixed   -- model F: the s20 static global set S, greedy coverage fill at
             90% of total designated bytes, computed once from the t0 base
             certificates (write-back items enter unprotected)

Common random numbers via hashed uniforms make the erasure draws identical
across variants.  R=200 realizations per cell.

Validations:
  V1  p=0 reproduces the s25 erasure-free loop accounting exactly
  V2  full protection (S = all items) at p=0.20 also reproduces it
  V3  fixed-S byte cost matches the s20 archived 90%-budget row
  V4  wrong serves stay at zero in every realization (restoration never
      launders staleness)

Outputs: data/protected_closed_loop.json + console summary.
"""

import json, os, io, random, hashlib, importlib.util, contextlib
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.normpath(os.path.join(HERE, '..', 'data'))

spec = importlib.util.spec_from_file_location(
    's25', os.path.join(HERE, 's25_closed_loop_system.py'))
s25 = importlib.util.module_from_spec(spec)
with contextlib.redirect_stdout(io.StringIO()):
    spec.loader.exec_module(s25)

by_pid, lab, HITS, DRIFT = s25.by_pid, s25.lab, s25.HITS, s25.DRIFT
TICKS, T = s25.TICKS, s25.T
state_of, verify = s25.state_of, s25.verify
base_certificate, fallback = s25.base_certificate, s25.fallback
gen = s25.gen
regen_lab = s25.regen_lab
A4MAP = {'正确': 1, '错误': 0, '不完整': 0, '无法判断': 0}

PS = (0.05, 0.10, 0.20)
R = int(os.environ.get('S27_R', '200'))
BUDGET_FRac = 0.90

# deterministic caches ------------------------------------------------------
ST = {}           # (pid, tag) -> evidence state
FB = {}           # (pid, tag) -> (answer, cert) fallback outcome
BASE_CERT = {}    # pid -> base certificate


def st_of(pid, tag):
    key = (pid, tag)
    if key not in ST:
        x = by_pid[pid]
        ST[key] = state_of(tag, x['qid'], x['prop'], x.get('page_title'))
    return ST[key]


def fb_of(pid, tag):
    key = (pid, tag)
    if key not in FB:
        FB[key] = fallback(by_pid[pid], tag)
    return FB[key]


def base_cert_of(pid):
    if pid not in BASE_CERT:
        BASE_CERT[pid] = base_certificate(by_pid[pid])
    return BASE_CERT[pid]


def u01(*parts):
    h = hashlib.sha1('|'.join(map(str, parts)).encode()).hexdigest()
    return int(h[:12], 16) / float(16 ** 12)


# ---------------- protection universe from the t0 base certificates --------
freq, nbytes = {}, {}
for pid in HITS:
    cert = base_certificate(by_pid[pid])
    for k in cert['claims']:
        freq[k] = freq.get(k, 0) + 1
        nbytes.setdefault(k, len(str(k).encode()))
    for s in by_pid[pid]['certificate']['sentences']:
        freq[s['id']] = freq.get(s['id'], 0) + 1
        nbytes.setdefault(s['id'], len(s['text'].encode()))
items = sorted(freq, key=lambda k: (-freq[k], k))
total_bytes = sum(nbytes[k] for k in items)
budget = int(BUDGET_FRac * total_bytes)
S_fixed, used = set(), 0
for k in items:                       # freq-sorted greedy fill (as s20)
    if used + nbytes[k] <= budget:
        S_fixed.add(k)
        used += nbytes[k]
print(f'protection universe: {len(items)} items, {total_bytes} bytes; '
      f'fixed S: {len(S_fixed)} items, {used} bytes (budget {budget})')

# V3: match the s20 archived 90%-budget byte usage
s20 = json.load(open(f'{DATA}/fixed_budget_protection.json'))
row90 = [r for r in s20['rows'] if abs(r['c'] - 0.90) < 1e-9][0]
assert used == row90['bytes_used'], \
    f"V3 FAILED: {used} != s20's {row90['bytes_used']}"
print(f"V3 OK: fixed-S bytes match s20's archived 90%-budget row "
      f"({used}).")


def protected_set(mode, seed):
    if mode == 'none':
        return set()
    if mode == 'fixed':
        return S_fixed
    if mode == 'random':
        return {k for k in items if u01('prot', seed, k) < BUDGET_FRac}
    if mode == 'full':
        return None                         # protect-all sentinel
    raise ValueError(mode)


def observed_verify(cert, st, contract, S, p, seed, tag, pid):
    """contract verdict over erasure-perturbed reads; restoration is
    transparent (restores the true current availability, never revives a
    genuinely dead item)."""
    def alive(key, true_alive, salt):
        if not true_alive:
            return False
        if u01('loss', seed, tag, pid, salt, key) < p:
            return S is None or key in S     # S=None: protect-all (V2)
        return True
    cobs = {k: alive(k, k in st['truthy'], 'c') for k in cert['claims']}
    sobs = {s: alive(s, s in st['sids'], 's') for s in cert['sentences']}
    if contract == 'claims':
        return all(cobs.values())
    if contract == 'asserted':
        return all(cobs[k] for k in cert['asserted'])
    ns = len(cert['sentences'])
    if contract == 'strict':
        return all(cobs.values()) and all(sobs.values())
    if contract == 'majority':
        return all(cobs.values()) and (
            ns == 0 or sum(sobs.values()) * 2 > ns)
    raise ValueError(contract)


def run_loop(contract, p, mode, seed):
    S = protected_set(mode, seed)
    store = {int(x['pid']): {'answer': gen[str(x['pid'])]['answer'],
                             'cert': base_cert_of(int(x['pid']))}
             for x in s25.usable}
    hits = set(HITS)
    acc = defaultdict(int)
    endpoint = {}
    for tag in TICKS:
        for pid in sorted(store):
            st = st_of(pid, tag)
            served = False
            false_decline = False
            if pid in hits:
                served = observed_verify(store[pid]['cert'], st, contract,
                                         S, p, seed, tag, pid)
                if served and tag == T and pid in DRIFT:
                    acc['wrong'] += 1
                if not served and verify(store[pid]['cert'], st, contract):
                    false_decline = True     # intact reads would have served
            if served:
                acc['served'] += 1
                endpoint[pid] = ('served',
                                 store[pid]['cert']['issued_at'])
                continue
            ans, cert = fb_of(pid, tag)
            if ans is None:
                acc['abstain'] += 1
                endpoint[pid] = ('abstain', None)
                continue
            acc['fallback'] += 1
            if false_decline:
                acc['false_decline'] += 1
            store[pid] = {'answer': ans, 'cert': cert}
            hits.add(pid)
            endpoint[pid] = ('regenerated', tag)
    # endpoint correct deliveries (published accounting, as s26)
    correct = 0
    for pid, (how, origin) in endpoint.items():
        if how == 'abstain':
            continue
        if how == 'regenerated':
            if origin == T and pid in regen_lab:
                correct += regen_lab[pid]['R1_adjudicated'] == '正确'
            else:
                correct += 1
        elif pid in DRIFT:
            pass
        elif origin == s25.SNAPS[0]:
            correct += A4MAP.get(lab[pid].get('A4', '无法判断'), 0)
        else:
            correct += 1
    acc['endpoint_correct'] = correct
    return acc


# ------------------------------ V1 / V2 ------------------------------------
print('V1/V2: erasure-free reproduction checks ...')
expect_fb = {'asserted': 29, 'strict': 53}
for contract in ('asserted', 'strict'):
    a = run_loop(contract, 0.0, 'none', seed=0)
    b = run_loop(contract, 0.20, 'full', seed=0)
    for name, r in (('p=0', a), ('p=0.20+full', b)):
        assert r['fallback'] == expect_fb[contract] and r['wrong'] == 0, \
            f'V1/V2 FAILED for {contract} {name}: {dict(r)}'
print('  V1 OK: p=0 reproduces the s25 loop (asserted 29, strict 53 '
      'fallbacks, zero wrong).')
print('  V2 OK: full protection at p=0.20 is identical (transparent, '
      'non-laundering restoration).')

# ------------------------------ main sweep ---------------------------------
results = {}
for contract in ('asserted', 'strict'):
    for mode in ('none', 'random', 'fixed'):
        for p in PS:
            cells = [run_loop(contract, p, mode, seed=1000 + r)
                     for r in range(R)]
            agg = {
                'fallbacks_mean': round(sum(c['fallback'] for c in cells)
                                        / R, 1),
                'fallbacks_p95': sorted(c['fallback'] for c in cells)
                    [int(0.95 * R)],
                'false_declines_mean': round(
                    sum(c['false_decline'] for c in cells) / R, 1),
                'serves_mean': round(sum(c['served'] for c in cells) / R, 1),
                'abstains_mean': round(sum(c['abstain'] for c in cells)
                                       / R, 1),
                'endpoint_correct_mean': round(
                    sum(c['endpoint_correct'] for c in cells) / R, 1),
                'wrong_max': max(c['wrong'] for c in cells),
            }
            assert agg['wrong_max'] == 0, 'V4 FAILED: wrong serve detected'
            results[f'{contract}/{mode}/p={p}'] = agg
print('V4 OK: zero wrong serves in every realization of every cell.')

print(f'\n==== protection in the closed loop (means over {R} erasure '
      f'realizations, 3 ticks) ====')
print(f'{"cell":26s} {"fallbacks":>9s} {"p95":>5s} {"falseDec":>8s} '
      f'{"serves":>7s} {"corr@T":>7s}')
for k, a in results.items():
    print(f'{k:26s} {a["fallbacks_mean"]:9.1f} {a["fallbacks_p95"]:5d} '
          f'{a["false_declines_mean"]:8.1f} {a["serves_mean"]:7.1f} '
          f'{a["endpoint_correct_mean"]:7.1f}')

out = {
    'model': 'read-erasure overlay on the s25 closed loop; audited '
             'restoration is transparent (restores true availability, '
             'never revives a genuinely decayed item); unrestored loss '
             'reads as unavailable and the conjunctive contract declines',
    'variants': ['none', 'random(0.90 per item)', 'fixed(90% bytes, s20 '
                 'greedy coverage set, static from t0)'],
    'realizations': R, 'ticks': TICKS,
    'protection_set': {'items': len(items), 'total_bytes': total_bytes,
                       'fixed_items': len(S_fixed), 'fixed_bytes': used},
    'validations': {'V1_p0_reproduces_s25': True,
                    'V2_full_protection_identity': True,
                    'V3_fixed_S_bytes_match_s20': used,
                    'V4_zero_wrong_all_cells': True},
    'erasure_free_reference': {'asserted_fallbacks': 29,
                               'strict_fallbacks': 53},
    'cells': results}
json.dump(out, open(f'{DATA}/protected_closed_loop.json', 'w'),
          ensure_ascii=False, indent=1)
print('\nsaved data/protected_closed_loop.json')

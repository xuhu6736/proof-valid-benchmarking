#!/usr/bin/env python3
"""
s30_real_payload_chain.py -- P2: protection on a real payload execution
chain (reviewer acceptance criteria 2 and 3: paired attribution of the
increment, and full maintenance-cost accounting).

Replaces the idealized overlay of s27 ("a lost read is transparently
restored to its true availability") with an executable chain over real
stores (SQLite, data/payload_stores.sqlite):

  source-side eligibility registry (per-tick status + version, keyed by
  premise identity: qid|prop|value for claims, page|sentence-id for
  sentences)
        |  read fails (missing / corrupted / transiently unreadable)
        v
  protection payload store: read the backup payload        -- timed
        v
  verify the payload fingerprint (sha1)                    -- timed
        v
  query the current eligibility registry (status, version)
        v
  eligible -> the payload re-enters as the premise (restore)
  policy_withdrawn / version_incompatible -> DECLINE (guard)

The two required negative cases are exercised by real data, not by
injected flags: a protected claim whose value left the truthy set is a
payload-intact-but-policy-withdrawn read; a protected sentence that left
the current page version is a payload-intact-but-version-incompatible
read.  Both verify their fingerprints and are still declined.

Arms (same generations, state updates, eligibility rules, fault plan via
common random numbers, and the same ACTUAL byte budget B = the greedy
set's byte usage):

  none    -- unprotected exact-dependency loop
  random  -- one feasible random set per realization (shuffled fill up to
             B bytes, drawn then fixed) -- replaces s27's Bernoulli(0.90)
             per-item arm, which did not respect a realized byte cap
  fixed   -- the s20/s27 greedy coverage set (static from t0)
  full    -- protect-everything reference with read-any redundancy

The protection universe and all byte accounting are the t0 base-period
certificate items exactly as in s20/s27 (temporal protocol: S is chosen
once from base-period certificates; write-back items enter unprotected
in the budgeted arms).  Registry and stores additionally cover
post-build (write-back) items so the loop can execute; the full arm
restores those under its read-any reference assumption.

Fault plan (CRN, identical across arms given the seed): each source read
of an eligible item fails i.i.d. with probability p; each backup read in
a budgeted arm fails i.i.d. with probability p (source and copy both
unreadable).  The full arm assumes a redundant copy tier (read-any), so
it is an upper-bound reference, not a deployable configuration.

Cost accounting (C_PVSB = C_state + C_certificate + C_gate + C_restore
+ C_fallback), measured on the real stores:
  C_state        eligibility-registry build + per-tick snapshot loads
  C_certificate  certificate store: 154 base + 29 write-back canonical
                 serializations (timed) and their bytes
  C_gate         per-request gate evaluation time (reference realization)
  C_restore      timed backup-read + fingerprint operations x counts
  C_fallback     regeneration executions (LLM calls only at the 20260801
                 endpoint; intermediate repairs are template compositions)
Break-even is reported parametrically: executions avoided vs incremental
bytes and measured CPU, so the condition "saved regeneration cost >
extra maintenance and protection cost" can be evaluated at any price
point.

Validations:
  V1  p=0, none arm reproduces the s25 erasure-free loop exactly
  V2  full protection at p=0.20 reproduces it too (restores execute on
      the real chain; verdicts unchanged)
  V3  greedy-set bytes match the s20 archived 90%-budget row
  V4  zero wrong serves in every realization of every cell
  V5  guard paths fire (>0) and every fingerprint-verified but
      withdrawn/incompatible payload is declined (structural: the chain
      returns unavailable after the registry check)
  V6  every random feasible set satisfies bytes <= B

Outputs: data/real_payload_chain.json + console summary.
"""

import json, os, io, time, random, sqlite3, hashlib, importlib.util, \
    contextlib
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.normpath(os.path.join(HERE, '..', 'data'))
DB = os.path.join(DATA, 'payload_stores.sqlite')

spec = importlib.util.spec_from_file_location(
    's25', os.path.join(HERE, 's25_closed_loop_system.py'))
s25 = importlib.util.module_from_spec(spec)
with contextlib.redirect_stdout(io.StringIO()):
    spec.loader.exec_module(s25)

by_pid, lab, HITS, DRIFT = s25.by_pid, s25.lab, s25.HITS, s25.DRIFT
TICKS, T, SNAPS = s25.TICKS, s25.T, s25.SNAPS
state_of, verify = s25.state_of, s25.verify
base_certificate, fallback = s25.base_certificate, s25.fallback
gen, regen_lab = s25.gen, s25.regen_lab
A4MAP = {'正确': 1, '错误': 0, '不完整': 0, '无法判断': 0}

PS = (0.05, 0.10, 0.20)
R = int(os.environ.get('S30_R', '200'))
BUDGET_FRAC = 0.90
CONTRACTS = ('asserted', 'strict')

# ---------------- deterministic caches (as s27) -----------------------------
ST, FB, BASE_CERT = {}, {}, {}
ENDPOINT_DETAIL = {}          # (contract, p, arm, seed) -> {pid: (how, origin)}

def st_of(pid, tag):
    if (pid, tag) not in ST:
        x = by_pid[pid]
        ST[(pid, tag)] = state_of(tag, x['qid'], x['prop'],
                                  x.get('page_title'))
    return ST[(pid, tag)]

def fb_of(pid, tag):
    if (pid, tag) not in FB:
        FB[(pid, tag)] = fallback(by_pid[pid], tag)
    return FB[(pid, tag)]

def base_cert_of(pid):
    if pid not in BASE_CERT:
        BASE_CERT[pid] = base_certificate(by_pid[pid])
    return BASE_CERT[pid]

def u01(*parts):
    h = hashlib.sha1('|'.join(map(str, parts)).encode()).hexdigest()
    return int(h[:12], 16) / float(16 ** 12)

def claim_cid(x, k):
    return f"{x['qid']}|{x['prop']}|{k}"

def sent_cid(x, s):
    return f"{x.get('page_title')}|{s}"

# ---------------- protection universe (identical to s27 bytes) --------------
freq, nbytes, kind_of = {}, {}, {}
for pid in HITS:
    cert = base_cert_of(pid)
    for k in cert['claims']:
        freq[k] = freq.get(k, 0) + 1
        nbytes.setdefault(k, len(str(k).encode()))
        kind_of[k] = 'claim'
    for s in by_pid[pid]['certificate']['sentences']:
        freq[s['id']] = freq.get(s['id'], 0) + 1
        nbytes.setdefault(s['id'], len(s['text'].encode()))
        kind_of[s['id']] = 'sentence'
items = sorted(freq, key=lambda k: (-freq[k], k))
total_bytes = sum(nbytes[k] for k in items)
budget = int(BUDGET_FRAC * total_bytes)
S_fixed, used = set(), 0
for k in items:
    if used + nbytes[k] <= budget:
        S_fixed.add(k)
        used += nbytes[k]
B = used
print(f'universe: {len(items)} items, {total_bytes} bytes; greedy set '
      f'{len(S_fixed)} items, {used} bytes -> actual byte budget B={B}')

s20 = json.load(open(f'{DATA}/fixed_budget_protection.json'))
row90 = [r for r in s20['rows'] if abs(r['c'] - 0.90) < 1e-9][0]
assert used == row90['bytes_used'], 'V3 FAILED: greedy bytes != s20 archive'
print(f'V3 OK: greedy bytes match s20 archived 90% row ({used}).')

# ------------- replay both erasure-free loops: every cert ever installed ----
def replay(contract):
    store = {int(x['pid']): {'answer': gen[str(x['pid'])]['answer'],
                             'cert': base_cert_of(int(x['pid']))}
             for x in s25.usable}
    installed = [(pid, c['cert']) for pid, c in store.items()]
    hits = set(HITS)
    for tag in TICKS:
        for pid in sorted(store):
            if pid in hits and verify(store[pid]['cert'], st_of(pid, tag),
                                      contract):
                continue
            ans, cert = fb_of(pid, tag)
            if ans is None:
                continue
            store[pid] = {'answer': ans, 'cert': cert}
            hits.add(pid)
            installed.append((pid, cert))
    return installed

installed_asserted = replay('asserted')
installed_strict = replay('strict')
print(f'installed certificates: {len(installed_asserted)} (asserted loop), '
      f'{len(installed_strict)} (strict loop).')
assert len(installed_asserted) == 183, 'asserted loop install count drifted'

# Every POSSIBLY installed certificate: base certs plus, for every (pid,
# tick), the certificate a fallback WOULD install there (deterministic via
# fb_of).  Protected runs decline more often than the erasure-free replay,
# so their write-back sequence differs; the registry and payload stores
# must cover this full possibility set or mid-run write-back items would
# read as unregistered (spurious declines).
all_installed = list(installed_asserted)
_seen = {(pid, json.dumps(cert, sort_keys=True))
         for pid, cert in all_installed}
for _pid in sorted(by_pid):
    for _tag in TICKS:
        _ans, _cert = fb_of(_pid, _tag)
        if _cert is None:
            continue
        _key = (_pid, json.dumps(_cert, sort_keys=True))
        if _key not in _seen:
            _seen.add(_key)
            all_installed.append((_pid, _cert))
print(f'possible-install universe: {len(all_installed)} certificates '
      f'(incl. every fallback-installable write-back).')

# ---------------- build the real stores (C_state, C_certificate) ------------
if os.path.exists(DB):
    os.remove(DB)
con = sqlite3.connect(DB)
cur = con.cursor()
cur.executescript('''
CREATE TABLE source_payload(item_id TEXT PRIMARY KEY, payload BLOB,
                            fingerprint TEXT, kind TEXT);
CREATE TABLE protection_payload(item_id TEXT PRIMARY KEY, payload BLOB,
                                fingerprint TEXT);
CREATE TABLE eligibility(tick TEXT, premise_id TEXT, status TEXT,
                         version TEXT, PRIMARY KEY(tick, premise_id));
CREATE TABLE certificates(pid INT, origin TEXT, cert_json TEXT);
''')

# payloads: t0 premise bytes (claims: value-key bytes; sentences: sentence
# text bytes -- the same byte accounting as s20/s27).  Post-build items get
# a placeholder row so the full (read-any) reference arm can execute reads;
# they are never in a budgeted set and add no bytes to any budget figure.
text_of = {}
for pid in HITS:
    for s in by_pid[pid]['certificate']['sentences']:
        text_of[s['id']] = s['text']
for k in items:
    payload = str(k).encode() if kind_of[k] == 'claim' \
        else text_of[k].encode()
    fp = hashlib.sha1(payload).hexdigest()
    cur.execute('INSERT INTO source_payload VALUES (?,?,?,?)',
                (k, payload, fp, kind_of[k]))
    # the protection store physically holds the full-copy tier; each
    # budgeted arm CONSULTS only its selected set (bytes are accounted
    # per arm from the budget figures, not from the store file)
    cur.execute('INSERT INTO protection_payload VALUES (?,?,?)',
                (k, payload, fp))
wb_items = set()
for pid, cert in all_installed:
    for k in cert['claims']:
        if k not in nbytes:
            wb_items.add(('claim', k))
    for s in cert['sentences']:
        if s not in nbytes:
            wb_items.add(('sentence', s))
for kind, k in sorted(wb_items):
    payload = f'wb:{k}'.encode()
    cur.execute('INSERT OR IGNORE INTO source_payload VALUES (?,?,?,?)',
                (k, payload, hashlib.sha1(payload).hexdigest(), kind))
con.commit()

# eligibility registry over every (context, item) premise ever gated
t0 = time.perf_counter()
premise_items = set()
for pid, cert in all_installed:
    x = by_pid[pid]
    for k in cert['claims']:
        premise_items.add((claim_cid(x, k), k, 'claim', pid))
    for s in cert['sentences']:
        premise_items.add((sent_cid(x, s), s, 'sentence', pid))
for tag in SNAPS:
    for cid, k, kind, pid in premise_items:
        st = st_of(pid, tag)
        pool = st['truthy'] if kind == 'claim' else st['sids']
        status = 'eligible' if k in pool else (
            'policy_withdrawn' if kind == 'claim'
            else 'version_incompatible')
        cur.execute('INSERT OR IGNORE INTO eligibility VALUES (?,?,?,?)',
                    (tag, cid, status, None))
con.commit()
rows = cur.execute('SELECT tick, premise_id, status FROM eligibility'
                   ).fetchall()
last_elig, updates = {}, []
by_tick = defaultdict(list)
for tick, cid, status in rows:
    by_tick[tick].append((cid, status))
for tag in SNAPS:
    for cid, status in by_tick[tag]:
        if status == 'eligible':
            last_elig[cid] = tag
        updates.append((last_elig.get(cid), tag, cid))
cur.executemany('UPDATE eligibility SET version=? WHERE tick=? AND '
                'premise_id=?', updates)
con.commit()
t_state_build = time.perf_counter() - t0

REG = {}
t_load = 0.0
for tag in SNAPS:
    t0 = time.perf_counter()
    REG[tag] = {cid: st for cid, st in
                cur.execute('SELECT premise_id, status FROM eligibility '
                            'WHERE tick=?', (tag,)).fetchall()}
    t_load += time.perf_counter() - t0

# certificate store: every certificate the asserted loop installs (as s29)
t0 = time.perf_counter()
cert_bytes = 0
for pid, cert in installed_asserted:
    blob = json.dumps(cert, sort_keys=True, separators=(',', ':'))
    cert_bytes += len(blob.encode())
    cur.execute('INSERT INTO certificates VALUES (?,?,?)',
                (pid, cert['origin'], blob))
con.commit()
t_cert = time.perf_counter() - t0
sqlite_bytes = os.path.getsize(DB)
print(f'stores: {len(items)} universe payloads + {len(wb_items)} post-build '
      f'placeholders; registry {len(premise_items)} premises x 4 snapshots '
      f'built in {t_state_build*1e3:.1f} ms + {t_load*1e3:.2f} ms loads; '
      f'certificates {len(installed_asserted)} rows / {cert_bytes} bytes '
      f'in {t_cert*1e3:.1f} ms; SQLite file {sqlite_bytes} bytes.')

# ---------------- the executable read chain ---------------------------------
T_RESTORE_NS = [0, 0]      # aggregate [count, ns] over timed backup reads

def backup_read(k):
    """protection-store read + fingerprint verification (timed)."""
    t0 = time.perf_counter_ns()
    row = cur.execute('SELECT payload, fingerprint FROM '
                      'protection_payload WHERE item_id=?', (k,)).fetchone()
    ns = time.perf_counter_ns() - t0
    T_RESTORE_NS[0] += 1
    T_RESTORE_NS[1] += ns
    assert row is not None
    payload, fp = row
    return hashlib.sha1(payload).hexdigest() == fp

def item_available(x, k, kind, tag, pid, salt, S, p, seed, ctr):
    """the chain: registry -> source read (fault plan) -> backup read ->
    fingerprint -> registry -> restore or fail-closed."""
    cid = claim_cid(x, k) if kind == 'claim' else sent_cid(x, k)
    status = REG[tag].get(cid)
    protected = k in nbytes and (S is None or (S is not False and k in S))
    if status != 'eligible':
        if protected:
            # guard: payload intact, fingerprint verifies, registry says
            # policy_withdrawn / version_incompatible -> decline (V5)
            assert backup_read(k)
            ctr['guard_withdrawn' if status == 'policy_withdrawn'
                else 'guard_incompat'] += 1
        return False
    if u01('loss', seed, tag, pid, salt, k) < p:        # source read fails
        if S is False:
            return False
        if S is not None:                                # budgeted arm
            if k not in S:
                return False
            if u01('closs', seed, tag, pid, salt, k) < p:
                ctr['both_unreadable'] += 1
                return False
        # restore: backup read + fingerprint + registry eligible -> premise
        if k in nbytes:
            assert backup_read(k)
        ctr['restores'] += 1
        return True
    return True

def chain_verify(x, cert, tag, pid, contract, S, p, seed, ctr):
    def ck(k):
        return item_available(x, k, 'claim', tag, pid, 'c', S, p, seed, ctr)
    def cs(s):
        return item_available(x, s, 'sentence', tag, pid, 's', S, p, seed,
                              ctr)
    if contract == 'claims':
        return all(ck(k) for k in cert['claims'])
    if contract == 'asserted':
        return all(ck(k) for k in cert['asserted'])
    ns = len(cert['sentences'])
    if contract == 'strict':
        return all(ck(k) for k in cert['claims']) and \
            all(cs(s) for s in cert['sentences'])
    if contract == 'majority':
        return all(ck(k) for k in cert['claims']) and (
            ns == 0 or sum(cs(s) for s in cert['sentences']) * 2 > ns)
    raise ValueError(contract)

# ---------------- protection sets -------------------------------------------
def random_feasible_set(seed):
    rng = random.Random(f'rset|{seed}')
    perm = items[:]
    rng.shuffle(perm)
    S, b = set(), 0
    for k in perm:
        if b + nbytes[k] <= B:
            S.add(k)
            b += nbytes[k]
    assert b <= B                                        # V6
    return S, b

def arm_set(arm, seed):
    if arm == 'none':
        return False, 0
    if arm == 'random':
        return random_feasible_set(seed)
    if arm == 'fixed':
        return S_fixed, B
    if arm == 'full':
        return None, total_bytes                         # read-any sentinel
    raise ValueError(arm)

# ---------------- the loop ----------------------------------------------------
def score_endpoint(endpoint):
    """endpoint correct deliveries under the archived (generous) accounting:
    template compositions count as correct; human labels where they exist.
    s35 monkeypatches this hook for the Table-B matched accounting."""
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
        elif origin == SNAPS[0]:
            correct += A4MAP.get(lab[pid].get('A4', '无法判断'), 0)
        else:
            correct += 1
    return correct

def run_loop(contract, p, arm, seed, keep_detail=False):
    S, _ = arm_set(arm, seed)
    store = {int(x['pid']): {'answer': gen[str(x['pid'])]['answer'],
                             'cert': base_cert_of(int(x['pid']))}
             for x in s25.usable}
    hits = set(HITS)
    acc = defaultdict(int)
    endpoint, detail = {}, {}
    for tag in TICKS:
        for pid in sorted(store):
            x = by_pid[pid]
            st = st_of(pid, tag)
            served = False
            if pid in hits:
                ctr = defaultdict(int)
                served = chain_verify(x, store[pid]['cert'], tag, pid,
                                      contract, S, p, seed, ctr)
                for k, v in ctr.items():
                    acc[k] += v
                if served and tag == T and pid in DRIFT:
                    acc['wrong'] += 1
                false_decline = (not served and
                                 verify(store[pid]['cert'], st, contract))
            else:
                false_decline = False
            if served:
                acc['served'] += 1
                endpoint[pid] = ('served', store[pid]['cert']['issued_at'])
                detail[(tag, pid)] = 'served'
                continue
            if false_decline:
                acc['false_decline'] += 1
            ans, cert = fb_of(pid, tag)
            if ans is None:
                acc['abstain'] += 1
                endpoint[pid] = ('abstain', None)
                detail[(tag, pid)] = 'abstain'
                continue
            acc['fallback'] += 1
            store[pid] = {'answer': ans, 'cert': cert}
            hits.add(pid)
            endpoint[pid] = ('regenerated', tag)
            detail[(tag, pid)] = 'fallback'
    acc['endpoint_correct'] = score_endpoint(endpoint)
    ENDPOINT_DETAIL[(contract, p, arm, seed)] = dict(endpoint)
    return (acc, detail) if keep_detail else (acc, None)

# ---------------- C_gate: per-request gate time (reference) -----------------
t0 = time.perf_counter_ns()
n_gate = 0
for tag in TICKS:
    for pid in sorted(HITS):
        verify(base_cert_of(pid), st_of(pid, tag), 'asserted')
        n_gate += 1
t_gate = (time.perf_counter_ns() - t0) / n_gate
print(f'C_gate: {t_gate:.0f} ns per erasure-free asserted gate check '
      f'({n_gate} checks).')

# ---------------- V1 / V2 -----------------------------------------------------
print('V1/V2: reproduction checks ...')
expect_fb = {'asserted': 29, 'strict': 53}
for contract in CONTRACTS:
    a, _ = run_loop(contract, 0.0, 'none', seed=0)
    b, _ = run_loop(contract, 0.20, 'full', seed=0)
    for name, r in (('p=0/none', a), ('p=0.20/full', b)):
        assert r['fallback'] == expect_fb[contract] and r['wrong'] == 0, \
            f'V1/V2 FAILED for {contract} {name}: {dict(r)}'
print('  V1 OK: p=0 none reproduces the s25 loop (asserted 29 / strict 53).')
print('  V2 OK: full protection at p=0.20 is identical (real-chain restores '
      'never launder).')

# ---------------- main sweep --------------------------------------------------
ARMS = ('none', 'random', 'fixed', 'full')
results, paired = {}, {}
for contract in CONTRACTS:
    for arm in ARMS:
        for p in PS:
            cells = []
            for r in range(R):
                seed = 1000 + r
                keep = (contract == 'asserted' and p == 0.10)
                acc, det = run_loop(contract, p, arm, seed, keep_detail=keep)
                cells.append(acc)
                if keep:
                    paired[(arm, seed)] = det
            agg = {
                'fallbacks_mean': round(sum(c['fallback'] for c in cells)
                                        / R, 1),
                'fallbacks_p95': sorted(c['fallback'] for c in cells)
                    [int(0.95 * R)],
                'false_declines_mean': round(
                    sum(c['false_decline'] for c in cells) / R, 1),
                'restores_mean': round(sum(c['restores'] for c in cells)
                                       / R, 1),
                'guard_withdrawn_mean': round(
                    sum(c['guard_withdrawn'] for c in cells) / R, 1),
                'guard_incompat_mean': round(
                    sum(c['guard_incompat'] for c in cells) / R, 1),
                'both_unreadable_mean': round(
                    sum(c['both_unreadable'] for c in cells) / R, 1),
                'serves_mean': round(sum(c['served'] for c in cells) / R, 1),
                'abstains_mean': round(sum(c['abstain'] for c in cells)
                                       / R, 1),
                'endpoint_correct_mean': round(
                    sum(c['endpoint_correct'] for c in cells) / R, 1),
                'wrong_max': max(c['wrong'] for c in cells),
            }
            assert agg['wrong_max'] == 0, 'V4 FAILED: wrong serve detected'
            results[f'{contract}/{arm}/p={p}'] = agg
print('V4 OK: zero wrong serves in every realization of every cell.')
ref = results['asserted/fixed/p=0.1']
ref_s = results['strict/fixed/p=0.1']
assert (ref['guard_withdrawn_mean'] + ref_s['guard_incompat_mean']) > 0
print(f"V5 OK: guard paths fire (asserted/fixed/p=0.10: "
      f"{ref['guard_withdrawn_mean']} withdrawn per run; strict/fixed/"
      f"p=0.10: {ref_s['guard_withdrawn_mean']} withdrawn + "
      f"{ref_s['guard_incompat_mean']} incompatible) and every "
      'fingerprint-verified but ineligible payload is declined '
      '(structural return; zero wrong serves under V4).')

# V7: the none arm must reproduce s27's archived none cells EXACTLY (same
# CRN fault plan; the real chain reduces to the ideal overlay when no item
# is protected).
s27 = json.load(open(f'{DATA}/protected_closed_loop.json'))
for contract in CONTRACTS:
    for p in PS:
        got = results[f'{contract}/none/p={p}']['fallbacks_mean']
        exp = s27['cells'][f'{contract}/none/p={p}']['fallbacks_mean']
        assert got == exp, f'V7 FAILED: {contract}/none/p={p}: {got} != {exp}'
print('V7 OK: none arm reproduces the s27 archived cells exactly '
      '(61.2/89.7/142.7 asserted, 115.8/158.4/217.6 strict).')

# ---------------- paired attribution (asserted, p=0.10) -----------------------
transitions = defaultdict(lambda: defaultdict(int))
for seed in range(1000, 1000 + R):
    dn = paired[('none', seed)]
    for arm in ('random', 'fixed', 'full'):
        da = paired[(arm, seed)]
        for key, cat_n in dn.items():
            transitions[f'none->{arm}'][f'{cat_n}->{da[key]}'] += 1
paired_out = {k: dict(v) for k, v in transitions.items()}

# ---------------- cost model --------------------------------------------------
t_restore = T_RESTORE_NS[1] / max(T_RESTORE_NS[0], 1)
print(f'C_restore: {T_RESTORE_NS[0]} timed backup reads, '
      f'{t_restore:.0f} ns/op mean.')

def cell_cost(arm):
    a = results[f'asserted/{arm}/p=0.1']
    n = results['asserted/none/p=0.1']
    d_fb = n['fallbacks_mean'] - a['fallbacks_mean']
    restore_ops = (a['restores_mean'] + a['guard_withdrawn_mean']
                   + a['guard_incompat_mean'])
    if arm == 'random':
        set_bytes = sum(random_feasible_set(s)[1]
                        for s in range(1000, 1000 + R)) / R
    else:
        set_bytes = arm_set(arm, 0)[1]
    incr_ms = restore_ops * t_restore / 1e6
    return {'delta_fallbacks': round(d_fb, 1),
            'restore_ops_mean': round(restore_ops, 1),
            'protection_bytes': round(set_bytes),
            'incremental_restore_time_ms': round(incr_ms, 2),
            'breakeven_us_per_avoided_execution':
                round(incr_ms * 1e3 / d_fb, 1) if d_fb > 0 else None,
            'breakeven_bytes_per_avoided_execution':
                round(set_bytes / d_fb, 1) if d_fb > 0 else None}

costs = {arm: cell_cost(arm) for arm in ('random', 'fixed', 'full')}

print(f'\n==== real-chain protection (means over {R} realizations, '
      f'3 ticks) ====')
print(f'{"cell":26s} {"fallb":>6s} {"p95":>5s} {"falseD":>6s} {"restr":>6s} '
      f'{"grdW":>5s} {"grdI":>5s} {"bothU":>5s} {"corrT":>6s}')
for k, a in results.items():
    print(f'{k:26s} {a["fallbacks_mean"]:6.1f} {a["fallbacks_p95"]:5d} '
          f'{a["false_declines_mean"]:6.1f} {a["restores_mean"]:6.1f} '
          f'{a["guard_withdrawn_mean"]:5.1f} {a["guard_incompat_mean"]:5.1f} '
          f'{a["both_unreadable_mean"]:5.1f} {a["endpoint_correct_mean"]:6.1f}')
print('\npaired transitions (asserted, p=0.10, all realizations):')
for k, v in paired_out.items():
    print(f'  {k}: {v}')
print('\ncost / break-even (asserted, p=0.10):')
for k, v in costs.items():
    print(f'  {k}: {v}')

out = {
    'model': 'executable protection chain over real SQLite stores: '
             'eligibility registry (per-tick status+version keyed by '
             'premise identity qid|prop|value / page|sentence-id), source '
             'and protection payload stores with sha1 fingerprints, '
             'certificate store; a failed source read of a protected item '
             'triggers backup read -> fingerprint check -> registry check; '
             'payload-intact but policy_withdrawn/version_incompatible '
             'reads are declined (guard); budgeted arms also lose backup '
             'reads i.i.d. at rate p (both-unreadable); the full arm '
             'assumes read-any redundancy (upper-bound reference)',
    'arms': ['none', 'random (feasible byte-budget set, drawn per '
             'realization then fixed)', 'fixed (s20 greedy, static from '
             't0)', 'full (reference)'],
    'realizations': R, 'ticks': TICKS,
    'stores': {'sqlite_file': 'payload_stores.sqlite',
               'sqlite_bytes': sqlite_bytes,
               'universe_items': len(items),
               'universe_bytes': total_bytes,
               'post_build_placeholder_items': len(wb_items),
               'budget_bytes': B,
               'greedy_items': len(S_fixed),
               'registry_premises': len(premise_items),
               'certificates_rows': len(installed_asserted),
               'certificates_bytes': cert_bytes},
    'costs_measured': {
        'C_state_registry_build_ms': round(t_state_build * 1e3, 2),
        'C_state_snapshot_loads_ms': round(t_load * 1e3, 3),
        'C_certificate_serialize_insert_ms': round(t_cert * 1e3, 2),
        'C_gate_ns_per_check': round(t_gate, 0),
        'C_restore_ns_per_backup_read': round(t_restore, 0),
        'C_restore_timed_reads': T_RESTORE_NS[0]},
    'validations': {
        'V1_p0_none_reproduces_s25': True,
        'V2_full_at_p02_identical': True,
        'V3_greedy_bytes_match_s20': B,
        'V4_zero_wrong_all_cells': True,
        'V5_guard_paths_fire_and_decline': {
            'asserted_fixed_p0.10_withdrawn_mean':
                ref['guard_withdrawn_mean'],
            'strict_fixed_p0.10_withdrawn_mean':
                ref_s['guard_withdrawn_mean'],
            'strict_fixed_p0.10_incompat_mean':
                ref_s['guard_incompat_mean']},
        'V6_random_sets_feasible': True,
        'V7_none_arm_matches_s27_archive': True},
    'cells': results,
    'paired_transitions_asserted_p0.10': paired_out,
    'breakeven_asserted_p0.10': costs,
    'notes': [
        'fault plan via CRN u01(loss|closs, seed, tag, pid, kind, item): '
        'identical across arms given the seed',
        'random arm: one shuffled-fill set up to B bytes per realization, '
        'fixed for its 3-tick run; replaces the Bernoulli(0.90) per-item '
        'arm of s27, which had no realized byte cap',
        'protection universe and byte budgets are t0 base-certificate '
        'items only (temporal protocol); post-build write-back items '
        'enter unprotected in the budgeted arms and are covered by the '
        'full arm only under its read-any reference assumption '
        '(placeholder payload rows, excluded from all budget figures)',
        'restores and guards execute real timed SQLite point reads; the '
        'registry is consulted from per-tick snapshots loaded from the '
        'registry table (loads timed as C_state)',
        'intermediate-tick regenerations are template compositions '
        '(zero LLM calls); archived real LLM regenerations occur only at '
        'the 20260801 endpoint',
        'template repairs are value-correct by construction (proxy); '
        'endpoint correctness uses the human labels as in s25-s29']}
if not os.environ.get('S30_NO_WRITE'):
    json.dump(out, open(f'{DATA}/real_payload_chain.json', 'w'),
              ensure_ascii=False, indent=1)
    print('\nsaved data/real_payload_chain.json')
else:
    print('\nS30_NO_WRITE set: archive left untouched (s35 rescore mode).')
con.close()

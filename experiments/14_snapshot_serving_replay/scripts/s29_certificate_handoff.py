#!/usr/bin/env python3
"""
s29_certificate_handoff.py  --  P1: certificate handoff under independent
verification (reviewer acceptance criterion 1: the execution loop holds --
certificates can be handed to an independent verifier, and revocation cannot
be laundered).

Design
------
Every certificate the closed loop installs (154 base certificates plus every
write-back certificate issued by the asserted-claims loop over the three
snapshot-update ticks) is serialized into a tamper-evident envelope binding
SIX elements:

  (1) request id (pid)           (4) full certificate claim set
  (2) answer payload digest      (5) evidence sentence ids
  (3) asserted claim set         (6) issuance snapshot

The envelope digest is sha256 over the canonical JSON of those six elements.
An INDEPENDENT verifier reads only the transferred envelopes (JSONL file) and
the state-provider view; it recomputes the digest, checks the served request
id and answer payload against the envelope, and only then evaluates the
contract against the current state.

The digest makes an honestly issued certificate tamper-EVIDENT (in deployment
it would be signed by the issuer); the experiment shows that no bound element
can be altered without detection, and that a certificate revoked by the
current state cannot be laundered back into acceptance.

Checks
------
  V1 positive control: for every envelope x 4 snapshots x 4 contracts the
     independent verifier's verdict equals the in-memory verify() verdict.
  V2 six tamper classes, injected into every envelope, all rejected:
     M1 payload substitution        M4 sentence substitution
     M2 claim-set laundering        M5 issuance forward-dating
     M3 asserted-set laundering     M6 cross-request replay
  V3 laundering focus: every base certificate that the 20260801 state has
     legitimately revoked (declined under the asserted contract) is still
     rejected after a laundering attempt (M2/M3).

Outputs: data/certificates_handoff.jsonl (the transferred artifact),
         data/certificate_handoff.json + console summary.
"""

import json, os, re, hashlib

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.normpath(os.path.join(HERE, '..', 'data'))
ANN = os.path.normpath(os.path.join(HERE, '..', 'annotation'))
REB = os.path.normpath(os.path.join(HERE, '..', '..', '12_real_dynamics_audit',
                                   'data', 'rebuilt_2026'))

SNAPS = ['20260401', '20260501', '20260701', '20260801']
TICKS = ['20260501', '20260701', '20260801']
T = '20260801'
CONTRACTS = ('strict', 'majority', 'claims', 'asserted')

# ---------------------------------------------------------------- helpers ---
def norm(s):
    return re.sub(r'[,\s]+', '', str(s).lower())

def wikitext_to_plain(wt):
    wt = re.sub(r"<ref[^>]*>.*?</ref>", " ", wt, flags=re.S)
    wt = re.sub(r"<ref[^>]*/>", " ", wt)
    wt = re.sub(r"\{\{[^{}]*\}\}", " ", wt)
    wt = re.sub(r"\{\{[^{}]*\}\}", " ", wt)
    wt = re.sub(r"\[\[([^|\]]*\|)?([^\]]*)\]\]", r"\2", wt)
    wt = re.sub(r"\[https?://[^\s\]]+\s*([^\]]*)\]", r"\1", wt)
    wt = re.sub(r"'{2,}", "", wt)
    wt = re.sub(r"<[^>]+>", " ", wt)
    wt = re.sub(r"^[=]+.*?[=]+$", " ", wt, flags=re.M)
    return wt

def sentences(text, min_len=40):
    parts = re.split(r"(?<=[.!?。！？])\s+|\n+", text)
    return [re.sub(r"\s+", " ", s).strip() for s in parts
            if len(s.strip()) >= min_len]

def sid(s):
    return hashlib.sha1(s.encode('utf-8')).hexdigest()[:12]

def val_key(v):
    if isinstance(v, dict):
        if 'id' in v: return 'Q:' + str(v['id'])
        if 'amount' in v: return 'A:' + str(v['amount'])
        if 'time' in v: return 'T:' + str(v['time'])[:11]
        if 'text' in v: return 'S:' + str(v['text'])
    return 'S:' + str(v)

def truthy_keys(ent, prop):
    out = set()
    for rank, snak, val in ent['claims'].get(prop, []):
        if rank in ('normal', 'preferred') and snak == 'value' and val is not None:
            out.add(val_key(val))
    return out

def truthy_items(ent, prop):
    out = []
    for rank, snak, val in ent['claims'].get(prop, []):
        if rank in ('normal', 'preferred') and snak == 'value' and val is not None:
            out.append((val_key(val), val))
    return out

def answer_asserts(answer, vtext):
    na, nv = norm(answer), norm(vtext)
    if nv and nv in na:
        return True
    m = re.match(r'(\d{4})', str(vtext))
    return bool(m and m.group(1) in str(answer))

# ------------------------------------------------------------- state load ---
print('loading snapshots ...')
snaps_ent = {t: json.load(open(f'{REB}/entities_v2_{t}.json')) for t in SNAPS}
pages = {t: json.load(open(f'{REB}/pages_{t}.json')) for t in SNAPS}
page_sids = {t: {title: {sid(s) for s in sentences(wikitext_to_plain(w))}
                 for title, w in pages[t].items()} for t in SNAPS}
title_map = json.load(open(f'{REB}/title_qid_map.json'))
qid2title = {int(v): k for k, v in title_map.items()}

def value_text(v):
    if isinstance(v, dict):
        if 'id' in v:
            vid = str(v['id']).lstrip('Q')
            try:
                return qid2title.get(int(vid), 'Q' + str(vid))
            except (TypeError, ValueError):
                return str(v['id'])
        if 'amount' in v: return str(v['amount']).lstrip('+')
        if 'time' in v: return str(v['time'])[1:11]
        if 'text' in v: return str(v['text'])
    return str(v)

def state_of(tag, qid, prop, page_title):
    ent = snaps_ent[tag].get(str(qid))
    return {'truthy': truthy_keys(ent, prop) if ent is not None else set(),
            'sids': page_sids[tag].get(page_title, set())}

# ------------------------------------------------------------- panel load ---
panel = json.load(open(f'{DATA}/evidence.json'))
gen = json.load(open(f'{DATA}/generation.json'))
lab = {int(k): v for k, v in json.load(open(f'{ANN}/answer_labels.json')).items()}
regen = json.load(open(f'{DATA}/regeneration.json'))

usable = [x for x in panel
          if gen[str(x['pid'])]['answer_ok'] and gen[str(x['pid'])]['paraphrase']]
assert len(usable) == 154
HITS = {p for p in lab if lab[p]['hit']}
by_pid = {int(x['pid']): x for x in usable}

# ------------------------------------------------------- certificate layer ---
def base_certificate(x):
    claims = [val_key(c['value']) for c in x['certificate']['claims']]
    ans = gen[str(x['pid'])]['answer']
    asserted = [val_key(c['value'])
                for c, vt in zip(x['certificate']['claims'], x['values_text'])
                if answer_asserts(ans, vt)]
    return {'claims': claims, 'asserted': asserted,
            'sentences': [s['id'] for s in x['certificate']['sentences']],
            'issued_at': SNAPS[0], 'origin': 'base'}

def current_certificate(x, tag, answer_text):
    ent = snaps_ent[tag].get(str(x['qid']))
    items = truthy_items(ent, x['prop']) if ent is not None else []
    claims = [k for k, _ in items]
    texts = [value_text(v) for _, v in items]
    sents = []
    for s in sentences(wikitext_to_plain(pages[tag].get(x.get('page_title'), ''))):
        if any(norm(vt) and norm(vt) in norm(s) for vt in texts):
            sents.append(sid(s))
        if len(sents) == 3:
            break
    return {'claims': claims, 'asserted': list(claims),
            'sentences': sents, 'issued_at': tag, 'origin': f'repaired@{tag}'}

def verify(cert, st, contract):
    cdead = [k for k in cert['claims'] if k not in st['truthy']]
    if contract == 'claims':
        return not cdead
    if contract == 'asserted':
        return not any(k in cdead for k in cert['asserted'])
    sdead = [s for s in cert['sentences'] if s not in st['sids']]
    ns = len(cert['sentences'])
    if contract == 'strict':
        return not cdead and not sdead
    if contract == 'majority':
        return not cdead and (ns == 0 or (ns - len(sdead)) * 2 > ns)
    raise ValueError(contract)

# ------------------------------ collect every certificate the loop installs --
print('replaying the asserted-claims loop to collect installed certificates ...')
def template_fallback(x, tag):
    ent = snaps_ent[tag].get(str(x['qid']))
    items = truthy_items(ent, x['prop']) if ent is not None else []
    if not items:
        return None, None
    answer = ', '.join(value_text(v) for _, v in items)
    return answer, current_certificate(x, tag, answer)

def fallback(x, tag):
    if tag == T:
        r = regen.get(str(x['pid']))
        if r is not None and r.get('regen_ok'):
            return r['regen_answer'], current_certificate(x, T, r['regen_answer'])
    return template_fallback(x, tag)

installed = []          # (pid, answer, cert) -- every cert the loop installs
store = {}
for x in usable:
    pid = int(x['pid'])
    store[pid] = {'answer': gen[str(pid)]['answer'], 'cert': base_certificate(x)}
    installed.append((pid, store[pid]['answer'], store[pid]['cert']))
hits = set(HITS)
for tag in TICKS:
    for pid in sorted(store):
        x = by_pid[pid]
        st = state_of(tag, x['qid'], x['prop'], x.get('page_title'))
        if pid in hits and verify(store[pid]['cert'], st, 'asserted'):
            continue
        ans, cert = fallback(x, tag)
        if ans is None:
            continue
        store[pid] = {'answer': ans, 'cert': cert}
        hits.add(pid)
        installed.append((pid, ans, cert))
print(f'  {len(installed)} certificates installed '
      f'(154 base + {len(installed) - 154} write-back).')

# ------------------------------------------------- serialization / binding ---
def answer_sha(answer):
    return hashlib.sha256(answer.encode('utf-8')).hexdigest()[:16]

def envelope_of(pid, answer, cert):
    """tamper-evident envelope binding the six elements."""
    core = {'pid': pid,
            'answer_sha': answer_sha(answer),
            'asserted': sorted(cert['asserted']),
            'claims': sorted(cert['claims']),
            'sentences': sorted(cert['sentences']),
            'issued_at': cert['issued_at']}
    canon = json.dumps(core, sort_keys=True, separators=(',', ':'))
    env = dict(core)
    env['origin'] = cert['origin']
    env['digest'] = hashlib.sha256(canon.encode('utf-8')).hexdigest()
    return env

envelopes = [envelope_of(pid, ans, cert) for pid, ans, cert in installed]
with open(f'{DATA}/certificates_handoff.jsonl', 'w') as f:
    for env in envelopes:
        f.write(json.dumps(env, ensure_ascii=False) + '\n')
print(f'  wrote {len(envelopes)} envelopes to data/certificates_handoff.jsonl')

# --------------------------------------------- independent verification entry
CORE_FIELDS = ('pid', 'answer_sha', 'asserted', 'claims', 'sentences',
               'issued_at')

def independent_verify(env, served_pid, served_answer, st, contract):
    """verifies a TRANSFERRED envelope against the state; returns (ok, why)."""
    core = {k: env[k] for k in CORE_FIELDS}
    canon = json.dumps(core, sort_keys=True, separators=(',', ':'))
    if hashlib.sha256(canon.encode('utf-8')).hexdigest() != env['digest']:
        return False, 'integrity: digest mismatch (bound element altered)'
    if env['pid'] != served_pid:
        return False, 'binding: request-id mismatch (cross-request replay)'
    if env['answer_sha'] != answer_sha(served_answer):
        return False, 'binding: answer-payload mismatch (substitution)'
    cert = {'claims': env['claims'], 'asserted': env['asserted'],
            'sentences': env['sentences']}
    ok = verify(cert, st, contract)
    return ok, 'contract ' + ('accept' if ok else 'decline')

# --------------------------------------- V1: positive control, all verdicts --
print('V1: independent verifier vs in-memory verify() ...')
answers_by_pid_cert = {(pid, id(c)): a for pid, a, c in installed}
n_cmp = n_agree = 0
for (pid, ans, cert), env in zip(installed, envelopes):
    x = by_pid[pid]
    for tag in SNAPS:
        st = state_of(tag, x['qid'], x['prop'], x.get('page_title'))
        for v in CONTRACTS:
            ok, _ = independent_verify(env, pid, ans, st, v)
            n_cmp += 1
            n_agree += (ok == verify(cert, st, v))
print(f'  V1: {n_agree}/{n_cmp} verdicts agree.')
assert n_agree == n_cmp, 'V1 FAILED: independent verifier disagrees'

# --------------------------------------- V2: six tamper classes, all caught --
print('V2: tamper injection ...')
def clone(env):
    return json.loads(json.dumps(env))

tampered_counts = {f'M{i}': 0 for i in range(1, 7)}
tampered_rejected = {f'M{i}': 0 for i in range(1, 7)}
pids_sorted = sorted(by_pid)
for i, ((pid, ans, cert), env) in enumerate(zip(installed, envelopes)):
    x = by_pid[pid]
    st = state_of(T, x['qid'], x['prop'], x.get('page_title'))
    dead_c = [k for k in env['claims'] if k not in st['truthy']]
    dead_a = [k for k in env['asserted'] if k not in st['truthy']]
    dead_s = [s for s in env['sentences'] if s not in st['sids']]
    live_s = sorted(st['sids']) or ['live_placeholder']

    def launder(field, dead, forge):
        """drop the revoked elements; if none/empty, forge a foreign one --
        either way the bound element changes."""
        lst = env[field]
        if not lst:
            return [forge]
        drop = dead or lst[:1]
        kept = [k for k in lst if k not in drop]
        return kept if kept != lst else [forge]

    def substitute_sents(env, dead, live):
        """replace dead sentence ids with live ones; degenerate cases (empty
        list, or replacement equal to the original) fall back to inserting a
        forged id -- every branch changes the bound element."""
        lst = env['sentences']
        if not lst:
            return [live[0]]
        drop = dead or lst[:1]
        new = [live[j % len(live)] if s in drop else s
               for j, s in enumerate(lst)]
        return new if new != lst else new + ['forged_sid']

    variants = {
        # M1 payload substitution: serve a different answer under this cert
        'M1': (env, pid, gen[str(pids_sorted[(i + 1) % len(pids_sorted)])]['answer']),
        # M2 claim-set laundering: drop the revoked claims
        'M2': ({**clone(env), 'claims': launder('claims', dead_c, 'Q:forged')},
               pid, ans),
        # M3 asserted-set laundering: drop the revoked asserted claims
        'M3': ({**clone(env), 'asserted': launder('asserted', dead_a, 'Q:forged')},
               pid, ans),
        # M4 sentence substitution: swap dead sentence ids for live ones
        'M4': ({**clone(env), 'sentences': substitute_sents(env, dead_s,
                                                            live_s)},
               pid, ans),
        # M5 issuance forward-dating
        'M5': ({**clone(env), 'issued_at': T}, pid, ans),
        # M6 cross-request replay: this cert presented for another request
        'M6': (env, pids_sorted[(i + 1) % len(pids_sorted)], ans),
    }
    for m, (menv, mpid, mans) in variants.items():
        if menv['issued_at'] == T and m == 'M5' and env['issued_at'] == T:
            menv['issued_at'] = SNAPS[0]     # backdating is equally tampering
        tampered_counts[m] += 1
        rejected_all = True
        for v in CONTRACTS:
            ok, _ = independent_verify(menv, mpid, mans, st, v)
            if ok:
                rejected_all = False
        tampered_rejected[m] += rejected_all
for m in sorted(tampered_counts):
    print(f'  {m}: {tampered_rejected[m]}/{tampered_counts[m]} rejected '
          f'under every contract.')
assert all(tampered_rejected[m] == tampered_counts[m] for m in tampered_counts), \
    'V2 FAILED: a tampered envelope was accepted'

# ----------------------- V3: laundering cannot resurrect a revoked cert -----
print('V3: laundering focus on state-revoked certificates ...')
revoked = []
for (pid, ans, cert), env in zip(installed, envelopes):
    if env['origin'] != 'base':
        continue
    x = by_pid[pid]
    st = state_of(T, x['qid'], x['prop'], x.get('page_title'))
    if not verify(cert, st, 'asserted'):
        revoked.append((pid, ans, env, st))
n_launder_rejected = 0
for pid, ans, env, st in revoked:
    for field in ('claims', 'asserted'):
        st2 = state_of(T, by_pid[pid]['qid'], by_pid[pid]['prop'],
                       by_pid[pid].get('page_title'))
        live = [k for k in env[field] if k in st2['truthy']]
        menv = {**clone(env), field: live}
        ok, _ = independent_verify(menv, pid, ans, st, 'asserted')
        n_launder_rejected += (not ok)
print(f'  V3: {len(revoked)} base certificates revoked by the {T} state; '
      f'{n_launder_rejected}/{2 * len(revoked)} laundering attempts rejected.')
assert n_launder_rejected == 2 * len(revoked), 'V3 FAILED: laundering succeeded'

# ------------------------------------------------------------------- output --
out = {
    'envelopes': len(envelopes),
    'base_certificates': 154,
    'writeback_certificates': len(installed) - 154,
    'bound_elements': list(CORE_FIELDS),
    'validations': {
        'V1_positive_control': {'comparisons': n_cmp, 'agreements': n_agree},
        'V2_tamper_injection': {
            m: {'injected': tampered_counts[m],
                'rejected_under_all_contracts': tampered_rejected[m]}
            for m in sorted(tampered_counts)},
        'V3_revocation_laundering': {
            'revoked_base_certificates_at_endpoint': len(revoked),
            'laundering_attempts': 2 * len(revoked),
            'rejected': n_launder_rejected}},
    'notes': [
        'the envelope digest binds six elements: request id, answer payload '
        'digest, asserted claim set, full claim set, evidence sentence ids, '
        'issuance snapshot; it makes an honestly issued certificate '
        'tamper-evident (a deployment would sign the digest)',
        'tamper classes: M1 payload substitution, M2 claim-set laundering, '
        'M3 asserted-set laundering, M4 sentence substitution, M5 issuance '
        'forward-dating, M6 cross-request replay',
        'the independent verifier reads only the transferred JSONL envelopes '
        'plus the state-provider view; integrity and binding checks run '
        'before contract evaluation']}
json.dump(out, open(f'{DATA}/certificate_handoff.json', 'w'),
          ensure_ascii=False, indent=1)
print('\nsaved data/certificate_handoff.json')
print(json.dumps(out['validations'], indent=1))

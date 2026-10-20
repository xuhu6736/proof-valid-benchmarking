#!/usr/bin/env python3
"""
s25_closed_loop_system.py  --  P1 Experiment A: runnable closed-loop prototype.

A small but complete serving system with the four components the reviewer
asked for:

  1. certificate output      -- every cache entry carries a certificate
                                {claims, asserted claims, sentences, issued_at}
  2. current-state upkeep    -- a StateProvider materializes the current
                                evidence state from the four real rebuilt
                                snapshots (2026-04/05/07/08)
  3. fallback                -- declined/missed requests are regenerated
                                (template-composed from current truthy values
                                at intermediate ticks; the real cached LLM
                                regenerations at the 2026-08 endpoint)
  4. cache update (write-back)-- regenerated answers are written back with a
                                fresh certificate built from the same current
                                evidence, so later ticks re-serve from cache

Policies run under an identical request stream (all 154 requests at each of
the three monthly ticks):

  similarity-static   serve any similar hit; never gate, never update
  TTL-1mo             blind refresh of every entry at every tick
  page-change-loop    serve iff the source page is unchanged since issue,
                      else regenerate + write back
  pvsb-strict-loop    strict contract gate + regeneration + write-back
  pvsb-asserted-loop  asserted-claims contract gate + regeneration + write-back

Built-in validations:
  V1  no-loop verifier on the base certificates at 20260801 reproduces the
      published endpoint counts (strict 89 / majority 96 / claims 104 /
      asserted 111 served of 147 hits) -- checked against age_curve.json
  V2  every certificate written back at 20260801 verifies at the 20260801
      state under every contract; every abstention has no current evidence
  V3  every request is accounted exactly once per tick per policy

Outputs: data/closed_loop_system.json + console summary.
"""

import json, os, re, hashlib
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.normpath(os.path.join(HERE, '..', 'data'))
ANN = os.path.normpath(os.path.join(HERE, '..', 'annotation'))
REB = os.path.normpath(os.path.join(HERE, '..', '..', '12_real_dynamics_audit',
                                   'data', 'rebuilt_2026'))

SNAPS = ['20260401', '20260501', '20260701', '20260801']
TICKS = ['20260501', '20260701', '20260801']     # serving ticks after t0 build

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
    """truthy (key, raw value) pairs, preserving the raw value for display."""
    out = []
    for rank, snak, val in ent['claims'].get(prop, []):
        if rank in ('normal', 'preferred') and snak == 'value' and val is not None:
            out.append((val_key(val), val))
    return out

def answer_asserts(answer, vtext):
    na = norm(answer)
    nv = norm(vtext)
    if nv and nv in na:
        return True
    m = re.match(r'(\d{4})', str(vtext))
    if m and m.group(1) in str(answer):
        return True
    return False

# ------------------------------------------------------------- state load ---
print('loading snapshots ...')
snaps_ent = {t: json.load(open(f'{REB}/entities_v2_{t}.json')) for t in SNAPS}
pages = {t: json.load(open(f'{REB}/pages_{t}.json')) for t in SNAPS}
page_sids, page_hash = {}, {}
for t in SNAPS:
    page_sids[t] = {}
    page_hash[t] = {}
    for title, w in pages[t].items():
        page_sids[t][title] = {sid(s) for s in sentences(wikitext_to_plain(w))}
        page_hash[t][title] = hashlib.sha1(w.encode('utf-8')).hexdigest()[:12]
title_map = json.load(open(f'{REB}/title_qid_map.json'))
qid2title = {int(v): k for k, v in title_map.items()}

def value_text(v):
    if isinstance(v, dict):
        if 'id' in v:
            vid = v['id']
            if isinstance(vid, str) and vid.startswith('Q'):
                vid = vid[1:]
            try:
                return qid2title.get(int(vid), 'Q' + str(vid))
            except (TypeError, ValueError):
                return str(v['id'])
        if 'amount' in v:
            return str(v['amount']).lstrip('+')
        if 'time' in v:
            return str(v['time'])[1:11]
        if 'text' in v:
            return str(v['text'])
    return str(v)

def state_of(tag, qid, prop, page_title):
    """current evidence state for one panel entry at one snapshot."""
    ent = snaps_ent[tag].get(str(qid))
    tk = truthy_keys(ent, prop) if ent is not None else set()
    sids_now = page_sids[tag].get(page_title, set())
    return {'truthy': tk, 'sids': sids_now,
            'page_hash': page_hash[tag].get(page_title)}

# ------------------------------------------------------------- panel load ---
panel = json.load(open(f'{DATA}/evidence.json'))
gen = json.load(open(f'{DATA}/generation.json'))
lab = {int(k): v for k, v in json.load(open(f'{ANN}/answer_labels.json')).items()}
incoming = {int(k): v for k, v in json.load(open(f'{ANN}/incoming_query_labels.json')).items()}
regen = json.load(open(f'{DATA}/regeneration.json'))
regen_lab = {int(k): v for k, v in json.load(open(f'{DATA}/regen_labels.json')).items()}

usable = [x for x in panel
          if gen[str(x['pid'])]['answer_ok'] and gen[str(x['pid'])]['paraphrase']]
assert len(usable) == 154
HITS = {p for p in lab if lab[p]['hit']}            # 147 similarity hits
MISSES = sorted(set(lab) - HITS)                    # 7 retrieval misses
DRIFT = {p for p in incoming if incoming[p]['drift_kind'] != 'none'}
T = '20260801'

by_pid = {int(x['pid']): x for x in usable}

# ------------------------------------------------------- certificate layer ---
def base_certificate(x):
    claims = [val_key(c['value']) for c in x['certificate']['claims']]
    ans = gen[str(x['pid'])]['answer']
    asserted = [val_key(c['value'])
                for c, vt in zip(x['certificate']['claims'], x['values_text'])
                if answer_asserts(ans, vt)]
    return {'claims': claims,
            'asserted': asserted,
            'sentences': [s['id'] for s in x['certificate']['sentences']],
            'issued_at': SNAPS[0], 'origin': 'base'}

def current_certificate(x, tag, answer_text):
    """fresh certificate built from the current evidence state at `tag`."""
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

# -------------------------------------------------------------- verifier ---
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

# -------------------------------------------------------------- fallback ---
def template_fallback(x, tag):
    """regenerate from current truthy evidence; abstain when none exists."""
    ent = snaps_ent[tag].get(str(x['qid']))
    items = truthy_items(ent, x['prop']) if ent is not None else []
    if not items:
        return None, None
    answer = ', '.join(value_text(v) for _, v in items)
    return answer, current_certificate(x, tag, answer)

def real_fallback_aug(x):
    """the real cached LLM regeneration at the 20260801 endpoint."""
    r = regen.get(str(x['pid']))
    if r is None or not r.get('regen_ok'):
        return None, None
    cert = current_certificate(x, T, r['regen_answer'])
    return r['regen_answer'], cert

def fallback(x, tag):
    if tag == T:
        ans, cert = real_fallback_aug(x)
        if ans is not None:
            return ans, cert
    return template_fallback(x, tag)

# --------------------------------------------------------------- policies ---
def run_policy(name):
    """Simulate one policy over the three ticks; return per-tick accounting."""
    store = {}      # pid -> {'answer':..., 'cert':...}
    for x in usable:
        store[int(x['pid'])] = {'answer': gen[str(x['pid'])]['answer'],
                                'cert': base_certificate(x)}
    hits = set(HITS)    # retrieval outcome; PVSB write-back aliases repair misses
    rows = []
    for tag in TICKS:
        acc = defaultdict(int)
        detail = {}
        for pid in sorted(store):
            x = by_pid[pid]
            st = state_of(tag, x['qid'], x['prop'], x.get('page_title'))
            hit = pid in hits
            served = False
            if name == 'TTL-1mo':
                pass                                   # blind refresh below
            elif hit:
                if name == 'similarity-static':
                    served = True
                elif name == 'page-change-loop':
                    served = (st['page_hash'] is not None and
                              st['page_hash'] == page_hash[
                                  store[pid]['cert']['issued_at']].get(
                                      x.get('page_title')))
                elif name == 'pvsb-strict-loop':
                    served = verify(store[pid]['cert'], st, 'strict')
                elif name == 'pvsb-asserted-loop':
                    served = verify(store[pid]['cert'], st, 'asserted')
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
            # ---- fallback path (decline, miss, or TTL blind refresh) ----
            ans, cert = fallback(x, tag)
            if ans is None:
                acc['abstain'] += 1
                detail[pid] = ['abstain', None]
                continue
            acc['fallback'] += 1
            if name != 'similarity-static':            # static never updates
                store[pid] = {'answer': ans, 'cert': cert}
                if name.startswith('pvsb'):
                    hits.add(pid)                      # evidence-checked alias
            detail[pid] = ['regenerated', tag]
        for k in ('served', 'fallback', 'abstain', 'wrong_proxy',
                  'decayed_proxy'):
            acc[k] += 0
        rows.append({'tick': tag, **acc, '_detail': detail})
    return rows

# ----------------------------------------------------------- V1 validation --
print('V1: endpoint reproduction check (no loop, base certs, 20260801) ...')
ac = json.load(open(f'{DATA}/age_curve.json'))
expect = {v: ac['endpoint'][v]['served'] for v in
          ('strict', 'majority', 'claims', 'asserted')}
got = {}
for v in ('strict', 'majority', 'claims', 'asserted'):
    n = 0
    for pid in HITS:
        x = by_pid[pid]
        st = state_of(T, x['qid'], x['prop'], x.get('page_title'))
        if verify(base_certificate(x), st, v):
            n += 1
    got[v] = n
print('  expected', expect)
print('  got      ', got)
assert got == expect, 'V1 FAILED: verifier does not reproduce endpoint counts'
print('  V1 OK: verifier reproduces the published endpoint counts exactly.')

# --------------------------------------------------------------- run all ---
policies = ['similarity-static', 'TTL-1mo', 'page-change-loop',
            'pvsb-strict-loop', 'pvsb-asserted-loop']
results = {}
for name in policies:
    print(f'running {name} ...')
    rows = run_policy(name)
    for r in rows:
        assert (r['served'] + r['fallback'] + r['abstain']) == 154, \
            f'V3 FAILED for {name} at {r["tick"]}'
    results[name] = rows
print('V3 OK: every request accounted exactly once per tick per policy.')

# ------------------------------- V2: written-back certs verify at issue -----
print('V2: write-back certificate soundness ...')
store = {}
for x in usable:
    store[int(x['pid'])] = {'answer': gen[str(x['pid'])]['answer'],
                            'cert': base_certificate(x)}
wb_check = {'verified': 0, 'failed': [], 'abstain_no_evidence': 0,
            'abstain_with_evidence': []}
hits = set(HITS)
for tag in TICKS:
    for pid in sorted(store):
        x = by_pid[pid]
        st = state_of(tag, x['qid'], x['prop'], x.get('page_title'))
        if pid in hits and verify(store[pid]['cert'], st, 'asserted'):
            continue
        ans, cert = fallback(x, tag)
        if ans is None:
            ent = snaps_ent[tag].get(str(x['qid']))
            if truthy_keys(ent, x['prop']) if ent is not None else set():
                wb_check['abstain_with_evidence'].append((pid, tag))
            else:
                wb_check['abstain_no_evidence'] += 1
            continue
        st2 = state_of(tag, x['qid'], x['prop'], x.get('page_title'))
        if all(verify(cert, st2, v) for v in
               ('strict', 'majority', 'claims', 'asserted')):
            wb_check['verified'] += 1
        else:
            wb_check['failed'].append((pid, tag))
        store[pid] = {'answer': ans, 'cert': cert}
        hits.add(pid)
assert not wb_check['failed'] and not wb_check['abstain_with_evidence']
print(f'  V2 OK: {wb_check["verified"]} written-back certificates verify at '
      f'their issue state under all four contracts; '
      f'{wb_check["abstain_no_evidence"]} abstentions all lack current evidence.')

# --------------------------------------- endpoint human-label cross-check ---
def endpoint_truth(rows):
    """final-tick deliveries, broken down by HUMAN labels where they exist:
    base-origin cached serves use the answer labels, real 20260801
    regenerations use the regeneration labels; intermediate template repairs
    are reported as proxy-correct (their asserted values are truthy by
    verifier construction); incoming-query drift overrides both."""
    det = rows[-1]['_detail']
    out = defaultdict(int)
    for pid, (how, origin) in det.items():
        if how == 'abstain':
            out['abstain'] += 1
        elif how == 'regenerated':
            rl = regen_lab.get(pid)
            if origin == T and rl is not None:
                out['regen_' + {'正确': 'correct', '不完整': 'incomplete',
                                '错误': 'wrong'}.get(rl['R1_adjudicated'],
                                                    'indet')] += 1
            else:
                out['regen_template_proxy'] += 1
        elif pid in DRIFT:
            out['served_drift_wrong'] += 1        # incoming-query drift
        elif origin == SNAPS[0]:
            a4 = lab[pid].get('A4', '无法判断')
            out['served_' + {'正确': 'correct', '错误': 'wrong',
                             '不完整': 'incomplete',
                             '无法判断': 'indet'}.get(a4, 'indet')] += 1
        else:
            out['served_repaired_proxy'] += 1
    return dict(out)

summary = {}
for name in policies:
    rows = results[name]
    ep = endpoint_truth(rows)
    correct_ep = sum(v for k, v in ep.items()
                     if k in ('served_correct', 'regen_correct',
                              'served_repaired_proxy', 'regen_template_proxy'))
    summary[name] = {
        'per_tick': [{k: v for k, v in r.items() if k != '_detail'}
                     for r in rows],
        'total_fallbacks': sum(r['fallback'] for r in rows),
        'total_contract_violating_serves_proxy':
            sum(r['wrong_proxy'] for r in rows),
        'total_partially_stale_serves_proxy':
            sum(r['decayed_proxy'] for r in rows),
        'total_abstains': sum(r['abstain'] for r in rows),
        'endpoint_breakdown': ep,
        'endpoint_correct_deliveries': correct_ep,
    }

print('\n================ closed-loop frontier (3 monthly ticks) ================')
print(f'{"policy":20s} {"fallbacks":>9s} {"viol*":>7s} {"p-stale*":>8s} '
      f'{"abstains":>8s} {"correct@T":>9s}')
for name in policies:
    s = summary[name]
    print(f'{name:20s} {s["total_fallbacks"]:9d} '
          f'{s["total_contract_violating_serves_proxy"]:7d} '
          f'{s["total_partially_stale_serves_proxy"]:8d} '
          f'{s["total_abstains"]:8d} {s["endpoint_correct_deliveries"]:9d}')
print('(* deterministic value proxy over all ticks: a serve whose asserted '
      'values are no\n longer truthy (viol) or whose non-asserted certificate '
      'claims are (p-stale).\n correct@T = correct deliveries at the 20260801 '
      'endpoint, human labels where they exist.)')
for name in policies:
    print(f'\n{name} endpoint breakdown:',
          summary[name]['endpoint_breakdown'])
    print(f'{name} per-tick:', summary[name]['per_tick'])

out = {'ticks': TICKS, 'policies': summary,
       'validations': {
           'V1_endpoint_reproduction': {'expected': expect, 'got': got},
           'V2_writeback_soundness': {k: v for k, v in wb_check.items()},
           'V3_accounting_complete': True},
       'notes': [
           'intermediate-tick regenerations are template-composed from the '
           'current truthy values of the same rebuilt snapshots; the 20260801 '
           'regenerations are the real cached LLM outputs with human labels',
           'wrong/decayed serves at intermediate ticks use the deterministic '
           'value proxy (asserted/claimed value no longer truthy); the '
           'endpoint breakdown uses the human answer labels and incoming-query '
           'drift labels',
           'PVSB write-back adds an evidence-checked retrieval alias, so a '
           'regenerated miss becomes a cache hit at later ticks']}
json.dump(out, open(f'{DATA}/closed_loop_system.json', 'w'),
          ensure_ascii=False, indent=1)
print('\nsaved data/closed_loop_system.json')

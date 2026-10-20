#!/usr/bin/env python3
"""s32_tableb_regenerate.py -- P3 / Table B: endpoint-fair regeneration.

The reviewer (round "accept the reported results", item III.1) asked us to
split the closed-loop frontier into a pure mechanism table (Table A) and an
endpoint answer-quality table (Table B) in which every delivered answer comes
from the SAME generation source: same input question, same target evidence,
same model prompt, same acceptance rule, same human annotation standard.

The s25 closed loop still delivers template-composed answers in two places:
  (i)  TTL-1mo: every entry is blind-refreshed at every tick; the 20260801
       endpoint falls back to the template whenever no real cached LLM
       regeneration exists for that pid (89 of 154 requests).
  (ii) the gated loops: a repair at an intermediate tick (20260501/20260701)
       writes back a template answer whose certificate is later served at the
       endpoint ("served", origin != base).

This script replays all arms, records the exact endpoint delivery source of
every request, collects the (pid, origin_tick) keys that were template
deliveries in ANY arm (deduplicated across arms -- one generation reused by
every arm that needs it, which is exactly the reviewer's "same reuse rule"),
and regenerates each key with the REAL model using the s9 prompt protocol
against the evidence of the ORIGIN TICK snapshot:

  target values    := truthy values of the property in the origin-tick
                      Wikidata entity view
  target sentences := up to 3 origin-tick page sentences mentioning a value

Keys whose origin-tick view has no truthy value are recorded as
no_current_evidence (Table B will count them as abstentions -- honest
accounting, no model call).  Keys at 20260801 for which s9 already attempted
a real regeneration that failed acceptance are re-attempted once (same
protocol; acceptance is stochastic).

Usage:
  python3 s32_tableb_regenerate.py --dry   # compute key set only, no API
  python3 s32_tableb_regenerate.py         # full run (checkpointed)

Outputs:
  data/tableb_endpoint_map.json   per arm per pid: delivery source + answer
  data/tableb_regeneration.json   key "pid|tick" -> regeneration record
"""
import json, os, re, sys, time, urllib.request, importlib.util, contextlib
from concurrent.futures import ThreadPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.normpath(os.path.join(HERE, '..', 'data'))
REB = os.path.normpath(os.path.join(HERE, '..', '..', '12_real_dynamics_audit',
                                   'data', 'rebuilt_2026'))
T = '20260801'

KEY = os.environ['KIMI_API_KEY']
BASE = os.environ['KIMI_BASE_URL'].rstrip('/')
MODEL = 'kimi-k2-turbo-preview'
SYS_A = ('You are a careful question-answering assistant. Answer strictly from '
         'the provided evidence. Reply with one short phrase or sentence only; '
         'do not add caveats.')

# ---------------------------------------------------------- load s25 engine --
spec = importlib.util.spec_from_file_location(
    's25', os.path.join(HERE, 's25_closed_loop_system.py'))
s25 = importlib.util.module_from_spec(spec)
with open(os.devnull, 'w') as dn, contextlib.redirect_stdout(dn):
    spec.loader.exec_module(s25)     # runs V1/V2/V3 silently; deterministic

ARMS = ['similarity-static', 'TTL-1mo', 'page-change-loop',
        'pvsb-strict-loop', 'pvsb-asserted-loop']

# ------------------------------------------------------------- replication --
def norm(s):
    return re.sub(r'[,\s]+', '', s.lower())

def accepted(answer, values_text):
    na = norm(answer)
    for vt in values_text:
        nv = norm(vt)
        if nv and nv in na:
            return True
        m = re.match(r'(\d{4})', vt)
        if m and m.group(1) in answer:
            return True
    return False

def chat(prompt, system, max_tokens=160, tries=4):
    payload = {'model': MODEL, 'max_tokens': max_tokens,
               'thinking': {'type': 'disabled'},
               'system': system,
               'messages': [{'role': 'user', 'content': prompt}]}
    for a in range(tries):
        try:
            req = urllib.request.Request(
                BASE + '/messages', data=json.dumps(payload).encode(),
                headers={'x-api-key': KEY, 'anthropic-version': '2023-06-01',
                         'content-type': 'application/json'}, method='POST')
            with urllib.request.urlopen(req, timeout=60) as r:
                d = json.loads(r.read())
            return ''.join(b.get('text', '') for b in d.get('content', [])
                           if b.get('type') == 'text').strip()
        except Exception:
            time.sleep(2 + a * 3)
    return None

labels1 = json.load(open(f'{DATA}/labels_20260801.json'))
labels0 = json.load(open(f'{DATA}/labels_20260401.json'))
extra_p = f'{DATA}/labels_extra.json'
labelsx = json.load(open(extra_p)) if os.path.exists(extra_p) else {}

def resolve_label(qid_str):
    qn = str(qid_str).lstrip('Q')
    if qn in labels1:
        return labels1[qn]
    if qn in labels0:
        return labels0[qn]
    return labelsx.get('Q' + qn, 'Q' + qn)

def tick_evidence(x, tick):
    """s9-style target evidence, but from the ORIGIN-TICK snapshot."""
    ent = s25.snaps_ent[tick].get(str(x['qid']))
    vtexts, seen = [], set()
    if ent is not None:
        for rank, snak, val in ent['claims'].get(x['prop'], []):
            if rank in ('normal', 'preferred') and snak == 'value' \
                    and val is not None:
                k = s25.val_key(val)
                if k in seen:
                    continue
                seen.add(k)
                if isinstance(val, dict) and 'id' in val:
                    vtexts.append(resolve_label(val['id']))
                elif isinstance(val, dict) and 'amount' in val:
                    vtexts.append(str(val['amount']).lstrip('+'))
                elif isinstance(val, dict) and 'time' in val:
                    vtexts.append(val['time'][1:11])
                else:
                    vtexts.append(str(val))
    sents = []
    title = x.get('page_title')
    wt = s25.pages[tick].get(title) if title else None
    if wt:
        for s in s25.sentences(s25.wikitext_to_plain(wt)):
            if any(norm(vt) and norm(vt) in norm(s) for vt in vtexts):
                sents.append(s)
            if len(sents) >= 3:
                break
    return vtexts, sents

def template_answer(x, tick):
    """what the s25 template fallback actually delivered at `tick`."""
    ans, _ = s25.template_fallback(x, tick)
    return ans

# ------------------------------------------------- endpoint delivery map ---
print('replaying arms to map endpoint delivery sources ...', flush=True)
endpoint_map, template_keys = {}, {}
for arm in ARMS:
    rows = s25.run_policy(arm)
    det = rows[-1]['_detail']
    em = {}
    for pid, (how, origin) in det.items():
        x = s25.by_pid[pid]
        if how == 'abstain':
            em[pid] = {'how': how, 'origin': None, 'source': 'abstain',
                       'answer': None}
        elif how == 'regenerated':
            r = s25.regen.get(str(pid))
            if origin == T and r is not None and r.get('regen_ok'):
                em[pid] = {'how': how, 'origin': origin,
                           'source': 'real-regen', 'answer': r['regen_answer']}
            else:
                em[pid] = {'how': how, 'origin': origin, 'source': 'template',
                           'answer': template_answer(x, origin)}
                template_keys.setdefault((pid, origin), []).append(arm)
        else:                                    # served*
            if origin == s25.SNAPS[0]:
                em[pid] = {'how': how, 'origin': origin,
                           'source': 'base-cached',
                           'answer': s25.gen[str(pid)]['answer']}
            else:
                em[pid] = {'how': how, 'origin': origin, 'source': 'template',
                           'answer': template_answer(x, origin)}
                template_keys.setdefault((pid, origin), []).append(arm)
    endpoint_map[arm] = {str(k): v for k, v in em.items()}

from collections import Counter
print('\nendpoint delivery source per arm:')
for arm in ARMS:
    c = Counter(v['source'] for v in endpoint_map[arm].values())
    print(f'  {arm:20s} {dict(c)}')
tc = Counter(t for _, t in template_keys)
print(f'\ntemplate keys to regenerate: {len(template_keys)}  per tick: {dict(tc)}')

json.dump(endpoint_map, open(f'{DATA}/tableb_endpoint_map.json', 'w'),
          ensure_ascii=False, indent=1)
print('saved data/tableb_endpoint_map.json')

if '--dry' in sys.argv:
    sys.exit(0)

# --------------------------------------------------------------- regenerate --
out_path = f'{DATA}/tableb_regeneration.json'
regen = json.load(open(out_path)) if os.path.exists(out_path) else {}

def regen_one(key):
    pid, tick = key
    x = s25.by_pid[pid]
    vtexts, sents = tick_evidence(x, tick)
    base = {'pid': pid, 'tick': tick, 'question': x['question'],
            'prop_name': x['prop_name'],
            'template_answer': template_answer(x, tick),
            'target_values_text': vtexts, 'target_sentences': sents,
            'arms_needing': sorted(set(template_keys[key]))}
    if not vtexts:
        return key, {**base, 'regen_answer': None, 'regen_ok': False,
                     'no_current_evidence': True}
    ev_lines = [f"- {x['prop_name']}: {', '.join(vtexts)}"]
    ev_lines += [f'- {s}' for s in sents]
    prompt = (f"Question: {x['question']}\n\nEvidence:\n"
              + '\n'.join(ev_lines) + '\n\nAnswer:')
    ans = chat(prompt, SYS_A)
    ok = ans is not None and accepted(ans, vtexts)
    if not ok:
        ans2 = chat(prompt + '\n(Your previous reply did not use the '
                    'evidence values. Answer again, naming the value from '
                    'the evidence.)', SYS_A)
        if ans2 is not None and accepted(ans2, vtexts):
            ans, ok = ans2, True
    return key, {**base, 'regen_answer': ans, 'regen_ok': bool(ok),
                 'no_current_evidence': False}

todo = [k for k in sorted(template_keys) if f'{k[0]}|{k[1]}' not in regen]
print(f'\ntodo: {len(todo)} / {len(template_keys)}', flush=True)
done = 0
with ThreadPoolExecutor(max_workers=6) as ex:
    for key, r in ex.map(regen_one, todo):
        regen[f'{key[0]}|{key[1]}'] = r
        done += 1
        if done % 10 == 0:
            json.dump(regen, open(out_path, 'w'), ensure_ascii=False)
            print(f'checkpoint {done}/{len(todo)}', flush=True)
json.dump(regen, open(out_path, 'w'), ensure_ascii=False, indent=1)

n_no = sum(1 for r in regen.values() if r['no_current_evidence'])
n_ok = sum(1 for r in regen.values() if r['regen_ok'])
n_fail = len(regen) - n_no - n_ok
print(f'DONE tableb regen: {len(regen)} keys | regen_ok {n_ok} | '
      f'accept-fail {n_fail} | no_current_evidence {n_no}')
print('saved data/tableb_regeneration.json')

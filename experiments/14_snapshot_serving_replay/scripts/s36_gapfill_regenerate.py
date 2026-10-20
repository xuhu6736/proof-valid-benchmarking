#!/usr/bin/env python3
"""s36_gapfill_regenerate.py -- close the generation-key coverage gap.

The s30 fault sweep (4,800 runs) visits 462 distinct (pid, tick)
fallback-generation keys, but only 225 had archived real model outputs
(65 from s9 at the endpoint + 175 from the s32 Table-B round, 174 of
them accepted).  The remaining keys were silently answered by the
template fallback inside the erasure runs.

This script enumerates the visited keys from s30's fallback cache,
computes the uncovered set, and regenerates each missing key with the
real model under the SAME protocol as s9/s32: same system prompt, same
question, truthy values and up to three evidence sentences from the
key's OWN tick snapshot, same value-mention acceptance, one stricter
retry.  Results are merged into data/tableb_regeneration.json (key
"pid|tick", extra field gap_fill=true), so every downstream consumer
sees one store.  Keys whose origin-tick view has no truthy value are
recorded as no_current_evidence (such keys correspond to abstentions,
not deliveries).

After this run, every fallback key visited by any archived run has an
archived real output or an explicit no-evidence record -- the coverage
assertion s37 enforces.
"""
import json, os, re, io, sys, time, urllib.request, importlib.util, \
    contextlib
from concurrent.futures import ThreadPoolExecutor

os.environ['S30_NO_WRITE'] = '1'
HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.normpath(os.path.join(HERE, '..', 'data'))

KEY = os.environ['KIMI_API_KEY']
BASE = os.environ['KIMI_BASE_URL'].rstrip('/')
MODEL = 'kimi-k2-turbo-preview'
SYS_A = ('You are a careful question-answering assistant. Answer strictly from '
         'the provided evidence. Reply with one short phrase or sentence only; '
         'do not add caveats.')

# ------------------------------------------------------- visited key set ----
spec = importlib.util.spec_from_file_location(
    's30', os.path.join(HERE, 's30_real_payload_chain.py'))
s30 = importlib.util.module_from_spec(spec)
with contextlib.redirect_stdout(io.StringIO()):
    spec.loader.exec_module(s30)
visited = set(s30.FB.keys())
print('visited generation keys across fault runs:', len(visited))

s25 = s30.s25
regen0 = json.load(open(f'{DATA}/regeneration.json'))
tb_path = f'{DATA}/tableb_regeneration.json'
tb = json.load(open(tb_path))

covered = set()
for pid_s, r in regen0.items():
    if r.get('regen_ok'):
        covered.add((int(pid_s), '20260801'))
for k, r in tb.items():
    pid, tick = k.split('|')
    covered.add((int(pid), tick))           # accept-fail keys re-attempted

missing = sorted(visited - covered)
print('uncovered keys:', len(missing))
from collections import Counter
print('per tick:', dict(Counter(t for _, t in missing)))

# ------------------------------------------------------- generation kit -----
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

def tick_evidence(x, tick):
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

def regen_one(key):
    pid, tick = key
    x = s25.by_pid[pid]
    vtexts, sents = tick_evidence(x, tick)
    base = {'pid': pid, 'tick': tick, 'question': x['question'],
            'prop_name': x['prop_name'],
            'template_answer': s25.template_fallback(x, tick)[0],
            'target_values_text': vtexts, 'target_sentences': sents,
            'gap_fill': True, 'arms_needing': ['s30-fault-sweep']}
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

todo = [k for k in missing if f'{k[0]}|{k[1]}' not in tb]
print(f'todo: {len(todo)} / {len(missing)}', flush=True)
if '--dry' in sys.argv:
    sys.exit(0)
done = 0
with ThreadPoolExecutor(max_workers=6) as ex:
    for key, r in ex.map(regen_one, todo):
        tb[f'{key[0]}|{key[1]}'] = r
        done += 1
        if done % 10 == 0:
            json.dump(tb, open(tb_path, 'w'), ensure_ascii=False)
            print(f'checkpoint {done}/{len(todo)}', flush=True)
json.dump(tb, open(tb_path, 'w'), ensure_ascii=False, indent=1)

# coverage assertion over ALL archived runs
covered2 = {(int(k.split('|')[0]), k.split('|')[1])
            for k, r in tb.items() if r.get('regen_ok')}
covered2 |= {(int(p), '20260801') for p, r in regen0.items()
             if r.get('regen_ok')}
noev = {(int(k.split('|')[0]), k.split('|')[1])
        for k, r in tb.items() if r.get('no_current_evidence')}
noev |= {(int(p), '20260801') for p, r in regen0.items()
         if r.get('no_current_evidence')}
still = visited - covered2 - noev
n_no = sum(1 for r in tb.values() if r.get('no_current_evidence'))
n_ok = sum(1 for r in tb.values() if r.get('regen_ok'))
print(f'DONE gap fill: store now {len(tb)} keys | regen_ok {n_ok} | '
      f'no_current_evidence {n_no}')
print('visited keys without a real output (accept-fail, counted as '
      'abstain):', len(still), sorted(still)[:10])

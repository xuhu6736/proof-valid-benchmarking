#!/usr/bin/env python3
"""s41_gpt_regenerate.py -- second-generator robustness archive.

Reviewer suggestion (round 77 follow-up): add a second generation model so
that the closed-loop conclusions do not rest on a single generator
(kimi-k2-turbo-preview).  This script re-runs EVERY model-generated endpoint
delivery source of Table B with GPT (via the Poe OpenAI-compatible API),
using the EXACT s9/s32 prompt protocol:

  * same system prompt (SYS_A), same user prompt shape, same evidence
    construction (origin-tick snapshot for template deliveries, endpoint
    tick 20260801 for s9-style regenerations), same max_tokens=160,
    same value-mention acceptance rule, same one-time re-ask nudge.

Scope: all 411 keys of the merged regeneration store
(data/tableb_regeneration.json: 175 Table-B endpoint keys from s32 plus 236
fault-sweep gap-fill keys from s36) plus all 65 s9 endpoint regeneration
keys.  Evidence shown to GPT is taken verbatim from the archived kimi
records, so both generators saw byte-identical prompts.  Base-cached answers
are identical across policy arms and are not regenerated.

Usage:
  POE_API_KEY=... python3 s41_gpt_regenerate.py --dry   # key set only
  POE_API_KEY=... python3 s41_gpt_regenerate.py         # full run (checkpointed)

Outputs:
  data/tableb_regeneration_gpt.json    s32 keys, GPT answers
  data/regeneration_gpt.json           s9 endpoint keys, GPT answers
"""
import json, os, sys, time, urllib.request, importlib.util, contextlib
from concurrent.futures import ThreadPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.normpath(os.path.join(HERE, '..', 'data'))
T = '20260801'
MODEL = os.environ.get('GPT_MODEL', 'gpt-5-mini')
BASE = os.environ.get('POE_BASE_URL', 'https://api.poe.com/v1').rstrip('/')
KEY = os.environ['POE_API_KEY']

# ------------------------------------------------- reuse s32 (dry) engine --
_argv = sys.argv
sys.argv = ['s32_tableb_regenerate.py', '--dry']
spec = importlib.util.spec_from_file_location(
    's32', os.path.join(HERE, 's32_tableb_regenerate.py'))
s32 = importlib.util.module_from_spec(spec)
with open(os.devnull, 'w') as dn, contextlib.redirect_stdout(dn):
    try:
        spec.loader.exec_module(s32)  # dry run: maps endpoints, no API calls
    except SystemExit:                # s32's --dry path ends with sys.exit(0)
        pass
sys.argv = _argv

SYS_A = s32.SYS_A
accepted = s32.accepted

# ------------------------------------------------------------- GPT client --
def chat(prompt, system, max_tokens=160, tries=4):
    payload = {'model': MODEL, 'max_tokens': max_tokens,
               'messages': [{'role': 'system', 'content': system},
                            {'role': 'user', 'content': prompt}]}
    for a in range(tries):
        try:
            req = urllib.request.Request(
                BASE + '/chat/completions', data=json.dumps(payload).encode(),
                headers={'Authorization': 'Bearer ' + KEY,
                         'content-type': 'application/json'}, method='POST')
            with urllib.request.urlopen(req, timeout=90) as r:
                d = json.loads(r.read())
            return d['choices'][0]['message']['content'].strip()
        except Exception as e:
            print(f'  retry {a+1}: {e}', flush=True)
            time.sleep(2 + a * 3)
    return None

def regen_with_protocol(question, prop_name, vtexts, sents):
    """Exact s9/s32 prompt protocol incl. the one-time acceptance nudge."""
    if not vtexts:
        return None, False
    ev_lines = [f"- {prop_name}: {', '.join(vtexts)}"]
    ev_lines += [f'- {s}' for s in sents]
    prompt = (f"Question: {question}\n\nEvidence:\n"
              + '\n'.join(ev_lines) + '\n\nAnswer:')
    ans = chat(prompt, SYS_A)
    ok = ans is not None and accepted(ans, vtexts)
    if not ok:
        ans2 = chat(prompt + '\n(Your previous reply did not use the '
                    'evidence values. Answer again, naming the value from '
                    'the evidence.)', SYS_A)
        if ans2 is not None and accepted(ans2, vtexts):
            ans, ok = ans2, True
    return ans, bool(ok)

# ---------------------------------- part 1: merged regeneration store -----
kimi_store = json.load(open(f'{DATA}/tableb_regeneration.json'))

def regen_one(k):
    r0 = kimi_store[k]                       # archived kimi record
    base = {'pid': r0['pid'], 'tick': r0['tick'], 'question': r0['question'],
            'prop_name': r0['prop_name'],
            'template_answer': r0.get('template_answer'),
            'target_values_text': r0['target_values_text'],
            'target_sentences': r0['target_sentences'],
            'generator': MODEL,
            'kimi_answer': r0.get('regen_answer'),
            'kimi_ok': r0.get('regen_ok')}
    if r0.get('no_current_evidence') or not r0['target_values_text']:
        return k, {**base, 'regen_answer': None, 'regen_ok': False,
                   'no_current_evidence': True}
    ans, ok = regen_with_protocol(r0['question'], r0['prop_name'],
                                  r0['target_values_text'],
                                  r0['target_sentences'])
    return k, {**base, 'regen_answer': ans, 'regen_ok': ok,
               'no_current_evidence': False}

out1 = f'{DATA}/tableb_regeneration_gpt.json'
regen = json.load(open(out1)) if os.path.exists(out1) else {}
todo = [k for k in sorted(kimi_store) if k not in regen]
print(f'part 1 (merged store keys) todo: {len(todo)} / {len(kimi_store)}',
      flush=True)
if '--dry' in sys.argv:
    n9 = len(json.load(open(f'{DATA}/regeneration.json')))
    print(f'part 2 (s9 endpoint keys) would regenerate: {n9}')
    sys.exit(0)

done = 0
with ThreadPoolExecutor(max_workers=6) as ex:
    for key, r in ex.map(regen_one, todo):
        regen[key] = r
        done += 1
        if done % 10 == 0:
            json.dump(regen, open(out1, 'w'), ensure_ascii=False)
            print(f'checkpoint p1 {done}/{len(todo)}', flush=True)
json.dump(regen, open(out1, 'w'), ensure_ascii=False, indent=1)
n_no = sum(1 for r in regen.values() if r['no_current_evidence'])
n_ok = sum(1 for r in regen.values() if r['regen_ok'])
print(f'part 1 DONE: {len(regen)} keys | ok {n_ok} | fail '
      f'{len(regen)-n_no-n_ok} | no-evidence {n_no}', flush=True)

# --------------------------------------- part 2: s9 endpoint regenerations --
s9old = json.load(open(f'{DATA}/regeneration.json'))
out2 = f'{DATA}/regeneration_gpt.json'
regen9 = json.load(open(out2)) if os.path.exists(out2) else {}

def regen9_one(pid_str):
    pid = int(pid_str)
    x = s32.s25.by_pid[pid]
    vtexts, sents = s32.tick_evidence(x, T)          # endpoint-tick evidence
    if not vtexts:
        return pid_str, {'target_values_text': vtexts,
                         'target_sentences': sents, 'regen_answer': None,
                         'regen_ok': False, 'no_current_evidence': True,
                         'generator': MODEL}
    ans, ok = regen_with_protocol(x['question'], x['prop_name'], vtexts, sents)
    return pid_str, {'target_values_text': vtexts, 'target_sentences': sents,
                     'regen_answer': ans, 'regen_ok': ok,
                     'no_current_evidence': False, 'generator': MODEL}

todo9 = [p for p in sorted(s9old, key=int) if p not in regen9]
print(f'part 2 (s9 keys) todo: {len(todo9)} / {len(s9old)}', flush=True)
done = 0
with ThreadPoolExecutor(max_workers=6) as ex:
    for pid_str, r in ex.map(regen9_one, todo9):
        regen9[pid_str] = r
        done += 1
        if done % 10 == 0:
            json.dump(regen9, open(out2, 'w'), ensure_ascii=False)
            print(f'checkpoint p2 {done}/{len(todo9)}', flush=True)
json.dump(regen9, open(out2, 'w'), ensure_ascii=False, indent=1)
n_no = sum(1 for r in regen9.values() if r.get('no_current_evidence'))
n_ok = sum(1 for r in regen9.values() if r['regen_ok'])
print(f'part 2 DONE: {len(regen9)} keys | ok {n_ok} | fail '
      f'{len(regen9)-n_no-n_ok} | no-evidence {n_no}', flush=True)
print('saved data/tableb_regeneration_gpt.json and data/regeneration_gpt.json')

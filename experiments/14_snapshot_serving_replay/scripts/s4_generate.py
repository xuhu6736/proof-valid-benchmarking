#!/usr/bin/env python3
"""Step 4: LLM base-answer generation + paraphrase queries (Kimi gateway).

For every panel pair:
  answer  := LLM(question, evidence S claims + evidence B sentences)   [base period]
  para    := LLM paraphrase of the question (for the similarity gate)
Validation: an answer is accepted iff it mentions >=1 cited value text
(normalised).  One retry with a stricter prompt; otherwise gen_fail.

Output: data/generation.json  (checkpointed; safe to resume)
"""
import json, os, re, time, urllib.request
from concurrent.futures import ThreadPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.normpath(os.path.join(HERE, '..', 'data'))

KEY = os.environ['KIMI_API_KEY']
BASE = os.environ['KIMI_BASE_URL'].rstrip('/')
MODEL = 'kimi-k2-turbo-preview'


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


def norm(s):
    s = s.lower()
    s = re.sub(r'[,\s]+', '', s)
    return s


def accepted(answer, values_text):
    na = norm(answer)
    for vt in values_text:
        nv = norm(vt)
        if nv and (nv in na):
            return True
        m = re.match(r'(\d{4})', vt)          # year-only match for dates
        if m and m.group(1) in answer:
            return True
    return False


SYS_A = ('You are a careful question-answering assistant. Answer strictly from '
         'the provided evidence. Reply with one short phrase or sentence only; '
         'do not add caveats.')
SYS_P = ('Rewrite the user question in different words while preserving its '
         'meaning exactly. Output only the rewritten question.')


def gen_one(x):
    ev_lines = [f"- {x['prop_name']}: {', '.join(x['values_text'])}"]
    for s in x['certificate']['sentences']:
        ev_lines.append(f'- {s["text"]}')
    prompt = (f"Question: {x['question']}\n\nEvidence:\n" + '\n'.join(ev_lines) +
              '\n\nAnswer:')
    ans = chat(prompt, SYS_A)
    ok = ans is not None and accepted(ans, x['values_text'])
    if not ok:
        ans2 = chat(prompt + '\n(Your previous reply did not use the evidence '
                    'values. Answer again, naming the value from the evidence.)',
                    SYS_A)
        if ans2 is not None and accepted(ans2, x['values_text']):
            ans, ok = ans2, True
    para = chat(x['question'], SYS_P, max_tokens=80)
    return x['pid'], {'answer': ans, 'answer_ok': bool(ok), 'paraphrase': para}


def main():
    panel = json.load(open(f'{DATA}/evidence.json'))
    pilot = int(os.environ.get('PILOT', '0'))
    if pilot:
        panel = panel[:pilot]
        out_path = f'{DATA}/generation_pilot.json'
    else:
        out_path = f'{DATA}/generation.json'
    gen = json.load(open(out_path)) if os.path.exists(out_path) else {}
    todo = [x for x in panel if str(x['pid']) not in gen]
    print('todo:', len(todo), '/', len(panel), flush=True)
    done = 0
    with ThreadPoolExecutor(max_workers=6) as ex:
        for pid, r in ex.map(gen_one, todo):
            gen[str(pid)] = r
            done += 1
            if done % 20 == 0:
                json.dump(gen, open(out_path, 'w'), ensure_ascii=False)
                print('checkpoint', done, '/', len(todo), flush=True)
    json.dump(gen, open(out_path, 'w'), ensure_ascii=False, indent=1)
    ok = sum(1 for r in gen.values() if r['answer_ok'])
    np_ = sum(1 for r in gen.values() if r['paraphrase'])
    print('DONE answers:', len(gen), 'accepted:', ok, 'with paraphrase:', np_)


if __name__ == '__main__':
    main()

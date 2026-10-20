#!/usr/bin/env python3
"""Translate the 154 questions + answers into Chinese (cached, batched)."""
import json, os, time, urllib.request
from concurrent.futures import ThreadPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.normpath(os.path.join(HERE, '..', 'data'))
KEY = os.environ['KIMI_API_KEY']
BASE = os.environ['KIMI_BASE_URL'].rstrip('/')

SYS = ('You are a precise bilingual translator. For each numbered item, '
       'translate the English question and answer into Simplified Chinese. '
       'Keep proper nouns in their common Chinese form if one exists, '
       'otherwise keep the original. Output ONLY lines in the format '
       '"<n>Q: <chinese question>" and "<n>A: <chinese answer>".')


def chat(prompt, tries=4):
    payload = {'model': 'kimi-k2-turbo-preview', 'max_tokens': 1500,
               'thinking': {'type': 'disabled'}, 'system': SYS,
               'messages': [{'role': 'user', 'content': prompt}]}
    for a in range(tries):
        try:
            req = urllib.request.Request(
                BASE + '/messages', data=json.dumps(payload).encode(),
                headers={'x-api-key': KEY, 'anthropic-version': '2023-06-01',
                         'content-type': 'application/json'}, method='POST')
            with urllib.request.urlopen(req, timeout=90) as r:
                d = json.loads(r.read())
            return ''.join(b.get('text', '') for b in d.get('content', [])
                           if b.get('type') == 'text').strip()
        except Exception:
            time.sleep(2 + a * 3)
    return None


def main():
    recs = json.load(open(f'{DATA}/annotation_data.json'))
    out_path = f'{DATA}/translations.json'
    tr = json.load(open(out_path)) if os.path.exists(out_path) else {}
    todo = [r for r in recs if str(r['pid']) not in tr]
    print('todo entries:', len(todo))
    batches = [todo[i:i + 5] for i in range(0, len(todo), 5)]

    def work(batch):
        lines = []
        for r in batch:
            lines.append(f"{r['pid']}Q: {r['question']}")
            lines.append(f"{r['pid']}A: {r['answer']}")
        txt = chat('\n'.join(lines))
        out = {}
        if txt:
            for ln in txt.splitlines():
                ln = ln.strip()
                for r in batch:
                    p = str(r['pid'])
                    if ln.startswith(p + 'Q:'):
                        out.setdefault(p, {})['q_zh'] = ln[len(p) + 2:].strip()
                    elif ln.startswith(p + 'A:'):
                        out.setdefault(p, {})['a_zh'] = ln[len(p) + 2:].strip()
        return out

    done = 0
    with ThreadPoolExecutor(max_workers=5) as ex:
        for out in ex.map(work, batches):
            for p, v in out.items():
                if v.get('q_zh') and v.get('a_zh'):
                    tr[p] = v
            done += 1
            if done % 5 == 0:
                json.dump(tr, open(out_path, 'w'), ensure_ascii=False)
                print('batch', done, '/', len(batches), 'translated:', len(tr), flush=True)
    json.dump(tr, open(out_path, 'w'), ensure_ascii=False, indent=1)
    print('DONE translated:', len(tr), '/ 154')


if __name__ == '__main__':
    main()

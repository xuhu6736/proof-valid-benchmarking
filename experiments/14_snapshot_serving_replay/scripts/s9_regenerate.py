#!/usr/bin/env python3
"""Step 9 (package 1d): regeneration closed loop for declined/missed requests.

The main replay stops at the routing decision ("not cache-served").  Here we
close the loop: every request NOT served under the strict contract (7
similarity misses + 58 gate declines = 65) is answered by fresh generation
from TARGET-snapshot evidence:

  target values    := truthy values of the property in the 2026-08-01
                      Wikidata entity view (labels from labels_20260801.json)
  target sentences := up to 3 sentences of the 2026-08-01 page mentioning a
                      target value text (same extraction pipeline as s5)

If the target view has no truthy value for the property, the request is
recorded as no_current_evidence (the system abstains honestly) and no model
call is made.  Otherwise the answer is generated with the same prompt shape
as s4 and validated by value mention (normalized containment, year-only for
dates).  One stricter retry, as in s4.

Output: data/regeneration.json  (checkpointed; safe to resume)
"""
import json, os, re, time, urllib.request
from concurrent.futures import ThreadPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.normpath(os.path.join(HERE, '..', 'data'))
REB = os.path.normpath(os.path.join(HERE, '..', '..', '12_real_dynamics_audit',
                                   'data', 'rebuilt_2026'))

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


# --- target-page sentence extraction (same pipeline as s5) ---
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


def val_key(v):
    if isinstance(v, dict):
        if 'id' in v: return 'Q:' + str(v['id'])
        if 'amount' in v: return 'A:' + str(v['amount'])
        if 'time' in v: return 'T:' + str(v['time'])[:11]
        if 'text' in v: return 'S:' + str(v['text'])
    return 'S:' + str(v)


SYS_A = ('You are a careful question-answering assistant. Answer strictly from '
         'the provided evidence. Reply with one short phrase or sentence only; '
         'do not add caveats.')


def main():
    panel = json.load(open(f'{DATA}/evidence.json'))
    gen = json.load(open(f'{DATA}/generation.json'))
    replay = {r['pid']: r for r in json.load(open(f'{DATA}/replay.json'))}
    e1 = json.load(open(f'{REB}/entities_v2_20260801.json'))
    pages1 = json.load(open(f'{REB}/pages_20260801.json'))
    labels1 = json.load(open(f'{DATA}/labels_20260801.json'))
    labels0 = json.load(open(f'{DATA}/labels_20260401.json'))
    extra_p = f'{DATA}/labels_extra.json'
    labelsx = json.load(open(extra_p)) if os.path.exists(extra_p) else {}

    def resolve_label(qid_str):
        """Label for a Wikidata QID: target dump -> base dump -> extra fetch."""
        qn = str(qid_str).lstrip('Q')
        if qn in labels1:
            return labels1[qn]
        if qn in labels0:
            return labels0[qn]
        q = 'Q' + qn
        return labelsx.get(q, q)

    usable = [x for x in panel if gen[str(x['pid'])]['answer_ok']
              and gen[str(x['pid'])]['paraphrase']]
    # strict contract: served iff hit AND all claims AND all sentences survive
    try:
        summ = json.load(open(f'{DATA}/replay_summary.json'))
        tau = summ.get('tau', 0.8364)
    except Exception:
        tau = 0.8364
    if isinstance(tau, dict):  # summary may store richer structure
        tau = tau.get('value', 0.8364)
    def served_strict(x):
        r = replay[x['pid']]
        return (r['sim_para'] >= tau and r['claims_dead'] == 0
                and r['sents_dead'] == 0)
    not_served = [x for x in usable if not served_strict(x)]
    print('not served under strict contract:', len(not_served))

    def target_evidence(x):
        ent = e1.get(x['qid'])
        vals, vtexts = [], []
        if ent is not None:
            seen = set()
            for rank, snak, val in ent['claims'].get(x['prop'], []):
                if rank in ('normal', 'preferred') and snak == 'value' \
                        and val is not None:
                    k = val_key(val)
                    if k in seen:
                        continue
                    seen.add(k)
                    vals.append(val)
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
        if title and title in pages1:
            for s in sentences(wikitext_to_plain(pages1[title])):
                if any(norm(vt) and norm(vt) in norm(s) for vt in vtexts):
                    sents.append(s)
                if len(sents) >= 3:
                    break
        return vtexts, sents

    out_path = f'{DATA}/regeneration.json'
    regen = json.load(open(out_path)) if os.path.exists(out_path) else {}

    def regen_one(x):
        pid = str(x['pid'])
        vtexts, sents = target_evidence(x)
        cached = gen[pid]['answer']
        if not vtexts:
            return x['pid'], {'target_values_text': [], 'target_sentences': [],
                              'regen_answer': None, 'regen_ok': False,
                              'no_current_evidence': True,
                              'differs_from_cached': None}
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
        return x['pid'], {'target_values_text': vtexts,
                          'target_sentences': sents,
                          'regen_answer': ans, 'regen_ok': bool(ok),
                          'no_current_evidence': False,
                          'differs_from_cached':
                              None if ans is None else norm(ans) != norm(cached)}

    todo = [x for x in not_served if str(x['pid']) not in regen]
    print('todo:', len(todo), '/', len(not_served), flush=True)
    done = 0
    with ThreadPoolExecutor(max_workers=6) as ex:
        for pid, r in ex.map(regen_one, todo):
            regen[str(pid)] = r
            done += 1
            if done % 10 == 0:
                json.dump(regen, open(out_path, 'w'), ensure_ascii=False)
                print('checkpoint', done, '/', len(todo), flush=True)
    json.dump(regen, open(out_path, 'w'), ensure_ascii=False, indent=1)

    n_no = sum(1 for r in regen.values() if r['no_current_evidence'])
    n_ok = sum(1 for r in regen.values() if r['regen_ok'])
    n_diff = sum(1 for r in regen.values() if r['differs_from_cached'])
    print(f'DONE regen: {len(regen)} | no_current_evidence {n_no} | '
          f'regen_ok {n_ok} | differs_from_cached {n_diff}')


if __name__ == '__main__':
    main()

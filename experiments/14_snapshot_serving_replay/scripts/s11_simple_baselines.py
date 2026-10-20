#!/usr/bin/env python3
"""Step 11 (package 2): simple serve-time alternatives to evidence gating.

Same similarity stage, same 147 hits, same human answer labels; each
baseline makes its own serve/decline decision:

  page-change     : decline iff the page wikitext differs at all
  entity-change   : decline iff the entity's claim JSON differs at all
  prop-change     : decline iff the property's truthy value SET differs
                    (pure additions also decline, unlike the claims gate)
  value-in-page   : decline iff any answer-asserted value text no longer
                    appears in the target page plain text (year-only for
                    dates, amount digits for quantities)
  llm-recheck     : decline iff an LLM judge, shown the question, the
                    cached answer, and the target evidence, does not say
                    the answer is still correct (parse failure / missing
                    target evidence => decline)

Metrics per baseline: decayed served (of the 8 decayed hits), correct
hits declined (of 133), indeterminate declined (of 6), total declined.

Deterministic baselines need no model calls; the LLM judge is cached in
data/llm_judge.json (checkpointed; safe to resume).

Output: data/simple_baselines.json
"""
import json, os, re, time, hashlib, urllib.request
from concurrent.futures import ThreadPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.normpath(os.path.join(HERE, '..', 'data'))
REB = os.path.normpath(os.path.join(HERE, '..', '..', '12_real_dynamics_audit',
                                   'data', 'rebuilt_2026'))
TAU = 0.8364

KEY = os.environ.get('KIMI_API_KEY')
BASE = os.environ.get('KIMI_BASE_URL', '').rstrip('/')
MODEL = 'kimi-k2-turbo-preview'


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


def answer_asserts(answer, vtext):
    na, nv = norm(answer), norm(vtext)
    if nv and nv in na:
        return True
    m = re.match(r'(\d{4})', str(vtext))
    if m and m.group(1) in str(answer):
        return True
    return False


def value_in_text(vtext, plain):
    nv, np_ = norm(vtext), norm(plain)
    if nv and nv in np_:
        return True
    m = re.match(r'(\d{4})', str(vtext))
    if m and m.group(1) in plain:
        return True
    return False


SYS_J = ('You are a strict fact-checking auditor. Given a question, a cached '
         'answer from an earlier date, and current evidence, decide whether '
         'the cached answer is still correct according to the current '
         'evidence. Reply with exactly one word: YES or NO.')


def judge(prompt, tries=4):
    payload = {'model': MODEL, 'max_tokens': 10,
               'thinking': {'type': 'disabled'}, 'system': SYS_J,
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


def main():
    panel = json.load(open(f'{DATA}/evidence.json'))
    gen = json.load(open(f'{DATA}/generation.json'))
    lab = {int(k): v for k, v in json.load(open(
        os.path.join(DATA, '..', 'annotation', 'answer_labels.json'))).items()}
    asserted = {int(k): v for k, v in
                json.load(open(f'{DATA}/asserted_claims.json'))['asserted'].items()}
    regen = {int(k): v for k, v in json.load(open(f'{DATA}/regeneration.json')).items()}

    e0 = json.load(open(f'{REB}/entities_v2_20260401.json'))
    e1 = json.load(open(f'{REB}/entities_v2_20260801.json'))
    p0 = json.load(open(f'{REB}/pages_20260401.json'))
    p1 = json.load(open(f'{REB}/pages_20260801.json'))
    usable = [x for x in panel if gen[str(x['pid'])]['answer_ok']
              and gen[str(x['pid'])]['paraphrase']]

    hits = [x for x in usable if lab[x['pid']]['hit']]
    assert len(hits) == 147

    # --- deterministic baselines ---
    dec = {}  # pid -> {baseline: True(serve)/False(decline)}
    for x in hits:
        pid = x['pid']
        ent0, ent1 = e0.get(x['qid']), e1.get(x['qid'])
        w0 = p0.get(x.get('page_title'))
        w1 = p1.get(x.get('page_title'))
        page_same = (w0 is not None and w1 is not None and w0 == w1)
        ent_same = (ent0 is not None and ent1 is not None
                    and ent0.get('claims') == ent1.get('claims'))
        prop_same = (ent0 is not None and ent1 is not None
                     and truthy_keys(ent0, x['prop']) == truthy_keys(ent1, x['prop']))
        plain1 = wikitext_to_plain(w1) if w1 is not None else ''
        ans = gen[str(pid)]['answer']
        ans_vals = [vt for c, vt in zip(x['certificate']['claims'],
                                        x['values_text'])
                    if answer_asserts(ans, vt)]
        val_ok = bool(ans_vals) and all(value_in_text(vt, plain1)
                                        for vt in ans_vals)
        dec[pid] = {'page-change': page_same, 'entity-change': ent_same,
                    'prop-change': prop_same, 'value-in-page': val_ok}

    # --- LLM judge on the same 147 hits ---
    jpath = f'{DATA}/llm_judge.json'
    jcache = json.load(open(jpath)) if os.path.exists(jpath) else {}

    def judge_one(x):
        pid = x['pid']
        r = regen.get(pid)
        if r is not None and not r['no_current_evidence']:
            vtexts, sents = r['target_values_text'], r['target_sentences']
        else:
            # served hits: build target evidence deterministically
            ent1 = e1.get(x['qid'])
            vtexts, sents = [], []
            if ent1 is not None:
                lab1 = json.load(open(f'{DATA}/labels_20260801.json'))
                lab0 = json.load(open(f'{DATA}/labels_20260401.json'))
                lx = json.load(open(f'{DATA}/labels_extra.json'))
                seen = set()
                for rank, snak, val in ent1['claims'].get(x['prop'], []):
                    if rank in ('normal', 'preferred') and snak == 'value' \
                            and val is not None:
                        k = val_key(val)
                        if k in seen:
                            continue
                        seen.add(k)
                        if isinstance(val, dict) and 'id' in val:
                            qn = str(val['id']).lstrip('Q')
                            vtexts.append(lab1.get(qn) or lab0.get(qn)
                                          or lx.get('Q' + qn, 'Q' + qn))
                        elif isinstance(val, dict) and 'amount' in val:
                            vtexts.append(str(val['amount']).lstrip('+'))
                        elif isinstance(val, dict) and 'time' in val:
                            vtexts.append(val['time'][1:11])
                        else:
                            vtexts.append(str(val))
            w1 = p1.get(x.get('page_title'))
            if w1:
                plain1 = wikitext_to_plain(w1)
                from s9_regenerate import sentences  # reuse splitter
                for s in sentences(plain1):
                    if any(norm(vt) and norm(vt) in norm(s) for vt in vtexts):
                        sents.append(s)
                    if len(sents) >= 3:
                        break
        if not vtexts:
            return pid, {'verdict': 'NO_EVIDENCE', 'raw': None}
        ev = [f"- {x['prop_name']}: {', '.join(vtexts)}"] + \
             [f'- {s}' for s in sents]
        prompt = (f"Question: {x['question']}\n"
                  f"Cached answer (from 2026-04-01): {gen[str(pid)]['answer']}\n\n"
                  f"Current evidence (2026-08-01):\n" + '\n'.join(ev) +
                  "\n\nIs the cached answer still correct according to the "
                  "current evidence? Reply YES or NO.")
        raw = judge(prompt)
        v = None
        if raw:
            up = raw.strip().upper()
            if up.startswith('YES'):
                v = 'YES'
            elif up.startswith('NO'):
                v = 'NO'
        return pid, {'verdict': v or 'PARSE_FAIL', 'raw': raw}

    todo = [x for x in hits if str(x['pid']) not in jcache]
    print('judge todo:', len(todo), '/', len(hits), flush=True)
    if todo and KEY and BASE:
        done = 0
        with ThreadPoolExecutor(max_workers=6) as ex:
            for pid, r in ex.map(judge_one, todo):
                jcache[str(pid)] = r
                done += 1
                if done % 10 == 0:
                    json.dump(jcache, open(jpath, 'w'), ensure_ascii=False)
                    print('judge checkpoint', done, '/', len(todo), flush=True)
        json.dump(jcache, open(jpath, 'w'), ensure_ascii=False, indent=1)
    elif todo:
        print('WARNING: no API key; skipping judge calls')

    for x in hits:
        v = jcache.get(str(x['pid']), {}).get('verdict')
        dec[x['pid']]['llm-recheck'] = (v == 'YES')

    # --- metrics ---
    def metrics(flag):
        sv = [p for p in dec if dec[p][flag]]
        dl = [p for p in dec if not dec[p][flag]]
        return {
            'served': len(sv),
            'decayed_served': sum(1 for p in sv
                                  if lab[p]['lab'] in ('wrong', 'partial')),
            'correct_declined': sum(1 for p in dl if lab[p]['lab'] == 'correct'),
            'indet_declined': sum(1 for p in dl if lab[p]['lab'] == 'indet'),
            'declined': len(dl),
        }

    out = {'n_hits': len(hits),
           'labels': {'wrong': 3, 'partial': 5, 'correct': 133, 'indet': 6},
           'baselines': {b: metrics(b) for b in
                         ['page-change', 'entity-change', 'prop-change',
                          'value-in-page', 'llm-recheck']},
           'judge_verdicts': {k: sum(1 for r in jcache.values()
                                     if r['verdict'] == k)
                              for k in ('YES', 'NO', 'NO_EVIDENCE',
                                        'PARSE_FAIL')},
           'per_entry': {str(p): dec[p] for p in dec}}
    json.dump(out, open(f'{DATA}/simple_baselines.json', 'w'), indent=1)
    print('\nbaseline                served  decayed-served  correct-declined  indet-declined')
    for b, m in out['baselines'].items():
        print(f"{b:22s} {m['served']:4d}  {m['decayed_served']:6d}        "
              f"{m['correct_declined']:4d}/133      {m['indet_declined']}")
    print('judge verdicts:', out['judge_verdicts'])


if __name__ == '__main__':
    main()

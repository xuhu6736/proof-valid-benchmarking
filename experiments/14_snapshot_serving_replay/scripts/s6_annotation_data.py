#!/usr/bin/env python3
"""Build annotation sheet dataset: one record per cached entry (154).

Merges panel/evidence, generation (answer+paraphrase), replay (gate outcomes),
and target-period claim values + sentence survival.
Output: data/annotation_data.json
"""
import json, os, re, hashlib
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.normpath(os.path.join(HERE, '..', 'data'))
REB = os.path.normpath(os.path.join(HERE, '..', '..', '12_real_dynamics_audit',
                                   'data', 'rebuilt_2026'))

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
    return [re.sub(r"\s+", " ", s).strip() for s in parts if len(s.strip()) >= min_len]

def sid(s):
    return hashlib.sha1(s.encode('utf-8')).hexdigest()[:12]

def render(v, labels):
    if isinstance(v, dict):
        if 'id' in v:
            q = str(v['id']); q = q[1:] if q.startswith('Q') else q
            return labels.get(q, 'Q' + q)
        if 'amount' in v:
            a = v['amount'].lstrip('+')
            try:
                f = float(a)
                return f'{f:,.0f}' if f == int(f) else f'{f:,.2f}'
            except Exception:
                return a
        if 'time' in v:
            m = re.match(r'\+?(\d+)-(\d{2})-(\d{2})', v['time'])
            return f'{m.group(1)}-{m.group(2)}-{m.group(3)}' if m else v['time']
        if 'text' in v:
            return str(v['text'])
    return str(v)

def vkey(v):
    if isinstance(v, dict):
        if 'id' in v: return 'Q:' + str(v['id'])
        if 'amount' in v: return 'A:' + str(v['amount'])
        if 'time' in v: return 'T:' + str(v['time'])[:11]
        if 'text' in v: return 'S:' + str(v['text'])
    return 'S:' + str(v)

def truthy(ent, prop):
    out = []
    for rank, snak, val in ent['claims'].get(prop, []):
        if rank in ('normal', 'preferred') and snak == 'value' and val is not None:
            out.append(val)
    return out

def main():
    panel = {x['pid']: x for x in json.load(open(f'{DATA}/evidence.json'))}
    gen = json.load(open(f'{DATA}/generation.json'))
    replay = {r['pid']: r for r in json.load(open(f'{DATA}/replay.json'))}
    lab0 = json.load(open(f'{DATA}/labels_20260401.json'))
    lab1 = json.load(open(f'{DATA}/labels_20260801.json'))
    e1 = json.load(open(f'{REB}/entities_v2_20260801.json'))
    pages1 = json.load(open(f'{REB}/pages_20260801.json'))

    recs = []
    for pid, r in sorted(replay.items()):
        x = panel[r['pid']]
        ent1 = e1.get(x['qid'])
        t1 = truthy(ent1, x['prop']) if ent1 else []
        base_keys = {vkey(v) for v in x['values_base']}
        tgt_keys = {vkey(v) for v in t1}
        removed = [render(v, lab0) for v in x['values_base'] if vkey(v) not in tgt_keys]
        added = [render(v, lab1) for v in t1 if vkey(v) not in base_keys]
        target_vals = [render(v, lab1) for v in t1]
        # sentence survival
        title = x.get('page_title')
        tgt_sids = set()
        if title and title in pages1:
            tgt_sids = {sid(s) for s in sentences(wikitext_to_plain(pages1[title]))}
        sents = [{'text': s['text'], 'alive': s['id'] in tgt_sids}
                 for s in x['certificate']['sentences']]
        recs.append({
            'pid': x['pid'], 'qid': x['qid'], 'entity': x['label'],
            'prop': x['prop'], 'prop_name': x['prop_name'],
            'kind': x['kind'],
            'question': x['question'],
            'paraphrase': gen[str(x['pid'])]['paraphrase'],
            'answer': gen[str(x['pid'])]['answer'],
            'page_title': title,
            'base_values': x['values_text'],
            'target_values': target_vals,
            'removed_values': removed, 'added_values': added,
            'sentences': sents,
            # --- answer key (appendix only) ---
            'stale': x['stale'], 'sim': r['sim_para'],
            'claims_dead': r['claims_dead'], 'sents_dead': r['sents_dead'],
            'gate_pass': r['gate_pass'],
        })
    json.dump(recs, open(f'{DATA}/annotation_data.json', 'w'),
              ensure_ascii=False, indent=1)
    print('records:', len(recs))
    print('stale:', sum(1 for r in recs if r['stale']))
    n_dead = sum(1 for r in recs for s in r['sentences'] if not s['alive'])
    print('dead sentences total:', n_dead)

if __name__ == '__main__':
    main()

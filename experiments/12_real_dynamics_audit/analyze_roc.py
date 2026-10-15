import json, re, html
from collections import Counter, defaultdict

def parse_entity(txt):
    """解析 Wikidata 实体 JSON → (label, truthy statement set)"""
    txt = html.unescape(txt)
    try:
        e = json.loads(txt)
    except Exception:
        return None, None
    label = None
    try:
        label = e.get('labels', {}).get('en', {}).get('value')
    except Exception:
        pass
    stmts = set()
    for prop, slist in (e.get('claims') or {}).items():
        pref = [s for s in slist if s.get('rank') == 'preferred']
        use = pref if pref else [s for s in slist if s.get('rank') == 'normal']
        for s in use:
            snak = s.get('mainsnak', {})
            dv = snak.get('datavalue')
            if not dv:
                continue
            v = dv.get('value')
            if isinstance(v, dict):
                if 'id' in v: key = v['id']
                elif 'text' in v: key = v['text']
                elif 'time' in v: key = v['time']
                elif 'amount' in v: key = v['amount']
                else: key = json.dumps(v, sort_keys=True)
            else:
                key = str(v)
            stmts.add((prop, key))
    return label, stmts

apr = json.load(open('/tmp/pvb_data/entities_20260401.json'))
aug = json.load(open('/tmp/pvb_data/entities_20260801.json'))
tq = json.load(open('/tmp/pvb_data/title_qid_map.json'))
qid2title = {v['qid']: k for k, v in tq.items()}

ent = {}
for qid_s, txt in apr.items():
    qid = int(qid_s)
    label, s0 = parse_entity(txt)
    if s0 is not None:
        ent.setdefault(qid, {})['s0'] = s0
        if label: ent[qid]['label'] = label
for qid_s, txt in aug.items():
    qid = int(qid_s)
    label, s1 = parse_entity(txt)
    if s1 is not None:
        ent.setdefault(qid, {})['s1'] = s1
        if label and not ent[qid].get('label'): ent[qid]['label'] = label

diff = {}
for qid, d in ent.items():
    if 's0' in d and 's1' in d:
        removed = d['s0'] - d['s1']
        added = d['s1'] - d['s0']
        diff[qid] = {"label": d.get('label'), "removed": sorted(removed), "added": sorted(added),
                     "n_s0": len(d['s0']), "n_s1": len(d['s1'])}
n = len(diff)
n_changed = sum(1 for v in diff.values() if v['removed'])
print(f"entities with statements both dates: {n}")
print(f"entities with >=1 removed (superseded) statement: {n_changed} ({100*n_changed/n:.1f}%)")
prop_removed = Counter()
for v in diff.values():
    for p, k in v['removed']:
        prop_removed[p] += 1
print("top removed properties:", prop_removed.most_common(15))
json.dump({str(k): v for k, v in diff.items()}, open('/tmp/pvb_data/statement_diff.json','w'))

# 一些可读示例
shown = 0
for qid, v in sorted(diff.items(), key=lambda kv: -len(kv[1]['removed'])):
    if v['removed'] and v['label'] and shown < 12:
        for p, k in v['removed'][:2]:
            print(f"  {v['label']} (Q{qid}) {p}: -{k}")
        shown += 1

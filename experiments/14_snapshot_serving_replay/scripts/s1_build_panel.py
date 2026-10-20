#!/usr/bin/env python3
"""Stage 1 / Step 1: build the serving-replay QA panel.

Window: 2026-04-01 (base) -> 2026-08-01 (target), rebuilt snapshots.

A QA pair = (entity qid, property).  Cited certificate statements are the
base-period *truthy* values of that property.  A pair is labelled
  stale  iff at least one cited (prop, value) is no longer truthy at target
         (premise-erasure semantics; value *additions* do not make it stale);
  fresh  otherwise (identical set or additions only).

Panel = ALL stale pairs for the templated properties + 2x matched fresh
controls (seed 42).  Output: data/panel.json
"""
import json, random, os
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.normpath(os.path.join(HERE, '..', 'data'))
REB = os.path.normpath(os.path.join(HERE, '..', '..', '12_real_dynamics_audit',
                                   'data', 'rebuilt_2026'))

# property -> (question template, answer kind)  kind: qid|time|quantity|string
TEMPLATES = {
    'P106': ('What is the occupation of {L}?', 'qid'),
    'P136': ('What genre is {L}?', 'qid'),
    'P131': ('In which administrative territorial entity is {L} located?', 'qid'),
    'P166': ('Which award did {L} receive?', 'qid'),
    'P19':  ('Where was {L} born?', 'qid'),
    'P69':  ('Where was {L} educated?', 'qid'),
    'P175': ('Who performed {L}?', 'qid'),
    'P361': ('What is {L} part of?', 'qid'),
    'P527': ('What are the parts of {L}?', 'qid'),
    'P39':  ('Which position does {L} hold?', 'qid'),
    'P54':  ('Which sports team does {L} play for?', 'qid'),
    'P102': ('Which political party does {L} belong to?', 'qid'),
    'P123': ('Who published {L}?', 'qid'),
    'P264': ('Which record label is {L} signed to?', 'qid'),
    'P27':  ('What is the country of citizenship of {L}?', 'qid'),
    'P17':  ('In which country is {L} located?', 'qid'),
    'P641': ('Which sport does {L} play?', 'qid'),
    'P495': ('What is the country of origin of {L}?', 'qid'),
    'P159': ('Where is the headquarters of {L} located?', 'qid'),
    'P20':  ('Where did {L} die?', 'qid'),
    'P57':  ('Who directed {L}?', 'qid'),
    'P58':  ('Who was the screenwriter of {L}?', 'qid'),
    'P86':  ('Who composed the music for {L}?', 'qid'),
    'P50':  ('Who is the author of {L}?', 'qid'),
    'P35':  ('Who is the head of state of {L}?', 'qid'),
    'P6':   ('Who is the head of government of {L}?', 'qid'),
    'P1082': ('What is the population of {L}?', 'quantity'),
    'P2046': ('What is the area of {L}?', 'quantity'),
    'P571':  ('When was {L} established?', 'time'),
    'P577':  ('When was {L} published or released?', 'time'),
    'P570':  ('When did {L} die?', 'time'),
}


def truthy(ent):
    out = defaultdict(list)
    for p, lst in ent['claims'].items():
        for rank, snak, val in lst:
            if rank in ('normal', 'preferred') and snak == 'value' and val is not None:
                out[p].append(val)
    return out


def val_key(p, v):
    """canonical comparable key for a claim value"""
    if isinstance(v, dict):
        if 'id' in v:   return 'Q:' + str(v['id'])
        if 'amount' in v: return 'A:' + str(v['amount'])
        if 'time' in v: return 'T:' + str(v['time'])[:11]  # day precision
        if 'text' in v: return 'S:' + str(v['text'])
    return 'S:' + str(v)


def val_qid(v):
    if isinstance(v, dict) and 'id' in v:
        q = str(v['id'])
        return q[1:] if q.startswith('Q') else q
    return None


def main():
    e0 = json.load(open(f'{REB}/entities_v2_20260401.json'))
    e1 = json.load(open(f'{REB}/entities_v2_20260801.json'))
    pairs = []
    for qid, ent in e0.items():
        if qid not in e1:
            continue
        lab = ent.get('label')
        if not lab:
            continue
        t0, t1 = truthy(ent), truthy(e1[qid])
        for p in TEMPLATES:
            if p not in t0:
                continue
            keys0 = {val_key(p, v) for v in t0[p]}
            keys1 = {val_key(p, v) for v in t1.get(p, [])}
            removed = keys0 - keys1
            stale = len(removed) > 0
            pairs.append({
                'qid': qid, 'label': lab, 'prop': p,
                'question': TEMPLATES[p][0].format(L=lab),
                'kind': TEMPLATES[p][1],
                'values_base': t0[p],          # raw values (certificate items)
                'stale': stale,
            })
    stale = [x for x in pairs if x['stale']]
    fresh = [x for x in pairs if not x['stale']]
    random.seed(42)
    random.shuffle(fresh)
    n_ctrl = min(len(fresh), 2 * len(stale))
    panel = stale + fresh[:n_ctrl]
    random.shuffle(panel)
    for i, x in enumerate(panel):
        x['pid'] = i
    os.makedirs(DATA, exist_ok=True)
    json.dump(panel, open(f'{DATA}/panel.json', 'w'), ensure_ascii=False, indent=1)

    qids = set()
    for x in panel:
        for v in x['values_base']:
            q = val_qid(v)
            if q:
                qids.add(q)
    json.dump(sorted(qids), open(f'{DATA}/label_qids.json', 'w'))
    print('pairs total:', len(pairs), ' stale:', len(stale),
          ' fresh kept:', n_ctrl, ' panel:', len(panel))
    print('distinct value qids needing labels:', len(qids))
    from collections import Counter
    print('panel by prop:', Counter(x['prop'] for x in panel).most_common())
    print('stale by prop:', Counter(x['prop'] for x in stale).most_common())


if __name__ == '__main__':
    main()

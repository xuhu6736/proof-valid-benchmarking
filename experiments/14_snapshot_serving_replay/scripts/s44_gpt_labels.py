#!/usr/bin/env python3
"""s44_gpt_labels.py -- parse the two GPT annotation sheets, audit
agreement, and build the GPT label store.

The 99-entry sheet (s43) was judged independently by the same two
annotators under the same R1 operational definitions as the earlier
rounds.  Adjudication follows the established protocol: the first
annotator adjudicates disagreements.

Label-store construction for every Table-B endpoint regeneration key:
  * GPT answer byte-identical (normalised) to the kimi answer -> inherit
    that key's adjudicated label (the text already judged by the humans);
  * differing answer -> this round's adjudicated label;
  * accept-fail / no-current-evidence -> no label (counted as abstention).

Outputs:
  annotation/gpt_ann1_parsed.json, annotation/gpt_ann2_parsed.json
  annotation/gpt_agreement.json
  data/gpt_labels.json     key "pid|tick" -> label record (incl. origin)
"""
import zipfile, re, json, os
from collections import Counter
from lxml import etree

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.normpath(os.path.join(HERE, '..', 'data'))
ANN = os.path.normpath(os.path.join(HERE, '..', 'annotation'))
PAPER = os.path.normpath(os.path.join(HERE, '..', '..', '..', '..'))

W = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"
OPT = ["正确", "错误", "不完整", "无法判断"]


def text_of(el):
    return "".join(t.text or "" for t in el.iter(W + "t"))


def parse_sheet(path):
    root = etree.fromstring(zipfile.ZipFile(path).read("word/document.xml"))
    out = {}
    for tbl in root.iter(W + "tbl"):
        txt = text_of(tbl)
        m = re.search(r"条目 Entry #(\d+) \| 证据期 (\d{4})-(\d{2})-(\d{2})",
                      txt)
        if not m:
            continue
        key = f"{m.group(1)}|{m.group(2)}{m.group(3)}{m.group(4)}"
        ann = None
        for row in tbl.iter(W + "tr"):
            rt = text_of(row)
            if "R1" in rt and "正确性" in rt:
                ann = rt
        assert ann is not None, f"no annotation row for {key}"
        i = ann.find("R1 ")
        j = ann.find("备注", i)
        seg = ann[i: j if j != -1 else len(ann)]
        vi = seg.find("√")
        assert vi != -1, f"missing mark {key}"
        tail = seg[vi + 1: vi + 40].lstrip()
        hit = next((o for o in OPT if tail.startswith(o)), None)
        assert hit, f"unmatched option {key}: {tail[:30]!r}"
        mn = re.search(r"备注 notes:\s*(.*)$", ann)
        out[key] = {"R1": hit,
                    "notes": re.sub(r"_+$", "", mn.group(1).strip()
                                    if mn else "").strip()}
    return out


def kappa(p, q, items):
    n = len(items)
    po = sum(p[i] == q[i] for i in items) / n
    cp, cq = Counter(p[i] for i in items), Counter(q[i] for i in items)
    pe = sum((cp[k] / n) * (cq[k] / n) for k in set(cp) | set(cq))
    return po, (po - pe) / (1 - pe) if pe < 1 else 1.0


def norm(s):
    return re.sub(r'[,\s]+', '', (s or '').lower())


def main():
    a1 = parse_sheet(os.path.join(
        PAPER, 'PVSB_GPT再生标注表_99条（人工1）.docx'))
    a2 = parse_sheet(os.path.join(
        PAPER, 'PVSB_GPT再生标注表_99条（人工2）.docx'))
    keys = json.load(open(os.path.join(ANN, 'gpt_sheet_keys.json')))
    assert set(a1) == set(a2) == set(keys), (
        f'key mismatch: sheet1 {len(a1)}, sheet2 {len(a2)}, '
        f'expected {len(keys)}')
    json.dump(a1, open(os.path.join(ANN, 'gpt_ann1_parsed.json'), 'w'),
              ensure_ascii=False, indent=1)
    json.dump(a2, open(os.path.join(ANN, 'gpt_ann2_parsed.json'), 'w'),
              ensure_ascii=False, indent=1)

    po, ka = kappa({k: a1[k]['R1'] for k in keys},
                   {k: a2[k]['R1'] for k in keys}, keys)
    disagree = {k: [a1[k]['R1'], a2[k]['R1'],
                    a1[k]['notes'], a2[k]['notes']]
                for k in keys if a1[k]['R1'] != a2[k]['R1']}
    adj = {k: (a1[k]['R1'] if a1[k]['R1'] == a2[k]['R1']
               else a1[k]['R1']) for k in keys}   # ann1 adjudicates
    agree_rep = {'n': len(keys), 'raw_agreement': po, 'cohens_kappa': ka,
                 'disagreements': disagree,
                 'dist_ann1': dict(Counter(a1[k]['R1'] for k in keys)),
                 'dist_ann2': dict(Counter(a2[k]['R1'] for k in keys))}
    json.dump(agree_rep, open(os.path.join(ANN, 'gpt_agreement.json'), 'w'),
              ensure_ascii=False, indent=1)
    print(f'99 fresh labels: raw agreement {po:.1%}, kappa {ka:.3f}, '
          f'disagreements {len(disagree)}')
    for k, d in disagree.items():
        print('  DIS', k, d[0], '/', d[1])

    # ------------------------- build the GPT label store ------------------
    ktab = json.load(open(f'{DATA}/tableb_regeneration.json'))
    gtab = json.load(open(f'{DATA}/tableb_regeneration_gpt.json'))
    k9 = json.load(open(f'{DATA}/regeneration.json'))
    g9 = json.load(open(f'{DATA}/regeneration_gpt.json'))
    tl = json.load(open(f'{DATA}/tableb_labels.json'))
    rl = json.load(open(f'{DATA}/regen_labels.json'))

    store, n_inherit, n_fresh = {}, 0, 0
    for k, g in gtab.items():
        if not g['regen_ok']:
            continue                       # abstain: no label
        kr = ktab.get(k, {})
        if kr.get('regen_ok') and norm(kr.get('regen_answer')) == \
                norm(g['regen_answer']):
            src = tl.get(k)
            if src is not None:
                store[k] = {'R1_adjudicated': src['R1_adjudicated'],
                            'origin': 'inherited-identical'}
                n_inherit += 1
            continue
        if k in adj:
            store[k] = {'R1_adjudicated': adj[k], 'origin': 'gpt-round'}
            n_fresh += 1
    for pid, g in g9.items():
        if not g['regen_ok']:
            continue
        kr = k9.get(pid, {})
        k = f'{pid}|20260801'
        if kr.get('regen_ok') and norm(kr.get('regen_answer')) == \
                norm(g['regen_answer']):
            src = rl.get(pid)
            if src is not None:
                store[k] = {'R1_adjudicated': src['R1_adjudicated'],
                            'origin': 'inherited-identical'}
                n_inherit += 1
            continue
        if k in adj:
            store[k] = {'R1_adjudicated': adj[k], 'origin': 'gpt-round'}
            n_fresh += 1
    json.dump(store, open(f'{DATA}/gpt_labels.json', 'w'),
              ensure_ascii=False, indent=1)
    print(f'GPT label store: {len(store)} labels '
          f'({n_inherit} inherited-identical, {n_fresh} fresh)')
    print('label distribution:',
          dict(Counter(v['R1_adjudicated'] for v in store.values())))
    print('saved data/gpt_labels.json')


if __name__ == '__main__':
    main()

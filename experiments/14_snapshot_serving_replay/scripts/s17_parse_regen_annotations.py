#!/usr/bin/env python3
"""Step 17: parse the two completed regeneration annotation sheets and
recompute closed-loop answer-level truth (P3 second half).

Reads annotation/PVSB_regeneration_sheet_ann1.docx and
annotation/PVSB_regeneration_sheet_ann2.docx (same √/□ marking
convention as the cached-answer sheet), computes per-item agreement
and Cohen's kappa for R1/R2, adjudicates disagreements (first
annotator adjudicates, as in the cached-answer round), and derives
the answer-level truth of the 51 regenerated answers.

Writes: annotation/regen_ann1_parsed.json,
        annotation/regen_ann2_parsed.json,
        annotation/regen_agreement.json,
        data/regen_labels.json
"""
import zipfile, re, json, os
from collections import Counter
from lxml import etree

HERE = os.path.dirname(os.path.abspath(__file__))
ANN = os.path.normpath(os.path.join(HERE, '..', 'annotation'))
DATA = os.path.normpath(os.path.join(HERE, '..', 'data'))

W = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"
OPT = {
    "R1": ["正确", "错误", "不完整", "无法判断"],
    "R2": ["支持", "部分支持", "不支持", "无法判断"],
}


def text_of(el):
    return "".join(t.text or "" for t in el.iter(W + "t"))


def parse_sheet(path):
    root = etree.fromstring(zipfile.ZipFile(path).read("word/document.xml"))
    out = {}
    for tbl in root.iter(W + "tbl"):
        txt = text_of(tbl)
        m = re.search(r"条目 Entry #(\d+)", txt)
        if not m:
            continue
        pid = int(m.group(1))
        ann = None
        for row in tbl.iter(W + "tr"):
            rt = text_of(row)
            if "R1" in rt and "正确性" in rt:
                ann = rt
        assert ann is not None, f"no annotation row for entry {pid}"
        rec = {}
        for q in ["R1", "R2"]:
            i = ann.find(q + " ")
            j = len(ann)
            for q2 in ["R1", "R2", "备注"]:
                if q2 == q:
                    continue
                k = ann.find(q2 + " " if q2 != "备注" else "备注", i + 3)
                if k != -1 and k < j:
                    j = k
            seg = ann[i:j]
            vi = seg.find("√")
            assert vi != -1, f"missing mark entry {pid} {q}"
            tail = seg[vi + 1: vi + 40].lstrip()
            hit = next((o for o in OPT[q] if tail.startswith(o)), None)
            assert hit, f"unmatched option entry {pid} {q}: {tail[:30]!r}"
            rec[q] = hit
        mn = re.search(r"备注 notes:\s*(.*)$", ann)
        rec["notes"] = re.sub(r"_+$", "", mn.group(1).strip() if mn else "").strip()
        out[pid] = rec
    return out


def kappa(p, q, items):
    n = len(items)
    po = sum(p[i] == q[i] for i in items) / n
    cp, cq = Counter(p[i] for i in items), Counter(q[i] for i in items)
    pe = sum((cp[k] / n) * (cq[k] / n) for k in set(cp) | set(cq))
    return po, (None if pe >= 1 else (po - pe) / (1 - pe))


def main():
    a1 = parse_sheet(os.path.join(ANN, 'PVSB_regeneration_sheet_ann1.docx'))
    a2 = parse_sheet(os.path.join(ANN, 'PVSB_regeneration_sheet_ann2.docx'))
    pids = sorted(set(a1) & set(a2))
    assert len(pids) == 51, f'expected 51, got {len(pids)}'

    json.dump({str(i): a1[i] for i in pids},
              open(os.path.join(ANN, 'regen_ann1_parsed.json'), 'w'),
              ensure_ascii=False, indent=1)
    json.dump({str(i): a2[i] for i in pids},
              open(os.path.join(ANN, 'regen_ann2_parsed.json'), 'w'),
              ensure_ascii=False, indent=1)

    agree = {}
    for q in ("R1", "R2"):
        po, k = kappa({i: a1[i][q] for i in pids},
                      {i: a2[i][q] for i in pids}, pids)
        agree[q] = {'agree': round(po, 4),
                    'kappa': (None if k is None else round(k, 3))}
    dis = [{'pid': i, 'item': q, 'ann1': a1[i][q], 'ann2': a2[i][q]}
           for i in pids for q in ("R1", "R2") if a1[i][q] != a2[i][q]]

    # adjudication: first annotator adjudicates (consistent with the
    # cached-answer round); record disagreements for the artifact
    adj = {i: a1[i]['R1'] for i in pids}
    labels = {str(i): {'R1_ann1': a1[i]['R1'], 'R1_ann2': a2[i]['R1'],
                       'R2_ann1': a1[i]['R2'], 'R2_ann2': a2[i]['R2'],
                       'R1_adjudicated': adj[i],
                       'notes_ann1': a1[i]['notes'],
                       'notes_ann2': a2[i]['notes']} for i in pids}
    json.dump(labels, open(f'{DATA}/regen_labels.json', 'w'),
              ensure_ascii=False, indent=1)

    c1 = Counter(a1[i]['R1'] for i in pids)
    c2 = Counter(a2[i]['R1'] for i in pids)
    out = {'n': len(pids),
           'agreement': agree,
           'n_disagreements': len(dis), 'disagreements': dis,
           'R1_ann1': dict(c1), 'R1_ann2': dict(c2),
           'R2_ann1': dict(Counter(a1[i]['R2'] for i in pids)),
           'R2_ann2': dict(Counter(a2[i]['R2'] for i in pids)),
           'adjudicated_R1': dict(Counter(adj.values()))}
    json.dump(out, open(os.path.join(ANN, 'regen_agreement.json'), 'w'),
              ensure_ascii=False, indent=1)

    print('R1 agreement: %.1f%% kappa %s' % (agree['R1']['agree'] * 100,
                                             agree['R1']['kappa']))
    print('R2 agreement: %.1f%% kappa %s' % (agree['R2']['agree'] * 100,
                                             agree['R2']['kappa']))
    print('disagreements:', dis)
    print('R1 ann1:', dict(c1))
    print('R1 ann2:', dict(c2))
    print('R1 adjudicated:', dict(Counter(adj.values())))
    print('R2 ann1:', out['R2_ann1'])
    print('R2 ann2:', out['R2_ann2'])
    n_same_notes = sum(1 for i in pids if a1[i]['notes'] == a2[i]['notes'])
    print('identical notes:', n_same_notes, '/', len(pids))


if __name__ == '__main__':
    main()

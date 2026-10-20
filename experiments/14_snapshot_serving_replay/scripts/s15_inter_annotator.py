#!/usr/bin/env python3
"""Step 15: inter-annotator agreement audit (P3, reviewer item).

Parses the two independently completed annotation sheets
(annotation/PVSB_annotation_sheet_ann1.docx = first annotator,
annotation/PVSB_annotation_sheet_ann2.docx = second annotator),
computes per-item agreement and Cohen's kappa for A1-A5, derives
answer-level decay labels under each annotator, and reports the
sensitivity of the headline counts (decayed set, interception) to
the choice of annotator.

Writes: annotation/annotations_ann1_parsed.json,
        annotation/annotations_ann2_parsed.json,
        annotation/agreement.json
"""
import json, os, sys
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
ANN = os.path.normpath(os.path.join(HERE, '..', 'annotation'))
sys.path.insert(0, HERE)
from parse_annotations import parse_sheet  # noqa: E402

Q = ['A1', 'A2', 'A3', 'A4', 'A5']


def decay(a):
    if a['A4'] == '错误':
        return 'wrong'
    if a['A4'] == '不完整' and a['A5'] in ('使其失效', '部分影响'):
        return 'partial'
    if a['A4'] == '无法判断':
        return 'indet'
    return 'correct'


def kappa(p, q, items):
    n = len(items)
    po = sum(p[i] == q[i] for i in items) / n
    cp, cq = Counter(p[i] for i in items), Counter(q[i] for i in items)
    pe = sum((cp[k] / n) * (cq[k] / n) for k in set(cp) | set(cq))
    return po, (None if pe >= 1 else (po - pe) / (1 - pe))


def main():
    a1 = parse_sheet(os.path.join(ANN, 'PVSB_annotation_sheet_ann1.docx'))
    a2 = parse_sheet(os.path.join(ANN, 'PVSB_annotation_sheet_ann2.docx'))
    pids = sorted(set(a1) & set(a2))
    assert len(pids) == 154

    json.dump({str(i): a1[i] for i in pids},
              open(os.path.join(ANN, 'annotations_ann1_parsed.json'), 'w'),
              ensure_ascii=False, indent=1)
    json.dump({str(i): a2[i] for i in pids},
              open(os.path.join(ANN, 'annotations_ann2_parsed.json'), 'w'),
              ensure_ascii=False, indent=1)

    agree = {}
    for q in Q:
        po, k = kappa({i: a1[i][q] for i in pids},
                      {i: a2[i][q] for i in pids}, pids)
        agree[q] = {'agree': round(po, 4),
                    'kappa': (None if k is None else round(k, 3))}
    dis = [{'pid': i, 'item': q, 'ann1': a1[i][q], 'ann2': a2[i][q]}
           for i in pids for q in Q if a1[i][q] != a2[i][q]]

    lab = json.load(open(os.path.join(ANN, 'answer_labels.json')))
    dec = {n: sorted(i for i in pids if decay(a[i]) in ('wrong', 'partial'))
           for n, a in (('ann1', a1), ('ann2', a2))}
    changed = [{'pid': i, 'ann1': decay(a1[i]), 'ann2': decay(a2[i])}
               for i in pids if decay(a1[i]) != decay(a2[i])]
    # interception under ann2's decayed set
    inter = {}
    for v in ('strict', 'majority', 'claims'):
        inter[v] = sum(1 for i in dec['ann2']
                       if lab[str(i)]['hit'] and not lab[str(i)][v])
    sim_served_decayed = sum(1 for i in dec['ann2'] if lab[str(i)]['hit'])

    out = {
        'n_entries': len(pids), 'n_cells': len(pids) * 5,
        'n_disagreements': len(dis),
        'item_level_agreement': round(1 - len(dis) / (len(pids) * 5), 4),
        'per_question': agree,
        'disagreements': dis,
        'decayed_ann1': dec['ann1'], 'decayed_ann2': dec['ann2'],
        'derived_label_changes': changed,
        'ann2_interception': inter,
        'ann2_decayed_hits_served_by_similarity': sim_served_decayed,
    }
    json.dump(out, open(os.path.join(ANN, 'agreement.json'), 'w'),
              ensure_ascii=False, indent=1)

    print('item-level agreement: %.1f%% (%d/%d cells)'
          % (out['item_level_agreement'] * 100,
             len(pids) * 5 - len(dis), len(pids) * 5))
    for q in Q:
        print(f"  {q}: agree {agree[q]['agree']*100:.1f}%  kappa {agree[q]['kappa']}")
    print('decayed ann1:', dec['ann1'])
    print('decayed ann2:', dec['ann2'])
    print('derived-label changes:', changed)
    print('ann2 interception (decayed hits declined):', inter,
          '| ann2 decayed hits:', sim_served_decayed)


if __name__ == '__main__':
    main()

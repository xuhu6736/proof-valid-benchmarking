#!/usr/bin/env python3
"""Step 16: build the regeneration annotation sheet (P3 second half).

Generates annotation/PVSB_regeneration_sheet.docx: a bilingual
per-entry sheet for the 51 regenerated answers that passed
value-mention acceptance in the closed loop (s9).  For each entry
the sheet lists the question, the paraphrase, the cached answer,
the REGENERATED answer, the target-period values, and the
target-period evidence sentences (markup cleaned), and provides two
annotation items to be judged independently by two annotators:

  R1 再生答案目标期正确性 correctness of the regenerated answer at
      the 2026-08-01 snapshot: 正确 / 错误 / 不完整 / 无法判断
  R2 目标期证据支持度 target-evidence support:
      支持 / 部分支持 / 不支持 / 无法判断

Gate decisions, dependency labels, and the cached-answer labels are
not shown.  The sheet uses the same √/□ marking convention as the
cached-answer sheet so parse_annotations-style tooling applies.
"""
import json, os, re, html
from docx import Document
from docx.shared import Pt
from docx.enum.text import WD_ALIGN_PARAGRAPH

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.normpath(os.path.join(HERE, '..', 'data'))
OUT = os.path.normpath(os.path.join(HERE, '..', 'annotation',
                                    'PVSB_regeneration_sheet.docx'))


def clean(wt):
    wt = html.unescape(wt)
    wt = re.sub(r"<ref[^>]*>.*?</ref>", " ", wt, flags=re.S)
    wt = re.sub(r"<ref[^>]*/>", " ", wt)
    wt = re.sub(r"\{\{[^{}]*\}\}", " ", wt)
    wt = re.sub(r"\{\{[^{}]*\}\}", " ", wt)
    wt = re.sub(r"\[\[([^|\]]*\|)?([^\]]*)\]\]", r"\2", wt)
    wt = re.sub(r"\[https?://[^\s\]]+\s*([^\]]*)\]", r"\1", wt)
    wt = re.sub(r"'{2,}", "", wt)
    wt = re.sub(r"<[^>]+>", " ", wt)
    return re.sub(r"\s+", " ", wt).strip()


def main():
    reg = json.load(open(f'{DATA}/regeneration.json'))
    ev = {x['pid']: x for x in json.load(open(f'{DATA}/evidence.json'))}
    gen = json.load(open(f'{DATA}/generation.json'))
    tr = json.load(open(f'{DATA}/translations.json'))
    ok = sorted(int(k) for k, v in reg.items() if v['regen_ok'])
    assert len(ok) == 51

    doc = Document()
    st = doc.styles['Normal']
    st.font.name = 'Times New Roman'
    st.font.size = Pt(10)

    doc.add_heading('PVSB 再生答案标注表 Regenerated-Answer Annotation Sheet', 0)
    doc.add_paragraph(
        '说明 Instructions：共 51 条，均为闭环中通过值提及验收的再生答案。'
        '请独立判断每个再生答案在 2026-08-01 目标快照下的质量，'
        '在对应选项前把 □ 改为 √。门控判定与缓存答案的标签未向您展示。'
        ' For each entry, mark exactly one option per item by changing '
        'the leading □ to √.  Gate decisions and cached-answer labels '
        'are withheld.')
    legend = doc.add_table(rows=3, cols=3)
    legend.style = 'Table Grid'
    legend.rows[0].cells[0].text = '标注项 Item'
    legend.rows[0].cells[1].text = '比较双方 Compare'
    legend.rows[0].cells[2].text = '判断内容 Judges'
    legend.rows[1].cells[0].text = 'R1'
    legend.rows[1].cells[1].text = '再生答案 ↔ 目标期值与证据句'
    legend.rows[1].cells[2].text = '目标期正确性 correctness at target snapshot'
    legend.rows[2].cells[0].text = 'R2'
    legend.rows[2].cells[1].text = '目标期证据句 ↔ 再生答案'
    legend.rows[2].cells[2].text = '证据支持度 evidence support'
    doc.add_paragraph('')

    for n, pid in enumerate(ok):
        x = ev[pid]
        r = reg[str(pid)]
        g = gen[str(pid)]
        t = doc.add_table(rows=8, cols=2)
        t.style = 'Table Grid'
        t.rows[0].cells[0].text = (f'条目 Entry #{pid}   {x["label"]} '
                                   f'(Q{x["qid"]})   {x["prop"]} '
                                   f'{x["prop_name"]}')
        t.rows[0].cells[1].text = ''
        zh = tr.get(str(pid), {}).get('q_zh', '')
        t.rows[1].cells[0].text = '问题 Question'
        t.rows[1].cells[1].text = x['question'] + ('\n' + zh if zh else '')
        t.rows[2].cells[0].text = '改写查询 Paraphrase'
        t.rows[2].cells[1].text = g['paraphrase']
        t.rows[3].cells[0].text = '缓存答案（已拒绝） Cached answer (declined)'
        t.rows[3].cells[1].text = g['answer']
        t.rows[4].cells[0].text = '再生答案 Regenerated answer'
        t.rows[4].cells[1].text = r['regen_answer']
        t.rows[5].cells[0].text = '目标期值 Target values'
        t.rows[5].cells[1].text = '; '.join(r['target_values_text']) or '—'
        t.rows[6].cells[0].text = '目标期证据句 Target evidence sentences'
        t.rows[6].cells[1].text = '\n'.join(
            f'[{i+1}] {clean(s)}' for i, s in enumerate(r['target_sentences'])) or '—'
        t.rows[7].cells[0].text = '标注 Annotation'
        t.rows[7].cells[1].text = (
            'R1 再生答案目标期正确性 correctness at target:  '
            '□ 正确 correct   □ 错误 incorrect   □ 不完整 incomplete   '
            '□ 无法判断 indeterminate\n'
            'R2 目标期证据支持度 evidence support:  '
            '□ 支持 supported   □ 部分支持 partial   □ 不支持 unsupported   '
            '□ 无法判断 indeterminate\n'
            '备注 notes: ')
        doc.add_paragraph('')

    doc.save(OUT)
    print('wrote', OUT, 'with', len(ok), 'entries')


if __name__ == '__main__':
    main()

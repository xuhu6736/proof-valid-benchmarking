#!/usr/bin/env python3
"""s53_make_109_sheet.py -- one-entry supplementary annotation sheet.

The claim-level equality ablation (s51/s52) and the equality-proxy mapping
(s48) deliver, at the 2026-08-01 endpoint, one archived answer that no core
policy delivers: the regeneration written back for request 109 at the
intermediate tick 2026-07-01 (key '109|20260701').  It is the only
unlabeled endpoint delivery in those configurations.  This sheet asks the
two annotators for the same R1 judgment as the 174-entry Table-B sheet:
correctness of the regenerated answer at its ORIGIN-TICK (2026-07-01)
snapshot.  One annotation clears both unassessed cells.

Outputs:
  annotation/PVSB_tableb_109_sheet.docx                    (working copy)
  <paper dir>/PVSB_TableB补充标注表_109_20260701.docx      (deliverable)
"""
import json, os, re, html, shutil
from docx import Document
from docx.shared import Pt

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.normpath(os.path.join(HERE, '..', 'data'))
OUT = os.path.normpath(os.path.join(HERE, '..', 'annotation',
                                    'PVSB_tableb_109_sheet.docx'))
PAPER_DIR = os.path.normpath(os.path.join(HERE, '..', '..', '..', '..'))

TICK_LABEL = {'20260501': '2026-05-01', '20260701': '2026-07-01',
              '20260801': '2026-08-01'}
KEY = '109|20260701'


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
    reg = json.load(open(f'{DATA}/tableb_regeneration.json'))
    ev = {x['pid']: x for x in json.load(open(f'{DATA}/evidence.json'))}
    gen = json.load(open(f'{DATA}/generation.json'))
    tr = json.load(open(f'{DATA}/translations.json'))
    r = reg[KEY]
    pid = int(KEY.split('|')[0])
    tick = KEY.split('|')[1]
    x = ev[pid]

    doc = Document()
    st = doc.styles['Normal']
    st.font.name = 'Times New Roman'
    st.font.size = Pt(10)

    doc.add_heading(
        'PVSB Table B 补充标注表 Supplementary Annotation Sheet (1 entry)',
        0)
    doc.add_paragraph(
        '背景 Background：本表仅 1 条。属性级相等消融回路（s51/s52）与相等映射'
        '（s48）在 2026-08-01 终点交付了一个核心策略均未在终点交付的归档答案：'
        '条目 109 在中间期 2026-07-01 再生的回写答案。它是这些配置中唯一未标注的'
        '终点交付。请按与 174 条 Table B 标注表完全相同的口径，独立判断该再生答案'
        '在其证据期（2026-07-01）快照下的正确性，在对应选项前把 □ 改为 √。'
        '门控判定与各臂机制标签未向您展示。'
        ' Please judge the regenerated answer\'s correctness at its '
        'origin-tick (2026-07-01) snapshot, using exactly the same R1 '
        'rubric as the 174-entry sheet; mark one option by changing the '
        'leading □ to √.  Gate decisions and per-arm mechanism labels '
        'are withheld.')

    leg = doc.add_table(rows=5, cols=2)
    leg.style = 'Table Grid'
    leg.rows[0].cells[0].text = 'R1 档位 Level'
    leg.rows[0].cells[1].text = '操作定义 Operational definition'
    for i, (lv, df) in enumerate([
            ('正确 correct', '答案与当期目标值一致且完整回答了问题'),
            ('错误 incorrect', '答案与当期目标值矛盾，或给出目标值之外的值'),
            ('不完整 incomplete', '方向正确但漏掉了部分当期目标值'),
            ('无法判断 indeterminate', '证据不足以按上述三档判定')]):
        leg.rows[i + 1].cells[0].text = lv
        leg.rows[i + 1].cells[1].text = df
    doc.add_paragraph('')

    t = doc.add_table(rows=10, cols=2)
    t.style = 'Table Grid'
    t.rows[0].cells[0].text = (
        f'条目 Entry #{pid} | 证据期 {TICK_LABEL[tick]}   '
        f'{x["label"]} (Q{x["qid"]})   {x["prop"]} {x["prop_name"]}')
    t.rows[0].cells[1].text = ''
    zh = tr.get(str(pid), {}).get('q_zh', '')
    t.rows[1].cells[0].text = '问题 Question'
    t.rows[1].cells[1].text = x['question'] + ('\n' + zh if zh else '')
    t.rows[2].cells[0].text = '改写查询 Paraphrase'
    t.rows[2].cells[1].text = gen[str(pid)]['paraphrase']
    t.rows[3].cells[0].text = '缓存答案（基期） Cached answer (base)'
    t.rows[3].cells[1].text = gen[str(pid)]['answer']
    t.rows[4].cells[0].text = '原模板交付答案 Previous template delivery'
    t.rows[4].cells[1].text = r['template_answer'] or '—'
    t.rows[5].cells[0].text = '再生答案 Regenerated answer (2026-07-01)'
    t.rows[5].cells[1].text = r['regen_answer']
    t.rows[6].cells[0].text = '当期目标值 Target values @ 2026-07-01'
    t.rows[6].cells[1].text = '; '.join(r['target_values_text']) or '—'
    t.rows[7].cells[0].text = '当期证据句 Evidence sentences @ 2026-07-01'
    t.rows[7].cells[1].text = '\n'.join(
        f'[{i+1}] {clean(s)}' for i, s in enumerate(r['target_sentences'])
    ) or '—'
    t.rows[8].cells[0].text = '终点交付说明 Endpoint delivery note'
    t.rows[8].cells[1].text = (
        '该再生答案被回写后在 2026-08-01 终点作为缓存答案交付；'
        '交付时其声明集与当期状态一致（门控通过）。'
        'This regeneration was written back and served from cache at the '
        '2026-08-01 endpoint; its claim set matched the endpoint state '
        '(gate passed).')
    t.rows[9].cells[0].text = '标注 Annotation'
    t.rows[9].cells[1].text = (
        'R1 再生答案当期正确性 correctness at origin tick (2026-07-01):  '
        '□ 正确 correct   □ 错误 incorrect   □ 不完整 incomplete   '
        '□ 无法判断 indeterminate\n'
        '备注 notes: ')

    doc.save(OUT)
    print('wrote', OUT)
    dst = os.path.join(PAPER_DIR, 'PVSB_TableB补充标注表_109_20260701.docx')
    shutil.copyfile(OUT, dst)
    print('deliverable copy:', dst)


if __name__ == '__main__':
    main()

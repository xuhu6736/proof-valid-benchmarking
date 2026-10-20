#!/usr/bin/env python3
"""s33_make_tableb_sheet.py -- Table B annotation sheet.

Builds a bilingual per-entry annotation sheet for the regenerations that
passed value-mention acceptance in s32 (the endpoint-fair regeneration set:
every answer the closed-loop arms previously delivered as a template,
regenerated with the real model against the origin-tick evidence).

For each entry the sheet lists the entry key (pid | origin tick), the
question (EN + ZH), the TEMPLATE answer that the system previously delivered
at that point, the REGENERATED answer, the origin-tick target values and
evidence sentences, and the arm(s) whose endpoint delivery this key feeds.
One annotation item, judged independently by two annotators:

  R1 再生答案当期正确性 correctness of the regenerated answer at the
      ORIGIN-TICK snapshot: 正确 / 错误 / 不完整 / 无法判断

R1 操作定义（与 154 条缓存答案表 A1-A5、51 条再生表口径衔接）:
  正确    -- 答案与当期目标值一致且完整回答了问题；
  错误    -- 答案与当期目标值矛盾，或给了目标值之外的值；
  不完整  -- 答案方向正确但漏掉了部分当期目标值；
  无法判断-- 证据不足以前述三档判定。

Gate decisions, mechanism labels, and other arms' labels are not shown.
Same √/□ marking convention as the earlier sheets.

Outputs:
  annotation/PVSB_tableb_sheet.docx          (working copy)
  <paper dir>/PVSB_TableB再生标注表_N条.docx  (deliverable copy)
"""
import json, os, re, html, shutil
from docx import Document
from docx.shared import Pt

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.normpath(os.path.join(HERE, '..', 'data'))
OUT = os.path.normpath(os.path.join(HERE, '..', 'annotation',
                                    'PVSB_tableb_sheet.docx'))
PAPER_DIR = os.path.normpath(os.path.join(HERE, '..', '..', '..', '..'))

TICK_LABEL = {'20260501': '2026-05-01', '20260701': '2026-07-01',
              '20260801': '2026-08-01'}


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
    ok = sorted((k for k, v in reg.items() if v['regen_ok']),
                key=lambda k: (int(k.split('|')[0]), k.split('|')[1]))
    print('entries with regen_ok:', len(ok))

    doc = Document()
    st = doc.styles['Normal']
    st.font.name = 'Times New Roman'
    st.font.size = Pt(10)

    doc.add_heading(
        'PVSB Table B 再生标注表 Endpoint-Fair Regeneration Sheet', 0)
    doc.add_paragraph(
        f'说明 Instructions：共 {len(ok)} 条。闭环实验先前以模板拼装方式交付的答案，'
        '现已统一改用真实模型按同一提示词、同一当期证据再生（键 = 条目号 | 证据期）。'
        '请独立判断每个再生答案在其证据期快照下的正确性，'
        '在对应选项前把 □ 改为 √。门控判定与各臂机制标签未向您展示。'
        ' For each entry, mark exactly one option by changing the leading '
        '□ to √.  Gate decisions and per-arm mechanism labels are withheld.')
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

    for n, k in enumerate(ok):
        r = reg[k]
        pid, tick = k.split('|')
        pid = int(pid)
        x = ev[pid]
        t = doc.add_table(rows=9, cols=2)
        t.style = 'Table Grid'
        t.rows[0].cells[0].text = (
            f'条目 Entry #{pid} | 证据期 {TICK_LABEL.get(tick, tick)}   '
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
        t.rows[5].cells[0].text = '再生答案 Regenerated answer'
        t.rows[5].cells[1].text = r['regen_answer']
        t.rows[6].cells[0].text = '当期目标值 Target values @ tick'
        t.rows[6].cells[1].text = '; '.join(r['target_values_text']) or '—'
        t.rows[7].cells[0].text = '当期证据句 Evidence sentences @ tick'
        t.rows[7].cells[1].text = '\n'.join(
            f'[{i+1}] {clean(s)}' for i, s in enumerate(r['target_sentences'])
        ) or '—'
        t.rows[8].cells[0].text = '标注 Annotation'
        t.rows[8].cells[1].text = (
            'R1 再生答案当期正确性 correctness at origin tick:  '
            '□ 正确 correct   □ 错误 incorrect   □ 不完整 incomplete   '
            '□ 无法判断 indeterminate\n'
            '备注 notes: ')
        doc.add_paragraph('')

    doc.save(OUT)
    print('wrote', OUT, 'with', len(ok), 'entries')
    dst = os.path.join(PAPER_DIR, f'PVSB_TableB再生标注表_{len(ok)}条.docx')
    shutil.copyfile(OUT, dst)
    print('deliverable copy:', dst)


if __name__ == '__main__':
    main()

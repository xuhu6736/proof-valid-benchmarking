// Generate the bilingual (EN-CN) per-entry annotation review sheet for the
// 154 cached entries of experiment 14 (snapshot serving replay).
// Data is read at runtime from annotation_data.json + translations.json.
import fs from "node:fs";
import path from "node:path";
import {
  AlignmentType, Document, Footer, Header, HeadingLevel, Packer, PageNumber,
  Paragraph, TextRun, Table, TableRow, TableCell, WidthType, ShadingType,
  PageBreak, convertInchesToTwip,
} from "docx";

const outputPath = process.argv[2];
if (!outputPath) throw new Error("Usage: node make_annotation_docx.js /abs/out.docx");

const HERE = path.dirname(decodeURIComponent(new URL(import.meta.url).pathname));
const DATA = path.join(HERE, "..", "data");
const recs = JSON.parse(fs.readFileSync(path.join(DATA, "annotation_data.json"), "utf8"));
const trs = JSON.parse(fs.readFileSync(path.join(DATA, "translations.json"), "utf8"));

const PROP_ZH = {
  P106: "职业", P136: "体裁/类型", P131: "所在行政实体", P166: "获得奖项",
  P19: "出生地", P69: "毕业院校", P175: "表演者", P361: "所属整体",
  P527: "组成部分", P39: "担任职务", P54: "所属运动队", P102: "所属政党",
  P123: "出版者", P264: "唱片公司", P27: "国籍", P17: "所在国家",
  P641: "运动项目", P495: "原产国", P159: "总部所在地", P20: "死亡地",
  P57: "导演", P58: "编剧", P86: "作曲", P50: "作者", P35: "国家元首",
  P6: "政府首脑", P1082: "人口", P2046: "面积", P571: "成立时间",
  P577: "出版/发布日期", P570: "逝世日期",
};

const font = { ascii: "Times New Roman", hAnsi: "Times New Roman", cs: "Times New Roman", eastAsia: "SimSun" };

// Clean raw wiki-extracted text: unescape HTML entities, drop citation
// markup (<ref>...</ref>), template braces ({{efn|...}}), residual tags.
function clean(t) {
  let s = String(t);
  s = s.replace(/&quot;/g, '"').replace(/&apos;/g, "'")
       .replace(/&lt;/g, "<").replace(/&gt;/g, ">").replace(/&amp;/g, "&");
  s = s.replace(/<ref\b[^>]*\/>/gi, "");          // self-closing refs
  s = s.replace(/<ref\b[^>]*>[\s\S]*?<\/ref>/gi, ""); // ref pairs
  let prev;
  do { prev = s; s = s.replace(/\{\{[^{}]*\}\}/g, ""); } while (s !== prev); // nested templates
  s = s.replace(/<\/?[a-zA-Z][^>]*>/g, "");        // any leftover tags
  return s.replace(/\s{2,}/g, " ").replace(/\s+([,.;:)])/g, "$1").replace(/([(])\s+/g, "$1").trim();
}
const run = (text, options = {}) => new TextRun({ text: String(text), font, size: 20, ...options });
const para = (children, options = {}) => new Paragraph({
  spacing: { after: 120, line: 280 },
  ...options,
  children: Array.isArray(children) ? children : [children],
});
const h1 = (text) => para(run(text, { bold: true, size: 30 }), { heading: HeadingLevel.HEADING_1, spacing: { before: 240, after: 160 } });
const h2 = (text) => para(run(text, { bold: true, size: 24 }), { heading: HeadingLevel.HEADING_2, spacing: { before: 200, after: 120 } });
const brk = () => new Paragraph({ children: [new PageBreak()] });

const W_LABEL = 1900, W_CONT = 7126, W_TOTAL = W_LABEL + W_CONT;
const cellMargins = { top: 80, bottom: 80, left: 120, right: 120 };
const labelCell = (text) => new TableCell({
  width: { size: W_LABEL, type: WidthType.DXA }, margins: cellMargins,
  shading: { type: ShadingType.CLEAR, fill: "F2F5F7" },
  children: [para(run(text, { bold: true, size: 18 }), { spacing: { after: 0 } })],
});
const contCell = (children) => new TableCell({
  width: { size: W_CONT, type: WidthType.DXA }, margins: cellMargins,
  children: Array.isArray(children) ? children : [children],
});
const row = (label, content) => new TableRow({ children: [labelCell(label), contCell(content)] });
const tpara = (children, options = {}) => para(children, { spacing: { after: 60, line: 260 }, ...options });
const trun = (text, options = {}) => run(text, { size: 18, ...options });

const CB = "□ ";

function entryTable(r) {
  const tr = trs[String(r.pid)] || {};
  const rows = [];
  // header row
  rows.push(new TableRow({ children: [new TableCell({
    width: { size: W_TOTAL, type: WidthType.DXA }, margins: cellMargins, columnSpan: 2,
    shading: { type: ShadingType.CLEAR, fill: "DCE5EC" },
    children: [tpara([
      trun(`条目 Entry #${r.pid}   `, { bold: true, size: 20 }),
      trun(`${r.entity} (Q${r.qid})   `, { bold: true, size: 20 }),
      trun(`${r.prop} ${r.prop_name} / ${PROP_ZH[r.prop] || ""}`, { size: 20 }),
    ], { spacing: { after: 0 } })],
  })] }));
  rows.push(row("问题 Question", [
    tpara(trun(r.question)),
    tpara(trun(tr.q_zh || "（缺翻译）", { color: "44546A" })),
  ]));
  rows.push(row("改写查询 Paraphrase", [tpara(trun(r.paraphrase || "—"))]));
  rows.push(row("缓存答案 Answer", [
    tpara(trun(r.answer)),
    tpara(trun(tr.a_zh || "（缺翻译）", { color: "44546A" })),
  ]));
  rows.push(row("指定声明 Claims（基期 base）",
    r.base_values.map((v) => tpara(trun(`– ${v}`)))));
  const sentParas = r.sentences.length
    ? r.sentences.map((s) => tpara([
        trun(s.alive ? "[存活 alive] " : "[目标期消失 ABSENT] ", s.alive ? { color: "2E7D32" } : { bold: true, color: "B23A2E" }),
        trun(clean(s.text)),
      ]))
    : [tpara(trun("（本条证书无证据句 no designated sentences）", { color: "667788" }))];
  rows.push(row("指定句子 Sentences（基期提取）", sentParas));
  const st = [];
  st.push(tpara([trun("基期值 base: ", { bold: true }), trun(r.base_values.join("; ") || "—")]));
  st.push(tpara([trun("目标期值 target: ", { bold: true }), trun(r.target_values.join("; ") || "—")]));
  if (r.removed_values.length)
    st.push(tpara([trun("目标期移除 removed: ", { bold: true, color: "B23A2E" }), trun(r.removed_values.join("; "), { color: "B23A2E" })]));
  if (r.added_values.length)
    st.push(tpara([trun("目标期新增 added: ", { bold: true, color: "2E7D32" }), trun(r.added_values.join("; "), { color: "2E7D32" })]));
  if (!r.removed_values.length && !r.added_values.length)
    st.push(tpara(trun("两期指定属性值集合一致（无增删）no change in the designated value set", { color: "667788" })));
  rows.push(row("两期状态 Status", st));
  const ann = [
    tpara([trun("A1 基期答案正确性 base-answer correctness:  ", { bold: true }),
           trun(CB + "正确 correct   " + CB + "错误 incorrect   " + CB + "不完整 incomplete   " + CB + "无法判断 indeterminate")]),
    tpara([trun("A2 证据支持度 evidence support:  ", { bold: true }),
           trun(CB + "支持 supported   " + CB + "部分支持 partial   " + CB + "不支持 unsupported   " + CB + "无法判断")]),
    tpara([trun("A3 释义保真 paraphrase fidelity:  ", { bold: true }),
           trun(CB + "保真 faithful   " + CB + "不保真 unfaithful   " + CB + "无法判断")]),
    tpara([trun("A4 目标期答案仍正确 answer still correct at 2026-08-01:  ", { bold: true }),
           trun(CB + "正确   " + CB + "错误   " + CB + "不完整   " + CB + "无法判断")]),
    tpara([trun("A5 两期变化是否影响答案断言 does the change affect the answer:  ", { bold: true }),
           trun(CB + "使其失效 invalidates   " + CB + "部分影响 partial   " + CB + "不影响 no effect   " + CB + "无变化可评 no change   " + CB + "无法判断")]),
    tpara([trun("备注 notes: ", { bold: true }), trun("____________________________________________")]),
  ];
  rows.push(row("标注 Annotation", ann));
  return new Table({
    width: { size: W_TOTAL, type: WidthType.DXA },
    columnWidths: [W_LABEL, W_CONT],
    rows,
  });
}

// ---------------- instructions page ----------------
const AW = [640, 2230, 820, 1000, 900, 1050, 1060, 1326];
const aCell = (text, w, options = {}) => new TableCell({
  width: { size: w, type: WidthType.DXA }, margins: cellMargins,
  ...(options.shade ? { shading: { type: ShadingType.CLEAR, fill: options.shade } } : {}),
  children: [tpara(run(String(text), { size: 16, ...(options.bold ? { bold: true } : {}) }), { spacing: { after: 0 } })],
});
const instr = [
  h1("PVSB 快照服务回放：逐条标注审阅表（中英对照）"),
  para(run("Per-Entry Annotation Sheet for the PVSB Snapshot Serving Replay (Bilingual)", { bold: true, size: 22, color: "44546A" })),
  para([run("用途 Purpose：", { bold: true }),
        run("本表用于对快照服务回放实验（2026-04-01 基期 → 2026-08-01 目标期）缓存的 154 条问答条目进行人工答案级标注。每条包含：问题（中英）、改写查询、缓存答案（中英）、指定证据（声明与句子）、两期状态（基期值/目标期值/增删）以及五个标注问题。")]),
  para([run("This sheet supports answer-level human annotation of the 154 cached question–answer entries in the snapshot serving replay (base 2026-04-01 → target 2026-08-01). Each block lists the question, its paraphrase, the cached answer, the designated evidence (claims and sentences), the two-period value status, and five annotation questions.", { color: "44546A" })]),
  h2("标注问题与选项 Annotation questions and options"),
  para([run("A1 基期答案正确性：", { bold: true }), run("答案是否正确回答了原问题（以基期 2026-04-01 为准）。Does the answer correctly answer the original question as of the base snapshot?")]),
  para([run("A2 证据支持度：", { bold: true }), run("答案中的事实是否被所列指定证据（声明+句子）支持。Are the answer's facts supported by the listed designated evidence?")]),
  para([run("A3 释义保真：", { bold: true }), run("改写查询是否保持原问题的实体、属性与量词。Does the paraphrase preserve the entity, property and quantifier of the original question?")]),
  para([run("A4 目标期答案仍正确：", { bold: true }), run("在目标期 2026-08-01，该缓存答案是否仍正确、完整。Is the cached answer still correct and complete at the target snapshot? 操作参照：将“缓存答案”的内容与“两期状态”中的目标期值比对（并参考指定句子的存活/消失标记）；目标期值是机械投影而非绝对真值，请作综合判断，拿不准选“无法判断”。Reference: compare the cached answer against the target-period values in the Status row (aided by sentence survival marks); the target values are a mechanical projection, not ground truth — use your judgment and “indeterminate” when unsure.")]),
  para([run("A5 变化影响：", { bold: true }), run("两期观察到的变化（值增删、句子消失）是否影响答案的实际断言。Do the observed changes affect the answer's actual assertion? 五档：使其失效＝核心断言被推翻；部分影响＝部分断言值失效但核心仍成立；不影响＝变化不触及答案断言（含仅新增值、规范化假象）；无变化可评＝两期值集一致且无句子消失；无法判断。Five options: invalidates = core assertion overturned; partial = some asserted values lost but the core still holds; no effect = changes do not touch the asserted facts (incl. pure additions, normalization artifacts); no change = identical value set and no sentence loss; indeterminate.")]),
  h2("五问操作定义一览 Operational definitions at a glance"),
  new Table({
    width: { size: W_TOTAL, type: WidthType.DXA },
    columnWidths: [900, 3600, 4526],
    rows: [
      new TableRow({ children: [
        aCell("标注项 Item", 900, { bold: true, shade: "DCE5EC" }),
        aCell("比较双方 Compare", 3600, { bold: true, shade: "DCE5EC" }),
        aCell("判断内容 Judges", 4526, { bold: true, shade: "DCE5EC" }),
      ] }),
      ...[
        ["A1", "缓存答案 ↔ 基期值", "事实正确性（基期）factual correctness at base"],
        ["A2", "缓存答案 ↔ 声明＋句子", "证据支持度，重点看句子 evidence support, esp. sentences"],
        ["A3", "改写查询 ↔ 原问题", "实体、属性、量词保持 entity/property/quantifier preserved"],
        ["A4", "缓存答案 ↔ 目标期值＋句子存活", "事实正确性（目标期）factual correctness at target"],
        ["A5", "两期增删/句消失 ↔ 答案断言", "变化的归因影响，五档 attribution of impact, 5 options"],
      ].map(([a, b, c]) => new TableRow({ children: [
        aCell(a, 900, { bold: true }), aCell(b, 3600), aCell(c, 4526),
      ] })),
    ],
  }),
  para(run("注意 Note：A2 与 A1 可能不一致——答案与基期值吻合但句子未覆盖全部断言时，A1 正确而 A2 部分支持；纯新增值不影响答案已断言的事实（A5 选“不影响”），完整性问题由 A4 的“不完整”承接。A2 may diverge from A1: if the answer matches the base values but the sentences do not cover every assertion, mark A1 correct and A2 partial. Pure additions do not invalidate asserted facts (A5 = no effect); incompleteness is captured by A4.", { size: 18, color: "44546A" })),
  h2("标注约定 Conventions"),
  para(run("1. 请在所选 □ 内打 √ 或高亮；无法判断时请选“无法判断”并在备注说明。Mark √ in the chosen box or highlight it; use “indeterminate” when unsure and explain in notes.")),
  para(run("2. “[目标期消失 ABSENT]”仅表示该规范化字符串未再出现在目标页，不代表事实一定变化。“ABSENT” only means the normalized string no longer appears on the target page; it does not by itself imply a factual change.")),
  para(run("3. 表格中不含任何门控决定或实验标签，以避免判断偏倚；门控结果与依赖标签集中在文末“答案对照表”，请完成全部标注后再查阅。Gate decisions and dependency labels are withheld from the entry blocks to avoid bias; they are listed in the answer-key appendix at the end — consult it only after completing all annotations.")),
  para(run("4. 属性缩写 Property codes：P106 职业 occupation；P136 体裁 genre；P131 所在行政实体 located in；P166 奖项 award；P19 出生地 birth place；P69 毕业院校 educated at；P175 表演者 performer；P361 所属整体 part of；P39 职务 position；P54 运动队 team；P102 政党 party；P27 国籍 citizenship；P17 国家 country；P641 运动 sport；P495 原产国 origin；P159 总部 HQ；P20 死亡地 death place；P57 导演 director；P58 编剧 screenwriter；P86 作曲 composer；P50 作者 author；P1082 人口 population；P2046 面积 area；P571 成立 inception；P577 发布日期 publication date；P570 逝世日期 death date；P264 唱片公司 record label；P123 出版者 publisher；P527 组成部分 has part；P35 国家元首 head of state；P6 政府首脑 head of government。", { size: 18 })),
];

// ---------------- answer key appendix ----------------
const aHead = new TableRow({ children: [
  aCell("#", AW[0], { bold: true, shade: "DCE5EC" }), aCell("实体 Entity", AW[1], { bold: true, shade: "DCE5EC" }),
  aCell("属性 Prop", AW[2], { bold: true, shade: "DCE5EC" }), aCell("依赖陈旧 stale", AW[3], { bold: true, shade: "DCE5EC" }),
  aCell("相似命中 hit", AW[4], { bold: true, shade: "DCE5EC" }), aCell("声明缺失 cl. dead", AW[5], { bold: true, shade: "DCE5EC" }),
  aCell("句子缺失 sn. dead", AW[6], { bold: true, shade: "DCE5EC" }), aCell("严格门 strict gate", AW[7], { bold: true, shade: "DCE5EC" }),
] });
const aRows = recs.map((r) => new TableRow({ children: [
  aCell(r.pid, AW[0]), aCell(r.entity, AW[1]), aCell(r.prop, AW[2]),
  aCell(r.stale ? "是 yes" : "否 no", AW[3], r.stale ? { bold: true } : {}),
  aCell(r.sim >= 0.8364 ? "是 yes" : "否 no", AW[4]),
  aCell(`${r.claims_dead}`, AW[5]), aCell(`${r.sents_dead}`, AW[6]),
  aCell(r.sim < 0.8364 ? "未命中 miss" : (r.gate_pass ? "服务 serve" : "拒绝 decline"), AW[7]),
] }));

// ---------------- assemble ----------------
const children = [];
children.push(...instr);
children.push(brk());
children.push(h1("逐条审阅表 Per-Entry Review Blocks（154 条）"));
for (const r of recs) {
  children.push(entryTable(r));
  children.push(para(run(""), { spacing: { after: 200 } }));
}
children.push(brk());
children.push(h1("答案对照表 Answer Key（标注完成后查阅 / consult after annotating）"));
children.push(para(run("依赖陈旧=至少一个指定基期值在目标期投影 truthy 视图中缺失；相似命中=释义相似度≥0.8364；严格门=相似命中且全部指定声明与句子存活。stale = a designated base value absent from the target projected truthy view; hit = paraphrase similarity ≥ 0.8364; strict gate = hit AND all designated claims and sentences survive.", { size: 18, color: "44546A" })));
children.push(new Table({
  width: { size: AW.reduce((a, b) => a + b, 0), type: WidthType.DXA },
  columnWidths: AW,
  rows: [aHead, ...aRows],
}));

const doc = new Document({
  features: { updateFields: false },
  sections: [{
    headers: { default: new Header({ children: [para(run("PVSB 快照服务回放 · 逐条标注审阅表", { bold: true, size: 18 }), { alignment: AlignmentType.CENTER })] }) },
    footers: { default: new Footer({ children: [para(new TextRun({ children: [PageNumber.CURRENT], font, size: 18 }), { alignment: AlignmentType.CENTER })] }) },
    children,
  }],
});

fs.writeFileSync(outputPath, await Packer.toBuffer(doc));
console.log("written:", outputPath);

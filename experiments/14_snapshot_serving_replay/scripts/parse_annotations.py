#!/usr/bin/env python3
"""Parse the completed bilingual annotation sheet (docx) into per-entry labels.

Reads annotation/PVSB_annotation_sheet.docx and writes
annotation/annotations_parsed.json (raw A1-A5 choices + notes) and
annotation/answer_labels.json (derived answer-level decay labels and
per-variant gate decisions joined with data/annotation_data.json).

Label derivation (see Section 6.6 of the paper):
  wrong    : A4 = 错误 (answer wrong at target snapshot)
  partial  : A4 = 不完整 AND A5 in {使其失效, 部分影响} (change-induced decay)
  indet    : A4 = 无法判断
  correct  : otherwise (incl. pre-existing/base-defect incompleteness)
"""
import zipfile, re, json, sys
from pathlib import Path
from lxml import etree

HERE = Path(__file__).resolve().parent.parent  # experiment root
DOCX = HERE / "annotation" / "PVSB_annotation_sheet.docx"
DATA = HERE / "data" / "annotation_data.json"
TAU = 0.8364  # fifth percentile of genuine paraphrase similarities

W = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"
OPT = {
    "A1": ["正确", "错误", "不完整", "无法判断"],
    "A2": ["支持", "部分支持", "不支持", "无法判断"],
    "A3": ["保真", "不保真", "无法判断"],
    "A4": ["正确", "错误", "不完整", "无法判断"],
    "A5": ["使其失效", "部分影响", "不影响", "无变化可评", "无法判断"],
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
            if "A1" in rt and "正确性" in rt:
                ann = rt
        assert ann is not None, f"no annotation row for entry {pid}"
        rec = {}
        for q in ["A1", "A2", "A3", "A4", "A5"]:
            i = ann.find(q + " ")
            j = len(ann)
            for q2 in ["A1", "A2", "A3", "A4", "A5", "备注"]:
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

def main():
    anns = parse_sheet(DOCX)
    assert len(anns) == 154, f"expected 154 entries, got {len(anns)}"
    (HERE / "annotation" / "annotations_parsed.json").write_text(
        json.dumps(anns, ensure_ascii=False, indent=1))

    data = {r["pid"]: r for r in json.load(open(DATA))}

    def decay(pid):
        a = anns[pid]
        if a["A4"] == "错误":
            return "wrong"
        if a["A4"] == "不完整" and a["A5"] in ("使其失效", "部分影响"):
            return "partial"
        if a["A4"] == "无法判断":
            return "indet"
        return "correct"

    labels = {}
    for pid, r in data.items():
        ns = len(r["sentences"])
        labels[str(pid)] = {
            "hit": r["sim"] >= TAU,
            "dep_stale": r["stale"],
            "strict": r["claims_dead"] == 0 and r["sents_dead"] == 0,
            "majority": r["claims_dead"] == 0 and (ns == 0 or (ns - r["sents_dead"]) * 2 > ns),
            "claims": r["claims_dead"] == 0,
            "lab": decay(pid),
            "A1": anns[pid]["A1"], "A4": anns[pid]["A4"], "A5": anns[pid]["A5"],
        }
    (HERE / "annotation" / "answer_labels.json").write_text(
        json.dumps(labels, ensure_ascii=False, indent=1))
    from collections import Counter
    print("A4:", Counter(a["A4"] for a in anns.values()))
    print("A5:", Counter(a["A5"] for a in anns.values()))
    print("decay labels:", Counter(v["lab"] for v in labels.values()))

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Stage 9 (A-2 final): parse the double-annotated 20-item fallback
spot-check sheets, audit inter-annotator agreement, adjudicate
disagreements conservatively (disagreement -> the system-less-favourable
label), and emit fallback_spotcheck_results.json.  Also archives the
annotated sheets under annotation/.

Inputs (workspace root):  PVSB_主场景回退标注表_20条（人工1/2）.docx
Master ledger:            fallback_spotcheck_sample20.json
Output:                   fallback_spotcheck_results.json
"""
import json
import os
import re
import shutil

from docx import Document

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
LEVELS = ["正确", "不完整", "错误", "无法判断"]
# severity order for conservative adjudication: higher index = less
# favourable to the system
SEVERITY = {"正确": 0, "不完整": 1, "无法判断": 2, "错误": 3}


def parse_sheet(path):
    doc = Document(path)
    labels = {}
    for t in doc.tables[1:]:
        hdr = t.rows[0].cells[0].text
        n = int(re.search(r"Entry #(\d+)", hdr).group(1))
        cell = t.rows[4].cells[1].text
        marks = [lv for lv in LEVELS
                 if re.search(r"[√✓✔]\s*" + lv, cell)]
        if len(marks) != 1:
            raise ValueError(f"entry {n}: expected 1 mark, got {marks}")
        labels[n] = marks[0]
    if len(labels) != 20:
        raise ValueError(f"{path}: expected 20 items, got {len(labels)}")
    return labels


def wilson(k, n, z=1.96):
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * (p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5 / den
    return [round(100 * (c - h), 1), round(100 * (c + h), 1)]


def main():
    a1 = parse_sheet(os.path.join(
        ROOT, "PVSB_主场景回退标注表_20条（人工1）.docx"))
    a2 = parse_sheet(os.path.join(
        ROOT, "PVSB_主场景回退标注表_20条（人工2）.docx"))
    items = json.load(open(os.path.join(
        HERE, "fallback_spotcheck_sample20.json")))["items"]

    agree = sum(1 for n in a1 if a1[n] == a2[n])
    # Cohen's kappa over the levels actually used
    cats = sorted(set(a1.values()) | set(a2.values()))
    n_items = len(a1)
    pe = sum((sum(1 for v in a1.values() if v == c) / n_items)
             * (sum(1 for v in a2.values() if v == c) / n_items)
             for c in cats)
    po = agree / n_items
    kappa = (po - pe) / (1 - pe) if pe < 1 else 1.0

    adjudicated = {}
    disagreements = []
    for n in a1:
        if a1[n] == a2[n]:
            adjudicated[n] = a1[n]
        else:
            adjudicated[n] = max((a1[n], a2[n]), key=lambda x: SEVERITY[x])
            disagreements.append(
                {"n": n, "annotator1": a1[n], "annotator2": a2[n],
                 "adjudicated": adjudicated[n],
                 "rule": "conservative (system-less-favourable)"})

    correct = sum(1 for v in adjudicated.values() if v == "正确")
    auto_accept = sum(1 for it in items if it["auto_correct"])
    false_accept = sum(1 for it in items
                       if it["auto_correct"]
                       and adjudicated[it["n"]] != "正确")
    auto_reject_confirmed = sum(
        1 for it in items if not it["auto_correct"]
        and adjudicated[it["n"]] != "正确")

    per_item = [{
        "n": it["n"], "gen": it["gen"], "key": it["key"],
        "tick": it["tick"], "auto_correct": it["auto_correct"],
        "annotator1": a1[it["n"]], "annotator2": a2[it["n"]],
        "adjudicated": adjudicated[it["n"]],
    } for it in items]

    results = {
        "sample": "20 of 41 executed fallback regenerations (seed 42)",
        "annotators": "two independent annotators, blinded to "
                      "auto-judgments and gate decisions",
        "agreement": {"raw": f"{agree}/20",
                      "raw_pct": round(100 * po, 1),
                      "cohens_kappa": round(kappa, 2)},
        "disagreements": disagreements,
        "adjudication_rule": "conservative: disagreement resolves to "
                             "the system-less-favourable label",
        "adjudicated_correct": correct,
        "adjudicated_correct_pct": round(100 * correct / 20, 1),
        "adjudicated_correct_wilson95": wilson(correct, 20),
        "auto_value_mention": {
            "accepted_in_sample": auto_accept,
            "false_accepts": false_accept,
            "false_accept_rate_pct": round(100 * false_accept
                                           / auto_accept, 1),
            "rejected_in_sample": 20 - auto_accept,
            "reject_confirmed_incorrect": auto_reject_confirmed},
        "per_item": per_item,
    }
    out = os.path.join(HERE, "fallback_spotcheck_results.json")
    json.dump(results, open(out, "w"), ensure_ascii=False, indent=1)
    print("saved", out)

    ann = os.path.join(HERE, "annotation")
    os.makedirs(ann, exist_ok=True)
    for tag in ("人工1", "人工2"):
        src = os.path.join(
            ROOT, f"PVSB_主场景回退标注表_20条（{tag}）.docx")
        shutil.copy2(src, os.path.join(ann, os.path.basename(src)))
    print("archived annotated sheets ->", ann)

    print(f"agreement {agree}/20 ({100*po:.1f}%), kappa={kappa:.2f}")
    print(f"adjudicated correct {correct}/20 "
          f"({100*correct/20:.1f}% {wilson(correct,20)})")
    print(f"auto false-accept {false_accept}/{auto_accept}")


if __name__ == "__main__":
    main()

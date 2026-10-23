#!/usr/bin/env python3
"""Stage 12 (operating-window closure, final): parse the double-annotated
40-item base-answer sheets, audit agreement, adjudicate conservatively,
and produce a design-weighted point estimate of b (the error rate of
non-stale cached answers) with a stratified-sampling variance (FPC).
Then evaluate the Eq.(window) verdict against the fallback spot-check
estimate of r (9/20 correct -> r = 11/20, Wilson [34.2, 74.2]).

Inputs (workspace root): PVSB_主场景基缓存答案标注表_40条（人工1/2）.docx
Ledger:                  base_answer_sample40.json
Output:                  base_answer_results.json
"""
import json
import os
import re
import shutil

from docx import Document

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
LEVELS = ["正确", "不完整", "错误", "无法判断"]
SEVERITY = {"正确": 0, "不完整": 1, "无法判断": 2, "错误": 3}
# stratum population sizes (from perpart_rescore.json over 130 accepted)
STRATA = {("kimi", "flagged"): 16, ("kimi", "clean"): 46,
          ("gpt", "flagged"): 17, ("gpt", "clean"): 51}
N_POP = 130
# fallback spot-check (stage9) adjudication
R_CORRECT, R_N = 9, 20


def parse_sheet(path):
    doc = Document(path)
    labels = {}
    for t in doc.tables[1:]:
        hdr = t.rows[0].cells[0].text
        n = int(re.search(r"Entry #(\d+)", hdr).group(1))
        cell = t.rows[4].cells[1].text
        marks = [lv for lv in LEVELS if re.search(r"[√✓✔]\s*" + lv, cell)]
        if len(marks) != 1:
            raise ValueError(f"entry {n}: expected 1 mark, got {marks}")
        labels[n] = marks[0]
    if len(labels) != 40:
        raise ValueError(f"{path}: expected 40 items, got {len(labels)}")
    return labels


def wilson(k, n, z=1.96):
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * (p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5 / den
    return [round(100 * (c - h), 1), round(100 * (c + h), 1)]


def main():
    a1 = parse_sheet(os.path.join(
        ROOT, "PVSB_主场景基缓存答案标注表_40条（人工1）.docx"))
    a2 = parse_sheet(os.path.join(
        ROOT, "PVSB_主场景基缓存答案标注表_40条（人工2）.docx"))
    ledger = json.load(open(os.path.join(HERE,
                                         "base_answer_sample40.json")))
    items = ledger["items"]

    agree = sum(1 for n in a1 if a1[n] == a2[n])
    cats = sorted(set(a1.values()) | set(a2.values()))
    pe = sum((sum(1 for v in a1.values() if v == c) / 40)
             * (sum(1 for v in a2.values() if v == c) / 40)
             for c in cats)
    kappa = (agree / 40 - pe) / (1 - pe) if pe < 1 else 1.0

    adjudicated, disagreements = {}, []
    for n in a1:
        if a1[n] == a2[n]:
            adjudicated[n] = a1[n]
        else:
            adjudicated[n] = max((a1[n], a2[n]),
                                 key=lambda x: SEVERITY[x])
            disagreements.append({"n": n, "ann1": a1[n], "ann2": a2[n],
                                  "adjudicated": adjudicated[n]})

    # ---- design-weighted (Horvitz-Thompson) estimates ----
    # error definition: primary = not 正确 (错误+不完整+无法判断),
    # sensitivity = 错误 only.
    def weighted(error_set):
        num = den = 0.0
        var = 0.0
        strata = {}
        for it in items:
            h = (it["gen"], it["stratum"])
            Nh = STRATA[h]
            err = 1.0 if adjudicated[it["n"]] in error_set else 0.0
            num += Nh / 10 * err      # each stratum sampled 10
            den += Nh / 10
            strata.setdefault(h, []).append(err)
        for h, errs in strata.items():
            Nh = STRATA[h]
            nh = len(errs)
            ph = sum(errs) / nh
            var += Nh * Nh * (1 - nh / Nh) * ph * (1 - ph) / max(nh - 1, 1)
        p = num / den
        se = (var ** 0.5) / N_POP
        return {"estimate_pct": round(100 * p, 1),
                "se_pct": round(100 * se, 1),
                "ci95_pct": [round(100 * max(0.0, p - 1.96 * se), 1),
                             round(100 * min(1.0, p + 1.96 * se), 1)]}

    b_primary = weighted({"错误", "不完整", "无法判断"})
    b_wrong_only = weighted({"错误"})

    # ---- window verdict ----
    r = (R_N - R_CORRECT) / R_N            # 0.55
    r_ci = [x / 100 for x in wilson(R_N - R_CORRECT, R_N)]
    b = b_primary["estimate_pct"] / 100
    upper_kimi, upper_gpt = 0.225 * (1 - b), 0.194 * (1 - b)
    lower_ttl1d = 0.0011 * (1 - b)
    verdict = {
        "r_estimate_pct": round(100 * r, 1),
        "r_ci95_pct": wilson(R_N - R_CORRECT, R_N),
        "r_minus_b_pct": round(100 * (r - b), 1),
        "upper_bound_kimi_pct": round(100 * upper_kimi, 1),
        "upper_bound_gpt_pct": round(100 * upper_gpt, 1),
        "lower_bound_ttl1d_pct": round(100 * lower_ttl1d, 2),
        "gate_beats_similarity_only": (r - b) < upper_kimi,
        "gate_beats_ttl1d": (r - b) > lower_ttl1d,
    }

    results = {
        "sample": "40 of 130 accepted base answers, stratified "
                  "(10 per-part-flagged + 10 clean per generator), "
                  "seed 42",
        "annotators": "two independent annotators, blinded to "
                      "per-part verdicts and gate labels",
        "agreement": {"raw": f"{agree}/40",
                      "raw_pct": round(100 * agree / 40, 1),
                      "cohens_kappa": round(kappa, 2)},
        "disagreements": disagreements,
        "adjudication_rule": "conservative (system-less-favourable)",
        "b_estimate_primary": b_primary,
        "b_definition_primary": "P(label != 正确), design-weighted",
        "b_wrong_only": b_wrong_only,
        "label_distribution": {
            lv: sum(1 for v in adjudicated.values() if v == lv)
            for lv in LEVELS},
        "window_verdict": verdict,
        "per_item": [{"n": it["n"], "gen": it["gen"], "key": it["key"],
                      "stratum": it["stratum"],
                      "ann1": a1[it["n"]], "ann2": a2[it["n"]],
                      "adjudicated": adjudicated[it["n"]]}
                     for it in items],
    }
    out = os.path.join(HERE, "base_answer_results.json")
    json.dump(results, open(out, "w"), ensure_ascii=False, indent=1)
    print("saved", out)

    ann = os.path.join(HERE, "annotation")
    os.makedirs(ann, exist_ok=True)
    for tag in ("人工1", "人工2"):
        src = os.path.join(
            ROOT, f"PVSB_主场景基缓存答案标注表_40条（{tag}）.docx")
        shutil.copy2(src, os.path.join(ann, os.path.basename(src)))
    print("archived ->", ann)
    print(json.dumps({"agreement": results["agreement"],
                      "b_primary": b_primary,
                      "b_wrong_only": b_wrong_only,
                      "labels": results["label_distribution"],
                      "verdict": verdict}, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()

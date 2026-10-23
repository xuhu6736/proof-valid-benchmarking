#!/usr/bin/env python3
"""Stage 10 (operating-window closure, part 1): per-part re-scoring of
archived answers -- no model calls.

Re-scores (a) the 41 executed fallback regenerations against tick-k
truth and (b) the 62/68 accepted base answers against t0 truth, using
PER-PART value matching (each claim's value must occur inside its own
numbered part), and contrasts with the original set-level value-mention
acceptance.  For the 20 double-annotated spot-check regenerations the
per-part verdicts are compared with the human-adjudicated labels.

Outputs: perpart_rescore.json
"""
import json
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))


def norm(s):
    return re.sub(r"[,\s]+", "", s.lower())


def value_in(v, text):
    nt = norm(text)
    dv = re.findall(r"\d+(?:\.\d+)?", v.replace(",", ""))
    if dv:
        return all(t in nt for t in dv)
    return norm(v) in nt


def split_parts(text):
    """Split '(1) .. (2) .. (3) ..' into {1: .., 2: .., 3: ..}."""
    toks = re.split(r"\((\d)\)\s*", text)
    parts = {}
    for i in range(1, len(toks) - 1, 2):
        parts[int(toks[i])] = toks[i + 1].strip()
    return parts


def perpart_verdict(answer, values):
    parts = split_parts(answer)
    missing = []
    unparsed = len(parts) < len(values)
    for j, v in enumerate(values, 1):
        if v is None:
            continue
        seg = parts.get(j, "") if not unparsed else answer
        if not value_in(v, seg):
            missing.append({"part": j, "value": v})
    return {"accept": not missing, "missing": missing,
            "unparsed": unparsed}


def main():
    panel = json.load(open(os.path.join(HERE, "panel21.json")))["panel"]
    out = {"regenerations": {}, "base_answers": {},
           "spotcheck_vs_human": {}}

    # ---------- (a) 41 executed fallback regenerations ----------
    fb = json.load(open(os.path.join(HERE, "fallback_regen.json")))
    sc = json.load(open(os.path.join(
        HERE, "fallback_spotcheck_results.json")))
    human = {it["n"]: it for it in sc["per_item"]}
    for gen in ("kimi", "gpt"):
        for key, v in fb[gen].items():
            if "correct" not in v:
                continue
            pp = perpart_verdict(v["regen"], v["current_values"])
            out["regenerations"][f"{gen}|{key}"] = {
                "set_level_accept": v["correct"],
                "per_part_accept": pp["accept"],
                "per_part_missing": pp["missing"],
                "unparsed": pp["unparsed"]}

    # ---------- (b) 62/68 accepted base answers vs t0 truth ----------
    for gen in ("kimi", "gpt"):
        ans = json.load(open(os.path.join(
            HERE, f"answers_{gen}.json")))["answers"]
        for key, a in ans.items():
            if not a.get("accepted"):
                continue
            idx = int(key.rsplit("#", 1)[1])
            t0 = [c["value_t0"] for c in panel[idx]["claims"]]
            pp = perpart_verdict(a["answer"], t0)
            out["base_answers"][f"{gen}|{key}"] = {
                "per_part_accept": pp["accept"],
                "per_part_missing": pp["missing"],
                "unparsed": pp["unparsed"]}

    # ---------- (c) per-part vs human labels on the 20-item check ---
    ledger = json.load(open(os.path.join(
        HERE, "fallback_spotcheck_sample20.json")))["items"]
    agree = false_accept = false_reject = 0
    detail = []
    for it in ledger:
        rec = out["regenerations"][f"{it['gen']}|{it['key']}"]
        hum = human[it["n"]]["adjudicated"]
        human_ok = hum == "正确"
        pp_ok = rec["per_part_accept"]
        if pp_ok == human_ok:
            agree += 1
        elif pp_ok and not human_ok:
            false_accept += 1
        else:
            false_reject += 1
        detail.append({"n": it["n"], "gen": it["gen"], "key": it["key"],
                       "per_part_accept": pp_ok,
                       "human": hum,
                       "set_level_accept": it["auto_correct"]})
    out["spotcheck_vs_human"] = {
        "n": len(ledger), "agree": agree,
        "false_accept": false_accept, "false_reject": false_reject,
        "set_level_agree": sum(
            1 for it in ledger
            if it["auto_correct"] == (human[it["n"]]["adjudicated"]
                                      == "正确")),
        "detail": detail}

    # ---------- summary ----------
    regen = out["regenerations"]
    base = out["base_answers"]
    out["summary"] = {
        "regen_set_level_accept":
            f"{sum(r['set_level_accept'] for r in regen.values())}/41",
        "regen_per_part_accept":
            f"{sum(r['per_part_accept'] for r in regen.values())}/41",
        "base_per_part_accept":
            {g: f"{sum(r['per_part_accept'] for k, r in base.items() if k.startswith(g))}"
             f"/{sum(1 for k in base if k.startswith(g))}"
             for g in ("kimi", "gpt")},
        "spotcheck_per_part_vs_human":
            f"{agree}/{len(ledger)} agree, "
            f"{false_accept} false-accept, {false_reject} false-reject",
        "spotcheck_set_level_vs_human":
            f"{out['spotcheck_vs_human']['set_level_agree']}/{len(ledger)}"
            " agree",
    }
    path = os.path.join(HERE, "perpart_rescore.json")
    json.dump(out, open(path, "w"), ensure_ascii=False, indent=1)
    print("saved", path)
    print(json.dumps(out["summary"], ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()

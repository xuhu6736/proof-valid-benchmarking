"""第四轮：FR<=11% 约束下的前沿细化 + McNemar 显著性 + 工作量估计重算。"""
import json, math, re

WORK = "/tmp/pvb_data"
OUTP = "/mnt/agents/output/pvb_pilot"
SCHEMA_PROPS = {"P421"}
exec(open("/mnt/agents/output/pvb_pilot/constructor_lab.py").read().split("def main()")[0])

pages0 = json.load(open(f"{WORK}/pages_20260401.json"))
pages1 = json.load(open(f"{WORK}/pages_20260801.json"))
tq = json.load(open(f"{WORK}/title_qid_map.json"))
ents0 = json.load(open(f"{WORK}/entities_v2_20260401.json"))
diff = {int(k): v for k, v in json.load(open(f"{OUTP}/statement_diff_v2_2026.json")).items()}

rows = []
for title, qid in tq.items():
    if qid not in diff or title not in pages0 or title not in pages1:
        continue
    s0 = sentences(wikitext_to_plain(pages0[title]))
    if len(s0) < 5:
        continue
    s1 = sentences(wikitext_to_plain(pages1[title]))
    harm = any(p not in SCHEMA_PROPS for p, _ in diff[qid]["removed"])
    pats = []
    if str(qid) in ents0:
        for prop, sl in ents0[str(qid)]["claims"].items():
            for rank, snak, v in sl:
                pats.extend(value_patterns(prop, v))
    rows.append({"title": title, "qid": qid, "harm": harm, "s0": s0, "s1": s1,
                 "pats": sorted(set(pats), key=len, reverse=True)})

for r in rows:
    r["s1_toks"] = [toks(s) for s in r["s1"]]
    s1_set = set(r["s1"])
    r["lost_any"] = sum(1 for s in r["s0"] if s not in s1_set)

def sel_val(r):
    return [s for s in r["s0"] if any(p in s for p in r["pats"])]

def brk(sel, r, fuzzy=1.0, quorum=1):
    return cert_break(sel, r["s1"], r["s1_toks"], fuzzy=fuzzy, quorum=quorum)

def OR2(a, b):
    if a is None and b is None:
        return None
    return bool(a) or bool(b)

def cascade(r, fz=0.9, fb=3):
    sv = sel_val(r)
    return brk(sv, r, fuzzy=fz) if sv else brk(select_lead(r["s0"], fb), r)

specs = {}
for m in (8, 9, 10, 12):
    specs[f"casc_or_lost{m}"] = lambda r, m=m: OR2(cascade(r), r["lost_any"] >= m)
specs["casc_or_dense1"] = lambda r: OR2(cascade(r), brk(select_dense(r["s0"], 1), r, fuzzy=0.9))
specs["casc_or_dense1_lost10"] = lambda r: OR2(OR2(cascade(r), brk(select_dense(r["s0"], 1), r, fuzzy=0.9)),
                                               r["lost_any"] >= 10)
specs["casc_or_dense1_lost12"] = lambda r: OR2(OR2(cascade(r), brk(select_dense(r["s0"], 1), r, fuzzy=0.9)),
                                               r["lost_any"] >= 12)
# 参考：lead-k 曲线 + 纯 cascade
for k in (1, 3, 5, 10):
    specs[f"lead{k}"] = lambda r, k=k: brk(select_lead(r["s0"], k), r)
specs["cascade_only"] = cascade

res = {"constructors": {}}
for name, fn in specs.items():
    for r in rows:
        r[name] = fn(r)
    res["constructors"][name] = evaluate(rows, name)
    e = res["constructors"][name]
    print(f"{name}: int {e['int']:.3f} [{e['int_ci'][0]:.3f},{e['int_ci'][1]:.3f}] | "
          f"fr {e['fr']:.3f} [{e['fr_ci'][0]:.3f},{e['fr_ci'][1]:.3f}]  (i={e['i']}/{e['n_ch']}, f={e['f']}/{e['n_un']})", flush=True)

# McNemar（连续校正卡方 + 正态 p）
def mcnemar_p(b, c):
    if b + c == 0:
        return 1.0
    chi2 = (abs(b - c) - 1) ** 2 / (b + c)
    # 卡方1的自由度上尾 ≈ 正态双尾
    z = math.sqrt(chi2)
    p = 2 * (1 - (0.5 * (1 + math.erf(z / math.sqrt(2)))))
    return p

res["mcnemar_vs_lead5"] = {}
for name in specs:
    b_i = sum(1 for r in rows if r["harm"] and r[name] and not r["lead5"])
    c_i = sum(1 for r in rows if r["harm"] and not r[name] and r["lead5"])
    b_f = sum(1 for r in rows if not r["harm"] and r[name] and not r["lead5"])
    c_f = sum(1 for r in rows if not r["harm"] and not r[name] and r["lead5"])
    res["mcnemar_vs_lead5"][name] = {"int_b": b_i, "int_c": c_i, "int_p": round(mcnemar_p(b_i, c_i), 5),
                                     "fr_b": b_f, "fr_c": c_f, "fr_p": round(mcnemar_p(b_f, c_f), 5)}
print(json.dumps(res["mcnemar_vs_lead5"], indent=1), flush=True)

json.dump(res, open(f"{OUTP}/constructor_results_v4.json", "w"), indent=1)
print("saved v4", flush=True)

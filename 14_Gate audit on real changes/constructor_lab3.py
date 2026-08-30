"""构造器实验室第三轮（规范句段口径）：级联回退 + 精细组合，FR≈10% 下最大化拦截。
另重算 P421 占比等论文修正所需数字。
"""
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
    removed = diff[qid]["removed"]
    harm = any(p not in SCHEMA_PROPS for p, _ in removed)
    pats = []
    if str(qid) in ents0:
        for prop, sl in ents0[str(qid)]["claims"].items():
            for rank, snak, v in sl:
                pats.extend(value_patterns(prop, v))
    pats = sorted(set(pats), key=len, reverse=True)
    rows.append({"title": title, "qid": qid, "harm": harm, "s0": s0, "s1": s1, "pats": pats})

for r in rows:
    r["s1_toks"] = [toks(s) for s in r["s1"]]
    s1_set = set(r["s1"])
    r["lost_any"] = sum(1 for s in r["s0"] if s not in s1_set)


def sel_val(r, k=None):
    sel = [s for s in r["s0"] if any(p in s for p in r["pats"])]
    return sel if k is None else sel[:k]


def brk(sel, r, fuzzy=1.0, quorum=1):
    return cert_break(sel, r["s1"], r["s1_toks"], fuzzy=fuzzy, quorum=quorum)


def OR2(a, b):
    if a is None and b is None:
        return None
    return bool(a) or bool(b)


specs = {}
for th in (0.85, 0.9):
    specs[f"val_all_fz{th}"] = lambda r, th=th: brk(sel_val(r), r, fuzzy=th)
# 级联：值提及证书为空时回退 lead-k
for fb in (1, 3, 5):
    specs[f"cascade_valFz9_lead{fb}"] = lambda r, fb=fb: (
        brk(sel_val(r), r, fuzzy=0.9) if sel_val(r) else brk(select_lead(r["s0"], fb), r))
# 级联 + 大改检测 OR
for m in (6, 7, 8):
    specs[f"cascadeFz9_or_lost{m}"] = lambda r, m=m: OR2(
        brk(sel_val(r), r, fuzzy=0.9) if sel_val(r) else brk(select_lead(r["s0"], 3), r),
        r["lost_any"] >= m)
# 值提及 OR 定义句（模糊）
specs["valFz9_or_lead1Fz9"] = lambda r: OR2(brk(sel_val(r), r, fuzzy=0.9),
                                            brk(select_lead(r["s0"], 1), r, fuzzy=0.9))
# 级联 OR dense1（模糊）
specs["cascadeFz9_or_dense1Fz9"] = lambda r: OR2(
    brk(sel_val(r), r, fuzzy=0.9) if sel_val(r) else brk(select_lead(r["s0"], 3), r),
    brk(select_dense(r["s0"], 1), r, fuzzy=0.9))

res = {"constructors": {}}
for name, fn in specs.items():
    for r in rows:
        r[name] = fn(r)
    e = evaluate(rows, name)
    res["constructors"][name] = e
    print(f"{name}: int {e['int']:.3f} {e['int_ci']} | fr {e['fr']:.3f} {e['fr_ci']}  (i={e['i']}/{e['n_ch']}, f={e['f']}/{e['n_un']})", flush=True)

for r in rows:
    r["lead5"] = brk(select_lead(r["s0"], 5), r)
res["constructors"]["lead5"] = evaluate(rows, "lead5")
res["mcnemar_vs_lead5"] = {}
for name in specs:
    res["mcnemar_vs_lead5"][name] = {
        "int_b": sum(1 for r in rows if r["harm"] and r[name] and not r["lead5"]),
        "int_c": sum(1 for r in rows if r["harm"] and not r[name] and r["lead5"]),
        "fr_b": sum(1 for r in rows if not r["harm"] and r[name] and not r["lead5"]),
        "fr_c": sum(1 for r in rows if not r["harm"] and not r[name] and r["lead5"])}

# ---- 论文修正所需：P421 占比（v2 口径） ----
tot_rem = sum(len(v["removed"]) for v in diff.values())
ex_rem = sum(1 for v in diff.values() for p, _ in v["removed"] if p not in SCHEMA_PROPS)
res["p421"] = {"total_removed_stmts": tot_rem, "ex_schema_removed": ex_rem,
               "p421_share": round((tot_rem - ex_rem) / tot_rem, 4)}
print("P421 share:", res["p421"], flush=True)
# 证书规模统计
sizes = [len(sel_val(r)) for r in rows]
res["val_cert_size"] = {"mean": round(sum(sizes) / len(sizes), 2),
                        "nonempty": round(sum(s > 0 for s in sizes) / len(sizes), 4),
                        "median": sorted(sizes)[len(sizes) // 2]}
print("val cert sizes:", res["val_cert_size"], flush=True)

json.dump(res, open(f"{OUTP}/constructor_results_v3.json", "w"), indent=1)
print("saved constructor_results_v3.json", flush=True)

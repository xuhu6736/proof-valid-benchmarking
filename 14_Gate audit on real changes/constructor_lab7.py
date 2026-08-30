"""第七轮：金标准对照——同一 1,511 实体、同一语句层真值上，
SOTA 相似度信号 vs lead-k 文本哈希 vs 冠军证书的正面比较。

相似度信号：基快照与目标快照页面（导语/全文）的词元 Jaccard 相似度，
阈值 tau 扫描。embedding 缓存对小编辑更迟钝，故词元相似度是对 SOTA
有利的保守代理（高估其拦截能力）。
"""
import json, math

DATA = "/mnt/agents/output/pvb_pilot/data/api"
OUTP = "/mnt/agents/output/pvb_pilot"
SCHEMA_PROPS = {"P421"}
exec(open("/mnt/agents/output/pvb_pilot/constructor_lab.py").read().split("def main()")[0])

pages0 = json.load(open(f"{DATA}/pages_20260401.json"))
pages1 = json.load(open(f"{DATA}/pages_20260801.json"))
tq = json.load(open(f"{DATA}/title_qid_map.json"))
ents0 = json.load(open(f"{DATA}/entities_v2_20260401.json"))
diff = {int(k): v for k, v in json.load(open(f"{OUTP}/statement_diff_v2_2026.json")).items()}

def lead_text(plain, nsent=5):
    return " ".join(sentences(plain)[:nsent]) if sentences(plain) else plain[:1500]

rows = []
for title, qid in tq.items():
    if qid not in diff or title not in pages0 or title not in pages1:
        continue
    p0 = wikitext_to_plain(pages0[title]); p1 = wikitext_to_plain(pages1[title])
    s0 = sentences(p0)
    if len(s0) < 5:
        continue
    harm = any(p not in SCHEMA_PROPS for p, _ in diff[qid]["removed"])
    lt0, lt1 = set(toks(lead_text(p0))), set(toks(lead_text(p1)))
    ft0, ft1 = set(toks(" ".join(s0))), set(toks(" ".join(sentences(p1))))
    sim_lead = len(lt0 & lt1) / max(1, len(lt0 | lt1))
    sim_full = len(ft0 & ft1) / max(1, len(ft0 | ft1))
    pats = []
    if str(qid) in ents0:
        for prop, sl in ents0[str(qid)]["claims"].items():
            for rank, snak, v in sl:
                pats.extend(value_patterns(prop, v))
    rows.append({"harm": harm, "s0": s0, "s1": sentences(p1),
                 "sim_lead": sim_lead, "sim_full": sim_full,
                 "pats": sorted(set(pats), key=len, reverse=True)})

n_ch = sum(r["harm"] for r in rows); n_un = len(rows) - n_ch
print(f"joined={len(rows)} harms={n_ch} non-harms={n_un}")

# ---------- 相似度分布：受害 vs 未受害 ----------
import statistics as st
sh = [r["sim_lead"] for r in rows if r["harm"]]; su = [r["sim_lead"] for r in rows if not r["harm"]]
fh = [r["sim_full"] for r in rows if r["harm"]]; fu = [r["sim_full"] for r in rows if not r["harm"]]
print(f"lead-sim  harm: mean {st.mean(sh):.4f} med {st.median(sh):.4f} | "
      f"non-harm: mean {st.mean(su):.4f} med {st.median(su):.4f}")
print(f"full-sim  harm: mean {st.mean(fh):.4f} med {st.median(fh):.4f} | "
      f"non-harm: mean {st.mean(fu):.4f} med {st.median(fu):.4f}")

# ---------- SOTA 相似缓存 ROC：serve iff sim >= tau ----------
# 拦截率 = P(sim<tau | harm)；误拦截 = P(sim<tau | non-harm)
def roc(feat):
    out = []
    for tau in [0.70, 0.75, 0.80, 0.85, 0.90, 0.92, 0.94, 0.95, 0.96, 0.97, 0.98, 0.99]:
        i = sum(1 for r in rows if r["harm"] and r[feat] < tau)
        f = sum(1 for r in rows if not r["harm"] and r[feat] < tau)
        out.append((tau, i / n_ch, f / n_un, i, f))
    return out

print("\ntau | 导语相似度: 拦截 / 误拦截 || 全文相似度: 拦截 / 误拦截")
for (t1, i1, f1, _, _), (t2, i2, f2, _, _) in zip(roc("sim_lead"), roc("sim_full")):
    print(f"{t1:.2f} | {i1*100:5.1f}% / {f1*100:5.1f}% || {i2*100:5.1f}% / {f2*100:5.1f}%")

# ---------- 冠军证书（复核 + 与相似度的合取） ----------
def sel_val(r): return [s for s in r["s0"] if any(p in s for p in r["pats"])]
for r in rows:
    r["s1_toks"] = [toks(s) for s in r["s1"]]
    s1_set = set(r["s1"])
    r["lost_any"] = sum(1 for s in r["s0"] if s not in s1_set)
    sv = sel_val(r)
    casc = (cert_break(sv, r["s1"], r["s1_toks"], fuzzy=0.9) if sv
            else cert_break(select_lead(r["s0"], 3), r["s1"], r["s1_toks"]))
    dense = cert_break(select_dense(r["s0"], 1), r["s1"], r["s1_toks"], fuzzy=0.9)
    r["champ"] = bool(casc or dense or r["lost_any"] >= 12)

i = sum(1 for r in rows if r["harm"] and r["champ"]); f = sum(1 for r in rows if not r["harm"] and r["champ"])
print(f"\nchampion 复核: int {i/n_ch*100:.1f}% ({i}/{n_ch}) fr {f/n_un*100:.1f}% ({f}/{n_un})")

# ---------- 合取分析：相似度信号能否再过滤冠军的误拦截 ----------
# 冠军误拦截的 153 个未受害实体，其导语相似度分布如何？
fc = [r["sim_lead"] for r in rows if not r["harm"] and r["champ"]]
print(f"冠军误拦截实体的导语相似度: mean {st.mean(fc):.4f} med {st.median(fc):.4f}, "
      f"其中 sim<0.95 的 {sum(1 for x in fc if x<0.95)}/{len(fc)}")

# 受害实体中冠军未拦截者（72 个）的相似度：是否真的完全无文本变化？
ic = [r for r in rows if r["harm"] and not r["champ"]]
nochange = sum(1 for r in ic if r["lost_any"] == 0)
print(f"冠军未拦截受害 {len(ic)} 个，其中零句丢失 {nochange} 个；"
      f"其导语相似度 mean {st.mean([r['sim_lead'] for r in ic]):.4f}")

# ---------- 端到端组合数字（4 个月 horizon, 面板受害率 8.0%） ----------
harm_rate = 0.0799
rocL = roc("sim_lead")
def at(t): 
    c=[x for x in rocL if abs(x[0]-t)<1e-9][0]; return c[1], c[2]
summary=[]
for name, inter, fr in [("sim_tau0.95",)+at(0.95), ("sim_tau0.96",)+at(0.96),
                        ("sim_tau0.98",)+at(0.98), ("lead5", 0.3385, 0.1332),
                        ("champion", i/n_ch, f/n_un)]:
    stale = harm_rate*(1-inter); miss = (1-harm_rate)*fr
    summary.append({"name":name,"interception":inter,"fr":fr,
                    "stale_serve_panel":stale,"miss_panel":miss})
    print(f"{name}: stale {stale*100:.2f}% miss {miss*100:.2f}%")
json.dump({"n_ch": n_ch, "n_un": n_un,
           "lead_sim": {"harm_mean": st.mean(sh), "harm_med": st.median(sh),
                        "nonharm_mean": st.mean(su), "nonharm_med": st.median(su)},
           "full_sim": {"harm_mean": st.mean(fh), "nonharm_mean": st.mean(fu)},
           "sim_roc_lead": roc("sim_lead"), "sim_roc_full": roc("sim_full"),
           "summary": summary,
           "champion_missed_harms": len(ic), "champion_missed_zero_loss": nochange,
           "champion_fr_lead_sim_mean": st.mean(fc),
           "champion_fr_below_095": sum(1 for x in fc if x<0.95)},
          open(f"{OUTP}/goldstandard_panel_2026.json", "w"), indent=1)
print("saved goldstandard_panel_2026.json")

import json, re, math
from collections import Counter

def wikitext_to_plain(wt):
    wt = re.sub(r"<ref[^>]*>.*?</ref>", " ", wt, flags=re.S)
    wt = re.sub(r"<ref[^>]*/>", " ", wt)
    wt = re.sub(r"\{\{[^{}]*\}\}", " ", wt)
    wt = re.sub(r"\{\{[^{}]*\}\}", " ", wt)
    wt = re.sub(r"\[\[([^|\]]*\|)?([^\]]*)\]\]", r"\2", wt)
    wt = re.sub(r"\[https?://[^\s\]]+\s*([^\]]*)\]", r"\1", wt)
    wt = re.sub(r"'{2,}", "", wt)
    wt = re.sub(r"<[^>]+>", " ", wt)
    wt = re.sub(r"^[=]+.*?[=]+$", " ", wt, flags=re.M)
    return wt

def sentences(text, min_len=40):
    parts = re.split(r"(?<=[.!?。！？])\s+|\n+", text)
    return [re.sub(r"\s+", " ", s).strip() for s in parts if len(s.strip()) >= min_len]

def wilson(k, n, z=1.96):
    if n == 0: return (0, 0, 0)
    p = k / n
    denom = 1 + z*z/n
    c = (p + z*z/(2*n)) / denom
    w = z * math.sqrt(p*(1-p)/n + z*z/(4*n*n)) / denom
    return p, c - w, c + w

pages_apr = json.load(open('/tmp/pvb_data/pages_20260401.json'))
pages_aug = json.load(open('/tmp/pvb_data/pages_20260801.json'))
tq = json.load(open('/tmp/pvb_data/title_qid_map.json'))
diff = {int(k): v for k, v in json.load(open('/tmp/pvb_data/statement_diff.json')).items()}

SCHEMA_PROPS = {"P421"}  # 时区建模式批量迁移

rows = []
for title, info in tq.items():
    qid = info['qid']
    if qid not in diff: continue
    if title not in pages_apr or title not in pages_aug: continue
    s_apr = sentences(wikitext_to_plain(pages_apr[title]))
    s_aug = set(sentences(wikitext_to_plain(pages_aug[title])))
    if len(s_apr) < 5: continue
    removed = diff[qid]['removed']
    fact_changed = len(removed) > 0
    fact_changed_sem = any(p not in SCHEMA_PROPS for p, _ in removed)
    # 多档证书：lead-1/3/5/10
    certs = {}
    for k in (1, 3, 5, 10):
        certs[k] = any(s not in s_aug for s in s_apr[:k])
    # 值承载证书：含被移除字面值的句子
    val_bearing = None
    lits = [k for p, k in removed if not k.startswith('Q')]
    if lits:
        vb_sents = [s for s in s_apr if any(re.sub(r'^\+', '', k) in s for k in lits)]
        val_bearing = (len(vb_sents), any(s not in s_aug for s in vb_sents))
    rows.append({"title": title, "qid": qid, "label": diff[qid]['label'],
                 "fact_changed": fact_changed, "fact_changed_sem": fact_changed_sem,
                 "n_removed": len(removed), "certs": certs, "val_bearing": val_bearing})

n = len(rows)
n_changed = sum(r['fact_changed'] for r in rows)
n_changed_sem = sum(r['fact_changed_sem'] for r in rows)
print(f"joined entities: {n}; fact-changed (any removal): {n_changed} ({100*n_changed/n:.1f}%); ex-schema: {n_changed_sem} ({100*n_changed_sem/n:.1f}%)")

res = {"n_joined": n, "n_changed": n_changed, "n_changed_sem": n_changed_sem}
for k in (1, 3, 5, 10):
    for lab, key in [("all", "fact_changed"), ("sem", "fact_changed_sem")]:
        ch = [r for r in rows if r[key]]
        un = [r for r in rows if not r[key]]
        i_k = sum(r['certs'][k] for r in ch)   # 拦截（证书破损|事实变更）
        f_k = sum(r['certs'][k] for r in un)   # 误拒（证书破损|事实未变）
        p_i, lo_i, hi_i = wilson(i_k, len(ch))
        p_f, lo_f, hi_f = wilson(f_k, len(un))
        res[f"lead{k}_{lab}"] = {
            "interception": round(p_i, 4), "interception_ci": [round(lo_i,4), round(hi_i,4)], "n_ch": len(ch),
            "false_reject": round(p_f, 4), "false_reject_ci": [round(lo_f,4), round(hi_f,4)], "n_un": len(un)}
        print(f"lead-{k} [{lab}]: interception {i_k}/{len(ch)} = {p_i:.3f} [{lo_i:.3f},{hi_i:.3f}] | "
              f"false-reject {f_k}/{len(un)} = {p_f:.3f} [{lo_f:.3f},{hi_f:.3f}]")

# 值承载证书
vb = [r for r in rows if r['val_bearing'] and r['val_bearing'][0] > 0]
vb_broken = sum(1 for r in vb if r['val_bearing'][1])
p, lo, hi = wilson(vb_broken, len(vb))
res["value_bearing_cert"] = {"n": len(vb), "broken": vb_broken, "rate": round(p,4), "ci": [round(lo,4), round(hi,4)]}
print(f"value-bearing certs: {vb_broken}/{len(vb)} = {p:.3f} [{lo:.3f},{hi:.3f}] (含被移除值的旧句子在 T_new 消失/改变的比例)")

json.dump(res, open('/tmp/pvb_data/roc_results.json','w'), indent=1)
json.dump(rows, open('/tmp/pvb_data/roc_rows.json','w'))
print("saved roc_results.json")

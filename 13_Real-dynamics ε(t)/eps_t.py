"""ε(t) 实测曲线: 句子级 (enwiki 5 期) + 声明级 (wikidata 4 期), 均以 20260401 为基期。
输出 /tmp/pvb_data/eps_t.json 并复制到 /mnt/agents/output/pvb_pilot/。
"""
import json, math, re
from statsmodels.stats.proportion import proportion_confint

# 与 module2_analyze.py 完全一致的规整管线(复制以避免导入副作用)
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

WORK = "/tmp/pvb_data"
OUT = "/mnt/agents/output/pvb_pilot"
BASE = "20260401"
TXT_DATES = ["20260501", "20260601", "20260701", "20260801"]
STMT_DATES = ["20260501", "20260701", "20260801"]
SCHEMA_PROPS = {"P421"}

def wilson(k, n):
    lo, hi = proportion_confint(k, n, alpha=0.05, method="wilson")
    return [round(lo, 4), round(hi, 4)]

# ---------- 句子级 ----------
pages = {d: json.load(open(f"{WORK}/pages_{d}.json")) for d in [BASE] + TXT_DATES}
base_sents = {}
for t, wt in pages[BASE].items():
    s = sentences(wikitext_to_plain(wt))
    if s:
        base_sents[t] = s
print("base pages with sentences:", len(base_sents), flush=True)

text_rows = []
for d in TXT_DATES:
    rows, cert5_broken, cert10_broken, edited = [], 0, 0, 0
    for t, s0 in base_sents.items():
        if t not in pages[d]:
            continue
        s1 = set(sentences(wikitext_to_plain(pages[d][t])))
        lost = sum(1 for s in s0 if s not in s1)
        rows.append((len(s0), lost))
        if pages[BASE][t] != pages[d][t]:
            edited += 1
        cert5_broken += any(s not in s1 for s in s0[:5])
        cert10_broken += any(s not in s1 for s in s0[:10])
    n_p = len(rows)
    tot = sum(r[0] for r in rows)
    lost = sum(r[1] for r in rows)
    eps = lost / tot
    per_page = [r[1] / r[0] for r in rows]
    text_rows.append({
        "window": f"{BASE}->{d}", "months": (int(d[4:6]) - 4), "n_pages": n_p,
        "n_sentences": tot, "sentences_lost": lost,
        "eps_text": eps, "eps_text_ci95": wilson(lost, tot),
        "per_page_eps_mean": sum(per_page) / n_p,
        "page_edit_rate": edited / n_p,
        "lead5_cert_break": cert5_broken / n_p, "lead5_cert_break_ci95": wilson(cert5_broken, n_p),
        "lead10_cert_break": cert10_broken / n_p,
    })
    print("text", d, "eps", round(eps, 4), flush=True)

# ---------- 声明级 ----------
def truthy_keys(claims):
    """truthy 陈述键集合: preferred 优先于 normal; 键 = (prop, 值规范化)"""
    out = set()
    for prop, stmts in claims.items():
        prefs = [s for s in stmts if s[0] == "preferred"]
        use = prefs if prefs else [s for s in stmts if s[0] == "normal"]
        for rank, snaktype, dv in use:
            if snaktype != "value" or dv is None:
                continue
            if isinstance(dv, dict):
                v = dv.get("id") or dv.get("text") or dv.get("time") or dv.get("amount")
            else:
                v = str(dv)
            if v is not None:
                out.add((prop, str(v)))
    return out

title_qid = json.load(open(f"{WORK}/title_qid_map.json"))
ents = {d: json.load(open(f"{WORK}/entities_{d}.json")) for d in [BASE] + STMT_DATES}
base_truthy = {q: truthy_keys(ents[BASE][q]) for q in ents[BASE]}
print("base entities:", len(base_truthy), flush=True)

stmt_rows = []
for d in STMT_DATES:
    common = [q for q in base_truthy if q in ents[d]]
    tot = lost = lost_ex = tot_ex = 0
    ent_any_removal = ent_any_removal_ex = 0
    for q in common:
        s0, s1 = base_truthy[q], truthy_keys(ents[d][q])
        if not s0:
            continue
        rem = s0 - s1
        tot += len(s0); lost += len(rem)
        s0e = {s for s in s0 if s[0] not in SCHEMA_PROPS}
        reme = {s for s in rem if s[0] not in SCHEMA_PROPS}
        tot_ex += len(s0e); lost_ex += len(reme)
        ent_any_removal += bool(rem)
        ent_any_removal_ex += bool(reme)
    n_e = len(common)
    stmt_rows.append({
        "window": f"{BASE}->{d}", "months": (int(d[4:6]) - 4), "n_entities": n_e,
        "n_statements": tot, "removed": lost,
        "eps_stmt": lost / tot, "eps_stmt_ci95": wilson(lost, tot),
        "eps_stmt_exschema": lost_ex / tot_ex, "eps_stmt_exschema_ci95": wilson(lost_ex, tot_ex),
        "entity_any_removal": ent_any_removal / n_e,
        "entity_any_removal_ci95": wilson(ent_any_removal, n_e),
        "entity_any_removal_exschema": ent_any_removal_ex / n_e,
        "entity_any_removal_exschema_ci95": wilson(ent_any_removal_ex, n_e),
    })
    print("stmt", d, "eps", round(lost / tot, 4), "ex-schema", round(lost_ex / tot_ex, 4), flush=True)

res = {"base": BASE, "text": text_rows, "stmt": stmt_rows,
       "note": "eps_text = 基期句子在目标期消失比例(最小长度40字符); eps_stmt = 基期 truthy 陈述在目标期被移除比例; exschema 排除 P421 时区批量迁移"}
json.dump(res, open(f"{WORK}/eps_t.json", "w"), indent=1)
json.dump(res, open(f"{OUT}/eps_t.json", "w"), indent=1)
print("saved eps_t.json", flush=True)

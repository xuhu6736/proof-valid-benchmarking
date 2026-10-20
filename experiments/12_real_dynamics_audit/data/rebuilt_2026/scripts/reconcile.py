"""派生层重算与全量对账：eps_t（文本+声明）与 statement_diff，对照保留锚点。
Wilson 区间用纯 Python 复刻 statsmodels proportion_confint(method="wilson")。
"""
import json, math, os, re

HERE = os.path.dirname(os.path.abspath(__file__))
RET = os.path.join(HERE, "..", "proof-valid-benchmarking",
                   "experiments", "12_real_dynamics_audit")
BASE = "20260401"
TXT_DATES = ["20260501", "20260601", "20260701", "20260801"]
STMT_DATES = ["20260501", "20260701", "20260801"]
SCHEMA_PROPS = {"P421"}

def wilson(k, n):
    z = 1.959963984540054
    p = k / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return [round(center - half, 4), round(center + half, 4)]

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

def truthy_keys(claims):
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

# ---------- 文本层 ----------
pages = {d: json.load(open(f"{HERE}/pages_{d}.json")) for d in [BASE] + TXT_DATES}
base_sents = {}
for t, wt in pages[BASE].items():
    s = sentences(wikitext_to_plain(wt))
    if s:
        base_sents[t] = s
text_rows = []
for d in TXT_DATES:
    rows, cert5, cert10, edited = [], 0, 0, 0
    for t, s0 in base_sents.items():
        if t not in pages[d]:
            continue
        s1 = set(sentences(wikitext_to_plain(pages[d][t])))
        lost = sum(1 for s in s0 if s not in s1)
        rows.append((len(s0), lost))
        if pages[BASE][t] != pages[d][t]:
            edited += 1
        cert5 += any(s not in s1 for s in s0[:5])
        cert10 += any(s not in s1 for s in s0[:10])
    n_p = len(rows)
    tot = sum(r[0] for r in rows)
    lost = sum(r[1] for r in rows)
    text_rows.append({
        "window": f"{BASE}->{d}", "n_pages": n_p, "n_sentences": tot,
        "sentences_lost": lost, "eps_text": lost / tot,
        "eps_text_ci95": wilson(lost, tot),
        "per_page_eps_mean": sum(r[1] / r[0] for r in rows) / n_p,
        "page_edit_rate": edited / n_p,
        "lead5_cert_break": cert5 / n_p, "lead5_cert_break_ci95": wilson(cert5, n_p),
        "lead10_cert_break": cert10 / n_p,
    })

# ---------- 声明层 ----------
ents = {d: json.load(open(f"{HERE}/entities_v2_{d}.json")) for d in [BASE] + STMT_DATES}
base_truthy = {q: truthy_keys(e["claims"]) for q, e in ents[BASE].items()}
stmt_rows = []
for d in STMT_DATES:
    common = [q for q in base_truthy if q in ents[d]]
    tot = lost = lost_ex = tot_ex = 0
    ent_rem = ent_rem_ex = 0
    for q in common:
        s0, s1 = base_truthy[q], truthy_keys(ents[d][q]["claims"])
        if not s0:
            continue
        rem = s0 - s1
        tot += len(s0); lost += len(rem)
        s0e = {s for s in s0 if s[0] not in SCHEMA_PROPS}
        reme = {s for s in rem if s[0] not in SCHEMA_PROPS}
        tot_ex += len(s0e); lost_ex += len(reme)
        ent_rem += bool(rem); ent_rem_ex += bool(reme)
    n_e = len(common)
    stmt_rows.append({
        "window": f"{BASE}->{d}", "n_entities": n_e, "n_statements": tot,
        "removed": lost, "eps_stmt": lost / tot, "eps_stmt_ci95": wilson(lost, tot),
        "eps_stmt_exschema": lost_ex / tot_ex,
        "eps_stmt_exschema_ci95": wilson(lost_ex, tot_ex),
        "entity_any_removal": ent_rem / n_e,
        "entity_any_removal_exschema": ent_rem_ex / n_e,
        "entity_any_removal_exschema_ci95": wilson(ent_rem_ex, n_e),
    })

# ---------- statement diff ----------
final = {q: truthy_keys(e["claims"]) for q, e in ents["20260801"].items()}
diff = {}
for q, s0 in base_truthy.items():
    if q not in final:
        continue
    s1 = final[q]
    diff[q] = {"label": ents[BASE][q].get("label"),
               "removed": sorted([list(t) for t in (s0 - s1)]),
               "added": sorted([list(t) for t in (s1 - s0)]),
               "n_s0": len(s0), "n_s1": len(s1)}

# ---------- 对账 ----------
ret_eps = json.load(open(f"{RET}/eps_t.json"))
ret_stmt = json.load(open(f"{RET}/eps_stmt_v2.json"))
ret_diff = json.load(open(f"{RET}/statement_diff_v2_2026.json"))

print("=== TEXT WINDOWS ===")
for got, want in zip(text_rows, ret_eps["text"]):
    keys = ["n_pages", "n_sentences", "sentences_lost", "page_edit_rate",
            "lead5_cert_break", "lead10_cert_break", "eps_text"]
    bad = [k for k in keys if abs(got[k] - want[k]) > 1e-9]
    ci_ok = got["eps_text_ci95"] == want["eps_text_ci95"]
    print(got["window"], "MATCH" if not bad else f"DIFF {bad}",
          f"(eps={got['eps_text']:.6f} vs {want['eps_text']:.6f}, ci={'ok' if ci_ok else 'DIFF'})")

print("=== STMT WINDOWS ===")
for got, want in zip(stmt_rows, ret_stmt):
    keys = ["n_entities", "n_statements", "removed", "eps_stmt",
            "eps_stmt_exschema", "entity_any_removal", "entity_any_removal_exschema"]
    bad = [k for k in keys if abs(got[k] - want[k]) > 1e-9]
    print(got["window"], "MATCH" if not bad else f"DIFF {[(k, got[k], want[k]) for k in bad]}")

print("=== STATEMENT DIFF ===")
common = set(diff) & set(ret_diff)
n_exact = 0
mism = []
for q in common:
    g, w = diff[q], ret_diff[q]
    if g["removed"] == w["removed"] and g["added"] == w["added"] \
       and g["n_s0"] == w["n_s0"] and g["n_s1"] == w["n_s1"]:
        n_exact += 1
    else:
        mism.append(q)
print(f"entities rebuilt={len(diff)} retained={len(ret_diff)} common={len(common)}")
print(f"per-entity exact (removed+added+n_s0+n_s1): {n_exact}/{len(common)}")
if mism:
    q = mism[0]
    print("first mismatch", q, diff[q], "vs", ret_diff[q])

json.dump({"base": BASE, "text": text_rows, "stmt": stmt_rows},
          open(f"{HERE}/eps_t_rebuilt.json", "w"), indent=1)
json.dump(diff, open(f"{HERE}/statement_diff_rebuilt.json", "w"), indent=1)
print("saved eps_t_rebuilt.json / statement_diff_rebuilt.json")

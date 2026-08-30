"""声明层重建（v2 实体，正确 QID 键）：
1) statement_diff_v2.json: 基期->末期 truthy 陈述 per-entity removed/added
2) eps_stmt_v2.json: 三个 horizon 的声明级/实体级抹除率（ex-P421 口径）
"""
import json, math, sys

WORK = "/tmp/pvb_data"
OUTP = "/mnt/agents/output/pvb_pilot"
BASE = "20260401"
STMT_DATES = ["20260501", "20260701", "20260801"]
SCHEMA_PROPS = {"P421"}


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


def wilson(k, n):
    from statsmodels.stats.proportion import proportion_confint
    lo, hi = proportion_confint(k, n, alpha=0.05, method="wilson")
    return [round(lo, 4), round(hi, 4)]


def val_str(t):
    p, v = t
    return [p, v]


ents = {d: json.load(open(f"{WORK}/entities_v2_{d}.json")) for d in [BASE] + STMT_DATES}
for d in ents:
    print(d, "entities:", len(ents[d]), flush=True)
base_truthy = {q: truthy_keys(e["claims"]) for q, e in ents[BASE].items()}

# ---------- per-horizon 声明级统计（替代 eps_t.json 的 stmt 行） ----------
rows = []
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
    rows.append({
        "window": f"{BASE}->{d}", "months": (int(d[4:6]) - 4), "n_entities": n_e,
        "n_statements": tot, "removed": lost,
        "eps_stmt": lost / tot, "eps_stmt_ci95": wilson(lost, tot),
        "eps_stmt_exschema": lost_ex / tot_ex, "eps_stmt_exschema_ci95": wilson(lost_ex, tot_ex),
        "entity_any_removal": ent_rem / n_e, "entity_any_removal_ci95": wilson(ent_rem, n_e),
        "entity_any_removal_exschema": ent_rem_ex / n_e,
        "entity_any_removal_exschema_ci95": wilson(ent_rem_ex, n_e),
    })
    print("stmt", d, "eps", round(lost / tot, 4), "ex-schema", round(lost_ex / tot_ex, 4),
          "ent_ex", round(ent_rem_ex / n_e, 4), flush=True)
json.dump(rows, open(f"{OUTP}/eps_stmt_v2.json", "w"), indent=1)

# ---------- base->final diff ----------
final = {q: truthy_keys(e["claims"]) for q, e in ents["20260801"].items()}
diff = {}
for q, s0 in base_truthy.items():
    if q not in final:
        continue
    s1 = final[q]
    removed = sorted(s0 - s1)
    added = sorted(s1 - s0)
    if True:  # 保留全部共同实体（含无变化），供 ROC 全量联结
        diff[q] = {"label": ents[BASE][q].get("label"),
                   "removed": [val_str(t) for t in removed],
                   "added": [val_str(t) for t in added],
                   "n_s0": len(s0), "n_s1": len(s1)}
json.dump(diff, open(f"{OUTP}/statement_diff_v2_2026.json", "w"), indent=1)
print("diff entities:", len(diff), flush=True)

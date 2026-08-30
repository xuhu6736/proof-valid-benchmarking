"""PVB 事后构造器实验室：在 2026 真实动态面板上构造并评估文本证书构造器族。
输入（镜像重建于 /tmp/pvb_data）：pages_{04,08}.json, title_qid_map.json, entities_20260401.json
危害真值：/mnt/agents/output/pvb_pilot/statement_diff_2026.json（qid->removed/added）
用法: python3 constructor_lab.py [validate|run]
"""
import json, math, os, re, sys
from collections import Counter

WORK = "/tmp/pvb_data"
OUTP = "/mnt/agents/output/pvb_pilot"
SCHEMA_PROPS = {"P421"}


# ---------- 文本规整（与 roc_close.py / eps_t.py 完全一致） ----------
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


def toks(s):
    return {t for t in re.findall(r"[a-z0-9]+", s.lower()) if len(t) >= 2}


def wilson(k, n, z=1.96):
    if n == 0:
        return (0.0, 0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    w = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return p, c - w, c + w


# ---------- 基期声明值 → 文本匹配模式 ----------
YEAR_RE = re.compile(r"\b(1[0-9]{3}|20[0-9]{2})\b")
NUM_RE = re.compile(r"\d")


def value_patterns(prop, v):
    """把 truthy 声明值规范化为可在正文中匹配的模式列表（保守、高精度）"""
    if isinstance(v, dict):  # wd_pipeline slim: {"id"/"text"/"time"/"amount"}
        v = v.get("id") or v.get("text") or v.get("time") or v.get("amount") or ""
    v = str(v)
    pats = []
    if v.startswith("Q"):
        return pats  # 实体值暂不解析标签（无标签解析器）
    if v.startswith("+" ) or v.startswith("-"):
        if "T" in v:  # 时间 +1956-07-09T00:00:00Z
            m = re.match(r"^[+](\d{3,4})-", v)
            if m:
                pats.append(m.group(1))
        else:  # 数量 +1000000 / +1.0E6
            try:
                x = float(v)
                if x == int(x) and abs(x) >= 1000:
                    pats.append(f"{int(x):,}")
                    pats.append(str(int(x)))
            except ValueError:
                pass
        return pats
    if len(v) >= 4 and not v.isdigit():
        pats.append(v)
    return pats


# ---------- 构造器族 ----------
def select_lead(s0, k):
    return list(s0[:k])


def select_dense(s0, k):
    scored = sorted(enumerate(s0), key=lambda x: (-(len(YEAR_RE.findall(x[1])) + min(len(NUM_RE.findall(x[1])), 6) / 10), x[0]))
    return [s0[i] for i, _ in scored[:k]]


def select_valmention(s0, pats_all, k=None):
    sel = [s for s in s0 if any(p in s for p in pats_all)]
    return sel if k is None else sel[:k]


def select_self(s0, label, k=None):
    if not label or len(label) < 4:
        return []
    low = label.lower()
    sel = [s for s in s0 if low in s.lower()]
    return sel if k is None else sel[:k]


def cert_break(sel, s1_list, s1_toks, fuzzy=1.0, quorum=1):
    if not sel:
        return None  # 证书为空：不适用
    lost = 0
    s1_set = set(s1_list)
    for s in sel:
        if fuzzy >= 1.0:
            gone = s not in s1_set
        else:
            st = toks(s)
            gone = not any(len(st & tt) / max(1, len(st | tt)) >= fuzzy for tt in s1_toks)
        lost += gone
    return lost >= quorum


def evaluate(rows, key):
    ch = [r for r in rows if r["harm"] and r[key] is not None]
    un = [r for r in rows if not r["harm"] and r[key] is not None]
    i_k = sum(r[key] for r in ch)
    f_k = sum(r[key] for r in un)
    pi, lo_i, hi_i = wilson(i_k, len(ch))
    pf, lo_f, hi_f = wilson(f_k, len(un))
    return {"int": round(pi, 4), "int_ci": [round(lo_i, 4), round(hi_i, 4)],
            "fr": round(pf, 4), "fr_ci": [round(lo_f, 4), round(hi_f, 4)],
            "i": i_k, "n_ch": len(ch), "f": f_k, "n_un": len(un)}


def main():
    pages0 = json.load(open(f"{WORK}/pages_20260401.json"))
    pages1 = json.load(open(f"{WORK}/pages_20260801.json"))
    tq = json.load(open(f"{WORK}/title_qid_map.json"))
    ents0 = json.load(open(f"{WORK}/entities_v2_20260401.json"))  # v2: 正确 QID 键
    diff = {int(k): v for k, v in json.load(open(f"{OUTP}/statement_diff_v2_2026.json")).items()}

    rows = []
    for title, qid in tq.items():
        if qid not in diff or title not in pages0 or title not in pages1:
            continue
        s0 = sentences(wikitext_to_plain(pages0[title]))
        if len(s0) < 5:
            continue
        s1 = sentences(wikitext_to_plain(pages1[title]))
        removed = [r for r in diff[qid]["removed"]]
        harm = any(p not in SCHEMA_PROPS for p, _ in removed)
        # 基期声明值模式（仅字面量）；v2 实体结构 {"claims":..., "label":...}
        pats = []
        if str(qid) in ents0:
            for prop, sl in ents0[str(qid)]["claims"].items():
                for rank, snak, v in sl:
                    pats.extend(value_patterns(prop, v))
        pats = sorted(set(pats), key=len, reverse=True)
        label = (ents0.get(str(qid)) or {}).get("label") or diff[qid].get("label")
        rows.append({"title": title, "qid": qid, "harm": harm,
                     "any_removed": bool(removed),
                     "s0": s0, "s1": s1, "pats": pats, "label": label})
    n = len(rows)
    nh = sum(r["harm"] for r in rows)
    print(f"joined={n} harms_ex_schema={nh} (期望 1378 / 58)", flush=True)

    # ---------- 校验基线 ----------
    res = {"n_joined": n, "n_harm": nh, "constructors": {}}
    for r in rows:
        r["s1_toks"] = [toks(s) for s in r["s1"]]
        r["lost_any"] = sum(1 for s in r["s0"] if s not in set(r["s1"]))
    for k in (1, 3, 5, 10):
        key = f"lead{k}"
        for r in rows:
            r[key] = cert_break(select_lead(r["s0"], k), r["s1"], r["s1_toks"])
        res["constructors"][key] = evaluate(rows, key)
        e = res["constructors"][key]
        print(f"{key}: int {e['int']:.3f} {e['int_ci']} | fr {e['fr']:.3f} {e['fr_ci']}", flush=True)

    # ---------- 天花板诊断 ----------
    diag = {}
    harmed = [r for r in rows if r["harm"]]
    diag["harm_with_any_sent_lost"] = sum(r["lost_any"] > 0 for r in harmed) / len(harmed)
    unharmed = [r for r in rows if not r["harm"]]
    diag["unharmed_with_any_sent_lost"] = sum(r["lost_any"] > 0 for r in unharmed) / len(unharmed)
    # 危害实体的被移除字面值是否出现在基期正文中（值提及构造的覆盖上限）
    cov = cov_hit = 0
    for r in harmed:
        lits = [v for p, v in diff[r["qid"]]["removed"]
                if p not in SCHEMA_PROPS and not v.startswith("Q")]
        if not lits:
            continue
        pats_r = []
        for v in lits:
            pats_r.extend(value_patterns("", v))
        if not pats_r:
            continue
        cov += 1
        s1_set = set(r["s1"])
        if any(any(p in s for p in pats_r) and s not in s1_set for s in r["s0"]):
            cov_hit += 1
    diag["harm_lit_coverable"] = cov / len(harmed)
    diag["harm_lit_coverable_and_lost"] = cov_hit / len(harmed)
    print("diagnostics:", json.dumps(diag, indent=1), flush=True)
    res["diagnostics"] = diag

    if "validate" in sys.argv:
        json.dump(res, open(f"{OUTP}/constructor_validate.json", "w"), indent=1)
        return

    # ---------- 构造器族评估 ----------
    specs = {}
    for k in (1, 2, 3, 5, 8):
        specs[f"dense{k}"] = lambda r, k=k: cert_break(select_dense(r["s0"], k), r["s1"], r["s1_toks"])
    specs["valmention_all"] = lambda r: cert_break(select_valmention(r["s0"], r["pats"]), r["s1"], r["s1_toks"])
    specs["valmention_top5"] = lambda r: cert_break(select_valmention(r["s0"], r["pats"], 5), r["s1"], r["s1_toks"])
    specs["valmention_q2"] = lambda r: cert_break(select_valmention(r["s0"], r["pats"]), r["s1"], r["s1_toks"], quorum=2)
    specs["selfmention_top5"] = lambda r: cert_break(select_self(r["s0"], r["label"], 5), r["s1"], r["s1_toks"])
    specs["lead1+val_top4"] = lambda r: cert_break(
        select_lead(r["s0"], 1) + [s for s in select_valmention(r["s0"], r["pats"]) if s != r["s0"][0]][:4],
        r["s1"], r["s1_toks"])
    for th in (0.9, 0.8):
        specs[f"lead5_fuzzy{th}"] = lambda r, th=th: cert_break(select_lead(r["s0"], 5), r["s1"], r["s1_toks"], fuzzy=th)
        specs[f"val_fuzzy{th}"] = lambda r, th=th: cert_break(select_valmention(r["s0"], r["pats"]), r["s1"], r["s1_toks"], fuzzy=th)
    for m in (1, 2, 3, 5):
        specs[f"lost>={m}"] = lambda r, m=m: (r["lost_any"] >= m)

    for name, fn in specs.items():
        for r in rows:
            r[name] = fn(r)
        e = evaluate(rows, name)
        res["constructors"][name] = e
        cert_sizes = [len(r["s0"]) for r in rows[:0]]  # placeholder
        print(f"{name}: int {e['int']:.3f} {e['int_ci']} | fr {e['fr']:.3f} {e['fr_ci']}  (i={e['i']}/{e['n_ch']}, f={e['f']}/{e['n_un']})", flush=True)

    # ---------- 与 lead-5 的配对 McNemar ----------
    def mcnemar(key_a, key_b, mask):
        b = sum(1 for r in rows if mask(r) and r[key_a] and not r[key_b])
        c = sum(1 for r in rows if mask(r) and not r[key_a] and r[key_b])
        if b + c == 0:
            return (b, c, 1.0)
        chi2 = (abs(b - c) - 1) ** 2 / (b + c)
        from math import erf
        p = 1 - erf(math.sqrt(chi2 / 2)) * 0  # placeholder, replaced below
        return (b, c, chi2)

    res["mcnemar"] = {}
    for name in specs:
        b_int = sum(1 for r in rows if r["harm"] and r[name] and not r["lead5"])
        c_int = sum(1 for r in rows if r["harm"] and not r[name] and r["lead5"])
        b_fr = sum(1 for r in rows if not r["harm"] and r[name] and not r["lead5"])
        c_fr = sum(1 for r in rows if not r["harm"] and not r[name] and r["lead5"])
        res["mcnemar"][name] = {"int_b": b_int, "int_c": c_int, "fr_b": b_fr, "fr_c": c_fr}

    # 精简 rows 以便持久化（去掉句子本体）
    slim_rows = [{"title": r["title"], "qid": r["qid"], "harm": r["harm"],
                  "n_s0": len(r["s0"]), "n_pats": len(r["pats"]), "lost_any": r["lost_any"],
                  **{k: r[k] for k in list(res["constructors"].keys())}}
                 for r in rows]
    json.dump({"summary": res, "rows": slim_rows}, open(f"{OUTP}/constructor_results.json", "w"), indent=1)
    print("saved constructor_results.json", flush=True)


if __name__ == "__main__":
    main()

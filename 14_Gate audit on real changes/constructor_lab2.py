"""构造器实验室第二轮：链接密度选择、模糊阈值细化、组合证书（OR/quorum）。
目标：在 FR≈10% 约束下最大化拦截率，并给出与 lead-5 的配对显著性。
"""
import json, math, re, sys
from collections import Counter

WORK = "/tmp/pvb_data"
OUTP = "/mnt/agents/output/pvb_pilot"
SCHEMA_PROPS = {"P421"}

exec(open("/mnt/agents/output/pvb_pilot/constructor_lab.py").read().split("def main()")[0])


def sentences_with_links(wt, min_len=40):
    """保留链接信息的句子抽取：返回 [(plain_sentence, n_links)]，与 sentences() 对齐"""
    wt = re.sub(r"<ref[^>]*>.*?</ref>", " ", wt, flags=re.S)
    wt = re.sub(r"<ref[^>]*/>", " ", wt)
    wt = re.sub(r"\{\{[^{}]*\}\}", " ", wt)
    wt = re.sub(r"\{\{[^{}]*\}\}", " ", wt)
    wt = re.sub(r"'{2,}", "", wt)
    wt = re.sub(r"<[^>]+>", " ", wt)
    wt = re.sub(r"^[=]+.*?[=]+$", " ", wt, flags=re.M)
    parts = re.split(r"(?<=[.!?。！？])\s+|\n+", wt)
    out = []
    for p in parts:
        n_links = len(re.findall(r"\[\[[^\]]+\]\]", p))
        plain = re.sub(r"\[\[([^|\]]*\|)?([^\]]*)\]\]", r"\2", p)
        plain = re.sub(r"\[https?://[^\s\]]+\s*([^\]]*)\]", r"\1", plain)
        plain = re.sub(r"\s+", " ", plain).strip()
        if len(plain) >= min_len:
            out.append((plain, n_links))
    return out


def main():
    pages0 = json.load(open(f"{WORK}/pages_20260401.json"))
    pages1 = json.load(open(f"{WORK}/pages_20260801.json"))
    tq = json.load(open(f"{WORK}/title_qid_map.json"))
    ents0 = json.load(open(f"{WORK}/entities_v2_20260401.json"))
    diff = {int(k): v for k, v in json.load(open(f"{OUTP}/statement_diff_v2_2026.json")).items()}

    rows = []
    for title, qid in tq.items():
        if qid not in diff or title not in pages0 or title not in pages1:
            continue
        sl0 = sentences_with_links(pages0[title])
        if len(sl0) < 5:
            continue
        s0 = [p for p, _ in sl0]
        links0 = [nl for _, nl in sl0]
        s1 = sentences(wikitext_to_plain(pages1[title]))
        removed = diff[qid]["removed"]
        harm = any(p not in SCHEMA_PROPS for p, _ in removed)
        pats = []
        if str(qid) in ents0:
            for prop, sl in ents0[str(qid)]["claims"].items():
                for rank, snak, v in sl:
                    pats.extend(value_patterns(prop, v))
        pats = sorted(set(pats), key=len, reverse=True)
        label = (ents0.get(str(qid)) or {}).get("label") or diff[qid].get("label")
        rows.append({"title": title, "qid": qid, "harm": harm,
                     "s0": s0, "links0": links0, "s1": s1, "pats": pats, "label": label})
    n = len(rows); nh = sum(r["harm"] for r in rows)
    print(f"joined={n} harms={nh}", flush=True)
    for r in rows:
        r["s1_toks"] = [toks(s) for s in r["s1"]]
        s1_set = set(r["s1"])
        r["lost_any"] = sum(1 for s in r["s0"] if s not in s1_set)

    def sel_link(r, k):
        idx = sorted(range(len(r["s0"])), key=lambda i: (-r["links0"][i], i))
        return [r["s0"][i] for i in idx[:k] if r["links0"][i] > 0]

    def sel_val(r, k=None):
        sel = [s for s in r["s0"] if any(p in s for p in r["pats"])]
        return sel if k is None else sel[:k]

    def brk(sel, r, fuzzy=1.0, quorum=1):
        return cert_break(sel, r["s1"], r["s1_toks"], fuzzy=fuzzy, quorum=quorum)

    def OR(*flags):
        return any(f for f in flags if f is not None) if any(f is not None for f in flags) else None

    specs = {}
    # 链接密度证书
    for k in (3, 5, 8):
        specs[f"link{k}"] = lambda r, k=k: brk(sel_link(r, k), r)
    # 值提及 + 模糊阈值细化
    for th in (0.85, 0.9):
        for k in (5, 8, 12):
            specs[f"val_top{k}_fz{th}"] = lambda r, k=k, th=th: brk(sel_val(r, k), r, fuzzy=th)
        specs[f"val_all_fz{th}"] = lambda r, th=th: brk(sel_val(r), r, fuzzy=th)
        specs[f"val_all_fz{th}_q2"] = lambda r, th=th: brk(sel_val(r), r, fuzzy=th, quorum=2)
    # 组合证书（OR）
    specs["valFz9+link5"] = lambda r: OR(brk(sel_val(r), r, fuzzy=0.9), brk(sel_link(r, 5), r))
    specs["valFz9+dense3"] = lambda r: OR(brk(sel_val(r), r, fuzzy=0.9),
                                          brk(select_dense(r["s0"], 3), r))
    specs["valFz9+lost5"] = lambda r: OR(brk(sel_val(r), r, fuzzy=0.9), (r["lost_any"] >= 5))
    specs["valFz9+lead1fz9"] = lambda r: OR(brk(sel_val(r), r, fuzzy=0.9),
                                            brk(select_lead(r["s0"], 1), r, fuzzy=0.9))
    specs["val5+link3_fz9"] = lambda r: OR(brk(sel_val(r, 5), r, fuzzy=0.9), brk(sel_link(r, 3), r, fuzzy=0.9))
    specs["val5+link3+dense2_fz9"] = lambda r: OR(brk(sel_val(r, 5), r, fuzzy=0.9),
                                                 brk(sel_link(r, 3), r, fuzzy=0.9),
                                                 brk(select_dense(r["s0"], 2), r, fuzzy=0.9))
    res = {"n_joined": n, "n_harm": nh, "constructors": {}}
    for name, fn in specs.items():
        for r in rows:
            r[name] = fn(r)
        e = evaluate(rows, name)
        res["constructors"][name] = e
        sizes = [len(sel_val(r)) for r in rows] if name.startswith("val") else [k for k in (0,)]
        print(f"{name}: int {e['int']:.3f} {e['int_ci']} | fr {e['fr']:.3f} {e['fr_ci']}  (i={e['i']}/{e['n_ch']}, f={e['f']}/{e['n_un']})", flush=True)

    # 冠军与 lead-5 的配对比较
    for r in rows:
        r["lead5"] = brk(select_lead(r["s0"], 5), r)
    res["constructors"]["lead5"] = evaluate(rows, "lead5")
    res["mcnemar_vs_lead5"] = {}
    for name in specs:
        b_i = sum(1 for r in rows if r["harm"] and r[name] and not r["lead5"])
        c_i = sum(1 for r in rows if r["harm"] and not r[name] and r["lead5"])
        b_f = sum(1 for r in rows if not r["harm"] and r[name] and not r["lead5"])
        c_f = sum(1 for r in rows if not r["harm"] and not r[name] and r["lead5"])
        res["mcnemar_vs_lead5"][name] = {"int_b": b_i, "int_c": c_i, "fr_b": b_f, "fr_c": c_f}

    # 证书规模（公平性）
    res["cert_size"] = {
        "lead5": 5,
        "valmention_mean": round(sum(min(len(sel_val(r)), 1e9) for r in rows) / n, 2),
        "valmention_nonempty": sum(1 for r in rows if sel_val(r)) / n,
        "link5_mean_avail": round(sum(min(len(sel_link(r, 5)), 5) for r in rows) / n, 2),
    }
    slim = [{"title": r["title"], "qid": r["qid"], "harm": r["harm"], "lost_any": r["lost_any"],
             **{k: r[k] for k in list(specs.keys()) + ["lead5"]}} for r in rows]
    json.dump({"summary": res, "rows": slim}, open(f"{OUTP}/constructor_results_v2.json", "w"), indent=1)
    print("saved constructor_results_v2.json", flush=True)


if __name__ == "__main__":
    main()

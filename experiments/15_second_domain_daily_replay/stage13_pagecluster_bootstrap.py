#!/usr/bin/env python3
"""Stage 13: page-cluster bootstrap re-analysis of the dual-generator
daily replay (reviewer round-8 request 3.8/B1).

Answer-level resampling (stages 4/7) treats answers from the same
Wikipedia page as independent; answers on one page share the page's
edit process, so this script re-computes every headline contrast with
the bootstrap unit set to the PAGE (cluster = all accepted answers
and their tick cases belonging to one page).  Contrasts:

  B1'  sent-asserted stale-delivery rate: Kimi - GPT, paired on the
       pages cached by BOTH generators (common-page subset)
  B2'  sent-asserted vs TTL-14d stale-delivery rate, per generator
  B3'  sent-asserted vs TTL-1d regeneration share, per generator
  M'   sent-asserted vs GC-overlap / FC-equality / FC-claim-eq
       (stale rate and regeneration share), per generator

All contrasts are paired page-cluster bootstraps (10,000 resamples,
seed 42).  Outputs pagecluster_bootstrap.json.
"""
import json
import math
import os
import random
import re

HERE = os.path.dirname(os.path.abspath(__file__))
TICKS = [f"2026-07-{d:02d}" for d in range(1, 22)]
BOOT = 10000
SEED = 42


def norm(s):
    return re.sub(r"[,\s]+", "", s.lower())


def value_in_text(value, ntext):
    dv = re.findall(r"\d+(?:\.\d+)?", value.replace(",", ""))
    if dv:
        return all(t in ntext for t in dv)
    return norm(value) in ntext


def build_cases(gen):
    """Merged case builder (stage 4 + stage 7 fields), with page id."""
    panel = json.load(open(os.path.join(HERE, "panel21.json")))["panel"]
    answers = json.load(open(os.path.join(HERE, f"answers_{gen}.json")))["answers"]
    series = json.load(open(os.path.join(HERE, "series21.json")))["pages"]
    cases = []
    for i, a in enumerate(panel):
        key = f"{a['page']}#{i}"
        if not answers.get(key, {}).get("accepted"):
            continue
        page = series[a["page"]]
        sers = page["series"]
        h0 = sers[TICKS[0]]["hash"]
        ib0 = sers[TICKS[0]]["infobox"]
        for age, day in enumerate(TICKS):
            if age == 0:
                continue
            snap = sers.get(day)
            if snap is None:
                continue
            nlead = norm(snap["lead"])
            valids = [c["valid"].get(day) for c in a["claims"]]
            ibk = snap["infobox"]
            cases.append({
                "aid": i, "page": a["page"], "day": day, "age": age,
                "stale": any(v is False for v in valids),
                "n_valid": sum(1 for v in valids if v is True),
                "alive_assert": all(c["alive"].get(day) for c in a["claims"]),
                "alive_ctx": a["context_alive"].get(day, True),
                "hash_same": snap["hash"] == h0,
                "infobox_changed": any(ib0.get(k) != ibk.get(k)
                                       for k in set(ib0) | set(ibk)),
                "vals_in_lead": all(value_in_text(c["value_t0"], nlead)
                                    for c in a["claims"]),
            })
    return cases


ARMS = {
    "sent-asserted": lambda c: c["alive_assert"],
    "TTL-1d":        lambda c: c["age"] <= 1,
    "TTL-14d":       lambda c: c["age"] <= 14,
    "GC-overlap":    lambda c: (c["hash_same"] and c["n_valid"] >= 2
                                and c["vals_in_lead"]),
    "FC-equality":   lambda c: c["hash_same"] and c["n_valid"] == 3,
    "FC-claim-eq":   lambda c: c["n_valid"] == 3,
}


def page_level(cases, gate):
    """per-page aggregates for cluster bootstrap"""
    per = {}
    for c in cases:
        d = per.setdefault(c["page"], {"n": 0, "served": 0, "sd": 0,
                                       "rej": 0})
        d["n"] += 1
        if gate(c):
            d["served"] += 1
            d["sd"] += c["stale"]
        else:
            d["rej"] += 1
    return per


def boot_paired(perA, perB, stat, rng):
    ids = sorted(set(perA) & set(perB))
    diffs = []
    for _ in range(BOOT):
        samp = [ids[rng.randrange(len(ids))] for _ in ids]
        diffs.append(stat({i: perA[i] for i in samp})
                     - stat({i: perB[i] for i in samp}))
    diffs.sort()
    return (diffs[int(0.025 * BOOT)], diffs[int(0.975 * BOOT) - 1],
            len(ids))


def stat_stale(per):
    s = sum(d["sd"] for d in per.values())
    n = sum(d["served"] for d in per.values())
    return s / n if n else 0.0


def stat_regen(per):
    r = sum(d["rej"] for d in per.values())
    n = sum(d["n"] for d in per.values())
    return r / n if n else 0.0


def main():
    rng = random.Random(SEED)
    pers = {}
    meta = {}
    for gen in ("kimi", "gpt"):
        cases = build_cases(gen)
        meta[gen] = {"n_cases": len(cases),
                     "n_cached_answers": len({c["aid"] for c in cases}),
                     "n_pages": len({c["page"] for c in cases})}
        for name, gate in ARMS.items():
            pers[(gen, name)] = page_level(cases, gate)

    out = {"unit": "page", "boot": BOOT, "seed": SEED, "meta": meta,
           "contrasts": {}}
    print(f"pages: kimi={meta['kimi']['n_pages']} gpt={meta['gpt']['n_pages']}")

    # B1': generator contrast, paired on pages cached by both
    lo, hi, np_ = boot_paired(pers[("kimi", "sent-asserted")],
                              pers[("gpt", "sent-asserted")],
                              stat_stale, rng)
    out["contrasts"]["B1p_kimi_minus_gpt_stalerate"] = {
        "ci": [lo, hi], "n_common_pages": np_}
    print(f"B1' asserted stale Kimi-GPT (pages={np_}): "
          f"{100*lo:+.2f}..{100*hi:+.2f} pp")

    for gen in ("kimi", "gpt"):
        lo, hi, np_ = boot_paired(pers[(gen, "sent-asserted")],
                                  pers[(gen, "TTL-14d")], stat_stale, rng)
        out["contrasts"][f"B2p_{gen}_asserted_minus_ttl14_stale"] = {
            "ci": [lo, hi], "n_pages": np_}
        print(f"B2' {gen} asserted-TTL14d stale: "
              f"{100*lo:+.2f}..{100*hi:+.2f} pp")
        lo, hi, np_ = boot_paired(pers[(gen, "sent-asserted")],
                                  pers[(gen, "TTL-1d")], stat_regen, rng)
        out["contrasts"][f"B3p_{gen}_asserted_minus_ttl1_regen"] = {
            "ci": [lo, hi], "n_pages": np_}
        print(f"B3' {gen} asserted-TTL1d regen: "
              f"{100*lo:+.2f}..{100*hi:+.2f} pp")
        for base in ("GC-overlap", "FC-equality", "FC-claim-eq"):
            lo, hi, np_ = boot_paired(pers[(gen, "sent-asserted")],
                                      pers[(gen, base)], stat_stale, rng)
            out["contrasts"][f"Mp_{gen}_asserted_minus_{base}_stale"] = {
                "ci": [lo, hi], "n_pages": np_}
            lo, hi, np_ = boot_paired(pers[(gen, "sent-asserted")],
                                      pers[(gen, base)], stat_regen, rng)
            out["contrasts"][f"Mp_{gen}_asserted_minus_{base}_regen"] = {
                "ci": [lo, hi], "n_pages": np_}
            print(f"M' {gen} asserted-{base}: "
                  f"stale {100*out['contrasts'][f'Mp_{gen}_asserted_minus_{base}_stale']['ci'][0]:+.2f}.."
                  f"{100*out['contrasts'][f'Mp_{gen}_asserted_minus_{base}_stale']['ci'][1]:+.2f} pp, "
                  f"regen {100*lo:+.2f}..{100*hi:+.2f} pp")

    json.dump(out, open(os.path.join(HERE, "pagecluster_bootstrap.json"),
                        "w"), indent=1)
    print("\nsaved pagecluster_bootstrap.json")


if __name__ == "__main__":
    main()

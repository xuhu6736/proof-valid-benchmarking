#!/usr/bin/env python3
"""Stage 7: mechanism-level baseline arms on the dual-generator daily
panel (GroundedCache-style overlap, FinCacheServe-style equality,
claim-level equality ablation), using the mapping conventions of
Supplementary Section S5 (s48_mechanism_arms.py) ported to this
scenario's signals:

  GC-overlap   serve iff page hash unchanged
               AND >= ceil(3*0.5)=2 of the 3 claims remain valid
               AND every asserted t0 value still occurs in the
                   current lead (normalized digit/substring match)
  FC-equality  serve iff page hash unchanged AND all 3 claims valid
  FC-claim-eq  serve iff all 3 claims valid (no page-version guard)

All arms share the same maintained state (tick snapshots), the same
dual caches (accepted answers), and the same regeneration accounting
as stage 4. FreshCache's hash-based change detection corresponds to
the page-hash reference already reported; no FreshCache risk
predictor is evaluated (see S5 discussion).
Outputs mechanism_arms21.json and paired bootstrap contrasts.
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


def wilson(k, n, z=1.96):
    if n == 0:
        return (0.0, 0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (p, max(0.0, (c - h) / d), min(1.0, (c + h) / d))


def pct(t):
    return f"{100*t[0]:.2f} [{100*t[1]:.2f},{100*t[2]:.2f}]"


def build_cases(gen):
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
        for day in TICKS[1:]:
            snap = sers.get(day)
            if snap is None:
                continue
            nlead = norm(snap["lead"])
            valids = [c["valid"].get(day) for c in a["claims"]]
            n_valid = sum(1 for v in valids if v is True)
            stale = any(v is False for v in valids)
            cases.append({
                "aid": i, "day": day, "stale": stale,
                "n_valid": n_valid,
                "alive_assert": all(c["alive"].get(day) for c in a["claims"]),
                "alive_ctx": a["context_alive"].get(day, True),
                "hash_same": snap["hash"] == h0,
                "vals_in_lead": all(value_in_text(c["value_t0"], nlead)
                                    for c in a["claims"]),
            })
    return cases


ARMS = {
    "sent-asserted": lambda c: c["alive_assert"],
    "GC-overlap":    lambda c: (c["hash_same"] and c["n_valid"] >= 2
                                and c["vals_in_lead"]),
    "FC-equality":   lambda c: c["hash_same"] and c["n_valid"] == 3,
    "FC-claim-eq":   lambda c: c["n_valid"] == 3,
    "page-hash":     lambda c: c["hash_same"],
    "TTL-1d":        lambda c: c["age"] <= 1 if "age" in c else False,
}


def eval_arm(cases, gate):
    served = rej = sd = inter = fr = 0
    n_stale = 0
    for c in cases:
        n_stale += c["stale"]
        if gate(c):
            served += 1
            sd += c["stale"]
        else:
            rej += 1
            inter += c["stale"]
            fr += (not c["stale"])
    return {"n": len(cases), "served": served, "rejected": rej,
            "n_stale": n_stale, "stale_deliv": sd,
            "stale_rate": wilson(sd, served),
            "intercept": inter, "intercept_rate": wilson(inter, n_stale),
            "false_rej": fr,
            "fr_valid": wilson(fr, len(cases) - n_stale),
            "regen_share": rej / len(cases) if cases else 0.0}


def answer_level(cases, gate):
    per = {}
    for c in cases:
        d = per.setdefault(c["aid"], {"n": 0, "served": 0, "sd": 0, "rej": 0})
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
    return diffs[int(0.025 * BOOT)], diffs[int(0.975 * BOOT) - 1]


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
    results, pers = {}, {}
    for gen in ("kimi", "gpt"):
        cases = build_cases(gen)
        results[gen] = {"n_cases": len(cases), "arms": {}}
        for name in ("sent-asserted", "GC-overlap", "FC-equality",
                     "FC-claim-eq", "page-hash"):
            gate = ARMS[name]
            results[gen]["arms"][name] = eval_arm(cases, gate)
            pers[(gen, name)] = answer_level(cases, gate)

    for gen in ("kimi", "gpt"):
        print(f"\n=== {gen.upper()} cases={results[gen]['n_cases']}")
        print(f"{'arm':<15}{'stale-deliv %':>22}{'intercept %':>20}"
              f"{'regen %':>9}{'FR|valid %':>12}")
        for name in ("sent-asserted", "GC-overlap", "FC-equality",
                     "FC-claim-eq", "page-hash"):
            a = results[gen]["arms"][name]
            print(f"{name:<15}{pct(a['stale_rate']):>22}"
                  f"{pct(a['intercept_rate']):>20}"
                  f"{100*a['regen_share']:>8.1f}{pct(a['fr_valid']):>12}")

    boot = {}
    for gen in ("kimi", "gpt"):
        for base in ("GC-overlap", "FC-equality", "FC-claim-eq"):
            lo, hi = boot_paired(pers[(gen, "sent-asserted")],
                                 pers[(gen, base)], stat_stale, rng)
            boot[f"{gen}_asserted_minus_{base}_stale"] = [lo, hi]
            print(f"{gen}: asserted-{base} stale {100*lo:+.2f}..{100*hi:+.2f} pp")
            lo, hi = boot_paired(pers[(gen, "sent-asserted")],
                                 pers[(gen, base)], stat_regen, rng)
            boot[f"{gen}_asserted_minus_{base}_regen"] = [lo, hi]
            print(f"{gen}: asserted-{base} regen {100*lo:+.2f}..{100*hi:+.2f} pp")

    results["bootstrap"] = boot
    json.dump(results, open(os.path.join(HERE, "mechanism_arms21.json"), "w"),
              indent=1)
    print("\nsaved mechanism_arms21.json")


if __name__ == "__main__":
    main()

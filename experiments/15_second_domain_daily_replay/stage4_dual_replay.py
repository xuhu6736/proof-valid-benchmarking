#!/usr/bin/env python3
"""Stage 4 (A-3): dual-generator replay with real cached answers.

Cache per generator = accepted t0 answers only. Gate arms decide
serve/reject per answer-tick case; rejection triggers regeneration
(assumed fresh and correct; a real-regeneration sample is judged in
stage 5 / A-2). Truth is deterministic value supersession.

Outputs dual_replay_results.json and prints a LaTeX-ready dual table
(arm x generator) plus paired-bootstrap contrasts:
  B1  sent-asserted stale-delivery: Kimi vs GPT (generator sensitivity)
  B2  sent-asserted vs TTL-14d stale delivery (near-equal budget)
  B3  sent-asserted vs TTL-1d regeneration share (near-equal quality)
Cluster bootstrap over answers (10,000 resamples, seed 42).
"""
import json
import math
import os
import random

HERE = os.path.dirname(os.path.abspath(__file__))
TICKS = [f"2026-07-{d:02d}" for d in range(1, 22)]
BOOT = 10000
SEED = 42


def wilson(k, n, z=1.96):
    if n == 0:
        return (0.0, 0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (p, max(0.0, (c - h) / d), min(1.0, (c + h) / d))


def pct(trip):
    return f"{100*trip[0]:.2f} [{100*trip[1]:.2f},{100*trip[2]:.2f}]"


def build_cases(gen):
    panel = json.load(open(os.path.join(HERE, "panel21.json")))["panel"]
    answers = json.load(open(os.path.join(HERE, f"answers_{gen}.json")))["answers"]
    series = json.load(open(os.path.join(HERE, "series21.json")))["pages"]
    cases = []
    for i, a in enumerate(panel):
        key = f"{a['page']}#{i}"
        if not answers.get(key, {}).get("accepted"):
            continue                      # not in this generator's cache
        page = series[a["page"]]
        sers = page["series"]
        h0 = sers[TICKS[0]]["hash"]
        ib0 = sers[TICKS[0]]["infobox"]
        for day in TICKS[1:]:
            snap = sers.get(day)
            if snap is None:
                continue
            stale = any(c["valid"].get(day) is False for c in a["claims"])
            alive_assert = all(c["alive"].get(day) for c in a["claims"])
            alive_ctx = a["context_alive"].get(day, True)
            ibk = snap["infobox"]
            cases.append({"aid": i, "day": day, "stale": stale,
                          "alive_assert": alive_assert,
                          "alive_ctx": alive_ctx,
                          "page_changed": snap["hash"] != h0,
                          "infobox_changed":
                              any(ib0.get(k) != ibk.get(k)
                                  for k in set(ib0) | set(ibk))})
    return cases


AGE = {d: i for i, d in enumerate(TICKS)}
ARMS = {
    "similarity-only": lambda c: True,
    "sent-asserted":   lambda c: c["alive_assert"],
    "sent-strict":     lambda c: c["alive_assert"] and c["alive_ctx"],
    "infobox-diff":    lambda c: not c["infobox_changed"],
    "page-hash":       lambda c: not c["page_changed"],
    "TTL-1d":          lambda c: AGE[c["day"]] <= 1,
    "TTL-3d":          lambda c: AGE[c["day"]] <= 3,
    "TTL-7d":          lambda c: AGE[c["day"]] <= 7,
    "TTL-14d":         lambda c: AGE[c["day"]] <= 14,
}


def eval_arm(cases, gate):
    served = rej = sd = inter = fr = 0
    n_stale = 0
    for c in cases:
        if c["stale"]:
            n_stale += 1
        if gate(c):
            served += 1
            if c["stale"]:
                sd += 1
        else:
            rej += 1
            if c["stale"]:
                inter += 1
            else:
                fr += 1
    n_valid = len(cases) - n_stale
    base_avoided = None
    return {"n": len(cases), "served": served, "rejected": rej,
            "n_stale": n_stale, "stale_deliv": sd,
            "stale_rate": wilson(sd, served),
            "intercept": inter, "intercept_rate": wilson(inter, n_stale),
            "false_rej": fr, "fr_valid": wilson(fr, n_valid),
            "regen_share": rej / len(cases) if cases else 0.0}


def answer_level(cases, gate):
    """per-answer aggregates for cluster bootstrap"""
    per = {}
    for c in cases:
        d = per.setdefault(c["aid"], {"n": 0, "served": 0, "sd": 0,
                                      "rej": 0, "stale": 0})
        d["n"] += 1
        d["stale"] += c["stale"]
        if gate(c):
            d["served"] += 1
            d["sd"] += c["stale"]
        else:
            d["rej"] += 1
    return per


def boot_contrast(perA, perB, stat, rng, paired=False):
    """CI for stat(A) - stat(B). stat maps per-answer dict -> float.
    paired=True resamples the shared answer ids."""
    if paired:
        ids = sorted(set(perA) & set(perB))
        diffs = []
        for _ in range(BOOT):
            samp = [ids[rng.randrange(len(ids))] for _ in ids]
            a = stat({i: perA[i] for i in samp})
            b = stat({i: perB[i] for i in samp})
            diffs.append(a - b)
    else:
        idsA, idsB = sorted(perA), sorted(perB)
        diffs = []
        for _ in range(BOOT):
            sa = [idsA[rng.randrange(len(idsA))] for _ in idsA]
            sb = [idsB[rng.randrange(len(idsB))] for _ in idsB]
            diffs.append(stat({i: perA[i] for i in sa})
                         - stat({i: perB[i] for i in sb}))
    diffs.sort()
    lo, hi = diffs[int(0.025 * BOOT)], diffs[int(0.975 * BOOT) - 1]
    return lo, hi


def stat_stale_rate(per):
    s = sum(d["sd"] for d in per.values())
    n = sum(d["served"] for d in per.values())
    return s / n if n else 0.0


def stat_regen_share(per):
    r = sum(d["rej"] for d in per.values())
    n = sum(d["n"] for d in per.values())
    return r / n if n else 0.0


def main():
    rng = random.Random(SEED)
    results = {}
    pers = {}
    for gen in ("kimi", "gpt"):
        cases = build_cases(gen)
        results[gen] = {"n_cases": len(cases),
                        "n_cached": len({c["aid"] for c in cases}),
                        "arms": {}}
        for name, gate in ARMS.items():
            results[gen]["arms"][name] = eval_arm(cases, gate)
            pers[(gen, name)] = answer_level(cases, gate)

    for gen in ("kimi", "gpt"):
        r0 = results[gen]
        print(f"\n=== {gen.upper()}  cached={r0['n_cached']}  cases={r0['n_cases']} "
              f"stale={r0['arms']['similarity-only']['n_stale']} "
              f"({100*r0['arms']['similarity-only']['n_stale']/r0['n_cases']:.2f}%)")
        print(f"{'arm':<16}{'stale-deliv %':>22}{'intercept %':>20}"
              f"{'regen %':>9}{'FR|valid %':>12}")
        for name in ARMS:
            a = r0["arms"][name]
            inter = pct(a["intercept_rate"]) if a["n_stale"] else "--"
            print(f"{name:<16}{pct(a['stale_rate']):>22}{inter:>20}"
                  f"{100*a['regen_share']:>8.1f}{pct(a['fr_valid']):>12}")

    # ---------------- paired bootstrap contrasts ----------------
    boot = {}
    a_k, a_g = pers[("kimi", "sent-asserted")], pers[("gpt", "sent-asserted")]
    lo, hi = boot_contrast(a_k, a_g, stat_stale_rate, rng)
    boot["B1_kimi_minus_gpt_stalerate"] = [lo, hi]
    print(f"\nB1 asserted stale-rate Kimi-GPT: {100*lo:+.2f} .. {100*hi:+.2f} pp")

    for gen in ("kimi", "gpt"):
        lo, hi = boot_contrast(pers[(gen, "sent-asserted")],
                               pers[(gen, "TTL-14d")], stat_stale_rate,
                               rng, paired=True)
        boot[f"B2_{gen}_asserted_minus_ttl14"] = [lo, hi]
        print(f"B2 {gen}: asserted-TTL14d stale-rate {100*lo:+.2f} .. {100*hi:+.2f} pp")
        lo, hi = boot_contrast(pers[(gen, "sent-asserted")],
                               pers[(gen, "TTL-1d")], stat_regen_share,
                               rng, paired=True)
        boot[f"B3_{gen}_asserted_minus_ttl1_regen"] = [lo, hi]
        print(f"B3 {gen}: asserted-TTL1d regen-share {100*lo:+.2f} .. {100*hi:+.2f} pp")

    results["bootstrap"] = boot
    json.dump(results, open(os.path.join(HERE, "dual_replay_results.json"), "w"),
              indent=1)
    print("\nsaved dual_replay_results.json")


if __name__ == "__main__":
    main()

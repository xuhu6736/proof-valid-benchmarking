#!/usr/bin/env python3
"""Stage 14: per-part-filtered replay (reviewer round-9 priority-2 request).

The headline replay caches every base answer accepted under SET-level
matching.  Stage 10 re-scored those base answers under per-part
(part-wise) matching and flagged 16/62 Kimi and 17/68 GPT answers whose
answer text does not mention every claim part.  Those flagged answers
are exactly the population for which the sentence contract's precision
had to be *transported* rather than measured.  This script repeats the
routing analysis with the cache restricted to the per-part-retained
answers, so that rejection precision, stale-delivery rates, coverage
and fallback demand are measured directly on the filtered cache instead
of being carried over by a transport assumption.

Outputs perpart_filtered_replay.json:
  per generator and per arm: cases, served coverage, stale-delivery
  rate, rejections, rejection precision (stale rejects / all rejects),
  fallback demand share; plus paired page-cluster bootstrap contrasts
  (asserted vs TTL-1d/TTL-14d/GC-overlap/FC-equality/FC-claim-eq) on
  stale rate and regeneration share, computed exactly as in stage 13.
"""
import json
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


def build_cases(gen, keep):
    """Same case builder as stage 13, but restricted to answers whose
    base answer survives per-part matching (keep = set of aids)."""
    panel = json.load(open(os.path.join(HERE, "panel21.json")))["panel"]
    answers = json.load(open(os.path.join(HERE, f"answers_{gen}.json")))["answers"]
    series = json.load(open(os.path.join(HERE, "series21.json")))["pages"]
    cases = []
    for i, a in enumerate(panel):
        if i not in keep:
            continue
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
    per = {}
    for c in cases:
        d = per.setdefault(c["page"], {"n": 0, "served": 0, "sd": 0,
                                       "rej": 0, "rej_stale": 0})
        d["n"] += 1
        if gate(c):
            d["served"] += 1
            d["sd"] += c["stale"]
        else:
            d["rej"] += 1
            d["rej_stale"] += c["stale"]
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


def arm_summary(per):
    n = sum(d["n"] for d in per.values())
    served = sum(d["served"] for d in per.values())
    sd = sum(d["sd"] for d in per.values())
    rej = sum(d["rej"] for d in per.values())
    rej_stale = sum(d["rej_stale"] for d in per.values())
    return {
        "cases": n,
        "coverage": served / n if n else 0.0,
        "stale_delivery_rate": sd / served if served else 0.0,
        "rejections": rej,
        "rejection_precision": rej_stale / rej if rej else None,
        "fallback_demand": rej / n if n else 0.0,
    }


def main():
    rng = random.Random(SEED)
    pp = json.load(open(os.path.join(HERE, "perpart_rescore.json")))
    base = pp["base_answers"]
    keeps = {}
    for gen in ("kimi", "gpt"):
        keeps[gen] = set()
        for k, v in base.items():
            g, rest = k.split("|", 1)
            if g != gen:
                continue
            aid = int(rest.rsplit("#", 1)[1])
            if v["per_part_accept"]:
                keeps[gen].add(aid)

    out = {"unit": "page", "boot": BOOT, "seed": SEED,
           "filter": "cache restricted to base answers passing "
                     "per-part matching (stage 10 rescore)",
           "arms": {}, "contrasts": {}}
    pers = {}
    for gen in ("kimi", "gpt"):
        cases = build_cases(gen, keeps[gen])
        out.setdefault("meta", {})[gen] = {
            "n_cached_answers": len({c["aid"] for c in cases}),
            "n_cases": len(cases),
            "n_pages": len({c["page"] for c in cases})}
        print(f"{gen}: {out['meta'][gen]['n_cached_answers']} cached answers, "
              f"{len(cases)} cases")
        for name, gate in ARMS.items():
            per = page_level(cases, gate)
            pers[(gen, name)] = per
            s = arm_summary(per)
            out["arms"][f"{gen}|{name}"] = s
            rp = (f"{100*s['rejection_precision']:.1f}%"
                  if s["rejection_precision"] is not None else "n/a")
            print(f"  {name:14s} cov={100*s['coverage']:5.1f}% "
                  f"stale={100*s['stale_delivery_rate']:5.2f}% "
                  f"rej={s['rejections']:4d} prec={rp} "
                  f"fb={100*s['fallback_demand']:5.1f}%")

    for gen in ("kimi", "gpt"):
        for base_name in ("TTL-14d", "GC-overlap", "FC-equality",
                          "FC-claim-eq"):
            lo, hi, np_ = boot_paired(pers[(gen, "sent-asserted")],
                                      pers[(gen, base_name)],
                                      stat_stale, rng)
            out["contrasts"][f"{gen}_asserted_minus_{base_name}_stale"] = {
                "ci": [lo, hi], "n_pages": np_}
            print(f"{gen} asserted-{base_name} stale: "
                  f"{100*lo:+.2f}..{100*hi:+.2f} pp (pages={np_})")
        lo, hi, np_ = boot_paired(pers[(gen, "sent-asserted")],
                                  pers[(gen, "TTL-1d")], stat_regen, rng)
        out["contrasts"][f"{gen}_asserted_minus_TTL-1d_regen"] = {
            "ci": [lo, hi], "n_pages": np_}
        print(f"{gen} asserted-TTL1d regen: "
              f"{100*lo:+.2f}..{100*hi:+.2f} pp (pages={np_})")

    json.dump(out, open(os.path.join(HERE, "perpart_filtered_replay.json"),
                        "w"), indent=1)
    print("\nsaved perpart_filtered_replay.json")


if __name__ == "__main__":
    main()

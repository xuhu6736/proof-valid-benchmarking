#!/usr/bin/env python3
"""Stage 15: zero-denominator audit of the page-cluster bootstraps
(reviewer round-10 request S5).

For every reported paired contrast in stage 13 (full cache) and
stage 14 (per-part-filtered cache), count bootstrap draws in which the
served denominator of a stale-rate statistic is zero on either arm.
stat_stale falls back to 0.0 when the denominator is zero; if no draw
has a zero denominator, that convention never fires and the reported
intervals are unaffected.

Output: zero_denominator_check.json
"""
import json
import os
import random

import stage13_pagecluster_bootstrap as s13
import stage14_perpart_filtered_replay as s14

BOOT = 10000
SEED = 42


def served(per, ids):
    return sum(per[i]["served"] for i in ids)


def audit(perA, perB, rng):
    ids = sorted(set(perA) & set(perB))
    zeroA = zeroB = 0
    for _ in range(BOOT):
        samp = [ids[rng.randrange(len(ids))] for _ in ids]
        if served(perA, samp) == 0:
            zeroA += 1
        if served(perB, samp) == 0:
            zeroB += 1
    return {"n_pages": len(ids), "zero_den_A": zeroA, "zero_den_B": zeroB}


def main():
    rng = random.Random(SEED)
    out = {"boot": BOOT, "seed": SEED, "full_cache": {}, "filtered": {}}

    # full-cache (stage 13) contrasts
    pers = {}
    for gen in ("kimi", "gpt"):
        cases = s13.build_cases(gen)
        for name, gate in s13.ARMS.items():
            pers[(gen, name)] = s13.page_level(cases, gate)
    for gen in ("kimi", "gpt"):
        for arm in ("TTL-14d", "TTL-1d", "GC-overlap", "FC-equality",
                    "FC-claim-eq"):
            key = f"{gen}|sent-asserted vs {arm}"
            out["full_cache"][key] = audit(pers[(gen, "sent-asserted")],
                                           pers[(gen, arm)], rng)
    out["full_cache"]["kimi vs gpt|sent-asserted"] = audit(
        pers[("kimi", "sent-asserted")], pers[("gpt", "sent-asserted")],
        rng)

    # filtered-cache (stage 14) contrasts
    pp = json.load(open(os.path.join(s14.HERE, "perpart_rescore.json")))
    keeps = {"kimi": set(), "gpt": set()}
    for k, v in pp["base_answers"].items():
        g, rest = k.split("|", 1)
        if v["per_part_accept"]:
            keeps[g].add(int(rest.rsplit("#", 1)[1]))
    fpers = {}
    for gen in ("kimi", "gpt"):
        cases = s14.build_cases(gen, keeps[gen])
        for name, gate in s14.ARMS.items():
            fpers[(gen, name)] = s14.page_level(cases, gate)
    for gen in ("kimi", "gpt"):
        for arm in ("TTL-14d", "TTL-1d", "GC-overlap", "FC-equality",
                    "FC-claim-eq"):
            key = f"{gen}|sent-asserted vs {arm}"
            out["filtered"][key] = audit(fpers[(gen, "sent-asserted")],
                                         fpers[(gen, arm)], rng)

    any_zero = any(v["zero_den_A"] or v["zero_den_B"]
                   for grp in (out["full_cache"], out["filtered"])
                   for v in grp.values())
    out["any_zero_denominator_draw"] = any_zero
    json.dump(out, open(os.path.join(s14.HERE,
                                     "zero_denominator_check.json"),
                        "w"), indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()

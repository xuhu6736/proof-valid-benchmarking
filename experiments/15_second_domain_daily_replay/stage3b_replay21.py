#!/usr/bin/env python3
"""Stage 3: deterministic replay of gate arms over the 7-day panel.

Arms:
  similarity-only      serve everything (no gate)
  sentence-asserted    serve iff all 3 asserted support sentences survive
  sentence-claims      asserted + context-claim sentences (here = asserted,
                       claims granularity recorded for completeness)
  sentence-strict      all certificate sentences incl. context sentence
  infobox-any-key      reject if any infobox key of the page changed
  page-hash            reject if page revision hash changed
  TTL-1d / 3d / 7d     reject by certificate age

Truth: an answer is stale at tick k iff any asserted infobox value was
superseded (normalized inequality). False rejection = gate fires while all
asserted values still valid. Regeneration fallback = fresh (correct) answer.
Horvitz-Thompson weights applied for the complete stratum.
"""
import json
import math
import os
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
TICKS = [f"2026-07-{d:02d}" for d in range(1, 22)]


def wilson(k, n, z=1.96):
    if n == 0:
        return (0.0, 0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (round(p, 4), round(max(0.0, (c - h) / d), 4), round(min(1.0, (c + h) / d), 4))


def main():
    panel = json.load(open(os.path.join(HERE, "panel21.json")))["panel"]
    series = json.load(open(os.path.join(HERE, "series21.json")))["pages"]

    # precompute per-answer per-tick state
    cases = []
    for a in panel:
        w = a["weight"]
        page = series[a["page"]]
        sers = page["series"]
        for day in TICKS[1:]:                    # serve from t0 cert over ticks 1..7
            snap = sers.get(day)
            if snap is None:
                continue
            stale = any(c["valid"].get(day) is False for c in a["claims"])
            alive_assert = all(c["alive"].get(day) for c in a["claims"])
            alive_ctx = a["context_alive"].get(day, True)
            # page-level signals
            h0 = sers[TICKS[0]]["hash"]
            page_changed = snap["hash"] != h0
            ib0, ibk = sers[TICKS[0]]["infobox"], snap["infobox"]
            infobox_changed = any(ib0.get(k) != ibk.get(k) for k in set(ib0) | set(ibk))
            cases.append({"w": w, "day": day, "stale": stale,
                          "alive_assert": alive_assert, "alive_ctx": alive_ctx,
                          "page_changed": page_changed,
                          "infobox_changed": infobox_changed})

    N = sum(c["w"] for c in cases)
    W_stale = sum(c["w"] for c in cases if c["stale"])
    print(f"answer-tick cases: {len(cases)} (weighted {N:.0f}); "
          f"stale prevalence: {W_stale:.1f}/{N:.0f} = {W_stale/N:.4f}")

    age = {d: i for i, d in enumerate(TICKS)}   # t0 -> day i means age i days

    arms = {
        "similarity-only":  lambda c: True,
        "sent-asserted":    lambda c: c["alive_assert"],
        "sent-strict":      lambda c: c["alive_assert"] and c["alive_ctx"],
        "infobox-any-key":  lambda c: not c["infobox_changed"],
        "page-hash":        lambda c: not c["page_changed"],
        "TTL-1d":           lambda c: age[c["day"]] <= 1,
        "TTL-3d":           lambda c: age[c["day"]] <= 3,
        "TTL-7d":           lambda c: age[c["day"]] <= 7,
        "TTL-14d":          lambda c: age[c["day"]] <= 14,
    }

    rows = {}
    for name, gate in arms.items():
        served = rejected = stale_deliv = false_rej = intercept = 0.0
        for c in cases:
            ok = gate(c)
            if ok:
                served += c["w"]
                if c["stale"]:
                    stale_deliv += c["w"]
            else:
                rejected += c["w"]
                if c["stale"]:
                    intercept += c["w"]
                else:
                    false_rej += c["w"]
        n_s = served or 1
        n_r = rejected or 1
        rows[name] = {
            "served": round(served), "rejected": round(rejected),
            "stale_delivered": round(stale_delivered := stale_deliv, 1),
            "stale_rate": wilson(round(stale_deliv), round(n_s)),
            "intercepted": round(intercept, 1),
            "intercept_rate": wilson(round(intercept), round(W_stale)),
            "false_rejected": round(false_rej, 1),
            "false_rej_rate": wilson(round(false_rej), round(n_r)),
            "false_rej_given_valid": wilson(round(false_rej), round(N - W_stale)),
            "regenerations": round(rejected),
        }

    hdr = f"{'arm':<16} {'served':>7} {'stale-deliv':>11} {'stale%':>14} {'intercept%':>18} {'false-rej%':>16} {'regen':>6}"
    print("\n" + hdr)
    for name, r in rows.items():
        sr, fr, ir = r["stale_rate"], r["false_rej_rate"], r["intercept_rate"]
        print(f"{name:<16} {r['served']:>7} {r['stale_delivered']:>11} "
              f"{sr[0]*100:>5.2f} [{sr[1]*100:.2f},{sr[2]*100:.2f}] "
              f"{ir[0]*100:>7.1f} [{ir[1]*100:.1f},{ir[2]*100:.1f}] "
              f"{fr[0]*100:>6.2f} [{fr[1]*100:.2f},{fr[2]*100:.2f}] "
              f"{r['regenerations']:>6}"
              f"  FR|valid={r['false_rej_given_valid'][0]*100:.2f}%")

    # efficiency: regenerations per avoided stale delivery (vs similarity-only)
    base = rows["similarity-only"]["stale_delivered"]
    print("\nefficiency vs similarity-only:")
    for name, r in rows.items():
        if name == "similarity-only":
            continue
        avoided = base - r["stale_delivered"]
        regen = r["regenerations"]
        ratio = (regen / avoided) if avoided > 0 else float("inf")
        print(f"  {name:<16} avoided={avoided:>5.1f}  regen={regen:>5}  "
              f"regen/avoided={ratio:>8.1f}")

    json.dump({"n_cases": len(cases), "N_weighted": round(N, 1),
               "W_stale": round(W_stale, 1), "arms": rows},
              open(os.path.join(HERE, "replay_results21.json"), "w"), indent=1)
    print("\nsaved replay_results.json")


if __name__ == "__main__":
    main()

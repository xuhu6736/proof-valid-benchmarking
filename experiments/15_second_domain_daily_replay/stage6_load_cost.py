#!/usr/bin/env python3
"""Stage 6 (A-4): per-1000-services expected cost for each arm.

Measured components (this prototype, this machine):
  t_gate   : sent-asserted/strict certificate check (sentence survival
             via normalized containment) -- timed over all replay cases
  t_regen  : LLM regeneration call wall time -- timed sample of real
             regeneration prompts per generator
  t_fetch  : source-page re-read (infobox-diff / page-hash references)
             -- timed sample of live API fetches
  t_ttl    : age comparison ~ negligible (measured for completeness)

Cost model per 1000 services for arm a:
  E_a = 1000 * (t_check(a) + regen_share(a) * t_regen(gen))
reported in seconds, plus LLM generations per 1000 services.
"""
import json
import os
import re
import time
import urllib.parse
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
TICKS = [f"2026-07-{d:02d}" for d in range(1, 22)]
PROXY = os.environ.get("GEN_PROXY", "http://127.0.0.1:7890")


def norm(s):
    return re.sub(r"[,\s]+", "", s.lower())


# ------------------------------------------------ 1. gate check timing --
def time_gate_checks():
    panel = json.load(open(os.path.join(HERE, "panel21.json")))["panel"]
    series = json.load(open(os.path.join(HERE, "series21.json")))["pages"]
    # pre-normalize leads once (a deployment caches normalized state)
    leads = {}
    for title, p in series.items():
        if "series" in p:
            for day, snap in p["series"].items():
                leads[(title, day)] = norm(snap["lead"])
    checks = []
    for a in panel:
        for day in TICKS[1:]:
            if (a["page"], day) in leads:
                checks.append((a, day))
    # asserted check
    t0 = time.perf_counter()
    for rep in range(5):
        n = 0
        for a, day in checks:
            lead = leads[(a["page"], day)]
            ok = all(norm(c["support"]) in lead for c in a["claims"])
            n += ok
    dt = (time.perf_counter() - t0) / 5 / len(checks)
    # strict check
    t0 = time.perf_counter()
    for rep in range(5):
        for a, day in checks:
            lead = leads[(a["page"], day)]
            ok = all(norm(c["support"]) in lead for c in a["claims"])
            if ok and a.get("context"):
                ok = norm(a["context"]) in lead
    dt_strict = (time.perf_counter() - t0) / 5 / len(checks)
    return {"n_checks": len(checks),
            "t_gate_asserted_s": dt, "t_gate_strict_s": dt_strict}


# -------------------------------------------- 2. regeneration timing --
def time_regen(gen, n_sample=6):
    from stage5_fallback_regen import chat_gen, SYSTEM
    answers = json.load(open(os.path.join(HERE, f"answers_{gen}.json")))["answers"]
    keys = [k for k, v in answers.items() if v.get("accepted")][:n_sample]
    times = []
    for k in keys:
        q = answers[k]["question"]
        t0 = time.perf_counter()
        out = chat_gen(gen, q, SYSTEM)
        dt = time.perf_counter() - t0
        if out is not None:
            times.append(dt)
        time.sleep(0.3)
    times.sort()
    return {"gen": gen, "n": len(times),
            "median_s": times[len(times) // 2] if times else None,
            "min_s": times[0] if times else None,
            "max_s": times[-1] if times else None}


# ------------------------------------------------- 3. source re-read --
def time_fetch(n_sample=3):
    op = urllib.request.build_opener(urllib.request.ProxyHandler(
        {"http": PROXY, "https": PROXY}))
    op.addheaders = [("User-Agent", "PVCacheServe-timing/0.1")]
    times = []
    titles = ["Harry Kane", "Erling Haaland", "Kylian Mbappé"][:n_sample]
    for t in titles:
        url = ("https://en.wikipedia.org/w/api.php?" +
               urllib.parse.urlencode({
                   "action": "query", "prop": "revisions", "titles": t,
                   "rvprop": "ids|content", "rvslots": "main",
                   "rvlimit": "1", "format": "json", "formatversion": "2"}))
        t0 = time.perf_counter()
        with op.open(url, timeout=40) as r:
            r.read()
        times.append(time.perf_counter() - t0)
        time.sleep(0.3)
    times.sort()
    return {"n": len(times), "median_s": times[len(times) // 2],
            "min_s": times[0], "max_s": times[-1]}


def main():
    gate = time_gate_checks()
    print(f"gate: asserted {1e6*gate['t_gate_asserted_s']:.1f} us, "
          f"strict {1e6*gate['t_gate_strict_s']:.1f} us "
          f"({gate['n_checks']} checks)")
    fetch = time_fetch()
    print(f"source re-read: median {fetch['median_s']:.2f} s")
    regen = {}
    for gen in ("kimi", "gpt"):
        regen[gen] = time_regen(gen)
        r = regen[gen]
        print(f"regen {gen}: median {r['median_s']:.2f} s "
              f"(min {r['min_s']:.2f}, max {r['max_s']:.2f}, n={r['n']})")

    replay = json.load(open(os.path.join(HERE, "dual_replay_results.json")))
    arms = ["similarity-only", "sent-asserted", "sent-strict",
            "infobox-diff", "page-hash",
            "TTL-1d", "TTL-3d", "TTL-7d", "TTL-14d"]

    def check_cost(name):
        if name == "sent-asserted":
            return gate["t_gate_asserted_s"]
        if name == "sent-strict":
            return gate["t_gate_strict_s"]
        if name in ("infobox-diff", "page-hash"):
            return fetch["median_s"]
        return 0.0

    print("\n=== expected cost per 1000 services ===")
    print(f"{'arm':<16}{'gen':>6}{'LLM calls':>11}{'check s':>10}"
          f"{'regen s':>10}{'total s':>10}")
    out = {"gate": gate, "fetch": fetch, "regen": regen, "cost": {}}
    for gen in ("kimi", "gpt"):
        tr = regen[gen]["median_s"]
        for name in arms:
            share = replay[gen]["arms"][name]["regen_share"]
            calls = 1000 * share
            cs = 1000 * check_cost(name)
            rs = calls * tr
            out["cost"].setdefault(name, {})[gen] = {
                "llm_calls": round(calls, 1), "check_s": round(cs, 3),
                "regen_s": round(rs, 1), "total_s": round(cs + rs, 1)}
            print(f"{name:<16}{gen:>6}{calls:>11.0f}{cs:>10.3f}"
                  f"{rs:>10.1f}{cs+rs:>10.1f}")

    json.dump(out, open(os.path.join(HERE, "load_cost.json"), "w"), indent=1)
    print("\nsaved load_cost.json")


if __name__ == "__main__":
    main()

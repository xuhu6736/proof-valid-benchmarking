#!/usr/bin/env python3
"""Stage 2: build the replay panel from series.json.

Anti-degeneracy structure (mirrors exp-14):
  certificate premise = supporting sentence(s) in the t0 lead containing
                        the asserted value   (gate checks sentence survival)
  endpoint truth      = infobox key-value still valid at the tick
The two naturally diverge (lead lags infobox -> missed stale; decorative
sentence edits with value still correct -> false rejection).

An answer = 3 claims from the same page. Stratified sampling: superseded
stratum (any claim superseded by t7) fully included + random complete
stratum, with Horvitz-Thompson weights recorded (seed 42).
"""
import json
import os
import random
import re
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
TICKS = [f"2026-07-{d:02d}" for d in range(1, 22)]
N_ANSWERS_TARGET = 67          # ~200 claims at 3 claims/answer
SEED = 42

BAD_KEY = re.compile(r"color|image|flag|size|border|alt|caption|logo|"
                     r"map|website|url|link|footnotes?|module|below|header")
BAD_VAL = re.compile(r"\.(png|jpe?g|svg|gif|tif?f)\b", re.I)


def norm(s):
    s = s.replace(",", "").replace("\u2013", "-").replace("\u2014", "-")
    s = re.sub(r"\s+", " ", s).strip().lower()
    return s


def digit_tokens(s):
    return re.findall(r"\d+(?:\.\d+)?", s.replace(",", ""))


def sentences(text):
    parts = re.split(r"(?<=[.!?])\s+", text)
    return [p.strip() for p in parts if len(p.strip()) > 15]


def value_in_sentence(value, sent):
    nv, ns = norm(value), norm(sent)
    if not nv:
        return False
    dv = digit_tokens(value)
    if dv:                                   # numeric-anchored claim
        ds = set(digit_tokens(sent))
        if not all(t in ds for t in dv):
            return False
        # also require any alphabetic tail of the value to appear
        tail = re.sub(r"[\d,.\s$€£%]", "", value).strip()
        if tail and norm(tail) not in ns:
            return False
        return True
    return nv in ns                          # textual value


def find_support(value, sents):
    for s in sents:
        if value_in_sentence(value, s):
            return s
    return None


def claim_ok(key, value):
    if BAD_KEY.search(key) or BAD_VAL.search(value):
        return False
    if not value or len(value) > 120 or len(value) < 2:
        return False
    if not digit_tokens(value) and len(value) < 4:
        return False
    return True


def build_page_claims(page):
    """Return list of claims with t0 support sentences, plus context sent."""
    s0 = page["series"].get(TICKS[0])
    if not s0:
        return [], None, Counter()
    sents = sentences(s0["lead"])
    claims, dropped = [], Counter()
    for key, value in sorted(s0["infobox"].items()):
        if not claim_ok(key, value):
            dropped["filtered"] += 1
            continue
        sup = find_support(value, sents)
        if sup is None:
            dropped["no_support"] += 1
            continue
        claims.append({"key": key, "value_t0": value, "support": sup})
    # context sentence: a lead sentence containing none of the claim values
    ctx = None
    for s in sents:
        if not any(value_in_sentence(c["value_t0"], s) for c in claims):
            ctx = s
            break
    return claims, ctx, dropped


def per_tick_validity(page, key, v0):
    """value still valid at each tick (normalized equality)."""
    out = {}
    for day in TICKS:
        snap = page["series"].get(day)
        if snap is None:
            out[day] = None
            continue
        cur = snap["infobox"].get(key)
        out[day] = (cur is not None and norm(cur) == norm(v0))
    return out


def per_tick_survival(page, sentence):
    ns = norm(sentence)
    out = {}
    for day in TICKS:
        snap = page["series"].get(day)
        out[day] = None if snap is None else (ns in norm(snap["lead"]))
    return out


def main():
    series = json.load(open(os.path.join(HERE, "series21.json")))["pages"]
    rng = random.Random(SEED)

    pool_claims = []            # all eligible claims (pool level stats)
    page_bundles = []           # per-page claim groups for answer building
    drop = Counter()
    for title, page in series.items():
        if "error" in page or "series" not in page:
            continue
        claims, ctx, dropped = build_page_claims(page)
        drop.update(dropped)
        if not claims:
            drop["page_no_claim"] += 1
            continue
        page_bundles.append({"title": title, "group": page["group"],
                             "claims": claims, "context": ctx})
        pool_claims += [(title, page, c) for c in claims]

    print(f"pages with claims: {len(page_bundles)}")
    print(f"pool claims (support sentence found): {len(pool_claims)}  dropped={dict(drop)}")

    # ---- pool-level supersession curve (natural prevalence) ----
    def curve(claims_list, day):
        n = k = 0
        for title, page, c in claims_list:
            v = per_tick_validity(page, c["key"], c["value_t0"]).get(day)
            if v is not None:
                n += 1
                k += (not v)
        return n, k

    pool_stats = {}
    for day in (TICKS[1], TICKS[3], TICKS[7], TICKS[14], TICKS[20]):  # 1/3/7/14/20d
        n, k = curve(pool_claims, day)
        pool_stats[day] = {"n": n, "superseded": k,
                           "rate": round(k / n, 4) if n else None}
        print(f"pool supersession @{day}: {k}/{n} = {pool_stats[day]['rate']}")

    # ---- answers: 3 claims from same page ----
    answers = []
    for b in page_bundles:
        page = series[b["title"]]
        cs = b["claims"]
        if len(cs) < 3:
            drop["page_lt3"] += 1
            continue
        for i in range(0, len(cs) - 2, 3):
            trio = cs[i:i + 3]
            ans = {"page": b["title"], "group": b["group"], "claims": [],
                   "context": b["context"], "certificate": []}
            for c in trio:
                validity = per_tick_validity(page, c["key"], c["value_t0"])
                survival = per_tick_survival(page, c["support"])
                ans["claims"].append({"key": c["key"], "value_t0": c["value_t0"],
                                      "support": c["support"],
                                      "valid": validity, "alive": survival})
                ans["certificate"].append(c["support"])
            if b["context"]:
                ans["certificate"].append(b["context"])
                ans["context_alive"] = per_tick_survival(page, b["context"])
            else:
                ans["context_alive"] = {d: True for d in TICKS}
            answers.append(ans)

    # ---- stratification ----
    def ans_superseded(a):
        return any(a["claims"][j]["valid"].get(TICKS[-1]) is False
                   for j in range(3))

    sup = [a for a in answers if ans_superseded(a)]
    comp = [a for a in answers if not ans_superseded(a)]
    print(f"answers total={len(answers)}  superseded-stratum={len(sup)}  complete-stratum={len(comp)}")

    n_sup = len(sup)
    n_comp = max(0, N_ANSWERS_TARGET - n_sup)
    if n_comp > len(comp):
        n_comp = len(comp)
    sel_comp = rng.sample(comp, n_comp)
    for a in sup:
        a["stratum"], a["weight"] = "superseded", 1.0
    p_comp = n_comp / len(comp) if comp else 1.0
    for a in sel_comp:
        a["stratum"], a["weight"] = "complete", round(1.0 / p_comp, 4)
    panel = sup + sel_comp
    n_claims = sum(3 for _ in panel)
    print(f"panel: {len(panel)} answers / {3*len(panel)} claims "
          f"(sup={n_sup}, comp={n_comp}, HT weight={round(1.0/p_comp,3) if comp else 1})")

    json.dump({"seed": SEED, "ticks": TICKS, "panel": panel},
              open(os.path.join(HERE, "panel21.json"), "w"))
    json.dump({"pool_claims": len(pool_claims), "pool_stats": pool_stats,
               "answers_total": len(answers), "sup_stratum": n_sup,
               "comp_stratum": len(comp), "comp_sampled": n_comp,
               "p_comp": p_comp, "drops": dict(drop)},
              open(os.path.join(HERE, "pool_stats21.json"), "w"), indent=1)
    print("saved panel.json, pool_stats.json")


if __name__ == "__main__":
    main()

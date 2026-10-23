#!/usr/bin/env python3
"""Stage 0 (A-1): template one multi-part question per panel answer and
generate the t0 cached answer with two generators (Kimi / GPT).

Each answer = 3 same-page claims. The question asks one sub-question per
claim (key-stem templates with a generic fallback). The model receives
the t0 certificate evidence (3 support sentences + 1 context sentence)
and must answer using only that evidence. Acceptance = lexical
value-mention check on all 3 asserted values (normalized).

Usage:
  python stage0_generate_answers.py kimi     # uses KIMI_API_KEY/KIMI_BASE_URL
  python stage0_generate_answers.py gpt      # uses POE_API_KEY
Checkpointed per answer; re-running skips completed entries.
"""
import json
import os
import re
import sys
import time
import urllib.request

PROXY = os.environ.get("GEN_PROXY", "http://127.0.0.1:7890")
OPENER = urllib.request.build_opener(
    urllib.request.ProxyHandler({"http": PROXY, "https": PROXY}))

HERE = os.path.dirname(os.path.abspath(__file__))
T0_HUMAN = "1 July 2026"

GEN = sys.argv[1] if len(sys.argv) > 1 else "kimi"
if GEN == "kimi":
    BASE = os.environ["KIMI_BASE_URL"].rstrip("/")
    KEY = os.environ["KIMI_API_KEY"]
    MODEL = "kimi-k2-turbo-preview"
    STYLE = "anthropic"
else:
    BASE = os.environ.get("POE_BASE_URL", "https://api.poe.com/v1").rstrip("/")
    KEY = os.environ["POE_API_KEY"]
    MODEL = os.environ.get("GPT_MODEL", "gpt-5-mini")
    STYLE = "openai"

OUT = os.path.join(HERE, f"answers_{GEN}.json")


def chat(prompt, system, max_tokens=200, tries=5):
    if STYLE == "anthropic":
        payload = {"model": MODEL, "max_tokens": max_tokens,
                   "thinking": {"type": "disabled"}, "system": system,
                   "messages": [{"role": "user", "content": prompt}]}
        headers = {"x-api-key": KEY, "anthropic-version": "2023-06-01",
                   "content-type": "application/json"}
        url = BASE + "/messages"
    else:
        payload = {"model": MODEL, "max_tokens": max_tokens,
                   "messages": [{"role": "system", "content": system},
                                {"role": "user", "content": prompt}]}
        headers = {"Authorization": "Bearer " + KEY,
                   "content-type": "application/json"}
        url = BASE + "/chat/completions"
    for a in range(tries):
        try:
            req = urllib.request.Request(url, data=json.dumps(payload).encode(),
                                         headers=headers, method="POST")
            with OPENER.open(req, timeout=90) as r:
                d = json.loads(r.read())
            if STYLE == "anthropic":
                return "".join(b.get("text", "") for b in d.get("content", [])
                               if b.get("type") == "text").strip()
            return d["choices"][0]["message"]["content"].strip()
        except Exception:
            time.sleep(min(45, 3 + a * 5))
    return None


# ------------------------------------------------------------ questions --
def stem_of(key):
    return re.sub(r"\d+$", "", key).strip()


def human(stem):
    return stem.replace("_", " ")


def sub_question(page, key, value):
    st, hs = stem_of(key), human(stem_of(key))
    if st == "full_name":
        return f"What is {page}'s full name?"
    if st == "position":
        return f"What position does {page} play?"
    if st == "currentclub":
        return f"Which club does {page} currently play for?"
    if st in ("clubs", "youthclubs"):
        n = key[len(st):]
        kind = "senior-career" if st == "clubs" else "youth-career"
        return (f"Which club is listed as {kind} entry {n} "
                f"for {page}?")
    if st == "nationalgoals":
        return f"How many goals has {page} scored for their national team?"
    if st == "nationalcaps":
        return f"How many appearances (caps) does {page} have for their national team?"
    if st == "nationalyears":
        return f"In which year did {page} start playing for this national team?"
    if st == "coach":
        return f"Who is the head coach of {page}?"
    if st == "captain":
        return f"Who is the captain of {page}?"
    if st == "association":
        return f"Which association governs {page}?"
    if st == "home stadium":
        return f"What is the home stadium of {page}?"
    if st == "clubnumber":
        return f"What is {page}'s squad number?"
    if st in ("total storms", "total hurricanes", "total depressions",
              "total disturbances", "total intense", "total super"):
        return f"How many {hs} were there in the {page} as of {T0_HUMAN}?"
    if st == "fatalities":
        return f"How many fatalities were recorded in the {page}?"
    if st == "damages":
        return f"What were the total damages of the {page} (in USD)?"
    if st == "strongest storm name":
        return f"What was the strongest storm of the {page}?"
    if st == "ace":
        return f"What was the accumulated cyclone energy (ACE) of the {page}?"
    return f"What is the {hs} of {page}, as listed in its Wikipedia infobox?"


def build_prompt(ans):
    page = ans["page"]
    qs = [sub_question(page, c["key"], c["value_t0"]) for c in ans["claims"]]
    evidence = list(dict.fromkeys(c["support"] for c in ans["claims"]))
    if ans.get("context") and ans["context"] not in evidence:
        evidence.append(ans["context"])
    q_block = "\n".join(f"({i+1}) {q}" for i, q in enumerate(qs))
    e_block = "\n".join(f"- {e}" for e in evidence)
    return (f"Answer the following three questions about {page} as of "
            f"{T0_HUMAN}, using only the evidence below.\n"
            f"{q_block}\n\n"
            f"Evidence from the English Wikipedia article \"{page}\" "
            f"({T0_HUMAN}):\n{e_block}\n\n"
            f"Answer all three parts concisely, numbered (1) (2) (3).")


SYSTEM = ("You are a precise assistant. Use only the provided evidence. "
          "State the specific values asked for. Keep the whole answer "
          "under 90 words.")


# ----------------------------------------------------------- acceptance --
def norm(s):
    return re.sub(r"[,\s]+", "", s.lower())


def accepted(answer, claims):
    na = norm(answer)
    missing = []
    for c in claims:
        v = c["value_t0"]
        dv = re.findall(r"\d+(?:\.\d+)?", v.replace(",", ""))
        if dv:
            if not all(t in na for t in dv):
                missing.append(v)
        elif norm(v) not in na:
            missing.append(v)
    return (not missing), missing


def main():
    panel = json.load(open(os.path.join(HERE, "panel21.json")))["panel"]
    data = {"generator": MODEL, "style": STYLE, "answers": {}}
    if os.path.exists(OUT):
        data = json.load(open(OUT))
    n_done = n_acc = 0
    for i, ans in enumerate(panel):
        key = f"{ans['page']}#{i}"
        if key in data["answers"]:
            n_done += 1
            n_acc += data["answers"][key]["accepted"]
            continue
        prompt = build_prompt(ans)
        text = chat(prompt, SYSTEM)
        if text is None:
            print(f"[fail] {key}", flush=True)
            continue
        ok, missing = accepted(text, ans["claims"])
        data["answers"][key] = {
            "page": ans["page"], "group": ans["group"],
            "question": prompt, "answer": text,
            "accepted": ok, "missing": missing}
        json.dump(data, open(OUT, "w"), ensure_ascii=False)
        n_done += 1
        n_acc += ok
        print(f"[{n_done}/{len(panel)}] {key} accepted={ok}", flush=True)
        time.sleep(0.4)
    print(f"\n{GEN} ({MODEL}): {n_acc}/{n_done} accepted "
          f"({100*n_acc/max(1,n_done):.1f}%)")
    print("saved", OUT)


if __name__ == "__main__":
    main()

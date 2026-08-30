
#!/usr/bin/env python3
"""E7: grounded operating-point comparison (replaces proxy Table 1 in main text).

Compares, on the REAL evidence-leaf distributions of HotpotQA (7405 questions)
and MuSiQue (2417 answerable), three cache semantics at erasure rate eps:
  (a) similarity-only transparent cache: serves every query; unproved reuse ->
      premise error = 1 - R(eps)  where R(eps) = E[(1-eps)^{e_q}]
  (b) proof-valid transparent cache: serves iff the full certificate is
      available -> throughput R(eps), premise error 0 by Theorem 1
  (c) proof-valid coded cache with budget B = 0.3*kappa:
      throughput BinomialCDF(B, kappa, eps), error 0
All exponents are MEASURED per question/paragraph -- no synthetic class sizes.
Output: grounded_operating_results.json, tab_grounded_operating.tex
"""
import json
import numpy as np, pandas as pd
from scipy.stats import binom

EQ = None
def hotpot_eq():
    df = pd.read_parquet('data/hotpot_validation.parquet')
    # deduplicate (title, sent_id) evidence pairs, as in the grounded audit
    return np.array([len(set(zip(sf['title'], sf['sent_id'].tolist())))
                     for sf in df['supporting_facts']], dtype=float)

def musique_eq():
    out = []
    with open('data/musique_full_dev.jsonl') as f:
        for line in f:
            r = json.loads(line)
            if not r.get('answerable', True):
                continue
            paras = r.get('paragraphs', [])
            n = sum(1 for p in paras if p.get('is_supporting', False))
            if n > 0:
                out.append(float(n))
    return np.array(out)

def row(e_arr, eps):
    k = e_arr.astype(int)
    Rt = (1.0 - eps) ** k                      # per-instance transparent availability
    B = np.maximum(1, np.round(0.3 * k)).astype(int)
    Rc = binom.cdf(B, k, eps)                  # coded availability at budget B
    return dict(
        sim_only_error=float(np.mean(1.0 - Rt)),
        pv_transparent_throughput=float(np.mean(Rt)),
        pv_coded_throughput=float(np.mean(Rc)),
        mean_kappa=float(np.mean(k)))

results = {}
for name, fn in [('hotpotqa', hotpot_eq), ('musique', musique_eq)]:
    e = fn()
    results[name] = {'n': int(len(e)), 'eps': {}}
    for eps in (0.1, 0.3, 0.5):
        results[name]['eps'][str(eps)] = row(e, eps)
        r = results[name]['eps'][str(eps)]
        print(f"[E7] {name} eps={eps}: sim-only error={r['sim_only_error']:.3f} | "
              f"PV-transparent thr={r['pv_transparent_throughput']:.3f} (err 0) | "
              f"PV-coded thr={r['pv_coded_throughput']:.3f} (err 0)")

json.dump(results, open('grounded_operating_results.json', 'w'), indent=2)

def fmt(x): return f"{x:.3f}"
lines = [r"\begin{tabular}{llcccc}", r"\toprule",
 r"Dataset & $\varepsilon$ & Sim.-only error & PV transp.~thr. & PV coded thr. ($B{=}0.3\kappa$) & mean $\kappa$ \\", r"\midrule"]
for name, label in [('hotpotqa','HotpotQA'), ('musique','MuSiQue')]:
    for eps in (0.1, 0.3, 0.5):
        r = results[name]['eps'][str(eps)]
        lines.append(f"{label if eps==0.1 else ''} & {eps} & {fmt(r['sim_only_error'])} & {fmt(r['pv_transparent_throughput'])} & {fmt(r['pv_coded_throughput'])} & {fmt(r['mean_kappa'])} \\\\")
    lines.append(r"\midrule")
lines = lines[:-1] + [r"\bottomrule", r"\end{tabular}"]
open('tab_grounded_operating.tex','w').write('\n'.join(lines))
print("[E7] grounded_operating_results.json, tab_grounded_operating.tex saved")

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Trace operating points (Appendix A.1--A.3 protocol -> Tables A1 and A3).

Implements, end to end, the proof-valid premise-erasure stress test on the two
released vCache semantic-cache benchmarks:

  * SemBenchmarkSearchQueries (other_60k_with_responses.parquet), first 20,000
    rows -> 6,113 classes; GPTCache-style static threshold at the 0.70 quantile
    (the 0.90 quantile saturates at 1.0 on this trace, see Appendix A.2);
  * SemBenchmarkLmArena (train.parquet), first 20,000 rows -> 3,372 classes;
    static threshold at the 0.90 quantile.

Protocol (Appendix A):
  kappa_C = |C|;  x_C = ceil(kappa_C/2);  n_C = kappa_C - x_C  (exposed leaves)
  s+  = leave-one-out nearest within-class cosine (singletons receive 1)
  s-  = nearest foreign-class centroid cosine
  tau_static = quantile_q({s+});  tau_vc = argmax_tau { coverage : FR({s-}) <= delta }
  Per eps in the grid: one seeded erasure pass over exposed leaves; GPTCache and
  vCache serve on s+ or s- above threshold; the proof-valid gate serves only
  own-class hits whose certificate survives.  Errors: premise error (own-class
  serve with broken certificate) + cross-class false hit, per served prompt.
  Throughput = PV served / GPTCache served at the same eps.

The archived result files trace_endpoints_results.json (LmArena + SQ context)
and trace_endpoints_sq_results.json (supplementary SQ run at q=0.70) were
produced by this protocol; if the trace embeddings are present locally this
script re-executes the full pipeline and checks the archived numbers cell by
cell.  Run:

  HF_ENDPOINT=https://hf-mirror.com python3 run_trace_endpoints_experiment.py

Data sources (Hugging Face):
  datasets/vCache/SemBenchmarkSearchQueries/resolve/main/other_60k_with_responses.parquet
  datasets/vCache/SemBenchmarkLmArena/resolve/main/train.parquet
"""
import argparse, json, os, sys
import numpy as np

SQ_URL = "datasets/vCache/SemBenchmarkSearchQueries/resolve/main/other_60k_with_responses.parquet"
LA_URL = "datasets/vCache/SemBenchmarkLmArena/resolve/main/train.parquet"
MAX_ROWS = 20000
DELTA = 0.02                      # false-hit target (vCache-style calibration)
EPS_GRID = [0.0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5]
SEED = 42
EMB_COL, RESP_COL = "emb_gte", "response_llama_3_8b"

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")


def hf_url(base, path):
    return f"{base.rstrip('/')}/{path}"


def load_trace(base, url, name):
    """Download the trace parquet (cached under data/) and return a DataFrame."""
    import pandas as pd
    dst = os.path.join(DATA, name)
    if not (os.path.exists(dst) and os.path.getsize(dst) > 0):
        import urllib.request
        full = hf_url(base, url)
        print(f"[download] {full}\n        -> {dst}", flush=True)
        urllib.request.urlretrieve(full, dst)
    return pd.read_parquet(dst)


def similarities(df):
    """Leave-one-out within-class positives and foreign-centroid negatives."""
    emb = np.asarray(df[EMB_COL].tolist(), dtype=np.float32)
    emb /= np.linalg.norm(emb, axis=1, keepdims=True) + 1e-12
    labels = df["id_set"].astype(str).values
    classes, inv = np.unique(labels, return_inverse=True)
    n = len(df)
    # class centroids (unit-normalized mean)
    C = len(classes)
    cent = np.zeros((C, emb.shape[1]), dtype=np.float32)
    np.add.at(cent, inv, emb)
    cent /= np.bincount(inv)[:, None].astype(np.float32)
    cent /= np.linalg.norm(cent, axis=1, keepdims=True) + 1e-12
    # positives: leave-one-out nearest within class (blockwise)
    s_pos = np.ones(n, dtype=np.float32)
    for c in range(C):
        idx = np.nonzero(inv == c)[0]
        if len(idx) == 1:
            continue                      # singleton: s+ = 1 by convention
        S = emb[idx] @ emb[idx].T
        np.fill_diagonal(S, -np.inf)
        s_pos[idx] = S.max(1)
    # negatives: nearest foreign centroid
    S = emb @ cent.T
    S[np.arange(n), inv] = -np.inf
    s_neg = S.max(1)
    return s_pos, s_neg, classes, inv


def calibrate(s_pos, s_neg, quantile):
    tau_static = float(np.quantile(s_pos, quantile))
    grid = np.linspace(0.0, 1.0, 2001)
    cov = np.array([(s_pos >= t).mean() for t in grid])
    fr = np.array([(s_neg >= t).mean() for t in grid])
    ok = fr <= DELTA
    tau_vc = float(grid[ok][np.argmax(cov[ok])]) if ok.any() else float(grid[np.argmin(fr)])
    return tau_static, tau_vc


def stress(df, s_pos, s_neg, tau_static, tau_vc, classes, inv):
    """One seeded erasure pass per eps, per Appendix A.3 (Algorithm 1)."""
    rng = np.random.default_rng(SEED)
    kappa = np.bincount(inv).astype(np.int64)
    n_resid = kappa - (kappa + 1) // 2          # n_C = kappa_C - ceil(kappa_C/2)
    out = []
    for eps in EPS_GRID:
        survive = rng.random((len(classes), int(n_resid.max()))) >= eps
        ok_leaf = np.ones(len(classes), dtype=bool)
        for c in range(len(classes)):
            ok_leaf[c] = survive[c, :n_resid[c]].all()
        o = ok_leaf[inv]                         # certificate survival per query
        rec = {"eps": float(eps)}
        for name, tau in (("GPTCache", tau_static), ("vCache", tau_vc)):
            h_pos = s_pos >= tau
            h_neg = (~h_pos) & (s_neg >= tau)
            served = h_pos | h_neg
            n_served = int(served.sum())
            premise_err = int((h_pos & ~o).sum())
            false_hit = int(h_neg.sum())
            rec[name] = {
                "served": n_served,
                "premise_err_rate": premise_err / max(n_served, 1),
                "falsehit_rate": false_hit / max(n_served, 1),
                "total_err_rate": (premise_err + false_hit) / max(n_served, 1),
                "hit_prob_theory": float(np.mean((1 - eps) ** n_resid)),
                "hit_prob_empirical": float(o[h_pos].mean()) if h_pos.any() else 1.0,
            }
        h_pos_s = s_pos >= tau_static
        pv_served = h_pos_s & o
        rec["proofvalid"] = {
            "served": int(pv_served.sum()),
            "premise_err_rate": 0.0,
            "falsehit_rate": 0.0,
            "total_err_rate": 0.0,
            "hit_prob_theory": float(np.mean((1 - eps) ** n_resid)),
            "hit_prob_empirical": float(o[h_pos_s].mean()) if h_pos_s.any() else 1.0,
            "throughput_vs_gptcache": float(pv_served.sum() / max((s_pos >= tau_static).sum(), 1)),
        }
        out.append(rec)
    return out


def check_against_archive(results):
    """Cell-by-cell check of Tables A1/A3 against the archived JSONs."""
    ref_sq = json.load(open(os.path.join(HERE, "trace_endpoints_sq_results.json")))
    ref_la = json.load(open(os.path.join(HERE, "trace_endpoints_results.json")))
    refs = {"SemBenchmarkSearchQueries": ref_sq["datasets"]["SemBenchmarkSearchQueries"],
            "SemBenchmarkLmArena": ref_la["datasets"]["SemBenchmarkLmArena"]}
    n_bad = 0
    for ds_name, res in results.items():
        ref = refs[ds_name]
        for key in ("thr_static", "thr_vcache", "mean_sim_pos", "mean_sim_neg"):
            if abs(res[key] - ref[key]) > 5e-7:
                print(f"  MISMATCH {ds_name}.{key}: {res[key]:.7f} vs archived {ref[key]:.7f}")
                n_bad += 1
        for pe, pr in zip(res["per_eps"], ref["per_eps"]):
            for pol in ("GPTCache", "vCache", "proofvalid"):
                for key in ("served", "total_err_rate", "hit_prob_empirical"):
                    a, b = pe[pol][key], pr[pol][key]
                    tol = 0.5 if key == "served" else 5e-4
                    if abs(a - b) > tol:
                        print(f"  MISMATCH {ds_name} eps={pe['eps']} {pol}.{key}: {a} vs archived {b}")
                        n_bad += 1
    print(f"[check] {'all cells match the archived results' if n_bad == 0 else f'{n_bad} mismatches'}")
    return n_bad == 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hf-endpoint", default="https://huggingface.co")
    args = ap.parse_args()
    os.makedirs(DATA, exist_ok=True)

    results = {}
    for ds_name, url, fname, q in (
        ("SemBenchmarkSearchQueries", SQ_URL, "trace_searchqueries.parquet", 0.70),
        ("SemBenchmarkLmArena", LA_URL, "trace_lmarena.parquet", 0.90),
    ):
        df = load_trace(args.hf_endpoint, url, fname).head(MAX_ROWS)
        s_pos, s_neg, classes, inv = similarities(df)
        tau_static, tau_vc = calibrate(s_pos, s_neg, q)
        per_eps = stress(df, s_pos, s_neg, tau_static, tau_vc, classes, inv)
        results[ds_name] = {
            "rows": int(len(df)), "classes": int(len(classes)),
            "emb_col": EMB_COL, "resp_col": RESP_COL,
            "static_quantile": q, "delta": DELTA, "seed": SEED,
            "thr_static": tau_static, "thr_vcache": tau_vc,
            "mean_sim_pos": float(s_pos.mean()), "mean_sim_neg": float(s_neg.mean()),
            "per_eps": per_eps,
        }
        print(f"[{ds_name}] rows={len(df)} classes={len(classes)} "
              f"tau_static={tau_static:.7f} tau_vc={tau_vc:.7f}")

    out = os.path.join(HERE, "trace_endpoints_rerun_results.json")
    json.dump({"config": {"max_per_dataset": MAX_ROWS, "delta": DELTA,
                          "kappa_rule": "size", "x_frac": 0.5,
                          "eps_list": ",".join(str(e) for e in EPS_GRID)},
               "datasets": results}, open(out, "w"), indent=1)
    print(f"[saved] {out}")
    ok = check_against_archive(results)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()

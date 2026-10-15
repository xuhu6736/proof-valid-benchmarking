#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Premise-erasure trustworthy-caching benchmark on vCache real query logs (v2, fixed).

Fixes over v1 (review findings):
  [F1] vCache is now genuinely modelled: an adaptive threshold calibrated to a target
       error rate delta over CROSS-class nearest neighbours (false hits), distinct from
       the GPTCache static threshold. Previously both shared one threshold and overlapped.
  [F2] Threshold now operates over the FULL similarity distribution (intra-class paraphrase
       reuse AND cross-class false hits), reproducing vCache's real operating point instead
       of only intra-class hits at thr=1.0. We report the full hit/error trade-off.
  [F3] Fig.3 uses the PER-CLASS kappa distribution (not a single k_med); Fig.4 uses each
       class's measured certificate footprint, so both figures carry real-data information.

Honesty of the premise overlay (unchanged, documented):
  The datasets carry (prompt, id_set, embedding, response) but NO explicit premise layer.
  We instantiate it explicitly: each equivalence class (id_set) is a derivation object whose
  answer depends on kappa premises (knowledge units); premises are erased i.i.d. at rate eps
  (premise revocation / deletion). kappa is instantiated from the class's observable
  footprint (--kappa_rule). This is an explicit, reproducible construction.
"""
import argparse, json, os, glob
from math import comb
import numpy as np
import pandas as pd

# ----------------------------- small exact utils -----------------------------

def binom_cdf(n, r, eps):
    if r < 0: return 0.0
    if r >= n: return 1.0
    return sum(comb(n, j) * eps**j * (1 - eps) ** (n - j) for j in range(r + 1))

def r_mds_star(n, eps, delta):
    c, target = 0.0, 1.0 - delta
    for r in range(0, n + 1):
        c += comb(n, r) * eps**r * (1 - eps) ** (n - r)
        if c >= target: return r
    return n

def norm_rows(M):
    return M / (np.linalg.norm(M, axis=1, keepdims=True) + 1e-12)

# ----------------------------- data loading ----------------------------------

EMB_CANDIDATES = ["emb_gte", "emb_e5_large_v2", "emb_text-embedding-3-small"]
RESP_CANDIDATES = ["response_llama_3_8b", "response_gpt-4o-mini",
                   "response_llama_3_70b", "response_gpt-4.1"]

def find_file(data_dir, key):
    for f in sorted(glob.glob(os.path.join(data_dir, "*.parquet"))):
        if key.lower() in os.path.basename(f).lower():
            return f
    return None

def pick_column(cols, wanted, candidates):
    if wanted and wanted in cols: return wanted
    for c in candidates:
        if c in cols: return c
    for c in cols:
        if any(cand.split("_", 1)[-1] in c for cand in candidates): return c
    return None

def load_trimmed(path, emb_col, resp_col, max_rows, out_trim):
    import pyarrow.parquet as pq
    cols = pq.ParquetFile(path).schema_arrow.names
    ecol = pick_column(cols, emb_col, EMB_CANDIDATES)
    rcol = pick_column(cols, resp_col, RESP_CANDIDATES)
    idcol = "id_set" if "id_set" in cols else ("ID_Set" if "ID_Set" in cols else None)
    need = [c for c in ["prompt", idcol, ecol, rcol] if c]
    if ecol is None: raise RuntimeError(f"No embedding column in {path}: {cols}")
    if idcol is None: raise RuntimeError(f"No id_set/ID_Set column in {path}: {cols}")
    df = pd.read_parquet(path, columns=need)
    if max_rows and max_rows > 0:
        df = df.iloc[:max_rows].reset_index(drop=True)
    df = df.rename(columns={idcol: "id_set", ecol: "emb", rcol: "response"})
    os.makedirs(os.path.dirname(out_trim), exist_ok=True)
    df.to_parquet(out_trim, index=False)
    return df, ecol, rcol

# ----------------------------- premise overlay -------------------------------

KAPPA_CONST = 8

def build_classes(df, kappa_rule):
    classes = []
    for cid, g in df.groupby("id_set"):
        size = len(g)
        if kappa_rule == "size": kappa = max(1, size)
        elif kappa_rule == "sqrt": kappa = max(1, int(np.ceil(np.sqrt(size))))
        else: kappa = KAPPA_CONST
        embs = np.stack([np.asarray(e, dtype=np.float32) for e in g["emb"].values])
        classes.append({"id": cid, "size": size, "kappa": kappa, "emb": embs})
    return classes

# --------------------- similarity structure (true + false hits) --------------

def class_similarity_structure(df, classes, max_neg_pairs):
    """Per query: similarity to own class (true-hit evidence) and to the nearest
    FOREIGN class (false-hit evidence). Returns arrays aligned with df rows."""
    rng = np.random.default_rng(0)
    class_index = {c["id"]: i for i, c in enumerate(classes)}
    centroids = norm_rows(np.stack([c["emb"].mean(axis=0) for c in classes]))
    qembs = norm_rows(np.stack([np.asarray(e, np.float32) for e in df["emb"].values]))
    qcls = df["id_set"].values
    n = len(df)
    sim_pos = np.zeros(n)   # to own class
    sim_neg = np.zeros(n)   # to nearest foreign class (false-hit evidence)
    C = len(classes)
    # Map each row to its position inside its class so we can exclude itself (leave-one-out).
    # Within a class, df rows appear in groupby order; track per-class cursor.
    cursor = {}
    for i in range(n):
        ci = class_index[qcls[i]]
        j = cursor.get(ci, 0)
        cursor[ci] = j + 1
        own = norm_rows(classes[ci]["emb"])
        s = own @ qembs[i]
        s[j] = -1.0                     # exclude the query itself (leave-one-out)
        if own.shape[0] > 1:
            sim_pos[i] = float(np.max(s))
        else:
            sim_pos[i] = 1.0            # singleton class: it is its own evidence
        # false-hit evidence: similarity to nearest foreign centroid
        sims = centroids @ qembs[i]
        sims[ci] = -1.0
        sim_neg[i] = float(np.max(sims))
    return qembs, qcls, class_index, sim_pos, sim_neg

def calibrate_threshold(sim_pos, sim_neg, delta):
    """vCache adaptive threshold: the threshold whose FOREIGN-class false-hit rate
    (fraction of queries whose nearest foreign sim >= threshold) meets delta. vCache
    guarantees an error bound while maximising hits, so among thresholds satisfying the
    bound we pick the one with the most true hits; if none meets delta, fall back to the
    threshold with the lowest false-hit rate (most conservative operating point)."""
    lo = float(min(sim_neg.min(), sim_pos.min()))
    thr_grid = np.linspace(max(0.0, lo - 1e-3), 1.0, 121)
    best_feasible, best_fallback, best_fallback_fhr = None, None, np.inf
    for thr in thr_grid:
        fhr = float(np.mean(sim_neg >= thr))
        if fhr <= delta:
            hits = float(np.mean(sim_pos >= thr))       # maximise coverage under the bound
            if best_feasible is None or hits > float(np.mean(sim_pos >= best_feasible)):
                best_feasible = thr
        if fhr < best_fallback_fhr:
            best_fallback_fhr, best_fallback = fhr, thr
    return float(best_feasible if best_feasible is not None else best_fallback)

# ----------------------------- simulation ------------------------------------

def simulate(df, classes, sim_pos, sim_neg, class_index, eps, delta,
             thr_static, thr_vcache, x_frac, seed=42):
    """One pass at erasure rate eps, with both static and adaptive thresholds over
    the full similarity distribution (true hits via sim_pos, false hits via sim_neg).

    A cache *serves* when max(sim_pos, sim_neg_as_hit) >= its threshold.
      - correct serve: matched own class (sim_pos>=thr) AND premises survive.
      - false hit (embedding-level): served on foreign sim (sim_neg>=thr) -> always error.
      - premise-layer error: served on own class but premises erased.
    proof-valid: only serves when sim_pos>=thr AND premises survive -> no premise error,
    and it rejects foreign matches (it must produce a proof), so no false hit either.
    """
    rng = np.random.default_rng(seed)
    n = len(df)
    qcls = df["id_set"].values

    def run(thr, premise_aware):
        served = err_premise = err_falsehit = 0
        hits_theory, hits_emp = [], []
        for i in range(n):
            ci = class_index[qcls[i]]
            kappa = classes[ci]["kappa"]
            x = int(np.ceil(x_frac * kappa))
            n_unc = max(0, kappa - x)
            p_theory = (1 - eps) ** n_unc
            emp_ok = bool(np.all(rng.random(n_unc) > eps)) if n_unc > 0 else True
            hits_theory.append(p_theory)
            hits_emp.append(1.0 if emp_ok else 0.0)

            own_hit = sim_pos[i] >= thr
            foreign_hit = (not own_hit) and (sim_neg[i] >= thr)

            if premise_aware:
                # proof-valid: needs own-class match AND surviving premises
                if own_hit and emp_ok:
                    served += 1
                # no premise error, no false hit by construction
            else:
                if own_hit:
                    served += 1
                    if not emp_ok:
                        err_premise += 1
                elif foreign_hit:
                    served += 1
                    err_falsehit += 1
        tot_err = err_premise + err_falsehit
        return {
            "served": served,
            "premise_err_rate": (err_premise / served) if served else 0.0,
            "falsehit_rate": (err_falsehit / served) if served else 0.0,
            "total_err_rate": (tot_err / served) if served else 0.0,
            "hit_prob_theory": float(np.mean(hits_theory)),
            "hit_prob_empirical": float(np.mean(hits_emp)),
        }

    pv = run(thr_static, premise_aware=True)
    pv["throughput_vs_gptcache"] = (pv["served"] / run(thr_static, False)["served"]
                                    if run(thr_static, False)["served"] else 0.0)
    return {
        "eps": eps,
        "GPTCache": run(thr_static, premise_aware=False),
        "vCache": run(thr_vcache, premise_aware=False),
        "proofvalid": pv,
    }

# ----------------------------- hybrid frontier (per-class) --------------------

def hybrid_frontier_perclass(classes, eps, delta):
    """For each class, the cheapest total storage (x+r) at each eta; then average the
    normalised frontier over the real kappa distribution. Returns per-eta mean + the
    asymptotic line eta+(1-eta)*eps for reference."""
    etas = np.linspace(0.5, 0.95, 19)
    acc = {float(e): [] for e in etas}
    for c in classes:
        k = c["kappa"]
        for eta in etas:
            best = None
            for total in range(0, 2 * k + 1):
                x = int(round(eta * total)); r = total - x
                if x < 0 or r < 0 or x > k: continue
                if (1 - eps) ** (k - x) * binom_cdf(k, r, eps) >= 1 - delta:
                    best = total; break
            if best is not None:
                acc[float(eta)].append(best / k)
    pts = [{"eta": e, "sigma_over_k": float(np.mean(v)),
            "asymptotic": e + (1 - e) * eps}
           for e, v in acc.items() if v]
    return {"eps": eps, "n_classes": len(classes), "points": pts}

# ----------------------------- certificate footprint (per-class) --------------

def certificate_footprint(classes, eps, delta):
    """Fig.4 with real data: x-axis = each class's measured certificate footprint
    (kappa, the premise count); y-axis = storage floor r*_MDS(kappa, eps, delta).
    Aggregate by kappa, weighted by class frequency -> reflects the real class-size
    distribution of the dataset."""
    agg = {}
    for c in classes:
        k = c["kappa"]
        agg.setdefault(k, []).append(r_mds_star(k, eps, delta))
    xs = sorted(agg.keys())
    return {"eps": eps,
            "points": [{"kappa": int(k),
                        "n_classes": len(agg[k]),
                        "storage_floor": float(np.mean(agg[k]))} for k in xs]}

# ----------------------------- main -------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="data")
    ap.add_argument("--out_dir", default="results")
    ap.add_argument("--emb_col", default="emb_gte")
    ap.add_argument("--resp_col", default="response_llama_3_8b")
    ap.add_argument("--max_per_dataset", type=int, default=20000)
    ap.add_argument("--delta", type=float, default=0.02)
    ap.add_argument("--static_quantile", type=float, default=0.90,
                    help="quantile of intra-class sim for GPTCache static threshold")
    ap.add_argument("--kappa_rule", default="size", choices=["size", "sqrt", "const"])
    ap.add_argument("--x_frac", type=float, default=0.5)
    ap.add_argument("--eps_list", default="0,0.05,0.1,0.15,0.2,0.3,0.4,0.5")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    eps_list = [float(x) for x in args.eps_list.split(",")]

    summary = {"config": vars(args), "datasets": {}}

    for key, label in [("SearchQueries", "SemBenchmarkSearchQueries"),
                       ("LmArena", "SemBenchmarkLmArena")]:
        path = find_file(args.data_dir, key)
        if not path:
            print(f"[skip] {label}: no parquet in {args.data_dir}"); continue
        out_trim = os.path.join(args.out_dir, "data_trimmed", f"{label}_trimmed.parquet")
        print(f"[load] {label} from {path}")
        df, ecol, rcol = load_trimmed(path, args.emb_col, args.resp_col,
                                      args.max_per_dataset, out_trim)
        print(f"       rows={len(df)} classes={df['id_set'].nunique()} emb={ecol} resp={rcol}")
        classes = build_classes(df, args.kappa_rule)

        qembs, qcls, class_index, sim_pos, sim_neg = class_similarity_structure(
            df, classes, max_neg_pairs=5)
        thr_static = float(np.quantile(np.clip(sim_pos, 0, 1), args.static_quantile))
        thr_vcache = calibrate_threshold(sim_pos, sim_neg, args.delta)
        print(f"       thr_static={thr_static:.3f}  thr_vcache(adaptive)={thr_vcache:.3f}")
        print(f"       mean sim_pos={sim_pos.mean():.3f}  mean sim_neg={sim_neg.mean():.3f}")

        per_eps = []
        for eps in eps_list:
            m = simulate(df, classes, sim_pos, sim_neg, class_index, eps, args.delta,
                         thr_static, thr_vcache, args.x_frac, seed=42)
            per_eps.append(m)
            g, v, p = m["GPTCache"], m["vCache"], m["proofvalid"]
            print(f"   eps={eps:.2f} | floor theo={g['hit_prob_theory']:.3f} emp={g['hit_prob_empirical']:.3f} | "
                  f"gptErr={g['total_err_rate']:.3f}(p{g['premise_err_rate']:.2f}/f{g['falsehit_rate']:.2f}) "
                  f"vErr={v['total_err_rate']:.3f}(p{v['premise_err_rate']:.2f}/f{v['falsehit_rate']:.2f}) "
                  f"pvErr={p['total_err_rate']:.3f} | pvThroughput={p['throughput_vs_gptcache']:.3f}")

        frontier = hybrid_frontier_perclass(classes, eps=0.1, delta=args.delta)
        cert = certificate_footprint(classes, eps=0.1, delta=args.delta)

        summary["datasets"][label] = {
            "rows": len(df), "classes": int(df["id_set"].nunique()),
            "emb_col": ecol, "resp_col": rcol,
            "thr_static": thr_static, "thr_vcache": thr_vcache,
            "mean_sim_pos": float(sim_pos.mean()), "mean_sim_neg": float(sim_neg.mean()),
            "per_eps": per_eps,
            "hybrid_frontier": frontier,
            "certificate_footprint": cert,
        }

    with open(os.path.join(args.out_dir, "results_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n[done] wrote {os.path.join(args.out_dir,'results_summary.json')}")
    try:
        from make_figures import make_all
        make_all(summary, args.out_dir)
        print("[done] figures ->", args.out_dir)
    except Exception as e:
        print("[warn] plotting skipped:", repr(e))

if __name__ == "__main__":
    main()

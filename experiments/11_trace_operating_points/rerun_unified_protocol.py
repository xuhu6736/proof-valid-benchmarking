#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unified-protocol rerun of the trace endpoint experiment (protocol v3).

Protocol v3 (shared-mask, query-local):
  * One erasure mask is drawn per (erasure rate, request): request i draws
    n_{C(i)} independent Bernoulli(eps) indicators for its exposed leaves;
    o_i = 1 iff all survive.  The SAME mask (o_i) is shared by the static,
    adaptive, oracle-filtered, and proof-valid arms at that rate.
  * The proof-valid arm uses the STATIC threshold and serves exactly the
    own-class static hits whose certificate survives, so the deterministic
    counting identity
        N_PV = N_own,static - N_premise,static
    holds; it is asserted before results are written.
  * Availability audit population: ALL requests (R_hat = mean o_i vs
    R_bar = mean p_i); PV reuse ratio = N_PV / N_static.

Data loading uses DuckDB (the trimmed parquets ship with the repo); all
similarity/threshold logic is imported from run_experiment.py (single
source of truth).  Writes updated trace_endpoints_results.json (LmArena,
quantile 0.90) and trace_endpoints_sq_results.json (SearchQueries,
quantile 0.70); pre-v3 artifacts are preserved as *_legacymask.json.
"""
import json, os, sys
import duckdb
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import run_experiment as RE

EPS_LIST = [0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5]
DELTA = 0.02
X_FRAC = 0.5
SEED = 42


def load_df(path, max_rows=20000):
    con = duckdb.connect()
    return con.sql(
        f"SELECT id_set, emb FROM read_parquet('{path}') LIMIT {max_rows}"
    ).df()


def resp_column_duckdb(path):
    con = duckdb.connect()
    cols = [r[0] for r in con.sql(
        f"DESCRIBE SELECT * FROM read_parquet('{path}')").fetchall()]
    return RE.pick_column(cols, None, RE.RESP_CANDIDATES)


def simulate_shared(sim_pos, sim_neg, qcls, class_index, classes, eps,
                    thr_static, thr_vcache, x_frac=X_FRAC, seed=SEED):
    """One erasure rate, one shared query-local mask, all policy arms."""
    rng = np.random.default_rng(seed)
    n = len(sim_pos)
    p = np.empty(n)
    o = np.ones(n, dtype=bool)
    n_unc_arr = np.empty(n, dtype=int)
    for i in range(n):
        kappa = classes[class_index[qcls[i]]]["kappa"]
        x = int(np.ceil(x_frac * kappa))
        n_unc = max(0, kappa - x)
        n_unc_arr[i] = n_unc
        p[i] = (1.0 - eps) ** n_unc
        o[i] = bool(np.all(rng.random(n_unc) > eps)) if n_unc > 0 else True

    def arm(thr):
        own = sim_pos >= thr
        foreign = (~own) & (sim_neg >= thr)
        served_mask = own | foreign
        served = int(served_mask.sum())
        prem = int((own & ~o).sum())
        fh = int(foreign.sum())
        return {
            "served": served,
            "own_serves": int(own.sum()),
            "foreign_serves": fh,
            "premise_errors": prem,
            "premise_err_rate": (prem / served) if served else 0.0,
            "falsehit_rate": (fh / served) if served else 0.0,
            "total_err_rate": ((prem + fh) / served) if served else 0.0,
        }

    stat = arm(thr_static)
    adap = arm(thr_vcache)
    pv_served = int(((sim_pos >= thr_static) & o).sum())
    identity_lhs = pv_served
    identity_rhs = stat["own_serves"] - stat["premise_errors"]
    assert identity_lhs == identity_rhs, (
        f"counting identity violated: {identity_lhs} != {identity_rhs}")
    balance_lhs = stat["served"] - pv_served
    balance_rhs = stat["foreign_serves"] + stat["premise_errors"]
    assert balance_lhs == balance_rhs, (
        f"served-mass balance violated: {balance_lhs} != {balance_rhs}")

    pv = {
        "served": pv_served,
        "premise_err_rate": 0.0,
        "falsehit_rate": 0.0,
        "total_err_rate": 0.0,
        "throughput_vs_gptcache": (pv_served / stat["served"]
                                   if stat["served"] else 0.0),
    }
    avail = {
        "R_hat_all_requests": float(o.mean()),
        "R_bar_theory": float(p.mean()),
        "abs_dev": float(abs(o.mean() - p.mean())),
        "SE_query_local": float(np.sqrt(np.sum(p * (1 - p))) / n),
    }
    return {
        "eps": eps,
        "GPTCache": {**stat, "hit_prob_theory": avail["R_bar_theory"],
                     "hit_prob_empirical": avail["R_hat_all_requests"]},
        "vCache": {**adap, "hit_prob_theory": avail["R_bar_theory"],
                   "hit_prob_empirical": avail["R_hat_all_requests"]},
        "proofvalid": {**pv, "hit_prob_theory": avail["R_bar_theory"],
                       "hit_prob_empirical": avail["R_hat_all_requests"]},
        "oracle_static": {
            "own_serves": stat["own_serves"],
            "premise_errors": stat["premise_errors"],
            "conditional_err": (stat["premise_errors"] / stat["own_serves"]
                                if stat["own_serves"] else 0.0),
        },
        "availability": avail,
        "count_identity_N_PV_eq_N_own_minus_N_premise": True,
        "count_identity_N_static_minus_N_PV_eq_N_false_plus_N_premise": True,
    }


def run_dataset(key, label, quantile):
    path = RE.find_file("data", key)
    print(f"[load] {label} from {path}", flush=True)
    df = load_df(path)
    rcol = resp_column_duckdb(path)
    print(f"       rows={len(df)} classes={df['id_set'].nunique()} resp={rcol}",
          flush=True)
    classes = RE.build_classes(df, "size")
    qembs, qcls, class_index, sim_pos, sim_neg = RE.class_similarity_structure(
        df, classes, max_neg_pairs=5)
    thr_static = float(np.quantile(np.clip(sim_pos, 0, 1), quantile))
    thr_vcache = RE.calibrate_threshold(sim_pos, sim_neg, DELTA)
    print(f"       thr_static={thr_static:.7f} thr_vcache={thr_vcache:.7f}",
          flush=True)
    per_eps = []
    for eps in EPS_LIST:
        m = simulate_shared(sim_pos, sim_neg, qcls, class_index, classes,
                            float(eps), thr_static, thr_vcache)
        per_eps.append(m)
        g, v, pv = m["GPTCache"], m["vCache"], m["proofvalid"]
        print(f"   eps={eps:.2f} | dev={m['availability']['abs_dev']:.5f} | "
              f"statErr={g['total_err_rate']:.4f} orErr="
              f"{m['oracle_static']['conditional_err']:.4f} "
              f"adErr={v['total_err_rate']:.4f} | PV={pv['served']} "
              f"thr={pv['throughput_vs_gptcache']:.4f}", flush=True)
    frontier = RE.hybrid_frontier_perclass(classes, eps=0.1, delta=DELTA)
    cert = RE.certificate_footprint(classes, eps=0.1, delta=DELTA)
    return {
        "rows": len(df), "classes": int(df["id_set"].nunique()),
        "emb_col": "emb_gte", "resp_col": rcol,
        "thr_static": thr_static, "thr_vcache": thr_vcache,
        "mean_sim_pos": float(sim_pos.mean()),
        "mean_sim_neg": float(sim_neg.mean()),
        "per_eps": per_eps,
        "hybrid_frontier": frontier,
        "certificate_footprint": cert,
    }


def proto_meta():
    proto = ("query-local shared-mask v3: one Bernoulli mask per "
             "(rate, request) shared by all policy arms; availability "
             "audited over all requests; PV arm = static threshold + "
             "own-class + survival; identities N_PV=N_own-N_premise and "
             "N_static-N_PV=N_false+N_premise asserted")
    thr_meta = {
        "static": "np.quantile(clip(sim_pos,0,1), q) per dataset config",
        "adaptive": "121-point linspace grid on [max(0,lo-1e-3),1], "
                    "lo=min(sim_pos.min(),sim_neg.min()); feasible set "
                    "mean(sim_neg>=thr)<=delta=0.02, maximize mean(sim_pos>=thr); "
                    "comparison operator >=, scores float32",
    }
    return proto, thr_meta


def run_sq07():
    proto, thr_meta = proto_meta()
    # SearchQueries, static quantile 0.70 (reported panel)
    sq = run_dataset("SearchQueries", "SemBenchmarkSearchQueries", 0.70)
    out_sq = {"config": {"data_dir": "data", "emb_col": "emb_gte",
                         "max_per_dataset": 20000, "delta": DELTA,
                         "static_quantile": 0.70, "kappa_rule": "size",
                         "x_frac": X_FRAC,
                         "eps_list": ",".join(str(e) for e in EPS_LIST),
                         "protocol": proto, "thresholds": thr_meta},
              "note": ("Protocol v3 shared-mask rerun (2026-10-13); "
                       "supersedes the pre-v3 mask stream. Legacy artifact: "
                       "trace_endpoints_sq_results_legacymask.json"),
              "datasets": {"SemBenchmarkSearchQueries": sq}}
    json.dump(out_sq, open("trace_endpoints_sq_results.json", "w"), indent=1)
    print("[saved] trace_endpoints_sq_results.json", flush=True)

def run_main():
    proto, thr_meta = proto_meta()
    # LmArena, static quantile 0.90; keep SQ 0.90 panel in the main file
    sq90 = run_dataset("SearchQueries", "SemBenchmarkSearchQueries", 0.90)
    la = run_dataset("LmArena", "SemBenchmarkLmArena", 0.90)
    out_main = {"config": {"data_dir": "data", "emb_col": "emb_gte",
                           "max_per_dataset": 20000, "delta": DELTA,
                           "static_quantile": 0.90, "kappa_rule": "size",
                           "x_frac": X_FRAC,
                           "eps_list": ",".join(str(e) for e in EPS_LIST),
                           "protocol": proto, "thresholds": thr_meta},
                "note": ("Protocol v3 shared-mask rerun (2026-10-13). "
                         "SearchQueries at quantile 0.90 is a legacy "
                         "reference panel; the reported SearchQueries panel "
                         "uses quantile 0.70 (trace_endpoints_sq_results.json). "
                         "Legacy artifact: "
                         "trace_endpoints_results_legacymask.json"),
                "datasets": {"SemBenchmarkSearchQueries": sq90,
                             "SemBenchmarkLmArena": la}}
    json.dump(out_main, open("trace_endpoints_results.json", "w"), indent=1)
    print("[saved] trace_endpoints_results.json", flush=True)




if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    if which in ("sq07", "all"):
        run_sq07()
    if which in ("main", "all"):
        run_main()

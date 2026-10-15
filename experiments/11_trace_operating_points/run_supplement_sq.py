#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Supplementary run: SemBenchmarkSearchQueries with a lower GPTCache static quantile.

Rationale: at the default --static_quantile 0.9 the intra-class similarity 90%-quantile
on SearchQueries saturates at 1.000, so the GPTCache static baseline degenerates to a
near-no-op (it almost never serves). Lowering the quantile brings the static baseline to
a meaningful operating point, completing the three-way comparison on this dataset.

This script reuses every function from run_experiment.py (single source of truth) and
only overrides the static quantile and the dataset subset. Run it from the same folder
as run_experiment.py:

    python run_supplement_sq.py --data_dir data --out_dir results_v2_sq07 \
        --emb_col emb_gte --delta 0.02 --static_quantile 0.7 --max_per_dataset 20000
"""
import argparse, json, os, sys

# import the fixed experiment module (single source of truth)
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import run_experiment as RE


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="data")
    ap.add_argument("--out_dir", default="results_v2_sq07")
    ap.add_argument("--emb_col", default="emb_gte")
    ap.add_argument("--resp_col", default="response_llama_3_8b")
    ap.add_argument("--max_per_dataset", type=int, default=20000)
    ap.add_argument("--delta", type=float, default=0.02)
    ap.add_argument("--static_quantile", type=float, default=0.7,
                    help="lower quantile so the GPTCache static baseline reaches a "
                         "meaningful operating point on SearchQueries")
    ap.add_argument("--kappa_rule", default="size", choices=["size", "sqrt", "const"])
    ap.add_argument("--x_frac", type=float, default=0.5)
    ap.add_argument("--eps_list", default="0,0.05,0.1,0.15,0.2,0.3,0.4,0.5")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    eps_list = [float(x) for x in args.eps_list.split(",")]

    summary = {"config": vars(args),
               "note": "Supplementary SearchQueries run with lower GPTCache static "
                       "quantile to avoid the degenerate thr=1.0 baseline.",
               "datasets": {}}

    label = "SemBenchmarkSearchQueries"
    path = RE.find_file(args.data_dir, "SearchQueries")
    if not path:
        print(f"[skip] no SearchQueries parquet in {args.data_dir}")
        return
    out_trim = os.path.join(args.out_dir, "data_trimmed", f"{label}_trimmed.parquet")
    print(f"[load] {label} from {path}")
    df, ecol, rcol = RE.load_trimmed(path, args.emb_col, args.resp_col,
                                     args.max_per_dataset, out_trim)
    print(f"       rows={len(df)} classes={df['id_set'].nunique()} emb={ecol} resp={rcol}")
    classes = RE.build_classes(df, args.kappa_rule)

    qembs, qcls, class_index, sim_pos, sim_neg = RE.class_similarity_structure(
        df, classes, max_neg_pairs=5)
    thr_static = float(RE.np.quantile(RE.np.clip(sim_pos, 0, 1), args.static_quantile))
    thr_vcache = RE.calibrate_threshold(sim_pos, sim_neg, args.delta)
    print(f"       thr_static(q={args.static_quantile})={thr_static:.3f}  "
          f"thr_vcache(adaptive)={thr_vcache:.3f}")
    print(f"       mean sim_pos={sim_pos.mean():.3f}  mean sim_neg={sim_neg.mean():.3f}")

    per_eps = []
    for eps in eps_list:
        m = RE.simulate(df, classes, sim_pos, sim_neg, class_index, eps, args.delta,
                        thr_static, thr_vcache, args.x_frac, seed=42)
        per_eps.append(m)
        g, v, p = m["GPTCache"], m["vCache"], m["proofvalid"]
        print(f"   eps={eps:.2f} | floor theo={g['hit_prob_theory']:.3f} "
              f"emp={g['hit_prob_empirical']:.3f} | "
              f"gptErr={g['total_err_rate']:.3f}(f{g['falsehit_rate']:.2f}/p{g['premise_err_rate']:.2f}) "
              f"vcErr={v['total_err_rate']:.3f}(f{v['falsehit_rate']:.2f}/p{v['premise_err_rate']:.2f}) "
              f"pvErr={p['total_err_rate']:.3f} | pvThr={p['throughput_vs_gptcache']:.3f}")

    frontier = RE.hybrid_frontier_perclass(classes, eps=0.1, delta=args.delta)
    cert = RE.certificate_footprint(classes, eps=0.1, delta=args.delta)

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

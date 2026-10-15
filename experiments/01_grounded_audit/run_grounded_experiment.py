#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Grounded validation experiment (evidence-plan steps 1-2)
for "Proof-Valid Benchmarking for Approximate LLM Caching under Premise Erasures".

Pipeline
--------
Step 0  Download data:
        * HotpotQA distractor validation split (7,405 questions, human-annotated
          supporting facts)  -- HF dataset hotpotqa/hotpot_qa, config "distractor".
        * id_set columns of the two vCache SemBenchmark traces (read by HTTP range
          requests; only the class-id column is fetched, not the full parquets).
Step 1  Distribution audit: annotated evidence counts e_q vs semantic-class sizes,
        question-level and class-level Spearman rank correlations.
Step 2  Grounded stress test: i.i.d. erasure of annotated evidence items; exact law
        R(eps) = mean_q (1-eps)^{e_q} vs Monte-Carlo simulation; proxy-implied curve
        with exponent |C(q)| on the same questions; measured proxy bias checked
        against the proxy-sensitivity bound  |R_hat - R| <= Delta * ln(1/(1-eps)).

Outputs (written to --out):
        fig5_grounded_audit_scis.pdf / .png
        fig6_grounded_stress_scis.pdf / .png
        results_grounded.json

Dependencies: numpy pandas scipy matplotlib pyarrow fsspec aiohttp requests
              sentence-transformers torch
Usage:    python3 run_grounded_experiment.py --out ./results
          python3 run_grounded_experiment.py --out ./results --hf-endpoint https://hf-mirror.com
"""
import argparse, json, os
import numpy as np
import pandas as pd

HOTPOT_URL = "datasets/hotpotqa/hotpot_qa/resolve/main/distractor/validation-00000-of-00001.parquet"
SQ_URL     = "datasets/vCache/SemBenchmarkSearchQueries/resolve/main/other_60k_with_responses.parquet"
LA_URL     = "datasets/vCache/SemBenchmarkLmArena/resolve/main/train.parquet"
EPS_LIST   = np.array([0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5])
SEED       = 42
N_REP      = 20
TAU        = 0.75          # embedding class threshold (sensitivity: 0.70 / 0.80 reported)
MAX_ROWS   = 20000         # the paper's evaluated prefix of each trace


def hf_url(base, path):
    return f"{base.rstrip('/')}/{path}"


def download(url, dst):
    if os.path.exists(dst) and os.path.getsize(dst) > 0:
        return dst
    import urllib.request
    print(f"[download] {url}\n        -> {dst}", flush=True)
    urllib.request.urlretrieve(url, dst)
    return dst


def read_remote_column(url, column):
    """Read a single column from a remote parquet via HTTP range requests."""
    import fsspec, pyarrow.parquet as pq
    with fsspec.open(url, "rb") as f:
        pf = pq.ParquetFile(f)
        t = pf.read(columns=[column])
    return t.column(column).to_pylist()


def class_sizes_from_ids(ids):
    return pd.Series(ids).value_counts().values.astype(int)


def threshold_classes(emb, tau, block=1024):
    """Connected components of the cosine-similarity threshold graph."""
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import connected_components
    n = emb.shape[0]
    rows, cols = [], []
    for i in range(0, n, block):
        S = emb[i:i + block] @ emb.T
        r, c = np.nonzero(S >= tau)
        keep = (i + r) < c
        rows.append((i + r)[keep]); cols.append(c[keep])
    rows = np.concatenate(rows); cols = np.concatenate(cols)
    A = csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(n, n))
    return connected_components(A, directed=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="./results")
    ap.add_argument("--data", default="./data")
    ap.add_argument("--hf-endpoint", default="https://huggingface.co",
                    help="use https://hf-mirror.com where huggingface.co is unreachable")
    ap.add_argument("--tau", type=float, default=TAU)
    ap.add_argument("--n-rep", type=int, default=N_REP)
    ap.add_argument("--seed", type=int, default=SEED)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    os.makedirs(args.data, exist_ok=True)
    os.environ.setdefault("HF_ENDPOINT", args.hf_endpoint)
    base = args.hf_endpoint

    # ---------------- Step 0: data ----------------
    hp_path = download(hf_url(base, HOTPOT_URL), os.path.join(args.data, "hotpot_validation.parquet"))
    df = pd.read_parquet(hp_path)
    n = len(df)
    print(f"[data] HotpotQA validation: {n} questions")

    sq_ids = read_remote_column(hf_url(base, SQ_URL), "id_set")[:MAX_ROWS]
    la_ids = read_remote_column(hf_url(base, LA_URL), "ID_Set")[:MAX_ROWS]
    sq_sizes = class_sizes_from_ids(sq_ids)
    la_sizes = class_sizes_from_ids(la_ids)
    print(f"[data] SearchQueries first {MAX_ROWS}: {len(sq_sizes)} classes, "
          f"mean {sq_sizes.mean():.3f}, max {sq_sizes.max()}")
    print(f"[data] LmArena      first {MAX_ROWS}: {len(la_sizes)} classes, "
          f"mean {la_sizes.mean():.3f}, max {la_sizes.max()}")

    # ---------------- Step 1: distribution audit ----------------
    e_q = np.array([len(set(zip(sf["title"], sf["sent_id"].tolist())))
                    for sf in df["supporting_facts"]])
    hist = dict(zip(*[a.tolist() for a in np.unique(e_q, return_counts=True)]))
    print(f"[audit] evidence counts e_q: hist {hist}, mean {e_q.mean():.4f}, max {e_q.max()}")

    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    emb = model.encode(df["question"].tolist(), batch_size=256,
                       normalize_embeddings=True, convert_to_numpy=True)
    from scipy.stats import spearmanr
    audit = {}
    for tau in [0.70, args.tau, 0.80]:
        ncomp, labels = class_dummy = threshold_classes(emb, tau)
        sizes = np.bincount(labels)
        cs_q = sizes[labels]
        rho_q, p_q = spearmanr(cs_q, e_q)
        pc_e = pd.Series(e_q).groupby(labels).mean()
        pc_s = pd.Series(cs_q).groupby(labels).first()
        rho_c, p_c = spearmanr(pc_s, pc_e)
        audit[tau] = dict(n_classes=int(ncomp), singleton_frac=float((sizes == 1).mean()),
                          max_size=int(sizes.max()),
                          question_level_spearman=[float(rho_q), float(p_q)],
                          class_level_spearman=[float(rho_c), float(p_c)])
        print(f"[audit] tau={tau}: classes {ncomp}, singletons {(sizes==1).mean()*100:.1f}%, "
              f"Spearman q-level rho={rho_q:.4f} (p={p_q:.3g}), "
              f"class-level rho={rho_c:.4f} (p={p_c:.3g})")
    ncomp, labels = threshold_classes(emb, args.tau)
    sizes = np.bincount(labels)
    cs_q = sizes[labels]

    # ---------------- Step 2: grounded stress test ----------------
    rng = np.random.default_rng(args.seed)
    R_exact = np.array([np.mean((1 - e) ** e_q) for e in EPS_LIST])
    R_sim = []
    for eps in EPS_LIST:
        surv = rng.random((args.n_rep, n, int(e_q.max()))) >= eps
        pad = np.arange(e_q.max())[None, :] >= e_q[:, None]      # padding always survives
        surv = surv | pad[None, :, :]
        R_sim.append(surv.all(axis=2).mean(axis=1))
    R_sim = np.array(R_sim)
    max_dev = float(np.abs(R_exact - R_sim.mean(axis=1)).max())
    print(f"[stress] exact-law vs Monte-Carlo: max |dev| = {max_dev:.5f}")

    # proxy-implied curve on the same questions, and measured proxy bias
    R_proxy = np.array([np.mean((1 - e) ** cs_q) for e in EPS_LIST])
    bias = np.abs(R_proxy - R_exact)
    Delta = np.abs(cs_q - e_q)
    print(f"[stress] proxy bias: max {bias.max():.4f} at eps={EPS_LIST[bias.argmax()]}, "
          f"mean Delta = {Delta.mean():.3f}, max Delta = {Delta.max()}")
    bound_ok = all(b <= (Delta.mean() * np.log(1 / (1 - e)) + 1e-12 if e > 0 else 0) + 1e-12
                   for b, e in zip(bias, EPS_LIST))
    print(f"[stress] sensitivity bound |R_hat-R| <= mean(Delta)*ln(1/(1-eps)) holds: {bound_ok}")

    # trace-proxy footprint curves (prompt-frequency weighted, x = 0)
    eps_grid = np.linspace(0, 0.5, 51)
    R_grounded_g = np.array([np.mean((1 - e) ** e_q) for e in eps_grid])
    R_hp_proxy_g = np.array([np.mean((1 - e) ** cs_q) for e in eps_grid])
    per_prompt = lambda ids: pd.Series(ids).map(pd.Series(ids).value_counts()).values
    sq_pp, la_pp = per_prompt(sq_ids), per_prompt(la_ids)
    R_sq_g = np.array([np.mean((1 - e) ** sq_pp) for e in eps_grid])
    R_la_g = np.array([np.mean((1 - e) ** la_pp) for e in eps_grid])

    # ---------------- figures ----------------
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    ax = axes[0]
    ax.hist(sq_sizes, bins=np.arange(0.5, 17.5, 1), histtype="step", lw=2,
            label=f"SearchQueries class size (mean {sq_sizes.mean():.2f})", color="tab:blue")
    ax.hist(la_sizes, bins=np.arange(0.5, 17.5, 1), histtype="step", lw=2,
            label=f"LmArena class size (mean {la_sizes.mean():.2f})", color="tab:orange")
    ax.hist(e_q, bins=np.arange(0.5, 9.5, 1), histtype="step", lw=2.4,
            label=f"HotpotQA evidence count $e_q$ (mean {e_q.mean():.2f})", color="tab:green")
    ax.set_yscale("log"); ax.set_xlabel("footprint value"); ax.set_ylabel("count (log scale)")
    ax.set_title("(a) footprint distributions", fontsize=11)
    ax.legend(fontsize=8.5); ax.grid(alpha=.3)
    ax = axes[1]
    joint = np.zeros((5, 8), dtype=int)
    for a_, b_ in zip(np.clip(cs_q, 1, 5), e_q):
        joint[a_ - 1, b_ - 1] += 1
    im = ax.imshow(joint.T, origin="lower", aspect="auto", cmap="viridis",
                   extent=[0.5, 5.5, 0.5, 8.5])
    ax.set_xticks([1, 2, 3, 4, 5]); ax.set_xticklabels(["1", "2", "3", "4", "5+"])
    ax.set_yticks(range(2, 9))
    ax.set_xlabel(r"embedding class size $|C(q)|$ ($\tau=%.2f$)" % args.tau)
    ax.set_ylabel(r"annotated evidence count $e_q$")
    ax.set_title(f"(b) joint distribution (Spearman $\\rho$={audit[args.tau]['question_level_spearman'][0]:.3f})",
                 fontsize=11)
    fig.colorbar(im, ax=ax, label="questions")
    fig.tight_layout()
    fig.savefig(os.path.join(args.out, "fig5_grounded_audit_scis.pdf"), bbox_inches="tight")
    fig.savefig(os.path.join(args.out, "fig5_grounded_audit.png"), dpi=160, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.6, 4.6))
    ax.plot(eps_grid, R_grounded_g, "-", lw=2.2, color="tab:green",
            label=r"exact law, grounded exponent $e_q$: $\frac{1}{n}\sum_q (1-\varepsilon)^{e_q}$")
    ax.plot(EPS_LIST, R_sim.mean(axis=1), "o", ms=7, mfc="none", mew=1.8, color="tab:green",
            label=f"Monte-Carlo erasure of annotated facts ({args.n_rep} reps)")
    ax.plot(eps_grid, R_hp_proxy_g, "--", lw=2, color="tab:red",
            label=r"proxy-implied, exponent $|C(q)|$ on same questions")
    ax.fill_between(eps_grid, R_grounded_g, R_hp_proxy_g, color="tab:red", alpha=0.12,
                    label=f"measured proxy bias (max {bias.max():.3f} at $\\varepsilon$="
                          f"{EPS_LIST[bias.argmax()]})")
    ax.plot(eps_grid, R_sq_g, ":", lw=1.6, color="tab:blue",
            label="SearchQueries trace proxy (per-prompt class size)")
    ax.plot(eps_grid, R_la_g, ":", lw=1.6, color="tab:orange",
            label="LmArena trace proxy (per-prompt class size)")
    ax.set_xlabel(r"premise erasure rate $\varepsilon$")
    ax.set_ylabel("certificate-survival probability")
    ax.set_ylim(0, 1.02); ax.grid(alpha=.3); ax.legend(fontsize=8.5, loc="upper right")
    fig.tight_layout()
    fig.savefig(os.path.join(args.out, "fig6_grounded_stress_scis.pdf"), bbox_inches="tight")
    fig.savefig(os.path.join(args.out, "fig6_grounded_stress.png"), dpi=160, bbox_inches="tight")
    plt.close(fig)

    # ---------------- results json ----------------
    res = {
        "dataset": {"name": "HotpotQA distractor validation (hotpotqa/hotpot_qa)",
                    "n_questions": int(n)},
        "evidence_counts": {"hist": hist, "mean": float(e_q.mean()),
                            "median": float(np.median(e_q)), "max": int(e_q.max())},
        "embedding_class_audit": audit,
        "grounded_stress": {"eps": EPS_LIST.tolist(), "R_exact": R_exact.tolist(),
                            "R_sim_mean": R_sim.mean(axis=1).tolist(),
                            "max_theory_sim_dev": max_dev},
        "proxy_bias": {"R_proxy": R_proxy.tolist(), "R_grounded": R_exact.tolist(),
                       "max_abs_bias": float(bias.max()),
                       "argmax_eps": float(EPS_LIST[bias.argmax()]),
                       "Delta_mean": float(Delta.mean()), "Delta_max": float(Delta.max()),
                       "mean_delta_bound_holds": bool(bound_ok)},
        "trace_class_sizes": {
            "SQ": {"classes": int(len(sq_sizes)), "mean": float(sq_sizes.mean()),
                   "max": int(sq_sizes.max())},
            "LA": {"classes": int(len(la_sizes)), "mean": float(la_sizes.mean()),
                   "max": int(la_sizes.max())}},
        "footprint_curve_at_eps_0.1": {"grounded": float(R_grounded_g[10]),
                                       "hp_proxy": float(R_hp_proxy_g[10]),
                                       "SQ_proxy": float(R_sq_g[10]),
                                       "LA_proxy": float(R_la_g[10])},
        "config": {"tau": args.tau, "n_rep": args.n_rep, "seed": args.seed,
                   "eps_list": EPS_LIST.tolist(), "max_rows": MAX_ROWS,
                   "embedding_model": "sentence-transformers/all-MiniLM-L6-v2"},
    }
    with open(os.path.join(args.out, "results_grounded.json"), "w") as f:
        json.dump(res, f, indent=1)
    print(f"[done] figures and results_grounded.json written to {args.out}")


if __name__ == "__main__":
    main()

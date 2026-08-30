# Proof-Valid Benchmarking for Approximate LLM Caching under Premise Erasures

Experimental code and result artifacts for the paper

> **Proof-Valid Benchmarking for Approximate LLM Caching under Premise Erasures**
> Jianfeng Xu — Koguan School of Law, China Institute for Smart Justice, School of Computer Science, Shanghai Jiao Tong University
> ChinaXiv:202608.00158v1 · *Sci China Inf Sci* (under review)

- Paper (ChinaXiv): `ChinaXiv202608.00158v1_Proof-Valid-Caching(1).pdf`
- Repository: https://github.com/xuhu6736/proof-valid-benchmarking

---

## Overview

Approximate caches for large language models reuse a stored answer when a new prompt is *semantically close* to a cached one. Similarity alone, however, does not establish that the evidence supporting the stored answer is still available. **Proof-valid benchmarking (PVB)** redefines cache serving as a proof obligation: a cached answer may be served only with a **certificate** whose premises survive a controlled **premise-erasure stress test**. An embedding hit whose premises have been erased is *declined and regenerated*, rather than silently served.

The paper provides:

1. **The PVB framework** — response-conditioned evidence-support DAGs, premise-leaf feature records, and an explicit certificate-construction interface.
2. **Exact reliability laws** — the residual-premise law `Pr[serve] = (1 - ε)^{e_q(S)}`, with closed-form reliability thresholds and the exact transparent-storage baseline.
3. **Protection mechanisms and their limits** — semantic-module selection (NP-complete, with a greedy logarithmic guarantee), a certificate-based coded converse (proof *size*, not embedding dimension, sets the storage floor), and the corrected residual-code hybrid law under a common erasure realization.
4. **Three-layer experimental closure** plus a **real-dynamics harm audit** on live 2026 Wikipedia/Wikidata snapshots.

### Headline results

| Result | Value |
| --- | --- |
| Exact law vs. Monte-Carlo (grounded HotpotQA) | agreement within **0.00095** |
| Exact law vs. instrumented pipeline | agreement within **0.0016** |
| Certificate construction / gate cost | **38 µs** / **1.6 µs** per question |
| Module-sharing savings (ρ = 0.5) | **44.6%** (HotpotQA), **71.9%** (MuSiQue) |
| Coded advantage at equal budget | **+0.51** (κ = 10), **+0.77** (κ = 20) |
| Measured premise erasure ε(t), 1→4 months | **1.61% → 5.21%** |
| Gate interception / false rejection on real 2026 changes | **44.6%** / **11.1%** |

---

## Repository structure

Each numbered directory implements one experiment from the paper (Table A4). The naming follows the experiment inventory in Appendix A.6.

| Directory | Experiment | Data | Key artifact(s) | Paper |
| --- | --- | --- | --- | --- |
| `01_Grounded audit + stress` | Distribution audit + grounded stress test, proxy-bias measurement | HP + SQ/LA | `results_grounded.json` | §6.1, Fig. 2 |
| `02_Constructor noise` | Constructor-error sensitivity (Proposition 4 tightness) | HP | `e13_constructor_noise_results.json` | §6.1 |
| `03_Coded regime + converse` | Coded/hybrid regime map + certificate floor (Thm. 5-6) | HP | `coded_regime_results.json` | §6.2, Fig. 3 |
| `04_OR redundancy (loose_strict)` | Disjunctive certificate families (Proposition 2 bookends) | HP | `e11_or_strict_results.json` | §6.2, Fig. 4 |
| `05_Module selection` | Module selection at the NP-complete boundary (Thm. 3, Prop. 4) | HP + MQ | `module_selection_results.json`, `e12_module_musique_results.json` | §6.2, Fig. 5 |
| `06_Gain curve (natural certs)` | Coded-advantage curve on natural full-context certificates | HP + MQ | `e6b_natural_context_results.json` | §6.2, Fig. 6 |
| `07_Grounded operating points` | Grounded operating-point comparison (three cache semantics) | HP + MQ | `grounded_operating_results.json`, `TableA1.tex` | §6.2, Tab. A1 |
| `08_Instrumented pipeline` | Instrumented-provenance pipeline, closed-loop validation | HP (66,581 docs) | `e8_pipeline_results.json` | §6.3, Fig. 7 |
| `09_Correlated erasure` | Document-level correlated + bursty erasure (i.i.d. boundary) | HP + MQ | `e9_correlated_results.json` | §6.3, Fig. 8 |
| `10_Pareto frontiers` | Error-vs-throughput Pareto frontiers (similarity-only vs. PVB) | HP | `e10_pareto_results.json` | §6.4, Fig. 9 |
| `11_Regeneration acceptance` | Embedding-judge acceptance test for the regeneration envelope | HP | `regeneration_results.json` | App. B, Fig. B1 |
| `12_Trace operating points` | End-to-end stress test on the two vCache benchmark traces | SQ + LA | `trace_endpoints_results.json`, `trace_endpoints_sq_results.json`, `TableA3.tex` | App. A, Tab. A3 |
| `13_Real-dynamics ε(t)` | Measured premise-erasure curve ε(t) on 2026 Wikipedia/Wikidata snapshots | EN + WD 2026 | `eps_t.json`, `eps_stmt_v2.json`, `Table1.tex` | §6.5, Fig. 10(a) |
| `14_Gate audit on real changes` | End-to-end interception/false-rejection audit of the serving gate | EN + WD 2026, TW panel | `constructor_results_v4.json`, `goldstandard_panel_2026.json`, `Table2.tex` | §6.5, Fig. 11 |
| `15_Harm scale + popularity` | Harm scale by subject popularity (WikiFactDiff legacy) | WikiFactDiff | `wfd_popularity_workload.json`, `workload_corrected_2026.json` | §6.5, Fig. 10(b) |

**Data key:** HP = HotpotQA distractor validation (7,405 questions); MQ = MuSiQue full dev split (2,417 answerable); SQ/LA = vCache SemBenchmark SearchQueries/LmArena (20,000 prompts each); TW = TemporalWiki probe subjects; EN/WD = Wikimedia enwiki/wikidatawiki 2026 snapshots.

> **Note on figure naming.** Committed figure PDFs are named by their *internal* experiment labels (`fig1a_*`, `fig2a_*`, …, `fig15_*`), which do not coincide with the paper's figure numbers. The mapping is: `fig1a/b/c` → paper Fig. 2(a/b/c); `fig2a/b` → Fig. 3(a/b); `fig3` → Fig. 4; `fig4a/b` → Fig. 5(a/b); `fig5a/b` → Fig. 6(a/b); `fig7` → Fig. B1; `fig13` → Fig. 7; `fig14` → Fig. 8; `fig15` → Fig. 9.

---

## Installation

Python 3.9+ recommended. The scripts depend on:

```bash
pip install numpy pandas scipy matplotlib pyarrow statsmodels \
            sentence-transformers torch fsspec aiohttp requests
```

The encoder used throughout is `all-MiniLM-L6-v2` (via `sentence-transformers`); `torch` is required only as a sentence-transformers backend.

---

## Data preparation

Most experiments read two grounded datasets from a local `data/` directory:

| File | Source | Notes |
| --- | --- | --- |
| `data/hotpot_validation.parquet` | `datasets/hotpotqa/hotpot_qa` (config `distractor`, validation split) | human-annotated `supporting_facts` |
| `data/musique_full_dev.jsonl` | `datasets/MuSiQue` (full dev split) | 4,834 records, 2,417 answerable |
| `data/question_emb.npy`, `data/doc_emb.npy` | generated by `08_Instrumented pipeline/embed_docs_sliced.py` | `all-MiniLM-L6-v2` embeddings |

Some scripts resolve `data/` relative to the working directory (`data/hotpot_validation.parquet`), others relative to their own location (`os.path.dirname(__file__)/data/…`). The most robust setup is a shared download plus a `data/` **symlink** in each experiment folder.

- `01_Grounded audit + stress` and `12_Trace operating points` fetch the vCache SemBenchmark traces (and HotpotQA, in the case of 01) directly over HTTP range requests, so they do not require the parquet to be pre-downloaded.
- `13`–`15` implement the **real-dynamics audit**: they download Wikimedia dump snapshots (enwiki/wikidatawiki, 2026) and read/write under `/tmp/pvb_data` and `/mnt/agents/output/pvb_pilot` (Linux paths). These are heavyweight downloads and are best run on a Linux machine with adequate disk.

---

## Reproducing the experiments

From the repository root, with `data/` prepared as above:

```bash
# §6.1  Grounded audit + stress (downloads HotpotQA + vCache traces over HTTP)
python3 "01_Grounded audit + stress/run_grounded_experiment.py" --out ./results

# §6.1  Constructor-noise sensitivity
python3 "02_Constructor noise/run_constructor_noise_experiment.py"

# §6.2  Coded regime + converse
python3 "03_Coded regime + converse/run_coded_regime_experiment.py"

# §6.2  OR redundancy (loose vs. strict certificates)
python3 "04_OR redundancy (loose_strict)/run_or_redundancy_strict_experiment.py"

# §6.2  Module selection (HotpotQA + MuSiQue)
python3 "05_Module selection/run_module_selection_experiment.py"
python3 "05_Module selection/run_module_selection_musique.py"

# §6.2  Gain curve on natural full-context certificates
python3 "06_Gain curve (natural certs)/run_natural_context_curve_experiment.py"

# §6.2  Grounded operating points
python3 "07_Grounded operating points/run_grounded_operating_experiment.py"

# §6.3  Instrumented pipeline (requires precomputed question/doc embeddings)
python3 "08_Instrumented pipeline/embed_docs_sliced.py"
python3 "08_Instrumented pipeline/run_pipeline_experiment.py"

# §6.3  Correlated + bursty erasure
python3 "09_Correlated erasure/run_correlated_erasure_experiment.py"

# §6.4  Pareto frontiers
python3 "10_Pareto frontiers/run_pareto_frontier_experiment.py"

# App. B  Regeneration acceptance
python3 "11_Regeneration acceptance/run_regeneration_experiment.py"

# App. A  Trace operating points (vCache traces; re-checks archived numbers if embeddings present)
python3 "12_Trace operating points/run_trace_endpoints_experiment.py"

# §6.5  Real-dynamics audit (heavy; Wikimedia dumps, Linux paths)
python3 "13_Real-dynamics ε(t)/full_orchestrator.py"
python3 "14_Gate audit on real changes/constructor_lab.py" validate
python3 "15_Harm scale + popularity/make_wfd_popularity_workload.py"
```

All stochastic experiments use **seed 42**, the erasure grid `{0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5}`, and a reliability target `δ = 0.02` (the paper's main configuration). Committed result JSONs were produced by the scripts as written; re-running regenerates the figures and re-checks the archived numbers.

---

## Result artifacts

Beyond the scripts, each folder contains the archived artifacts from the paper:

- **JSON result summaries** — the endpoint values cited in the paper (e.g. `results_grounded.json`, `e8_pipeline_results.json`, `eps_t.json`), read verbatim by the paper's tables and figures; nothing is smoothed or extrapolated.
- **Figure PDFs** — `*_scis.pdf` files (see the naming note above).
- **LaTeX tables** — `Table1.tex` (measured ε(t)), `Table2.tex` (gate operating points), `TableA1.tex` (grounded operating points), `TableA3.tex` (trace operating points).

---

## Citation

```bibtex
@misc{xu2026proofvalid,
  title   = {Proof-Valid Benchmarking for Approximate LLM Caching under Premise Erasures},
  author  = {Jianfeng Xu},
  year    = {2026},
  note    = {ChinaXiv:202608.00158v1},
  howpublished = {\url{https://chinaxiv.org/abs/202608.00158}},
  journal = {Sci China Inf Sci (under review)}
}
```

---

## License

This repository accompanies a paper under review. The code and artifacts are released for reproducibility purposes; please cite the paper if you use them.

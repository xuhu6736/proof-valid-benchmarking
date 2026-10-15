# Proof-Valid Serving and Benchmarking for Approximate LLM Caches — Experiment Artifact

This repository contains the complete experimental programs, data, and result
artifacts for the paper

> J. Xu, *Proof-Valid Serving and Benchmarking for Approximate LLM Caches
> under Premise Erasure* (companion to *Partial Path Interception in
> Derivation DAGs: Exact Reliability Laws and a Sharp Complexity Dichotomy*,
> arXiv:2608.11782).

Every number, table, and figure in the paper can be audited against the
retained result artifacts in this repository, and all figures and tables can
be regenerated from them. Unless stated otherwise, all stochastic experiments
use **seed 42** and **20 Monte-Carlo repetitions**. Microsecond timings were
measured on an Apple M3 Pro as a separate memory-resident context-record
microbenchmark (see
`experiments/08_instrumented_pipeline/remeasure_timing_m3pro.json`).

**Reproducibility boundary (snapshot change-detection audit).** The locally
derived per-page JSON inputs of the original 2026 snapshot run are no longer
retained in the authors' local environment. The configuration-level counts,
paired comparison summaries (`constructor_results_v4.json`), and the retained
per-entity outputs (`constructor_results.json`, 1,511-entity panel) are
archived here, and the paper's matched selector×matcher comparison
(`constructor_ablation_2x2.json`) is a re-aggregation of those retained
outputs; end-to-end re-execution of the snapshot extraction pipeline requires
recovering the corresponding historical snapshot inputs and panel mappings.

## Repository layout

```
data/                          Shared inputs for the benchmark experiments
  hotpot_validation.parquet    HotpotQA distractor validation split (7,405 questions)
  question_emb.npy             Question embeddings (all-MiniLM-L6-v2)
  trace_class_sizes.npz        vCache trace class-size distributions (proxy layer)
  musique_full_dev.jsonl       MuSiQue full dev set (4,834 records; 2,417 answerable)
  musique_question_emb.npy     MuSiQue question embeddings
  gptcache_matches.npz         Cached GPTCache match index for the pipeline run
  azure_conv_slice.csv         13.3 h Azure conversation-trace slice (replay input)
experiments/
  01_grounded_audit/           Fig. 2 — footprint audit + grounded stress test
  02_constructor_noise/        Constructor-noise sensitivity (Appendix)
  03_coded_regime/             Fig. 3 — coded regime map, MDS parity, converse band
  04_or_redundancy/            Fig. 9 — loose/strict OR redundancy (bookends)
  05_module_selection/         Fig. 4 — shared-root module selection (HotpotQA + MuSiQue)
  06_gain_curve/               Fig. 10 — coded-advantage gain curve on natural certificates
  07_grounded_operating_points/ Table 2 — grounded operating-point availability
  08_instrumented_pipeline/    Fig. 5 — retrieval grounding, record availability, overhead
  09_correlated_erasure/       Fig. 6 — document-correlated and matched-marginal burst erasure
  10_pareto_frontiers/         Fig. 11 — error–throughput operating curves (1x4 panels)
  11_trace_operating_points/   Tables 5–6 — trace configuration and operating points
                               (SemBenchmark SearchQueries + LmArena)
  12_real_dynamics_audit/      Fig. 8, Table 4 — 2026 Wikipedia/Wikidata
                               real-dynamics audit; Fig. 12, Tables 8–9 —
                               snapshot change-detection audit and constructor frontier
  13_production_trace_replay/  Fig. 7, Table 3 — Azure conversation-trace cost
                               projection and trace-driven single-server queueing check
```

Each experiment folder contains the run script(s), the canonical result
JSON(s) cited in the paper's experiment-inventory table, the figure-replot
script (where the published figure was re-rendered for the final layout), and
the final figure PDF(s) exactly as they appear in the paper.

## Environment

Python 3.10+ with: `numpy pandas scipy matplotlib pyarrow requests`
(plus `sentence-transformers torch` only if embeddings must be regenerated —
all embedding artifacts needed to reproduce the reported numbers are already
included in `data/`).

## Data sources

Included in this repository:

- **HotpotQA** distractor validation split and **MuSiQue** dev set (in
  `data/`; originals: HF `hotpotqa/hotpot_qa`, `dgslibisey/MuSiQue`).
- **WikiFactDiff** 2021-01-04 → 2023-02-27 atomic-fact diff
  (`experiments/12_real_dynamics_audit/data/wikifactdiff_20210104_20230227.parquet`;
  original: the WikiFactDiff release).
- **TemporalWiki probe pages** (`experiments/12_real_dynamics_audit/data/TWiki_Probes.zip`).
- **SemBenchmark traces** (Apache-2.0: HF `vCache/SemBenchmarkSearchQueries`,
  `vCache/SemBenchmarkLmArena`). The 20,000-row trimmed working copies are in
  `experiments/11_trace_operating_points/data/`. The full parquets are *not*
  bundled (12 GB); `_dl_trace_emb.py` refetches the required columns from the
  Hugging Face hub over HTTP range requests.
- **2026 English Wikipedia / Wikidata snapshots** (2026-04-01 … 2026-08-01):
  public Wikimedia dumps, re-downloaded by
  `experiments/12_real_dynamics_audit/fetch_api.py` and the
  `wd_pipeline*.py` / `rebuild_*.py` chain. The derived per-entity statement
  diff and audit results are included as JSON.

## Experiment-to-paper map

| Paper item | Folder | Main script(s) | Canonical result JSON(s) |
|---|---|---|---|
| Fig. 2 grounded audit + stress | `01_grounded_audit` | `run_grounded_experiment.py`, `measure_overhead.py` | `results_grounded.json`, `overhead_results.json` |
| Constructor noise (Appendix) | `02_constructor_noise` | `run_constructor_noise_experiment.py` | `e13_constructor_noise_results.json`, `e13_unbalanced_sensitivity.json` |
| Fig. 3 coded regime + converse | `03_coded_regime` | `run_coded_regime_experiment.py` | `coded_regime_results.json` |
| Fig. 9 OR redundancy | `04_or_redundancy` | `run_or_redundancy_strict_experiment.py`, `compute_exact_family_agreement.py` | `e11_or_strict_results.json`, `e11_exact_family_agreement.json` |
| Fig. 4 module selection | `05_module_selection` | `run_module_selection_experiment.py`, `run_module_selection_musique_experiment.py`, `verify_closed_form.py` | `module_selection_results.json`, `e12_module_musique_results.json`, `module_closed_form_verification.json` |
| Fig. 10 gain curve | `06_gain_curve` | `run_gain_curve_experiment.py`, `run_natural_context_curve_experiment.py` | `gain_curve_results.json`, `e6b_natural_context_results.json` |
| Table 2 grounded operating points | `07_grounded_operating_points` | `run_grounded_operating_experiment.py` | `grounded_operating_results.json` |
| Fig. 5 instrumented pipeline | `08_instrumented_pipeline` | `embed_docs_sliced.py` (doc embeddings), `run_pipeline_experiment.py`, `run_retrieval_grounding_experiment.py`, `remeasure_timing_m3pro.py`, `measure_gate_intact_m3pro.py` | `e8_pipeline_results.json`, `remeasure_timing_m3pro.json`, `remeasure_gate_intact_m3pro.json`, `e8_timings_m3pro.npz` |
| Fig. 6 correlated erasure | `09_correlated_erasure` | `run_correlated_erasure_experiment.py`, `run_correlated_erasure_matched.py` | `e9_correlated_results_matched.json` (canonical; `e9_correlated_results.json` is the earlier unmatched burst arm) |
| Fig. 7 + Table 3 production-trace projection + queueing | `13_production_trace_replay` | `replay_azure_trace.py` | `azure_replay_results.json` |
| Fig. 11 operating curves | `10_pareto_frontiers` | `run_pareto_frontier_experiment.py` | `e10_pareto_results.json` |
| Tables 5–6 trace operating points | `11_trace_operating_points` | `run_experiment.py`, `run_supplement_sq.py` | `trace_endpoints_results.json`, `trace_endpoints_sq_results.json` |
| Fig. 8 + Table 4 real dynamics | `12_real_dynamics_audit` | `eps_t.py`, `wd_pipeline.py`, `wd_pipeline_v2.py`, `module1_wikifactdiff.py`, `module2_analyze.py`, `make_figures_eps_v2.py` | `eps_t.json`, `eps_stmt_v2.json`, `wfd_popularity_workload.json`, `workload_corrected_2026.json` |
| Fig. 12 + Tables 8–9 gate-audit operating points and ablation | `12_real_dynamics_audit` | `constructor_lab.py` … `constructor_lab4.py`, `compute_matched_ablation.py`, `make_frontier_fig.py`, `analyze_roc.py`, `roc_close.py` | `constructor_results_v4.json`, `constructor_results.json` (per-entity), `constructor_ablation_2x2.json`, `statement_diff_v2_2026.json` |

## Naming notes (please read before citing artifact files)

- `trace_endpoints_results.json` / `trace_endpoints_sq_results.json` are the
  canonical names used in the paper. They are, respectively, the LmArena main
  run summary (static quantile 0.9) and the supplementary SearchQueries run
  summary (static quantile 0.70); the paper's endpoint table reads LmArena
  rows from the former and SearchQueries rows from the latter.
- The joined 1,511-entity gold-standard panel (proxy labels per entity) is
  not stored as a separate file: its per-entity constructor outputs are
  retained in `constructor_results.json`, its configuration-level aggregate
  statistics are exactly `constructor_results_v4.json` (130 proxy-positive /
  1,381 proxy-negative entities), and the proxy labels derive from
  `statement_diff_v2_2026.json` (per-entity removed-statement diffs).
  Rebuilding the panel itself additionally requires the historical snapshot
  page inputs, which are no longer retained locally (see the reproducibility
  boundary above).
- Figure-replot scripts (`replot_*.py`, `make_figures_eps_v2.py`,
  `make_frontier_fig.py`) only re-render published figures from the archived
  JSONs; they never recompute measurements.

## GitHub size note

`experiments/11_trace_operating_points/data/SemBenchmarkLmArena_trimmed.parquet`
is 101 MB, marginally above GitHub's 100 MB per-file limit — track it with
Git LFS, or exclude it and regenerate with `_dl_trace_emb.py`. All other
files are below the limit. The full repository is intended to be archived on
Zenodo (or another DOI minting service) as-is.

## License

The SemBenchmark trace data are redistributed under their original
Apache-2.0 license; HotpotQA, MuSiQue, WikiFactDiff, TemporalWiki, and
Wikimedia dump derivatives remain under their respective original licenses.
Scripts and derived result JSONs are released for academic reproduction of
the paper.

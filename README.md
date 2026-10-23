# Proof-Valid Serving and Benchmarking for Approximate LLM Caches — Experiment Artifact

This repository contains the complete experimental programs, data, and result
artifacts for the paper

> J. Xu, *Proof-Valid Serving and Benchmarking for Approximate LLM Caches
> under Premise Erasure* (companion to *Partial Path Interception in
> Derivation DAGs: Exact Reliability Laws and a Sharp Complexity Dichotomy*,
> arXiv:2608.11782).

**Permanent archive:** this repository is archived at Zenodo,
[doi:10.5281/zenodo.23502164](https://doi.org/10.5281/zenodo.23502164)
(the concept DOI always resolves to the latest version).
The tree corresponding to the KAIS submission is tagged
`kais-submission`; the Zenodo record archives that same tree.
Section and table numbers below refer to the KAIS submission version of the
manuscript.

Every number, table, and figure in the paper can be audited against the
retained result artifacts in this repository, and all figures and tables can
be regenerated from them. Unless stated otherwise, all stochastic experiments
use **seed 42** and **20 Monte-Carlo repetitions**. Microsecond timings were
measured on an Apple M3 Pro as a separate memory-resident context-record
microbenchmark (see
`experiments/08_instrumented_pipeline/remeasure_timing_m3pro.json`).

**Reproducibility of the snapshot audits.** The 2026 snapshot inputs
have been rebuilt end to end: all five English Wikipedia page snapshots,
four Wikidata entity views, the title↔QID panel mappings, and the
WikiFactDiff legacy split were re-fetched from public Wikimedia mirrors
and re-derived with the archived extraction code. Every paper-cited
count from the original run reproduces **exactly** from the rebuilt
inputs (sentence- and claim-level change rates, the 1,511-entity
constructor table, and the detection operating points including the
paired discordant counts). The rebuilt inputs and the reconciliation
report live in `experiments/12_real_dynamics_audit/data/rebuilt_2026/`
(see `RECONCILIATION.md` in the same folder tree), so the snapshot
pipeline is now re-executable end to end.

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
  01_grounded_audit/           footprint audit + grounded stress test
  02_constructor_noise/        constructor-noise sensitivity
  03_coded_regime/             coded regime map, MDS parity, converse band
  04_or_redundancy/            loose/strict OR redundancy (bookends)
  05_module_selection/         shared-root module selection (HotpotQA + MuSiQue)
  06_gain_curve/               coded-advantage gain curve on natural certificates
  07_grounded_operating_points/ grounded operating-point availability
  08_instrumented_pipeline/    retrieval grounding, record availability, overhead
  09_correlated_erasure/       document-correlated and matched-marginal burst erasure
  10_pareto_frontiers/         error–throughput operating curves (1x4 panels)
  11_trace_operating_points/   trace configuration and operating points
                               (SemBenchmark SearchQueries + LmArena)
  12_real_dynamics_audit/      2026 Wikipedia/Wikidata real-dynamics audit;
                               snapshot change-detection audit and constructor
                               frontier; data/rebuilt_2026/ — rebuilt snapshot
                               inputs (pages ×5, entities ×4, panel mappings,
                               statement diff) + RECONCILIATION.md + rebuild scripts
  13_production_trace_replay/  Azure conversation-trace cost projection
                               and trace-driven single-server queueing check
  14_snapshot_serving_replay/  §7.3 — end-to-end monthly serving replay
                               of the full pipeline on the real
                               2026-04-01 → 2026-08-01 snapshot change (LLM answer
                               generation, certificate gate, two serving arms,
                               closed-loop regeneration, cost decomposition,
                               simple serve-time baselines, protection-recovery
                               overlay + revert negative control; human answer
                               labels included)
  15_second_domain_daily_replay/ §7.2 — main-scenario daily frozen-cache
                               replay: 100 candidate pages (99
                               usable) tracked over 21 daily ticks (2026-07-01 →
                               2026-07-21); lead-sentence certificate premises
                               vs. infobox value truth; dual-generator (Kimi/GPT)
                               caches with real answer generation, fallback
                               regeneration, and double-annotated audits; gate
                               arms vs. TTL/page-hash/infobox-diff baselines;
                               page-cluster bootstrap (stage13) and
                               per-part-filtered replay (stage14); the initial
                               48-page pilot pool is included (stage1b–stage3b)
  16_dag_correctness_test/       §7.1 + Supplementary Section S1 — exhaustive
                               small-schema check of the
                               residual-failure-path lemma — 400 random
                               conjunctive DAG schemas, all 990,560
                               erasure/cache states agree (no model calls)
```

The authoritative experiment→result map, with section and table
cross-references for the KAIS submission, is the inventory in Online
Resource 1, Section S10.

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
| Grounded audit + stress | `01_grounded_audit` | `run_grounded_experiment.py`, `measure_overhead.py` | `results_grounded.json`, `overhead_results.json` |
| Constructor noise (Appendix) | `02_constructor_noise` | `run_constructor_noise_experiment.py` | `e13_constructor_noise_results.json`, `e13_unbalanced_sensitivity.json` |
| Coded regime + converse | `03_coded_regime` | `run_coded_regime_experiment.py` | `coded_regime_results.json` |
| OR redundancy | `04_or_redundancy` | `run_or_redundancy_strict_experiment.py`, `compute_exact_family_agreement.py` | `e11_or_strict_results.json`, `e11_exact_family_agreement.json` |
| Module selection | `05_module_selection` | `run_module_selection_experiment.py`, `run_module_selection_musique_experiment.py`, `verify_closed_form.py` | `module_selection_results.json`, `e12_module_musique_results.json`, `module_closed_form_verification.json` |
| Gain curve | `06_gain_curve` | `run_gain_curve_experiment.py`, `run_natural_context_curve_experiment.py` | `gain_curve_results.json`, `e6b_natural_context_results.json` |
| Grounded operating points | `07_grounded_operating_points` | `run_grounded_operating_experiment.py` | `grounded_operating_results.json` |
| Instrumented pipeline | `08_instrumented_pipeline` | `embed_docs_sliced.py` (doc embeddings), `run_pipeline_experiment.py`, `run_retrieval_grounding_experiment.py`, `remeasure_timing_m3pro.py`, `measure_gate_intact_m3pro.py` | `e8_pipeline_results.json`, `remeasure_timing_m3pro.json`, `remeasure_gate_intact_m3pro.json`, `e8_timings_m3pro.npz` |
| Correlated erasure | `09_correlated_erasure` | `run_correlated_erasure_experiment.py`, `run_correlated_erasure_matched.py` | `e9_correlated_results_matched.json` (canonical; `e9_correlated_results.json` is the earlier unmatched burst arm) |
| Production-trace projection + queueing | `13_production_trace_replay` | `replay_azure_trace.py` | `azure_replay_results.json` |
| Error–throughput operating curves | `10_pareto_frontiers` | `run_pareto_frontier_experiment.py` | `e10_pareto_results.json` |
| Trace operating points | `11_trace_operating_points` | `run_experiment.py`, `run_supplement_sq.py` | `trace_endpoints_results.json`, `trace_endpoints_sq_results.json` |
| Real dynamics | `12_real_dynamics_audit` | `eps_t.py`, `wd_pipeline.py`, `wd_pipeline_v2.py`, `module1_wikifactdiff.py`, `module2_analyze.py`, `make_figures_eps_v2.py` | `eps_t.json`, `eps_stmt_v2.json`, `wfd_popularity_workload.json`, `workload_corrected_2026.json` |
| Gate-audit operating points and ablation | `12_real_dynamics_audit` | `constructor_lab.py` … `constructor_lab4.py`, `compute_matched_ablation.py`, `make_frontier_fig.py`, `analyze_roc.py`, `roc_close.py` | `constructor_results_v4.json`, `constructor_results.json` (per-entity), `constructor_ablation_2x2.json`, `statement_diff_v2_2026.json` |
| §7.3 monthly snapshot serving replay | `14_snapshot_serving_replay` | `s1_build_panel.py` … `s52_claim_equality_loop_alias.py` (see `RESULTS.md`) | `replay_summary.json`, `replay.json`, cached `generation.json` and `embeddings.json`, `age_curve.json`, `simple_baselines.json`, `protection_recovery.json`, `current_claims/`, `annotation/` (incl. `incoming_query_labels.json`), `incoming_query_crosstab.json`, `alternative_certificates.json`, `qualified_alternatives.json`, `fixed_budget_protection.json`, `retrieval_scaling.json`, `unified_strategy_table.json`, `paired_transition.json`, `population_reweight.json`, `population_reweight_schemeA.json`, `closed_loop_system.json`, `sensitivity_ttl_vs_loop.json`, `protected_closed_loop.json`, `staged_timing_failclosed.json`, `certificates_handoff.jsonl`, `certificate_handoff.json`, `real_payload_chain.json`, `payload_stores.sqlite`, `frozen_independent_check.json`, `tableb_regeneration.json`, `tableb_endpoint_map.json`, `tableb_labels.json`, `tableb_endpoint_quality.json`, `tableb_protected_loop.json`, `real_driven_loop.json`, `real_driven_sweep.json` (Decimal re-aggregated; pre-correction backup `real_driven_sweep_binary_rounding_backup.json`), `residual_decomposition.json`, `paired_transitions_real.json`, `frozen_panel_generation.json`, `frozen_panel_check.json`, `tableb_regeneration_gpt.json`, `regeneration_gpt.json`, `gpt_generator_compare.json` (second-generator replication, GPT via Poe API), `mechanism_arms.json` (mechanism-level comparison arms), `gate_alias_cross.json` (crossed gate-by-alias control), `gate_alias_request_diff.json` (per-request alias toggle diff), `annotation/pairing_audit.json` (second-generator answer-pairing audit), `claim_equality_loop.json` (claim-level equality ablation arm, aliases off), `claim_equality_loop_alias.json` (same ablation, aliases on) |
| §7.2 daily frozen-cache replay (main scenario, dual-generator) | `15_second_domain_daily_replay` | `stage1c_fetch_expanded.py`, `stage1d_static_fill.py`, `stage2e_build_panel_prox.py`, `stage0_generate_answers.py`, `stage4_dual_replay.py`, `stage5_fallback_regen.py`, `stage6_load_cost.py`, `stage7_mechanism_arms21.py`, `stage8_spotcheck_sheet.py`, `stage9_parse_spotcheck.py` | `series21.json`, `panel21.json`, `pool_stats21.json`, `answers_kimi.json`, `answers_gpt.json`, `dual_replay_results.json`, `fallback_regen.json`, `load_cost.json`, `mechanism_arms21.json`, `fallback_spotcheck_sample20.json`, `fallback_spotcheck_results.json` (double-annotated fallback spot check: agreement 19/20, Cohen's κ=0.90, adjudicated correct 9/20 [25.8, 65.8]), `perpart_rescore.json` (per-part re-scoring: regenerations 24/41 vs 37/41 set-level, 7 of 10 false accepts rejected, base answers flagged 16/62 Kimi, 17/68 GPT), `base_answer_sample40.json`, `base_answer_results.json` (double-annotated base-answer audit: agreement 40/40, κ=1.0, design-weighted b = 22.8% [20.6, 25.1]; Eq.(window) verdict: r−b ≈ 32 points exceeds the upper bound vs similarity-only, clears the TTL-1d lower bound by two orders of magnitude), `annotation/` (returned annotator sheets) |

## Naming notes (please read before citing artifact files)

- **Result precedence.** Primary erasure-free endpoint quality uses the
  label-aware endpoint fields in `real_driven_loop.json`;
  `tableb_endpoint_quality.json` retains the annotation-agreement and
  bootstrap blocks, but its per-arm endpoint counts are superseded by the
  real-driven accounting. Replacement-generator endpoint delivery uses
  `gpt_reexec_endpoint.json`. Fault routing, p95 values, and guard counts
  use `real_driven_sweep.json`; label-aware fault quality uses
  `real_driven_loop.json` (its `endpoint_correct_mean`-style fields in the
  sweep file are routing aggregates, not human-labeled quality). Daily
  quality projections are recomputed from retained item labels, acceptance
  decisions, and routing counts; the legacy confidence intervals and the
  embedded `window_verdict` field in `base_answer_results.json` are
  superseded and not authoritative.
- **Multiplicity-preserving page-cluster bootstrap.** `stage13`/`stage14`
  key each bootstrap draw by sampling position, so a page drawn repeatedly
  contributes its counts repeatedly. The current
  `pagecluster_bootstrap.json` and `perpart_filtered_replay.json` were
  regenerated under this implementation (10,000 resamples, seed 42); the
  pre-fix outputs are kept as `*_premultfix_backup.json` for provenance and
  must not be cited. `stage15_zero_denominator_check.py` sums served counts
  over the draw list and audits the same convention.
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
  The panel and all derived counts rebuild exactly from the snapshot inputs
  in `data/rebuilt_2026/` (see the reproducibility note above).
- Experiment 14 caches all LLM answers and embeddings
  (`generation.json`, `embeddings.json`); `s5_replay.py` therefore
  recomputes every reported replay metric deterministically. Regenerating
  the answers themselves requires an OpenAI/Anthropic-compatible LLM
  endpoint (the paper's run used `kimi-k2-turbo-preview` for generation and
  `bge-m3` embeddings via an API gateway). The second-generator replication
  (`s41_gpt_regenerate.py`, `s42_gpt_compare.py`) likewise replays
  deterministically from `tableb_regeneration_gpt.json`; re-running the GPT
  generation itself requires a Poe API key in `POE_API_KEY` (the paper's
  run used `gpt-5-mini`).
- Figure-replot scripts (`replot_*.py`, `make_figures_eps_v2.py`,
  `make_frontier_fig.py`) only re-render published figures from the archived
  JSONs; they never recompute measurements.

## GitHub size note

Two files exceed GitHub's 100 MB per-file limit and are tracked with
Git LFS (see `.gitattributes`):
`experiments/11_trace_operating_points/data/SemBenchmarkLmArena_trimmed.parquet`
(101 MB; regenerate with `_dl_trace_emb.py`) and
`experiments/12_real_dynamics_audit/data/wikifactdiff_20210104_20230227.parquet`
(178 MB; the complete, untruncated WikiFactDiff legacy split — re-download
from the WikiFactDiff release). All other files, including the rebuilt 2026
snapshot inputs, are below the limit. The full repository is archived
as-is on Zenodo
([doi:10.5281/zenodo.23502164](https://doi.org/10.5281/zenodo.23502164)).

## License

The SemBenchmark trace data are redistributed under their original
Apache-2.0 license; HotpotQA, MuSiQue, WikiFactDiff, TemporalWiki, and
Wikimedia dump derivatives remain under their respective original licenses.
Scripts and derived result JSONs are released for academic reproduction of
the paper.

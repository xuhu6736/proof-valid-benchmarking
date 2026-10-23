# Experiment 15 — Second-domain daily replay with source-versioned truth

Forty-eight English Wikipedia pages (World Cup players and national teams,
tournament and season pages, stable controls) tracked over 21 daily ticks,
2026-07-01 → 2026-07-21. Certificate premise = the t0 lead sentence stating
an infobox value; endpoint truth = whether the same infobox key still holds
the t0 value at tick k. The pilot pipeline (stages 1b–3b) uses deterministic
labels (normalized value equality) with no model calls and no human
annotation; the dual-generator extension below (stages 0–15) adds real
Kimi/GPT answer generation, fallback regeneration, and double-annotated
audits over the expanded 100-page pool.

## Pipeline

1. `stage1b_fetch_series21.py` — fetch boundary revisions per tick via the
   Wikimedia API; writes `series21.json` (per page, per tick: cleaned infobox
   key-values, lead plain text, revision hash). Checkpointed per page;
   re-running skips completed pages.
2. `stage2b_build_panel21.py` — build the claim pool (t0 infobox values that
   also appear in a t0 lead sentence), group three same-page claims per
   answer, record per-tick value validity and sentence survival; writes
   `panel21.json` (66 answers / 198 claims, all eligible answers included,
   seed 42) and `pool_stats21.json` (pool-level supersession curve).
3. `stage3b_replay21.py` — deterministic replay of gate arms
   (similarity-only, sentence-asserted, sentence-strict, infobox-any-key,
   page-hash, TTL-1d/3d/7d/14d) over the 1,320 answer-tick cases; writes
   `replay_results21.json` (stale delivery, interception, regeneration share,
   false rejection given valid, Wilson intervals, regenerations per avoided
   stale delivery).

## Dual-generator extension (A-1/A-2/A-3)

4. `stage1c_fetch_expanded.py` + `stage1d_static_fill.py` — expand the
   pool to 100 candidate pages (99 usable; 12 zero-edit pages filled
   with their exact t0 boundary revision and flagged
   `static_in_window`).
5. `stage2e_build_panel_prox.py` — keyword- and proximity-constrained
   claim anchoring (numeric claims need a key-topic keyword within 8
   tokens of the value); writes the 99-answer / 297-claim `panel21.json`.
6. `stage0_generate_answers.py kimi|gpt` — template one 3-part question
   per answer and generate the t0 cached answer with Kimi
   (`kimi-k2-turbo-preview`) and GPT (`gpt-5-mini` via Poe);
   lexical value-mention acceptance -> `answers_kimi.json`
   (62/99 accepted), `answers_gpt.json` (68/99).
7. `stage4_dual_replay.py` — dual-generator replay over the two real
   caches with paired cluster bootstrap (10,000 resamples, seed 42)
   -> `dual_replay_results.json`.
8. `stage5_fallback_regen.py` — same-generator real regeneration at
   each cached answer's first asserted-gate rejection, judged against
   tick-k truth -> `fallback_regen.json` (Kimi 17/19, GPT 20/22
   correct) and a 20-row human spot-check sheet
   (`PVSB_主场景回退抽检表.docx`).

9. `stage6_load_cost.py` — measured component costs (gate check
   microseconds, regeneration call seconds, source re-read seconds)
   and the per-1000-services expected-cost model
   -> `load_cost.json`.

10. `stage7_mechanism_arms21.py` — mechanism-level baseline arms on the
   dual caches (GroundedCache-style overlap, FinCacheServe-style
   equality, claim-level equality ablation; S5 mapping conventions)
   with paired bootstrap -> `mechanism_arms21.json`.
11. `stage8_spotcheck_sheet.py` — regenerate the seed-42 twenty-item
   human spot-check sample of fallback regenerations and emit the
   double-annotation sheets (auto-judgment withheld, four-level
   checkboxes) -> `fallback_spotcheck_sample20.json` (master ledger
   with auto-judgments) and `PVSB_主场景回退标注表_20条（人工1/2）.docx`
   at the workspace root.
12. `stage9_parse_spotcheck.py` — parse the two returned sheets, audit
   inter-annotator agreement (19/20 raw, Cohen's kappa 0.90),
   adjudicate the single disagreement conservatively, and emit
   `fallback_spotcheck_results.json`: human-adjudicated correct 9/20
   (45.0%, Wilson 95% [25.8, 65.8]) versus lexical value-matching
   acceptance 19/20 in the sample (10 false accepts).  Returned sheets
   are archived under `annotation/`.
13. `stage10_perpart_rescore.py` — per-part value-matching re-scoring of
   the archived answers, no model calls -> `perpart_rescore.json`:
   regenerations accepted 24/41 per-part versus 37/41 set-level; seven
   of the ten spot-check false accepts rejected (one false reject, a
   surface-form variant); base answers flagged 16/62 (Kimi) and 17/68
   (GPT); per-part agrees with the human labels on 16/20 spot-check
   items versus 10/20 for set-level.
14. `stage11_base_answer_sheet.py` — stratified 40-answer sample of
   accepted base answers (10 per-part-flagged + 10 clean per generator,
   seed 42, design weights recorded) and double-annotation sheets ->
   `base_answer_sample40.json` and
   `PVSB_主场景基缓存答案标注表_40条（人工1/2）.docx` at the workspace
   root.
15. `stage12_parse_base_annotations.py` — parse the two returned
   base-answer sheets, audit agreement (40/40 raw, kappa 1.0),
   adjudicate conservatively, and emit `base_answer_results.json`:
   design-weighted b = 22.8% [20.6, 25.1] (wrong-only 5.0% [2.2, 7.8]);
   per-part flagging matches human labels on 38/40; the Eq.(window)
   verdict: excess error r-b ~ 32 points exceeds the daily panel's
   upper bound (17.4 Kimi / 15.0 GPT) while clearing the TTL-1d lower
   bound (0.08) by two orders of magnitude.  Returned sheets archived
   under `annotation/`.

Note: `stage1b/1c/1d` require network access to the Wikimedia API.
Stages 2-4 are purely local and deterministic.
16. `stage13_pagecluster_bootstrap.py` — page-cluster bootstrap
    re-analysis of every headline paired contrast (bootstrap unit =
    Wikipedia page, since answers on one page share its edit process;
    42 Kimi / 47 GPT pages, 41 shared), no model calls ->
    `pagecluster_bootstrap.json`: B1' cross-generator stale-delivery
    difference paired on the 41 shared pages [-0.01,+0.02] pp;
    B2'/B3' and the GC-overlap/FC-equality regeneration contrasts are
    slightly wider than the answer-level intervals with no conclusion
    changed (case counts verified identical to stages 4/7).
17. `stage14_perpart_filtered_replay.py` — restricted-cache replay:
    routing re-run after restricting each base cache to the entries
    retained by per-part matching (46/62 Kimi, 51/68 GPT), no model
    calls -> `perpart_filtered_replay.json`: asserted-gate fallback
    demand 16.7%/17.1%, rejection precision 23.4%/25.9%,
    served-conditional stale rates 0.13%/0.12%, with the paired
    page-cluster TTL contrasts retained.
18. `stage15_zero_denominator_check.py` — bootstrap zero-denominator
    audit for every reported paired contrast (10,000 resamples,
    seed 42) -> `zero_denominator_check.json`: no resample draw
    produces an empty denominator in either arm of any contrast.

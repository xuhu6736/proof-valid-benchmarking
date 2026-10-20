# Experiment 14 — Snapshot Serving Replay (full Fig. 1 architecture on real data)

**Window:** enwiki/Wikidata snapshots 2026-04-01 (base) → 2026-08-01 (target),
from the rebuilt, reconciled snapshots in
`../12_real_dynamics_audit/data/rebuilt_2026/`.

**Purpose.** The first end-to-end *serving-level* evaluation of the PVSB
architecture (companion paper Fig. 1): an answer cache whose entries carry
certificates, replayed against a real 4-month erasure event, compared against
a similarity-only baseline (GPTCache-style).

## Protocol

1. **Panel** (`s1_build_panel.py`): 32 templated factoid properties
   (occupation, genre, located-in, award, population, dates, …).
   A QA pair = (entity, property); certificate statements = the base-period
   truthy values of that property. A pair is *stale* iff ≥1 cited value is no
   longer truthy at target (premise-erasure semantics; additions do not count).
   Panel = **all 52 stale pairs + 104 matched fresh controls** (seed 42),
   154 usable after answer validation, covering 143 distinct entities.
2. **Labels** (`s2_scan_labels.py`, `s2b_fetch_labels.py`): EN labels for the
   251 distinct value QIDs resolved from the same two dumps via the
   multistream index + HTTP Range block fetches (248/251 base, 247/250 target;
   missing labels fall back to the QID string).
3. **Evidence + certificate** (`s3_evidence.py`): up to 3 base-page sentences
   mentioning a cited value (constructor-identical normalisation), plus the
   structured claims. Mean certificate: 2.79 claims + 2.46 sentences.
4. **Generation** (`s4_generate.py`): LLM (Kimi K2 Turbo, thinking disabled,
   temperature default) answers each question strictly from the provided
   evidence; the answer is accepted only if it names ≥1 cited value
   (154/156 accepted). A paraphrase of each question is generated for the
   similarity gate.
5. **Replay** (`s5_replay.py`): queries (paraphrases) hit the cache;
   similarity = cosine over `bge_m3_embed` embeddings (1024-d);
   τ calibrated to the 5th percentile of genuine-paraphrase similarities
   (τ = 0.8364). The certificate gate re-verifies every cited claim against
   the target-period truthy view and every cited sentence (sha1 of the
   normalised sentence) against the target page.

## Headline results (147 cache hits at τ = 0.8364)

Dependency-level routing outcomes. "Stale" below means
*dependency-stale*: ≥1 designated base claim value absent from the
target projected truthy view — **not** an independently verified
answer error. Because the stale label and the claim gate are evaluated
on the same projected claim keys, gated policies declining all
claim-stale hits is the expected behaviour of a correct
implementation (contract consistency), not an independent detection
finding. The replay instantiates the answer cache \(K\) of the
companion Fig. 1 with **S = ∅**: designated claims and sentences are
audited source-premise dependencies, not protection-store items.

| arm | served | stale served | stale share of served | intact hits declined |
|---|---|---|---|---|
| similarity-only baseline | 147 | 43 | 29.3% [22.5, 37.1] | – |
| strict dependency gate (claims + all sentences) | 89 | 0 | 0% [0, 4.1] | 15/104 (14.4%) [8.9, 22.4] |
| claims + majority sentences | 96 | 0 | 0% | 8/104 (7.7%) [3.9, 14.4] |
| claims only | 104 | 0 | 0% | 0/104 |

- The three gate variants are *different dependency contracts*, not
  parameter settings of one contract.
- All 15 strict-gate declines on claim-intact hits are triggered by
  the sentence-identity requirement (designated claims survive in
  every case); string absence alone proves neither pure rewording nor
  answer invalidity. On this claim-labelled panel, sentence-identity
  requirements add declines without adding interceptions — the price
  of a stricter contract, not evidence that sentences are redundant.
- Stratified hit rates: stale 43/50 (86.0%) vs intact 104/104 (100%);
  all 7 similarity misses fall in the stale stratum.
- Fallback counts (65 strict / 58 majority / 50 claims-only not
  cache-served out of 154) are potential regeneration demand, not
  executed regenerations.

## Files

```
scripts/  s1_build_panel.py s2_scan_labels.py s2b_fetch_labels.py
          s3_evidence.py s4_generate.py s5_replay.py
data/     panel.json label_qids.json label_hits_*.json labels_*.json
          evidence.json generation.json embeddings.json
          replay.json replay_summary.json
```

## Reproducibility notes

- Requires the rebuilt snapshots (experiment 12, `data/rebuilt_2026/`).
- LLM/embeddings: any OpenAI/Anthropic-compatible endpoint works; the run
  used Kimi gateway (`kimi-k2-turbo-preview`, `bge_m3_embed`). All generations
  are cached in `data/generation.json` / `data/embeddings.json`, so the replay
  metrics (`s5_replay.py`) recompute deterministically without further calls.

## Answer-level human annotation (2026-10-16)

All 154 cached entries were annotated blind at the answer level by one
author (sheet: `annotation/PVSB_annotation_sheet.docx`; parser:
`scripts/parse_annotations.py`; labels: `annotation/annotations_parsed.json`,
`annotation/answer_labels.json`). Five questions per entry: A1 base
correctness, A2 evidence support, A3 paraphrase fidelity, A4 target
correctness (vs. target-period values + sentence survival), A5
attribution (invalidates / partial / no effect / no change /
indeterminate).

Headline counts:

- A1: 141 correct, 1 wrong, 7 incomplete, 5 indeterminate.
- A2: 109 fully supported, 45 partial (sentences do not mention every
  designated claim).
- A3: 151 faithful; 3 unfaithful (#8, #52, #95 swap the relation).
- A4: 126 correct, 3 wrong, 18 incomplete, 7 indeterminate.
- A5: 3 invalidates, 6 partial, 54 no effect, 86 no change, 5
  indeterminate.
- Decayed answers (wrong, or change-induced partial): 9/154, all in the
  dependency-stale arm, 8 of them similarity hits. Conversely 33/50
  dependency-stale entries (66%) remain fully correct at the answer
  level — dependency staleness over-flags answer decay by an order of
  magnitude.

Answer-level gate outcomes over the 147 similarity hits (labels: 3
wrong, 5 partial, 133 correct, 6 indeterminate):

| policy | served | decayed served | correct declined | indet declined |
|---|---|---|---|---|
| similarity-only | 147 | 8 (3+5) | 0/133 | 0 |
| strict gate | 89 | 0 | 45/133 (33.8% [26.3,42.2]) | 5 |
| claims+majority | 96 | 0 | 39/133 (29.3% [22.3,37.6]) | 4 |
| claims only | 104 | 0 | 31/133 (23.3% [16.9,31.2]) | 4 |

All three contracts intercept all 8 decayed hits; sentence conjuncts
add no interception and 8–14 extra still-correct declines. Similarity
only: 5.4% decayed served [2.8,10.4], 2.0% outright wrong [0.7,5.8].

## Package 1 audits (2026-10-16)

**Unified cache retrieval** (`s6_unified_retrieval.py` →
`data/unified_retrieval.json`): full 154×154 query–question cosine
matrix from cached embeddings. At τ=0.8364 every query's top-1 match is
its own entry: 147 accepted, 0 misrouted, 7 missed — identical
decisions to the paired check. 5 queries have an impostor above τ but
the genuine entry always ranks first (min margin 0.0708, p5 0.1188,
median 0.4518).

**Out-of-sample τ** (`s7_tau_oos.py` → `data/tau_oos.json`):
entity-grouped 5-fold CV (seed 42), τ = 5th percentile of train genuine
sims. Fold τ ∈ [0.8384, 0.8706]; held-out paired hits 144/154 (93.6%,
worst fold 86.7%); unified retrieval on held-out entries: 144 correct,
0 misrouted, 10 missed.

**Cluster bootstrap** (`s8_cluster_bootstrap.py` →
`data/cluster_bootstrap.json`): 10,000 entity-resamples (seed 42) over
the 147 hits. Point [nominal Wilson] → cluster-boot95:

- dep-stale share, similarity-only: 29.3% → [20.7, 37.6]
- strict intact declines (dep level): 14.4% → [7.9, 21.4]
- decayed served share (answer level): 5.4% → [1.4, 10.6]
- correct-hit declines: strict 33.8% → [25.2, 42.4]; majority 29.3% →
  [20.9, 38.0]; claims-only 23.3% → [15.6, 31.2]

**Closed-loop regeneration** (`s9_regenerate.py` →
`data/regeneration.json`, `data/closed_loop.json`): all 65 requests not
served under the strict contract re-answered from target-snapshot
evidence (target truthy values + ≤3 target-page sentences; labels from
the two dump scans + `data/labels_extra.json`, 17 QIDs resolved via the
Wikidata API). 51/65 obtain a validated fresh answer (value-mention
acceptance), 14 abstain honestly (property emptied in the target view),
0 generation failures. All 9 decayed entries regenerate answers that
differ from the cache and cite current values. Final delivery under the
strict contract: 140/154 validated current answers (89 cached: 88
correct + 1 indeterminate; 51 regenerated), 14 abstentions, 0 decayed
served. 6 of 7 similarity misses abstain — their answers survive only
in Wikipedia sentences while the Wikidata property was emptied
(claim-sourced regeneration cannot repair this asymmetry).

## False-decline cost decomposition (2026-10-16, s10)

`scripts/s10_cost_decomposition.py` → `data/age_curve.json`,
`data/asserted_claims.json`, `data/fig_cache_age.pdf` (copied to the
paper `fig/`).  No model calls; reuses rebuilt intermediate snapshots
(2026-05-01, 2026-07-01).

**A. Prevalence reweighting (4-month endpoint).** Natural pair-level
staleness prevalence 52/6580 = 0.79%; panel enrichment ≈ 37×.
Population decline rates: strict 15.1%, majority 8.4%, claims-only
0.79%, asserted 0.68%. Estimated answer-decay prevalence ≈ 0.14% of
requests (0.79% × 9/50).

**B. Cache-age response.** Staleness onset among the 50
eventually-stale entries: 0% / 48% / 96% / 100% at 0/1/3/4 months.
Intact-conditional decline rates (enrichment-free; the 104 controls
are a random intact sample): strict 0 / 4.8 / 12.5 / 14.4%; majority
0 / 1.0 / 5.8 / 7.7%. Population decline-rate estimates: 1 month —
strict 5.2%, majority 1.3%, claims 0.4%, asserted 0.2%.

**C. Asserted-claims contract.** Answers name 1.69 of 2.79 designated
claims on average. Gating only answer-asserted claims at the endpoint:
serves 111 hits (7 stale hits whose dead claims are unasserted — all
7 annotated still correct / no effect), declines 24/133 still-correct
hits (18.0% [12.4, 25.4]), intercepts all 9 decayed entries. Dominates
the claims-only contract on both axes (104 served, 23.3% declined).
TTL reference: blind refresh of the 4-month cache regenerates 100%.

## Simple serve-time baselines (2026-10-16, s11)

Same 147 post-similarity hits and the same human answer-level labels
(8 decayed hits / 133 still-correct / 6 unjudged). Script:
scripts/s11_simple_baselines.py; outputs data/simple_baselines.json
and data/llm_judge.json.

| baseline                     | served | decayed served | still-correct declined |
|------------------------------|--------|----------------|------------------------|
| page-change (page text hash) | 43     | 1              | 92/133 (69%)           |
| entity-change (claims JSON)  | 56     | 0              | 79/133 (59%)           |
| prop-change (truthy props)   | 100    | 0              | 35/133 (26%)           |
| value-in-page                | 97     | 3              | 41/133 (31%)           |
| llm-recheck (LLM judge)      | 129    | 3              | 10/133 (7.5%)          |
| PVSB asserted-claims         | 111    | 0              | 24/133 (18%)           |

LLM-judge verdict distribution: 129 YES / 10 NO / 8 NO_EVIDENCE
(counted as declines) / 0 parse failures.

Missed-decay details:
- page-change misses #125 (Helen Deutsch): Wikidata-only decay, page
  untouched — page checks are blind to Wikidata-only decay.
- value-in-page misses #24, #48, #78.
- llm-recheck misses #24 (judged YES), #78, #125; #48/#50 judged NO.
- prop-change declines exactly 4 more entries than claims-only:
  #21/#60/#109/#119 — all pure-addition entries annotated correct /
  no effect, confirming the gate's addition-tolerant design.

## Protection recovery + revert negative control (2026-10-16, s12)

Script: scripts/s12_protection_recovery.py; output
data/protection_recovery.json; raw revert refetches in
data/current_claims/ (Wikidata wbgetclaims, 2026-10-16).

**Arm 1 — read-erasure overlay on the 147 hits.** Each certificate
item's read lost independently with prob p; fraction c protected by
transparent re-audited copies; per-hit survival (1-p(1-c))^n.
Expected serves (baseline without erasure: 111 asserted / 89 strict;
both serve zero decayed hits):

| contract | p    | no prot | c=0.5 | c=0.9 | c=0.95 | c=1.0 |
|----------|------|---------|-------|-------|--------|-------|
| asserted | 0.01 | 109.3   | 110.1 | 110.8 | 110.9  | 111   |
| asserted | 0.05 | 102.9   | 106.8 | 110.1 | 110.6  | 111   |
| asserted | 0.10 | 95.6    | 102.9 | 109.3 | 110.1  | 111   |
| asserted | 0.20 | 82.1    | 95.6  | 107.6 | 109.3  | 111   |
| strict   | 0.01 | 85.2    | 87.1  | 88.6  | 88.8   | 89    |
| strict   | 0.05 | 71.7    | 79.8  | 87.1  | 88.0   | 89    |
| strict   | 0.10 | 58.0    | 71.7  | 85.2  | 87.1   | 89    |
| strict   | 0.20 | 37.7    | 58.0  | 81.5  | 85.2   | 89    |

Mean gated items per served hit: 1.59 (asserted) vs 4.43 (strict) —
certificate size is the availability multiplier.  Seeded Monte Carlo
(2000 runs, seed 7) matches analytic expectations within its 95%
percentile band.  Copy-trust negative control: restoring base-period
copies WITHOUT the currency audit serves 147/147 incl. all 8 decayed
hits — unaudited protection is staleness laundering.

**Arm 2 — revert control (real data).** Current claim state of all
nine decayed (entity, property) pairs refetched 2026-10-16 (ten
weeks after the target snapshot).  Surviving answer-asserted claims:
pid24 0/1, pid33 3/5, pid48 0/1, pid50 0/1, pid78 2/5, pid107 2/4,
pid112 1/2, pid125 4/5, pid51 2/3 — NO entry fully reverted, so all
nine remain correctly declined; partial restoration cannot resurrect
an entry under the conjunctive gate.  Gate verdicts are pure
functions of current evidence, so a genuine revert would be
re-accepted with no bookkeeping (change-log invalidation is one-way).

## Two-axis label cross-tabulation (2026-10-16, P0 correction)

Reviewer-grade fix: target-period quality and temporal decay are
separate axes.  The 147 hits split as 121 confirmed correct (A4
correct, not decayed), 12 not-decayed but incomplete (base-period
gaps or pure additions), 8 decayed (3 wrong + 5 partially
invalidated), 6 indeterminate.  The former "133 still correct"
group mixed the 121 with the 12; all answer-level false-decline
rates re-denominatored to 121, with the 12 reported separately.

| mechanism   | served | decayed served | correct declined /121 | incompl. declined /12 | indet. /6 |
|-------------|--------|----------------|----------------------|-----------------------|-----------|
| similarity  | 147    | 8              | 0                    | 0                     | 0         |
| strict      | 89     | 0              | 41 (33.9% [26.1,42.7]) | 4 | 5         |
| majority    | 96     | 0              | 36 (29.8% [22.3,38.4]) | 3 | 4         |
| claims      | 104    | 0              | 28 (23.1% [16.5,31.4]) | 3 | 4         |
| asserted    | 111    | 0              | 21 (17.4% [11.6,25.1]) | 3 | 4         |
| page-change | 43     | 1              | 85 (70.2%)           | 7                     | 5         |
| entity-change | 56   | 0              | 67 (55.4%)           | 12                    | 4         |
| prop-change | 100    | 0              | 28 (23.1%)           | 7                     | 4         |
| value-in-page | 97   | 3              | 35 (28.9%)           | 6                     | 4         |
| llm-recheck | 129    | 3              | 9  (7.4%)            | 1                     | 3         |

Entity-change declines all 12/12 not-decayed-incomplete entries (any
JSON change flags them).  Closed-loop strict-served 89 = 80 correct
+ 8 incomplete-not-decayed + 1 indeterminate.  Exact
asserted-dependency validator decisions coincide with the asserted
gate by construction (same dependency set, state, and rule).
Cluster bootstrap (s8, corrected label rule, 10,000 resamples):
decayed-served share 5.4% [1.4,10.6]; correct-declined strict
33.9% [24.8,43.0], majority 29.8% [20.9,38.8], claims 23.1%
[15.0,31.5].  Clopper-Pearson 95% lower bound for 8/8 interception:
63.1%.  Data: data/crosstab_labels.json.

## Tau calibration audit (reviewer item 4)

tau = 0.8363771188718344 is the 7th order statistic of the 154
genuine-paraphrase similarities (code: sims[int(0.05*n)-1], 0-based
index 6), NOT the linear-interpolation 5th percentile (0.8613).
Counts: 6 below, 1 exactly equal (pid 51, treated as not accepted),
147 strictly above.  Text must not claim ">= tau, no ties".

## Protection-recovery dual-model revision (reviewer item 12)

s12 rewritten and rerun: Arm 1 now reports two protection models.
Model R (random placement): each certificate item copied
independently with probability c in {0.9, 0.95}, items cited by
several certificates share one placement indicator; analytic
survival (1-p(1-c))^n averaged over placement and loss.  Model F
(fixed budget): a fixed set of x = floor(0.9n) items protected per
hit; survival (1-p)^(n-x) over the actually exposed items, matching
the companion's R = (1-p)^{e(S)}.  The old single-model c=0.5
column is removed.

Arm 1, asserted contract (baseline 111 serves, mean cert 1.59):
p=0.01: unprot 109.3, R0.9 110.8, R0.95 110.9, F0.9 109.9, full 111
p=0.05: 102.9, 110.1, 110.6, 105.4, 111
p=0.10: 95.6, 109.3, 110.1, 99.7, 111
p=0.20: 82.1, 107.6, 109.3, 88.5, 111
Arm 1, strict contract (baseline 89 serves, mean cert 4.43):
p=0.01: 85.2, 88.6, 88.8, 88.1, 89
p=0.05: 71.7, 87.1, 88.0, 84.3, 89
p=0.10: 58.0, 85.2, 87.1, 79.7, 89
p=0.20: 37.7, 81.5, 85.2, 70.4, 89

Monte-Carlo check (model R, 2000 runs, seed 7, p=0.10): asserted
R0.9 mean 109.30 SE 0.03 |err| 0.02; R0.95 mean 110.12 SE 0.02
|err| 0.01; strict R0.9 mean 85.18 SE 0.04 |err| 0.01; R0.95 mean
87.03 SE 0.03 |err| 0.03.  Reported as mean/SE/absolute error, not
percentile bands.  Copy-trust negative control unchanged (147/147
incl. 8 decayed).  Arm 2 revert control unchanged.

Teaching point for the text: random placement benefits from
averaging luck on short certificates (109.3/111 at p=0.10, c=0.9);
a fixed 90% budget has no such luck (99.7), and the gap widens with
certificate size (strict 85.2 vs 79.7).  Partial coverage improves
availability smoothly; exact baseline restoration only at full
coverage.  Threshold wording ("coverage has a threshold character",
"only full coverage restores the baseline exactly") removed from
the text.

## Tau archiving correction (appendix)

Decisions use the four-decimal operating point tau=0.8364 with >=,
against which 7 genuine similarities fall below, 0 tie, 147 clear.
The full-precision tau_0 = 0.8363771188718344 is attained by pid 51
and is excluded by the rounded operating point.  The
linear-interpolation 5th percentile (0.8613, 146 hits) belongs only
to the trace-quantile convention of Appendix trace-preprocessing
and plays no role in the replay.  Appendix A.5 now states the exact
selection rule (7th order statistic, zero-based index 6), the
full-precision value, the 7/0/147 counts, and this attribution.

## Conditional population reweighting (reviewer item 11, s13)

`scripts/s13_population_reweight.py` → `data/population_reweight.json`,
regenerated `data/fig_cache_age.pdf/.png` (copied to the paper `fig/`).
No model calls.

**1. Deterministic full-universe measurement.**  Pair-level staleness
recomputed for the full universe of 6,580 templated (entity, property)
pairs at every rebuilt snapshot (2026-04-01/05-01/07-01/08-01),
replicating s1's universe construction exactly.  Verified: 6,580 pairs,
52 stale at the 4-month endpoint, 0 entities missing at intermediate
snapshots.  Deterministic per-age prevalence: 0 / 0.38% / 0.78% /
0.79%; onset share of eventually-stale: 0 / 48.1% / 98.1% / 100%
(matches the panel-onset scaling it replaces: 0/48/96/100).
Per-property 4-month decay d_z is strongly heterogeneous: P361 10.1%,
P175 4.3%, P136 4.0%, P2046 2.9%, P131 2.0% ... down to zero for 16 of
the 31 templated properties.

**2. Conditional reweighting.**  Population decline rate of variant v
at age t = sum_z w_z [ d_z(t) h_S r_S(v,t) + (1-d_z(t)) h_I r_I(v,t) ],
with w_z = n_z/N, stratum hit rates h_S = 43/50 = 0.86 and
h_I = 104/104 = 1.00 (declines occur only on hits), and panel
hit-conditional decline rates r_S/r_I pooled across properties.  New
vs old (pooled 52/6580 x panel onset) estimates:

| variant  | age | conditional (new) | pooled (old) |
|----------|-----|-------------------|--------------|
| strict   | 4m  | 14.99% | 15.1% |
| majority | 4m  | 8.31%  | 8.4%  |
| claims   | 4m  | 0.68%  | 0.79% |
| asserted | 4m  | 0.57%  | 0.66% |
| strict   | 1m  | 4.93%  | 5.2%  |
| majority | 1m  | 1.09%  | 1.3%  |
| claims   | 1m  | 0.13%  | 0.4%  |
| asserted | 1m  | 0.10%  | 0.2%  |

(strict/majority move little because sentence-driven intact declines
dominate and r_S ~ 1.0; claims/asserted drop by the h_S factor: stale
requests miss the cache 14% of the time and cannot be gate-declined.)
Per-property r_S sensitivity at the endpoint (properties with >=3
stale hits): strict/majority/claims = 1.0 everywhere; asserted ranges
0.556 (P136) / 0.625 (P106) / 1.0 (P175, P131, P19, P361).  Answer-
decay prevalence unchanged: 0.79% x 9/50 ~= 0.14% of requests.

Text updates: main reweighting paragraph (new numbers, deterministic
measurement, h_S/h_I conditioning, d_z heterogeneity), Figure
fig_cache_age caption, Limits paragraph, Appendix A.5 protocol
sentence (s13 + population_reweight.json), Discussion budget figure
(15.0% strict).

## Unfaithful-paraphrase three-case table (reviewer item 8)

Annotation question A3 flags exactly 3 of 154 paraphrases as
unfaithful, all P175 performer questions narrowed to on-screen
contributors:

| pid | entity | paraphrase drift | sim | claims at target | gate verdicts |
|-----|--------|------------------|-----|------------------|----------------|
| 8   | The Fifth Element | "Who acted in..." | 0.892 | P175 emptied (sole designated = sole asserted claim dead) | all four contracts decline; similarity-only serves |
| 52  | As Good as It Gets  | "Which actor or actress starred..." | 0.869 | P175 emptied (same) | same |
| 95  | Cars 3              | "Which actors provided the voices..." | 0.866 | Newman removed from P175, kept as P86 composer (mis-attribution repaired) | same |

All three: hit (sim > tau), dependency-stale, answer annotated
correct at target (A4 correct, A5 no effect).  Verified by direct
recomputation against the 2026-08-01 snapshot: strict = majority =
claims = asserted = decline (sdead = 0 in all three; the decline is
triggered solely by the dead asserted claim).  The drift caused no
wrong serve on this panel, but the cases isolate the gap:
certificates vouch for evidence currency, not question fidelity;
had the claims been intact, every contract would have served a
composer's name to a question about actors.  Added as
Table paraphrase-drift + one paragraph in the annotation
discussion; relation-preservation paraphrase check flagged as
future work.

## Staged timing + fail-closed injection (P2, s14)

`scripts/s14_staged_timing_failclosed.py` →
`data/staged_timing_failclosed.json`.  No model calls; real replay
path (147 hits at the 2026-08-01 target), Apple M3 Pro, 3,000
randomized hit evaluations, perf_counter_ns.

**A. Staged timing (median us; mean / p95 / p99 in json).**
retrieval (exact 154-doc brute force, 1024-dim cosine) 6.1;
certificate access 0.1; claim checks 0.4; sentence checks 1.5.
Totals: strict 8.2 us (p99 12.7), claims-level 6.7 us (sentence
stage skipped).  Retrieval dominates; the evidence-verification
stages are microsecond-scale.  Query embedding production (bge-m3
service) is excluded as a shared fixed cost of any semantic cache.
(Re-measured 2026-10-17 on the same M3 Pro to sync the archive with
the paper; medians jitter ~10% run-to-run, all conclusions
unchanged.)

**B. Fail-closed fault injection (serves out of 147 hits).**
Control (no fault): 89/96/104/111 = strict/majority/claims/asserted,
exactly the baseline verdict counts.  F1 entity evidence unavailable:
0/0/0/0.  F2 page evidence unavailable: 0/0 for strict/majority,
104/111 unchanged for claims/asserted (by construction).  F3
corrupted claim value read: 0/0/0/0.  F4 certificate unreadable:
0/0/0/0.  F5 corrupted query embedding (NaN): retrieval miss, no
serve, 0/0/0/0.

Text: new paragraph "Staged timing and fail-closed behavior" before
the Limits paragraph; Appendix A.5 protocol sentence added;
staged_timing_failclosed.json added to the data inventory row.

## Inter-annotator agreement + regeneration annotation round (P3, s15/s16)

**A. Second independent annotator (154 cached entries).**
`scripts/s15_inter_annotator.py` parses the two completed sheets
(`annotation/PVSB_annotation_sheet_ann1.docx` = first annotator =
the existing adjudicated labels, `..._ann2.docx` = second
independent annotator) → `annotation/annotations_ann1_parsed.json`,
`annotations_ann2_parsed.json`, `agreement.json`.

Item-level agreement 96.2% (741/770 cells).  Per question:
A1 96.8% (kappa 0.81), A2 96.1% (0.90), A3 98.0% (kappa 0.0 --
only 3 positive cells, kappa uninformative), A4 95.5% (0.86),
A5 94.8% (0.91).  29 disagreeing cells, adjudicated by the first
annotator against the sheet's operational definitions; all reported
counts unchanged.

Sensitivity under ann2's raw labels: decayed set 9 -> 13
(+8/52/71/95/155, -48).  The three additions pid 8/52/95 are the
Table-6 paraphrase-drift cases (ann2 read the composer answer as
wrong for the drifted actor query) -- an interpretation divergence,
not a factual one; pid 71/155 borderline partial attributions;
pid 48 wrong->indet.  ALL 13 ann2-decayed entries are
dependency-stale and declined by every gated contract (12 hits, all
declined; pid 51 a miss), so interception claims are insensitive to
adjudication.  Only the similarity-only decayed-serve share moves:
8/147 = 5.4% -> 12/147 = 8.2%.

**B. Regeneration annotation sheet (51 answers).**
`scripts/s16_make_regen_sheet.py` →
`annotation/PVSB_regeneration_sheet.docx` (copy at the paper root:
`PVSB_再生答案标注表_51条.docx`).  Per entry: question (EN+ZH),
paraphrase, cached (declined) answer, regenerated answer,
target-period values, up-to-3 cleaned target evidence sentences;
items R1 (correctness at target: 正确/错误/不完整/无法判断) and R2
(target-evidence support: 支持/部分支持/不支持/无法判断), same √/□
marking convention.  Gate decisions and cached-answer labels
withheld.  Awaiting completion by both annotators; on return, parse
with the parse_annotations machinery, recompute the closed-loop
delivery counts (51 value-mention-accepted -> answer-level truth),
and update the closed-loop paragraph and abstract wording
("delivered" -> verified share as measured).

## Regenerated-answer answer-level truth (P3回填, s17)

`scripts/s17_parse_regen_annotations.py` parses the two completed
regeneration sheets (`annotation/PVSB_regeneration_sheet_ann1.docx`,
`..._ann2.docx`) → `annotation/regen_ann1_parsed.json`,
`regen_ann2_parsed.json`, `regen_agreement.json`,
`data/regen_labels.json` (adjudicated = first annotator, consistent
with the cached-answer round; 8 disagreeing cells listed in the
audit).

Agreement: R1 92.2% (kappa 0.67), R2 92.2% (kappa 0.88).
R1 adjudicated: 43 correct / 8 incomplete / 0 wrong / 0 indet
(ann2 raw: 45/6).  R2 ann1: 19 supported / 11 partial / 21
unsupported (ann2: 18/15/18) -- the unsupported share reflects the
sentence-coverage gap of claim-sourced regeneration (target values
from Wikidata claims often lack a target-page sentence), consistent
with the Limits note.

Closed-loop truth upgrade (was: 51 passed value-mention acceptance,
"delivered, not verified"): 43 correct + 8 incomplete, none wrong,
none indeterminate.  Final delivery: 89 cached (80 correct, 8
incomplete-not-decayed, 1 indet) + 51 regenerated (43 correct, 8
incomplete) = 140 responses, 14 abstentions, 0 annotated-decayed
cached served, 0 annotated-wrong regenerated delivered.

Text updates: abstract (43 correct / 8 incomplete / none wrong),
closed-loop paragraph (annotation round described; "remains to be
done" sentence removed), Appendix A.5 (sheet + parser + audit files
archived), data inventory row (+ regen_labels.json).

Note for the record: 48/51 notes fields are identical between the
two sheets -- the marks differ in 8 cells, so the choices were made
on separate copies, but the notes text appears shared; the
independence claim in the paper covers the R1/R2 marks only.

## s18 incoming-query truth axis (2026-10-17)

`scripts/s18_incoming_query.py` → `annotation/incoming_query_labels.json`
(three-axis labels: Q_original / Q_incoming / D_temporal for all 154),
`data/incoming_query_crosstab.json`.

The three paraphrase-drift cases (pid 8/52/95, P175 performer wording)
are still correct for the cached question but wrong for the drifted
incoming query.  Recomputed under incoming-query truth (147 hits):
8 decayed + 3 drifted + 118 confirmed-correct + 12 incomplete + 6
indeterminate.  Correct-decline rates: strict 38/118 = 32.2% [23.9,
41.4], majority 33/118 = 28.0% [20.1, 37.0], claims 25/118 = 21.2%
[14.2, 29.7], asserted 18/118 = 15.3% [9.3, 23.0] (Clopper-Pearson).
Similarity-only wrong-for-incoming serves: 11/147 = 7.5% (8 decayed +
3 drifted).  All four gated contracts decline all 8 decayed and all 3
drifted hits (evidence reasons, coincident with drift).  Closed loop:
all three drifted queries abstain (target P175 evidence empty), so no
drifted-intent answer is delivered.  `s8_cluster_bootstrap.py` updated
accordingly (drift axis): wrong-incoming share 7.5% [2.8, 12.8],
strict correct declines 32.2% [23.5, 40.8] (entity-cluster, B=10000).

## s19 alternative certificates (disjunctive family)

`scripts/s19_alternative_certificates.py` → `data/alternative_certificates.json`.

Family = {claims certificate, sentence certificate}; serve if ANY
validates.  On the 147 hits: family-OR serves 137 at a 3.4%
confirmed-correct decline rate but serves 8 wrong-for-incoming answers
(5 decayed + 3 drifted; 5.4%) -- in all 8 the sentence certificate
survives while the claims certificate is dead (Wikipedia sentences lag
Wikidata edits; the certificates do not fail independently).
Sentences-only shows the same wrong set.  All conjunction-family
contracts (strict 89 / majority 96 / claims 104 / asserted 111) serve
zero wrong answers.  Conclusion: alternative certificates buy
availability only under independent failure; single-source panels must
conjoin, corroborate, or scope to answer-asserted claims.

## s20 budget-matched fixed-set protection

`scripts/s20_fixed_budget_protection.py` → `data/fixed_budget_protection.json`.

720 unique designated items, 65,508 total bytes (sentence text bytes;
claim key bytes).  Fixed global set S filled greedily by coverage
frequency vs random per-item placement at the SAME expected bytes; 500
i.i.d. erasure realizations per cell.  At 90% budget: 115/147 entries
fully protected (e_q(S)=0); fixed beats random -- asserted 111.0 vs
107.6, strict 85.3 vs 81.7 expected serves at p=0.20.  At 50% budget
the ordering inverts (strict 77.4 vs 79.7 at p=0.05): partially
protected entries spend bytes without being saved under conjunctive
fail-closed gates.  No protection at p=0.10: strict 58.1, asserted
95.7.

Extension (2026-10-17, sharing audit): designated-item reuse factor
782/720 = 1.09; reference-count distribution {1: 677, 2: 30, 3: 7,
4: 6}; multi-referenced items carry only 2.9% of designated bytes
(unique claims 368, sentence items 352).  A feasible-set random
baseline (uniform over budget-feasible subsets, 50 draws x 100
failures per cell) is statistically indistinguishable from per-item
Bernoulli placement.  A de-sharing ablation (deduplicated shared
items) changes strict expected serves by at most 2.7 and asserted by
0.0 across cells, so the fixed-set advantage comes from
coverage-aware selection, not from cross-entry sharing.  Updated
90%-budget numbers: strict 85.4 vs 81.0, asserted 111.0 vs 107.4
(p=0.20).

## s21 retrieval scaling curve

`scripts/s21_retrieval_scaling.py` → `data/retrieval_scaling.json`.

Brute-force cosine retrieval, bge-m3 dim 1024, float32, single thread,
Apple M3 Pro: N=154 -> 4.0 us; 1e3 -> 18.1 us; 1e4 -> 454.5 us;
1e5 -> 4.37 ms; 1e6 -> 44.7 ms median per query.  Retrieval scales
linearly and dominates beyond ~1e4 entries; evidence-verification
stages stay microsecond-scale and cache-size independent.

## s22 unified strategy table (2026-10-17)

`scripts/s22_unified_strategy.py` → `data/unified_strategy_table.json`.

No new model calls were needed: the non-serve sets are nested
(asserted 43 <= claims 50 <= majority 58 <= strict 65, and
similarity-only's 7 misses lie inside strict's 65), and regeneration
outcomes depend only on the request, so the cached strict closed-loop
regeneration (51 regen + 14 abstain, answer-level labels in
`data/regen_labels.json`) covers every policy's fallback.

Under incoming-query truth, over all 154 requests, with MUTUALLY
EXCLUSIVE quality classes (2026-10-17 recalibration: wrong = 3
decayed-wrong + 3 drifted; incomplete = 12 change-independent + 5
partially invalidated; decayed/drifted are attribution tags on a
separate axis, not quality classes):

| policy | correct | incomplete | wrong | indet | abstain |
|---|---|---|---|---|---|
| similarity-only | 119 | 17 | 6 | 6 | 6 |
| strict | 123 | 16 | 0 | 1 | 14 |
| majority | 121 | 17 | 0 | 2 | 14 |
| claims | 122 | 16 | 0 | 2 | 14 |
| asserted | 122 | 16 | 0 | 2 | 14 |

Attribution axis: the gates avoid 8 decayed + 3 drifted serves that
similarity-only delivers (6 of the 11 annotated wrong for the
incoming request, 5 partially invalidated).

Headline: every gated contract delivers more correct answers than
similarity-only (121-123 vs 119 of 154) with zero wrong deliveries
against 6; the paired correct-delivery gain is positive but NOT
significant (see s24).  The 14 abstentions include all three drifted
queries.  Written into main.tex as Table "Unified strategy view" +
paragraph after the closed-loop paragraph; abstract updated
accordingly.

## s23 population reweighting, scheme A (2026-10-17)

`scripts/s23_reweight_schemeA.py` → `data/population_reweight_schemeA.json`;
`data/fig_cache_age.pdf` redrawn and copied to `fig/`.

Scheme A (current-period stratification): D(t) = d(t) h_S r_{S_t}(t)
+ (1-d(t)) h_I r_{I_t}(t), where r_{S_t}, r_{I_t} are the decay and
intact decline rates of the COHORT that is stale/intact at time t
(claims-side rates deterministic; asserted-cohort assumption stated
in the paper).  Regeneration demand, percent of 154 requests:
1 month strict 5.1 / majority 1.3 / claims 0.33 / asserted 0.1;
3 months 13.1 / 6.4 / 0.67 / 0.5; 4 months 15.0 / 8.3 / 0.68 / 0.57
(four-month values identical to the previous mixed-cohort scheme).

## s24 paired transitions + delta-Y bootstrap (2026-10-17)

`scripts/s24_paired_transition.py` → `data/paired_transition.json`.

Paired transition counts (similarity-only -> strict): correct->correct
111, correct->incomplete 4, correct->abstain 3, incomplete->correct 6,
incomplete->incomplete 11, indet->correct 3, indet->abstain 2,
indet->indet 1, miss->abstain 6, miss->correct 1, wrong->correct 2,
wrong->incomplete 1, wrong->abstain 3.  Upgrades 11 (6 incomplete + 3
indet + 2 wrong), downgrades 7 (4 correct->incomplete + 3
correct->abstain).

Delta-Y (correct deliveries minus similarity-only's), entity-cluster
bootstrap B=10000 over 143 entities: strict +2.60 pts [-3.85, +9.26],
majority +1.30, claims/asserted +1.95 -- every interval crosses zero.
Headline downgraded accordingly: "6 -> 0 wrong deliveries, correct
deliveries 119 -> 121-123, abstentions 6 -> 14; paired correct-delivery
gain positive but not significant."

## s25 runnable closed-loop prototype (P1 Experiment A, 2026-10-17)

`scripts/s25_closed_loop_system.py` → `data/closed_loop_system.json`.

Four components assembled into a runnable loop: certificate output
(claims + asserted claims + sentence ids + issue date per entry),
current-state upkeep (the four real rebuilt evidence snapshots
2026-04/05/07/08), fallback (template composition from current truthy
values at intermediate ticks; the real cached LLM regenerations at the
20260801 endpoint), and write-back (fresh certificate from the same
current evidence + evidence-checked retrieval alias for regenerated
misses).  Three snapshot-update ticks (cache ages one, three, and
four months: 2026-05-01, 2026-07-01, 2026-08-01; no June Wikidata
view was rebuilt) replay the same 154 requests.

Self-checks, all passing:
- V1: the independent verifier on the base certificates at 20260801
  reproduces the published endpoint counts exactly (strict 89 /
  majority 96 / claims 104 / asserted 111 of 147 hits).
- V2: all 29 written-back certificates verify at their issue state
  under all four contracts; all 40 abstentions coincide with absent
  current evidence.
- V3: every request accounted exactly once per tick per policy.
- V4 (bonus): the static baseline's endpoint breakdown reproduces
  Table 9's similarity-only row exactly (119/17/6/6/6) from the raw
  snapshots alone.

Closed-loop frontier (3 ticks, 462 arrivals; violating serves use the
deterministic value proxy; endpoint columns use the human labels):

| policy | fallbacks | violating serves | abstains | correct@T | wrong@T |
|---|---|---|---|---|---|
| similarity, static cache | 3 | 83 | 18 | 119 | 6 |
| TTL, blind refresh at every tick | 422 | 0 | 40 | 132 | 0 |
| page-change detect + loop | 174 | 22 | 35 | 130 | 0 |
| PVSB strict gate + loop | 53 | 0 | 40 | 130 | 0 |
| PVSB asserted gate + loop | 29 | 0 | 40 | 128 | 0 |

Key findings: (i) the asserted-claims loop matches the blind
per-tick refresh's zero violations at 6.9% of its regeneration budget
(29 vs 422); (ii) write-back converts the one-shot fallback demand
(65 strict / 43 asserted) into incremental maintenance -- the final
tick executes only 4 (strict) and 2 (asserted) regenerations;
(iii) correct deliveries rise above the one-shot closed loop
(130/128 vs 123/122) because repairs restore availability;
(iv) page-change detection spends 6x the asserted loop's budget and
still leaks 22 violating serves at intermediate ticks (Wikidata
claims move without the article text changing); (v) the three
meaning-drifted queries abstain at every tick -- their evidence
(P175 performer values) was withdrawn already at 20260501 -- so the
loop never serves them, a coincidence of evidence withdrawal rather
than drift detection.

Honesty labels: intermediate-tick regenerations are template
compositions from the same snapshots; intermediate-tick correctness
is the deterministic value proxy; only the endpoint uses the human
labels.  Written into main.tex as the "runnable closed-loop
prototype" paragraph + Table "Closed-loop frontier" after the unified
strategy table; abstract and Discussion cost paragraph updated.

## s26 TTL-vs-loop sensitivity defence (2026-10-17)

`scripts/s26_sensitivity_ttl_vs_loop.py` → `data/sensitivity_ttl_vs_loop.json`.
Re-executes s25 as a module (identical rollout), then:

A. Paired entity-cluster bootstrap (B=10000, 143 clusters, seed
20261017) of endpoint correct-delivery differences, published
accounting: TTL - asserted = +4, 95% CI [-5.03, +13.17],
P(TTL better) = 0.77; TTL - strict = +2, CI [-6.87, +10.79];
TTL - page-change = +2, CI [-4.75, +8.94].  On this sample the
direction and magnitude of the correct-delivery differences are
undetermined (intervals cross zero without implying equivalence).

B. rho-parametric crossover (charging every template regeneration the
same accuracy rho instead of free proxy-correctness):
TTL(rho) = 43 + 89 rho; asserted(rho) = 101 + 27 rho -> rho* = 0.9355;
strict(rho) = 83 + 47 rho -> rho* = 0.9524; page-change
56 + 74 rho -> rho* = 0.8667.  Observed panel regeneration accuracy
43/51 = 0.843, Wilson 95% [0.720, 0.918] -- below both PVSB
crossovers.  At rho = 0.843: TTL 118.0 vs asserted 123.8, strict
122.6, page-change 118.4.  Blind refresh re-rolls 100
human-confirmed-correct cached answers and wins only if regeneration
from current evidence exceeds 93.5% accuracy.

Marginal cost of the refresh's face-value edge: 98.2 (asserted),
184.5 (strict), 124.0 (page-change) extra regenerations per extra
correct delivery.

Gap decomposition: +11 change-independent imperfect cached answers
(9 incomplete + 2 indeterminate; evidence intact, served as
contracted) - 7 net incomplete among the refresh's own real
regenerations = +4.

Written into main.tex as the "Sensitivity of the frontier comparison"
paragraph after the closed-loop analysis.

## s27 protection inside the closed loop (P2 Experiment C, 2026-10-17)

`scripts/s27_protected_closed_loop.py` → `data/protected_closed_loop.json`
(runtime ~16 s after caching; deterministic hashed common random numbers).

Read-erasure observation layer on the s25 loop: at each tick the gate's
read of every certificate item is lost i.i.d. with probability p; a lost
read of a protected item is transparently restored to its TRUE current
availability (never revives a genuinely decayed item); an unrestored
loss reads as unavailable and the conjunctive contract declines, paying
a real fallback + write-back (new items enter unprotected under the
static set).  Variants: none / random(0.90 per item) / fixed(s20's
static 90%-byte greedy coverage set, rebuilt and byte-matched).

Validations: V1 p=0 reproduces the s25 loop exactly (asserted 29,
strict 53 fallbacks); V2 protect-all at p=0.20 is identical
(transparent restoration); V3 fixed-S bytes == s20 archived 90% row
(58,954 of 58,957 budget); V4 zero wrong serves in all 3,600
realization runs (2 contracts x 3 variants x 3 p x 200).

Means over 200 realizations, regeneration executions over 3 ticks
(p95 in parentheses):

| contract | protection | p=0.05 | p=0.10 | p=0.20 |
|---|---|---|---|---|
| asserted | none | 61.2 (70) | 89.7 (103) | 142.7 (156) |
| asserted | random | 33.8 (38) | 38.6 (44) | 48.0 (55) |
| asserted | fixed | 30.6 (33) | 32.1 (36) | 35.0 (39) |
| strict | none | 115.8 (126) | 158.4 (169) | 217.6 (229) |
| strict | random | 62.1 (68) | 70.4 (78) | 85.9 (95) |
| strict | fixed | 57.6 (61) | 61.9 (66) | 69.5 (75) |

Erasure-free reference: 29 (asserted), 53 (strict).  Unprotected
inflation at p=0.10 is 3.1x/3.0x and dominated by false declines
(60.0/109.9 regenerations of evidence-intact entries).  Fixed S holds
the budget to +11%/+17% over erasure-free and beats equal-budget
random in every cell; residual false declines concentrate in the
unprotected byte tail and write-back certs' new items.  Endpoint
correct deliveries stay within -1.5/+3.2 of 128/130 in all cells
(the unprotected variants gain up to +1.8 asserted / +3.2 strict by
re-rolling change-independent
imperfect answers -- the TTL frontier trade again): read erasure is
priced in regeneration budget, not delivered correctness.

Written into main.tex as "Protection inside the closed loop" +
Table "Read erasure inside the closed loop"; Discussion auditability
advantage updated.

## s28 — Anchored lexical alternative acceptance rule (negative control upgraded to positive validation)

The naive sentence-certificate branch (s19) launders the dead
claims branch's signal: 8 decayed-or-drifted serves, 6 of them wrong.
s28 designs an anchored lexical acceptance rule on this development
panel and freezes it before evaluation — an alternative branch must
establish support for the current request–response pair from
evidence whose currency is independently anchored:

  anchored(sentence) := survives on the current page AND mentions
                        >=1 currently-truthy value of the property
  covered(value v)   := v truthy in the current claim state OR
                        mentioned in >=1 anchored surviving sentence
  qualified branch   := >=1 certificate sentence (no vacuous pass)
                        AND all survive AND >=1 anchored AND every
                        answer-asserted value covered

The rule is lexical, not semantic: a surviving sentence mentioning
a stale value alongside a live one can still cover the stale value
(disclosed in main.tex Discussion as an exploratory heuristic).

Validations: V1 naive rules reproduce the s19 archive exactly
(122/8/17, 137/8/4, 104/0/25); V2 all 8 laundered serves
(8/24/48/50/52/95/112/125) rejected by the anchored branch; V3
five-class accounting over all 154 requests sums to 154 for both
anchored families.

Policy comparison over the 147 hits (served / decayed / drifted /
wrong% / correct declined / rate):

| policy | served | dec | drf | wrong% | corr-dec | rate |
|---|---|---|---|---|---|---|
| claims-only | 104 | 0 | 0 | 0.0 | 25 | 21.2% |
| sentences-naive | 122 | 5 | 3 | 5.4 | 17 | 14.4% |
| family-OR-naive (claims primary) | 137 | 5 | 3 | 5.4 | 4 | 3.4% |
| sents-qualified | 13 | 0 | 0 | 0.0 | 105 | 89.0% |
| family-OR-qualified | 108 | 0 | 0 | 0.0 | 21 | 17.8% |
| asserted OR naive sentences | 141 | 5 | 3 | 5.4 | 0 | 0.0% |
| asserted OR anchored sentences | 115 | 0 | 0 | 0.0 | 14 | 11.9% |

Same-primary comparison (the paper's Table 8 rows): asserted only
111/0-wrong/18-declined; asserted OR naive sentences 141/8-wrong/
0-declined (best availability of any policy measured); asserted OR
anchored sentences 115/0-wrong/14-declined.  The anchored family is
nested in the naive family under the same primary; qualification
converts exactly 26 naive serves into declines: 8 decayed-or-drifted
plus 18 entries (14 correct, 1 incomplete, 3 indeterminate) whose
surviving text never anchors the answer against the current claim
state.  The claims-primary naive family (137) and the
asserted-primary anchored family (115) are NOT nested: |N and
Q| = 111, |N\Q| = 26, |Q\N| = 4, net 137-115 = 22 — reported in
the text to pre-empt the "26 != 137-115" misreading.

The anchored sentence branch alone accepts 13 entries, all confirmed
correct; the family recovers 4 confirmed-correct entries the
asserted gate alone declines (77/121/133/136).  Five-class over all
154 requests (fallback from the archived regeneration realization),
identical for both anchored families:
correct 123, incomplete 15, wrong 0, indet 2, abstain 14 — the strict
policy's correct-delivery count with zero wrong, at 115 rather than 89
cached serves.

Written into main.tex as the "Alternative certificates: from a
negative control to an anchored lexical acceptance rule" paragraph,
the last two rows of the simple-baselines table, and the Discussion
fourth boundary (exploratory heuristic, frozen before evaluation).

---

## s29_certificate_handoff.py — certificate handoff under independent verification

Reviewer acceptance criterion 1 (the execution loop holds): certificates
can be handed to an independent verifier, and revocation cannot be
laundered.  Every certificate the asserted-claims closed loop installs —
154 base certificates plus 29 write-back certificates issued over the
three snapshot-update ticks — is serialized into a tamper-evident envelope
(`data/certificates_handoff.jsonl`) binding six elements:

  (1) request id (pid)           (4) full certificate claim set
  (2) answer payload digest      (5) evidence sentence ids
  (3) asserted claim set         (6) issuance snapshot

The envelope digest is sha256 over the canonical JSON of those six
elements.  An independent verifier reads only the transferred envelopes
plus the state-provider view; it recomputes the digest and checks the
served request id and answer payload BEFORE contract evaluation.  The
digest makes an honestly issued certificate tamper-evident (a deployment
would sign it); the experiment shows no bound element can be altered
without detection.

Validations (data/certificate_handoff.json):

| check | result |
|---|---|
| V1 positive control: independent vs in-memory verdicts | 2928/2928 agree (183 envelopes x 4 snapshots x 4 contracts) |
| V2 M1 payload substitution | 183/183 rejected under every contract |
| V2 M2 claim-set laundering | 183/183 rejected |
| V2 M3 asserted-set laundering | 183/183 rejected |
| V2 M4 sentence substitution | 183/183 rejected |
| V2 M5 issuance forward/back-dating | 183/183 rejected |
| V2 M6 cross-request replay | 183/183 rejected |
| V3 revocation laundering focus | 43 base certs revoked by the 20260801 state; 86/86 laundering attempts (claim-set and asserted-set trimming) rejected |

V3 is the "revocation cannot be laundered" clause: the 43 base
certificates that the endpoint state legitimately declines under the
asserted contract stay declined after a laundering attempt, because
trimming a revoked element breaks the digest binding before the contract
is ever evaluated.

Written into main.tex §7 (certificate handoff paragraph of the
closed-loop subsection).

---

## s30_real_payload_chain.py — protection on a real payload execution chain

Reviewer acceptance criteria 2 (paired attribution) and 3 (maintenance-cost
accounting); also fixes the s27 random arm (Bernoulli(0.90) per-item had no
realized byte cap).  Replaces the idealized restoration overlay with an
executable chain over real SQLite stores (`data/payload_stores.sqlite`):
source/protection payload stores with sha1 fingerprints, a per-tick
eligibility registry keyed by premise identity (qid|prop|value for claims,
page|sentence-id for sentences; status + last-eligible version), and a
certificate store (183 rows / 30,878 bytes).

Chain: source read fails -> protection-store backup read (timed) ->
fingerprint check -> registry check -> eligible: restore as premise;
policy_withdrawn / version_incompatible: DECLINE (guard).  The two required
negative cases are exercised by real data: payload-intact but
policy-withdrawn claims and payload-intact but version-incompatible
sentences both verify fingerprints and are still declined (asserted/fixed/
p=0.10: 49.9 withdrawn guards/run; strict/fixed/p=0.10: 56.5 withdrawn +
14.0 incompatible).  Budgeted arms also lose backup reads i.i.d. at rate p
(source and copy both unreadable); the full arm assumes read-any redundancy
(upper-bound reference).

Arms (same generations, state updates, eligibility rules, CRN fault plan,
same ACTUAL byte budget B = 58,954 bytes = greedy usage): none /
random-feasible-set (shuffled fill <= B, drawn per realization then fixed) /
fixed (s20 greedy, static from t0) / full (reference).

Validations: V1 p=0 reproduces s25 (asserted 29 / strict 53); V2 full at
p=0.20 identical; V3 greedy bytes == s20 90% row (58,954); V4 zero wrong
serves in all 4,800 runs; V5 guards fire and always decline; V6 every
random set feasible; V7 none arm reproduces the s27 archived none cells
EXACTLY (61.2/89.7/142.7 asserted, 115.8/158.4/217.6 strict).

Headline cells (mean fallbacks over 200 realizations, 3 ticks):

| cell | p=0.05 | p=0.10 | p=0.20 |
|---|---|---|---|
| asserted none | 61.2 | 89.7 | 142.7 |
| asserted random | 34.3 | 43.1 | 67.3 |
| asserted fixed | 32.0 | 38.3 | 59.3 |
| asserted full (ref) | 29.0 | 29.0 | 29.0 |
| strict none | 115.8 | 158.4 | 217.6 |
| strict random | 64.3 | 81.5 | 124.0 |
| strict fixed | 60.9 | 75.4 | 115.6 |
| strict full (ref) | 53.0 | 53.0 | 53.0 |

Fixed beats random feasible-set placement in every cell (38.3 vs 43.1 at
asserted p=0.10).  The fixed arm's residual over the erasure-free 29 is
9.3 false declines plus 6.3 both-unreadable losses and the unprotected
tail/write-back items.

Paired attribution (asserted, p=0.10, 200 realizations = 92,400
request-ticks), none -> fixed: 66,452 served->served; 10,279
fallback->served (restore -> verify -> cached serve, 51.4/run); 7,669
fallback->fallback (38.3/run); 8,000 abstain->abstain.  none -> full:
12,148 fallback->served.

Cost accounting (measured on the stores): C_state registry build 13.0 ms +
1.3 ms per-tick snapshot loads; C_certificate 183 rows / 30,878 bytes in
1.4 ms; C_gate 503 ns per erasure-free asserted check; C_restore 5,652 ns
per timed backup read (638k reads); C_fallback = loop regeneration
executions (LLM calls only at the 20260801 endpoint; intermediate repairs
are template compositions).  Break-even (asserted, p=0.10, fixed arm):
51.4 executions avoided per 3-tick run at 0.62 ms incremental restore CPU
and 58,954 protection bytes -> 12.1 us CPU and 1,147 bytes per avoided
regeneration execution; parametric in the per-execution regeneration
price (random 46.6 avoided / 12.1 us / 1,265 B; full 60.7 / 11.2 us /
1,079 B).

Written into main.tex §7 "Protection inside the closed loop" (rewritten),
Table tab:protected-loop (new numbers + full row), Discussion cost and
auditability sentences, Reproducibility Statement.

---

## s31_frozen_independent_check.py — frozen-rule independent check

Reviewer acceptance criterion 4.  Frozen artifacts: the four dependency
contracts, the anchored lexical acceptance rule (zero revisions since
freezing), the t0 base certificates.  Nothing re-tuned.

**Leg 1 — entity holdout (2026-08 endpoint, human labels).**  Design-trail
= entities of the 12 entries that drove the rule-design analysis (8
laundered serves + 4 anchored admissions): 10 entities / 15 hits.
Holdout = 126 entities / 132 hits, still containing 3 decayed-or-drifted
entries (non-vacuous).  Frozen anchored family on holdout: 110 served,
**zero wrong**, 13 confirmed-correct declined (11.6%, identical to the
asserted gate's holdout row); naive control on design-trail serves 15/15
with 8 wrong vs the anchored family's 5 served / 0 wrong.  By construction
of the split, the availability GAIN over the asserted gate is concentrated
in the design-trail subset (holdout anchored row == asserted row); the
holdout therefore validates SAFETY (zero wrong, same decline profile), not
the gain — the gain rests on the four human-verified admissions (Leg 3)
and awaits a fresh panel.  Honesty note: rule wording was informed by
panel-wide failure analysis; the holdout is pristine for evaluation (no
holdout label consulted in any revision) but not for wording.

**Leg 2 — forward check at 2026-10-16.**  The nine refetched decayed
(entity, property) pairs (live Wikidata, data/current_claims/) all remain
declined by the frozen claims-level contracts at the forward date
(partial asserted survival 0/1, 3/5, 0/1, 0/1, 2/5, 2/4, 1/2, 4/5, 2/3 —
conjunctive gate never readmits).  The anchored rule's sentence component
is not evaluable forward (no 2026-10 page snapshot; network egress
unavailable this run) — disclosed.

**Legs 3+4 — admission dossiers and challenge audit.**  The frozen rule's
availability gain over the asserted gate is exactly 4 entries
(77/121/133/136); every one is human-confirmed correct (A4=正确), and
every one is a claim-store PRECISION shift, not a value change:

| pid | entity / property | cached answer | claim store now | anchoring current text |
|---|---|---|---|---|
| 77 | As Good as It Gets / publication date | "1997 (theatrical Dec 25, 1997; home media May 19, 1998)" | exact dates 1997-12-23/25, 1998-02-12 (year value dropped) | "is a 1997 American romantic comedy film" |
| 121 | Domenico Gizzi / date of death | "1745" | 1745-01-01 (precision shift from year) | "(Arpino, 1680 – Naples, 1745)" |
| 133 | Ugrin Csák / date of death | "1311" | 1311-01-01 (precision shift) | "died in 1311" (two sentences) |
| 136 | The Fifth Element / publication date | "1997" | exact 1997 dates (May 7/9, Aug 22/28) | "is a 1997 English-language French film" |

The disclosed co-occurrence loophole SHAPE (a surviving sentence mentions
the sentence-covered value alongside live values) fires in all four
admissions (2/1/3/3 shapes), but in each case the covered value is
entailed by the live values (same year), so the loophole did not
materialize into a wrong serve; the vulnerability remains real for
contradictory co-occurrence and is disclosed as such in the text.

Written into main.tex §7 ("A frozen independent check") and the
Discussion fourth boundary (out-of-sample sentence updated).

## Table B: endpoint-fair regeneration + labels (s32–s35, 2026-10-18)

Reviewer ask (acceptance-criteria round): split the closed-loop frontier
into a pure mechanism table (A) and an endpoint answer-quality table (B)
with one generation standard — same question, same origin-tick evidence,
same prompt/acceptance rule, same reuse rule, same human annotation.

**s32 (endpoint map + unified regeneration).** Replaying all five arms
shows where template compositions were delivered at the endpoint:
TTL-1mo 89/154, page-change 74, strict loop 47, asserted loop 27,
similarity-static 0.  Deduplicated (pid, origin-tick) keys: 175
(2026-08: 89, 2026-07: 56, 2026-05: 30).  Each key regenerated with the
real model (kimi-k2-turbo-preview) using the s9 prompt protocol against
the ORIGIN-tick snapshot evidence; one generation shared across arms.
Result: 174 regen_ok, 1 accept-fail (24|20260701 → abstain), 0
no-current-evidence.  Data: data/tableb_endpoint_map.json,
data/tableb_regeneration.json.

**s33/s34 (double annotation + adjudication).** 174 entries annotated
blind by two annotators (sheet annotation/PVSB_tableb_sheet.docx);
R1 agreement 99.4%, Cohen's kappa 0.977, one disagreement; first
annotator adjudicates (as in both earlier rounds).  Data:
data/tableb_labels.json, annotation/tableb_agreement.json.

**Table B (endpoint quality, /154, matched standard):**

| policy | correct | incomplete | wrong | indet | abstain |
|---|---|---|---|---|---|
| similarity, static cache | 119 | 17 | 6 | 6 | 6 |
| TTL-1mo | 119 | 21 | 0 | 0 | 14 |
| page-change-loop | 121 | 18 | 0 | 1 | 14 |
| pvsb-strict-loop | 122 | 17 | 0 | 1 | 14 |
| pvsb-asserted-loop | 124 | 14 | 0 | 2 | 14 |

The mixed-accounting edge reversed: TTL drops 132 → 119 once its 89
template deliveries carry real labels; the asserted loop (124) now
nominally leads the blind refresh.  Paired entity-cluster bootstrap
(B=10,000, seed 20261017, 143 clusters): TTL − asserted = −5, 95% CI
[-13.2, +3.0], P(TTL better) = 0.086.  Pooled regeneration accuracy
193/225 = 85.8%, Wilson 95% [0.806, 0.897].  Data:
data/tableb_endpoint_quality.json.

**s35 (erasure sweep re-scored).** s30 re-run with the scorer
monkeypatched to the Table-B label resolution (S30_NO_WRITE guard;
archive real_payload_chain.json untouched; unpatched re-run reproduces
the archived endpoint means exactly).  Erasure-free references are now
124 (asserted) / 122 (strict).  Endpoint correct means:
asserted/none 115.6/109.3/100.8 at p=0.05/0.10/0.20; asserted/fixed
123.2/121.6/116.1; asserted/full 124.0 flat; strict/none
100.5/88.9/79.6; strict/fixed 119.1/114.0/100.3; strict/full 122.0
flat.  Read erasure now visibly costs delivered correctness when
unprotected — the proxy accounting had masked it — and protection buys
correctness as well as budget.  Data: data/tableb_protected_loop.json.

Note: real_payload_chain.json's costs_measured block was refreshed by an
identical-code re-run on the same machine during s35 development
(registry build 17.8 ms, snapshot loads 1.6 ms, gate 0.6 µs, backup read
5.7 µs unchanged); all deterministic content (cells, transitions,
breakeven counts/bytes) is unchanged.  main.tex timing figures were
synced to the refreshed artifact.

Note (2026-10-18, superseding the timing half of the note above): a final
identical-code re-run refreshed costs_measured again (registry build
11.33 ms, per-tick snapshot load 1.403 ms, gate 504 ns, backup read
4,882 ns over 638,092 timed reads; breakeven restore-time block
recomputed accordingly: random/fixed/full = 0.49/0.54/0.59 ms for s30's
own cells).  Deterministic content is unchanged.  main.tex and the
supplement are synced to THIS artifact: 11.3 ms / 1.4 ms / 0.5 µs /
4.9 µs, and the headline restore-path figure is the run-level estimate
102.7 mean restore-path operations per run (real-driven asserted/fixed/
p=0.10 cell: 53.4 restores + 49.3 withdrawn guards) x 4.9 µs = 0.50 ms,
i.e. approximately 15 µs per avoided accepted regeneration against the
paired 34.405 reduction.  The older 0.62 ms / 12.1 µs (s30 section) and
17.8 ms / 5.7 µs (s35 note) figures are superseded timing measurements
of the same deterministic protocol.

## Real-driven routing, closed decomposition, paired attribution, frozen new-object panel (s36–s40)

**s36 (gap-fill regeneration).** The s30 fault sweep actually visits 462
(pid, tick) regeneration keys; only 225 had archived real outputs. s36
regenerated the 236 missing keys with the real model under the s9/s32
protocol (checkpointed, real API calls) and merged them into
`data/tableb_regeneration.json` (`gap_fill: true`).  Final coverage:
462/462 = 421 real outputs + 40 honest abstentions (no current evidence)
+ 1 acceptance failure counted as abstention (24|20260701).  Data:
`data/tableb_regeneration.json`.

**s37 (real-output-driven closed loop).** Fallback routing driven by
archived real outputs; write-back certificates assert exactly the values
the real answer asserts.  Erasure-free loop unchanged (asserted/strict
identical: 29/53 fallbacks, 0 violations, endpoint 124/122).  Erasure-free
mechanism table updates: TTL 421 fallbacks / 41 abstains; page-change 174
fallbacks, 20 violating (proxy), 6 partially-stale, 36 abstains; endpoint
page-change row 121/19/0/1/13.  Fault cells sit BELOW the template-driven
archive (real write-back certificates assert smaller premise sets).
Endpoint quality uses labelled-only scoring: gap-fill deliveries without
human labels are never scored (193 distinct unlabeled endpoint keys across
all fault runs; per-run unlabeled means archived per cell).  Data:
`data/real_driven_loop.json`.

**s37b (full-counter real-driven sweep).** All 24 cells, 200 CRN seeds:
fallback means + p95, false declines, restores, guard counters,
both-unreadable, zero wrong serves asserted in every cell.  Headline
(asserted, p=0.10): none 68.6 (78), random 37.4 (42), fixed 34.2 (38),
full 29.0.  Data: `data/real_driven_sweep.json`.

**s37c (decimal re-aggregation).** Same protocol as s37b, re-aggregated
with Decimal ROUND_HALF_UP and exact integer sums per cell
(`fallback_sum` fields).  Only one displayed cell changes: asserted /
fixed / p=0.10 moves 34.1 -> 34.2 (exact sum 6,830, mean 34.15; the old
34.1 was a binary-float `round()` artifact).  `real_driven_sweep.json`
was regenerated with a `rounding` note; the pre-correction file is kept
as `data/real_driven_sweep_binary_rounding_backup.json`.

**s38 (closed residual decomposition).** Fixed arm, asserted, p=0.10,
200 seeds, real-driven state; per fallback event aligned with the
same-seed erasure-free run, mutually exclusive classes: necessary 28.48 +
both-unreadable 5.36 + tail-loss 0.30 = 34.15 per run — closes against the
34.2 fallback mean (exact 34.15).  Supersedes the earlier non-additive "9.3 / 6.3"
wording.  Data: `data/residual_decomposition.json`.

**s39 (paired attribution, real-driven).** Asserted, p=0.10, 92,400 paired
request-ticks: of the none arm's 13,711 fallback events the fixed arm
converts 7,823 to restore-verify-serve (39.1/run) and leaves 5,888
(29.4/run); abstentions untouched (7,908 -> 7,908).  State divergence moves
942 request-ticks (4.7/run) serve->fallback and 86 serve->abstain (real
regenerated certificates assert smaller sets; disclosed in text).  Net:
(7,823-942)/200 = 34.405 fewer accepted regenerations per run over exact
totals 13,711 and 6,830 (displayed cell means 68.6 and 34.2 are
consistent with these totals); no serve ever becomes a WRONG serve.
Data: `data/paired_transitions_real.json`.

**s40 (frozen new-object panel).** Acceptance criterion 4 on the
protection chain.  Frozen selection: non-design qids, page at all four
snapshots, s3's frozen sentence-retrieval rule, Random(20261017) shuffle,
first 50 pairs whose freshly generated base answer passes the s4
acceptance rule (181 candidates tried; fresh-only — the stale class was
exhausted by the design panel; disclosed).  Same SQLite read chain, frozen
greedy 90%-byte rule (154/162 items, 13,377/14,887 bytes), p=0.10,
asserted, 100 seeds.  All five checks pass: (a) 15.23 restores/run, zero
fingerprint failures; (b) no natural ineligible-but-intact case —
deterministic eligibility flip of a protected asserted premise declines
through the guard path; (c) 1.95 both-unreadable/run, zero wrong serves;
(d) write-back premises never falsely restored (none arose naturally —
deterministic forced-loss test restores nothing); (e) fixed arm cuts mean
fallbacks 16.8 -> 1.9 (paired diff 14.9, bootstrap 95% CI [14.1, 15.7]).
Data: `data/frozen_panel_generation.json`, `data/frozen_panel_check.json`.

## Second-generator replication (s41–s42, 2026-10-19)

**s41 (GPT regeneration archive).** Every archived regeneration key was
re-run with a second generator, `gpt-5-mini` (Poe OpenAI-compatible API;
`POE_API_KEY` env var), under the byte-identical s9/s32 prompt protocol:
same system prompt, question, origin-tick evidence, value-mention
acceptance rule, and one-time re-ask nudge.  Coverage: all 411 merged-store
keys (175 Table-B endpoint keys + 236 fault-sweep gap-fill keys; the 40
no-current-evidence keys carried over without a model call) plus all 65 s9
endpoint keys.  Results: store acceptance identical in aggregate to the
kimi archive (370 accepted / 1 failed / 40 no-evidence — same profile);
endpoint store 51 accepted of 65, exactly matching kimi's 51/65.
Data: `data/tableb_regeneration_gpt.json`, `data/regeneration_gpt.json`.

**s42 (comparison).** Acceptance cross-tab vs the kimi archive: 369
both-accepted, 40 both-no-evidence, 1 kimi-only, 1 gpt-only (store); 51/14
both-accepted/both-failed (endpoint store).  Among the 369 both-accepted
keys, 98.1% of answer pairs share at least one target value (59.3%
identical after normalisation).  Rebuilding all five policy arms from the
same endpoint delivery map with GPT answers and scoring by the endpoint
value-mention proxy: counts differ from the kimi archive by at most one
delivery per arm (TTL 140→139, page-change 137→138 proxy-correct; the two
PVSB loops identical at 140 served / 140 proxy-correct); only 2 of 770
arm-deliveries change class, one in each direction.  Base-cached answers
are identical across arms and were not regenerated; no human labels are
transferred to the GPT variant.
Data: `data/gpt_generator_compare.json`.  Paper: supplementary Section S4.4,
Table S4.5; main text paragraph after Table 4 discussion.

## GPT human-labelled Table B (s43–s45, 2026-10-19)

**s43 (annotation sheet).** Keys whose GPT answer differs in normalised
text from the kimi answer cannot inherit the kimi-era labels (that would
be label transfer); identical-text answers legitimately inherit (104 of
174 Table-B keys, 22 of 51 s9 keys).  The sheet collects exactly the
remaining keys: 69 store keys (66 same-target-value, 3 different-value)
+ 29 endpoint keys + 1 newly served key (kimi accept-fail, GPT accepted)
= 99 entries, same bilingual format and R1 definitions as the earlier
rounds, kimi answers/labels withheld.  Data: `annotation/gpt_sheet_keys.json`.

**s44 (parse + audit + label store).** Two independent annotators: raw
agreement 98.0%, Cohen's kappa 0.961, 2 disagreements (4|20260801,
144|20260801), first annotator adjudicates (established protocol).
Fresh-99 distribution: 54 正确 / 42 不完整 / 2 错误 / 1 无法判断 — versus
19 不完整 among the same keys' kimi labels: terser GPT answers pass
value-mention acceptance but are judged incomplete more often.
Label store: 225 labels (126 inherited-identical + 99 fresh).
Data: `annotation/gpt_ann{1,2}_parsed.json`,
`annotation/gpt_agreement.json`, `data/gpt_labels.json`.

**s45 (GPT Table B).** Human-labelled GPT endpoint quality (categories
mutually exclusive, rows sum to 154):
similarity-static 119/17/6/6/6 (unchanged); TTL-1mo 105/32/1/1/15;
page-change 109/28/2/2/13; strict 115/24/0/1/14; asserted
120/18/0/2/14 (correct/incomplete/wrong/indet/abstain).
Policy ordering preserved; correct counts 105–120 versus kimi 119–124.
Paired entity-cluster bootstrap (B=10000, seed 20261017), TTL minus
asserted loop: −15.0 [−24,−6]; kimi minus GPT on the same asserted loop:
+4.1 [0,+9] — the apparent gap partly reflects where GPT answers land
(refresh-heavy arms deliver more of them).  Maintenance counts (421 vs 29)
generator-independent.  Conclusion written into supplementary S4.4:
endpoint answer quality is generator-dependent (a property of the
generator, not of the gating contracts); delivery structure, acceptance
decisions, and maintenance counts are generator-robust.
Data: `data/gpt_tableb_endpoint_quality.json`.

## GPT maintenance re-execution (s46–s47, 2026-10-19)

**s46 (gpt_maintenance).** Asserted-claim sets re-extracted from the
GPT-5-mini answers, certificates rebuilt, and all five policy
trajectories re-run with per-tick gating, write-back, and the
archive-coverage assertion over every reachable key.  The
accepted-regeneration counts of all five policies are unchanged from
the primary-generator run (blind refresh 421, asserted loop 29);
per-tick distributions shift by at most one accepted regeneration at a
single tick.  Data: `data/gpt_maintenance_counts.json`.

**s47 (gpt_reexec_endpoint).** The re-executed endpoint routes are
checked against a direct substitution reconstruction that replaces each
archived answer in the primary delivery map; they coincide at every
endpoint delivery except two bookkeeping differences under page-change
maintenance, affecting neither the origin tick nor the quality label of
a delivery.  Data: `data/gpt_reexec_endpoint.json`.

## Mechanism-level comparison arms (s48, 2026-10-20)

**s48 (mechanism_arms).** Three mechanism-level baseline arms mapped
onto the same real-driven closed-loop replay (s37 convention: archived
real generations at every tick, coverage assertion enforced, Table-B
endpoint labelling; zero new model calls): a GroundedCache-style router
(shared τ retrieval gate + certificate-claim overlap ≥ 0.5 + source-page
hash + per-value lexical support), a FinCacheServe-style fingerprint
gate (source-page hash unchanged + exact truthy-claim-set equality), and
a FreshCache-style answer-tier risk gate (per-property monthly survival
calibrated on the full rebuilt population, 186,163 claim observations
over 3,593 properties; risk = 1 − s^age ≤ 0.10).  Results (three ticks,
462 request-ticks; endpoint quality over 154 requests, primary
generator): grounded-router 181 regenerations / 0 violating serves /
endpoint 121 correct; finserve-fingerprint 187 / 0 / 120 correct (one
gap-fill delivery unlabeled, never scored); freshcache-risk degenerates
— the largest calibrated panel risk is 0.0958 (P361, monthly survival
0.9670, age 3) < 0.10, so the gate never fires and the arm coincides
with similarity-only reuse (3 / 83 / 119 correct, 6 wrong).  V3
accounting complete; V5 zero coverage misses.  Written into main-text
§7.3 and Supplementary S4.5 (Tables S4.6–S4.7).
Data: `data/mechanism_arms.json`.

## Crossed gate-by-alias control (s49, 2026-10-20)

**s49 (gate_alias_cross).** Crossed control to isolate the retrieval-alias
maintenance step from the gate granularity in the mechanism-arm comparison
(s48): three gates — asserted-claim verification (s25.verify mode
`asserted`), the GroundedCache-style overlap proxy (s48.grounded_serve),
and the FinCacheServe-style equality proxy (s48.finserve_serve) — each run
with the retrieval-alias step disabled and enabled, on the identical base
cache, request stream, screen, evidence states, archived fallbacks,
acceptance rule, and write-back (s37 real-driven convention; zero new
model calls).  Results (462 request-ticks; endpoint quality over 154
requests, primary generator):

| gate | alias off | alias on | aliases installed | endpoint correct |
|---|---|---|---|---|
| asserted | 31 | 29 | 1 | 124 / 124 |
| overlap-proxy | 181 | 181 | 1 | 121 / 121 |
| equality-proxy | 187 | 187 | 1 | 120 / 120 |

The alias step contributes at most two avoided regenerations in any
configuration (asserted: 31 vs 29); the proxy gates' excess over the
asserted gate (152 and 158 regenerations) is entirely attributable to
gate granularity, not to the alias maintenance.  Endpoint quality is
unchanged by the alias step within each gate.  Validations: V3
accounting complete (154 requests per tick per configuration), V5 zero
coverage misses, R1 anchors reproduced (asserted|alias-on 29,
overlap-proxy|alias-off 181, equality-proxy|alias-off 187).
Written into main-text §7.3 (Exploratory gate mappings paragraph) and
Supplementary S4.5 ("Crossed gate-by-alias control" paragraph).
Script: `scripts/s49_gate_alias_cross.py`.
Data: `data/gate_alias_cross.json`.

**s50 (gate_alias_request_diff).** Request-level decision diff for the s49
crossed control: per (tick, request), compares routing decisions between
alias-off and alias-on for each gate.  Result: exactly two changed cells,
both `regenerated -> served` for the same request (pid 51 at 20260701 and
20260801), in the asserted configuration; zero changed cells in both proxy
configurations.  The net count changes (-2, 0, 0) therefore correspond to
actual single-direction request-level changes, not offsetting flips.
Script: `scripts/s50_alias_request_diff.py`.
Data: `data/gate_alias_request_diff.json`.

**Pairing audit (2026-10-20).** File-level verification of the GPT label
store and the 98-key common-accepted comparison: 225 = 126
inherited-identical + 99 fresh; 99 = 98 paired + 1 unpaired
(24|20260701: kimi accept-fail, GPT accepted, freshly annotated); on the 98
paired keys (69 merged-store + 29 s9-endpoint), GPT incomplete = 41 versus
kimi incomplete = 19 (13 store via `tableb_labels.json` + 6 s9-endpoint via
first-annotator-adjudicated `regen_ann1_parsed.json`).
Data: `annotation/pairing_audit.json`.

## Claim-level equality ablation (s51, 2026-10-20)

**s51 (claim_equality_loop).** Predicate ablation of the FinCacheServe
mapping: drops the source-page version condition and keeps only exact
equality of the queried property's truthy-value set, with write-back, under
the s37 real-driven convention (no model calls; archived outputs only).

- Accepted regenerations 43 (per tick 13/26/4); violating serves 0;
  partially-stale serves 0; abstentions 40.
- Endpoint quality: 123 correct, 15 incomplete, 0 wrong, 2 indeterminate,
  14 abstentions. The single previously unlabeled endpoint delivery
  (request 109, intermediate-tick write-back, the same archived key as the
  equality mapping's unassessed delivery) was annotated by both annotators
  via the s53 one-entry sheet and adjudicated incomplete.
- Two-path predicate decomposition: equality alone 43, + page version 187
  (adds 144); page version alone (page-change loop) 174, + equality 187
  (adds 13). Predicate effects are non-additive.
- Validations: V3 accounting complete; V5 zero coverage misses.

Script: `scripts/s51_claim_equality_loop.py`.
Data: `data/claim_equality_loop.json`.

**s52 (claim_equality_loop_alias).** Same claim-level equality gate with
evidence-checked retrieval-alias expansion enabled (s25/s49 alias-on
mechanism), giving the matched-alias-setting comparison against the
asserted loop's 29. No new model calls (archived outputs only).

- Accepted regenerations 41 (per tick 13/25/3), versus 43 alias-off and
  29 for asserted alias-on; violating serves 0; partially-stale serves 0;
  abstentions 40; aliases installed 1.
- Endpoint quality unchanged: 123 correct, 15 incomplete, 0 wrong,
  2 indeterminate, 14 abstentions — the request-109 key (same archived
  key as in s51 and the equality-proxy mapping) is now annotated
  incomplete via the s53 one-entry sheet.
- Validations: V3 accounting complete; V5 zero coverage misses.

Script: `scripts/s52_claim_equality_loop_alias.py`.
Data: `data/claim_equality_loop_alias.json`.

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Production-trace cost projection and trace-driven queueing check.

Input : azure_conv_slice.csv -- first 48 MiB HTTP-range slice of
        AzureLLMInferenceTrace_conv_1week.csv (Azure Public Dataset,
        release dataset-llm-2024; schema TIMESTAMP,ContextTokens,GeneratedTokens).
        Reproduce with:
          curl -L -r 0-50331647 -o azure_conv_slice.csv \
            https://github.com/Azure/AzurePublicDataset/releases/download/dataset-llm-2024/AzureLLMInferenceTrace_conv_1week.csv
        (the final, possibly truncated, line is dropped by the parser:
         only complete CSV records are used)

Windowing rules (as implemented):
  * timestamps are parsed at the release precision (microseconds) and
    verified to be nondecreasing in file order; ties retain that order
    and are preserved without jitter;
  * minute/second windows are aligned to the first timestamp (offset windows),
    not to wall-clock minutes;
  * the final partial minute and partial second are excluded from
    means and peaks.

The trace carries no prompt content (privacy), so no semantic hit rate can be
measured.  We transfer the measured local costs of the instrumented pipeline
(M3 Pro medians; Appendix timing environment) without scaling by Azure token
counts:
    gate check        0.67 us   (intact-set median; early-exit median 0.42 us)
    record construct  9.0 us    (per-question median, miss/write path only)

Queueing check (single server, FCFS, initially empty, arrival times from the
trace, queueing arithmetic in integer nanoseconds relative to the first
arrival):
    f_0 = 0,  s_i = max(t_i, f_{i-1}),  f_i = s_i + c_i,  w_i = s_i - t_i.
The waiting queue excludes the request in service:
    Q(t) = |{j : t_j <= t < s_j}|,  Q_max = max_t Q(t);
Q_max is attained at an arrival time and is computed directly as well as
via the unfinished-at-arrival count (both are asserted equal).
Service-time scenarios: fixed medians (670 ns; 9670 ns), a 100x sensitivity
(967000 ns), and empirical service-time resampling -- each service time is
the sum of independent draws with replacement from the record-construction
and early-exit checking timing samples in e8_timings_m3pro.npz; the reported
resampling result is from one run with seed 42.

Outputs:
  azure_replay_results.json  -- all statistics reported in the paper
  fig_azure_replay.pdf       -- two-panel figure (arrival rate; projected demand)
"""
import csv, json, math, os, sys
import datetime as dt

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SLICE = os.path.join(HERE, 'azure_conv_slice.csv')
NPZ = os.path.join(HERE, 'e8_timings_m3pro.npz')
OUT_JSON = os.path.join(HERE, 'azure_replay_results.json')
OUT_FIG = os.path.join(HERE, 'fig_azure_replay.pdf')

CHECK_NS = 670           # gate check, intact-set median (ns)
CONSTRUCT_NS = 9000      # record construction, per-question median (ns)
DECODE_BAND = (20.0, 50.0, 100.0)   # tokens/s, stated sensitivity band

# ---------------------------------------------------------------- load
ts_us = []   # integer microseconds since slice start
gen = []
ctx = []
with open(SLICE, newline='') as f:
    r = csv.reader(f)
    header = next(r)
    assert header[:3] == ['TIMESTAMP', 'ContextTokens', 'GeneratedTokens'], header
    t0 = None
    for row in f:
        p = row.rstrip('\n').split(',')
        if len(p) != 3:
            continue                      # truncated final line of the range slice
        try:
            t = dt.datetime.fromisoformat(p[0])
            c = int(p[1]); g = int(p[2])
        except ValueError:
            continue
        if t0 is None:
            t0 = t
        ts_us.append((t - t0) // dt.timedelta(microseconds=1))
        ctx.append(c); gen.append(g)

n = len(ts_us)
arr_us = np.asarray(ts_us, dtype=np.int64)
assert np.all(arr_us[1:] >= arr_us[:-1]), 'timestamps not nondecreasing in file order'
t_ns = arr_us * 1000                     # integer nanoseconds since first arrival
t = arr_us.astype(np.float64) * 1e-6     # seconds (rates/figure only)
span_s = float(t[-1])
span_h = span_s / 3600.0

# ------------------------------------------------- A. arrival dynamics
minutes = int(span_s // 60) + 1
per_min = [0] * minutes
per_sec = [0] * (int(span_s) + 1)
for x in t:
    per_min[int(x // 60)] += 1
    per_sec[int(x)] += 1
per_min = per_min[:int(span_s // 60)]      # drop the final partial minute
per_sec = per_sec[:int(span_s)]            # drop the final partial second

mean_min = sum(per_min) / len(per_min)
peak_min = max(per_min)
peak_sec = max(per_sec)
mean_rate = n / span_s

iats = np.diff(t)
cv_iat = float(iats.std(ddof=1) / iats.mean())   # Poisson process -> 1.0

# ------------------------------------ B. normalized processing demand
def demand_percent(counts, window_s, cost_ns):
    return [c * cost_ns * 1e-9 / window_s * 100.0 for c in counts]

util_gate = demand_percent(per_min, 60.0, CHECK_NS)
util_full = demand_percent(per_min, 60.0, CHECK_NS + CONSTRUCT_NS)
demand = {
    'check': {
        'mean_minute': sum(util_gate) / len(util_gate),
        'peak_minute': max(util_gate),
        'peak_second': peak_sec * CHECK_NS * 1e-9 * 100.0,
    },
    'construct_plus_check': {
        'mean_minute': sum(util_full) / len(util_full),
        'peak_minute': max(util_full),
        'peak_second': peak_sec * (CHECK_NS + CONSTRUCT_NS) * 1e-9 * 100.0,
    },
}

# ------------------------------------ C. trace-driven queueing check
def queue_stats(service_ns):
    """FCFS single server, initially empty; service_ns = int64 ns per request.

    Waiting queue length excludes the request in service:
        Q(t) = |{j : t_j <= t < s_j}|,  Q_max = max_t Q(t).
    Q increases only at arrivals, so Q_max is attained at an arrival time;
    it is computed directly and cross-checked against the count of earlier
    requests unfinished at each arrival (the two are equal for FCFS).
    """
    f = 0
    waits = np.empty(n, dtype=np.int64)
    starts = np.empty(n, dtype=np.int64)
    finishes = np.empty(n, dtype=np.int64)
    for i in range(n):
        s = t_ns[i] if t_ns[i] > f else f
        starts[i] = s
        waits[i] = s - t_ns[i]
        f = s + service_ns[i]
        finishes[i] = f
    done = np.searchsorted(finishes, t_ns, side='right')
    unfinished = np.arange(n) - done       # earlier jobs unfinished at arrival
    started = np.searchsorted(starts, t_ns, side='right')
    q_direct = (np.arange(n) + 1) - started   # arrived but not started at t_i
    assert int(q_direct.max()) == int(unfinished.max())
    return {
        'wait_median_us': float(np.median(waits) * 1e-3),
        'wait_p99_us': float(np.quantile(waits, 0.99) * 1e-3),
        'wait_max_us': float(waits.max() * 1e-3),
        'q_max': int(q_direct.max()),
        'residual_backlog_at_end': int(n - int(np.searchsorted(finishes, t_ns[-1], side='right'))),
        'utilization': float(service_ns.sum() * 1e-9 / span_s),
    }

rng = np.random.default_rng(42)
scenarios = {}
scenarios['check_fixed_0.67us'] = queue_stats(np.full(n, CHECK_NS, dtype=np.int64))
scenarios['construct_check_fixed_9.67us'] = queue_stats(np.full(n, CHECK_NS + CONSTRUCT_NS, dtype=np.int64))
if os.path.exists(NPZ):
    d = np.load(NPZ)
    gate = d['gate_us']; constr = d['constr_us']
    # empirical service-time resampling: independent draws with replacement
    # from the construction and early-exit checking samples, summed; one run.
    svc = np.round((rng.choice(constr, size=n) + rng.choice(gate, size=n)) * 1000.0).astype(np.int64)
    scenarios['construct_check_empirical_seed42'] = queue_stats(svc)

# cost sensitivity on the heavier fixed scenario
sens = {}
for m in [1, 10, 100, 1000]:
    st = queue_stats(np.full(n, (CHECK_NS + CONSTRUCT_NS) * m, dtype=np.int64))
    sens[str(m)] = {'utilization': st['utilization'], 'wait_max_us': st['wait_max_us'],
                    'q_max': st['q_max']}
breakeven_service_s = span_s / n           # mean service time saturating one core

# ------------------------------ D. overhead share of modeled latency
def pct_share(cost_ns, r):
    shares = sorted(cost_ns * 1e-9 / (g / r) for g in gen if g > 0)
    k50 = shares[len(shares) // 2]
    k99 = shares[min(len(shares) - 1, int(0.99 * len(shares)))]
    return 100.0 * k50, 100.0 * k99

share = {}
for r in DECODE_BAND:
    share[str(r)] = {
        'check_median_p99': pct_share(CHECK_NS, r),
        'construct_check_median_p99': pct_share(CHECK_NS + CONSTRUCT_NS, r),
    }

def quant(a, q):
    s = sorted(a)
    return s[min(len(s) - 1, int(q * len(s)))]

results = {
    'slice': {
        'source': 'AzureLLMInferenceTrace_conv_1week.csv (dataset-llm-2024 release), first 48 MiB HTTP range slice; complete CSV records only',
        'citation': 'Stojkovic et al., DynamoLLM, HPCA 2025',
        'requests': n,
        'span_hours': span_h,
        'windowing': 'file order; timestamps verified nondecreasing; ties kept without jitter; offset windows from first timestamp; final partial minute/second excluded',
    },
    'arrival': {
        'mean_rate_req_s': mean_rate,
        'mean_per_min': mean_min,
        'peak_per_min': peak_min,
        'peak_to_mean_min': peak_min / mean_min,
        'peak_per_sec': peak_sec,
        'interarrival_cv': cv_iat,
    },
    'tokens': {
        'context_mean': sum(ctx) / n,
        'generated_mean': sum(gen) / n,
        'generated_median': quant(gen, 0.5),
        'generated_p99': quant(gen, 0.99),
        'generated_max': max(gen),
    },
    'audit_costs_ns': {'gate_check': CHECK_NS, 'record_construct': CONSTRUCT_NS},
    'normalized_processing_demand_percent': demand,
    'queueing_single_server': scenarios,
    'queueing_protocol': {
        'server': 'single FCFS, initially empty; f_0 = 0',
        'arithmetic': 'integer nanoseconds relative to first arrival',
        'queue_length': 'waiting requests only; excludes the request in service',
        'empirical_resampling': 'sum of independent draws with replacement from construction and early-exit checking samples; one run, seed 42',
    },
    'cost_sensitivity_multiplier': sens,
    'breakeven_mean_service_ms_one_core': breakeven_service_s * 1e3,
    'overhead_share_of_modeled_latency_percent': share,
    'decode_band_tokens_per_s': DECODE_BAND,
}
with open(OUT_JSON, 'w') as f:
    json.dump(results, f, indent=2)

# ---------------------------------------------------------------- figure
import matplotlib
matplotlib.use('Agg')
# plain white style for consistency with the other manuscript figures
import matplotlib.pyplot as plt

xmin = [i / 60.0 for i in range(len(per_min))]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.2, 2.7))

ax1.plot(xmin, per_min, lw=0.8, color='#1f77b4')
ax1.axhline(mean_min, ls='--', lw=0.8, color='gray')
ax1.annotate(f'peak {peak_min}', xy=(per_min.index(peak_min) / 60.0, peak_min),
             xytext=(0.35, 0.88), textcoords='axes fraction', fontsize=7,
             arrowprops=dict(arrowstyle='-', lw=0.5, color='gray'))
ax1.text(0.02, 0.90, f'mean {mean_min:.0f}', transform=ax1.transAxes, fontsize=7, color='gray')
ax1.set_xlabel('Time (h)', fontsize=8)
ax1.set_ylabel('Requests / min', fontsize=8)
ax1.tick_params(labelsize=7)
ax1.grid(alpha=.3)
ax1.text(-0.14, 1.02, '(a)', transform=ax1.transAxes, fontsize=9, fontweight='bold')

ax2.plot(xmin, util_full, lw=0.8, color='#d62728', label='construction + check')
ax2.plot(xmin, util_gate, lw=0.8, color='#2ca02c', label='check')
ax2.set_xlabel('Time (h)', fontsize=8)
ax2.set_ylabel('Projected processing demand (%)', fontsize=8)
ax2.tick_params(labelsize=7)
ax2.grid(alpha=.3)
ax2.set_ylim(0, 0.046)
ax2.legend(fontsize=7, loc='upper left', framealpha=0.9)
ax2.text(-0.14, 1.02, '(b)', transform=ax2.transAxes, fontsize=9, fontweight='bold')

fig.tight_layout()
fig.savefig(OUT_FIG, bbox_inches='tight')
print(json.dumps(results, indent=2))

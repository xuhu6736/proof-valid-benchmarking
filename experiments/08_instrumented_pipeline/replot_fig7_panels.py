#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Replot Fig. 7 (instrumented pipeline) as three standalone single-axes
panels at final print size (~0.325 textwidth each), identical axes rects,
legends inside the frames with zero curve overlap:
  (a) retrieval grounding vs depth   -> legend lower right (empty region)
  (b) exact law vs pipeline          -> ylim headroom, legend upper right
  (c) instrumented overhead (CDFs)   -> ylim headroom, legend upper left
Data: e8_pipeline_results.json + e8_timings_m3pro.npz (M3 Pro remeasure);
annotation exact law from grounded parquet via duckdb."""
import json, os
import duckdb
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
res = json.load(open(os.path.join(HERE, 'e8_pipeline_results.json')))
t = np.load(os.path.join(HERE, 'e8_timings_m3pro.npz'))
constr_us, gate_us = t['constr_us'], t['gate_us']

HP = os.path.join(HERE, '..', '..', 'data', 'hotpot_validation.parquet')
sup = duckdb.sql("SELECT supporting_facts['title'] AS g FROM read_parquet(?)",
                 params=[HP]).fetchall()
e_ann = np.array([len(g[0]) for g in sup], dtype=float)
EPS_GRID = res['pred_vs_meas']['gold_in_retrieval']['eps']
pred_annot = [float(np.mean((1 - e) ** e_ann)) for e in EPS_GRID]

K_LIST = res['k_list']; K_MAIN = res['k_main']

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

FS = (2.1, 2.15)
RECT = [0.235, 0.19, 0.715, 0.73]
LAB, TCK, LEG = 8.0, 7.0, 5.8

def new_panel():
    fig = plt.figure(figsize=FS)
    return fig, fig.add_axes(RECT)

# ---- (a) retrieval grounding vs depth ----
fig, ax = new_panel()
ax.plot(K_LIST, [res['recall_at_k'][str(k)] for k in K_LIST], 'o-',
        ms=2.8, lw=1.4, label='gold-doc recall')
ax.plot(K_LIST, [res['full_support_at_k'][str(k)] for k in K_LIST], 's-',
        ms=2.8, lw=1.4, label='complete gold-title coverage')
ax.plot(K_LIST, [res['certifiable_at_k'][str(k)] for k in K_LIST], '^--',
        ms=2.8, lw=1.4, label='nonempty gold-title overlap')
ax.set_xlabel('retrieval depth $k$', fontsize=LAB)
ax.set_ylabel('fraction of questions', fontsize=LAB)
ax.tick_params(labelsize=TCK)
ax.set_ylim(0, 1.02); ax.grid(alpha=.3)
# tighter, corner-anchored legend: clears the orange coverage curve at k >= 3
ax.legend(fontsize=5.2, loc='lower right', bbox_to_anchor=(1.0, 0.0),
          framealpha=0.95, borderpad=0.25,
          labelspacing=0.2, handlelength=1.3, handletextpad=0.45)
fig.savefig(os.path.join(HERE, 'fig7a_retrieval_scis.pdf'))
fig.savefig(os.path.join(HERE, 'fig7a_retrieval_scis.png'), dpi=200)
plt.close(fig)

# ---- (b) exact-law prediction vs pipeline measurement ----
fig, ax = new_panel()
for cname, mk, lab in [('gold_in_retrieval', 'o', 'gold-retr.'),
                       ('full_context', 's', 'full-ctx')]:
    d = res['pred_vs_meas'][cname]
    ax.plot(d['eps'], d['predicted'], '-', lw=1.4, label=f'{lab}: exact')
    ax.plot(d['eps'], d['measured'], mk, mfc='none', ms=4.5, lw=1.2,
            label=f'{lab}: pipeline')
ax.plot(EPS_GRID, pred_annot, ':', color='gray', lw=1.4,
        label='annotation: exact')
ax.set_xlabel(r'premise erasure rate $\varepsilon$', fontsize=LAB)
ax.set_ylabel('record-set availability', fontsize=LAB)
ax.tick_params(labelsize=TCK)
ax.set_ylim(0, 1.30); ax.grid(alpha=.3)
# ylim headroom: the legend now sits entirely above the highest curve point
ax.legend(fontsize=LEG, loc='upper right', framealpha=0.95, borderpad=0.25,
          labelspacing=0.1, handlelength=1.4, handletextpad=0.5)
fig.savefig(os.path.join(HERE, 'fig7b_exactlaw_scis.pdf'))
fig.savefig(os.path.join(HERE, 'fig7b_exactlaw_scis.png'), dpi=200)
plt.close(fig)

# ---- (c) instrumented overhead ----
fig, ax = new_panel()
for arr, lab in [(constr_us, '$\\Pi$ construction\n(surrogate sets)'),
                 (gate_us, 'pooled check\n($\\varepsilon{=}0.1$, early exit)')]:
    xs = np.sort(arr)
    ax.plot(xs, np.arange(1, len(xs) + 1) / len(xs), label=lab, lw=1.5)
ax.set_xscale('log')
ax.set_xlabel(r'time per call or question ($\mu$s)', fontsize=LAB)
ax.set_ylabel('cumulative fraction', fontsize=LAB)
ax.tick_params(labelsize=TCK)
# ylim headroom: the narrow two-line legend floats entirely above the
# y = 1 plateaus that both CDFs reach on the right half of the axis
ax.set_ylim(0, 1.40); ax.grid(alpha=.3)
ax.legend(fontsize=5.4, loc='upper right', bbox_to_anchor=(0.99, 0.995),
          framealpha=0.95, borderpad=0.25,
          labelspacing=0.15, handlelength=1.3, handletextpad=0.45)
fig.savefig(os.path.join(HERE, 'fig7c_overhead_scis.pdf'))
fig.savefig(os.path.join(HERE, 'fig7c_overhead_scis.png'), dpi=200)
plt.close(fig)
print('saved fig7a/fig7b/fig7c pdf+png')

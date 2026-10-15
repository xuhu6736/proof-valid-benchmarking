#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Replot Fig. 3 as three standalone single-axes panels (a/b/c), each drawn at
final print size (~0.325 textwidth) with IDENTICAL axes rectangles so the row
aligns perfectly; legends sit below the axes (no in-plot overlap).
Data and style as in run_coded_regime_experiment.py (fig8/fig9); duckdb loader.
Outputs:
  fig3a_regime_map_scis.pdf          -- hybrid-coded regime map
  fig3b_parity_requirement_scis.pdf  -- MDS parity requirement
  fig3c_grounded_dist_scis.pdf       -- grounded certificate-complexity dist
"""
import os
import duckdb
import numpy as np
from scipy.stats import binom

EPS, DELTA = 0.1, 0.02
HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, '..', '..', 'grounded_experiment_完整交付包_E1-E14',
                    'data', 'hotpot_validation.parquet')

def r_mds_star(n, eps=EPS, delta=DELTA):
    for r in range(0, int(n) + 1):
        if binom.cdf(r, int(n), eps) >= 1 - delta:
            return r
    return int(n)

def hybrid_ratio(k, eta):
    x = int(round(eta * k)); n = k - x
    r = r_mds_star(n) if n > 0 else 0
    return (x + r) / k

def mixture_curve(sizes, etas):
    sizes = np.asarray(sizes, dtype=int)
    return np.array([np.mean([hybrid_ratio(k, e) for k in sizes]) for e in etas])

rows = duckdb.sql("SELECT supporting_facts['title'] AS g, len(context['title']) AS d "
                  "FROM read_parquet(?)", params=[DATA]).fetchall()
e_q = np.array([len(r[0]) for r in rows], dtype=int)
docs = np.array([r[1] for r in rows], dtype=int)
print(f'e_q mean {e_q.mean():.3f} range {e_q.min()}-{e_q.max()}; docs mean {docs.mean():.2f}')

etas = np.linspace(0, 1, 21)
asym = etas + (1 - etas) * EPS
curves = {'grounded_e': mixture_curve(e_q, etas), 'docs': mixture_curve(docs, etas)}
synth = {k: np.array([hybrid_ratio(k, e) for e in etas]) for k in (20, 50, 100)}

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

FS = (2.1, 2.15)                 # fixed canvas per panel (inches)
RECT = [0.22, 0.19, 0.72, 0.73]    # identical axes rect -> aligned row
LAB, TCK, LEG = 8.0, 7.0, 5.8

def new_panel():
    fig = plt.figure(figsize=FS)
    ax = fig.add_axes(RECT)
    return fig, ax

# ---- (a) regime map ----
fig, ax = new_panel()
ax.plot(etas, asym, 'k--', lw=1.2, label='asymptotic')
ax.plot(etas, curves['grounded_e'], 'o-', color='tab:green', lw=1.5, ms=2.6,
        label=r'grounded $e_q$')
ax.plot(etas, curves['docs'], 's-', color='tab:blue', lw=1.5, ms=2.6,
        label='per-document')
for k, col in zip((20, 50, 100), ['tab:orange', 'tab:purple', 'tab:brown']):
    ax.plot(etas, synth[k], '^-', color=col, lw=1.2, ms=2.6, label=rf'$\kappa={k}$')
ax.set_xlabel(r'transparent fraction $\eta$', fontsize=LAB)
ax.set_ylabel(r'storage ratio $\sigma/\kappa c_B$', fontsize=LAB)
ax.tick_params(labelsize=TCK)
ax.legend(fontsize=LEG, loc='lower right', framealpha=0.95, borderpad=0.3,
          labelspacing=0.25, handlelength=1.4, handletextpad=0.5)
ax.grid(alpha=.3); ax.set_ylim(0, 1.05)
fig.savefig(os.path.join(HERE, 'fig3a_regime_map_scis.pdf'))
fig.savefig(os.path.join(HERE, 'fig3a_regime_map_scis.png'), dpi=200)
plt.close(fig)

# ---- (b) MDS parity requirement ----
mu_vals = np.arange(1, 9)
floor = np.array([r_mds_star(m) for m in mu_vals])
floor_lo = np.array([max(r_mds_star(m) - 1, 0) for m in mu_vals])
fig, ax = new_panel()
ax.fill_between(mu_vals, floor_lo, floor, alpha=0.25, color='tab:green',
                label='converse band')
ax.plot(mu_vals, floor, 'o-', color='tab:green', lw=1.5, ms=2.8,
        label=r'MDS parity $r^*(\mu_q)$')
ax.plot(mu_vals, mu_vals * EPS, 'k--', lw=1.2, label=r'asymptotic $\mu_q\varepsilon$')
ax.set_xlabel(r'certificate complexity $\mu_q$', fontsize=LAB)
ax.set_ylabel(r'MDS parity requirement', fontsize=LAB)
ax.tick_params(labelsize=TCK)
ax.legend(fontsize=LEG, loc='upper left', framealpha=0.95, borderpad=0.3,
          labelspacing=0.25, handlelength=1.4, handletextpad=0.5)
ax.grid(alpha=.3); ax.set_ylim(-0.1, 3.15)
fig.savefig(os.path.join(HERE, 'fig3b_parity_requirement_scis.pdf'))
fig.savefig(os.path.join(HERE, 'fig3b_parity_requirement_scis.png'), dpi=200)
plt.close(fig)

# ---- (c) grounded distribution ----
vals, cnts = np.unique(e_q, return_counts=True)
fig, ax = new_panel()
ax.bar(vals, cnts, color='tab:blue', alpha=0.8)
ax.set_xlabel(r'$\mu_q = e_q$ (grounded)', fontsize=LAB)
ax.set_ylabel('questions', fontsize=LAB)
ax.set_yscale('log')
ax.tick_params(labelsize=TCK)
ax.grid(alpha=.3, axis='y')
fig.savefig(os.path.join(HERE, 'fig3c_grounded_dist_scis.pdf'))
fig.savefig(os.path.join(HERE, 'fig3c_grounded_dist_scis.png'), dpi=200)
plt.close(fig)
print('saved fig3a/fig3b/fig3c pdf+png')

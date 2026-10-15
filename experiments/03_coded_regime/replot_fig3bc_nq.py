#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Replot Fig. 4(b)/(c) [fig3b/fig3c] with designated-certificate size n_q
labels instead of minimum certificate complexity mu_q.  Same data, size,
and style as replot_fig3_panels.py.  Outputs written to this directory and
copied to the manuscript root."""
import os, shutil, sys
import duckdb
import numpy as np
from scipy.stats import binom

sys.path.insert(0, os.path.expanduser('~/Library/Application Support/kimi-desktop/daimon-share/daimon/runtime/python'))

EPS, DELTA = 0.1, 0.02
HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, '..', '..', 'data', 'hotpot_validation.parquet')
ROOT = os.path.join(HERE, '..', '..', '..')

def r_mds_star(n, eps=EPS, delta=DELTA):
    for r in range(0, int(n) + 1):
        if binom.cdf(r, int(n), eps) >= 1 - delta:
            return r
    return int(n)

rows = duckdb.sql("SELECT supporting_facts['title'] AS g FROM read_parquet(?)",
                  params=[DATA]).fetchall()
e_q = np.array([len(r[0]) for r in rows], dtype=int)
print(f'n={len(e_q)}, e_q mean {e_q.mean():.3f} range {e_q.min()}-{e_q.max()}')

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

FS = (2.1, 2.15)
RECT = [0.22, 0.26, 0.72, 0.66]
LAB, TCK, LEG = 8.0, 7.0, 5.8

def new_panel():
    fig = plt.figure(figsize=FS)
    ax = fig.add_axes(RECT)
    return fig, ax

# ---- (a) regime map (unchanged content, aligned axes rect) ----
etas = np.linspace(0, 1, 21)
asym = etas + (1 - etas) * EPS

def hybrid_ratio(k, eta):
    x = int(round(eta * k)); n = k - x
    r = r_mds_star(n) if n > 0 else 0
    return (x + r) / k

def mixture_curve(sizes, etas):
    sizes = np.asarray(sizes, dtype=int)
    return np.array([np.mean([hybrid_ratio(k, e) for k in sizes]) for e in etas])

rows2 = duckdb.sql("SELECT supporting_facts['title'] AS g, len(context['title']) AS d "
                   "FROM read_parquet(?)", params=[DATA]).fetchall()
docs = np.array([r[1] for r in rows2], dtype=int)
curves = {'grounded_e': mixture_curve(e_q, etas), 'docs': mixture_curve(docs, etas)}
synth = {k: np.array([hybrid_ratio(k, e) for e in etas]) for k in (20, 50, 100)}

fig, ax = new_panel()
ax.plot(etas, asym, 'k--', lw=0.6, label='asymptotic')
ax.plot(etas, curves['grounded_e'], 'o-', color='tab:green', lw=0.75, ms=2.6,
        label=r'grounded $e_q$')
ax.plot(etas, curves['docs'], 's-', color='tab:blue', lw=0.75, ms=2.6,
        label='dataset-context records')
for k, col in zip((20, 50, 100), ['tab:orange', 'tab:purple', 'tab:brown']):
    ax.plot(etas, synth[k], '^-', color=col, lw=0.6, ms=2.6, label=rf'$\kappa={k}$')
ax.set_xlabel(r'transparent fraction $\eta$', fontsize=LAB)
ax.set_ylabel(r'storage ratio $\sigma/(\kappa c_B)$', fontsize=LAB)
ax.tick_params(labelsize=TCK)
# compact legend: all curves exceed 0.42 for eta >= 0.36, so a narrower,
# shorter box anchored at the lower-right corner clears every curve;
# final nudge: 0.5 mm right and 0.5 mm down relative to the axes corner
ax.legend(fontsize=4.9, loc='lower right', bbox_to_anchor=(1.013, -0.014),
          framealpha=0.95, borderpad=0.25,
          labelspacing=0.2, handlelength=1.2, handletextpad=0.45)
ax.grid(alpha=.3); ax.set_ylim(0, 1.05)
fig.savefig(os.path.join(HERE, 'fig3a_regime_map_scis.pdf'))
fig.savefig(os.path.join(HERE, 'fig3a_regime_map_scis.png'), dpi=200)
plt.close(fig)

# ---- (b) MDS parity requirement for a designated certificate of size n_q ----
n_vals = np.arange(1, 9)
floor = np.array([r_mds_star(m) for m in n_vals])
floor_lo = np.array([max(r_mds_star(m) - 1, 0) for m in n_vals])
fig, ax = new_panel()
ax.fill_between(n_vals, floor_lo, floor, alpha=0.25, color='tab:green',
                label='large-alphabet reference band')
ax.plot(n_vals, floor, 'o-', color='tab:green', lw=0.75, ms=2.8,
        label=r'MDS parity $r^*(n_q)$')
ax.plot(n_vals, n_vals * EPS, 'k--', lw=0.6, label=r'linear reference $n_q\varepsilon$')
ax.set_xlabel(r'designated certificate size $n_q$', fontsize=LAB)
ax.set_ylabel(r'MDS parity requirement', fontsize=LAB)
ax.tick_params(labelsize=TCK)
# frameless legend: the box edge used to clip the r* plateau; text only
ax.legend(fontsize=LEG, loc='upper left', frameon=False, borderpad=0.3,
          labelspacing=0.25, handlelength=1.4, handletextpad=0.5)
ax.grid(alpha=.3); ax.set_ylim(-0.1, 3.62)
fig.savefig(os.path.join(HERE, 'fig3b_parity_requirement_scis.pdf'))
fig.savefig(os.path.join(HERE, 'fig3b_parity_requirement_scis.png'), dpi=200)
plt.close(fig)

# ---- (c) designated gold evidence-set sizes ----
vals, cnts = np.unique(e_q, return_counts=True)
fig, ax = new_panel()
ax.bar(vals, cnts, color='tab:blue', alpha=0.8)
ax.set_xlabel(r'designated evidence-set size $n_q$', fontsize=LAB)
ax.set_ylabel('questions', fontsize=LAB)
ax.set_yscale('log')
ax.tick_params(labelsize=TCK)
ax.grid(alpha=.3, axis='y')
fig.savefig(os.path.join(HERE, 'fig3c_grounded_dist_scis.pdf'))
fig.savefig(os.path.join(HERE, 'fig3c_grounded_dist_scis.png'), dpi=200)
plt.close(fig)

for f in ('fig3a_regime_map_scis.pdf', 'fig3b_parity_requirement_scis.pdf', 'fig3c_grounded_dist_scis.pdf'):
    shutil.copy(os.path.join(HERE, f), os.path.join(ROOT, f))
print('saved fig3b/fig3c (n_q labels) to experiment dir and manuscript root')

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Replot Fig. 2(c): grounded stress test with trace class-size references.

Reads results_grounded.json (same folder) and ../data/trace_class_sizes.npz.
Only change vs the original rendering: the in-figure annotation now reads
"max. component-size gap" instead of the ambiguous "bias".
"""
import json, os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
d = json.load(open(os.path.join(HERE, 'results_grounded.json')))
tc = np.load(os.path.join(HERE, '..', '..', 'data', 'trace_class_sizes.npz'))

eps = np.array(d['grounded_stress']['eps'])
R_exact = np.array(d['grounded_stress']['R_exact'])
R_sim = np.array(d['grounded_stress']['R_sim_mean'])
R_proxy = np.array(d['proxy_bias']['R_proxy'])
gap = d['proxy_bias']['max_abs_bias']

def trace_curve(key):
    cls = tc[key].astype(float)
    w = cls / cls.sum()                      # prompt-weighted, full exponents
    return np.array([float((w * (1 - e) ** cls).sum()) for e in eps])

sq = trace_curve('sq')
la = trace_curve('la')

import matplotlib
matplotlib.use('Agg')
# plain white style for consistency with the other manuscript figures
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(3.4, 2.6))
ax.fill_between(eps, R_exact, R_proxy, color='#d62728', alpha=0.08, lw=0)
ax.plot(eps, R_proxy, '--', color='#d62728', lw=1.6)
ax.plot(eps, sq, ':', color='#1f77b4', lw=1.6)
ax.plot(eps, la, ':', color='#ff7f0e', lw=1.6)
ax.plot(eps, R_exact, '-', color='#2ca02c', lw=1.8)
ax.plot(eps, R_sim, 'o', color='#2ca02c', mfc='none', ms=5, mew=1.4)
ax.text(0.805, 0.42, 'max. component-size\ngap: %.3f' % gap,
        transform=ax.transAxes, ha='center', va='center', fontsize=6.5,
        color='#a02020', linespacing=1.5)
ax.set_xlabel(r'premise erasure rate $\varepsilon$', fontsize=8)
ax.set_ylabel('certificate survival', fontsize=8)
ax.set_xlim(0, 0.5); ax.set_ylim(0, 1.02)
ax.tick_params(labelsize=7)
ax.grid(alpha=0.25)
fig.tight_layout()
fig.savefig(os.path.join(HERE, 'fig1c_grounded_stress_scis.pdf'), bbox_inches='tight')
fig.savefig(os.path.join(HERE, 'fig1c_grounded_stress_scis.png'), dpi=200, bbox_inches='tight')
print('fig1c replotted; gap =', gap)

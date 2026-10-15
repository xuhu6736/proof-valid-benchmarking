#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Replot fig2b (certificate floor) with 'MDS parity requirement' terminology,
replacing 'MDS floor'/'coded storage floor' labels.  Same data and style as
run_coded_regime_experiment.py fig9; duckdb loader (managed runtime)."""
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

rows = duckdb.sql("SELECT supporting_facts['title'] AS g FROM read_parquet(?)",
                  params=[DATA]).fetchall()
e_q = np.array([len(r[0]) for r in rows], dtype=int)

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

mu_vals = np.arange(1, 9)
floor = np.array([r_mds_star(m) for m in mu_vals])
floor_lo = np.array([max(r_mds_star(m) - 1, 0) for m in mu_vals])
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.0),
                               gridspec_kw={'width_ratios': [1.5, 1]})
ax1.fill_between(mu_vals, floor_lo, floor, alpha=0.25, color='tab:green',
                 label=r'$\sigma^*_{\rm pv}/c_B\in[\,r^*-1,\ r^*\,]$ (Thm. 5)')
ax1.plot(mu_vals, floor, 'o-', color='tab:green', lw=2,
         label=r'MDS parity requirement $r^*_{\rm MDS}(\mu_q)$')
ax1.plot(mu_vals, mu_vals * EPS, 'k--', lw=1.4, label=r'asymptotic $\mu_q\varepsilon$')
ax1.set_xlabel(r'certificate complexity $\mu_q$')
ax1.set_ylabel(r'MDS parity requirement (packets)')
ax1.set_title(rf'(a) Proof-valid MDS parity requirement, $\varepsilon={EPS}$, $\delta={DELTA}$')
ax1.legend(fontsize=8.5); ax1.grid(alpha=.3)
vals, cnts = np.unique(e_q, return_counts=True)
ax2.bar(vals, cnts, color='tab:blue', alpha=0.8)
ax2.set_xlabel(r'$\mu_q = e_q$ (grounded)'); ax2.set_ylabel('questions')
ax2.set_yscale('log')
ax2.set_title('(b) grounded certificate-complexity distribution')
ax2.grid(alpha=.3, axis='y')
fig.tight_layout()
fig.savefig(os.path.join(HERE, 'fig2b_certificate_floor_scis.pdf'), bbox_inches='tight')
fig.savefig(os.path.join(HERE, 'fig2b_certificate_floor_scis.png'), dpi=160, bbox_inches='tight')
print('saved fig2b_certificate_floor_scis.pdf/.png')

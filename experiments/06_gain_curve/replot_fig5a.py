#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Regenerate fig5a_reliability_budget_scis.pdf (paper Fig. 6(a)) directly from
gain_curve_results.json: reliability vs budget B/kappa for MDS parity (solid)
and transparent copies (dashed), eps=0.1.  Only the axis notation (k -> kappa)
and label placement differ from the previous figure; data unchanged."""
import json, os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
res = json.load(open(os.path.join(HERE, 'gain_curve_results.json')))
curves = res['curves']

fig, ax = plt.subplots(figsize=(3.4, 2.8))
# kappa labels sit in the empty band between the solid (MDS) and dashed
# (transparent) families, left of each dashed curve's midpoint -- placing
# them right of the midpoint made them cross their own dashed curves
labelpos = {3: (0.38, 0.875), 10: (0.39, 0.625), 20: (0.40, 0.395), 40: (0.38, 0.15)}
for k, col in zip((3, 10, 20, 40), ('tab:green', 'tab:blue', 'tab:orange', 'tab:red')):
    c = curves[str(k)]
    B = [b / k for b in c['B']]
    ax.plot(B, c['R_C'], '-', color=col, lw=1.8)
    ax.plot(B, c['R_T'], '--', color=col, lw=1.2, alpha=0.65)
    ax.text(labelpos[k][0], labelpos[k][1], rf'$\kappa={k}$', fontsize=8, color=col)
ax.plot([], [], 'k-', lw=1.8, label='MDS parity')
ax.plot([], [], 'k--', lw=1.2, alpha=0.65, label='transparent')
ax.set_xlabel(r'budget $B/\kappa$', fontsize=9)
ax.set_ylabel('reliability', fontsize=9)
ax.tick_params(labelsize=8)
ax.legend(fontsize=7.5, loc='lower right')
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(os.path.join(HERE, 'fig5a_reliability_budget_scis.pdf'), bbox_inches='tight')
fig.savefig(os.path.join(HERE, 'fig5a_reliability_budget_scis.png'), dpi=200, bbox_inches='tight')
print('saved fig5a_reliability_budget_scis.pdf/.png')

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Regenerate fig5b_gain_curve_scis.pdf (paper Fig. 6(b)) with beta notation:
coded-minus-transparent availability at B = round(beta*kappa), eps = 0.1.
R_T = (1-eps)^{kappa-B}, R_C = Pr[Binomial(kappa, eps) <= B]  (paper Sec. 6.2).
Anchor values verified against gain_curve_results.json gain_at_0.3budget:
kappa=3,10,20,40 -> gains 0.162, 0.509, 0.769, 0.948."""
import json, os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import binom

HERE = os.path.dirname(os.path.abspath(__file__))
EPS = 0.1

ks = np.arange(1, 61)
fig, ax = plt.subplots(figsize=(3.4, 2.8))
for beta, col in zip((0.2, 0.3, 0.5), ('tab:blue', 'tab:orange', 'tab:green')):
    gains = []
    for k in ks:
        B = int(round(beta * k))
        r_t = (1 - EPS) ** (k - B)
        r_c = binom.cdf(B, k, EPS)
        gains.append(r_c - r_t)
    ax.plot(ks, gains, '-', color=col, lw=1.8, label=rf'$\beta={beta}$')

# anchor check against the artifact
ref = json.load(open(os.path.join(HERE, 'gain_curve_results.json')))['gain_at_0.3budget']
for k, target in ((3, 0.162), (10, 0.509), (20, 0.769), (40, 0.948)):
    B = int(round(0.3 * k))
    g = binom.cdf(B, k, EPS) - (1 - EPS) ** (k - B)
    assert abs(g - target) < 2e-3, (k, g, target)
    assert abs(g - (ref[str(k)]['R_C'] - ref[str(k)]['R_T'])) < 1e-9
print('[check] anchors match gain_curve_results.json')

# label the dashed marker line at its top end: the strip y > 0.93 is empty
# for kappa <= 31 (all curves stay <= 0.90 there)
ax.axvline(2.43, color='tab:gray', ls='--', lw=1.1)
ax.text(3.2, 0.97, 'mean annotated $e_q\\approx 2.4$', fontsize=7,
        color='dimgray', va='top', ha='left')
ax.axhline(0, color='k', lw=0.8)
ax.set_xlabel('evidence size $\\kappa$', fontsize=9)
ax.set_ylabel('coded gain', fontsize=9)
ax.set_xticks([1, 20, 40, 60])
ax.set_xticklabels(['1', '20', '40', '60'])
ax.tick_params(labelsize=8)
# raised ~0.5 cm so the box clears the y = 0 reference line below
ax.legend(fontsize=8, loc='lower right', bbox_to_anchor=(0.98, 0.12))
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(os.path.join(HERE, 'fig5b_gain_curve_scis.pdf'), bbox_inches='tight')
fig.savefig(os.path.join(HERE, 'fig5b_gain_curve_scis.png'), dpi=200, bbox_inches='tight')
print('saved fig5b_gain_curve_scis.pdf/.png')

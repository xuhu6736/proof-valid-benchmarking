#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Replot Fig. 9 (error--throughput frontiers) as four separate panels for a
1x4 subfloat row (same spec as the Fig. 7/8 panel style).  Data: archived
e10_pareto_results.json only.  Panel order: (a) tau_c=0.65 eps=0.1,
(b) tau_c=0.65 eps=0.3, (c) tau_c=0.75 eps=0.1, (d) tau_c=0.75 eps=0.3.
Marker meaning is carried by the main caption (red circles similarity-only,
blue squares proof-valid), so panels carry no legend; the equal-threshold
arrow at tau=0.9 connects the two policies' actual operating points
(E_SO, T_SO) -> (0, T_PV); the 2% false-hit reference line is removed
(the x-axis is total error, not a false-hit axis)."""
import json
import os
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
D = json.load(open(os.path.join(HERE, 'e10_pareto_results.json')))

PANELS = [('a', 0.65, 0.1), ('b', 0.65, 0.3),
          ('c', 0.75, 0.1), ('d', 0.75, 0.3)]

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

FS = (1.56, 2.0)                 # 0.242\textwidth at \textwidth ~= 6.46in
RECT = [0.25, 0.175, 0.71, 0.775]
LAB, TCK, ANN = 6.5, 6.0, 5.8

for tag, ct, eps in PANELS:
    d = D[str(ct)]['eps'][str(eps)]
    so_err = np.array(d['sim_only_error']); so_thr = np.array(d['sim_only_throughput'])
    pv_err = np.array(d['pv_error']);       pv_thr = np.array(d['pv_throughput'])
    taus = np.array(d['taus'])

    fig = plt.figure(figsize=FS)
    a = fig.add_axes(RECT)
    a.plot(so_err, so_thr, 'o-', color='tab:red', ms=2.4, lw=1.0)
    a.plot(pv_err, pv_thr, 's-', color='tab:blue', ms=2.4, lw=1.0)
    i9 = int(np.argmin(np.abs(taus - 0.9)))
    a.annotate('', xy=(pv_err[i9] + 0.004, pv_thr[i9]),
               xytext=(so_err[i9], so_thr[i9]),
               arrowprops=dict(arrowstyle='->', color='tab:blue', lw=1.0))
    a.annotate(f'{so_err[i9]:.3f}$\\to${pv_err[i9]:.3f}',
               xy=(0.42, 0.30), xycoords='axes fraction',
               fontsize=ANN, color='tab:blue')
    xmax = max(so_err.max(), 0.05) * 1.18
    ymin = min(pv_thr.min(), so_thr.min()) - 0.045
    a.set_xlim(-xmax * 0.03, xmax)
    a.set_ylim(ymin, 1.03)
    a.set_xlabel('premise-error mass', fontsize=LAB)
    a.set_ylabel('cache-served fraction', fontsize=LAB)
    a.tick_params(labelsize=TCK)
    a.xaxis.set_major_locator(MaxNLocator(4))
    a.yaxis.set_major_locator(MaxNLocator(4))
    if tag == 'c':  # keep y tick labels 3 chars wide so the ylabel fits
        a.set_yticks([0.8, 0.9, 1.0])
    a.grid(alpha=0.3, lw=0.5)
    for ext in ('pdf', 'png'):
        fig.savefig(os.path.join(HERE, f'fig9{tag}_pareto_scis.{ext}'),
                    dpi=300 if ext == 'png' else None)
    plt.close(fig)
    print(f'saved fig9{tag}_pareto_scis.pdf/.png  (tau_c={ct}, eps={eps})')

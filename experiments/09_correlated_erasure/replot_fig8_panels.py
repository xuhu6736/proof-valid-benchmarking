#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Replot Fig. 8 (correlated premise erasure) as three standalone single-axes
panels at final print size (~0.325 textwidth each), identical axes rects,
legends inside the frames with zero curve overlap.  Data: archived
matched_marginal_rerun/e9_correlated_results_matched.json (the
matched-marginal burst arm that the paper's figure and text report;
iid/doc arms identical to the base JSON).  No re-run."""
import json, os
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
R = json.load(open(os.path.join(HERE, 'matched_marginal_rerun',
                                'e9_correlated_results_matched.json')))
EPS = R['hotpotqa']['eps']

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

FS = (2.1, 2.15)
RECT = [0.235, 0.19, 0.715, 0.73]
LAB, TCK, LEG = 8.0, 7.0, 5.8

def new_panel():
    fig = plt.figure(figsize=FS)
    return fig, fig.add_axes(RECT)

# ---- (a) HotpotQA: correlated erasure ----
d = R['hotpotqa']
fig, ax = new_panel()
ax.plot(EPS, d['pred_iid'], 'k-', lw=1.5,
        label=r'leaf-iid $(1{-}\varepsilon)^{e_q}$')
ax.plot(EPS, d['meas_iid'], 'ko', mfc='none', ms=3.6, lw=1.1)
ax.plot(EPS, d['pred_doc'], '-', color='tab:red', lw=1.5,
        label=r'doc-corr. $(1{-}\varepsilon)^{g_q}$')
ax.plot(EPS, d['meas_doc'], 's', color='tab:red', mfc='none', ms=3.6, lw=1.1)
ax.plot(EPS, d['meas_cluster'], '^--', color='tab:orange', ms=3.6, lw=1.3,
        label='burst (matched, meas.)')
ax.set_xlabel(r'erasure rate $\varepsilon$', fontsize=LAB)
ax.set_ylabel('certificate availability', fontsize=LAB)
ax.tick_params(labelsize=TCK)
ax.set_ylim(0, 1.02); ax.grid(alpha=.3)
ax.legend(fontsize=LEG, loc='upper right', framealpha=0.95, borderpad=0.3,
          labelspacing=0.2, handlelength=1.4, handletextpad=0.5)
fig.savefig(os.path.join(HERE, 'fig8a_hotpotqa_scis.pdf'))
fig.savefig(os.path.join(HERE, 'fig8a_hotpotqa_scis.png'), dpi=200)
plt.close(fig)

# ---- (b) iid-prediction bias under doc erasure (HotpotQA) ----
bias = d['iid_bias_under_doc']
fig, ax = new_panel()
ax.bar(range(len(EPS)), bias, color='tab:red', alpha=0.75)
ax.set_xticks(range(len(EPS)))
ax.set_xticklabels([str(e) for e in EPS])
ax.set_xlabel(r'erasure rate $\varepsilon$', fontsize=LAB)
ax.set_ylabel('group $-$ leaf-iid', fontsize=7.5)
ax.tick_params(labelsize=TCK)
ax.grid(alpha=.3, axis='y')
ax.set_ylim(0, 0.068)
fig.savefig(os.path.join(HERE, 'fig8b_bias_scis.pdf'))
fig.savefig(os.path.join(HERE, 'fig8b_bias_scis.png'), dpi=200)
plt.close(fig)

# ---- (c) MuSiQue control ----
d = R['musique']
fig, ax = new_panel()
ax.plot(EPS, d['pred_iid'], 'k-', lw=1.5, label='leaf-iid law')
ax.plot(EPS, d['meas_iid'], 'ko', mfc='none', ms=3.6, lw=1.1,
        label='iid (meas.)')
ax.plot(EPS, d['meas_doc'], 's--', color='tab:red', mfc='none', ms=3.6, lw=1.1,
        label='doc-corr. (meas.)')
ax.plot(EPS, d['meas_cluster'], '^--', color='tab:orange', mfc='none', ms=3.6,
        lw=1.3, label='fixed-size burst (meas.)')
ax.set_xlabel(r'erasure rate $\varepsilon$', fontsize=LAB)
ax.set_ylabel('certificate availability', fontsize=LAB)
ax.tick_params(labelsize=TCK)
ax.set_ylim(0, 1.02); ax.grid(alpha=.3)
ax.legend(fontsize=LEG, loc='upper right', framealpha=0.95, borderpad=0.3,
          labelspacing=0.2, handlelength=1.4, handletextpad=0.5)
fig.savefig(os.path.join(HERE, 'fig8c_musique_scis.pdf'))
fig.savefig(os.path.join(HERE, 'fig8c_musique_scis.png'), dpi=200)
plt.close(fig)
print('saved fig8a/fig8b/fig8c pdf+png')

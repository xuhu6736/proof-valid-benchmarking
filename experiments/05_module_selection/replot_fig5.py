#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Replot Fig. 5 (module savings) without the constant greedy/optimum ratio axis.

Reads module_selection_results.json (HotpotQA) and e12_module_musique_results.json
(MuSiQue) from the same folder.  The greedy/optimum ratio is identically one on
every audited component and is stated in text and caption; the twin axis carried
no information and is removed.
"""
import json, os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
hp = json.load(open(os.path.join(HERE, 'module_selection_results.json')))
mq = json.load(open(os.path.join(HERE, 'e12_module_musique_results.json')))

rhos = [r['rho'] for r in hp['per_rho']]
hp_sav = [r['saving_vs_leaf_only'] * 100 for r in hp['per_rho']]
mq_sav = [r['saving_vs_leaf_only'] * 100 for r in mq['per_rho']]
x = np.arange(len(rhos))

import matplotlib
matplotlib.use('Agg')
# plain white style for consistency with the other manuscript figures
import matplotlib.pyplot as plt

def style(ax):
    ax.set_xticks(x)
    ax.set_xticklabels(['0.5', '1', '2', '4'])
    ax.set_xlabel(r'root-cost weight $\rho$', fontsize=8)
    ax.set_ylabel('saving (%)', fontsize=8)
    ax.tick_params(labelsize=7)
    ax.grid(alpha=0.25)

fig, ax = plt.subplots(figsize=(2.9, 2.4))
ax.plot(x, hp_sav, 'o-', color='#1f77b4', lw=0.8, ms=4.5, label='optimal saving')
ax.legend(fontsize=7, loc='upper right')
style(ax)
ax.set_ylim(-3, 55)
fig.tight_layout()
fig.savefig(os.path.join(HERE, 'fig4a_module_hotpotqa_scis.pdf'), bbox_inches='tight')
fig.savefig(os.path.join(HERE, 'fig4a_module_hotpotqa_scis.png'), dpi=200, bbox_inches='tight')
plt.close(fig)

fig, ax = plt.subplots(figsize=(2.9, 2.4))
ax.plot(x, mq_sav, 's-', color='#2ca02c', lw=0.8, ms=4.5, label='MuSiQue')
ax.plot(x, hp_sav, 'o-', color='#1f77b4', lw=0.8, ms=4.5, label='HotpotQA')
ax.legend(fontsize=7, loc='upper right')
style(ax)
ax.set_ylim(-4, 84)
fig.tight_layout()
fig.savefig(os.path.join(HERE, 'fig4b_module_musique_scis.pdf'), bbox_inches='tight')
fig.savefig(os.path.join(HERE, 'fig4b_module_musique_scis.png'), dpi=200, bbox_inches='tight')
print('fig4a/fig4b replotted without ratio axis')

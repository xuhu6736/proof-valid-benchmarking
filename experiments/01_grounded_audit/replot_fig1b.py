#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Regenerate fig1b_joint_distribution_scis.pdf (Fig. 2(b) of the paper):
joint distribution of embedding-component size |C(q)| (tau=0.75) vs
annotated evidence count e_q on HotpotQA validation.  In-figure text
corrected: the cell counts sum to n=7,405 questions; 7,331 is the
number of embedding components.  Data: ../../data/ (parquet + npy);
Spearman value cross-checked against results_grounded.json."""
import json, os
import numpy as np
import duckdb
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from scipy.stats import spearmanr

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, '..', '..', 'data')
TAU = 0.75

df = duckdb.sql(
    "SELECT supporting_facts['title'] AS t, supporting_facts['sent_id'] AS s "
    "FROM read_parquet(?)",
    params=[os.path.join(DATA, 'hotpot_validation.parquet')]).fetchall()
e_q = np.array([len(set(zip(t, s))) for t, s in df])
n = len(e_q)

emb = np.load(os.path.join(DATA, 'question_emb.npy')).astype(np.float32)
emb = emb / (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-12)
S = emb @ emb.T
S[S < TAU] = 0
nc, labels = connected_components(csr_matrix(S), directed=False)
cs_q = np.bincount(labels)[labels]

rho_q, p_q = spearmanr(cs_q, e_q)
ref = json.load(open(os.path.join(HERE, 'results_grounded.json')))
rho_ref = ref['embedding_class_audit'][str(TAU)]['question_level_spearman']
print(f'[check] n={n}, components={nc}, Spearman rho={rho_q:.5f} '
      f'(artifact {rho_ref[0]:.5f}, p={rho_ref[1]:.4g})')
assert n == 7405 and nc == 7331 and abs(rho_q - rho_ref[0]) < 1e-9

joint = np.zeros((5, 8), dtype=int)
for a_, b_ in zip(np.clip(cs_q, 1, 5), e_q):
    joint[a_ - 1, b_ - 1] += 1

fig = plt.figure(figsize=(3.0, 2.25))
ax = fig.add_axes([0.16, 0.16, 0.68, 0.76])
im = ax.imshow(joint.T, origin='lower', aspect='auto', cmap='viridis',
               extent=[0.5, 5.5, 0.5, 8.5], norm=LogNorm(vmin=1, vmax=joint.max()))
for i in range(5):
    for j in range(8):
        v = joint[i, j]
        if v > 0:
            ax.text(i + 1, j + 1, f'{v:,}', ha='center', va='center',
                    fontsize=5.4,
                    color='white' if v < joint.max() * 0.45 else '0.15')
ax.set_xticks([1, 2, 3, 4, 5]); ax.set_xticklabels(['1', '2', '3', '4', '5+'])
ax.set_yticks(range(1, 9))
ax.set_xlabel(r'embedding-class size $|C(q)|$ ($\tau=0.75$)', fontsize=7)
ax.set_ylabel(r'evidence count $e_q$', fontsize=7)
ax.tick_params(labelsize=6)
ax.text(0.50, 0.95,
        f'Spearman $\\rho={rho_q:.3f}$\n$n={n:,}$ questions\n{nc:,} components',
        transform=ax.transAxes, fontsize=5.8, color='black', va='top',
        ha='left', bbox=dict(facecolor='white', alpha=0.75,
                             edgecolor='none', pad=1.2))
cb = fig.colorbar(im, ax=ax, pad=0.02)
cb.ax.tick_params(labelsize=6)
cb.set_label('questions', fontsize=6.5)

for ext in ('pdf', 'png'):
    fig.savefig(os.path.join(HERE, f'fig1b_joint_distribution_scis.{ext}'),
                dpi=300 if ext == 'png' else None)
print('saved fig1b_joint_distribution_scis.pdf/.png')

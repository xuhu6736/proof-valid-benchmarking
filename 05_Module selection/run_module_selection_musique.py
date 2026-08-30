#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
X5b: Module selection on MuSiQue (generalization check for E5).

E5 solved module selection exactly on HotpotQA's evidence-sharing components
(742 components, greedy = optimal everywhere, savings up to 44.6%).  A natural
worry is dataset specificity: HotpotQA's sharing structure is sparse.  MuSiQue
is composition-constructed from single-hop questions, so its paragraph pool is
shared BY CONSTRUCTION across multi-hop questions -- the ideal stress test for
generality.  Same protocol as E5: connected components of the question-sharing
graph over SUPPORTING paragraphs; exact brute-force optimum vs density-greedy;
root cost rho in {0.5, 1, 2, 4}.

Outputs: module_selection_musique_results.json, fig18_module_musique_scis.pdf/.png
"""
import json, os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
RHOS = [0.5, 1.0, 2.0, 4.0]


def load():
    q_docs = {}
    with open(os.path.join(DATA, 'musique_full_dev.jsonl')) as f:
        for line in f:
            r = json.loads(line)
            if not r.get('answerable', True):
                continue
            sup = {p['title'] for p in r['paragraphs'] if p.get('is_supporting', False)}
            if sup:
                q_docs[r['id']] = sup
    return q_docs


def build(q_docs):
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import connected_components
    doc_qs = {}
    for q, ds in q_docs.items():
        for d in ds:
            doc_qs.setdefault(d, []).append(q)
    singles = {d: qs for d, qs in doc_qs.items() if len(qs) >= 2}
    qs = list(q_docs)
    qi = {q: i for i, q in enumerate(qs)}
    rows, cols = [], []
    for d, members in singles.items():
        for a in members:
            for b in members:
                if qi[a] < qi[b]:
                    rows.append(qi[a]); cols.append(qi[b])
    A = csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(len(qs), len(qs)))
    n_comp, labels = connected_components(A, directed=False)
    comps = []
    for c in range(n_comp):
        members = [qs[i] for i in np.nonzero(labels == c)[0]]
        if len(members) >= 2:
            comps.append(members)
    return singles, comps


EXACT_CAP = 20  # brute-force only up to 20 candidate modules; LP bound beyond


def lp_bound(qs, q_docs, docs, rho, leaves):
    """LP relaxation of the module-selection set cover: certified lower bound."""
    from scipy.optimize import linprog
    nD, nQ = len(docs), len(qs)
    # vars: x_d (deploy module d), y_{qd} (leaf (q,d) stored directly)
    pairs = [(qi, d) for qi, q in enumerate(qs) for d in leaves[q]]
    nY = len(pairs)
    c = np.concatenate([np.full(nD, rho), np.ones(nY)])
    A_ub, b_ub = [], []
    di = {d: i for i, d in enumerate(docs)}
    for j, (qi, d) in enumerate(pairs):
        row = np.zeros(nD + nY)
        if d in di:
            row[di[d]] = -1
        row[nD + j] = -1
        A_ub.append(row); b_ub.append(-1.0)
    res = linprog(c, A_ub=np.array(A_ub), b_ub=np.array(b_ub),
                  bounds=(0, 1), method='highs')
    return res.fun if res.success else 0.0


def solve(qs, q_docs, singles, rho):
    docs = sorted({d for q in qs for d in q_docs[q] if d in singles})
    leaves = {q: sorted(q_docs[q]) for q in qs}
    leaf_only = sum(len(v) for v in leaves.values())

    def cost(modset):
        c = len(modset) * rho
        for q in qs:
            c += sum(0 if d in modset else 1 for d in leaves[q])
        return c

    # density greedy
    remaining = set(qs); modset = set()
    while True:
        best, best_gain = None, 0
        for d in docs:
            if d in modset:
                continue
            cover = [q for q in remaining if d in q_docs[q]]
            gain = len(cover) - rho
            if gain > best_gain:
                best, best_gain = d, gain
        if best is None or best_gain <= 0:
            break
        modset.add(best)
        remaining -= {q for q in remaining if best in q_docs[q]}
    greedy = cost(modset)

    if len(docs) <= EXACT_CAP:
        # brute force with pruning -> exact optimum
        best_c = greedy
        docs_sorted = sorted(docs, key=lambda d: -sum(d in q_docs[q] for q in qs))
        def rec(i, chosen, c_so_far):
            nonlocal best_c
            if c_so_far >= best_c:
                return
            if i == len(docs_sorted):
                best_c = min(best_c, c_so_far + sum(
                    sum(0 if d in chosen else 1 for d in leaves[q]) for q in qs))
                return
            rec(i + 1, chosen, c_so_far)
            rec(i + 1, chosen | {docs_sorted[i]}, c_so_far + rho)
        rec(0, set(), 0.0)
        return leaf_only, greedy, best_c, True
    # large component: LP certified lower bound
    lb = lp_bound(qs, q_docs, docs, rho, leaves)
    return leaf_only, greedy, lb, False


def main():
    q_docs = load()
    singles, comps = build(q_docs)
    sizes = [len(c) for c in comps]
    print(f"[X5b] {len(singles)} shared supporting docs, {len(comps)} components "
          f"covering {sum(sizes)} questions (sizes {min(sizes)}-{max(sizes)})")
    out = {'n_modules': len(singles), 'n_components': len(comps),
           'questions_in_components': int(sum(sizes)), 'per_rho': []}
    for rho in RHOS:
        lo = gr = op = 0.0
        worst = 1.0
        worst_ex = 1.0
        n_sub = 0
        n_exact = 0
        lo_ex = gr_ex = op_ex = 0.0
        for members in comps:
            l, g, o, exact = solve(members, q_docs, singles, rho)
            lo += l; gr += g; op += o
            n_exact += exact
            if exact:
                lo_ex += l; gr_ex += g; op_ex += o
                if g > o + 1e-9:
                    worst_ex = max(worst_ex, g / o)
            if g > o + 1e-9:
                n_sub += 1
                worst = max(worst, g / o)
        out['per_rho'].append(dict(
            rho=rho, leaf_only=lo, greedy=gr, optimal_or_lp=op,
            n_exact_components=n_exact,
            saving=1 - op / lo if lo else 0.0,
            greedy_over_optimal=gr / op if op else 1.0,
            greedy_over_optimal_exact_subset=(gr_ex / op_ex if op_ex else 1.0),
            worst_instance_ratio_exact=worst_ex,
            worst_instance_ratio=worst, n_suboptimal=n_sub))
        print(f"  rho={rho}: leaf-only {lo:.0f}, optimal/LP {op:.0f} "
              f"(saving {(1-op/lo)*100:.1f}%), greedy/opt {gr/op:.4f}, "
              f"exact-only greedy/opt {gr_ex/op_ex:.4f} (worst {worst_ex:.3f}), "
              f"exact components {n_exact}/{len(comps)}", flush=True)
    json.dump(out, open('module_selection_musique_results.json', 'w'), indent=2)

    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    rhos = [p['rho'] for p in out['per_rho']]
    sav = [p['saving'] * 100 for p in out['per_rho']]
    ax.plot(rhos, sav, 'o-', color='#2ca02c', label='MuSiQue optimal saving')
    ax.set_xlabel('module root cost $\\rho$ (leaf-packet units)')
    ax.set_ylabel('storage saving (%)', color='#2ca02c')
    ax2 = ax.twinx()
    rat = [p['greedy_over_optimal_exact_subset'] for p in out['per_rho']]
    ax2.plot(rhos, rat, 's--', color='#ff7f0e', label='greedy / optimum')
    ax2.set_ylabel('greedy / optimum', color='#ff7f0e')
    ax2.set_ylim(0.95, 2.1)
    ax.set_title(f'Module selection on MuSiQue ({len(comps)} components, '
                 f'{len(singles)} modules)')
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig('fig18_module_musique_scis.pdf'); fig.savefig('fig18_module_musique.png', dpi=200)
    print("[X5b] fig18_module_musique_scis.pdf saved")


if __name__ == '__main__':
    main()

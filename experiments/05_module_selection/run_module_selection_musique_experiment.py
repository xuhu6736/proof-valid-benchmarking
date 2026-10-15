#!/usr/bin/env python3
"""E12: module selection on MuSiQue evidence-sharing structure
(cross-corpus generalization check for E5).

MuSiQue gives each question 20 paragraphs with is_supporting flags; the
paragraph pool is shared across questions, so the evidence-sharing graph is
denser than HotpotQA's (which has 99.1% singleton classes).  We rebuild the
module-selection instances (connected components of the question-sharing
graph over supporting paragraphs, identified by paragraph idx) and re-run the
exact-vs-greedy comparison at rho in {0.5, 1, 2, 4}, alongside the HotpotQA
numbers from module_selection_results.json.

Outputs: e12_module_musique_results.json, fig17_module_musique_scis.pdf/.png
"""
import json
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from itertools import combinations

RHO_LIST = [0.5, 1.0, 2.0, 4.0]
MQ = 'data/musique_full_dev.jsonl'
OUT_JSON = 'e12_module_musique_results.json'
OUT_FIG = 'fig17_module_musique_scis'


def build():
    """q_docs: question -> set of supporting paragraph ids (global idx).
    Paragraph identity: (question_idx, paragraph_idx) is unique per question,
    but paragraphs are shared only if the SAME text recurs; MuSiQue assigns
    each paragraph a stable idx within a question, so sharing is established
    by identical paragraph_text.  Use text hash for cross-question identity."""
    q_docs, pidx_of = [], {}
    for line in open(MQ):
        r = json.loads(line)
        if not r.get('answerable', True):
            continue
        docs = set()
        for p in r['paragraphs']:
            if p.get('is_supporting'):
                key = p['paragraph_text'][:200]  # stable text prefix as id
                if key not in pidx_of:
                    pidx_of[key] = len(pidx_of)
                docs.add(pidx_of[key])
        if docs:
            q_docs.append(docs)
    # shared docs: used by >=2 questions
    from collections import Counter
    use = Counter()
    for d in q_docs:
        for x in d:
            use[x] += 1
    singles = {x for x, c in use.items() if c >= 2}
    # sharing graph components
    n = len(q_docs)
    rows, cols = [], []
    by_doc = {}
    for i, d in enumerate(q_docs):
        for x in d & singles:
            by_doc.setdefault(x, []).append(i)
    for x, qs in by_doc.items():
        for a, b in combinations(qs, 2):
            rows.append(a); cols.append(b)
    A = csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(n, n))
    nc, labels = connected_components(A, directed=False)
    comps = {}
    for i, c in enumerate(labels):
        comps.setdefault(c, []).append(i)
    comps = [c for c in comps.values() if len(c) >= 2]
    return q_docs, singles, comps


def solve(qs, q_docs, singles, rho):
    """Exact (brute force) vs density-greedy cost on one component."""
    cand = sorted({x for i in qs for x in q_docs[i] & singles})
    leaf = sum(len(q_docs[i]) for i in qs)

    def cost(chosen):
        c = rho * len(chosen)
        for i in qs:
            c += len(q_docs[i] - set(chosen))
        return c

    best_c, best_k = leaf, 0
    for k in range(1, len(cand) + 1):
        if k * rho >= best_c:
            break
        for comb in combinations(cand, k):
            best_c = min(best_c, cost(list(comb)))
    # density greedy
    chosen, covered = [], {i: set() for i in qs}
    cur = leaf
    while True:
        best_gain, best_x = 0, None
        for x in cand:
            if x in chosen:
                continue
            new_cov = sum(1 for i in qs if x in q_docs[i] and x not in covered[i])
            gain = new_cov - rho
            if gain > best_gain:
                best_gain, best_x = gain, x
        if best_x is None:
            break
        chosen.append(best_x)
        for i in qs:
            covered[i].add(best_x)
        cur = cost(chosen)
    return leaf, cur, best_c


def main():
    q_docs, singles, comps = build()
    n_q = sum(len(c) for c in comps)
    print(f'[E12] {len(singles)} shared-paragraph modules, {len(comps)} '
          f'components covering {n_q} questions '
          f'(sizes {min(len(c) for c in comps)}-{max(len(c) for c in comps)})')
    big = [c for c in comps if len(c) > 9]
    print(f'[E12] components with >9 questions (exhaustive cap): {len(big)}')

    per_rho = []
    for rho in RHO_LIST:
        L = G = O = 0
        worst = 1.0
        n_sub = 0
        for qs in comps:
            cand = {x for i in qs for x in q_docs[i] & singles}
            if len(cand) > 20 or len(qs) > 9:
                # greedy only for very large components (rare)
                leaf, g, o = solve_greedy_only(qs, q_docs, singles, rho)
                L += leaf; G += g; O += g
                continue
            leaf, g, o = solve(qs, q_docs, singles, rho)
            L += leaf; G += g; O += o
            if o > 0 and g / o > worst + 1e-9:
                worst = g / o
            if g > o + 1e-9:
                n_sub += 1
        per_rho.append({'rho': rho, 'leaf_only_cost': L, 'greedy_cost': G,
                        'optimal_cost': O, 'saving_vs_leaf_only': 1 - O / L,
                        'greedy_over_optimal': G / O if O else 1.0,
                        'worst_instance_ratio': worst, 'n_suboptimal': n_sub})
        print(f'[E12] rho={rho}: leaf {L}, optimal {O} '
              f'(saving {100*(1-O/L):.1f}%), greedy/opt {G/O:.4f}, '
              f'worst {worst:.3f}')

    json.dump({'n_modules': len(singles), 'n_components': len(comps),
               'questions_in_components': n_q, 'rho_list': RHO_LIST,
               'per_rho': per_rho}, open(OUT_JSON, 'w'), indent=2)
    print(f'[E12] {OUT_JSON} saved')

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    hp = json.load(open('module_selection_results.json'))
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    ax2 = ax.twinx()
    ax.plot(RHO_LIST, [100 * p['saving_vs_leaf_only'] for p in per_rho],
            'o-', color='tab:green', lw=2, label='MuSiQue optimal saving')
    ax.plot(RHO_LIST, [100 * p['saving_vs_leaf_only'] for p in hp['per_rho']],
            's--', color='tab:blue', lw=1.6, label='HotpotQA optimal saving (E5)')
    ax2.plot(RHO_LIST, [p['greedy_over_optimal'] for p in per_rho],
             'v:', color='tab:orange', label='MuSiQue greedy/optimal')
    ax2.plot(RHO_LIST, [p['greedy_over_optimal'] for p in hp['per_rho']],
             '^:', color='tab:red', label='HotpotQA greedy/optimal (E5)')
    ax.set_xlabel('module root cost $\\rho$ (leaf-packet units)')
    ax.set_ylabel('storage saving vs leaf-only (%)', color='tab:green')
    ax2.set_ylabel('greedy / optimum', color='tab:orange')
    ax.set_ylim(-2, 80); ax2.set_ylim(0.99, 1.06)
    ax.grid(alpha=0.3)
    l1, lb1 = ax.get_legend_handles_labels()
    l2, lb2 = ax2.get_legend_handles_labels()
    ax.legend(l1 + l2, lb1 + lb2, fontsize=7.5, loc='upper right')
    ax.set_title(f'E12 module selection on MuSiQue sharing structure\n'
                 f'({len(comps)} components, {len(singles)} modules, '
                 f'{n_q} questions)', fontsize=10)
    fig.tight_layout()
    fig.savefig(OUT_FIG + '.pdf'); fig.savefig(OUT_FIG + '.png', dpi=140)
    print(f'[E12] {OUT_FIG}.pdf/.png saved')


def solve_greedy_only(qs, q_docs, singles, rho):
    cand = sorted({x for i in qs for x in q_docs[i] & singles})
    leaf = sum(len(q_docs[i]) for i in qs)
    chosen = []
    def cost(ch):
        return rho * len(ch) + sum(len(q_docs[i] - set(ch)) for i in qs)
    cur = leaf
    while True:
        best_gain, best_x = 0, None
        for x in cand:
            if x not in chosen:
                gain = sum(1 for i in qs if x in q_docs[i]) - rho
                if gain > best_gain:
                    best_gain, best_x = gain, x
        if best_x is None:
            break
        chosen.append(best_x)
        cur = cost(chosen)
    return leaf, cur, cur


if __name__ == '__main__':
    main()

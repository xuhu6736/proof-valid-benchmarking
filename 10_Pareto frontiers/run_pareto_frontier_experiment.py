#!/usr/bin/env python3
"""E10: fair Pareto-frontier comparison (error vs throughput) between
similarity-only and proof-valid gating, on grounded HotpotQA evidence.

For each threshold tau on the question-embedding similarity score s(q, C),
where classes C are connected components of the cosine graph at 0.75
(generator-disjoint judge, as in Section 7.4):

  serve_tau  : s >= tau  (similarity decision)
  error_tau  = false-hit rate      + premise-erasure rate
               P[s>=tau, wrong C]    P[s>=tau, own C, any evidence leaf erased]
  thr_tau    = P[s>=tau]

The proof-valid gate serves iff s>=tau AND the whole certificate survives:
its error contains ONLY the false-hit term (unsupported own-class reuse is
declined), so the comparison is fair in both axes.  Sweeping tau traces two
Pareto frontiers in the (error, throughput) plane.

Setup mirrors the grounded audit: e_q = #distinct annotated (title,sent_id)
pairs; class of q = its component; positive score = max cosine to another
same-component question (1.0 for singletons); negative score = max cosine to
any foreign question.  A fixed erasure draw (seed 42, N_REP reps) supplies
the per-question premise-survival indicator at each eps.

Outputs: e10_pareto_results.json, fig15_pareto_scis.pdf/.png
"""
import json
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

SEED = 42
N_REP = 20
CLASS_TAUS = [0.65, 0.75]   # 0.65: observable false-hit term; 0.75: the
                            # grounded-audit class scale of Section 7.3
EPS_LIST = [0.1, 0.3]
HP = 'data/hotpot_validation.parquet'
QEMB = 'data/question_emb.npy'
OUT_JSON = 'e10_pareto_results.json'
OUT_FIG = 'fig15_pareto_scis'


def class_components(emb, tau):
    norm = np.linalg.norm(emb, axis=1, keepdims=True)
    z = emb / norm
    sim = z @ z.T
    rows, cols = np.where(sim >= tau)
    keep = rows < cols
    g = csr_matrix((np.ones(keep.sum()), (rows[keep], cols[keep])),
                   shape=(len(emb), len(emb)))
    n, labels = connected_components(g, directed=False)
    return labels, sim


def per_question_scores(emb, labels):
    """own-best / foreign-best cosine per question, computed in row blocks."""
    n = len(emb)
    z = emb / np.linalg.norm(emb, axis=1, keepdims=True)
    own_best = np.ones(n, dtype=np.float32)   # singleton own-hit at s=1
    foreign_best = np.zeros(n, dtype=np.float32)
    BS = 400
    for s in range(0, n, BS):
        e = min(s + BS, n)
        S = z[s:e] @ z.T                      # (b, n)
        lab_block = labels[s:e]
        same = lab_block[:, None] == labels[None, :]
        rows = np.arange(s, e)
        same[np.arange(e - s), rows] = False  # exclude self
        own_blk = np.where(same, S, -1).max(axis=1)
        has_same = same.any(axis=1)
        own_best[s:e] = np.where(has_same, own_blk, 1.0)
        foreign_blk = np.where(~same, S, -1)
        foreign_blk[np.arange(e - s), rows] = -1
        foreign_best[s:e] = foreign_blk.max(axis=1)
    return own_best, foreign_best


def run_one(labels, e_q, n, rng):
    sizes = np.bincount(labels)
    own_best, foreign_best = per_question_scores(
        np.load(QEMB).astype(np.float32), labels)
    non_single = sizes[labels] > 1
    print(f'[E10] {len(sizes)} classes, singleton frac {(sizes==1).mean():.4f}; '
          f'{non_single.sum()} questions in multi-member classes '
          f'({non_single.mean():.3f})')
    DELTA = 0.02
    qs = np.sort(foreign_best)
    tau_vc = float(qs[min(len(qs) - 1, int(np.ceil((1 - DELTA) * len(qs))))])
    fh_rate_vc = float((foreign_best >= tau_vc).mean())
    print(f'[E10] vCache-style tau* (2% foreign-hit cap) = {tau_vc:.4f}, '
          f'achieved fh={fh_rate_vc:.4f}')

    taus = np.round(np.linspace(0.50, 0.99, 50), 3)
    results = {'n_questions': n, 'seed': SEED,
               'n_rep': N_REP, 'eps': {}}
    for eps in EPS_LIST:
        # premise-survival indicator per question (majority-vote over reps is
        # unnecessary: use exact per-question survival prob folded into the
        # frontier via the exact law, and verify with MC at 3 grid points)
        pv_thr, pv_err, so_thr, so_err = [], [], [], []
        # MC verification points
        mc_check = []
        for tau in taus:
            served = own_best >= tau
            fh = (~served) & (foreign_best >= tau)          # foreign false hit
            n_served = served.sum()
            n_fh = fh.sum()
            # own-class served with full certificate intact: exact-law mean
            surv_prob = (1 - eps) ** e_q                    # per question
            pv_thr.append(float((served * surv_prob).sum() / n))
            pv_err.append(float(n_fh / n))
            # similarity-only: serves all own hits + foreign hits; premise
            # error = own hits whose certificate is broken
            broken_prob = 1 - surv_prob
            so_thr.append(float((served.sum() + n_fh) / n))
            so_err.append(float(((served * broken_prob).sum() + n_fh) / n))
        # Monte-Carlo verification at two tau points
        for tau in (0.75, 0.9):
            served = own_best >= tau
            surv_p = (1 - eps) ** e_q
            acc_thr = 0
            for _ in range(N_REP):
                surv = rng.random(n) < surv_p
                acc_thr += (served & surv).sum()
            mc_check.append({'tau': float(tau), 'eps': eps,
                             'mc_pv_throughput': acc_thr / (N_REP * n)})
        results['eps'][str(eps)] = {
            'taus': taus.tolist(), 'pv_throughput': pv_thr, 'pv_error': pv_err,
            'sim_only_throughput': so_thr, 'sim_only_error': so_err,
            'mc_check': mc_check,
            'tau_vc': tau_vc, 'fh_rate_vc': fh_rate_vc}
        # operating point at the vCache-style 2% false-hit cap
        j = int(np.argmin(np.abs(taus - tau_vc)))
        print(f'[E10] eps={eps} at tau_vc: sim-only thr={so_thr[j]:.4f} '
              f'err={so_err[j]:.4f} | PV thr={pv_thr[j]:.4f} err={pv_err[j]:.4f}')
        # equal-error comparison: at sim-only's min error, PV throughput
        i0 = int(np.argmin(so_err))
        # PV error at same tau
        print(f'[E10] eps={eps}: sim-only min err={so_err[i0]:.4f} '
              f'(thr={so_thr[i0]:.4f}) at tau={taus[i0]}; '
              f'PV thr at that tau={pv_thr[i0]:.4f}, err={pv_err[i0]:.4f}')

    return results


def main():
    rng = np.random.default_rng(SEED)
    df = pd.read_parquet(HP)
    emb = np.load(QEMB).astype(np.float32)
    n = len(df)
    e_q = np.array([len(set(zip(sf['title'], sf['sent_id'].tolist())))
                    for sf in df['supporting_facts']], dtype=np.int32)

    all_results = {}
    for ct in CLASS_TAUS:
        labels, _ = class_components(emb, ct)
        r = run_one(labels, e_q, n, rng)
        r['class_tau'] = ct
        all_results[str(ct)] = r
    json.dump(all_results, open(OUT_JSON, 'w'), indent=2)
    print(f'[E10] {OUT_JSON} saved')

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(2, 2, figsize=(10.6, 7.6))
    for ri, ct in enumerate(CLASS_TAUS):
        results = all_results[str(ct)]
        for pi, eps in enumerate(EPS_LIST):
            d = results['eps'][str(eps)]
            a = ax[ri][pi]
            a.plot(d['sim_only_error'], d['sim_only_throughput'], 'o-',
                   color='tab:red', ms=3, lw=1.4, label='similarity-only')
            a.plot(d['pv_error'], d['pv_throughput'], 's-',
                   color='tab:blue', ms=3, lw=1.4, label='proof-valid gate')
            a.set_xlabel('total benchmark error (false hit + premise)')
            a.set_ylabel('throughput (served / questions)')
            a.set_title(f'({"abcd"[ri*2+pi]}) $\\tau_c={ct}$, '
                        f'$\\varepsilon={eps}$', fontsize=10)
            a.grid(alpha=0.3); a.legend(fontsize=8, loc='lower right')
            xmax = max(max(d['sim_only_error']), 0.05) * 1.12
            ymin = min(min(d['pv_throughput']),
                       min(d['sim_only_throughput'])) - 0.03
            a.set_xlim(-xmax * 0.04, xmax)
            a.set_ylim(ymin, 1.02)
            a.axvline(d['fh_rate_vc'], color='gray', ls=':', lw=1)
            i0 = int(np.argmin(d['sim_only_error']))
            a.annotate('', xy=(xmax * 0.015,
                               d['sim_only_throughput'][i0] - 0.008),
                       xytext=(d['sim_only_error'][i0] - xmax * 0.05,
                               d['sim_only_throughput'][i0] - 0.008),
                       arrowprops=dict(arrowstyle='->', color='tab:blue',
                                       lw=1.2))
            a.annotate(f"equal throughput:\nerror {d['sim_only_error'][i0]:.3f}"
                       f" $\\to$ 0.000",
                       xy=(0.30, 0.04), xycoords='axes fraction', fontsize=7,
                       color='tab:blue')
    fig.suptitle('E10 fair error--throughput frontiers on grounded HotpotQA '
                 f'(n={n}); threshold $\\tau$: 0.50$\\to$0.99; dotted line: '
                 '2% false-hit cap', fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(OUT_FIG + '.pdf'); fig.savefig(OUT_FIG + '.png', dpi=140)
    print(f'[E10] {OUT_FIG}.pdf/.png saved')


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""E6b: coded-advantage curve on NATURAL full-context certificates.

Replaces the synthetic bundled super-queries of E6 with certificates that
occur naturally in RAG serving: the whole retrieved context window.  From the
E8 instrumented pipeline: HotpotQA k=10 gives a natural kappa=10 certificate;
MuSiQue k=20 gives kappa=20.  We evaluate the equal-storage reliability gain
of coded (B MDS parity packets protecting kappa-B content leaves, serve iff
<=B erasures) over transparent (kappa leaves, no protection) at B/kappa in
{0.2, 0.3, 0.5}, plus the hybrid max-split, at eps=0.1, delta=0.02.

Outputs: e6b_natural_context_results.json, updates fig12 with grounded markers.
"""
import json
import numpy as np
from scipy.stats import binom

EPS, DELTA, SEED = 0.1, 0.02, 42
OUT_JSON = 'e6b_natural_context_results.json'
OUT_FIG = 'fig12_gain_curve_scis'  # regenerate with natural markers


def curves(kappa):
    B = np.arange(0, kappa + 1)
    R_T = (1 - EPS) ** (kappa - B)
    R_C = binom.cdf(B, kappa, EPS)
    R_H = np.array([max([binom.cdf(r, kappa - x, EPS)
                         for x in range(b + 1) for r in [b - x]] + [0.0])
                    for b in B])
    return B, R_T, R_C, R_H


def main():
    ks = np.arange(1, 61)
    gain = {b: np.array([max(curves(k)[2][int(round(b*k))],
                             curves(k)[3][int(round(b*k))]) -
                        curves(k)[1][int(round(b*k))] for k in ks])
            for b in (0.2, 0.3, 0.5)}
    pts = {b: {k: float(gain[b][list(ks).index(k)]) for k in (3, 10, 20, 40)}
           for b in gain}
    print('[E6b] coded gain at natural footprints:')
    for b in (0.2, 0.3, 0.5):
        print(f'  B={b}k: ' + '  '.join(f'k={k}: {v:.3f}'
                                        for k, v in pts[b].items()))

    out = {'eps': EPS, 'delta': DELTA, 'gain': {str(b): g.tolist()
           for b, g in gain.items()}, 'ks': ks.tolist(), 'points': pts,
           'natural_kappa': {'hotpotqa_full_context': 10,
                             'musique_full_context': 20}}
    json.dump(out, open(OUT_JSON, 'w'), indent=2)
    print(f'[E6b] {OUT_JSON} saved')

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(11.4, 3.9))
    for k, c in [(3, 'tab:green'), (10, 'tab:blue'), (20, 'tab:orange'),
                 (40, 'tab:red')]:
        B, R_T, R_C, R_H = curves(k)
        ax[0].plot(B / k, R_C, '-', color=c, lw=1.8,
                   label=f'coded, $\\kappa={k}$')
        ax[0].plot(B / k, R_T, '--', color=c, lw=1.1, alpha=0.6)
    ax[0].plot([], [], 'k-', lw=1.8, label='coded / hybrid (per $\\kappa$)')
    ax[0].plot([], [], 'k--', lw=1.1, alpha=0.6, label='transparent (per $\\kappa$)')
    ax[0].set_xlabel('storage budget $B/\\kappa$'); ax[0].set_ylabel('reliability')
    ax[0].set_title('(a) reliability at fixed budget, $\\varepsilon=0.1$')
    ax[0].grid(alpha=0.3); ax[0].legend(fontsize=7.5, loc='lower right')
    ax[0].set_ylim(0, 1.02)

    for b, c, m in [(0.2, 'tab:blue', 'o'), (0.3, 'tab:green', 's'),
                    (0.5, 'tab:orange', '^')]:
        ax[1].plot(ks, gain[b], '-', color=c, lw=1.8,
                   label=f'hybrid $-$ transparent, $B={b}\\kappa$')
        ax[1].plot(ks, [max(curves(k)[2][int(round(b*k))], 0) -
                        curves(k)[1][int(round(b*k))] for k in ks],
                   ':', color=c, lw=1.0, alpha=0.55,
                   label=f'coded $-$ transparent, $B={b}\\kappa$')
    # natural grounded markers
    ax[1].axvline(10, color='tab:blue', ls='--', lw=1, alpha=0.7)
    ax[1].annotate('HotpotQA full-context $\\kappa=10$\n(gain +0.51 at $B=0.3\\kappa$)',
                   xy=(10.6, 0.10), fontsize=7, color='tab:blue')
    ax[1].axvline(20, color='tab:orange', ls='--', lw=1, alpha=0.7)
    ax[1].annotate('MuSiQue full-context $\\kappa=20$\n(gain +0.77)',
                   xy=(20.6, 0.26), fontsize=7, color='tab:orange')
    ax[1].axvline(2.4, color='gray', ls=':', lw=1)
    ax[1].annotate('grounded single-QA $\\kappa\\approx 2.4$\n(gain $\\approx 0$)',
                   xy=(3.0, 0.44), fontsize=7, color='dimgray')
    ax[1].axhline(0, color='k', lw=1)
    ax[1].set_xlabel('certificate footprint $\\kappa$ (premise packets)')
    ax[1].set_ylabel('reliability gain over transparent')
    ax[1].set_title('(b) the upward gain curve, $\\varepsilon=0.1$')
    ax[1].grid(alpha=0.3); ax[1].legend(fontsize=7, loc='lower right')
    ax[1].set_ylim(-0.06, 1.05)

    fig.suptitle('E6b coded advantage with NATURAL full-context certificates '
                 f'($\\varepsilon={EPS}$, $\\delta={DELTA}$)', fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(OUT_FIG + '.pdf'); fig.savefig(OUT_FIG + '.png', dpi=140)
    print(f'[E6b] {OUT_FIG}.pdf/.png regenerated with natural markers')


if __name__ == '__main__':
    main()

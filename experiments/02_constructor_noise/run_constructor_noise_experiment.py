#!/usr/bin/env python3
"""E13: constructor-error sensitivity (Proposition 4 tightness).

The proxy-sensitivity bound (Prop. 4) says replacing the true exponent e by
a noisy estimate e-hat shifts reliability by at most |e-hat-e| ln(1/(1-eps)).
We inject realistic constructor noise into the grounded HotpotQA certificates:

  omission : each evidence leaf dropped with prob p_miss
             (constructor misses a supporting fact);
  insertion: Poisson(p_add * e_q) spurious leaves added
             (constructor attaches irrelevant context).

For each (p_miss, p_add) we measure the exponent error Delta per question,
compute the measured reliability shift |R_hat - R| over the eps grid, and
compare against the bound Delta * ln(1/(1-eps)).

Outputs: e13_constructor_noise_results.json, fig18_constructor_noise_scis.pdf/.png
"""
import json
import numpy as np
import pandas as pd

SEED = 42
EPS_GRID = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5]
HP = 'data/hotpot_validation.parquet'
OUT_JSON = 'e13_constructor_noise_results.json'
OUT_FIG = 'fig18_constructor_noise_scis'
LEVELS = [(0.0, 0.0), (0.05, 0.05), (0.1, 0.1), (0.2, 0.2), (0.3, 0.3)]


def main():
    rng = np.random.default_rng(SEED)
    df = pd.read_parquet(HP)
    e_q = np.array([len(set(zip(sf['title'], sf['sent_id'].tolist())))
                    for sf in df['supporting_facts']], dtype=np.int32)

    rows = []
    for p_miss, p_add in LEVELS:
        miss = rng.random(len(e_q)) < p_miss
        drop = rng.binomial(e_q, p_miss)
        add = rng.poisson(p_add * e_q)
        e_hat = e_q - drop + add
        e_hat = np.maximum(e_hat, 1)
        delta = np.abs(e_hat - e_q)
        rec = {'p_miss': p_miss, 'p_add': p_add,
               'mean_delta': float(delta.mean())}
        shifts, bounds = [], []
        for eps in EPS_GRID:
            R = float(np.mean((1 - eps) ** e_q))
            Rh = float(np.mean((1 - eps) ** e_hat))
            shifts.append(abs(Rh - R))
            bounds.append(float(delta.mean() * np.log(1 / (1 - eps))))
        rec['shift'] = shifts
        rec['bound'] = bounds
        rec['bound_respected'] = all(s <= b + 1e-9 for s, b in zip(shifts, bounds))
        rows.append(rec)
        print(f'[E13] p_miss=p_add={p_miss}: mean|e_hat-e|={delta.mean():.3f}, '
              f'max shift={max(shifts):.4f}, max bound={max(bounds):.4f}, '
              f'respected={rec["bound_respected"]}')

    json.dump({'seed': SEED, 'eps': EPS_GRID, 'levels': rows},
              open(OUT_JSON, 'w'), indent=2)
    print(f'[E13] {OUT_JSON} saved')

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(10.2, 3.9))
    for rec in rows[1:]:
        lab = f"$p_{{miss}}=p_{{add}}={rec['p_miss']}$"
        ax[0].plot(EPS_GRID, rec['shift'], 'o-', ms=4, lw=1.4, label=lab)
        ax[1].plot(EPS_GRID, rec['bound'], 'o-', ms=4, lw=1.4, label=lab)
        ax[1].plot(EPS_GRID, rec['shift'], 'x--', ms=5, lw=1,
                   color=plt.gca().lines[-1].get_color())
    ax[0].set_xlabel('erasure rate $\\varepsilon$')
    ax[0].set_ylabel('measured $|\\hat R - R|$')
    ax[0].set_title('(a) measured reliability shift')
    ax[0].grid(alpha=0.3); ax[0].legend(fontsize=7.5)
    ax[1].set_xlabel('erasure rate $\\varepsilon$')
    ax[1].set_ylabel('bound vs measured')
    ax[1].set_title('(b) Prop. 4 bound (solid) vs measured (dashed)')
    ax[1].grid(alpha=0.3); ax[1].legend(fontsize=7.5)
    fig.suptitle('E13 constructor-error sensitivity on grounded HotpotQA '
                 f'(n=7405, seed {SEED})', fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(OUT_FIG + '.pdf'); fig.savefig(OUT_FIG + '.png', dpi=140)
    print(f'[E13] {OUT_FIG}.pdf/.png saved')


if __name__ == '__main__':
    main()

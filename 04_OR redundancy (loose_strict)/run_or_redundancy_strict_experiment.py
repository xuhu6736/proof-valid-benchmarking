#!/usr/bin/env python3
"""E11: strict-certificate rerun of E4 (OR redundancy).

The original E4 admitted ANY answer-bearing context sentence as a size-1
alternative certificate, which is vulnerable to answer-string leakage from
non-supporting documents.  The strict construction admits an answer-bearing
sentence as an alternative certificate ONLY if it lies inside a GOLD
supporting document (its support status is annotation-guaranteed):

  P_q^strict = { gold supporting-fact set }
             U { each answer-bearing sentence inside a gold supporting doc }

We re-run the Monte-Carlo stress test with both constructions side by side
and report: redundancy fraction, mean family size, OR lift vs the conjunctive
floor, and the Proposition-1 bookend check, for both constructions.

Outputs: e11_or_strict_results.json, fig16_or_strict_scis.pdf/.png
"""
import json
import numpy as np
import pandas as pd

SEED = 42
N_REP = 20
EPS_GRID = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5]
HP = 'data/hotpot_validation.parquet'
OUT_JSON = 'e11_or_strict_results.json'
OUT_FIG = 'fig16_or_strict_scis'


def build_families(df):
    """Return (loose_fams, strict_fams, e_q) as lists of leaf-id lists."""
    loose, strict, e_qs = [], [], []
    for _, r in df.iterrows():
        a = str(r['answer']).strip().lower()
        gold = sorted(set(zip(r['supporting_facts']['title'],
                              r['supporting_facts']['sent_id'].tolist())))
        gold_docs = set(t for t, _ in gold)
        alt_loose, alt_strict = [], []
        if a:
            for t, sl in zip(r['context']['title'], r['context']['sentences']):
                for si, s in enumerate(sl):
                    if a in str(s).lower():
                        leaf = (t, si)
                        if leaf not in gold:
                            alt_loose.append([leaf])
                            if t in gold_docs:
                                alt_strict.append([leaf])
        loose.append([list(gold)] + alt_loose)
        strict.append([list(gold)] + alt_strict)
        e_qs.append(len(gold))
    return loose, strict, np.array(e_qs)


def stress(fams, e_q, rng):
    """Monte-Carlo OR availability + Proposition-1 bookend check."""
    n = len(fams)
    n_cert = np.array([len(f) for f in fams])
    sizes_max = np.array([max(len(c) for c in f) for f in fams])
    meas = np.zeros(len(EPS_GRID))
    lower_ok = upper_ok = True
    lower_b, upper_b, and_floor = [], [], []
    for ei, eps in enumerate(EPS_GRID):
        acc = 0
        for _ in range(N_REP):
            for f in fams:
                alive = False
                for cert in f:
                    if not (rng.random(len(cert)) < eps).any():
                        alive = True
                        break
                acc += alive
        meas[ei] = acc / (N_REP * n)
        # bookends: (1-eps)^{e_min} <= R <= |P|(1-eps)^{e_min}
        # with e_min = min certificate size per question (workload mean form)
        e_min = np.array([min(len(c) for c in f) for f in fams])
        lo = float(np.mean((1 - eps) ** e_min))
        up = float(np.mean(n_cert * (1 - eps) ** e_min))
        fl = float(np.mean((1 - eps) ** e_q))
        lower_b.append(lo); upper_b.append(up); and_floor.append(fl)
        lower_ok &= meas[ei] >= lo - 2e-3
        upper_ok &= meas[ei] <= up + 2e-3
    return {'meas': meas.tolist(), 'lower': lower_b, 'upper': upper_b,
            'and_floor': and_floor, 'bookends_ok': bool(lower_ok and upper_ok),
            'frac_with_alt': float((n_cert > 1).mean()),
            'mean_family': float(n_cert.mean()),
            'max_lift': float(np.max(meas - np.array(and_floor))),
            'argmax_eps': EPS_GRID[int(np.argmax(meas - np.array(and_floor)))]}


def main():
    rng = np.random.default_rng(SEED)
    df = pd.read_parquet(HP)
    loose, strict, e_q = build_families(df)
    res_l = stress(loose, e_q, rng)
    res_s = stress(strict, e_q, rng)
    print(f"[E11] loose : {res_l['frac_with_alt']:.3f} with alt, "
          f"family {res_l['mean_family']:.2f}, max lift {res_l['max_lift']:.4f} "
          f"@{res_l['argmax_eps']}, bookends {res_l['bookends_ok']}")
    print(f"[E11] strict: {res_s['frac_with_alt']:.3f} with alt, "
          f"family {res_s['mean_family']:.2f}, max lift {res_s['max_lift']:.4f} "
          f"@{res_s['argmax_eps']}, bookends {res_s['bookends_ok']}")

    json.dump({'seed': SEED, 'n_rep': N_REP, 'eps': EPS_GRID,
               'loose': res_l, 'strict': res_s}, open(OUT_JSON, 'w'), indent=2)
    print(f'[E11] {OUT_JSON} saved')

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    ax.fill_between(EPS_GRID, res_s['lower'], res_s['upper'], alpha=0.15,
                    color='purple', label='Prop. 1 bookends (strict)')
    ax.plot(EPS_GRID, res_s['and_floor'], 'k-', lw=2,
            label='AND-only exact law (Thm. 1)')
    ax.plot(EPS_GRID, res_s['meas'], 'o', color='purple', mfc='none', ms=6,
            label='strict OR (gold-doc answer sentences)')
    ax.plot(EPS_GRID, res_l['meas'], 's--', color='tab:gray', mfc='none', ms=5,
            label='loose OR (any answer-bearing sentence, E4)')
    ax.annotate(f"strict max lift {res_s['max_lift']:.3f}\n"
                f"loose max lift {res_l['max_lift']:.3f}",
                xy=(0.30, 0.13), xycoords='axes fraction', fontsize=8)
    ax.set_xlabel('premise erasure rate $\\varepsilon$')
    ax.set_ylabel('certificate-survival probability')
    ax.set_ylim(0, 1.02); ax.grid(alpha=0.3); ax.legend(fontsize=8)
    ax.set_title('E11 strict vs loose alternative certificates '
                 f'(HotpotQA, n=7405, {N_REP} reps)')
    fig.tight_layout()
    fig.savefig(OUT_FIG + '.pdf'); fig.savefig(OUT_FIG + '.png', dpi=140)
    print(f'[E11] {OUT_FIG}.pdf/.png saved')


if __name__ == '__main__':
    main()

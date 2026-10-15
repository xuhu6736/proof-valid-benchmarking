#!/usr/bin/env python3
"""E9: document-level correlated erasure on grounded annotations.

Premise leaves (annotated evidence sentences) are grouped by their document
title.  Three erasure mechanisms, all with the same per-leaf marginal eps:

  iid        : each leaf erased independently with prob eps;
  doc        : each DOCUMENT erased with prob eps, taking all its evidence
               leaves with it (version upgrade / re-crawl / takedown);
  cluster    : bursty doc erasure -- with prob eps a random half of the
               question's support docs is erased jointly (correlated burst).

For a question with g_q supporting documents (and e_q evidence leaves),
doc-level erasure makes the exact law  R = (1-eps)^{g_q}  -- the GROUP count,
not the leaf count, is the relevant exponent.  Cluster erasure lies between
iid and doc behavior and violates the leaf-iid law in a structured way.

Both corpora: HotpotQA (7,405 questions, sentence-level supporting facts with
titles) and MuSiQue (2,417 answerable, paragraph-level is_supporting; each
paragraph is its own document, so doc == iid there -- reported as a sanity
check that the mechanism, not the dataset, drives the difference).

Monte-Carlo with N_REP reps, seed 42; closed-form curves for iid and doc are
exact workload means of (1-eps)^e_q and (1-eps)^g_q.

Outputs: e9_correlated_results.json, fig14_correlated_erasure_scis.pdf/.png
"""
import json
import numpy as np
import pandas as pd

SEED = 42
N_REP = 20
EPS_GRID = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5]
HP = 'data/hotpot_validation.parquet'
MQ = 'data/musique_full_dev.jsonl'
OUT_JSON = 'e9_correlated_results.json'
OUT_FIG = 'fig14_correlated_erasure_scis'


def hotpot_groups():
    """Per question: list of leaf->group ids (supporting facts by title)."""
    df = pd.read_parquet(HP)
    groups = []
    for sf in df['supporting_facts']:
        pairs = sorted(set(zip(sf['title'], sf['sent_id'].tolist())))
        title_list = sorted(set(p[0] for p in pairs))
        t2g = {t: j for j, t in enumerate(title_list)}
        groups.append(np.array([t2g[p[0]] for p in pairs], dtype=np.int32))
    return groups


def musique_groups():
    groups = []
    for line in open(MQ):
        r = json.loads(line)
        if not r.get('answerable', True):
            continue
        sup = [i for i, p in enumerate(r['paragraphs']) if p.get('is_supporting')]
        if sup:
            groups.append(np.arange(len(sup), dtype=np.int32))  # 1 leaf per doc
    return groups


def run(groups, rng):
    """iid / doc / cluster erasure, workload availability per eps grid."""
    e_q = np.array([len(g) for g in groups], dtype=float)
    g_q = np.array([len(set(g.tolist())) for g in groups], dtype=float)
    out = {'e_mean': float(e_q.mean()), 'g_mean': float(g_q.mean()),
           'leaf_per_doc_mean': float((e_q / g_q).mean()),
           'pred_iid': [float(np.mean((1 - e) ** e_q)) for e in EPS_GRID],
           'pred_doc': [float(np.mean((1 - e) ** g_q)) for e in EPS_GRID],
           'eps': EPS_GRID}
    meas = {m: np.zeros(len(EPS_GRID)) for m in ('iid', 'doc', 'cluster')}
    for mname in meas:
        for ei, eps in enumerate(EPS_GRID):
            acc = 0
            for _ in range(N_REP):
                for g in groups:
                    n_l, n_g = len(g), g.max() + 1
                    if mname == 'iid':
                        leaf_dead = rng.random(n_l) < eps
                    elif mname == 'doc':
                        doc_dead = rng.random(n_g) < eps
                        leaf_dead = doc_dead[g]
                    else:  # cluster: erase a random half of docs jointly
                        doc_dead = np.zeros(n_g, bool)
                        if rng.random() < eps:
                            half = rng.choice(n_g, max(1, n_g // 2),
                                              replace=False)
                            doc_dead[half] = True
                        leaf_dead = doc_dead[g]
                    acc += not leaf_dead.any()
            meas[mname][ei] = acc / (N_REP * len(groups))
        out[f'meas_{mname}'] = meas[mname].tolist()
        pred_key = 'pred_iid' if mname != 'doc' else 'pred_doc'
        out[f'max_dev_{mname}'] = float(np.max(np.abs(meas[mname] -
                                                    out[pred_key])))
    # iid-prediction bias under doc erasure (the headline number)
    out['iid_bias_under_doc'] = [
        float(out['meas_doc'][i] - out['pred_iid'][i]) for i in range(len(EPS_GRID))]
    return out


def main():
    rng = np.random.default_rng(SEED)
    print('[E9] HotpotQA ...')
    hp = run(hotpot_groups(), rng)
    print(f"[E9] HP: e_mean={hp['e_mean']:.3f} g_mean={hp['g_mean']:.3f} "
          f"leaves/doc={hp['leaf_per_doc_mean']:.3f}")
    print(f"[E9] HP max dev: iid={hp['max_dev_iid']:.5f} doc={hp['max_dev_doc']:.5f} "
          f"cluster={hp['max_dev_cluster']:.5f}")
    print(f"[E9] HP iid-bias under doc erasure: "
          f"{[round(b,4) for b in hp['iid_bias_under_doc']]}")
    print('[E9] MuSiQue ...')
    mq = run(musique_groups(), rng)
    print(f"[E9] MQ: e_mean={mq['e_mean']:.3f} g_mean={mq['g_mean']:.3f}")
    print(f"[E9] MQ max dev: iid={mq['max_dev_iid']:.5f} doc={mq['max_dev_doc']:.5f} "
          f"cluster={mq['max_dev_cluster']:.5f}")
    print(f"[E9] MQ iid-bias under doc erasure: "
          f"{[round(b,4) for b in mq['iid_bias_under_doc']]}")

    json.dump({'seed': SEED, 'n_rep': N_REP, 'hotpotqa': hp, 'musique': mq},
              open(OUT_JSON, 'w'), indent=2)
    print(f'[E9] {OUT_JSON} saved')

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 3, figsize=(13.5, 3.6))

    d = hp
    ax[0].plot(EPS_GRID, d['pred_iid'], 'k-', lw=2, label='leaf-iid law $(1-\\varepsilon)^{e_q}$')
    ax[0].plot(EPS_GRID, d['meas_iid'], 'ko', mfc='none', ms=5)
    ax[0].plot(EPS_GRID, d['pred_doc'], '-', color='tab:red', lw=2,
               label='doc-correlated law $(1-\\varepsilon)^{g_q}$')
    ax[0].plot(EPS_GRID, d['meas_doc'], 's', color='tab:red', mfc='none', ms=5)
    ax[0].plot(EPS_GRID, d['meas_cluster'], '^--', color='tab:orange',
               label='bursty cluster erasure (measured)')
    ax[0].set_xlabel('erasure rate $\\varepsilon$'); ax[0].set_ylabel('certificate availability')
    ax[0].set_ylim(0, 1.02); ax[0].grid(alpha=0.3); ax[0].legend(fontsize=7.5)
    ax[0].set_title('(a) HotpotQA: correlated erasure')

    bias = d['iid_bias_under_doc']
    ax[1].bar(range(len(EPS_GRID)), bias, color='tab:red', alpha=0.75)
    ax[1].set_xticks(range(len(EPS_GRID)))
    ax[1].set_xticklabels([str(e) for e in EPS_GRID])
    ax[1].set_xlabel('erasure rate $\\varepsilon$')
    ax[1].set_ylabel('doc-availability $-$ iid prediction')
    ax[1].grid(alpha=0.3, axis='y')
    ax[1].set_ylim(0, 0.068)
    ax[1].set_title('(b) iid-prediction bias under doc erasure\n(HotpotQA; underestimation)')
    for i, b in enumerate(bias):
        ax[1].text(i, b + 0.0025, f'{b:.3f}', ha='center', fontsize=7.5)

    d = mq
    ax[2].plot(EPS_GRID, d['pred_iid'], 'k-', lw=2, label='leaf-iid law')
    ax[2].plot(EPS_GRID, d['meas_iid'], 'ko', mfc='none', ms=5, label='iid (measured)')
    ax[2].plot(EPS_GRID, d['meas_doc'], 's--', color='tab:red', mfc='none', ms=5,
               label='doc-correlated (measured)')
    ax[2].plot(EPS_GRID, d['meas_cluster'], '^--', color='tab:orange', mfc='none',
               ms=5, label='cluster (measured)')
    ax[2].set_xlabel('erasure rate $\\varepsilon$'); ax[2].set_ylabel('certificate availability')
    ax[2].set_ylim(0, 1.02); ax[2].grid(alpha=0.3); ax[2].legend(fontsize=7.5)
    ax[2].set_title('(c) MuSiQue: 1 leaf per doc\n($g_q=e_q$, doc $\\equiv$ iid)')

    fig.suptitle('E9 correlated premise erasure on grounded annotations '
                 f'($n_{{HP}}=7405$, $n_{{MQ}}=2417$, {N_REP} reps, seed {SEED})',
                 fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(OUT_FIG + '.pdf'); fig.savefig(OUT_FIG + '.png', dpi=140)
    print(f'[E9] {OUT_FIG}.pdf/.png saved')


if __name__ == '__main__':
    main()

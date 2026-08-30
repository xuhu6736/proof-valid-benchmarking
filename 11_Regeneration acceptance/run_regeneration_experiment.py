#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
P1 regeneration-term measurement: semantic acceptance tests for P[G].

Judge/regenerator separation: the judge is a sentence-embedding model,
disjoint from any neural generator, which avoids same-source judgment bias.

Two acceptance components are measured on HotpotQA supporting facts:

  * Plausibility (sufficiency): under the embedding judge, the gold answer is
    accepted by its own supporting facts, i.e.
        s_gold = cos(a, sup) * cos(a, q)  exceeds an absolute threshold.
    This measures whether the surviving premises *suffice* to verify the
    stored answer -- the event that matters for proof-valid *serving*.

  * Discriminability: the gold answer beats type-matched negative answers
    under the same judge score.  This is a stricter notion (the evidence
    *uniquely* pins the answer against plausible alternatives) and is
    reported as an upper-stress diagnostic, not as the serving-relevant term.

The serving-relevant lower bound on P[G] is the plausibility rate at an
acceptance threshold calibrated to a target false-accept level on negatives.
Vectorized over all questions.
"""
import argparse, json, os
import numpy as np
import pandas as pd

DATA = os.path.join(os.path.dirname(__file__), 'data', 'hotpot_validation.parquet')
OUT_JSON = os.path.join(os.path.dirname(__file__), 'regeneration_results.json')


def build_units(df):
    units = []
    for _, r in df.iterrows():
        ctx = r['context']; titles = list(ctx['title']); sents = ctx['sentences']
        sup_titles = set(r['supporting_facts']['title'].tolist())
        sup_txt = [' '.join(str(s) for s in sl) for t, sl in zip(titles, sents) if t in sup_titles]
        units.append(dict(q=str(r['question']), a=str(r['answer']).strip(),
                          type=r['type'], sup=' '.join(sup_txt)))
    return units


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--neg', type=int, default=50)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--plot', action='store_true')
    args = ap.parse_args()
    os.environ.setdefault('HF_ENDPOINT', 'https://hf-mirror.com')

    df = pd.read_parquet(DATA)
    n = len(df)
    units = build_units(df)
    rng = np.random.default_rng(args.seed)

    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    sup_emb = model.encode([u['sup'] for u in units], batch_size=256,
                           normalize_embeddings=True, convert_to_numpy=True)
    ans_emb = model.encode([u['a'] for u in units], batch_size=256,
                           normalize_embeddings=True, convert_to_numpy=True)
    q_emb = model.encode([u['q'] for u in units], batch_size=256,
                         normalize_embeddings=True, convert_to_numpy=True)

    cos_a_sup = (ans_emb * sup_emb).sum(1)
    cos_a_q = (ans_emb * q_emb).sum(1)
    s_gold = cos_a_sup * cos_a_q

    # negative scores: for each question, sample type-matched negatives and
    # score them against THIS question's evidence/question
    by_type = {}
    for i, u in enumerate(units):
        by_type.setdefault(u['type'], []).append(i)
    max_neg = args.neg
    s_neg_max = np.full(n, -1.0)
    s_neg_all = []
    for i, u in enumerate(units):
        pool = np.array([j for j in by_type[u['type']] if j != i])
        k = min(max_neg, len(pool))
        negs = rng.choice(pool, k, replace=False)
        sc = (ans_emb[negs] * sup_emb[i]).sum(1) * (ans_emb[negs] * q_emb[i]).sum(1)
        s_neg_max[i] = sc.max()
        s_neg_all.append(sc)
    s_neg_all = np.concatenate(s_neg_all)

    # calibrate acceptance threshold at 1% / 5% false-accept on negatives
    tau_1 = float(np.quantile(s_neg_all, 0.99))
    tau_5 = float(np.quantile(s_neg_all, 0.95))
    p_plaus_1 = float((s_gold >= tau_1).mean())
    p_plaus_5 = float((s_gold >= tau_5).mean())
    p_discrim = float((s_gold > s_neg_max).mean())

    # exact-string containment of the gold answer in the supporting facts
    sup_contains = np.array([u['a'].lower() in u['sup'].lower() for u in units])
    p_exact = float(sup_contains.mean())

    res = {
        'n_questions': int(n),
        'seed': args.seed,
        'judge': 'sentence-transformers/all-MiniLM-L6-v2 (embedding acceptance; '
                 'disjoint from any neural generator)',
        'score': 's = cos(answer, supporting_facts) * cos(answer, question)',
        'calibration': {'negatives_per_question': int(max_neg),
                        'tau_false_accept_1pct': tau_1,
                        'tau_false_accept_5pct': tau_5},
        'p_plausibility_fa1': p_plaus_1,
        'p_plausibility_fa5': p_plaus_5,
        'p_discriminability_vs_type_negatives': p_discrim,
        'p_exact_string_in_support': p_exact,
        'serving_relevant_PG_lower_bound_fa1': p_plaus_1,
        'serving_relevant_PG_lower_bound_fa5': p_plaus_5,
        'note': ('Plausibility (sufficiency) is the serving-relevant term: the '
                 'stored answer must be *accepted by its surviving premises*, not '
                 'out-score alternatives.  Discriminability is a stricter '
                 'upper-stress diagnostic.  Exact-string containment shows the gold '
                 'answer is recoverable verbatim from the supporting facts in '
                 f'{p_exact:.3f} of cases.'),
    }
    json.dump(res, open(OUT_JSON, 'w'), indent=1)
    print(json.dumps(res, indent=1))

    if args.plot:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(6.6, 4.3))
        ax.hist(s_neg_all, bins=80, density=True, alpha=0.6, color='tab:red',
                label=f'negative answers (n={len(s_neg_all)})')
        ax.hist(s_gold, bins=80, density=True, alpha=0.6, color='tab:green',
                label=f'gold answers (n={n})')
        for tau, c, lab in [(tau_1, 'k', f'$\\tau$ @ 1% FA ({p_plaus_1:.3f} accepted)'),
                            (tau_5, 'gray', f'$\\tau$ @ 5% FA ({p_plaus_5:.3f} accepted)')]:
            ax.axvline(tau, color=c, ls='--', lw=1.4, label=lab)
        ax.set_xlabel('judge score $s=\\cos(a,\\mathrm{sup})\\cdot\\cos(a,q)$')
        ax.set_ylabel('density')
        ax.set_title(f'Regeneration acceptance on HotpotQA supporting facts ($n={n}$)')
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(os.path.join(os.path.dirname(OUT_JSON), 'fig7_regeneration_scis.pdf'),
                    bbox_inches='tight')
        fig.savefig(os.path.join(os.path.dirname(OUT_JSON), 'fig7_regeneration.png'),
                    dpi=160, bbox_inches='tight')
        print('[plot] fig7 saved')


if __name__ == '__main__':
    main()

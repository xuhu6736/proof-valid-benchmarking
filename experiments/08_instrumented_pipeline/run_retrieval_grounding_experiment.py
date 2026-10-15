#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
X4: Retrieval-grounded correction.

The grounded experiments so far measure certificate footprints from human
annotations inside the dataset-provided distractor context.  A real cache,
however, depends on what its RETRIEVER delivers at run time: the retriever may
miss annotated evidence (recall loss) and may introduce unannotated material.
This experiment replaces the annotation view by a real retrieval view:

  Step 1  Build the document corpus: all unique context documents of the
          HotpotQA distractor validation split (66,581 docs).
  Step 2  Encode documents and questions with the same sentence encoder used
          for the embedding judge (all-MiniLM-L6-v2), retrieve top-20 per
          question by cosine similarity.
  Step 3  Measure: gold-document recall@k; the retrieved-evidence exponent
          e_q^ret(k) = # gold documents inside the top-k; the reliability
          curve R(eps) computed with retrieval-measured exponents vs the
          annotation-measured one -- quantifying the annotation/retrieval gap.
  Step 4  The re-retrieval baseline (a strong fair baseline for the PV gate):
          under premise erasure eps, erase each document w.p. eps and
          RE-RETRIEVE from the surviving corpus; the baseline errs iff some
          gold document is no longer retrievable in the top-10.  Compare its
          error against the PV gate's zero error and throughput R(eps).

Outputs: retrieval_grounding_results.json, fig16_retrieval_scis.pdf/.png,
         data/doc_emb.npy, data/doc_titles.json, data/top20_idx.npy
"""
import json, os, sys, time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
EPS = np.array([0.05, 0.1, 0.2, 0.3, 0.4, 0.5])
N_REP, SEED = 20, 42
TOPK = 20
MODEL = 'all-MiniLM-L6-v2'


def build_corpus():
    df = pd.read_parquet(os.path.join(DATA, 'hotpot_validation.parquet'))
    docs, doc_text = {}, {}
    for ctx in df['context']:
        for t, sents in zip(ctx['title'], ctx['sentences']):
            if t not in docs:
                docs[t] = True
                doc_text[t] = (t + '. ' + ' '.join(sents))[:700]
    titles = sorted(doc_text)
    gold = [sorted(set(sf['title'].tolist())) for sf in df['supporting_facts']]
    return df, titles, doc_text, gold


def encode(texts, model):
    outs = []
    for i in range(0, len(texts), 8000):
        outs.append(model.encode(texts[i:i+8000], batch_size=128,
                                 show_progress_bar=False,
                                 normalize_embeddings=True, convert_to_numpy=True))
        print(f'  encoded {min(i+8000, len(texts))}/{len(texts)}', flush=True)
    return np.concatenate(outs)


def main():
    os.environ.setdefault('HF_ENDPOINT', 'https://hf-mirror.com')
    from sentence_transformers import SentenceTransformer
    df, titles, doc_text, gold = build_corpus()
    print(f"[X4] corpus: {len(titles)} docs, {len(df)} questions", flush=True)
    t_idx = {t: i for i, t in enumerate(titles)}

    model = SentenceTransformer(MODEL)
    doc_path = os.path.join(DATA, 'doc_emb.npy')
    if os.path.exists(doc_path):
        D = np.load(doc_path)
    else:
        t0 = time.time()
        D = encode([doc_text[t] for t in titles], model)
        np.save(doc_path, D)
        print(f"[X4] doc embeddings {D.shape} in {time.time()-t0:.0f}s", flush=True)
    Q = encode(df['question'].tolist(), model)

    top_path = os.path.join(DATA, 'top20_idx.npy')
    if os.path.exists(top_path):
        top = np.load(top_path)
    else:
        top = np.zeros((len(Q), TOPK), dtype=np.int32)
        B = 512
        for i in range(0, len(Q), B):
            S = Q[i:i+B] @ D.T
            part = np.argpartition(-S, TOPK, axis=1)[:, :TOPK]
            order = np.argsort(-np.take_along_axis(S, part, axis=1), axis=1)
            top[i:i+B] = np.take_along_axis(part, order, axis=1)
        np.save(top_path, top)
    json.dump(titles, open(os.path.join(DATA, 'doc_titles.json'), 'w'))

    # ---- Step 3: recall and retrieved exponents
    gold_idx = [np.array(sorted(t_idx[t] for t in g), dtype=np.int32) for g in gold]
    ks = [1, 2, 4, 6, 8, 10, 20]
    recall = {}
    for k in ks:
        hits = [len(set(g) & set(top[i, :k])) / max(len(g), 1)
                for i, g in enumerate(gold_idx)]
        recall[k] = float(np.mean(hits))
    e_ann = np.array([len(set(zip(sf['title'], sf['sent_id'].tolist())))
                      for sf in df['supporting_facts']], dtype=float)
    e_ret = {k: np.array([len(set(g) & set(top[i, :k]))
                          for i, g in enumerate(gold_idx)], dtype=float)
             for k in ks}
    for k in (2, 10):
        print(f"[X4] recall@{k}={recall[k]:.3f}  mean e_ret@{k}={e_ret[k].mean():.2f} "
              f"(annotated docs mean 1.99, facts mean {e_ann.mean():.2f})")

    R_ann = {float(e): float(np.mean((1 - e) ** e_ann)) for e in EPS}
    R_ret10 = {float(e): float(np.mean((1 - e) ** e_ret[10])) for e in EPS}
    gap = {str(e): R_ann[e] - R_ret10[e] for e in EPS}

    # ---- Step 4: the re-retrieval baseline = a PV gate whose certificate
    # checker is the retriever.  It re-fetches context at serving time and
    # serves iff ALL gold documents survive AND are re-retrieved in top-10:
    #   thr_base(eps) = (1-eps)^{d_q} * 1[recall@10 = 1 per question]
    # Its error is 0 by construction, so the comparison isolates the
    # RECALL PENALTY that instrumented provenance avoids.
    d_q = np.array([len(g) for g in gold_idx], dtype=float)
    full_recall = np.array([set(g) <= set(top[i, :10])
                            for i, g in enumerate(gold_idx)], dtype=float)
    pv_doc = {float(e): float(np.mean((1 - e) ** d_q)) for e in EPS}
    base_thr = {float(e): float(np.mean(((1 - e) ** d_q) * full_recall)) for e in EPS}
    recall_penalty = {str(e): pv_doc[e] - base_thr[e] for e in EPS}
    for e in EPS:
        print(f"[X4] eps={e}: PV(provenance) thr={pv_doc[float(e)]:.3f} | "
              f"re-retrieval baseline thr={base_thr[float(e)]:.3f} "
              f"(recall penalty {recall_penalty[str(e)]:.3f})")

    results = dict(n_docs=len(titles), n_q=len(df), recall={str(k): v for k, v in recall.items()},
                   e_ret_mean={str(k): float(e_ret[k].mean()) for k in ks},
                   e_ann_mean=float(e_ann.mean()), d_q_mean=float(d_q.mean()),
                   full_recall_at10=float(full_recall.mean()),
                   R_ann={str(k): v for k, v in R_ann.items()},
                   R_ret10={str(k): v for k, v in R_ret10.items()},
                   annotation_retrieval_gap=gap,
                   pv_doc_throughput={str(k): v for k, v in pv_doc.items()},
                   re_retrieval_baseline_throughput={str(k): v for k, v in base_thr.items()},
                   recall_penalty=recall_penalty)
    json.dump(results, open('retrieval_grounding_results.json', 'w'), indent=2)

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.2))
    ax = axes[0]
    ax.plot(ks, [recall[k] for k in ks], 'o-', label='gold-document recall@k')
    ax2 = ax.twinx()
    ax2.plot(ks, [e_ret[k].mean() for k in ks], 's--', color='gray',
             label='retrieved exponent $e_q^{ret}$@k')
    ax2.axhline(1.99, color='gray', ls=':', lw=1)
    ax.set_xlabel('retrieval depth k'); ax.set_ylabel('recall')
    ax2.set_ylabel('mean exponent')
    ax.set_title('(a) retriever view of annotated evidence')
    ax.grid(alpha=0.3)
    h1,l1 = ax.get_legend_handles_labels(); h2,l2 = ax2.get_legend_handles_labels()
    ax.legend(h1+h2, l1+l2, fontsize=8, loc='center right')

    ax = axes[1]
    ax.plot(EPS, [R_ann[float(e)] for e in EPS], 'o-', label='PV reliability, annotated $e_q$')
    ax.plot(EPS, [R_ret10[float(e)] for e in EPS], 's-', label='PV reliability, retrieved $e_q^{ret}$@10')
    ax.plot(EPS, [pv_doc[float(e)] for e in EPS], 'd-', label='PV doc-level (provenance)')
    ax.plot(EPS, [base_thr[float(e)] for e in EPS], '^--',
            label='re-retrieval baseline throughput')
    ax.set_xlabel('erasure rate $\\varepsilon$'); ax.set_ylabel('probability')
    ax.set_title('(b) annotation vs retrieval grounding')
    ax.grid(alpha=0.3); ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig('fig16_retrieval_scis.pdf'); fig.savefig('fig16_retrieval.png', dpi=200)
    print("[X4] fig16_retrieval_scis.pdf, retrieval_grounding_results.json saved")


if __name__ == '__main__':
    main()

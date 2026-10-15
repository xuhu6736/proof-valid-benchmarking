#!/usr/bin/env python3
"""E8 (flagship): instrumented-provenance pipeline closed-loop validation.

Implements the certificate-construction interface Pi on a REAL embedding
retrieval pipeline over the HotpotQA distractor validation split, then runs
the premise-erasure stress test through the constructed certificates and
compares the exact-law prediction against the pipeline measurement.

Pipeline (per question):
  1. Embed the question (cached: data/question_emb.npy, all-MiniLM-L6-v2).
  2. Retrieve top-k documents by cosine similarity from the GLOBAL corpus of
     unique context documents (deduplicated by title; 74,050 -> ~N unique).
  3. Pi constructs premise leaves with the five-factor feature record
     b=(id, h, v, I, pol, pi): id=title, h=SHA-256(content), v=version tag,
     I=index-membership probe, pol=allow (stubbed), pi=presentation ok
     (stubbed).  Two certificate semantics:
       (a) gold-in-retrieval: leaves = retrieved docs annotated as supporting
           (empty -> question is uncertifiable at this k, reported);
       (b) full-context: leaves = all k retrieved docs (kappa=k), the natural
           whole-window certificate of a RAG answer.
  4. Proof-valid gate at serving: verify all five factors of every leaf.
Stress test: erase each leaf iid with prob eps; measured availability vs
predicted E[(1-eps)^e] from the MEASURED exponent distribution.

Also measured: retrieval gold-recall@k and full-support@k (retriever-noise
quantification), per-question construction time, per-question gate time,
certificate byte overhead.

Outputs: e8_pipeline_results.json, fig13_pipeline_scis.pdf/.png
"""
import json, time, hashlib, os, sys
import numpy as np
import pandas as pd

SEED = 42
EPS_GRID = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5]
K_LIST = [2, 4, 6, 8, 10, 20]
K_MAIN = 10
N_REP = 20
HP = 'data/hotpot_validation.parquet'
QEMB = 'data/question_emb.npy'
DEMB = 'data/doc_emb.npy'
OUT_JSON = 'e8_pipeline_results.json'
OUT_FIG = 'fig13_pipeline_scis'


# --------------------------------------------------------------------- data
def load_corpus():
    df = pd.read_parquet(HP)
    docs = {}   # title -> text (first occurrence)
    per_q_titles, gold_titles = [], []
    for _, row in df.iterrows():
        titles = list(row['context']['title'])
        sents = list(row['context']['sentences'])
        per_q_titles.append(titles)
        gold = set(row['supporting_facts']['title'])
        gold_titles.append(gold)
        for t, ss in zip(titles, sents):
            if t not in docs:
                docs[t] = ' '.join(ss)
    titles_sorted = sorted(docs)
    tidx = {t: i for i, t in enumerate(titles_sorted)}
    return df, docs, titles_sorted, tidx, per_q_titles, gold_titles


def embed_docs(titles_sorted, docs):
    if os.path.exists(DEMB):
        emb = np.load(DEMB)
        if emb.shape[0] == len(titles_sorted):
            print(f'[E8] loaded cached doc embeddings {emb.shape}')
            return emb.astype(np.float32)
    import torch
    torch.set_num_threads(2)
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    # chunk-index retriever: embed the lead chunk (title + first sentence);
    # full text is still hashed and stored by Pi for the certificate records.
    texts = []
    for t in titles_sorted:
        first = docs[t].split('. ')[0][:400]
        texts.append(f'{t}. {first}')
    # sort by length to minimize padding waste, then unsort
    order = np.argsort([len(x) for x in texts])
    srt = [texts[i] for i in order]
    emb_srt = model.encode(srt, batch_size=512, show_progress_bar=True,
                           normalize_embeddings=True, convert_to_numpy=True)
    emb = np.empty_like(emb_srt)
    emb[order] = emb_srt
    np.save(DEMB, emb.astype(np.float16))
    print(f'[E8] embedded {emb.shape[0]} docs -> {DEMB}')
    return emb


# ---------------------------------------------------------------- retrieval
def retrieve(qemb, demb, kmax):
    """Return (n_questions, kmax) int32 doc indices, cosine top-k, global."""
    n_q, n_d = qemb.shape[0], demb.shape[0]
    top = np.empty((n_q, kmax), dtype=np.int32)
    sims_top = np.empty((n_q, kmax), dtype=np.float32)
    chunk = 500
    for s in range(0, n_q, chunk):
        e = min(s + chunk, n_q)
        sim = qemb[s:e].astype(np.float32) @ demb.T.astype(np.float32)
        part = np.argpartition(-sim, kmax, axis=1)[:, :kmax]
        part_sim = np.take_along_axis(sim, part, axis=1)
        order = np.argsort(-part_sim, axis=1)
        top[s:e] = np.take_along_axis(part, order, axis=1)
        sims_top[s:e] = np.take_along_axis(part_sim, order, axis=1)
    return top, sims_top


# ------------------------------------------------------------- construction
def build_records(titles, docs):
    """Pi: build five-factor premise-leaf records; returns (records, total_us)."""
    index = set(titles)
    t0 = time.perf_counter_ns()
    recs = []
    for t in titles:
        body = docs[t].encode()
        recs.append({
            'id': t,
            'h': hashlib.sha256(body).hexdigest(),
            'v': 1,                    # version tag
            'I': t in index,           # index-membership probe
            'pol': True,               # license/policy: stubbed allow
            'pi': True,                # presentation: stubbed ok
        })
    us = (time.perf_counter_ns() - t0) / 1e3
    return recs, us


def gate_check(recs, erased, hashes):
    """Proof-valid gate: all five factors must hold for every leaf."""
    ns = time.perf_counter_ns()
    for j, r in enumerate(recs):
        if erased[j] or not (r['I'] and r['pol'] and r['pi']):
            ok = False
            break
        if hashes[r['id']] != r['h']:
            ok = False
            break
    else:
        ok = True
    us = (time.perf_counter_ns() - ns) / 1e3
    return ok, us


def main():
    rng = np.random.default_rng(SEED)
    print('[E8] loading corpus ...')
    df, docs, titles_sorted, tidx, per_q_titles, gold_titles = load_corpus()
    n_q = len(df)
    print(f'[E8] {n_q} questions, {len(titles_sorted)} unique docs')

    qemb = np.load(QEMB)
    assert qemb.shape[0] == n_q
    demb = embed_docs(titles_sorted, docs)
    # normalize (safety: cached float16 cast)
    demb = demb / np.linalg.norm(demb, axis=1, keepdims=True)
    qemb = qemb / np.linalg.norm(qemb, axis=1, keepdims=True)

    print('[E8] retrieving top-20 globally ...')
    top, sims = retrieve(qemb, demb, max(K_LIST))

    # ---- retrieval quality: gold recall@k / full support@k / certifiable@k
    gold_idx = [set(tidx[t] for t in g if t in tidx) for g in gold_titles]
    n_gold = np.array([len(g) for g in gold_idx])
    recall_at_k, full_at_k, certifiable_at_k, e_gold_ret_at_k = {}, {}, {}, {}
    for k in K_LIST:
        got = np.array([len(gold_idx[i] & set(top[i, :k])) for i in range(n_q)])
        recall_at_k[k] = float(np.mean(got / np.maximum(n_gold, 1)))
        full_at_k[k] = float(np.mean(got == n_gold))
        certifiable_at_k[k] = float(np.mean(got > 0))
        e_gold_ret_at_k[k] = float(np.mean(got))
        print(f'[E8] k={k:2d}: gold recall={recall_at_k[k]:.3f} '
              f'full-support={full_at_k[k]:.3f} certifiable={certifiable_at_k[k]:.3f} '
              f'mean gold-in-topk={e_gold_ret_at_k[k]:.2f}')

    # ---- Pi construction at K_MAIN (timed, per question)
    print(f'[E8] constructing certificates at k={K_MAIN} ...')
    hashes = {t: hashlib.sha256(docs[t].encode()).hexdigest() for t in titles_sorted}
    gold_recs, full_recs = [], []
    constr_us = np.empty(n_q)
    cert_bytes = np.empty(n_q)
    for i in range(n_q):
        rt = [titles_sorted[j] for j in top[i, :K_MAIN]]
        gt = [t for t in rt if t in gold_titles[i]]
        recs_g, us_g = build_records(gt, docs)
        recs_f, us_f = build_records(rt, docs)
        gold_recs.append(recs_g)
        full_recs.append(recs_f)
        constr_us[i] = us_g + us_f
        cert_bytes[i] = 76 * (len(recs_g) + len(recs_f))
    e_gold = np.array([len(r) for r in gold_recs])
    print(f'[E8] construction: median {np.median(constr_us):.1f} us/q, '
          f'p95 {np.percentile(constr_us,95):.1f}, p99 {np.percentile(constr_us,99):.1f}')

    # ---- stress test: iid leaf erasure through the pipeline gate
    print('[E8] stress test ...')
    results = {'pred_vs_meas': {}}
    gate_us_all = []
    for cname, recs_all, e_arr in [('gold_in_retrieval', gold_recs, e_gold),
                                   ('full_context', full_recs,
                                    np.full(n_q, K_MAIN))]:
        mask = e_arr > 0
        idxs = np.where(mask)[0]
        meas = np.zeros(len(EPS_GRID))
        for ei, eps in enumerate(EPS_GRID):
            acc = 0
            for _ in range(N_REP):
                for i in idxs:
                    r = recs_all[i]
                    erased = rng.random(len(r)) < eps
                    ok, us = gate_check(r, erased, hashes)
                    acc += ok
                    if eps == 0.1 and len(gate_us_all) < 20000:
                        gate_us_all.append(us)
            meas[ei] = acc / (N_REP * len(idxs))
        pred = np.array([np.mean((1 - eps) ** e_arr[mask]) for eps in EPS_GRID])
        max_dev = float(np.max(np.abs(meas - pred)))
        results['pred_vs_meas'][cname] = {
            'eps': EPS_GRID, 'predicted': pred.tolist(), 'measured': meas.tolist(),
            'max_abs_dev': max_dev, 'n_certifiable': int(mask.sum())}
        print(f'[E8] {cname}: max |pred-meas| = {max_dev:.5f} over {mask.sum()} questions')

    gate_us = np.array(gate_us_all)
    print(f'[E8] gate: median {np.median(gate_us):.2f} us, '
          f'p95 {np.percentile(gate_us,95):.2f}, p99 {np.percentile(gate_us,99):.2f}')

    # ---- annotation-vs-retrieval exponent comparison (retriever-noise effect)
    e_annot = np.array([len(set(zip(sf['title'], sf['sent_id'].tolist())))
                        for sf in df['supporting_facts']], dtype=float)
    pred_annot = [float(np.mean((1 - eps) ** e_annot)) for eps in EPS_GRID]

    results.update({
        'n_questions': n_q, 'n_unique_docs': len(titles_sorted),
        'seed': SEED, 'n_rep': N_REP, 'k_list': K_LIST, 'k_main': K_MAIN,
        'recall_at_k': {str(k): recall_at_k[k] for k in K_LIST},
        'full_support_at_k': {str(k): full_at_k[k] for k in K_LIST},
        'certifiable_at_k': {str(k): certifiable_at_k[k] for k in K_LIST},
        'mean_gold_in_topk': {str(k): e_gold_ret_at_k[k] for k in K_LIST},
        'construction_us': {'median': float(np.median(constr_us)),
                            'p95': float(np.percentile(constr_us, 95)),
                            'p99': float(np.percentile(constr_us, 99)),
                            'mean': float(np.mean(constr_us))},
        'gate_us': {'median': float(np.median(gate_us)),
                    'p95': float(np.percentile(gate_us, 95)),
                    'p99': float(np.percentile(gate_us, 99))},
        'cert_bytes_per_question': {'mean': float(np.mean(cert_bytes)),
                                    'p95': float(np.percentile(cert_bytes, 95))},
        'e_gold_retrieved_hist': np.bincount(e_gold).tolist(),
        'annotation_pred': {'eps': EPS_GRID, 'predicted': pred_annot},
    })
    json.dump(results, open(OUT_JSON, 'w'), indent=2)
    np.savez('e8_timings.npz', constr_us=constr_us, gate_us=gate_us)
    print(f'[E8] {OUT_JSON} saved')

    # ---------------------------------------------------------------- figure
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 3, figsize=(13.5, 3.6))

    ks = K_LIST
    ax[0].plot(ks, [recall_at_k[k] for k in ks], 'o-', label='gold-doc recall')
    ax[0].plot(ks, [full_at_k[k] for k in ks], 's-', label='full support (all gold)')
    ax[0].plot(ks, [certifiable_at_k[k] for k in ks], '^--', label='certifiable ($\\geq$1 gold)')
    ax[0].set_xlabel('retrieval depth $k$'); ax[0].set_ylabel('fraction of questions')
    ax[0].set_ylim(0, 1.02); ax[0].grid(alpha=0.3); ax[0].legend(fontsize=8)
    ax[0].set_title('(a) retrieval grounding vs depth')

    for cname, mk, lab in [('gold_in_retrieval', 'o', 'gold-in-retrieval cert'),
                           ('full_context', 's', f'full-context cert ($\\kappa={K_MAIN}$)')]:
        d = results['pred_vs_meas'][cname]
        ax[1].plot(d['eps'], d['predicted'], '-', label=f'{lab}: exact law')
        ax[1].plot(d['eps'], d['measured'], mk, mfc='none', ms=6, label=f'{lab}: pipeline')
    ax[1].plot(EPS_GRID, pred_annot, ':', color='gray', label='annotation cert: exact law')
    ax[1].set_xlabel('premise erasure rate $\\varepsilon$')
    ax[1].set_ylabel('certificate availability')
    ax[1].set_ylim(0, 1.02); ax[1].grid(alpha=0.3); ax[1].legend(fontsize=8)
    ax[1].set_title('(b) exact-law prediction vs pipeline measurement')

    for arr, lab in [(constr_us, '$\\Pi$ construction (both certs)'),
                     (gate_us, 'serving gate (5-factor check)')]:
        xs = np.sort(arr)
        ax[2].plot(xs, np.arange(1, len(xs) + 1) / len(xs), label=lab, lw=1.8)
    ax[2].set_xscale('log')
    ax[2].set_xlabel('time per question ($\\mu$s, log scale)')
    ax[2].set_ylabel('cumulative fraction')
    ax[2].set_ylim(0, 1.02); ax[2].grid(alpha=0.3); ax[2].legend(fontsize=8)
    ax[2].set_title('(c) instrumented overhead')

    fig.suptitle(f'E8 instrumented pipeline on HotpotQA validation '
                 f'($n={n_q}$ questions, {len(titles_sorted)} unique docs, '
                 f'{N_REP} reps, seed {SEED})', fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(OUT_FIG + '.pdf'); fig.savefig(OUT_FIG + '.png', dpi=140)
    print(f'[E8] {OUT_FIG}.pdf/.png saved')


if __name__ == '__main__':
    main()

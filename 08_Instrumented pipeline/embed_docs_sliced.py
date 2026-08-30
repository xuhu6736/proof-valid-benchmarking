#!/usr/bin/env python3
"""Slice-wise document embedding with checkpointing (E8 helper).
Usage: python3 embed_docs_sliced.py <n_slices> <slice_idx>
Each call embeds one slice and writes data/doc_emb_part_<idx>.npy.
Merge step (slice_idx == -1) concatenates parts into data/doc_emb.npy."""
import sys, os, time
import numpy as np
import pandas as pd

n_slices, idx = int(sys.argv[1]), int(sys.argv[2])

# rebuild the corpus deterministically (sorted titles)
df = pd.read_parquet('data/hotpot_validation.parquet')
docs = {}
for _, row in df.iterrows():
    for t, ss in zip(row['context']['title'], row['context']['sentences']):
        if t not in docs:
            docs[t] = ' '.join(ss)
titles_sorted = sorted(docs)
N = len(titles_sorted)
bounds = [(i * N) // n_slices for i in range(n_slices + 1)]

if idx == -1:  # merge
    parts = [np.load(f'data/doc_emb_part_{i}.npy') for i in range(n_slices)]
    emb = np.concatenate(parts)
    assert emb.shape[0] == N
    np.save('data/doc_emb.npy', emb.astype(np.float16))
    print(f'merged {emb.shape} -> data/doc_emb.npy')
    sys.exit(0)

out = f'data/doc_emb_part_{idx}.npy'
if os.path.exists(out):
    print(f'slice {idx} already done'); sys.exit(0)
lo, hi = bounds[idx], bounds[idx + 1]
import torch
torch.set_num_threads(2)
from sentence_transformers import SentenceTransformer
m = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
ts = titles_sorted[lo:hi]
texts = [f'{t}. {docs[t].split(". ")[0][:400]}' for t in ts]
t0 = time.time()
emb = m.encode(texts, batch_size=256, show_progress_bar=False,
               normalize_embeddings=True, convert_to_numpy=True)
np.save(out, emb.astype(np.float16))
print(f'slice {idx}: {emb.shape} in {time.time()-t0:.0f}s -> {out}')

#!/usr/bin/env python3
"""Step 21: brute-force retrieval scaling curve.

Reviewer item: the staged-timing table times retrieval over the 154-entry
cache only; "retrieval dominates, deploy with an ANN index" presumes a
scaling claim.  Measure exact brute-force cosine retrieval (bge-m3 dim =
1024, float32, single thread) at cache sizes 154 / 1e3 / 1e4 / 1e5 / 1e6
with synthetic unit-norm embeddings, 200 queries per size (20 at 1e6),
median wall time per query.

Outputs: data/retrieval_scaling.json + printed table.
"""
import json, os, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.normpath(os.path.join(HERE, '..', 'data'))

rng = np.random.default_rng(20261017)
DIM = 1024
SIZES = [154, 1_000, 10_000, 100_000, 1_000_000]


def bench(n, nq):
    E = rng.standard_normal((n, DIM), dtype=np.float32)
    E /= np.linalg.norm(E, axis=1, keepdims=True)
    qs = rng.standard_normal((nq, DIM), dtype=np.float32)
    qs /= np.linalg.norm(qs, axis=1, keepdims=True)
    ts = []
    for q in qs:
        t0 = time.perf_counter()
        sims = E @ q
        int(np.argmax(sims))
        ts.append((time.perf_counter() - t0) * 1e6)
    ts.sort()
    return ts[len(ts) // 2], ts[int(0.99 * len(ts)) - 1]


def main():
    rows = []
    for n in SIZES:
        nq = 200 if n <= 100_000 else 20
        med, p99 = bench(n, nq)
        rows.append({'n': n, 'queries': nq,
                     'median_us': round(med, 1), 'p99_us': round(p99, 1)})
        print(f'N={n:>8,}  median {med:9.1f} us   p99 {p99:9.1f} us')
    json.dump({'dim': DIM, 'dtype': 'float32', 'rows': rows},
              open(f'{DATA}/retrieval_scaling.json', 'w'), indent=1)
    print('wrote data/retrieval_scaling.json')


if __name__ == '__main__':
    main()

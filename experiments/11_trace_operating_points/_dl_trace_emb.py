
import sys, site, time, traceback
sys.path.append(site.getusersitepackages())
import numpy as np, requests, pyarrow.parquet as pq

class RangeFile:
    """File-like object over HTTP range requests with timeout+retry."""
    def __init__(self, url, block=1<<23):
        self.url = url; self.pos = 0; self.block = block
        self.cache = {}
        r = requests.head(url, allow_redirects=True, timeout=60)
        self.size = int(r.headers['Content-Length'])
    def tell(self): return self.pos
    def seek(self, pos, whence=0):
        self.pos = pos if whence==0 else (self.pos+pos if whence==1 else self.size+pos)
        return self.pos
    def _fetch(self, b):
        if b in self.cache: return self.cache[b]
        lo, hi = b*self.block, min((b+1)*self.block, self.size)-1
        for attempt in range(5):
            try:
                r = requests.get(self.url, headers={'Range': f'bytes={lo}-{hi}'}, timeout=120)
                r.raise_for_status()
                self.cache[b] = r.content
                if len(self.cache) > 8: self.cache.pop(next(iter(self.cache)))
                return self.cache[b]
            except Exception as e:
                print(f'retry {attempt} block {b}: {type(e).__name__}', flush=True)
                time.sleep(3)
        raise IOError('range fetch failed')
    def read(self, n=-1):
        if n < 0: n = self.size - self.pos
        out = bytearray()
        while len(out) < n:
            b = (self.pos+len(out)) // self.block
            data = self._fetch(b)
            off = (self.pos+len(out)) % self.block
            take = min(len(data)-off, n-len(out))
            out += data[off:off+take]
        self.pos += n
        return bytes(out)
    def close(self): pass
    @property
    def closed(self): return False
    def readable(self): return True
    def seekable(self): return True
    def writable(self): return False
    def flush(self): pass

OUT='/mnt/agents/output/grounded_experiment/data'
JOBS = [
 ('sq','https://hf-mirror.com/datasets/vCache/SemBenchmarkSearchQueries/resolve/main/other_60k_with_responses.parquet'),
 ('la','https://hf-mirror.com/datasets/vCache/SemBenchmarkLmArena/resolve/main/train.parquet'),
]
for tag, url in JOBS:
    try:
        t0=time.time()
        import pyarrow as pa
        rf = RangeFile(url)
        pf = pq.ParquetFile(pa.PythonFile(rf, mode='r'))
        ids, embs = [], []
        import json as _json
        for batch in pf.iter_batches(batch_size=5000, columns=['id_set','emb_gte']):
            ids.extend(batch.column('id_set').to_pylist())
            arr = batch.column('emb_gte')
            import pyarrow as _pa
            if _pa.types.is_string(arr.type):
                rows = [np.fromstring(v.strip('[]'), sep=',', dtype=np.float32)
                        for v in arr.to_pylist()]
                embs.append(np.stack(rows))
            else:
                la = arr.combine_chunks()
                embs.append(la.flatten().to_numpy(zero_copy_only=False).reshape(la.length(), -1))
            print(f'{tag}: {len(ids)} rows, {time.time()-t0:.0f}s', flush=True)
            if len(ids) >= 20000: break
        emb = np.concatenate(embs)[:20000].astype(np.float32)
        np.savez(f'{OUT}/trace_raw_{tag}.npz', ids=np.array(ids[:20000], dtype=object), emb=emb)
        print(f'{tag} DONE {emb.shape} in {time.time()-t0:.0f}s', flush=True)
        del emb, embs, ids
    except Exception:
        traceback.print_exc()
print('all done', flush=True)

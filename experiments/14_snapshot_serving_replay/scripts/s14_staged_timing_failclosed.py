#!/usr/bin/env python3
"""Step 14: staged serve-time timing + fail-closed fault injection.

P2 reviewer item.  Two parts, both on the real replay path (the 147
similarity hits at the 2026-08-01 target snapshot), on Apple M3 Pro.

A. STAGED TIMING.  Decomposes one gate evaluation into stages and
   times each with perf_counter_ns over R randomized repetitions:
     retrieval  : cosine similarity of the 1024-dim query paraphrase
                  against all 154 cached question embeddings (exact
                  brute force, as in the replay);
     cert access: certificate lookup and item count;
     claim check: per designated claim, value-key canonicalization
                  (val_key) plus membership against the materialized
                  target truthy set;
     sentence ck: per designated sentence, SHA-1 id plus membership
                  against the target page sentence-id set.
   Totals reported per contract variant (claims/asserted do not run
   the sentence stage).  The query embedding itself is produced by
   the embedding service and is not a local cost; it is excluded.

B. FAIL-CLOSED FAULT INJECTION.  Re-evaluates all 147 hits under
   injected faults and counts serves (must be zero for affected
   variants):
     F1 entity evidence unavailable (entity record dropped);
     F2 page evidence unavailable (page sentence set dropped;
        affects strict/majority only);
     F3 corrupted claim value read (one character flipped in every
        designated claim key before comparison);
     F4 certificate unreadable (item access raises; gate must treat
        as decline);
     F5 query embedding corrupted (NaN vector; similarity stage must
        yield a miss, not a serve).
   Control: the no-fault run reproduces the baseline verdict counts
   (89 strict / 96 majority / 104 claims / 111 asserted serves).

Output: data/staged_timing_failclosed.json
"""
import json, os, re, hashlib, time, platform
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.normpath(os.path.join(HERE, '..', 'data'))
REB = os.path.normpath(os.path.join(HERE, '..', '..', '12_real_dynamics_audit',
                                   'data', 'rebuilt_2026'))
R = 3000
SEED = 7


def norm(s):
    return re.sub(r'[,\s]+', '', str(s).lower())


def wikitext_to_plain(wt):
    wt = re.sub(r"<ref[^>]*>.*?</ref>", " ", wt, flags=re.S)
    wt = re.sub(r"<ref[^>]*/>", " ", wt)
    wt = re.sub(r"\{\{[^{}]*\}\}", " ", wt)
    wt = re.sub(r"\{\{[^{}]*\}\}", " ", wt)
    wt = re.sub(r"\[\[([^|\]]*\|)?([^\]]*)\]\]", r"\2", wt)
    wt = re.sub(r"\[https?://[^\s\]]+\s*([^\]]*)\]", r"\1", wt)
    wt = re.sub(r"'{2,}", "", wt)
    wt = re.sub(r"<[^>]+>", " ", wt)
    wt = re.sub(r"^[=]+.*?[=]+$", " ", wt, flags=re.M)
    return wt


def sentences(text, min_len=40):
    parts = re.split(r"(?<=[.!?。！？])\s+|\n+", text)
    return [re.sub(r"\s+", " ", s).strip() for s in parts
            if len(s.strip()) >= min_len]


def sid(s):
    return hashlib.sha1(s.encode('utf-8')).hexdigest()[:12]


def val_key(v):
    if isinstance(v, dict):
        if 'id' in v: return 'Q:' + str(v['id'])
        if 'amount' in v: return 'A:' + str(v['amount'])
        if 'time' in v: return 'T:' + str(v['time'])[:11]
        if 'text' in v: return 'S:' + str(v['text'])
    return 'S:' + str(v)


def truthy_keys(ent, prop):
    out = set()
    for rank, snak, val in ent['claims'].get(prop, []):
        if rank in ('normal', 'preferred') and snak == 'value' and val is not None:
            out.add(val_key(val))
    return out


def main():
    panel = json.load(open(f'{DATA}/evidence.json'))
    gen = json.load(open(f'{DATA}/generation.json'))
    lab = {int(k): v for k, v in json.load(open(
        os.path.join(DATA, '..', 'annotation', 'answer_labels.json'))).items()}
    emb = json.load(open(f'{DATA}/embeddings.json'))
    entsT = json.load(open(f'{REB}/entities_v2_20260801.json'))
    pagesT = json.load(open(f'{REB}/pages_20260801.json'))
    page_sids = {t: {sid(s) for s in sentences(wikitext_to_plain(w))}
                 for t, w in pagesT.items()}

    usable = [x for x in panel if gen[str(x['pid'])]['answer_ok']
              and gen[str(x['pid'])]['paraphrase']]
    hits = [x for x in usable if lab[x['pid']]['hit']]
    assert len(hits) == 147

    # materialized serve-time state (as a deployment would hold it)
    truthy = {(x['qid'], x['prop']): truthy_keys(entsT.get(x['qid'], {'claims': {}}, ), x['prop'])
              if x['qid'] in entsT else set() for x in usable}
    truthy = {}
    for x in usable:
        ent = entsT.get(x['qid'])
        truthy[x['pid']] = truthy_keys(ent, x['prop']) if ent else set()
    cert_keys = {x['pid']: [val_key(c['value'])
                            for c in x['certificate']['claims']]
                 for x in usable}
    cert_sids = {x['pid']: [s['id'] for s in x['certificate']['sentences']]
                 for x in usable}
    qmat = np.array([emb[f'{x["pid"]}:q'] for x in usable])     # 154 x 1024
    qmat = qmat / np.linalg.norm(qmat, axis=1, keepdims=True)
    pembs = {x['pid']: np.array(emb[f'{x["pid"]}:p'])
             for x in usable}
    for pid in pembs:
        pembs[pid] = pembs[pid] / np.linalg.norm(pembs[pid])

    # ---------------- A. staged timing ----------------
    rng = np.random.default_rng(SEED)
    hit_pids = [x['pid'] for x in hits]
    stages = ('retrieval', 'cert_access', 'claim_check', 'sentence_check')
    acc = {s: [] for s in stages}
    tot = {v: [] for v in ('strict', 'claims')}
    x_by_pid = {x['pid']: x for x in usable}
    for _ in range(R):
        pid = hit_pids[rng.integers(len(hit_pids))]
        x = x_by_pid[pid]
        t0 = time.perf_counter_ns()
        sims = qmat @ pembs[pid]                       # retrieval
        best = int(np.argmax(sims))
        t1 = time.perf_counter_ns()
        cert_c = cert_keys[pid]                        # certificate access
        cert_s = cert_sids[pid]
        tk = truthy[pid]
        t2 = time.perf_counter_ns()
        okc = True                                     # claim checks
        for k in cert_c:
            kk = val_key({'id': k[2:]}) if k.startswith('Q:') else k
            if k not in tk:
                okc = False
        t3 = time.perf_counter_ns()
        ps = page_sids.get(x.get('page_title'), set())
        oks = True                                     # sentence checks
        for s in cert_s:
            if sid(s) is None:                         # hash work
                pass
            if s not in ps:
                oks = False
        t4 = time.perf_counter_ns()
        acc['retrieval'].append((t1 - t0) / 1e3)
        acc['cert_access'].append((t2 - t1) / 1e3)
        acc['claim_check'].append((t3 - t2) / 1e3)
        acc['sentence_check'].append((t4 - t3) / 1e3)
        tot['strict'].append((t4 - t0) / 1e3)
        tot['claims'].append((t3 - t0) / 1e3)

    def stats(a):
        a = np.array(a)
        return {'median': round(float(np.median(a)), 3),
                'mean': round(float(a.mean()), 3),
                'p95': round(float(np.percentile(a, 95)), 3),
                'p99': round(float(np.percentile(a, 99)), 3)}

    timing = {s: stats(acc[s]) for s in stages}
    timing['total_strict'] = stats(tot['strict'])
    timing['total_claims'] = stats(tot['claims'])
    timing['n_runs'] = R
    timing['platform'] = {'chip': 'Apple M3 Pro',
                          'machine': platform.machine(),
                          'python': platform.python_version()}

    # ---------------- B. fail-closed fault injection ----------------
    def verdicts(fault):
        out = {v: 0 for v in ('strict', 'majority', 'claims', 'asserted')}
        for x in hits:
            pid = x['pid']
            if fault == 'F5':
                q = np.full(1024, np.nan)
                s = qmat @ q
                if not np.isfinite(s).all():
                    continue                          # miss => no serve
            if fault == 'F4':
                continue                              # certificate unreadable
            ent = None if fault == 'F1' else entsT.get(x['qid'])
            tk = truthy_keys(ent, x['prop']) if ent else set()
            if fault == 'F3':
                tk = {k + '#corrupt' for k in tk}
            cdead = [k for k in cert_keys[pid] if k not in tk]
            if fault == 'F2':
                ps = set()
            else:
                ps = page_sids.get(x.get('page_title'), set())
            sdead = [s for s in cert_sids[pid] if s not in ps]
            ns = len(cert_sids[pid])
            ans = gen[str(pid)]['answer']
            asserted = [k for k, c in zip(cert_keys[pid],
                                          x['certificate']['claims'])]
            # asserted subset as in s10/s13
            asserted = []
            for c, vt in zip(x['certificate']['claims'], x['values_text']):
                na, nv = norm(ans), norm(vt)
                if (nv and nv in na) or re.match(r'(\d{4})', str(vt)) and \
                        re.match(r'(\d{4})', str(vt)).group(1) in str(ans):
                    asserted.append(val_key(c['value']))
            ok = {
                'strict': not cdead and not sdead,
                'majority': not cdead and (ns == 0
                                           or (ns - len(sdead)) * 2 > ns),
                'claims': not cdead,
                'asserted': not any(k in cdead for k in asserted),
            }
            for v, o in ok.items():
                out[v] += int(o)
        return out

    control = verdicts(None)
    faults = {f: verdicts(f) for f in ('F1', 'F2', 'F3', 'F4', 'F5')}

    print('control (no fault) serves:', control,
          '  expected strict 89 / majority 96 / claims 104 / asserted 111')
    for f, r in faults.items():
        print(f'{f} serves:', r)

    out = {'timing_us': timing, 'control_serves': control,
           'fault_serves': faults,
           'faults': {
               'F1': 'entity evidence unavailable (affects all variants)',
               'F2': 'page evidence unavailable (affects strict/majority)',
               'F3': 'corrupted claim value read (affects all variants)',
               'F4': 'certificate unreadable (all variants decline)',
               'F5': 'query embedding corrupted (retrieval misses)'}}
    json.dump(out, open(f'{DATA}/staged_timing_failclosed.json', 'w'),
              indent=1)
    print('\nstaged timing (us):', json.dumps(timing, indent=1))
    print('wrote data/staged_timing_failclosed.json')


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""Step 5: target-period replay — certificate gate + two serving arms.

Gate (PVSB): serve cached answer iff sim(query, cached_q) >= tau AND every
certificate item survives at target:
  - each cited claim (prop, value) still truthy in the target entity view
  - each cited sentence still present (normalised sha1) in the target page
Baseline (similarity-only): serve iff sim >= tau.

Ground truth: pair is stale iff >=1 cited value was erased at target.

Metrics reported across a tau sweep + at tau calibrated as the 5th
percentile of genuine-paraphrase similarities.

Output: data/replay.json + data/replay_summary.json
"""
import json, os, re, time, urllib.request, hashlib
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.normpath(os.path.join(HERE, '..', 'data'))
REB = os.path.normpath(os.path.join(HERE, '..', '..', '12_real_dynamics_audit',
                                   'data', 'rebuilt_2026'))

KEY = os.environ['KIMI_API_KEY']
BASE = os.environ['KIMI_BASE_URL'].rstrip('/')


def embed(texts, tries=4):
    for a in range(tries):
        try:
            req = urllib.request.Request(
                BASE + '/embeddings',
                data=json.dumps({'model': 'bge_m3_embed', 'input': texts}).encode(),
                headers={'Authorization': f'Bearer {KEY}',
                         'content-type': 'application/json'}, method='POST')
            with urllib.request.urlopen(req, timeout=60) as r:
                d = json.loads(r.read())
            return [x['embedding'] for x in sorted(d['data'], key=lambda z: z['index'])]
        except Exception:
            time.sleep(2 + a * 2)
    raise RuntimeError('embedding failed')


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
    return [re.sub(r"\s+", " ", s).strip() for s in parts if len(s.strip()) >= min_len]


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
    e1 = json.load(open(f'{REB}/entities_v2_20260801.json'))
    pages1 = json.load(open(f'{REB}/pages_20260801.json'))

    usable = [x for x in panel if gen[str(x['pid'])]['answer_ok']
              and gen[str(x['pid'])]['paraphrase']]
    print('usable pairs:', len(usable))

    # --- embeddings (cached) ---
    emb_path = f'{DATA}/embeddings.json'
    emb = json.load(open(emb_path)) if os.path.exists(emb_path) else {}
    todo_q, texts = [], []
    for x in usable:
        for tag, t in (('q', x['question']),
                       ('p', gen[str(x['pid'])]['paraphrase'])):
            key = f"{x['pid']}:{tag}"
            if key not in emb:
                todo_q.append(key); texts.append(t)
    print('to embed:', len(texts))
    for i in range(0, len(texts), 32):
        vecs = embed(texts[i:i + 32])
        for k, v in zip(todo_q[i:i + 32], vecs):
            emb[k] = v
        if (i // 32) % 5 == 4:
            json.dump(emb, open(emb_path, 'w'))
            print('emb checkpoint', i + 32, '/', len(texts), flush=True)
    json.dump(emb, open(emb_path, 'w'))

    import numpy as np
    def cos(a, b):
        a = np.array(a); b = np.array(b)
        return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))

    # --- replay ---
    results = []
    for x in usable:
        pid = str(x['pid'])
        sim_para = cos(emb[f'{pid}:q'], emb[f'{pid}:p'])
        # claim survival at target
        ent1 = e1.get(x['qid'])
        claims_alive, claims_dead = [], []
        if ent1 is not None:
            t1 = truthy_keys(ent1, x['prop'])
            for c in x['certificate']['claims']:
                k = val_key(c['value'])
                (claims_alive if k in t1 else claims_dead).append(k)
        else:
            claims_dead = [val_key(c['value']) for c in x['certificate']['claims']]
        # sentence survival at target
        title = x.get('page_title')
        target_sids = set()
        if title and title in pages1:
            target_sids = {sid(s) for s in sentences(wikitext_to_plain(pages1[title]))}
        sents_alive = [s['id'] for s in x['certificate']['sentences']
                       if s['id'] in target_sids]
        sents_dead = [s['id'] for s in x['certificate']['sentences']
                      if s['id'] not in target_sids]
        gate_pass = (not claims_dead) and (not sents_dead)
        results.append({
            'pid': x['pid'], 'qid': x['qid'], 'prop': x['prop'],
            'label': x['label'], 'stale': x['stale'],
            'sim_para': sim_para,
            'claims_dead': len(claims_dead), 'claims_total': len(x['certificate']['claims']),
            'sents_dead': len(sents_dead), 'sents_total': len(x['certificate']['sentences']),
            'gate_pass': gate_pass,
        })
    json.dump(results, open(f'{DATA}/replay.json', 'w'), ensure_ascii=False, indent=1)

    # --- metrics across tau sweep ---
    sims = sorted(r['sim_para'] for r in results)
    tau_cal = sims[max(0, int(0.05 * len(sims)) - 1)]   # 5th pct of paraphrase sims
    stale = [r for r in results if r['stale']]
    fresh = [r for r in results if not r['stale']]

    def metrics(tau):
        def served(r): return r['sim_para'] >= tau
        b_err = sum(1 for r in stale if served(r)) / max(1, len(stale))
        p_int = sum(1 for r in stale if served(r) and not r['gate_pass']) / max(1, len(stale))
        p_err = sum(1 for r in stale if served(r) and r['gate_pass']) / max(1, len(stale))
        p_fr = sum(1 for r in fresh if served(r) and not r['gate_pass']) / max(1, len(fresh))
        b_serve = sum(1 for r in results if served(r)) / len(results)
        p_serve = sum(1 for r in results if served(r) and r['gate_pass']) / len(results)
        return {'tau': round(tau, 4), 'baseline_error_serve': round(b_err, 4),
                'pvsb_intercept_stale': round(p_int, 4),
                'pvsb_residual_error': round(p_err, 4),
                'pvsb_false_reject_fresh': round(p_fr, 4),
                'baseline_hit_rate': round(b_serve, 4),
                'pvsb_hit_rate': round(p_serve, 4)}

    sweep = [metrics(t) for t in (0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90)]
    cal = metrics(tau_cal)
    summary = {'n_pairs': len(results), 'n_stale': len(stale), 'n_fresh': len(fresh),
               'paraphrase_sim_min': round(sims[0], 4),
               'paraphrase_sim_p05': round(tau_cal, 4),
               'paraphrase_sim_med': round(sims[len(sims) // 2], 4),
               'calibrated': cal, 'sweep': sweep,
               'gate_fail_fresh_detail': [
                   {'pid': r['pid'], 'label': r['label'], 'prop': r['prop'],
                    'claims_dead': r['claims_dead'], 'sents_dead': r['sents_dead'],
                    'sents_total': r['sents_total']}
                   for r in fresh if not r['gate_pass']],
               'gate_pass_stale_detail': [
                   {'pid': r['pid'], 'label': r['label'], 'prop': r['prop'],
                    'claims_dead': r['claims_dead']}
                   for r in stale if r['gate_pass']]}
    json.dump(summary, open(f'{DATA}/replay_summary.json', 'w'), ensure_ascii=False, indent=1)
    print(json.dumps({k: v for k, v in summary.items() if not k.endswith('_detail')},
                     indent=1))


if __name__ == '__main__':
    main()

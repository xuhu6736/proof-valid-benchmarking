#!/usr/bin/env python3
"""Step 20: fixed-global-set protection under a bytes budget.

Reviewer item: the s12 comparison gave model F a PER-ENTRY budget
(floor(0.9 n) of n items) while model R protects each item independently
with probability c -- unequal budgets (at n=1, F protects 0 items while R
protects 0.9 in expectation).  This step re-runs the comparison fairly:

  * one FIXED GLOBAL protected set S, chosen under a global BYTES budget
    (sentence text bytes; claim bytes = UTF-8 length of its value text),
    filled greedily by item coverage frequency across the 147-hit cache;
  * random per-item protection (model R) calibrated to the SAME expected
    byte budget;
  * multiple independent erasure realizations: every unprotected item is
    erased i.i.d. with probability p; protected items survive;
  * serves measured by ACTUAL gate evaluation on the surviving evidence
    (strict and asserted contracts), and e_q(S) reported directly.

Outputs: data/fixed_budget_protection.json + printed table.
"""
import json, os, random

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.normpath(os.path.join(HERE, '..', 'data'))
ANN = os.path.normpath(os.path.join(HERE, '..', 'annotation'))
REB = os.path.normpath(os.path.join(HERE, '..', '..', '12_real_dynamics_audit',
                                   'data', 'rebuilt_2026'))

random.seed(20261017)
REALIZ = 500
PS = (0.05, 0.10, 0.20)
CS = (0.50, 0.90, 0.95)


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
    lab = {int(k): v for k, v in
           json.load(open(os.path.join(ANN, 'answer_labels.json'))).items()}
    asserted = json.load(open(f'{DATA}/asserted_claims.json'))['asserted']
    panel = json.load(open(f'{DATA}/evidence.json'))
    entT = json.load(open(f'{REB}/entities_v2_20260801.json'))
    hits = sorted((p for p, v in lab.items() if v['hit']), key=int)

    # ---- per-entry required items at the target snapshot ----------------
    # required_strict[pid] = (claim keys, sentence ids); only entries whose
    # gate PASSES on the intact target snapshot can lose serves to erasure.
    entries = {}
    freq = {}
    nbytes = {}
    for x in panel:
        pid = x['pid']
        if pid not in hits:
            continue
        ent = entT.get(x['qid'])
        tk = truthy_keys(ent, x['prop']) if ent is not None else set()
        cks = [val_key(c['value']) for c in x['certificate']['claims']]
        sids = [s['id'] for s in x['certificate']['sentences']]
        akeys = asserted.get(str(pid), [])
        strict_ok = all(k in tk for k in cks) and bool(sids_ok(
            x, pid))
        entries[pid] = {
            'claims': cks, 'sents': sids, 'asserted': akeys,
            'claims_alive': all(k in tk for k in cks),
            'asserted_alive': all(k in tk for k in akeys),
        }
        for k in cks:
            freq[k] = freq.get(k, 0) + 1
            nbytes.setdefault(k, len(str(k).encode()))
        for s in x['certificate']['sentences']:
            freq[s['id']] = freq.get(s['id'], 0) + 1
            nbytes.setdefault(s['id'], len(s['text'].encode()))

    # sentence survival comes from replay (page text already resolved there)
    replay = {x['pid']: x for x in json.load(open(f'{DATA}/replay.json'))}
    for pid, e in entries.items():
        r = replay[pid]
        e['sents_alive'] = (r['sents_total'] == 0 or r['sents_dead'] == 0)
        e['strict_pass0'] = e['claims_alive'] and e['sents_alive']
        e['asserted_pass0'] = e['asserted_alive']

    items = sorted(freq, key=lambda k: (-freq[k], k))
    total_bytes = sum(nbytes[k] for k in items)
    total_cites = sum(freq.values())
    n_claim_items = len({k for p in hits for k in entries[p]['claims']})
    n_sent_items = len({k for p in hits for k in entries[p]['sents']})
    multi = [k for k in items if freq[k] > 1]
    cite_dist = {}
    for k in items:
        cite_dist[freq[k]] = cite_dist.get(freq[k], 0) + 1
    bytes_multi = sum(nbytes[k] for k in multi)
    bytes_single = total_bytes - bytes_multi
    print(f'unique designated items: {len(items)}  total bytes: '
          f'{total_bytes}')
    print(f'reuse factor = {total_cites}/{len(items)} = '
          f'{total_cites/len(items):.2f}')
    print(f'unique claim items: {n_claim_items}  unique sentence items: '
          f'{n_sent_items}')
    print(f'citation-count distribution (count: #items): '
          f'{dict(sorted(cite_dist.items()))}')
    print(f'bytes share: multi-cited {100*bytes_multi/total_bytes:.1f}%  '
          f'single-cited {100*bytes_single/total_bytes:.1f}%')

    def greedy_set(budget, cost=None):
        S, used = set(), 0
        cost = cost or nbytes
        for k in items:                      # items already freq-sorted
            if used + cost[k] <= budget:
                S.add(k)
                used += cost[k]
        return S, used

    # dedup-off ablation: a shared item must be stored once per citation,
    # so its greedy cost scales with its citation count
    cost_nodedup = {k: nbytes[k] * freq[k] for k in items}

    SET_DRAWS = 50          # random feasible-set draws
    REALIZ_FS = 100         # failure realizations per drawn set

    def random_feasible_sets(budget, q):
        """Sample Bernoulli(q) placements, keep those within budget."""
        sets = []
        tries = 0
        while len(sets) < SET_DRAWS and tries < 100 * SET_DRAWS:
            tries += 1
            cand = {k for k in items if random.random() < q}
            if sum(nbytes[k] for k in cand) <= budget:
                sets.append(cand)
        return sets

    def expected_serves(protected_check, realiz=None):
        """mean serves over i.i.d. erasure realizations."""
        realiz = realiz or REALIZ
        out = {}
        for gate in ('strict', 'asserted'):
            base = [p for p in hits
                    if entries[p][f'{gate}_pass0']]
            tot = 0.0
            for _ in range(realiz):
                for p in base:
                    req = (entries[p]['claims'] + entries[p]['sents']
                           if gate == 'strict' else entries[p]['asserted'])
                    ok = True
                    for k in req:
                        if not protected_check(k) and random.random() < p_cur:
                            ok = False
                            break
                    tot += ok
            out[gate] = round(tot / realiz, 1)
        return out

    rows = []
    for c in CS:
        budget = c * total_bytes
        S, used = greedy_set(budget)
        S_nd, used_nd = greedy_set(budget, cost_nodedup)
        q = used / total_bytes
        feas_sets = random_feasible_sets(budget, q)
        n_prot_entries = sum(
            1 for p in hits
            if all(k in S for k in entries[p]['claims'] + entries[p]['sents']))
        for p_cur in PS:
            globals()['p_cur'] = p_cur
            fixed = expected_serves(lambda k: k in S)
            fixed_nd = expected_serves(lambda k: k in S_nd)
            rnd = expected_serves(lambda k: random.random() < q)
            # random feasible sets: average over drawn sets x failures
            fs_acc = {'strict': 0.0, 'asserted': 0.0}
            for cand in feas_sets:
                r = expected_serves(lambda k: k in cand, REALIZ_FS)
                fs_acc['strict'] += r['strict']
                fs_acc['asserted'] += r['asserted']
            rnd_fs = {g: round(fs_acc[g] / len(feas_sets), 1)
                      for g in fs_acc}
            rows.append({'c': c, 'p': p_cur,
                         'bytes_used': used, 'q_random': round(q, 3),
                         'entries_fully_protected': n_prot_entries,
                         'fixed_S': fixed, 'random_R': rnd,
                         'random_feasible_set': rnd_fs,
                         'fixed_S_nodedup': fixed_nd,
                         'bytes_used_nodedup': used_nd})
            print(f"c={c:.2f} p={p_cur:.2f}  bytes={used}/{total_bytes} "
                  f"q_R={q:.3f}  e_q(S)=1 for {n_prot_entries}/147  "
                  f"strict fixed {fixed['strict']} vs rnd {rnd['strict']} "
                  f"vs feas {rnd_fs['strict']} vs nd {fixed_nd['strict']}  "
                  f"asserted fixed {fixed['asserted']} vs {rnd['asserted']} "
                  f"vs {rnd_fs['asserted']} vs {fixed_nd['asserted']}")

    # reference: no protection and full protection
    for p_cur in PS:
        none = expected_serves(lambda k: False)
        print(f"p={p_cur:.2f}  no protection: {none}")
    json.dump({'n_items': len(items), 'total_bytes': total_bytes,
               'reuse_stats': {
                   'total_citations': total_cites,
                   'reuse_factor': round(total_cites / len(items), 3),
                   'unique_claim_items': n_claim_items,
                   'unique_sentence_items': n_sent_items,
                   'citation_count_distribution':
                       {str(k): v for k, v in sorted(cite_dist.items())},
                   'multi_cited_items': len(multi),
                   'bytes_share_multi_cited':
                       round(bytes_multi / total_bytes, 4)},
               'realizations': REALIZ,
               'feasible_set_draws': SET_DRAWS,
               'rows': rows},
              open(f'{DATA}/fixed_budget_protection.json', 'w'), indent=1)
    print('wrote data/fixed_budget_protection.json')


def sids_ok(x, pid):
    return True   # placeholder resolved via replay in main


if __name__ == '__main__':
    main()

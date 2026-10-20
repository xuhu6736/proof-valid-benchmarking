#!/usr/bin/env python3
"""Step 12 (package 3): protection recovery + revert negative control.

Arm 1 — recoverable evidence-erasure (serving-side read loss).
At serve time the gate's read of each certificate item independently
fails with probability p (transient shard/read loss in the serving
stack).  Conditions on the same 147 similarity hits and the same human
answer labels, asserted-claims contract (the paper's recommended
variant), and also the strict contract for certificate-size contrast:

  no-protection   : serve iff the audited verdict is serve AND no
                    certificate item read is lost
  protected-audit : PVSB — lost items restored from the protection
                    cache S (transparent copies of the CURRENT source
                    state) and re-audited; verdict identical to the
                    no-erasure baseline
  copy-trust      : negative control — restored BASE-period copies are
                    trusted without a currency audit; every hit serves,
                    including all 8 decayed ones (staleness laundering)
  coverage c      : only a fraction c of certificate items is protected;
                    serve iff every lost item is covered —
                    (1 - p(1-c))^n per entry with n certificate items,
                    i.e. conjunctive certificates need all-paths
                    interception (paper A's dominator condition)

Analytic expectations plus one seeded Monte-Carlo realization
(R = 2000) for concrete counts.

Arm 2 — revert negative control (real data).
For the 9 answer-decayed entries we fetched the CURRENT Wikidata claim
state (2026-10-16, wbgetclaims; raw JSON in data/current_claims/).
An entry has reverted iff every answer-asserted base claim is truthy
again.  Report the three-time-point timeline (base accept / target
decline / now ?) and what change-detector baselines would do.

Output: data/protection_recovery.json
"""
import json, os, re
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.normpath(os.path.join(HERE, '..', 'data'))
REB = os.path.normpath(os.path.join(HERE, '..', '..', '12_real_dynamics_audit',
                                   'data', 'rebuilt_2026'))
T = '20260801'


def norm(s):
    return re.sub(r'[,\s]+', '', str(s).lower())


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


def answer_asserts(answer, vtext):
    na, nv = norm(answer), norm(vtext)
    if nv and nv in na:
        return True
    m = re.match(r'(\d{4})', str(vtext))
    if m and m.group(1) in str(answer):
        return True
    return False


def sid(s):
    import hashlib
    return hashlib.sha1(s.encode('utf-8')).hexdigest()[:12]


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


def main():
    panel = json.load(open(f'{DATA}/evidence.json'))
    gen = json.load(open(f'{DATA}/generation.json'))
    lab = {int(k): v for k, v in json.load(open(
        os.path.join(DATA, '..', 'annotation', 'answer_labels.json'))).items()}
    e1 = json.load(open(f'{REB}/entities_v2_{T}.json'))
    p1 = json.load(open(f'{REB}/pages_{T}.json'))
    page_sids = {t: {sid(s) for s in sentences(wikitext_to_plain(w))}
                 for t, w in p1.items()}

    usable = [x for x in panel if gen[str(x['pid'])]['answer_ok']
              and gen[str(x['pid'])]['paraphrase']]
    hits = [x for x in usable if lab[x['pid']]['hit']]
    assert len(hits) == 147

    # per-hit certificates and target-period verdicts (s10 logic)
    recs = []
    for x in hits:
        pid = x['pid']
        ans = gen[str(pid)]['answer']
        tk = truthy_keys(e1.get(x['qid'], {'claims': {}}), x['prop'])
        claim_keys = [val_key(c['value']) for c in x['certificate']['claims']]
        asserted = [k for c, vt, k in zip(x['certificate']['claims'],
                                          x['values_text'], claim_keys)
                    if answer_asserts(ans, vt)]
        sent_ids = [s['id'] for s in x['certificate']['sentences']]
        tsids = page_sids.get(x.get('page_title'), set())
        verdict = {
            'strict': all(k in tk for k in claim_keys)
                      and all(s in tsids for s in sent_ids),
            'claims': all(k in tk for k in claim_keys),
            'asserted': all(k in tk for k in asserted),
        }
        recs.append({'pid': pid, 'qid': x['qid'], 'prop': x['prop'],
                     'label': x['label'], 'question': x['question'],
                     'claim_keys': claim_keys, 'asserted': asserted,
                     'sent_ids': sent_ids, 'verdict': verdict,
                     'decayed': lab[pid]['A5'] in ('使其失效', '部分影响')})
    n_dec_hits = sum(r['decayed'] for r in recs)
    print(f'hits={len(recs)}, decayed hits={n_dec_hits}')

    # ---------------- Arm 1: erasure x protection, two models ----------------
    # Model R (randomized placement): each certificate item is
    # independently included in the protection cache with prob c; the
    # per-hit survival (1-p(1-c))^n averages over BOTH placement and
    # read loss.  Shared premises across entries would share one
    # placement indicator (not present here: certificates are
    # entry-specific).
    # Model F (fixed placement, the companion's fixed-S setting):
    # a fixed set of x_i = floor(c*n_i) items per certificate is
    # protected, leaving e_i = n_i - x_i exposed; survival is
    # (1-p)^{e_i}, the residual-exposure law.
    P_RATES = [0.01, 0.05, 0.10, 0.20]
    R = 2000
    rng = np.random.default_rng(7)

    arm1 = {}
    for contract in ('asserted', 'strict'):
        base_serve = [r for r in recs if r['verdict'][contract]]
        dec_base = sum(r['decayed'] for r in base_serve)
        ns = np.array([len(r['asserted'] if contract == 'asserted'
                           else r['claim_keys'] + r['sent_ids'])
                       for r in base_serve], dtype=float)
        rows = []
        for p in P_RATES:
            row = {'p': p, 'unprotected': float(((1 - p) ** ns).sum())}
            for c in (0.9, 0.95):
                row[f'rand_{c}'] = float(((1 - p * (1 - c)) ** ns).sum())
            xs = np.floor(0.9 * ns)
            row['fixed_0.9'] = float(((1 - p) ** (ns - xs)).sum())
            row['full'] = float(len(ns))
            rows.append(row)
        # Monte-Carlo check of model R at p = 0.10: mean error + SE
        p = 0.10
        mc = {}
        for name, c in [('rand_0.9', 0.9), ('rand_0.95', 0.95)]:
            counts = []
            for _ in range(R):
                tot = 0
                for n in ns.astype(int):
                    if n == 0:
                        tot += 1
                        continue
                    erase = rng.random(n) < p
                    cov = rng.random(n) < c
                    if not np.any(erase & ~cov):
                        tot += 1
                counts.append(tot)
            counts = np.array(counts, dtype=float)
            analytic = float(((1 - p * (1 - c)) ** ns).sum())
            mc[name] = {'mc_mean': float(counts.mean()),
                        'mc_se': float(counts.std(ddof=1) / np.sqrt(R)),
                        'analytic': analytic,
                        'abs_err': float(abs(counts.mean() - analytic))}
        arm1[contract] = {'baseline_served': len(base_serve),
                          'baseline_decayed_served': dec_base,
                          'mean_cert_size': float(ns.mean()),
                          'analytic': rows, 'mc_p010': mc}

    # copy-trust negative control: everything serves, incl. all decayed
    arm1['copy_trust_no_audit'] = {
        'served': len(recs),
        'decayed_served': sum(r['decayed'] for r in recs)}

    # ---------------- Arm 2: revert negative control ----------------
    cur_dir = f'{DATA}/current_claims'
    arm2 = {'fetched': False, 'entries': []}
    if os.path.isdir(cur_dir):
        dec = [r for r in recs if r['decayed']] + [
            # the one decayed non-hit, for completeness
        ]
        # add the decayed non-hit (pid 51) from usable
        for x in usable:
            pid = x['pid']
            if lab[pid]['A5'] in ('使其失效', '部分影响') and not lab[pid]['hit']:
                ans = gen[str(pid)]['answer']
                tk1 = truthy_keys(e1.get(x['qid'], {'claims': {}}), x['prop'])
                ck = [val_key(c['value']) for c in x['certificate']['claims']]
                asr = [k for c, vt, k in zip(x['certificate']['claims'],
                                             x['values_text'], ck)
                       if answer_asserts(ans, vt)]
                dec.append({'pid': pid, 'qid': x['qid'], 'prop': x['prop'],
                            'label': x['label'], 'asserted': asr,
                            'hit': False,
                            'verdict_target': all(k in tk1 for k in asr)})
        for r in dec:
            if 'verdict_target' not in r:
                r['hit'] = True
                r['verdict_target'] = False  # all decayed hits intercepted
            f = os.path.join(cur_dir, f"Q{r['qid'].lstrip('Q')}"
                                       f"_{r['prop']}.json")
            if not os.path.exists(f):
                f = os.path.join(cur_dir, f"{r['qid']}_{r['prop']}.json")
            if not os.path.exists(f):
                arm2['entries'].append({**{k: r[k] for k in
                                           ('pid', 'qid', 'prop', 'label',
                                            'asserted', 'hit',
                                            'verdict_target')},
                                        'now': 'not fetched'})
                continue
            raw = json.load(open(f))
            ent = {'claims': {}}
            for cl in raw.get('claims', {}).get(r['prop'], []):
                snak = cl.get('mainsnak', {})
                dv = snak.get('datavalue', {}).get('value')
                ent['claims'].setdefault(r['prop'], []).append(
                    (cl.get('rank', 'normal'), snak.get('snaktype', 'value'),
                     dv))
            tk_now = truthy_keys(ent, r['prop'])
            alive = [k for k in r['asserted'] if k in tk_now]
            reverted = len(alive) == len(r['asserted'])
            arm2['entries'].append({
                'pid': r['pid'], 'qid': r['qid'], 'prop': r['prop'],
                'label': r['label'], 'asserted': r['asserted'],
                'hit': r['hit'], 'verdict_target': r['verdict_target'],
                'asserted_alive_now': alive, 'reverted': reverted,
                'now_keys': sorted(tk_now)})
        arm2['fetched'] = any(e.get('now') != 'not fetched'
                              for e in arm2['entries'])

    out = {'arm1': arm1, 'arm2': arm2}
    json.dump(out, open(f'{DATA}/protection_recovery.json', 'w'), indent=1)

    # ---------------- console summary ----------------
    for contract in ('asserted', 'strict'):
        a = arm1[contract]
        print(f"\n== Arm 1, {contract} contract: baseline serves "
              f"{a['baseline_served']}/147 (decayed served "
              f"{a['baseline_decayed_served']}), mean certificate "
              f"{a['mean_cert_size']:.2f} items")
        print(f"{'p':>5} {'unprot':>8} {'r.9':>6} {'r.95':>6} "
              f"{'f.9':>6} {'full':>5}")
        for row in a['analytic']:
            print(f"{row['p']:>5.2f} {row['unprotected']:>8.1f} "
                  f"{row['rand_0.9']:>6.1f} {row['rand_0.95']:>6.1f} "
                  f"{row['fixed_0.9']:>6.1f} {row['full']:>5.0f}")
        print('  MC@p=0.10 (model R):',
              {k: f"mean {v['mc_mean']:.2f} SE {v['mc_se']:.2f} "
                  f"analytic {v['analytic']:.1f} |err| {v['abs_err']:.2f}"
               for k, v in a['mc_p010'].items()})
    ct = arm1['copy_trust_no_audit']
    print(f"\ncopy-trust no-audit: serves {ct['served']}/147 incl. "
          f"{ct['decayed_served']} decayed")
    print('\n== Arm 2 reverts:')
    for e in arm2['entries']:
        if e.get('now') == 'not fetched':
            print(f"  pid {e['pid']} {e['label']}: NOT FETCHED")
        else:
            print(f"  pid {e['pid']} {e['label']!r} {e['prop']}: "
                  f"reverted={e['reverted']} "
                  f"alive {len(e['asserted_alive_now'])}/{len(e['asserted'])}")


if __name__ == '__main__':
    main()

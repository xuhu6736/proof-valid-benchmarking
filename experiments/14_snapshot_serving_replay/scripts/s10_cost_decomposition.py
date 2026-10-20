#!/usr/bin/env python3
"""Step 10: false-decline cost decomposition (package A+B+C).

A. Prevalence reweighting: population-level decline rates at the 4-month
   endpoint, using the natural pair-level staleness prevalence 52/6580
   and the panel-conditional decline rates.
B. Cache-age response: the same gate variants evaluated against the
   intermediate 2026-05-01 and 2026-07-01 rebuilt snapshots (no model
   calls), giving decline rates as a function of cache age.
C. Asserted-claims contract: a gate that re-verifies only the
   designated base claims actually NAMED by the cached answer
   (normalized containment; year-only for dates) -- the annotation
   showed most removed claims were never asserted by the answer.

Outputs: data/age_curve.json, data/asserted_claims.json,
         fig_cache_age.pdf/.png (copied to the paper fig/ folder)
"""
import json, os, re, hashlib
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.normpath(os.path.join(HERE, '..', 'data'))
REB = os.path.normpath(os.path.join(HERE, '..', '..', '12_real_dynamics_audit',
                                   'data', 'rebuilt_2026'))
FIGDIR = os.path.normpath(os.path.join(HERE, '..', '..', '..', '..', 'fig'))

SNAPS = ['20260401', '20260501', '20260701', '20260801']
AGES = [0, 1, 3, 4]  # months of cache age at each snapshot
TAU = 0.8364


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


def answer_asserts(answer, vtext):
    """Does the cached answer name this value?  (same rule as s4 acceptance)"""
    na = norm(answer)
    nv = norm(vtext)
    if nv and nv in na:
        return True
    m = re.match(r'(\d{4})', str(vtext))
    if m and m.group(1) in str(answer):
        return True
    return False


def main():
    panel = json.load(open(f'{DATA}/evidence.json'))
    gen = json.load(open(f'{DATA}/generation.json'))
    replay = {r['pid']: r for r in json.load(open(f'{DATA}/replay.json'))}
    lab = {int(k): v for k, v in json.load(open(
        os.path.join(DATA, '..', 'annotation', 'answer_labels.json'))).items()}

    usable = [x for x in panel if gen[str(x['pid'])]['answer_ok']
              and gen[str(x['pid'])]['paraphrase']]
    assert len(usable) == 154

    # --- asserted claims per entry (variant C) ---
    asserted = {}  # pid -> list of claim val_keys asserted by the answer
    for x in usable:
        pid = x['pid']
        ans = gen[str(pid)]['answer']
        keys = []
        for c, vt in zip(x['certificate']['claims'], x['values_text']):
            if answer_asserts(ans, vt):
                keys.append(val_key(c['value']))
        assert keys, f"entry {pid} asserts no designated value (impossible)"
        asserted[pid] = keys

    # --- gate evaluation against every snapshot (variant B) ---
    snaps = {}
    for tag in SNAPS:
        ents = json.load(open(f'{REB}/entities_v2_{tag}.json'))
        pages = json.load(open(f'{REB}/pages_{tag}.json'))
        page_sids = {t: {sid(s) for s in sentences(wikitext_to_plain(w))}
                     for t, w in pages.items()}
        snaps[tag] = (ents, page_sids)

    per_entry = {}
    for x in usable:
        pid = x['pid']
        title = x.get('page_title')
        row = {}
        for tag in SNAPS:
            ents, page_sids = snaps[tag]
            ent = ents.get(x['qid'])
            tk = truthy_keys(ent, x['prop']) if ent is not None else set()
            cdead = [val_key(c['value']) for c in x['certificate']['claims']
                     if val_key(c['value']) not in tk]
            tsids = page_sids.get(title, set())
            sdead = [s['id'] for s in x['certificate']['sentences']
                     if s['id'] not in tsids]
            ns = len(x['certificate']['sentences'])
            row[tag] = {
                'claims_dead': len(cdead), 'sents_dead': len(sdead),
                'sents_total': ns,
                'asserted_dead': sum(1 for k in asserted[pid] if k in cdead),
                'n_asserted': len(asserted[pid]),
                'strict': not cdead and not sdead,
                'majority': not cdead and (ns == 0 or (ns - len(sdead)) * 2 > ns),
                'claims': not cdead,
                'asserted': not any(k in cdead for k in asserted[pid]),
            }
        per_entry[pid] = row
    # sanity: nothing dead at base
    base_bad = [p for p, r in per_entry.items()
                if r['20260401']['claims_dead'] or r['20260401']['sents_dead']]
    assert not base_bad, f'base-snapshot deaths: {base_bad}'

    hits = [p for p in (x['pid'] for x in usable) if lab[p]['hit']]
    assert len(hits) == 147
    intact_hits = [p for p in hits if not lab[p]['dep_stale']]
    stale_all = [p for p in lab if lab[p]['dep_stale']]  # all 50, incl. misses

    # --- enrichment-free conditional rates per snapshot ---
    onset = [round(sum(1 for p in stale_all
                       if per_entry[p][t]['claims_dead'] > 0)
                   / len(stale_all), 3) for t in SNAPS]
    intact_decline = {v: [round(100 * sum(1 for p in intact_hits
                                          if not per_entry[p][t][v])
                                / len(intact_hits), 2) for t in SNAPS]
                      for v in ('strict', 'majority')}
    P_STALE_4M = 52 / 6580
    pop_age = {}
    for v in ('strict', 'majority', 'claims', 'asserted'):
        row = []
        for i, t in enumerate(SNAPS):
            p_stale_t = P_STALE_4M * onset[i]
            r_intact = (sum(1 for p in intact_hits if not per_entry[p][t][v])
                        / len(intact_hits)) if v in ('strict', 'majority') else 0.0
            r_stale = 1.0 if v != 'asserted' else (
                sum(1 for p in stale_all if not per_entry[p][t]['asserted'])
                / len(stale_all))
            row.append(round(100 * (p_stale_t * r_stale
                                    + (1 - p_stale_t) * r_intact), 2))
        pop_age[v] = row

    # --- B: age curve (decline rates among the 147 hits) ---
    curve = {'ages': AGES, 'snapshots': SNAPS}
    for v in ('strict', 'majority', 'claims', 'asserted'):
        curve[v] = [round(100 * sum(1 for p in hits
                                    if not per_entry[p][t][v]) / len(hits), 2)
                    for t in SNAPS]
    curve['dep_stale_share'] = [round(100 * sum(1 for p in hits
        if per_entry[p][t]['claims_dead'] > 0) / len(hits), 2) for t in SNAPS]
    curve['staleness_onset'] = onset
    curve['intact_decline'] = intact_decline
    curve['population_decline_rate'] = pop_age

    # --- C: asserted-claims gate at the 4-month endpoint ---
    T = '20260801'
    def served(p, variant):
        return lab[p]['hit'] and per_entry[p][T][variant]
    ep = {}
    for name, v in [('strict', 'strict'), ('majority', 'majority'),
                    ('claims', 'claims'), ('asserted', 'asserted')]:
        sv = [p for p in lab if served(p, v)]
        dec_served = sum(1 for p in sv if lab[p]['lab'] in ('wrong', 'partial'))
        corr = [p for p in lab if lab[p]['hit'] and lab[p]['lab'] == 'correct']
        fd = sum(1 for p in corr if not served(p, v))
        ep[name] = {'served': len(sv), 'decayed_served': dec_served,
                    'correct_hits_declined': fd, 'n_correct': len(corr),
                    'indet_declined': sum(1 for p in lab
                        if lab[p]['hit'] and lab[p]['lab'] == 'indet'
                        and not served(p, v))}
    # dependency-level reading of the asserted variant for Table 5
    stale_hits = [p for p in lab if lab[p]['hit'] and lab[p]['dep_stale']]
    intact_hits = [p for p in lab if lab[p]['hit'] and not lab[p]['dep_stale']]
    dep = {'asserted_stale_served': sum(1 for p in stale_hits
                                        if per_entry[p][T]['asserted']),
           'asserted_intact_declined': sum(1 for p in intact_hits
                                           if not per_entry[p][T]['asserted']),
           'n_stale_hits': len(stale_hits), 'n_intact_hits': len(intact_hits)}
    # per-decayed-entry interception under asserted variant
    dec_detail = {p: {'asserted_dead': per_entry[p][T]['asserted_dead'],
                      'n_asserted': per_entry[p][T]['n_asserted'],
                      'intercepted': not per_entry[p][T]['asserted']}
                  for p in lab if lab[p]['lab'] in ('wrong', 'partial')}

    # --- A: population reweighting at the 4-month endpoint ---
    P_STALE = 52 / 6580
    def decline_rate_intact(v):
        return sum(1 for p in intact_hits
                   if not per_entry[p][T][v]) / len(intact_hits)
    def decline_rate_stale(v):
        return sum(1 for p in stale_hits
                   if not per_entry[p][T][v]) / len(stale_hits)
    reweight = {}
    for v in ('strict', 'majority', 'claims', 'asserted'):
        reweight[v] = {
            'p_decline_given_intact': round(decline_rate_intact(v), 4),
            'p_decline_given_stale': round(decline_rate_stale(v), 4),
            'population_decline_rate': round(
                P_STALE * decline_rate_stale(v)
                + (1 - P_STALE) * decline_rate_intact(v), 4),
        }
    reweight['natural_stale_prevalence'] = round(P_STALE, 4)
    reweight['answer_decay_prevalence_estimate'] = round(
        P_STALE * 9 / 50, 5)  # decayed answers per request over 4 months

    out = {'age_curve': curve, 'endpoint': ep, 'asserted_dep': dep,
           'decayed_detail': dec_detail, 'reweight': reweight,
           'asserted_counts': {str(p): len(asserted[p]) for p in asserted}}
    json.dump(out, open(f'{DATA}/age_curve.json', 'w'), indent=1)

    # --- figure: population-reweighted decline rate vs cache age ---
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(3.1, 2.3))
    ax = fig.add_axes([0.16, 0.17, 0.78, 0.74])
    ax.plot(AGES, pop_age['strict'], 'o-', ms=3, lw=1.2, color='#B23A2E',
            label='strict (claims + all sentences)')
    ax.plot(AGES, pop_age['majority'], 's-', ms=3, lw=1.2, color='#D3822A',
            label='claims + majority sentences')
    ax.plot(AGES, pop_age['claims'], '^-', ms=3, lw=1.2, color='#2E7D32',
            label='claims only')
    ax.plot(AGES, pop_age['asserted'], 'd-', ms=3, lw=1.2, color='#1F5FA8',
            label='asserted claims only')
    ax.set_xlabel('cache age (months)', fontsize=8)
    ax.set_ylabel('est. requests declined (%)', fontsize=8)
    ax.set_xticks(AGES)
    ax.set_xlim(-0.15, 4.3)
    ax.set_ylim(0, 17)
    ax.tick_params(labelsize=7)
    ax.grid(alpha=.3)
    ax.legend(fontsize=5.6, loc='upper left', framealpha=0.95,
              borderpad=0.3, labelspacing=0.25, handlelength=1.4,
              handletextpad=0.5)
    fig.savefig(f'{DATA}/fig_cache_age.pdf')
    fig.savefig(f'{DATA}/fig_cache_age.png', dpi=200)
    if os.path.isdir(FIGDIR):
        import shutil
        shutil.copy(f'{DATA}/fig_cache_age.pdf',
                    os.path.join(FIGDIR, 'fig_cache_age.pdf'))
    plt.close(fig)

    # --- console summary ---
    print('asserted claims per entry: mean %.2f of mean %.2f designated'
          % (np.mean([len(v) for v in asserted.values()]),
             np.mean([len(x['certificate']['claims']) for x in usable])))
    print('\nage curve (decline % among 147 hits):')
    for v in ('strict', 'majority', 'claims', 'asserted', 'dep_stale_share'):
        print(f'  {v:16s}', curve[v])
    print('staleness onset (frac of 50 eventually-stale with dead claim):',
          onset)
    print('intact-conditional decline %:', intact_decline)
    print('population decline-rate estimate vs age:', pop_age)
    print('\nendpoint answer-level (asserted variant):',
          json.dumps(ep['asserted'], indent=1))
    print('dependency-level asserted:', dep)
    print('decayed interception under asserted:',
          {p: d['intercepted'] for p, d in dec_detail.items()})
    print('\npopulation reweighting:', json.dumps(reweight, indent=1))

    json.dump({'asserted': {str(p): asserted[p] for p in asserted}},
              open(f'{DATA}/asserted_claims.json', 'w'), indent=1)


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""Step 13: deterministic full-universe measurement + conditional reweighting.

Reviewer item 11.  Two upgrades over the s10 estimator:

1. DETERMINISTIC MEASUREMENT over the full universe of templated
   (entity, property) pairs (the same universe s1 sampled the panel
   from): pair-level staleness is recomputed against EVERY rebuilt
   snapshot (2026-04-01/05-01/07-01/08-01), giving exact per-age
   prevalence and per-property decay rates d_z instead of the panel
   onset scaling (52/6580 x panel onset).

2. CONDITIONAL REWEIGHTING.  The population decline rate of a gate
   variant v at age t is estimated as
       sum_z w_z [ d_z(t) * h_S * r_S(v,t) + (1 - d_z(t)) * h_I * r_I(v,t) ]
   with w_z = n_z/N (property share of the universe), d_z(t) the
   deterministic per-property stale share, h_S = 43/50 and
   h_I = 104/104 the observed hit rates conditional on staleness
   (declines only occur on hits), and r_S/r_I the panel
   hit-conditional decline rates (pooled across properties; a
   per-property sensitivity range is reported).

Outputs: data/population_reweight.json, updated fig_cache_age.pdf/.png
"""
import json, os, re, hashlib
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.normpath(os.path.join(HERE, '..', 'data'))
REB = os.path.normpath(os.path.join(HERE, '..', '..', '12_real_dynamics_audit',
                                   'data', 'rebuilt_2026'))
FIGDIR = os.path.normpath(os.path.join(HERE, '..', '..', '..', '..', 'fig'))

SNAPS = ['20260401', '20260501', '20260701', '20260801']
AGES = [0, 1, 3, 4]

TEMPLATES = ['P106', 'P136', 'P131', 'P166', 'P19', 'P69', 'P175', 'P361',
             'P527', 'P39', 'P54', 'P102', 'P123', 'P264', 'P27', 'P17',
             'P641', 'P495', 'P159', 'P20', 'P57', 'P58', 'P86', 'P50',
             'P35', 'P6', 'P1082', 'P2046', 'P571', 'P577', 'P570']


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
    na = norm(answer)
    nv = norm(vtext)
    if nv and nv in na:
        return True
    m = re.match(r'(\d{4})', str(vtext))
    if m and m.group(1) in str(answer):
        return True
    return False


def main():
    # ---------- 1. deterministic full-universe measurement ----------
    snaps_ent = {t: json.load(open(f'{REB}/entities_v2_{t}.json'))
                 for t in SNAPS}
    e0, eT = snaps_ent['20260401'], snaps_ent['20260801']
    universe = []  # (qid, prop, base_keys)
    n_missing_intermediate = 0
    for qid, ent in e0.items():
        if qid not in eT or not ent.get('label'):
            continue
        for p in TEMPLATES:
            k0 = truthy_keys(ent, p)
            if k0:
                universe.append((qid, p, k0))
    N = len(universe)

    det = {t: 0 for t in SNAPS}             # stale pairs at each snapshot
    per_prop = defaultdict(lambda: {'n': 0, **{t: 0 for t in SNAPS}})
    for qid, p, k0 in universe:
        per_prop[p]['n'] += 1
        for t in SNAPS:
            ent = snaps_ent[t].get(qid)
            if ent is None:
                if t not in ('20260401', '20260801'):
                    n_missing_intermediate += 1
                k1 = set()          # entity absent => all values removed
            else:
                k1 = truthy_keys(ent, p)
            if k0 - k1:
                det[t] += 1
                per_prop[p][t] += 1
    assert det['20260401'] == 0
    print(f'universe pairs: {N} (s1 reported 6580)')
    print(f'stale at target: {det["20260801"]} (s1 reported 52)')
    print('deterministic prevalence per age:',
          {t: round(det[t] / N, 5) for t in SNAPS})
    print('deterministic onset share of eventually-stale:',
          [round(det[t] / det['20260801'], 3) for t in SNAPS])
    print('entities missing at an intermediate snapshot:',
          n_missing_intermediate)
    het = {p: round(d['20260801'] / d['n'], 4)
           for p, d in per_prop.items()}
    print('\nper-property 4-month decay rate d_z (n_z pairs):')
    for p, r in sorted(het.items(), key=lambda kv: -kv[1]):
        print(f'  {p:6s} n={per_prop[p]["n"]:4d}  d_z={r:.4f}')

    # ---------- 2. panel hit-conditional gate outcomes (as in s10) ----
    panel = json.load(open(f'{DATA}/evidence.json'))
    gen = json.load(open(f'{DATA}/generation.json'))
    lab = {int(k): v for k, v in json.load(open(
        os.path.join(DATA, '..', 'annotation', 'answer_labels.json'))).items()}
    usable = [x for x in panel if gen[str(x['pid'])]['answer_ok']
              and gen[str(x['pid'])]['paraphrase']]
    assert len(usable) == 154

    asserted = {}
    for x in usable:
        pid = x['pid']
        ans = gen[str(pid)]['answer']
        keys = [val_key(c['value'])
                for c, vt in zip(x['certificate']['claims'], x['values_text'])
                if answer_asserts(ans, vt)]
        asserted[pid] = keys

    snaps = {}
    for tag in SNAPS:
        ents = snaps_ent[tag]
        pages = json.load(open(f'{REB}/pages_{tag}.json'))
        page_sids = {t: {sid(s) for s in sentences(wikitext_to_plain(w))}
                     for t, w in pages.items()}
        snaps[tag] = (ents, page_sids)

    per_entry = {}
    for x in usable:
        pid = x['pid']
        row = {}
        for tag in SNAPS:
            ents, page_sids = snaps[tag]
            ent = ents.get(x['qid'])
            tk = truthy_keys(ent, x['prop']) if ent is not None else set()
            cdead = [val_key(c['value']) for c in x['certificate']['claims']
                     if val_key(c['value']) not in tk]
            tsids = page_sids.get(x.get('page_title'), set())
            sdead = [s['id'] for s in x['certificate']['sentences']
                     if s['id'] not in tsids]
            ns = len(x['certificate']['sentences'])
            row[tag] = {
                'strict': not cdead and not sdead,
                'majority': not cdead and (ns == 0
                                           or (ns - len(sdead)) * 2 > ns),
                'claims': not cdead,
                'asserted': not any(k in cdead for k in asserted[pid]),
            }
        per_entry[pid] = row

    hits = [p for p in lab if lab[p]['hit']]
    stale_hits = [p for p in hits if lab[p]['dep_stale']]
    intact_hits = [p for p in hits if not lab[p]['dep_stale']]
    stale_all = [p for p in lab if lab[p]['dep_stale']]
    intact_all = [p for p in lab if not lab[p]['dep_stale']]
    H_S = len(stale_hits) / len(stale_all)      # 43/50
    H_I = len(intact_hits) / len(intact_all)    # 104/104
    print(f'\nhit rates: h_S={len(stale_hits)}/{len(stale_all)}={H_S:.3f}'
          f'  h_I={len(intact_hits)}/{len(intact_all)}={H_I:.3f}')

    VARIANTS = ('strict', 'majority', 'claims', 'asserted')
    r_S = {v: [sum(1 for p in stale_hits if not per_entry[p][t][v])
               / len(stale_hits) for t in SNAPS] for v in VARIANTS}
    r_I = {v: [sum(1 for p in intact_hits if not per_entry[p][t][v])
               / len(intact_hits) for t in SNAPS] for v in VARIANTS}

    # ---------- 3. conditional reweighting ----------
    w = {p: d['n'] / N for p, d in per_prop.items()}
    d_z = {p: {t: d[t] / d['n'] for t in SNAPS}
           for p, d in per_prop.items()}

    cond, pooled_det, pooled_old = {}, {}, {}
    for v in VARIANTS:
        crow, prow, orow = [], [], []
        for i, t in enumerate(SNAPS):
            c = sum(w[z] * (d_z[z][t] * H_S * r_S[v][i]
                            + (1 - d_z[z][t]) * H_I * r_I[v][i])
                    for z in w)
            crow.append(round(100 * c, 2))
            p_det = det[t] / N
            prow.append(round(100 * (p_det * r_S[v][i]
                                     + (1 - p_det) * r_I[v][i]), 2))
        cond[v] = crow
        pooled_det[v] = prow
    # old estimator for reference: P_STALE_4M * panel onset
    onset_panel = [round(sum(1 for p in stale_all
                             if not per_entry[p][t]['claims'])
                         / len(stale_all), 3) for t in SNAPS]
    for v in VARIANTS:
        orow = []
        for i, t in enumerate(SNAPS):
            p_stale_t = (52 / 6580) * onset_panel[i]
            orow.append(round(100 * (p_stale_t * r_S[v][i]
                                     + (1 - p_stale_t) * r_I[v][i]), 2))
        pooled_old[v] = orow

    # per-property sensitivity of r_S (properties with >=3 stale hits)
    from collections import Counter
    cnt = Counter(next(x['prop'] for x in usable if x['pid'] == p)
                  for p in stale_hits)
    sens = {}
    T = '20260801'
    for prop, c in cnt.items():
        if c < 3:
            continue
        ps = [p for p in stale_hits
              if next(x['prop'] for x in usable if x['pid'] == p) == prop]
        sens[prop] = {v: round(sum(1 for p in ps
                                   if not per_entry[p][T][v]) / len(ps), 3)
                      for v in VARIANTS}
    print('\nper-property r_S sensitivity at endpoint (props with >=3 stale hits):',
          json.dumps(sens, indent=1))

    print('\npopulation decline-rate estimate vs age (%):')
    print(f'{"variant":10s} {"estimator":22s} 0m     1m     3m     4m')
    for v in VARIANTS:
        print(f'{v:10s} {"conditional (new)":22s}', cond[v])
        print(f'{"":10s} {"pooled, det. prev.":22s}', pooled_det[v])
        print(f'{"":10s} {"pooled, panel onset":22s}', pooled_old[v])

    out = {
        'universe_pairs': N,
        'det_stale_per_snapshot': det,
        'det_prevalence': {t: det[t] / N for t in SNAPS},
        'det_onset_share': [det[t] / det['20260801'] for t in SNAPS],
        'per_property': {p: dict(d) for p, d in per_prop.items()},
        'hit_rates': {'h_S': H_S, 'h_I': H_I,
                      'n_stale': len(stale_all), 'n_intact': len(intact_all)},
        'r_S': r_S, 'r_I': r_I,
        'conditional': cond, 'pooled_deterministic': pooled_det,
        'pooled_panel_onset': pooled_old,
        'property_sensitivity_r_S': sens,
    }
    json.dump(out, open(f'{DATA}/population_reweight.json', 'w'), indent=1)

    # ---------- 4. regenerate the cache-age figure ----------
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(3.1, 2.3))
    ax = fig.add_axes([0.16, 0.17, 0.78, 0.74])
    ax.plot(AGES, cond['strict'], 'o-', ms=3, lw=1.2, color='#B23A2E',
            label='strict (claims + all sentences)')
    ax.plot(AGES, cond['majority'], 's-', ms=3, lw=1.2, color='#D3822A',
            label='claims + majority sentences')
    ax.plot(AGES, cond['claims'], '^-', ms=3, lw=1.2, color='#2E7D32',
            label='claims only')
    ax.plot(AGES, cond['asserted'], 'd-', ms=3, lw=1.2, color='#1F5FA8',
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
    print('\nwrote data/population_reweight.json and fig_cache_age.pdf/.png')


if __name__ == '__main__':
    main()

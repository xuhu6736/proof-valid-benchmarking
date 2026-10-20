#!/usr/bin/env python3
"""Step 23: reweighting with a CONSISTENT stratification (scheme A).

Reviewer correction: the s13 estimator multiplied d_z(t) (staleness AT
age t) by r_S(t) measured on the EVENTUALLY-stale (4-month) cohort, so
the intermediate-age estimates measured "stale at t AND stale at T"
rather than "stale at t".  Scheme A stratifies by current staleness:

  D(t) = d(t) * h_S * r_{S_t}(t)  +  (1 - d(t)) * h_I * r_{I_t}(t)

where d(t) is the deterministic universe stale share AT age t and
r_{S_t} is the decline probability given a hit stale AT age t:
  * strict / majority / claims: r_{S_t} = 1 exactly (a designated claim
    missing at t is precisely the decline condition), so the claims-side
    term d(t)*h_S is DETERMINISTIC over the rebuilt universe;
  * asserted: r_{S_t}(t) is the panel hit-conditional asserted-decline
    rate at age t (measured on the eventually-stale cohort; transient
    stale-at-t pairs are assumed to behave alike -- stated caveat);
  * r_{I_t}(t): sentence-absence declines among claim-intact hits are
    not measurable in the universe (no page texts) and remain
    panel-conditional scenario estimates.

Outputs: data/population_reweight_schemeA.json, regenerated
fig_cache_age.pdf/.png (copied to the paper fig/ directory).
"""
import json, os, shutil

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.normpath(os.path.join(HERE, '..', 'data'))
FIGDIR = os.path.normpath(os.path.join(HERE, '..', '..', '..', '..', 'fig'))

SNAPS = ['20260401', '20260501', '20260701', '20260801']
AGES = [0, 1, 3, 4]


def main():
    pr = json.load(open(f'{DATA}/population_reweight.json'))
    d = [pr['det_prevalence'][t] for t in SNAPS]
    h_S, h_I = pr['hit_rates']['h_S'], pr['hit_rates']['h_I']
    r_S, r_I = pr['r_S'], pr['r_I']

    variants = ('strict', 'majority', 'claims', 'asserted')
    schemeA, det_part, panel_part = {}, {}, {}
    for v in variants:
        a_row, d_row, p_row = [], [], []
        for i in range(4):
            r_stale = 1.0 if v != 'asserted' else r_S[v][i]
            claims_side = d[i] * h_S * r_stale
            sent_side = (1 - d[i]) * h_I * r_I[v][i]
            a_row.append(round(100 * (claims_side + sent_side), 2))
            d_row.append(round(100 * claims_side, 3))
            p_row.append(round(100 * sent_side, 3))
        schemeA[v] = a_row
        det_part[v] = d_row
        panel_part[v] = p_row

    print('scheme-A decline-rate estimates (%) by cache age', AGES)
    for v in variants:
        print(f'  {v:9s} total {schemeA[v]}   '
              f'(deterministic claims part {det_part[v]}, '
              f'panel sentence part {panel_part[v]})')
    print('\nfor reference, s13 mixed-cohort estimates:')
    for v in variants:
        print(f'  {v:9s} {pr["conditional"][v]}')

    out = {
        'scheme': 'A: stratified by staleness AT age t; claims-side '
                  'decline deterministic over the rebuilt universe; '
                  'asserted stale-side rate panel-measured (cohort '
                  'caveat); sentence-side rates panel-conditional',
        'ages_months': AGES,
        'det_prevalence': pr['det_prevalence'],
        'hit_rates': pr['hit_rates'],
        'schemeA': schemeA,
        'deterministic_claims_part': det_part,
        'panel_sentence_part': panel_part,
        's13_mixed_cohort': pr['conditional'],
    }
    json.dump(out, open(f'{DATA}/population_reweight_schemeA.json', 'w'),
              indent=1)

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(3.1, 2.3))
    ax = fig.add_axes([0.16, 0.17, 0.78, 0.74])
    ax.plot(AGES, schemeA['strict'], 'o-', ms=3, lw=1.2, color='#B23A2E',
            label='strict (claims + all sentences)')
    ax.plot(AGES, schemeA['majority'], 's-', ms=3, lw=1.2, color='#D3822A',
            label='claims + majority sentences')
    ax.plot(AGES, schemeA['claims'], '^-', ms=3, lw=1.2, color='#2E7D32',
            label='claims only')
    ax.plot(AGES, schemeA['asserted'], 'd-', ms=3, lw=1.2, color='#1F5FA8',
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
        shutil.copy(f'{DATA}/fig_cache_age.pdf',
                    os.path.join(FIGDIR, 'fig_cache_age.pdf'))
    plt.close(fig)
    print('\nwrote data/population_reweight_schemeA.json and regenerated '
          'fig_cache_age.pdf/.png')


if __name__ == '__main__':
    main()

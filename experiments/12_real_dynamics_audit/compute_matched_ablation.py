#!/usr/bin/env python3
"""Matched selector x matcher comparison on the common eligible cohort.

Re-aggregates the retained per-entity boolean outputs in
constructor_results.json (same 1,511-entity panel and same run as the
configuration-level summaries in constructor_results_v4.json; no re-run of
the snapshot extraction pipeline).  All four configurations are evaluated on
the COMMON cohort of entities whose value-mention selection is nonempty
(lead-5 selections are never empty on this panel, since every entity has at
least five extracted base sentences):

  selection:  lead-5            | value-mention (all matches, no cap)
  matcher:    exact (fuzzy=1.0) | fuzzy-0.9 Jaccard token-set survival

Every rate, interval, and paired comparison therefore uses identical
denominators across the four cells.  Paired discordances use the
continuity-corrected asymptotic McNemar statistic of constructor_lab4.py.
The comparison is in-sample on the audit panel; it excludes the cascade
fallback and the revision triggers.

Outputs: constructor_ablation_2x2.json
"""
import json
import math

SRC = 'constructor_results.json'
OUT = 'constructor_ablation_2x2.json'

CONFIGS = [
    ('lead-5', 'exact', 'lead5'),
    ('lead-5', 'fuzzy-0.9', 'lead5_fuzzy0.9'),
    ('value-mention', 'exact', 'valmention_all'),
    ('value-mention', 'fuzzy-0.9', 'val_fuzzy0.9'),
]


def wilson(k, n, z=1.96):
    if n == 0:
        raise ValueError('empty comparison stratum')
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    w = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return p, c - w, c + w


def paired(cohort, a, b):
    bb = sum(1 for r in cohort if bool(r[a]) and not bool(r[b]))
    cc = sum(1 for r in cohort if not bool(r[a]) and bool(r[b]))
    if bb + cc == 0:
        return {'b': bb, 'c': cc, 'p': 1.0}
    chi2 = max(abs(bb - cc) - 1, 0) ** 2 / (bb + cc)
    return {'b': bb, 'c': cc, 'p': round(math.erfc(math.sqrt(chi2 / 2)), 5)}


def main():
    rows = json.load(open(SRC))['rows']
    keys = [k for _, _, k in CONFIGS]
    eligible = [r for r in rows if all(r.get(k) is not None for k in keys)]
    n_pos = sum(1 for r in eligible if r['harm'])
    n_neg = len(eligible) - n_pos

    cells = {}
    for sel, mat, key in CONFIGS:
        entry = {'selection': sel, 'matching': mat}
        for label, tag in [(True, 'proxy_positive'), (False, 'proxy_negative')]:
            cohort = [r for r in eligible if bool(r['harm']) == label]
            k = sum(1 for r in cohort if bool(r[key]))
            p, lo, hi = wilson(k, len(cohort))
            entry[tag] = {'k': k, 'n': len(cohort), 'rate': round(p, 4),
                          'ci': [round(lo, 4), round(hi, 4)]}
        cells[f'{sel} / {mat}'] = entry

    comp = {}
    for name, (a, b) in {
        'matcher_under_lead5': ('lead5_fuzzy0.9', 'lead5'),
        'matcher_under_valmention': ('val_fuzzy0.9', 'valmention_all'),
        'selector_under_exact': ('valmention_all', 'lead5'),
        'selector_under_fuzzy0.9': ('val_fuzzy0.9', 'lead5_fuzzy0.9'),
    }.items():
        comp[name] = {'a': a, 'b': b}
        for label, tag in [(True, 'proxy_positive'), (False, 'proxy_negative')]:
            cohort = [r for r in eligible if bool(r['harm']) == label]
            comp[name][tag] = paired(cohort, a, b)

    out = {
        'source': ('retained per-entity rows of constructor_results.json; '
                   'same 1,511-entity panel and rule version as '
                   'constructor_results_v4.json'),
        'common_cohort': {
            'definition': ('entities whose value-mention selection is '
                           'nonempty; lead-5 selections are never empty on '
                           'this panel'),
            'n': len(eligible), 'n_pos': n_pos, 'n_neg': n_neg,
        },
        'cells': cells,
        'paired_comparisons': comp,
        'scope': ('in-sample matched re-aggregation on the common cohort; '
                  'covers the selector and matcher components only, not the '
                  'lead-3 fallback or the revision triggers of the full '
                  'cascade'),
    }
    json.dump(out, open(OUT, 'w'), indent=1)
    for name, e in cells.items():
        print(f"{name}: pos {e['proxy_positive']['k']}/{e['proxy_positive']['n']}"
              f" = {e['proxy_positive']['rate']} {e['proxy_positive']['ci']} | "
              f"neg {e['proxy_negative']['k']}/{e['proxy_negative']['n']}"
              f" = {e['proxy_negative']['rate']} {e['proxy_negative']['ci']}")
    for name, c in comp.items():
        print(f"{name}: pos {c['proxy_positive']} neg {c['proxy_negative']}")
    print('saved', OUT)


if __name__ == '__main__':
    main()

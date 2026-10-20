#!/usr/bin/env python3
"""
s26_sensitivity_ttl_vs_loop.py -- sensitivity defence for the closed-loop
frontier (Table 10): is TTL's 132 vs the asserted loop's 128 correct
deliveries a real loss?

Two analyses, all local data, no model calls:

  A. Paired entity-cluster bootstrap (B=10000) of the endpoint
     correct-delivery difference TTL minus loop, per loop policy,
     under the published accounting of Table 10.

  B. rho-parametric crossover: give every template-composed regeneration
     (TTL has 89, the loops have 27/47/74) the same real-world accuracy
     rho instead of free proxy-correctness.  correct_TTL(rho) = 43 + 89*rho
     (43 observed real-regen correct), correct_loop(rho) = human/real
     correct + proxy_count*rho.  Report the crossover rho*, the observed
     panel regeneration accuracy 43/51 with its Wilson 95% interval, and
     the evaluation at that rate.

Outputs: data/sensitivity_ttl_vs_loop.json + console summary.
"""

import json, os, io, sys, random, importlib.util, contextlib
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.normpath(os.path.join(HERE, '..', 'data'))

# ---- load s25 as a module (re-executes the prototype; stdout silenced) -----
spec = importlib.util.spec_from_file_location(
    's25', os.path.join(HERE, 's25_closed_loop_system.py'))
s25 = importlib.util.module_from_spec(spec)
with contextlib.redirect_stdout(io.StringIO()):
    spec.loader.exec_module(s25)

by_pid, lab, regen_lab, DRIFT = s25.by_pid, s25.lab, s25.regen_lab, s25.DRIFT
T = s25.T
BASE = s25.SNAPS[0]
results = s25.results

A4MAP = {'正确': 1, '错误': 0, '不完整': 0, '无法判断': 0}

def correct_indicator(policy, pid):
    """published (generous) accounting of Table 10: template compositions
    count as correct; human labels used wherever they exist."""
    how, origin = results[policy][-1]['_detail'][pid]
    if how == 'abstain':
        return 0
    if how == 'regenerated':
        if origin == T and pid in regen_lab:
            return 1 if regen_lab[pid]['R1_adjudicated'] == '正确' else 0
        return 1                                    # template proxy
    if pid in DRIFT:
        return 0                                    # incoming-query drift
    if origin == BASE:
        return A4MAP.get(lab[pid].get('A4', '无法判断'), 0)
    return 1                                        # repaired proxy

def proxy_count(policy, pid):
    """1 if the delivery is a template-composed regeneration (proxy)."""
    how, origin = results[policy][-1]['_detail'][pid]
    if how == 'regenerated' and not (origin == T and pid in regen_lab):
        return 1
    if how != 'abstain' and how != 'regenerated' and origin != BASE \
            and pid not in DRIFT:
        return 1                                    # repaired proxy
    return 0

POLICIES = ['TTL-1mo', 'page-change-loop', 'pvsb-strict-loop',
            'pvsb-asserted-loop']
pids = sorted(by_pid)
ent_of = {pid: by_pid[pid]['qid'] for pid in pids}
clusters = defaultdict(list)
for pid in pids:
    clusters[ent_of[pid]].append(pid)
ents = sorted(clusters)
print(f'requests {len(pids)}, entity clusters {len(ents)}')

correct = {pol: {pid: correct_indicator(pol, pid) for pid in pids}
           for pol in POLICIES}
proxy = {pol: {pid: proxy_count(pol, pid) for pid in pids} for pol in POLICIES}
obs = {pol: sum(correct[pol].values()) for pol in POLICIES}
print('published endpoint correct deliveries:', obs)

# ---------------- A. paired entity-cluster bootstrap -----------------------
rng = random.Random(20261017)
B = 10000
boot = {pol: [] for pol in POLICIES if pol != 'TTL-1mo'}
for _ in range(B):
    draw = [ents[rng.randrange(len(ents))] for _ in ents]
    n = 0
    d = {pol: 0 for pol in boot}
    for e in draw:
        for pid in clusters[e]:
            n += 1
            for pol in boot:
                d[pol] += correct['TTL-1mo'][pid] - correct[pol][pid]
    for pol in boot:
        boot[pol].append(d[pol] * 154.0 / n)        # rescale to 154 requests

boot_ci = {}
for pol, xs in boot.items():
    xs.sort()
    boot_ci[pol] = {
        'delta_observed': obs['TTL-1mo'] - obs[pol],
        'ci95': [round(xs[int(0.025 * B)], 2), round(xs[int(0.975 * B)], 2)],
        'p_ttl_better': round(sum(1 for x in xs if x > 0) / B, 4)}
    print(f"A bootstrap TTL - {pol}: observed {boot_ci[pol]['delta_observed']}, "
          f"95% CI {boot_ci[pol]['ci95']}, P(TTL better) "
          f"{boot_ci[pol]['p_ttl_better']}")

# ---------------- B. rho-parametric crossover ------------------------------
real_correct_ttl = 43          # observed real-regen correct inside TTL's 140
n_proxy = {pol: sum(proxy[pol].values()) for pol in POLICIES}
base_correct = {pol: obs[pol] - n_proxy[pol] for pol in POLICIES}
print('proxy (template) counts:', n_proxy)
print('non-proxy correct:', base_correct)

rho_obs, n_regen = 43 / 51, 51
z = 1.959964
p = rho_obs
den = 1 + z * z / n_regen
cen = (p + z * z / (2 * n_regen)) / den
mar = z * ((p * (1 - p) / n_regen + z * z / (4 * n_regen ** 2)) ** 0.5) / den
wilson = [round(cen - mar, 3), round(cen + mar, 3)]

crossover = {}
for pol in POLICIES:
    if pol == 'TTL-1mo':
        continue
    # correct_TTL(rho) = base_correct_TTL + n_proxy_TTL*rho vs loop likewise
    a1, b1 = base_correct['TTL-1mo'], n_proxy['TTL-1mo']
    a2, b2 = base_correct[pol], n_proxy[pol]
    rho_star = (a2 - a1) / (b1 - b2) if b1 != b2 else None
    crossover[pol] = {
        'ttl_rho': f'{a1} + {b1}*rho', 'loop_rho': f'{a2} + {b2}*rho',
        'rho_star': round(rho_star, 4) if rho_star is not None else None,
        'ttl_at_rho_obs': round(a1 + b1 * rho_obs, 1),
        'loop_at_rho_obs': round(a2 + b2 * rho_obs, 1)}
    print(f"B crossover TTL vs {pol}: rho* = {crossover[pol]['rho_star']}, "
          f"at rho={rho_obs:.3f}: TTL {crossover[pol]['ttl_at_rho_obs']} vs "
          f"loop {crossover[pol]['loop_at_rho_obs']}")
print(f"observed regeneration accuracy {rho_obs:.3f} (43/51), "
      f"Wilson 95% {wilson}")

marginal = {}
for pol in POLICIES:
    if pol == 'TTL-1mo':
        continue
    extra_regen = 422 - (29 if 'asserted' in pol else
                         53 if 'strict' in pol else 174)
    gap = obs['TTL-1mo'] - obs[pol]
    marginal[pol] = round(extra_regen / gap, 1) if gap > 0 else None
print('marginal regenerations per extra correct delivery:', marginal)

out = {
    'accounting': 'published generous accounting of Table 10 (template '
                  'compositions count as correct)',
    'observed_correct': obs,
    'proxy_counts': n_proxy,
    'non_proxy_correct': base_correct,
    'A_paired_cluster_bootstrap': {
        'B': B, 'clusters': len(ents), 'seed': 20261017, 'results': boot_ci},
    'B_rho_crossover': {
        'rho_observed': round(rho_obs, 4),
        'rho_wilson95': wilson,
        'per_policy': crossover,
        'note': 'rho is the accuracy of regeneration from current evidence; '
                'rho* is the blind-refresh break-even accuracy'},
    'marginal_regen_per_correct': marginal}
json.dump(out, open(f'{DATA}/sensitivity_ttl_vs_loop.json', 'w'),
          ensure_ascii=False, indent=1)
print('\nsaved data/sensitivity_ttl_vs_loop.json')

#!/usr/bin/env python3
"""s45_gpt_tableb.py -- human-labelled GPT Table B.

Rebuilds the endpoint answer-quality table for the second-generator
(gpt-5-mini) archive using ONLY human answer-quality labels:
  * base-cached deliveries: unchanged from Table B (incoming-query drift
    override, else the A4 label) -- these answers are generator-independent;
  * real-regen / template deliveries: the GPT label store of s44
    (126 inherited-identical + 99 freshly adjudicated labels);
  * accept-fail / no-current-evidence regenerations: abstention, exactly
    as in the kimi Table B.

Also re-runs the paired entity-cluster bootstrap (B=10000, seed 20261017,
same rescaler as s34) on the GPT correct indicators, TTL-1mo minus each
looped policy, and a kimi-minus-GPT comparison on the asserted loop.

Outputs:
  data/gpt_tableb_endpoint_quality.json
"""
import json, os, random, io, contextlib, importlib.util
from collections import Counter, defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.normpath(os.path.join(HERE, '..', 'data'))

spec = importlib.util.spec_from_file_location(
    's25', os.path.join(HERE, 's25_closed_loop_system.py'))
s25 = importlib.util.module_from_spec(spec)
with contextlib.redirect_stdout(io.StringIO()):
    spec.loader.exec_module(s25)

emap = json.load(open(f'{DATA}/tableb_endpoint_map.json'))
gtab = json.load(open(f'{DATA}/tableb_regeneration_gpt.json'))
g9 = json.load(open(f'{DATA}/regeneration_gpt.json'))
glabels = json.load(open(f'{DATA}/gpt_labels.json'))
lab, DRIFT = s25.lab, s25.DRIFT
by_pid = s25.by_pid
ARMS = ['similarity-static', 'TTL-1mo', 'page-change-loop',
        'pvsb-strict-loop', 'pvsb-asserted-loop']
CAT = {'正确': 'correct', '错误': 'wrong', '不完整': 'incomplete',
       '无法判断': 'indet'}


def gpt_label(arm, pid):
    """(category, correct_indicator) for one endpoint delivery, GPT arm."""
    d = emap[arm][str(pid)]
    src = d['source']
    if src == 'abstain':
        return 'abstain', 0
    if src == 'real-regen':
        r = g9.get(str(pid))
        if r is None or not r['regen_ok']:
            return 'abstain', 0
        c = CAT.get(glabels.get(f'{pid}|20260801', {})
                    .get('R1_adjudicated', '无法判断'), 'indet')
        return c, int(c == 'correct')
    if src == 'template':
        key = f"{pid}|{d['origin']}"
        r = gtab.get(key)
        if r is None or not r['regen_ok']:
            return 'abstain', 0
        c = CAT.get(glabels.get(key, {}).get('R1_adjudicated', '无法判断'),
                    'indet')
        return c, int(c == 'correct')
    # base-cached: generator-independent, same as Table B
    if pid in DRIFT:
        return 'wrong', 0
    c = CAT.get(lab[pid].get('A4', '无法判断'), 'indet')
    return c, int(c == 'correct')


pids = sorted(by_pid)
tableB, correct_ind = {}, {}
for arm in ARMS:
    cnt = Counter()
    ci = {}
    for pid in pids:
        c, ok = gpt_label(arm, pid)
        cnt[c] += 1
        ci[pid] = ok
    assert sum(cnt.values()) == 154, (arm, cnt)
    tableB[arm] = dict(cnt)
    correct_ind[arm] = ci
    print(f"{arm:22s} {dict(cnt)}")

# ------------------------------------------------ paired cluster bootstrap --
ent_of = {pid: by_pid[pid]['qid'] for pid in pids}
clusters = defaultdict(list)
for pid in pids:
    clusters[ent_of[pid]].append(pid)
ents = sorted(clusters)
rng = random.Random(20261017)
B = 10000
loops = [a for a in ARMS if a != 'TTL-1mo']
boot = {a: [] for a in loops}
# kimi asserted-loop indicators for the cross-generator comparison
ktableB = json.load(open(f'{DATA}/tableb_endpoint_quality.json'))
kadj = json.load(open(f'{DATA}/tableb_labels.json'))
kregen_lab = s25.regen_lab
ktbregen = json.load(open(f'{DATA}/tableb_regeneration.json'))
k9 = json.load(open(f'{DATA}/regeneration.json'))


def kimi_label(arm, pid):
    d = emap[arm][str(pid)]
    src = d['source']
    if src == 'abstain':
        return 0
    if src == 'real-regen':
        rl = kregen_lab.get(pid)
        if rl is None:
            return 0
        return int(rl['R1_adjudicated'] == '正确')
    if src == 'template':
        key = f"{pid}|{d['origin']}"
        r = ktbregen.get(key)
        if r is None or not r['regen_ok']:
            return 0
        return int(kadj.get(key, {}).get('R1_adjudicated') == '正确')
    if pid in DRIFT:
        return 0
    return int(lab[pid].get('A4') == '正确')


kci = {a: {pid: kimi_label(a, pid) for pid in pids} for a in ARMS}
cross = []
for _ in range(B):
    draw = [ents[rng.randrange(len(ents))] for _ in ents]
    dd = {a: 0 for a in loops}
    cr = 0
    n = 0
    for e in draw:
        for pid in clusters[e]:
            n += 1
            for a in loops:
                dd[a] += correct_ind['TTL-1mo'][pid] - correct_ind[a][pid]
            cr += kci['pvsb-asserted-loop'][pid] - \
                correct_ind['pvsb-asserted-loop'][pid]
    for a in loops:
        boot[a].append(dd[a] * 154.0 / n)      # rescale as in s26/s34
    cross.append(cr * 154.0 / n)


def ci95(v):
    v = sorted(v)
    return [v[int(0.025 * B)], v[int(0.975 * B) - 1]]


boots = {a: {'mean': sum(v) / B, 'ci95': ci95(v)} for a, v in boot.items()}
cross_ci = {'mean': sum(cross) / B, 'ci95': ci95(cross)}
print('\nGPT Table B bootstrap (TTL minus loop, correct deliveries):')
for a, r in boots.items():
    print(f"  {a:22s} mean {r['mean']:+.2f}  95% {r['ci95']}")
print(f"kimi minus GPT asserted loop: mean {cross_ci['mean']:+.2f}  "
      f"95% {cross_ci['ci95']}")

out = {'generator': 'gpt-5-mini',
       'labels': '225 human labels (126 inherited-identical, 99 fresh; '
                 'see annotation/gpt_agreement.json)',
       'tableB_per_arm': tableB,
       'bootstrap_TTL_minus_loop': boots,
       'bootstrap_kimi_minus_gpt_asserted': cross_ci}
json.dump(out, open(f'{DATA}/gpt_tableb_endpoint_quality.json', 'w'),
          ensure_ascii=False, indent=1)
print('saved data/gpt_tableb_endpoint_quality.json')

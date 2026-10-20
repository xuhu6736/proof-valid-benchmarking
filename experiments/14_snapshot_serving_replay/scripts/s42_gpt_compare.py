#!/usr/bin/env python3
"""s42_gpt_compare.py -- second-generator robustness analysis.

Rebuilds every Table-B policy arm twice from the SAME endpoint delivery map
(data/tableb_endpoint_map.json): once with the archived kimi-k2-turbo-preview
regenerations and once with the GPT (gpt-5-mini) regenerations produced by
s41 under the identical prompt protocol.  Answers are scored by the
value-mention proxy against endpoint-tick (20260801) truthy values --
the same proxy the paper uses for acceptance; no human labels are reused
or invented for the GPT variant.

Outputs:
  data/gpt_generator_compare.json   per-arm proxy counts, cross-tabs,
                                    per-key disagreement list
"""
import json, os, re, sys, importlib.util, contextlib
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.normpath(os.path.join(HERE, '..', 'data'))
T = '20260801'

sys.argv = ['s32_tableb_regenerate.py', '--dry']
spec = importlib.util.spec_from_file_location(
    's32', os.path.join(HERE, 's32_tableb_regenerate.py'))
s32 = importlib.util.module_from_spec(spec)
with open(os.devnull, 'w') as dn, contextlib.redirect_stdout(dn):
    try:
        spec.loader.exec_module(s32)
    except SystemExit:
        pass

s25 = s32.s25
norm = s32.norm

emap = json.load(open(f'{DATA}/tableb_endpoint_map.json'))
ktab = json.load(open(f'{DATA}/tableb_regeneration.json'))
gtab = json.load(open(f'{DATA}/tableb_regeneration_gpt.json'))
k9 = json.load(open(f'{DATA}/regeneration.json'))
g9 = json.load(open(f'{DATA}/regeneration_gpt.json'))

# ------------------------------------------------- endpoint gold values ----
def gold_values(pid):
    x = s25.by_pid[int(pid)]
    ent = s25.snaps_ent[T].get(str(x['qid']))
    out = []
    if ent is not None:
        for rank, snak, val in ent['claims'].get(x['prop'], []):
            if rank in ('normal', 'preferred') and snak == 'value' \
                    and val is not None:
                if isinstance(val, dict) and 'id' in val:
                    out.append(s32.resolve_label(val['id']))
                elif isinstance(val, dict) and 'amount' in val:
                    out.append(str(val['amount']).lstrip('+'))
                elif isinstance(val, dict) and 'time' in val:
                    out.append(val['time'][1:11])
                else:
                    out.append(str(val))
    return out

def mentions(answer, values):
    if answer is None:
        return False
    na = norm(answer)
    for vt in values:
        nv = norm(vt)
        if nv and nv in na:
            return True
        m = re.match(r'(\d{4})', str(vt))
        if m and m.group(1) in answer:
            return True
    return False

# --------------------------------------------------- arm reconstruction ----
def deliver(pid, rec, gen):
    """Return (source_class, answer) for generator gen in {'kimi','gpt'}."""
    src = rec['source']
    if src == 'abstain':
        return 'abstain', None
    if src == 'base-cached':
        return 'serve', rec['answer']                     # identical both arms
    if src == 'real-regen':
        r = (k9 if gen == 'kimi' else g9).get(str(pid))
        if r is not None and r.get('regen_ok'):
            return 'serve', r['regen_answer']
        return 'abstain', None          # accept-fail / no-evidence: abstain
    # template source: real regeneration at (pid, origin tick)
    key = f"{pid}|{rec['origin']}"
    r = (ktab if gen == 'kimi' else gtab).get(key)
    if r is not None and r.get('regen_ok'):
        return 'serve', r['regen_answer']
    return 'abstain', None              # no_current_evidence or accept-fail

arms = list(emap)
gold = {pid: gold_values(pid) for pid in emap[arms[0]]}

report = {}
for arm in arms:
    row = {}
    for gen in ('kimi', 'gpt'):
        n_serve = n_corr = n_abstain = 0
        for pid, rec in emap[arm].items():
            cls, ans = deliver(pid, rec, gen)
            if cls == 'abstain':
                n_abstain += 1
            else:
                n_serve += 1
                n_corr += mentions(ans, gold[pid])
        row[gen] = {'served': n_serve, 'proxy_correct': n_corr,
                    'abstain': n_abstain}
    report[arm] = row

# ------------------------------------------------- cross-generator diffs ---
diffs = []
for arm in arms:
    for pid, rec in emap[arm].items():
        ck, ak = deliver(pid, rec, 'kimi')
        cg, ag = deliver(pid, rec, 'gpt')
        vk, vg = mentions(ak, gold[pid]), mentions(ag, gold[pid])
        if ck != cg or vk != vg:
            diffs.append({'arm': arm, 'pid': pid, 'source': rec['source'],
                          'kimi': [ck, bool(vk)], 'gpt': [cg, bool(vg)],
                          'kimi_answer': ak, 'gpt_answer': ag,
                          'gold': gold[pid]})

# same-key acceptance cross-tab (merged store) and s9 store
def crosstab(ks, gs):
    c = Counter()
    for k in ks:
        a = ks[k].get('regen_ok'), gs[k]['regen_ok']
        c[a] += 1
    return {str(k): v for k, v in sorted(c.items())}

out = {'proxy': 'endpoint-tick truthy value mention (same rule as acceptance)',
       'arms': report,
       'acceptance_crosstab_tableb_store': crosstab(ktab, gtab),
       'acceptance_crosstab_s9_store': crosstab(k9, g9),
       'n_delivery_diffs': len(diffs), 'delivery_diffs': diffs}
json.dump(out, open(f'{DATA}/gpt_generator_compare.json', 'w'),
          ensure_ascii=False, indent=1)

print(f'{"arm":22s} {"gen":5s} {"served":>6s} {"proxy-correct":>13s} '
      f'{"abstain":>7s}')
for arm in arms:
    for gen in ('kimi', 'gpt'):
        r = report[arm][gen]
        print(f'{arm:22s} {gen:5s} {r["served"]:6d} '
              f'{r["proxy_correct"]:13d} {r["abstain"]:7d}')
print('\nacceptance cross-tab (tableb store):',
      out['acceptance_crosstab_tableb_store'])
print('acceptance cross-tab (s9 store):', out['acceptance_crosstab_s9_store'])
print('delivery diffs:', len(diffs))
print('saved data/gpt_generator_compare.json')

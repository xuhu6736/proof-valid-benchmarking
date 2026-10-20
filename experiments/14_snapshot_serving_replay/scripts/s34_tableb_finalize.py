#!/usr/bin/env python3
"""s34_tableb_finalize.py -- P3 second half: parse the two completed
Table-B annotation sheets, audit agreement, and rebuild the closed-loop
endpoint quality table (Table B) with a single generation standard.

Inputs (paper directory):
  PVSB_TableB再生标注表_174条（人工1）.docx
  PVSB_TableB再生标注表_174条（人工2）.docx

Label resolution per endpoint delivery (data/tableb_endpoint_map.json):
  base-cached  -> incoming-query drift override, else human A4 label
  real-regen   -> adjudicated R1 label of the s9 regeneration round
  template     -> adjudicated R1 label of THIS round's real regeneration
                  at the (pid, origin-tick) key; the one accept-fail key
                  is counted as an abstention (honest accounting)
  abstain      -> abstain

Adjudication: first annotator adjudicates, as in both earlier rounds.

Then re-runs the paired entity-cluster bootstrap (B=10000, seed 20261017,
same rescaler as s26) on the Table-B correct indicators, TTL-1mo minus
each looped policy.

Outputs:
  annotation/tableb_ann1_parsed.json, annotation/tableb_ann2_parsed.json
  annotation/tableb_agreement.json
  data/tableb_labels.json               key "pid|tick" -> label record
  data/tableb_endpoint_quality.json     per-arm Table B + bootstrap
"""
import zipfile, re, json, os, random, io, contextlib, importlib.util
from collections import Counter, defaultdict
from lxml import etree

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.normpath(os.path.join(HERE, '..', 'data'))
ANN = os.path.normpath(os.path.join(HERE, '..', 'annotation'))
PAPER = os.path.normpath(os.path.join(HERE, '..', '..', '..', '..'))

W = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"
OPT = ["正确", "错误", "不完整", "无法判断"]


def text_of(el):
    return "".join(t.text or "" for t in el.iter(W + "t"))


def parse_sheet(path):
    root = etree.fromstring(zipfile.ZipFile(path).read("word/document.xml"))
    out = {}
    for tbl in root.iter(W + "tbl"):
        txt = text_of(tbl)
        m = re.search(r"条目 Entry #(\d+) \| 证据期 (\d{4})-(\d{2})-(\d{2})", txt)
        if not m:
            continue
        key = f"{m.group(1)}|{m.group(2)}{m.group(3)}{m.group(4)}"
        ann = None
        for row in tbl.iter(W + "tr"):
            rt = text_of(row)
            if "R1" in rt and "正确性" in rt:
                ann = rt
        assert ann is not None, f"no annotation row for {key}"
        i = ann.find("R1 ")
        j = ann.find("备注", i)
        seg = ann[i: j if j != -1 else len(ann)]
        vi = seg.find("√")
        assert vi != -1, f"missing mark {key}"
        tail = seg[vi + 1: vi + 40].lstrip()
        hit = next((o for o in OPT if tail.startswith(o)), None)
        assert hit, f"unmatched option {key}: {tail[:30]!r}"
        mn = re.search(r"备注 notes:\s*(.*)$", ann)
        out[key] = {"R1": hit,
                    "notes": re.sub(r"_+$", "", mn.group(1).strip()
                                    if mn else "").strip()}
    return out


def kappa(p, q, items):
    n = len(items)
    po = sum(p[i] == q[i] for i in items) / n
    cp, cq = Counter(p[i] for i in items), Counter(q[i] for i in items)
    pe = sum((cp[k] / n) * (cq[k] / n) for k in set(cp) | set(cq))
    return po, (None if pe >= 1 else (po - pe) / (1 - pe))


# ------------------------------------------------------------------ parse ---
f1 = os.path.join(PAPER, 'PVSB_TableB再生标注表_174条（人工1）.docx')
f2 = os.path.join(PAPER, 'PVSB_TableB再生标注表_174条（人工2）.docx')
a1, a2 = parse_sheet(f1), parse_sheet(f2)
keys = sorted(set(a1) & set(a2), key=lambda k: (int(k.split('|')[0]),
                                                k.split('|')[1]))
assert len(keys) == 174, f'expected 174, got {len(keys)} / {len(a1)} / {len(a2)}'

json.dump({k: a1[k] for k in keys},
          open(os.path.join(ANN, 'tableb_ann1_parsed.json'), 'w'),
          ensure_ascii=False, indent=1)
json.dump({k: a2[k] for k in keys},
          open(os.path.join(ANN, 'tableb_ann2_parsed.json'), 'w'),
          ensure_ascii=False, indent=1)

po, kp = kappa({k: a1[k]['R1'] for k in keys}, {k: a2[k]['R1'] for k in keys},
               keys)
dis = [{'key': k, 'ann1': a1[k]['R1'], 'ann2': a2[k]['R1']}
       for k in keys if a1[k]['R1'] != a2[k]['R1']]
adj = {k: a1[k]['R1'] for k in keys}          # first annotator adjudicates

labels = {k: {'R1_ann1': a1[k]['R1'], 'R1_ann2': a2[k]['R1'],
              'R1_adjudicated': adj[k],
              'notes_ann1': a1[k]['notes'], 'notes_ann2': a2[k]['notes']}
          for k in keys}
json.dump(labels, open(f'{DATA}/tableb_labels.json', 'w'),
          ensure_ascii=False, indent=1)
agree_out = {'n': len(keys), 'agreement': round(po, 4),
             'kappa': None if kp is None else round(kp, 3),
             'n_disagreements': len(dis), 'disagreements': dis,
             'R1_ann1': dict(Counter(a1[k]['R1'] for k in keys)),
             'R1_ann2': dict(Counter(a2[k]['R1'] for k in keys)),
             'adjudicated_R1': dict(Counter(adj.values()))}
json.dump(agree_out, open(os.path.join(ANN, 'tableb_agreement.json'), 'w'),
          ensure_ascii=False, indent=1)
print(f"R1 agreement: {po*100:.1f}%  kappa {agree_out['kappa']}")
print('R1 ann1:', agree_out['R1_ann1'])
print('R1 ann2:', agree_out['R1_ann2'])
print('adjudicated:', agree_out['adjudicated_R1'])
print('disagreements:', len(dis))

# ---------------------------------------------------------- Table B build ---
spec = importlib.util.spec_from_file_location(
    's25', os.path.join(HERE, 's25_closed_loop_system.py'))
s25 = importlib.util.module_from_spec(spec)
with contextlib.redirect_stdout(io.StringIO()):
    spec.loader.exec_module(s25)

emap = json.load(open(f'{DATA}/tableb_endpoint_map.json'))
tbregen = json.load(open(f'{DATA}/tableb_regeneration.json'))
lab, regen_lab, DRIFT = s25.lab, s25.regen_lab, s25.DRIFT
by_pid = s25.by_pid
ARMS = ['similarity-static', 'TTL-1mo', 'page-change-loop',
        'pvsb-strict-loop', 'pvsb-asserted-loop']
CAT = {'正确': 'correct', '错误': 'wrong', '不完整': 'incomplete',
       '无法判断': 'indet'}

def tableb_label(arm, pid):
    """(category, correct_indicator) for one endpoint delivery."""
    d = emap[arm][str(pid)]
    src = d['source']
    if src == 'abstain':
        return 'abstain', 0
    if src == 'real-regen':
        rl = regen_lab.get(pid)
        if rl is None:
            return 'indet', 0
        c = CAT.get(rl['R1_adjudicated'], 'indet')
        return c, int(c == 'correct')
    if src == 'template':
        key = f"{pid}|{d['origin']}"
        r = tbregen.get(key)
        if r is None or not r['regen_ok']:
            return 'abstain', 0               # accept-fail -> honest abstain
        c = CAT.get(adj.get(key, '无法判断'), 'indet')
        return c, int(c == 'correct')
    # base-cached
    if pid in DRIFT:
        return 'wrong', 0
    c = CAT.get(lab[pid].get('A4', '无法判断'), 'indet')
    return c, int(c == 'correct')

tableB, correct_ind = {}, {}
pids = sorted(by_pid)
for arm in ARMS:
    cnt = Counter()
    ci = {}
    for pid in pids:
        c, ok = tableb_label(arm, pid)
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
for _ in range(B):
    draw = [ents[rng.randrange(len(ents))] for _ in ents]
    n = 0
    dd = {a: 0 for a in loops}
    for e in draw:
        for pid in clusters[e]:
            n += 1
            for a in loops:
                dd[a] += correct_ind['TTL-1mo'][pid] - correct_ind[a][pid]
    for a in loops:
        boot[a].append(dd[a] * 154.0 / n)
boot_ci = {}
for a, xs in boot.items():
    xs.sort()
    obs = (sum(correct_ind['TTL-1mo'].values())
           - sum(correct_ind[a].values()))
    boot_ci[a] = {'delta_observed': obs,
                  'ci95': [round(xs[int(0.025 * B)], 2),
                           round(xs[int(0.975 * B)], 2)],
                  'p_ttl_better': round(sum(1 for x in xs if x > 0) / B, 4)}
    print(f"bootstrap TTL - {a}: observed {obs}, 95% CI "
          f"{boot_ci[a]['ci95']}, P(TTL better) {boot_ci[a]['p_ttl_better']}")

# ------------------------------------------------------- regen accuracy -----
n_real_ok = sum(1 for k in keys if adj[k] == '正确')
tot_ok = 43 + n_real_ok                    # 43/51 correct in the s9 round
tot_n = 51 + len(keys)
z = 1.959964
p = tot_ok / tot_n
den = 1 + z * z / tot_n
cen = (p + z * z / (2 * tot_n)) / den
mar = z * ((p * (1 - p) / tot_n + z * z / (4 * tot_n ** 2)) ** 0.5) / den
wilson = [round(cen - mar, 3), round(cen + mar, 3)]
print(f'pooled regeneration accuracy {tot_ok}/{tot_n} = {p:.3f} '
      f'Wilson {wilson}')

out = {'tableB_per_arm': tableB,
       'labels': 'annotation standard identical across sources; first '
                 'annotator adjudicates disagreements',
       'agreement': agree_out,
       'accept_fail_keys': [k for k, v in tbregen.items() if not v['regen_ok']
                            and not v['no_current_evidence']],
       'paired_cluster_bootstrap': {'B': B, 'clusters': len(ents),
                                    'seed': 20261017, 'results': boot_ci},
       'pooled_regen_accuracy': {'correct': tot_ok, 'n': tot_n,
                                 'rate': round(p, 4), 'wilson95': wilson}}
json.dump(out, open(f'{DATA}/tableb_endpoint_quality.json', 'w'),
          ensure_ascii=False, indent=1)
print('\nsaved data/tableb_labels.json, annotation/tableb_agreement.json, '
      'data/tableb_endpoint_quality.json')

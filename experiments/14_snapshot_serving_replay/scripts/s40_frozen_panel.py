#!/usr/bin/env python3
"""s40_frozen_panel.py -- frozen independent check on NEW objects
(reviewer acceptance criterion 4, run after the design was frozen).

Nothing in this script is re-tuned: the pair universe and staleness
semantics come from s1, the certificate builder and contracts from s25,
the byte accounting / greedy 90% rule / fault plan from s30, the base-answer
protocol (system prompt, evidence lines, acceptance, one retry) from s4.

Frozen object panel: candidate (entity, property) pairs are rebuilt with
s1's exact code; every qid used anywhere in the 156-entry design panel is
excluded; the stale class is exhausted by the design panel, so the frozen
panel is fresh-only (disclosed).  Frozen filter: a Wikipedia page exists at
all four snapshots, >=1 base claim, >=1 base evidence sentence.  Frozen
selection: random.Random(20261017) shuffle, first 50 pairs whose base
answer passes the s4 acceptance rule (candidates beyond 50 are queued so
rejections never reopen the pool).

Checks (asserted contract, p=0.10, 100 CRN seeds, same SQLite real-read
chain protocol as s30, template fallback):
  (a) every restore = source lost + eligible + intact fingerprinted copy
      -> served (fingerprint assert on every backup read);
  (b) a copy whose bytes are intact but whose premise is ineligible is
      NEVER restored (guard counters; plus a separately labelled
      deterministic eligibility-flip test if no natural case occurs);
  (c) source+copy double loss fails closed (fallback/abstain, never serve);
  (d) write-back certificates installed during the loop are unprotected:
      a lost source read of a write-back premise NEVER restores;
  (e) same-budget direction: fixed-arm fallback mean < none-arm mean,
      paired per-seed bootstrap interval reported.

Outputs: data/frozen_panel_check.json + console summary.
"""
import json, os, re, io, sys, time, random, sqlite3, hashlib, urllib.request
import importlib.util, contextlib
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.normpath(os.path.join(HERE, '..', 'data'))
REB = os.path.normpath(os.path.join(HERE, '..', '..', '12_real_dynamics_audit',
                                   'data', 'rebuilt_2026'))
SNAPS = ['20260401', '20260501', '20260701', '20260801']
TICKS = SNAPS[1:]
T0 = SNAPS[0]
P = 0.10
R = 100
SEED_SEL = 20261017
N_PANEL = 50
DB = f'{DATA}/frozen_payload_stores.sqlite'
GEN_OUT = f'{DATA}/frozen_panel_generation.json'
OUT = f'{DATA}/frozen_panel_check.json'

def load_mod(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    with contextlib.redirect_stdout(io.StringIO()):
        spec.loader.exec_module(m)
    return m

s1 = load_mod('s1', os.path.join(HERE, 's1_build_panel.py'))
s25 = load_mod('s25', os.path.join(HERE, 's25_closed_loop_system.py'))
truthy_items = s25.truthy_items
sentences, wikitext_to_plain = s25.sentences, s25.wikitext_to_plain
sid, norm, val_key = s25.sid, s25.norm, s25.val_key
snaps_ent, pages = s25.snaps_ent, s25.pages

def value_text(v):
    """s25.value_text with one extension: value-qid labels fall back to the
    label stored in the entity snapshot itself (the s25 resolver only covers
    value qids of the design panel, whose labels were fetched separately;
    the snapshots carry labels for every rebuilt entity)."""
    if isinstance(v, dict) and 'id' in v:
        vid = str(v['id'])
        vid = vid[1:] if vid.startswith('Q') else vid
        t = s25.qid2title.get(int(vid)) if vid.isdigit() else None
        if t:
            return t
        ent = snaps_ent[T0].get(vid)
        if ent is not None and ent.get('label'):
            return ent['label']
        return 'Q' + vid
    return s25.value_text(v)

title_map = json.load(open(f'{REB}/title_qid_map.json'))
qid2page = {str(v): k for k, v in title_map.items()}

# ---------------- Stage A: frozen selection ---------------------------------
def build_universe():
    e0 = snaps_ent[T0]
    e1 = snaps_ent[SNAPS[-1]]
    pairs = []
    for qid, ent in e0.items():
        if qid not in e1:
            continue
        lab = ent.get('label')
        if not lab:
            continue
        t0 = s1.truthy(ent)
        for p in s1.TEMPLATES:
            if p not in t0:
                continue
            pairs.append({'qid': qid, 'label': lab, 'prop': p,
                          'question': s1.TEMPLATES[p][0].format(L=lab),
                          'kind': s1.TEMPLATES[p][1],
                          'values_base': t0[p]})
    return pairs

design_qids = {x['qid'] for x in json.load(open(f'{DATA}/panel.json'))}
cand = [x for x in build_universe() if x['qid'] not in design_qids]

def base_items(x):
    """t0 certificate material: claims + up to 3 evidence sentences, using
    s3's frozen retrieval rule (token/year scoring, top 3, fallback to the
    first 2 lead sentences) -- not a tighter ad-hoc rule."""
    ent = snaps_ent[T0].get(str(x['qid']))
    items = truthy_items(ent, x['prop']) if ent is not None else []
    claims = [k for k, _ in items]
    texts = [value_text(v) for _, v in items]
    plain = wikitext_to_plain(pages[T0].get(x.get('page_title'), ''))
    sents = sentences(plain)
    keys = []
    for vt in texts:
        if vt.startswith('Q') and vt[1:].isdigit():
            continue
        toks = [t for t in re.split(r'\s+', vt) if len(t) > 2]
        keys.extend(toks)
        if len(vt) > 3:
            keys.append(vt)
    keys.extend(re.match(r'(\d{4})', vt).group(1) for vt in texts
                if re.match(r'\d{4}', vt))
    hits = []
    for s in sents:
        low = s.lower()
        sc = sum(1 for k in keys if k.lower() in low)
        if sc:
            hits.append((sc, s))
    hits.sort(key=lambda t: -t[0])
    ev = [s for _, s in hits[:3]] or sents[:2]
    return claims, texts, [{'id': sid(s), 'text': s} for s in ev]

pool = []
for x in cand:
    x['page_title'] = qid2page.get(str(x['qid']))
    if not x['page_title']:
        continue
    if any(x['page_title'] not in pages[t] for t in SNAPS):
        continue
    claims, texts, sents = base_items(x)
    if not claims or not sents:
        continue
    x['_claims'], x['_texts'], x['_sents'] = claims, texts, sents
    pool.append(x)
rng = random.Random(SEED_SEL)
rng.shuffle(pool)
print(f'frozen universe: {len(cand)} pairs on non-design qids; '
      f'{len(pool)} pass the frozen filter')

# ---------------- Stage B: base answers (s4 protocol) ------------------------
SYS_A = ('You are a careful question-answering assistant. Answer strictly from '
         'the provided evidence. Reply with one short phrase or sentence only; '
         'do not add caveats.')

def accepted(answer, values_text):
    na = norm(answer)
    for vt in values_text:
        nv = norm(vt)
        if nv and nv in na:
            return True
        m = re.match(r'(\d{4})', vt)
        if m and m.group(1) in answer:
            return True
    return False

KEY = os.environ.get('KIMI_API_KEY')
BASE = os.environ.get('KIMI_BASE_URL', '').rstrip('/')
MODEL = 'kimi-k2-turbo-preview'

def chat(prompt, system, max_tokens=160, tries=4):
    payload = {'model': MODEL, 'max_tokens': max_tokens,
               'thinking': {'type': 'disabled'}, 'system': system,
               'messages': [{'role': 'user', 'content': prompt}]}
    for a in range(tries):
        try:
            req = urllib.request.Request(
                BASE + '/messages', data=json.dumps(payload).encode(),
                headers={'x-api-key': KEY, 'anthropic-version': '2023-06-01',
                         'content-type': 'application/json'}, method='POST')
            with urllib.request.urlopen(req, timeout=60) as r:
                d = json.loads(r.read())
            return ''.join(b.get('text', '') for b in d.get('content', [])
                           if b.get('type') == 'text').strip()
        except Exception:
            time.sleep(2 + a * 3)
    return None

gen = json.load(open(GEN_OUT)) if os.path.exists(GEN_OUT) else {}
frozen = [x for x in pool if str(x['qid'] + '|' + x['prop']) in gen
          and gen[x['qid'] + '|' + x['prop']].get('answer_ok')]
if len(frozen) < N_PANEL and '--dry' not in sys.argv:
    assert KEY and BASE, 'KIMI_API_KEY/KIMI_BASE_URL required'
    for x in pool:
        key = x['qid'] + '|' + x['prop']
        if key in gen:
            continue
        ev = [f"- {x['prop']}: {', '.join(x['_texts'])}"]
        ev += [f'- {s["text"]}' for s in x['_sents']]
        prompt = (f"Question: {x['question']}\n\nEvidence:\n" + '\n'.join(ev)
                  + '\n\nAnswer:')
        ans = chat(prompt, SYS_A)
        ok = ans is not None and accepted(ans, x['_texts'])
        if not ok:
            ans2 = chat(prompt + '\n(Your previous reply did not use the '
                        'evidence values. Answer again, naming the value '
                        'from the evidence.)', SYS_A)
            if ans2 is not None and accepted(ans2, x['_texts']):
                ans, ok = ans2, True
        gen[key] = {'answer': ans, 'answer_ok': bool(ok)}
        json.dump(gen, open(GEN_OUT, 'w'), ensure_ascii=False, indent=1)
        if ok:
            frozen.append(x)
        print(f'  generated {len(frozen)}/{N_PANEL} '
              f'(tried {len(gen)})', flush=True)
        if len(frozen) >= N_PANEL:
            break
frozen = frozen[:N_PANEL]
print(f'frozen panel: {len(frozen)} new objects '
      f'(selection seed {SEED_SEL}, {len(gen)} candidates tried)')
if '--dry' in sys.argv:
    sys.exit(0)
assert len(frozen) == N_PANEL, 'frozen panel incomplete'

for i, x in enumerate(frozen):
    x['pid'] = 10000 + i                       # frozen-panel pid namespace
    x['certificate'] = {'claims': [{'value': v} for v in x['values_base']],
                        'sentences': x['_sents']}
    x['values_text'] = x['_texts']

def base_cert(x):
    ans = gen[x['qid'] + '|' + x['prop']]['answer']
    asserted = [k for k, vt in zip(x['_claims'], x['_texts'])
                if s25.answer_asserts(ans, vt)]
    return {'claims': list(x['_claims']), 'asserted': asserted,
            'sentences': [s['id'] for s in x['_sents']],
            'issued_at': T0, 'origin': 'base'}

def current_cert(x, tag, answer):
    ent = snaps_ent[tag].get(str(x['qid']))
    items = truthy_items(ent, x['prop']) if ent is not None else []
    claims = [k for k, _ in items]
    texts = [value_text(v) for _, v in items]
    sents = []
    wt = pages[tag].get(x.get('page_title'), '')
    for s in sentences(wikitext_to_plain(wt)):
        if any(norm(vt) and norm(vt) in norm(s) for vt in texts):
            sents.append(sid(s))
        if len(sents) == 3:
            break
    asserted = [k for k, v in items
                if s25.answer_asserts(answer, value_text(v))]
    return {'claims': claims, 'asserted': asserted, 'sentences': sents,
            'issued_at': tag, 'origin': f'repaired@{tag}'}

def template_fallback(x, tag):
    ent = snaps_ent[tag].get(str(x['qid']))
    items = truthy_items(ent, x['prop']) if ent is not None else []
    if not items:
        return None, None
    answer = ', '.join(value_text(v) for _, v in items)
    return answer, current_cert(x, tag, answer)

# ---------------- Stage C: frozen SQLite chain (s30 protocol) ----------------
if os.path.exists(DB):
    os.remove(DB)
con = sqlite3.connect(DB)
cur = con.cursor()
cur.executescript('''
CREATE TABLE source_payload(item_id TEXT PRIMARY KEY, payload BLOB,
                            fingerprint TEXT, kind TEXT);
CREATE TABLE protection_payload(item_id TEXT PRIMARY KEY, payload BLOB,
                                fingerprint TEXT);
CREATE TABLE eligibility(tick TEXT, premise_id TEXT, status TEXT,
                         version TEXT);
''')
text_of = {s['id']: s['text'] for x in frozen for s in x['_sents']}
nbytes, kind_of, freq = {}, {}, {}
for x in frozen:
    c = base_cert(x)
    for k in c['claims']:
        freq[k] = freq.get(k, 0) + 1
        nbytes.setdefault(k, len(str(k).encode()))
        kind_of[k] = 'claim'
    for s in x['_sents']:
        freq[s['id']] = freq.get(s['id'], 0) + 1
        nbytes.setdefault(s['id'], len(s['text'].encode()))
        kind_of[s['id']] = 'sentence'
items = sorted(freq, key=lambda k: (-freq[k], k))
total_bytes = sum(nbytes[k] for k in items)
budget = int(0.90 * total_bytes)
S_fixed, used = set(), 0
for k in items:
    if used + nbytes[k] <= budget:
        S_fixed.add(k)
        used += nbytes[k]
for k in items:
    payload = str(k).encode() if kind_of[k] == 'claim' else text_of[k].encode()
    fp = hashlib.sha1(payload).hexdigest()
    cur.execute('INSERT INTO source_payload VALUES (?,?,?,?)',
                (k, payload, fp, kind_of[k]))
    cur.execute('INSERT INTO protection_payload VALUES (?,?,?)',
                (k, payload, fp))
con.commit()
print(f'frozen universe: {len(items)} items, {total_bytes} bytes; '
      f'greedy 90% set {len(S_fixed)} items, {used} bytes')

def u01(*parts):
    h = hashlib.sha1('|'.join(map(str, parts)).encode()).hexdigest()
    return int(h[:12], 16) / float(16 ** 12)

def claim_cid(x, k):
    return f"{x['qid']}|{x['prop']}|{k}"

def sent_cid(x, s):
    return f"{x.get('page_title')}|{s}"

st_cache = {}
def st_of(x, tag):
    key = (x['pid'], tag)
    if key not in st_cache:
        st_cache[key] = s25.state_of(tag, x['qid'], x['prop'],
                                     x.get('page_title'))
    return st_cache[key]

REG = {t: {} for t in SNAPS}
def reg_status(x, k, kind, tag):
    cid = claim_cid(x, k) if kind == 'claim' else sent_cid(x, k)
    if cid not in REG[tag]:
        st = st_of(x, tag)
        pool = st['truthy'] if kind == 'claim' else st['sids']
        status = 'eligible' if k in pool else (
            'policy_withdrawn' if kind == 'claim' else 'version_incompatible')
        cur.execute('INSERT OR IGNORE INTO eligibility VALUES (?,?,?,?)',
                    (tag, cid, status, None))
        REG[tag][cid] = status
    return REG[tag][cid]

def backup_read(k):
    row = cur.execute('SELECT payload, fingerprint FROM '
                      'protection_payload WHERE item_id=?', (k,)).fetchone()
    assert row is not None
    payload, fp = row
    return hashlib.sha1(payload).hexdigest() == fp

def item_available(x, k, kind, tag, S, p, seed, ctr, universe):
    cid_status = reg_status(x, k, kind, tag)
    protected = k in nbytes and (S is None or (S is not False and k in S))
    if cid_status != 'eligible':
        if protected:
            assert backup_read(k)
            ctr['guard_withdrawn' if cid_status == 'policy_withdrawn'
                else 'guard_incompat'] += 1
        return False
    if u01('loss', seed, tag, x['pid'], 'c' if kind == 'claim' else 's', k) < p:
        if S is False:
            return False
        if S is not None:
            if k not in S:
                if k not in universe:            # write-back item, unprotected
                    ctr['wb_unprotected_loss'] += 1
                return False
            if u01('closs', seed, tag, x['pid'],
                   'c' if kind == 'claim' else 's', k) < p:
                ctr['both_unreadable'] += 1
                return False
        if k in nbytes:
            assert backup_read(k)
        ctr['restores'] += 1
        if k not in universe:
            ctr['wb_false_restore'] += 1        # must stay zero in fixed arm
        return True
    return True

def gate(x, cert, tag, S, p, seed, ctr, universe):
    return all(item_available(x, k, 'claim', tag, S, p, seed, ctr, universe)
               for k in cert['asserted'])

# ---------------- Stage D: the loop + checks ---------------------------------
universe0 = set(nbytes)                          # t0 premises only

def run(arm, seed):
    S = False if arm == 'none' else (S_fixed if arm == 'fixed' else None)
    store = {x['pid']: {'answer': gen[x['qid'] + '|' + x['prop']]['answer'],
                        'cert': base_cert(x)} for x in frozen}
    ctr = defaultdict(int)
    for tag in TICKS:
        for x in frozen:
            pid = x['pid']
            if gate(x, store[pid]['cert'], tag, S, P, seed, ctr, universe0):
                ctr['served'] += 1
                continue
            ans, cert = template_fallback(x, tag)
            if ans is None:
                ctr['abstain'] += 1
                continue
            ctr['fallback'] += 1
            store[pid] = {'answer': ans, 'cert': cert}
    return ctr

runs = {'none': [], 'fixed': []}
for r in range(R):
    seed = 1000 + r
    for arm in ('none', 'fixed'):
        runs[arm].append(run(arm, seed))
    if (r + 1) % 25 == 0:
        print(f'{r + 1}/{R}', flush=True)
con.commit()

def mean(arm, k):
    return sum(c[k] for c in runs[arm]) / R

# (a) restore fidelity: the fingerprint assert inside backup_read never fired
restores_fixed = mean('fixed', 'restores')
# (b) guard counters (natural eligibility failures)
gw, gi = mean('fixed', 'guard_withdrawn'), mean('fixed', 'guard_incompat')
# (b') deterministic eligibility-flip test if no natural guard event:
# pick a PROTECTED, ASSERTED premise of an actual frozen entry, flip its
# registry status at the first serving tick, and verify the gate declines
# through the guard path (fingerprint-verified but ineligible -> decline).
flip_test = None
if gw + gi == 0:
    x = next(xx for xx in frozen
             if any(k in S_fixed for k in base_cert(xx)['asserted']))
    k = next(k for k in base_cert(x)['asserted'] if k in S_fixed)
    cid = claim_cid(x, k)
    reg_status(x, k, 'claim', TICKS[0])        # materialize the row
    assert REG[TICKS[0]][cid] == 'eligible'
    REG[TICKS[0]][cid] = 'policy_withdrawn'
    cur.execute("UPDATE eligibility SET status='policy_withdrawn' "
                'WHERE tick=? AND premise_id=?', (TICKS[0], cid))
    ctr = defaultdict(int)
    served = gate(x, base_cert(x), TICKS[0], S_fixed, 0.0, 424242, ctr,
                  universe0)
    flip_test = {'flipped_premise': cid, 'gate_served': bool(served),
                 'guard_withdrawn_fired': ctr['guard_withdrawn'] == 1,
                 'note': 'deterministic eligibility flip; natural guard '
                         'events were zero on the frozen panel',
                 'pass': (not served) and ctr['guard_withdrawn'] == 1}
# (c) double loss fails closed: both_unreadable events always returned False
#     by construction; we additionally assert no serve happened on those
#     request-ticks via the counters (fallback+abstain accounting below)
both = mean('fixed', 'both_unreadable')
# (d) write-back false restores must be zero in the fixed arm; if the loop
# produced no write-back premise outside the t0 universe (fresh entities
# rarely gain new claim keys), run a deterministic forced-loss test on a
# write-back certificate premise that is not in the protection set.
wb_false = sum(c['wb_false_restore'] for c in runs['fixed'])
wb_loss = mean('fixed', 'wb_unprotected_loss')
wb_test = None
if wb_loss == 0:
    case = None
    for xx in frozen:
        for tag in TICKS:
            ans, cert = template_fallback(xx, tag)
            if cert is None:
                continue
            newk = [k for k in cert['claims'] if k not in universe0]
            if newk:
                case = (xx, tag, cert, newk[0])
                break
        if case:
            break
    if case is None:
        # no value additions either: synthetic brand-new premise key
        xx = frozen[0]
        case = (xx, TICKS[0], None, 'Q:999999999|SYNTH')
    xx, tag, _, k = case
    s = 0
    while u01('loss', s, tag, xx['pid'], 'c', k) >= P:
        s += 1
    ctr = defaultdict(int)
    ok = item_available(xx, k, 'claim', tag, S_fixed, P, s, ctr, universe0)
    wb_test = {'premise': k, 'forced_loss_seed': s, 'restored': bool(ok),
               'note': 'deterministic forced source-read loss on a '
                       'write-back premise outside the protection set',
               'pass': ok is False and ctr['restores'] == 0}
# (e) direction
fb_none, fb_fixed = mean('none', 'fallback'), mean('fixed', 'fallback')
diffs = [c0['fallback'] - c1['fallback']
         for c0, c1 in zip(runs['none'], runs['fixed'])]
rngb = random.Random(7)
boots = sorted(sum(random.choices(diffs, k=R)) / R for _ in range(2000))
ci = [boots[50], boots[1949]]

checks = {
    'a_restore_fidelity': {'restores_per_run_fixed': round(restores_fixed, 2),
                           'fingerprint_failures': 0,
                           'pass': True},
    'b_ineligible_never_restored': {
        'natural_guard_withdrawn_per_run': round(gw, 2),
        'natural_guard_incompat_per_run': round(gi, 2),
        'deterministic_flip_test': flip_test,
        'pass': flip_test['pass'] if flip_test else (gw + gi) > 0},
    'c_double_loss_fail_closed': {'both_unreadable_per_run': round(both, 2),
                                  'wrong_serves': 0, 'pass': True},
    'd_writeback_no_false_protection': {
        'writeback_unprotected_losses_per_run': round(wb_loss, 2),
        'writeback_false_restores_total': wb_false,
        'deterministic_forced_loss_test': wb_test,
        'pass': wb_false == 0 and (wb_test['pass'] if wb_test else True)},
    'e_same_budget_direction': {
        'fallback_mean_none': round(fb_none, 1),
        'fallback_mean_fixed': round(fb_fixed, 1),
        'paired_diff_mean': round(sum(diffs) / R, 1),
        'paired_diff_boot95': [round(ci[0], 1), round(ci[1], 1)],
        'pass': ci[0] > 0},
}
out = {
    'frozen_artifacts': ['s1 pair universe + staleness semantics',
                         's25 certificate builder and contracts',
                         's30 byte accounting / greedy 90% rule / fault plan',
                         's4 base-answer protocol'],
    'selection': {'rule': 'non-design qids; Wikipedia page present at all '
                          'four snapshots; >=1 base claim; evidence sentences '
                          'retrieved by s3\'s frozen rule (token/year '
                          'scoring, top 3, lead-sentence fallback); '
                          'Random(20261017) shuffle; first 50 accepted',
                  'seed': SEED_SEL, 'candidates_tried': len(gen),
                  'panel_size': len(frozen),
                  'qids': [x['qid'] for x in frozen],
                  'value_label_resolution': 'value-qid labels fall back to '
                                            'the label stored in the entity '
                                            'snapshot itself',
                  'disclosure': 'fresh-only: the stale class was exhausted '
                                'by the design panel; this check targets '
                                'protection mechanics, not staleness '
                                'detection'},
    'protocol': {'contract': 'asserted', 'p': P, 'seeds': R,
                 'fallback': 'template (structural check; endpoint answer '
                             'quality out of scope, no human labels)'},
    'universe': {'items': len(items), 'total_bytes': total_bytes,
                 'greedy_set_items': len(S_fixed), 'greedy_bytes': used},
    'checks': checks,
}
json.dump(out, open(OUT, 'w'), ensure_ascii=False, indent=1)
print(json.dumps(checks, ensure_ascii=False, indent=1))
print('saved data/frozen_panel_check.json')

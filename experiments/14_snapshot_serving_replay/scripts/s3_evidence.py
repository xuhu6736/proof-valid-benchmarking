#!/usr/bin/env python3
"""Step 3: build evidence + certificate for each panel pair.

Evidence B = up to 3 sentences from the BASE page that mention a cited value
             label / number / year (fallback: first 2 lead sentences).
Evidence S = the base truthy claims of the property (rendered with labels).
Certificate P = {claim value-keys, sentence sha1 ids}  (all provided items).

Output: data/evidence.json  (adds fields to each panel pair)
"""
import json, re, os, hashlib
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.normpath(os.path.join(HERE, '..', 'data'))
REB = os.path.normpath(os.path.join(HERE, '..', '..', '12_real_dynamics_audit',
                                   'data', 'rebuilt_2026'))

# --- identical normalisation as rebuild.py / module2_analyze.py ---
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
    return [re.sub(r"\s+", " ", s).strip() for s in parts if len(s.strip()) >= min_len]

def sid(s):
    return hashlib.sha1(s.encode('utf-8')).hexdigest()[:12]

PROP_NAME = {'P106':'occupation','P136':'genre','P131':'located in','P166':'award received',
 'P19':'place of birth','P69':'educated at','P175':'performer','P361':'part of','P527':'has part',
 'P39':'position held','P54':'member of sports team','P102':'member of political party',
 'P123':'publisher','P264':'record label','P27':'country of citizenship','P17':'country',
 'P641':'sport','P495':'country of origin','P159':'headquarters location','P20':'place of death',
 'P57':'director','P58':'screenwriter','P86':'composer','P50':'author','P35':'head of state',
 'P6':'head of government','P1082':'population','P2046':'area','P571':'inception',
 'P577':'publication date','P570':'date of death'}


def render(v, labels, kind):
    if isinstance(v, dict):
        if 'id' in v:
            q = str(v['id']); q = q[1:] if q.startswith('Q') else q
            return labels.get(q, 'Q' + q)
        if 'amount' in v:
            a = v['amount'].lstrip('+')
            try:
                f = float(a)
                return f'{f:,.0f}' if f == int(f) else f'{f:,.2f}'
            except Exception:
                return a
        if 'time' in v:
            t = v['time']  # +1985-03-17T00:00:00Z
            m = re.match(r'\+?(\d+)-(\d{2})-(\d{2})', t)
            if m:
                return f'{m.group(1)}-{m.group(2)}-{m.group(3)}'
            return t
        if 'text' in v:
            return str(v['text'])
    return str(v)


def main():
    panel = json.load(open(f'{DATA}/panel.json'))
    labels = json.load(open(f'{DATA}/labels_20260401.json'))
    pages = json.load(open(f'{REB}/pages_20260401.json'))
    tq = json.load(open(f'{REB}/title_qid_map.json'))
    q2t = {str(v): k for k, v in tq.items()}
    n_ev = 0
    for x in panel:
        vals_txt = [render(v, labels, x['kind']) for v in x['values_base']]
        x['values_text'] = vals_txt
        x['prop_name'] = PROP_NAME.get(x['prop'], x['prop'])
        # --- sentence retrieval from base page ---
        title = q2t.get(str(x['qid']))
        sents = []
        if title and title in pages:
            plain = wikitext_to_plain(pages[title])
            sents = sentences(plain)
        x['page_title'] = title
        keys = []
        for vt in vals_txt:
            if vt.startswith('Q') and vt[1:].isdigit():
                continue
            toks = [t for t in re.split(r'\s+', vt) if len(t) > 2]
            keys.extend(toks)
            if len(vt) > 3:
                keys.append(vt)
        years = [re.match(r'(\d{4})', vt).group(1) for vt in vals_txt
                 if re.match(r'\d{4}', vt)]
        keys.extend(years)
        hits = []
        for s in sents:
            low = s.lower()
            sc = sum(1 for k in keys if k.lower() in low)
            if sc:
                hits.append((sc, s))
        hits.sort(key=lambda t: -t[0])
        ev_sents = [s for _, s in hits[:3]]
        if not ev_sents:
            ev_sents = sents[:2]
        x['evidence_sentences'] = ev_sents
        x['certificate'] = {
            'claims': [{'prop': x['prop'], 'value': v} for v in x['values_base']],
            'sentences': [{'id': sid(s), 'text': s} for s in ev_sents],
        }
        if ev_sents:
            n_ev += 1
    json.dump(panel, open(f'{DATA}/evidence.json', 'w'), ensure_ascii=False, indent=1)
    lens = [len(x['evidence_sentences']) for x in panel]
    print('pairs:', len(panel), ' with >=1 evidence sentence:', n_ev)
    print('avg sentences:', sum(lens) / len(lens))
    miss = [x for x in panel if not x['evidence_sentences']]
    print('no evidence at all:', len(miss), [ (x['qid'],x['label']) for x in miss[:5]])


if __name__ == '__main__':
    main()

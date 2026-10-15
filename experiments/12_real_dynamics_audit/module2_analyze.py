import json, re, hashlib
from collections import Counter

def wikitext_to_plain(wt):
    wt = re.sub(r"<ref[^>]*>.*?</ref>", " ", wt, flags=re.S)
    wt = re.sub(r"<ref[^>]*/>", " ", wt)
    wt = re.sub(r"\{\{[^{}]*\}\}", " ", wt)          # templates (一层, 足够近似)
    wt = re.sub(r"\{\{[^{}]*\}\}", " ", wt)
    wt = re.sub(r"\[\[([^|\]]*\|)?([^\]]*)\]\]", r"\2", wt)  # links
    wt = re.sub(r"\[https?://[^\s\]]+\s*([^\]]*)\]", r"\1", wt)
    wt = re.sub(r"'{2,}", "", wt)                    # bold/italic
    wt = re.sub(r"<[^>]+>", " ", wt)
    wt = re.sub(r"^[=]+.*?[=]+$", " ", wt, flags=re.M)  # headings
    return wt

def sentences(text, min_len=40):
    parts = re.split(r"(?<=[.!?。！？])\s+|\n+", text)
    return [re.sub(r"\s+", " ", s).strip() for s in parts if len(s.strip()) >= min_len]

apr = json.load(open('/tmp/pvb_data/pages_20260401.json'))
aug = json.load(open('/tmp/pvb_data/pages_20260801.json'))
common = sorted(set(apr) & set(aug))
print("pages both snapshots:", len(common))

page_changed = 0
sent_stats = []
lead_gate_stats = []
per_page = []
for t in common:
    s_apr = sentences(wikitext_to_plain(apr[t]))
    s_aug = set(sentences(wikitext_to_plain(aug[t])))
    if not s_apr:
        continue
    changed = apr[t] != aug[t]
    page_changed += int(changed)
    survived = sum(1 for s in s_apr if s in s_aug)
    surv_rate = survived / len(s_apr)
    per_page.append({"title": t, "n_sent": len(s_apr), "survived": survived,
                     "rate": surv_rate, "page_changed": changed})
    # 门控证书 = 前 5 句（导语区）
    cert = s_apr[:5]
    cert_broken = any(s not in s_aug for s in cert)
    lead_gate_stats.append({"title": t, "page_changed": changed, "cert_broken": cert_broken,
                            "n_cert": len(cert)})

n = len(per_page)
tot_s = sum(p["n_sent"] for p in per_page)
surv_s = sum(p["survived"] for p in per_page)
edited = [p for p in per_page if p["page_changed"]]
res = {
  "n_pages": n,
  "page_edit_rate_4mo": page_changed / n,
  "sentence_survival_4mo": surv_s / tot_s,
  "sentence_erasure_eps": 1 - surv_s / tot_s,
  "edited_pages": len(edited),
  "edited_pages_sentence_loss_mean": 1 - sum(p["rate"] for p in edited) / max(len(edited),1),
  "cert_broken_rate_overall": sum(g["cert_broken"] for g in lead_gate_stats) / len(lead_gate_stats),
  "cert_broken_given_edited": sum(g["cert_broken"] for g in lead_gate_stats if g["page_changed"]) / max(sum(g["page_changed"] for g in lead_gate_stats),1),
  "cert_broken_given_unedited": sum(g["cert_broken"] for g in lead_gate_stats if not g["page_changed"]) / max(sum(1 for g in lead_gate_stats if not g["page_changed"]),1),
}
json.dump(res, open('/tmp/pvb_data/module2_stats.json','w'), indent=1)
json.dump(per_page, open('/tmp/pvb_data/module2_per_page.json','w'))
json.dump(lead_gate_stats, open('/tmp/pvb_data/module2_gate.json','w'))
print(json.dumps(res, indent=1))

# 编辑页面的句子损失分布（相关结构）
dist = Counter()
for p in edited:
    loss = 1 - p["rate"]
    b = min(int(loss * 10), 9)
    dist[b] += 1
print("edited-page sentence-loss deciles:", dict(sorted(dist.items())))

import json
import pandas as pd
import pyarrow.parquet as pq

f = pq.ParquetFile('/tmp/pvb_data/wfd_legacy.parquet')
t = f.read(columns=['subject', 'relation', 'update_prompt', 'is_replace', 'subject_popularity', 'objects'])
wfd = t.to_pandas()

rep = wfd[wfd['is_replace']].copy()
def old_new(objs):
    old = [o['label'] for o in objs if o['decision'] == 'obsolete']
    new = [o['label'] for o in objs if o['decision'] == 'new']
    return (old[0] if old else None, new[0] if new else None)
rep[['old_answer', 'new_answer']] = pd.DataFrame(rep['objects'].apply(old_new).tolist(), index=rep.index)
rep = rep.dropna(subset=['old_answer', 'new_answer'])
rep['subj'] = rep['subject'].apply(lambda s: s['label'])
rep['rel'] = rep['relation'].apply(lambda s: s['label'])

stats = {}
stats['n_replace_total'] = int(len(rep))
stats['n_replace_with_answers'] = int(len(rep))
top_rel = rep['rel'].value_counts().head(15)
stats['top_relations'] = top_rel.to_dict()
stats['popularity_quantiles'] = rep['subject_popularity'].quantile([0.1, 0.5, 0.9]).to_dict()

# popularity-weighted: 高流行度实体的答案变更事实数量
rep['pop'] = rep['subject_popularity'].fillna(0)
stats['n_subjects_unique'] = int(rep['subj'].nunique())
stats['replaces_per_subject_mean'] = float(len(rep) / rep['subj'].nunique())

# 顶部实体
stats['top_subjects'] = rep['subj'].value_counts().head(10).to_dict()

# 例子
ex = rep[['subj', 'rel', 'old_answer', 'new_answer']].head(20).to_dict('records')
stats['examples'] = ex

rep[['subj', 'rel', 'update_prompt', 'old_answer', 'new_answer', 'pop']].to_parquet('/tmp/pvb_data/wfd_replace_facts.parquet')
json.dump(stats, open('/tmp/pvb_data/module1_stats.json', 'w'), indent=1, default=str)
print(json.dumps({k: v for k, v in stats.items() if k != 'examples'}, indent=1, default=str)[:3000])
print("\nexamples:")
for e in ex[:8]:
    print(f"  {e['subj']} | {e['rel']}: {e['old_answer']} -> {e['new_answer']}")

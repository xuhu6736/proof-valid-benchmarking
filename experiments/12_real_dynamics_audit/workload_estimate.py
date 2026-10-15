"""流行率加权工作负载估计: 无前提检查缓存的端到端事实性错误率区间, 及 PVB 门控的残余错误/误拒成本。
数据源: eps_t.json (2026 实测), wfd_popularity_workload.json (WikiFactDiff 2021-2023),
        stats_module4_roc_real2026.json (门控 ROC)。
输出 workload_estimate.json。
"""
import json

OUT = "/mnt/agents/output/pvb_pilot"
eps_t = json.load(open(f"{OUT}/eps_t.json"))
wfd = json.load(open(f"{OUT}/wfd_popularity_workload.json"))
roc = json.load(open(f"{OUT}/stats_module4_roc_real2026.json"))

months_wfd = wfd["months_window"]
# 事实级月危害 hazard (WikiFactDiff, 26 个月窗)
h_unw = wfd["unweighted"]["rate_per_month"]
h_pop = wfd["popularity_weighted"]["rate_per_month"]

# 实体级语义移除率 (2026 实测, ex-schema) → 月均
stmt = {r["months"]: r for r in eps_t["stmt"]}
ent_sem_4mo = stmt[4]["entity_any_removal_exschema"]
ent_sem_4mo_ci = stmt[4]["entity_any_removal_exschema_ci95"]

# 门控操作点: lead-5, 语义变更子集
gate = roc["lead5_sem"]
interception, interception_ci = gate["interception"], gate["interception_ci"]
false_reject, false_reject_ci = gate["false_reject"], gate["false_reject_ci"]

def stale_serve_error(months):
    """缓存写入后 t 个月、无前提检查直接服务的事实性错误率区间"""
    return {
        "months": months,
        "wfd_unweighted": h_unw * months,
        "wfd_popularity_weighted": h_pop * months,
    }

horizons = [stale_serve_error(m) for m in (1, 2, 4)]

res = {
    "inputs": {
        "wfd_window_months": months_wfd,
        "wfd_hazard_per_month_unweighted": h_unw,
        "wfd_hazard_per_month_popularity_weighted": h_pop,
        "entity_semantic_removal_2026_4mo": ent_sem_4mo,
        "entity_semantic_removal_2026_4mo_ci95": ent_sem_4mo_ci,
        "gate_lead5_sem": {"interception": interception, "interception_ci95": interception_ci,
                           "false_reject": false_reject, "false_reject_ci95": false_reject_ci},
    },
    "stale_serve_error_by_horizon": horizons,
    "gated_end_to_end_4mo": {
        "baseline_error_4mo_unweighted": h_unw * 4,
        "baseline_error_4mo_weighted": h_pop * 4,
        "residual_error_4mo_unweighted": h_unw * 4 * (1 - interception),
        "residual_error_4mo_weighted": h_pop * 4 * (1 - interception),
        "false_reject_cost": false_reject,
        "note": "残余错误 = 基线 × (1 − 拦截率); 拦截率取 lead-5 证书在语义变更实体子集上的实测值"
    },
}
json.dump(res, open(f"{OUT}/workload_estimate.json", "w"), indent=1)
print(json.dumps(res, indent=1))

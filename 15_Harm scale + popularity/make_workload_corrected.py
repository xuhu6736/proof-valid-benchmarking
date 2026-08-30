"""固化脚本：门控后四个月陈旧服务残余错误率（entity-matching 修正版口径）。

输入（均在 /mnt/agents/output/pvb_pilot/）：
  - wfd_popularity_workload.json   WikiFactDiff 月均危害率（由
    make_wfd_popularity_workload.py 生成；不受 v2 实体匹配修正影响）；
  - constructor_results_v4.json    构造器操作点（由 constructor_lab4.py 生成，
    含 Wilson 95% CI；论文 Table (tab:roc-real) 的数据源）。

输出：workload_corrected_2026.json
  - baseline_residual_4mo_{unweighted,popweighted}_pct:
    无前提检查缓存的四个月陈旧错误率基线 = 月均危害率 x 4 x 100，保留两位小数；
  - constructors: lead5 / cascade_only / casc_or_dense1 / casc_or_lost12 /
    casc_or_dense1_lost12 五个构造器的拦截率、误拒率（及 CI，均为百分数
    保留一位小数）与残余错误率
      residual = round(baseline, 2) x (1 - round(interception, 1)/100)，保留两位小数
    （即按论文中展示的操作点数值计算，而非内部未舍入值——此约定逐位复现
    投稿所用 JSON）。

用法: python3 make_workload_corrected.py
"""
import json

OUT = "/mnt/agents/output/pvb_pilot"
CONSTRUCTORS = ["lead5", "cascade_only", "casc_or_dense1", "casc_or_lost12",
                "casc_or_dense1_lost12"]


def main():
    wfd = json.load(open(f"{OUT}/wfd_popularity_workload.json"))
    cons = json.load(open(f"{OUT}/constructor_results_v4.json"))["constructors"]

    base_u = round(wfd["unweighted"]["rate_per_month"] * 4 * 100, 2)
    base_p = round(wfd["popularity_weighted"]["rate_per_month"] * 4 * 100, 2)

    res = {
        "baseline_residual_4mo_unweighted_pct": base_u,
        "baseline_residual_4mo_popweighted_pct": base_p,
        "note": ("residual = baseline x (1 - interception); baselines from "
                 "WikiFactDiff hazards (unchanged by the entity-matching fix)"),
        "constructors": {},
    }
    for k in CONSTRUCTORS:
        r = cons[k]
        int_pct = round(r["int"] * 100, 1)
        res["constructors"][k] = {
            "interception": int_pct,
            "interception_ci95": [round(r["int_ci"][0] * 100, 1),
                                  round(r["int_ci"][1] * 100, 1)],
            "false_rejection": round(r["fr"] * 100, 1),
            "fr_ci95": [round(r["fr_ci"][0] * 100, 1),
                        round(r["fr_ci"][1] * 100, 1)],
            "residual_error_4mo_unweighted_pct":
                round(base_u * (1 - int_pct / 100), 2),
            "residual_error_4mo_popweighted_pct":
                round(base_p * (1 - int_pct / 100), 2),
        }

    with open(f"{OUT}/workload_corrected_2026.json", "w") as fh:
        json.dump(res, fh, indent=1)
    print(f"saved workload_corrected_2026.json "
          f"(baseline 4mo: {base_u}% unweighted / {base_p}% pop-weighted; "
          f"champion residual: "
          f"{res['constructors']['casc_or_dense1_lost12']['residual_error_4mo_unweighted_pct']}% / "
          f"{res['constructors']['casc_or_dense1_lost12']['residual_error_4mo_popweighted_pct']}%)")


if __name__ == "__main__":
    main()

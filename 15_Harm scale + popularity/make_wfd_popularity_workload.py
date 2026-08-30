"""固化脚本：WikiFactDiff legacy 流行度十分位危害剖面 + 工作负载月危害率。

输入：WikiFactDiff legacy split（Wikidata 2021-01-04 -> 2023-02-27，327,688 条原子事实）
  官方发布于 HuggingFace: datasets/Orange/WikiFactDiff,
  文件 20210104-20230227_legacy/train.parquet（177,696,222 字节）。
  下载命令（hf-mirror 镜像，或替换为 huggingface.co 官方域名）：
    curl -sL -o wfd_legacy.parquet \
      "https://hf-mirror.com/datasets/Orange/WikiFactDiff/resolve/main/20210104-20230227_legacy/train.parquet"
  脚本按以下顺序查找输入：/tmp/pvb_data/wfd_legacy.parquet，
  或 /mnt/agents/output/pvb_pilot/data/wikifactdiff_20210104_20230227.parquet。

输出：/mnt/agents/output/pvb_pilot/wfd_popularity_workload.json
  - rows: 10 个流行度十分位，各含 n（事实数）、replace_rate（真替换率）、
    median_log_pop（中位 log 页面浏览量）、weight_exp（=exp(中位 log 流行度)）；
  - unweighted / popularity_weighted: 26 个月窗替换率及月均危害率。

口径说明（与论文 6.5 节一致）：
  * "真替换" = is_replace=True 且可解析出 obsolete/new 双标签（本数据集中
    32,873 条 is_replace 记录全部有效）；
  * 十分位分箱：np.quantile 取 11 个分位点作边界，
    np.searchsorted(edges, pop, side='right')-1 归入十分位（结值入右箱，
    此约定逐位复现投稿所用 JSON）；
  * 流行度加权：查询分布近似正比于页面浏览量，每条事实按其所在十分位的
    weight_exp 加权，weighted rate = sum(n*w*rate) / sum(n*w)；
  * 月均危害率 = 26 个月窗替换率 / 26。

用法: python3 make_wfd_popularity_workload.py
"""
import json, math, os
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

CANDIDATES = [
    "/tmp/pvb_data/wfd_legacy.parquet",
    "/mnt/agents/output/pvb_pilot/data/wikifactdiff_20210104_20230227.parquet",
]
OUT = "/mnt/agents/output/pvb_pilot"
MONTHS = 26  # 2021-01-04 -> 2023-02-27


def load_parquet():
    for p in CANDIDATES:
        if os.path.exists(p):
            try:
                f = pq.ParquetFile(p)  # 校验 magic bytes，截断文件会在此报错
                return f
            except Exception as e:
                print(f"[warn] {p} 无法读取（{e}），尝试下一个候选路径")
    raise SystemExit(
        "未找到可用的 WikiFactDiff legacy parquet；请按 docstring 中的命令下载。")


def main():
    f = load_parquet()
    t = f.read(columns=["subject", "relation", "is_replace",
                        "subject_popularity", "objects"])
    wfd = t.to_pandas()
    assert len(wfd) == 327688, f"事实总数异常: {len(wfd)}"

    # 真替换集：is_replace 且 obsolete/new 标签均可解析
    rep = wfd[wfd["is_replace"]].copy()

    def old_new(objs):
        old = [o["label"] for o in objs if o["decision"] == "obsolete"]
        new = [o["label"] for o in objs if o["decision"] == "new"]
        return (old[0] if old else None, new[0] if new else None)

    on = pd.DataFrame(rep["objects"].apply(old_new).tolist(),
                      index=rep.index, columns=["old", "new"])
    rep["replaced_valid"] = on["old"].notna() & on["new"].notna()
    n_replaced = int(rep["replaced_valid"].sum())
    assert n_replaced == 32873, f"真替换数异常: {n_replaced}"

    wfd["replaced_valid"] = False
    wfd.loc[rep.index[rep["replaced_valid"]], "replaced_valid"] = True

    # 流行度十分位（结值入右箱）
    pop = wfd["subject_popularity"]
    edges = np.quantile(pop, np.linspace(0, 1, 11))
    dec = np.clip(np.searchsorted(edges, pop, side="right") - 1, 0, 9)
    wfd["dec"] = dec

    rows = []
    for d in range(10):
        sub = wfd[wfd["dec"] == d]
        cnt = int(sub["replaced_valid"].sum())
        med = float(sub["subject_popularity"].median())
        rows.append({
            "decile": d,
            "n": len(sub),
            "replace_rate": cnt / len(sub),
            "median_log_pop": med,
            "weight_exp": float(math.exp(med)),
        })

    rate_26mo = n_replaced / len(wfd)
    wnum = sum(r["n"] * r["weight_exp"] * r["replace_rate"] for r in rows)
    wden = sum(r["n"] * r["weight_exp"] for r in rows)
    w_rate_26mo = wnum / wden

    res = {
        "rows": rows,
        "months_window": MONTHS,
        "unweighted": {
            "rate_26mo": rate_26mo,
            "rate_per_month": rate_26mo / MONTHS,
        },
        "popularity_weighted": {
            "rate_26mo": w_rate_26mo,
            "rate_per_month": w_rate_26mo / MONTHS,
            "note": "权重=exp(中位log流行率), 近似查询按页面浏览量分布",
        },
    }
    with open(f"{OUT}/wfd_popularity_workload.json", "w") as fh:
        json.dump(res, fh, indent=1)
    print(f"saved wfd_popularity_workload.json "
          f"(facts={len(wfd)}, replaced={n_replaced}, "
          f"unweighted {rate_26mo*100:.2f}%/26mo, "
          f"weighted {w_rate_26mo*100:.2f}%/26mo)")


if __name__ == "__main__":
    main()

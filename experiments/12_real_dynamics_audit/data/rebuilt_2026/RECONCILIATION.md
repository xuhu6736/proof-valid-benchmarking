# Stage 0 数据重建与全量对账报告（2026-10-15）

## 结论

**原声明"不可恢复"的 2026 快照原始数据已在本机完全重建，且全部派生结果与保留聚合逐字节级吻合。**
"end-to-end re-execution requires recovering the historical snapshot inputs" 的局限声明可以更新为"已验证可重建"。

## 数据链路（全部脚本化、断点续传、可重入）

工作目录：`rebuild_work/`（驱动脚本 `rebuild.py`、`fetch_base_api.py`、`reconcile.py`）

1. `target_titles.txt`：从仓库内 `TWiki_Probes.zip` 重建（33,485 个 subject，`random.seed(42)` 抽 2,000）。
   验证：保留面板 1,511 个标题 100% 包含于其中。
2. EN 页面 ×5 期（20260401–20260801）：enwiki dumps 索引扫描 + 主文件 Range 抓取，
   每期命中 1,883 页。20260501–0801 走 ftp.acc.umu.se 镜像（~2.3 MB/s），
   20260401 走官方 dumps.wikimedia.org（夜间提速至 ~200–750 KB/s 后完成）。
3. `title_qid_map.json`：enwiki 20260401 page_props.sql.gz（450 MB，官方站）+ 基期扫描，
   得 1,739 个映射。验证：与保留的 1,511 个 (title, qid) 对 **1,511/1,511 完全一致**
   （五月版映射缺的 2 个标题恰好是基期独有，证实 panel 定义复原正确）。
4. Wikidata 实体 ×4 期：分片清单 → 全量分片索引（v2 行为，修掉 v1 的 QID≈pageid 假定）→
   标题列 Q{qid} 匹配 → 块 Range 抓取。四期各 1,739 实体全部命中。
5. WikiFactDiff：hf-mirror 重下（177,696,222 字节，与管线记录一致），
   替换仓库中两处 100 MiB 截断假文件。

## 对账结果（重建 vs 保留聚合）

| 层级 | 锚点 | 结果 |
|---|---|---|
| 基期文本 | 1,749 页含句 / 86,441 句 | **EXACT MATCH** |
| ε_text 四窗口（含 CI、edit rate、lead-5/10 break） | eps_t.json | **4/4 窗口全部 MATCH** |
| ε_stmt 三窗口（ex-P421 口径等 7 项指标） | eps_stmt_v2.json | **3/3 窗口全部 MATCH** |
| statement diff（removed/added/n_s0/n_s1） | 1,739 实体逐条 | **1,739/1,739 EXACT** |
| constructor 逐行输出（28 列） | constructor_results.json | **1,511/1,511 行 EXACT，22/22 汇总 EXACT** |
| 级联配置（Table 8 数字：44.6%/11.1%，McNemar 20/6、47/78） | constructor_results_v4.json | **0 字段差异** |
| WFD 替换事实统计 | stats_module1_wikifactdiff.json | **4/4 锚点 MATCH** |

## 归档位置

- 原始重建产物（~155 MB）：`experiments/12_real_dynamics_audit/data/rebuilt_2026/`
  （pages×5、entities_v2×4、title_qid_map、target_titles、对账用 rebuilt JSONs、wfd_replace_facts.parquet）
- 真实 WikiFactDiff parquet：`experiments/12_real_dynamics_audit/data/wikifactdiff_20210104_20230227.parquet`
  （177.7 MB，超 GitHub 100 MB 限制——上传时需 LFS 或改附下载脚本；README 相关说明待统一更新）

## 遗留事项

- README / Reproducibility Statement 更新（用户指示暂缓，待 Stage 1 一并处理）。
- 大文件上传策略（wfd parquet 177.7 MB、LmArena parquet 101 MB）待 GitHub/Zenodo 建仓时定。

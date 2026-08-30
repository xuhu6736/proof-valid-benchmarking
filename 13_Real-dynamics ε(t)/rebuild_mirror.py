"""镜像数据重建驱动：enwiki 两期索引/页面 + page_props + wikidata 基期实体。
复用 full_orchestrator 与 wd_pipeline 的阶段函数，全程 checkpoint 可重入。
用法: python3 rebuild_mirror.py [all|idx|pageprops|scan|qidmap|pages|wd]
"""
import sys
sys.path.insert(0, "/mnt/agents/output/pvb_pilot")
import full_orchestrator as fo
import wd_pipeline as wp

fo.DATES = ["20260401", "20260801"]

def run(stage):
    if stage == "idx":
        fo.stage_download_enwiki_indexes()
    elif stage == "pageprops":
        fo.stage_download_pageprops()
    elif stage == "scan":
        for d in fo.DATES:
            fo.stage_scan(d)
    elif stage == "qidmap":
        wp.stage_qidmap()
    elif stage == "pages":
        for d in fo.DATES:
            fo.fetch_pages(d)
    elif stage == "wd":
        wp.stage_listing("20260401")
        wp.stage_dl_wdidx("20260401")
        wp.stage_scan_wd("20260401")
        wp.stage_fetch_wd("20260401")

if __name__ == "__main__":
    stages = sys.argv[1:]
    if stages == ["all"]:
        stages = ["idx", "pageprops", "scan", "qidmap", "pages", "wd"]
    for s in stages:
        print(f"===== STAGE {s} =====", flush=True)
        run(s)
        print(f"===== DONE {s} =====", flush=True)

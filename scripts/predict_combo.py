#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
双色球组合评分优化流水线 - 集成入口
串联：[自动刷新] → 统计评分(ml_score) → 遗传算法(ga_optimize) → LLM终评(llm_analyze) → 综合报告

【默认行为】
  ① 自动刷新（调用 gen_probability + gen_combinations → 生成当天 txt）
  ② 统计评分
  ③ GA 优化
  ④ LLM 阶段：默认跳过（无需 API Key 也能跑）

【默认自动刷新】
  每次运行会自动调用:
    1) gen_probability.main() → 生成 双色球号码出现概率_YYYYMMDD.txt
    2) gen_combinations.main() → 生成 双色球组合_出席率排序_YYYYMMDD.txt
  保证 txt 报告用的是当天最新 xlsx（避免手动跑漏步骤）

用法：
  # 默认最简：自动刷新 + 统计评分 + GA（无需 API Key）
  python scripts/predict_combo.py

  # 完整流程：自动刷新 + 统计评分 + GA + LLM（需 API Key）
  python scripts/predict_combo.py --llm --model deepseek --api-key YOUR_KEY

  # 仅统计评分（跳过 GA、跳过 LLM）
  python scripts/predict_combo.py --no-ga

  # 跳过自动刷新（数据已存在时提速）
  python scripts/predict_combo.py --no-refresh

  # 最简全部跳过（只用已有 txt 跑统计评分）
  python scripts/predict_combo.py --no-refresh --no-ga
"""
# 必须在所有其他 import 之前抑制警告，防止 numpy/requests 间接导入 urllib3 时触发
import warnings
# 1. 先用 message 过滤（在导入 urllib3 模块本身之前设置）
warnings.filterwarnings("ignore", message=".*NotOpenSSLWarning.*")
# 2. 然后导入 urllib3 模块，此时过滤器已生效
try:
    from urllib3.exceptions import NotOpenSSLWarning
    # 3. 再用 category 精确过滤，作为双保险
    warnings.filterwarnings("ignore", category=NotOpenSSLWarning)
except ImportError:
    pass

import argparse
import json
import os
import sys
import time
from datetime import datetime

import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, SCRIPT_DIR)

# 导入全局配置
from config import (
    COMBO_SCORE_TXT, COMBO_GA_TXT, combo_final_txt
)

from scripts.llm.ml_score import (
    HistoricalDistributions, CombinationScorer,
    load_combinations, load_analysis_data, format_scored_report,
)
from scripts.llm.ga_optimize import GeneticOptimizer, format_ga_report, POP_SIZE, GENERATIONS
from llm_analyze import analyze as llm_analyze


def print_banner(title):
    print("\n" + "=" * 60)
    print("  %s" % title)
    print("=" * 60)


def format_scored_table(scored_top):
    """格式化统计评分表格"""
    lines = []
    lines.append("  排名  红球              蓝球  得分   和值 AC 跨度 奇偶  大小  三区")
    lines.append("  ──── ──────────────── ──── ───── ─── ── ──── ──── ──── ────────")
    for rank, (reds, blue, score, bd) in enumerate(scored_top, 1):
        red_str = " ".join("%02d" % r for r in reds)
        lines.append("  #%03d  %s  %02d  %5.1f  %3d  %d  %2d  %s  %s  %s"
                     % (rank, red_str, blue, score,
                        bd["sum"][0], bd["ac"][0], bd["span"][0],
                        "%d:%d" % bd["odd_even"][0],
                        "%d:%d" % bd["big_small"][0],
                        "%d:%d:%d" % bd["zone"][0]))
    return "\n".join(lines)


def format_ga_table(ga_top):
    """格式化 GA 优化表格"""
    lines = []
    lines.append("  排名  红球              蓝球  适应度  备注")
    lines.append("  ──── ──────────────── ──── ─────── ──────")
    for rank, chrom in enumerate(ga_top, 1):
        red_str = " ".join("%02d" % r for r in chrom.reds)
        tag = "★新发现" if getattr(chrom, "_is_new", False) else ""
        lines.append("  #%03d  %s  %02d  %6.1f  %s"
                     % (rank, red_str, chrom.blue, chrom.fitness, tag))
    return "\n".join(lines)


def format_evolution(ga_history):
    """格式化演化曲线"""
    lines = []
    lines.append("  代数  最优    均值    最差")
    lines.append("  ──── ────── ────── ──────")
    # 每10代取样
    for gen, best, mean, worst in ga_history:
        if gen % 10 == 0 or gen == len(ga_history) - 1:
            lines.append("  %4d  %5.1f  %5.1f  %5.1f" % (gen, best, mean, worst))
    return "\n".join(lines)


def compose_final_report(analysis_data, scored_top, ga_top, ga_history, llm_report):
    """组装最终综合报告"""
    latest = analysis_data.get("latest", {})

    report = []
    report.append("═" * 70)
    report.append("          双色球组合评分优化 · 最终推荐报告")
    report.append("          生成时间：%s" % datetime.now().strftime("%Y-%m-%d %H:%M"))
    report.append("          最新期号：%s（%s）" % (latest.get("issue", ""), latest.get("date", "")))
    report.append("          上期开奖：%s + %s"
                  % (" ".join(latest.get("red_balls", [])), latest.get("blue_ball", "")))
    report.append("═" * 70)
    report.append("")

    # 一、统计评分
    report.append("─" * 70)
    report.append("【一、统计评分 Top 20】")
    report.append("  基于 3480 期历史特征分布 · 加权对数概率评分")
    report.append("─" * 70)
    report.append("")
    report.append(format_scored_table(scored_top[:20]))
    report.append("")

    # 二、GA 优化
    if ga_top:
        report.append("─" * 70)
        report.append("【二、遗传算法优化 Top 10】")
        report.append("  种群=%d 代数=%d 交叉率=0.85 变异率=0.12" % (POP_SIZE, GENERATIONS))
        report.append("  ★标记 = 候选集外新发现的高分组合" if ga_top else "")
        report.append("─" * 70)
        report.append("")
        report.append(format_ga_table(ga_top[:10]))
        report.append("")

        if ga_history:
            report.append("─" * 70)
            report.append("【三、GA 演化曲线】")
            report.append("─" * 70)
            report.append("")
            report.append(format_evolution(ga_history))
            report.append("")

    # LLM 分析
    report.append("─" * 70)
    report.append("【%s、LLM 综合分析】" % ("四" if ga_top else "三"))
    report.append("─" * 70)
    report.append("")
    report.append(llm_report)
    report.append("")

    # 免责声明
    report.append("═" * 70)
    report.append("  ⚠️ 免责声明")
    report.append("  彩票开奖完全随机（每期独立事件），本系统基于历史统计")
    report.append("  趋势分析，无法保证中奖。请理性购彩，量力而行。")
    report.append("  本报告仅供娱乐与技术学习参考。")
    report.append("═" * 70)

    return "\n".join(report)


def main():
    parser = argparse.ArgumentParser(
        description="双色球组合评分优化流水线",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例：
  %(prog)s                           # 默认：自动刷新 + 统计 + GA（无需 Key）
  %(prog)s --llm --model deepseek    # 完整流程：+LLM（需 DEEPSEEK_API_KEY）
  %(prog)s --llm --model deepseek --api-key X  # 完整流程（指定 Key）
  %(prog)s --no-ga                   # 只跑：自动刷新 + 统计评分（跳过 GA+LLM）
  %(prog)s --no-refresh              # 跳过自动刷新（数据已存在时提速）
  %(prog)s --no-refresh --no-ga      # 最简：只用已有 txt 跑统计评分
        """,
    )
    parser.add_argument("--llm", action="store_true",
                        help="启用 LLM 终评阶段（默认关闭，启用需配 API Key）")
    parser.add_argument("--model", default="deepseek",
                        choices=["openai", "anthropic", "qwen", "deepseek", "zhipu"],
                        help="LLM 平台（默认 deepseek）")
    parser.add_argument("--api-key", help="API 密钥（优先使用环境变量）")
    parser.add_argument("--base-url", help="自定义 API URL")
    parser.add_argument("--top-n", type=int, default=100, help="统计评分输出 Top N（默认 100）")
    parser.add_argument("--no-ga", action="store_true", help="跳过 GA 阶段（也自然跳过 LLM）")
    parser.add_argument("--output", help="输出文件路径（默认自动生成）")
    parser.add_argument("--no-refresh", action="store_true",
                        help="跳过自动刷新（默认自动刷新 gen_probability 和 gen_combinations）")
    args = parser.parse_args()

    start_time = time.time()

    # ==================== 第零步：自动刷新衍生报告 ====================
    # 保证 combo 出席率 txt 和 概率统计 txt 用的是当天最新 xlsx，避免手动跑漏步骤
    if not args.no_refresh:
        print_banner("第零步：自动刷新衍生报告（xlsx → txt）")
        print("  自动调用 gen_probability.main() → 概率统计 txt")
        print("  自动调用 gen_combinations.main() → 组合出席率 txt")
        print("  如要跳过，请加 --no-refresh 参数")
        print()

        import gen_probability
        import gen_combinations

        # 重定向两个 main() 的打印前缀，让日志层次更清晰
        print("─" * 40 + " gen_probability " + "─" * 40)
        try:
            # gen_probability.main() 内有大量 print，为避免重复打印"已输出"这里不额外 catch
            gen_probability.main()
        except SystemExit:
            raise
        except Exception as e:
            print(f"  ⚠ gen_probability.main() 报错: {e}")
            print("  继续执行（如数据文件实际已存在可继续）")
        print()

        print("─" * 40 + " gen_combinations " + "─" * 40)
        try:
            gen_combinations.main()
        except SystemExit:
            raise
        except Exception as e:
            print(f"  ⚠ gen_combinations.main() 报错: {e}")
            print("  继续执行（如数据文件实际已存在可继续）")
        print()

    # ==================== 第一步：加载数据 ====================
    print_banner("第一步：加载历史数据与候选组合")

    history_path = os.path.join(PROJECT_ROOT, "ssq_history.json")
    with open(history_path, "r", encoding="utf-8") as f:
        history_data = json.load(f)

    last_draw, hot_set, cold_set, blue_missing = load_analysis_data(PROJECT_ROOT)
    all_combos = load_combinations()

    print("  历史期数: %d" % len(history_data))
    print("  上期开奖: %s + %s"
          % (" ".join("%02d" % x for x in last_draw),
             json.load(open(os.path.join(PROJECT_ROOT, "ssq_analysis.json"), encoding="utf-8"))["latest"]["blue_ball"]))
    print("  候选组合: %d 组" % len(all_combos))

    # ==================== 第二步：统计评分 ====================
    print_banner("第二步：构建历史特征分布 + 统计评分")

    dist = HistoricalDistributions(history_data)
    scorer = CombinationScorer(dist, last_draw, hot_set, cold_set, blue_missing)

    # 验证
    latest_blue = int(json.load(open(os.path.join(PROJECT_ROOT, "ssq_analysis.json"), encoding="utf-8"))["latest"]["blue_ball"])
    latest_score, _ = scorer.score(last_draw, latest_blue)
    abnormal_score, _ = scorer.score([1, 2, 3, 4, 5, 6], 1)
    print("  分布验证: 和值峰值=%d, AC4-8=%.1f%%, 奇偶3:3=%.1f%%"
          % (dist.sum_bins[np.argmax(dist.sum_hist)],
             sum(dist.ac_hist[4:9]) / dist.ac_hist.sum() * 100,
             dist.odd_even_hist.get((3, 3), 0) / sum(dist.odd_even_hist.values()) * 100))
    print("  上期开奖得分: %.1f | 异常组合(01-06)得分: %.1f" % (latest_score, abnormal_score))

    print("  正在对 %d 组合评分..." % len(all_combos))
    scored = scorer.rank_combinations(all_combos)
    print("  最高分: %.1f | 中位数: %.1f | 最低分: %.1f"
          % (scored[0][2], scored[len(scored) // 2][2], scored[-1][2]))

    # 输出统计评分报告
    scored_path = COMBO_SCORE_TXT
    format_scored_report(scored[:args.top_n], scored_path)
    print("  已输出: %s" % scored_path)

    # ==================== 第三步：GA 优化 ====================
    ga_top = []
    ga_history = []

    if not args.no_ga:
        print_banner("第三步：遗传算法优化")
        print("  种群=%d 代数=%d 精英=%d 交叉=0.85 变异=0.12"
              % (POP_SIZE, GENERATIONS, int(POP_SIZE * 0.05)))

        seed_combos = [(r, b) for r, b, _, _ in scored[:50]]
        ga = GeneticOptimizer(scorer, seed_combos, all_combos)
        ga_top, ga_history = ga.evolve(verbose=True)

        # 标记新发现
        candidate_set = set((frozenset(r), b) for r, b in all_combos)
        new_count = 0
        for chrom in ga_top:
            chrom._is_new = (frozenset(chrom.reds), chrom.blue) not in candidate_set
            if chrom._is_new:
                new_count += 1
        print("  Top %d 中有 %d 个候选集外新发现" % (len(ga_top), new_count))

        # 输出 GA 报告
        ga_path = COMBO_GA_TXT
        format_ga_report(ga_top, ga_history, ga_path)
        print("  已输出: %s" % ga_path)
    else:
        print_banner("第三步：遗传算法优化（已跳过）")

    # ==================== 第四步：LLM 终评 ====================
    print_banner("第四步：LLM 终评分析")

    if not args.llm:
        print("  已跳过（默认关闭，需传 --llm 启用；--no-ga 模式下也无 GA 候选可送 LLM）")
        # 降级报告
        from llm_analyze import _fallback_report
        llm_report = _fallback_report(scored, ga_top)
    else:
        llm_report = llm_analyze(
            model_type=args.model,
            api_key=args.api_key,
            base_url=args.base_url,
            scored_top=scored[:10],
            ga_top=ga_top[:5] if ga_top else [],
            project_root=PROJECT_ROOT,
        )

    # ==================== 第五步：生成综合报告 ====================
    print_banner("第五步：生成综合报告")

    final_report = compose_final_report(
        json.load(open(os.path.join(PROJECT_ROOT, "ssq_analysis.json"), encoding="utf-8")),
        scored[:args.top_n], ga_top[:20] if ga_top else [], ga_history, llm_report,
    )

    # 输出
    if args.output:
        output_path = args.output
    else:
        output_path = combo_final_txt()

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(final_report + "\n")

    elapsed = time.time() - start_time
    print("  报告已保存: %s" % output_path)
    print("  总耗时: %.1fs" % elapsed)

    # 预览
    print("\n" + "=" * 60)
    print("  最终推荐 Top 5 预览:")
    print("=" * 60)
    for rank, (reds, blue, score, bd) in enumerate(scored[:5], 1):
        red_str = " ".join("%02d" % r for r in reds)
        print("  第%d注：%s + %02d  (得分:%.1f, 和值:%d, AC:%d, 奇偶:%d:%d)"
              % (rank, red_str, blue, score, bd["sum"][0], bd["ac"][0],
                 bd["odd_even"][0][0], bd["odd_even"][0][1]))

    if ga_top:
        print("\n  GA 优化最佳（候选集外新发现）:")
        for rank, chrom in enumerate(ga_top[:3], 1):
            if getattr(chrom, "_is_new", False):
                red_str = " ".join("%02d" % r for r in chrom.reds)
                print("  第%d注：%s + %02d  (适应度:%.1f) ★新发现"
                      % (rank, red_str, chrom.blue, chrom.fitness))


if __name__ == "__main__":
    main()

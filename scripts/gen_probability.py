# -*- coding: utf-8 -*-
"""
双色球号码出现概率统计
从 最新的双色球全部号码汇总_YYYYMMDD.xlsx 统计每个红球(1-33)和蓝球(1-16)的出现概率

两个维度：
  1. 组级出现率 = 包含该号码的复式组数 / 总组数
     （该号码在26组复式中有多少组选了它）
  2. 组合级出现率 = 包含该号码的展开组合数 / 总组合数
     （展开为6+1单式后，该号码在多少个组合中出现）
"""
import os
import sys
from itertools import combinations
from collections import defaultdict

import pandas as pd

# 导入全局配置
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import (
    get_summary_xlsx_or_exit, PROBABILITY_TXT
)

# ===================== 路径配置 =====================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
EXCEL_PATH = get_summary_xlsx_or_exit()   # 自动定位最新汇总 xlsx
OUTPUT_PATH = PROBABILITY_TXT

# 红球全选阈值：红球数 >= 此值的组跳过展开（组合爆炸）
FULL_SET_THRESHOLD = 20


def parse_nums(cell):
    if pd.isna(cell) or str(cell).strip() == "":
        return []
    return [int(p.strip()) for p in str(cell).split(",") if p.strip().isdigit()]


def load_groups():
    df = pd.read_excel(EXCEL_PATH)
    groups = []
    for _, row in df.iterrows():
        red = parse_nums(row.get("红球", ""))
        blue = parse_nums(row.get("蓝球", ""))
        img = str(row.get("图片文件名", ""))
        if red and blue:
            groups.append({
                "red": set(red),
                "blue": set(blue),
                "img": img,
                "red_count": len(red),
                "blue_count": len(blue),
            })
    return groups


def main():
    groups = load_groups()
    total_groups = len(groups)

    # ===================== 1. 组级出现率 =====================
    red_group_count = defaultdict(int)  # 红球 → 出现在多少组
    blue_group_count = defaultdict(int)  # 蓝球 → 出现在多少组

    for g in groups:
        for r in g["red"]:
            red_group_count[r] += 1
        for b in g["blue"]:
            blue_group_count[b] += 1

    # ===================== 2. 展开为 6+1 单式组合 =====================
    red_combo_count = defaultdict(int)  # 红球 → 出现在多少个展开组合
    blue_combo_count = defaultdict(int)  # 蓝球 → 出现在多少个展开组合
    total_combos = 0
    unique_combos = set()

    for g in groups:
        red_list = sorted(g["red"])
        blue_list = sorted(g["blue"])

        if len(red_list) >= FULL_SET_THRESHOLD:
            continue  # 跳过全选组

        for red_combo in combinations(red_list, 6):
            for blue in blue_list:
                key = (frozenset(red_combo), blue)
                if key not in unique_combos:
                    unique_combos.add(key)
                    for r in red_combo:
                        red_combo_count[r] += 1
                    blue_combo_count[blue] += 1
                    total_combos += 1

    # ===================== 3. 输出到 txt =====================
    lines = []
    lines.append("=" * 70)
    lines.append("         双色球号码出现概率统计报告")
    lines.append("=" * 70)
    lines.append("")
    lines.append(f"【数据来源】 {os.path.basename(EXCEL_PATH)}")
    lines.append(f"【复式组数】 {total_groups} 组")
    lines.append(f"【展开组合】 {total_combos} 组（跳过全选组红球≥{FULL_SET_THRESHOLD}）")
    lines.append("")
    lines.append("  组级出现率   = 包含该号码的复式组数 / {0}组".format(total_groups))
    lines.append("  组合级出现率 = 包含该号码的展开组合数 / {0}组".format(total_combos))
    lines.append("  理论概率     = 单注中该号码的概率（红球6/33=18.2%，蓝球1/16=6.3%）")
    lines.append("")

    # --- 红球概率 ---
    lines.append("─" * 70)
    lines.append("【红球 1-33 出现概率】")
    lines.append("─" * 70)
    lines.append("")
    lines.append(f"  号码  组级出现率          组合级出现率         理论概率    偏差")
    lines.append(f"  ──── ────────────────── ────────────────── ──────── ────────")

    # 按组级出现率降序排列
    red_sorted = sorted(range(1, 34), key=lambda r: -red_group_count.get(r, 0))

    for r in red_sorted:
        g_count = red_group_count.get(r, 0)
        g_pct = g_count / total_groups * 100

        c_count = red_combo_count.get(r, 0)
        c_pct = c_count / total_combos * 100 if total_combos > 0 else 0

        theory = 6 / 33 * 100  # 18.18%
        deviation = c_pct - theory

        # 可视化条
        bar_len = int(g_pct / 2)
        bar = "█" * bar_len

        lines.append(
            f"  {r:02d}   {g_count:2d}/{total_groups} = {g_pct:5.1f}% {bar:<22s}"
            f"  {c_count:4d}/{total_combos} = {c_pct:5.1f}%   "
            f"{theory:5.1f}%   {deviation:+5.1f}%"
        )

    # 红球统计摘要
    lines.append("")
    lines.append("  ── 红球统计摘要 ──")
    red_pcts = [red_group_count.get(r, 0) / total_groups * 100 for r in range(1, 34)]
    combo_pcts = [red_combo_count.get(r, 0) / total_combos * 100 if total_combos > 0 else 0 for r in range(1, 34)]
    lines.append(f"  组级出现率  最高: {max(red_pcts):.1f}%  最低: {min(red_pcts):.1f}%  "
                 f"均值: {sum(red_pcts)/len(red_pcts):.1f}%")
    lines.append(f"  组合级出现率 最高: {max(combo_pcts):.1f}%  最低: {min(combo_pcts):.1f}%  "
                 f"均值: {sum(combo_pcts)/len(combo_pcts):.1f}%")
    lines.append(f"  理论概率: {6/33*100:.1f}%（每注选6个红球/共33个）")

    # 红球热度分级
    hot_reds = [r for r in range(1, 34) if red_group_count.get(r, 0) / total_groups >= 0.35]
    warm_reds = [r for r in range(1, 34) if 0.25 <= red_group_count.get(r, 0) / total_groups < 0.35]
    cold_reds = [r for r in range(1, 34) if red_group_count.get(r, 0) / total_groups < 0.25]
    lines.append("")
    lines.append(f"  热号(≥35%): {' '.join(f'{r:02d}' for r in hot_reds) if hot_reds else '无'}")
    lines.append(f"  温号(25-35%): {' '.join(f'{r:02d}' for r in warm_reds) if warm_reds else '无'}")
    lines.append(f"  冷号(<25%): {' '.join(f'{r:02d}' for r in cold_reds) if cold_reds else '无'}")
    lines.append("")

    # --- 蓝球概率 ---
    lines.append("─" * 70)
    lines.append("【蓝球 1-16 出现概率】")
    lines.append("─" * 70)
    lines.append("")
    lines.append(f"  号码  组级出现率          组合级出现率         理论概率    偏差")
    lines.append(f"  ──── ────────────────── ────────────────── ──────── ────────")

    blue_sorted = sorted(range(1, 17), key=lambda b: -blue_group_count.get(b, 0))

    for b in blue_sorted:
        g_count = blue_group_count.get(b, 0)
        g_pct = g_count / total_groups * 100

        c_count = blue_combo_count.get(b, 0)
        c_pct = c_count / total_combos * 100 if total_combos > 0 else 0

        theory = 1 / 16 * 100  # 6.25%
        deviation = c_pct - theory

        bar_len = int(g_pct / 2)
        bar = "█" * bar_len

        lines.append(
            f"  {b:02d}   {g_count:2d}/{total_groups} = {g_pct:5.1f}% {bar:<22s}"
            f"  {c_count:4d}/{total_combos} = {c_pct:5.1f}%   "
            f"{theory:5.1f}%   {deviation:+5.1f}%"
        )

    # 蓝球统计摘要
    lines.append("")
    lines.append("  ── 蓝球统计摘要 ──")
    blue_pcts = [blue_group_count.get(b, 0) / total_groups * 100 for b in range(1, 17)]
    blue_combo_pcts = [blue_combo_count.get(b, 0) / total_combos * 100 if total_combos > 0 else 0 for b in range(1, 17)]
    lines.append(f"  组级出现率  最高: {max(blue_pcts):.1f}%  最低: {min(blue_pcts):.1f}%  "
                 f"均值: {sum(blue_pcts)/len(blue_pcts):.1f}%")
    lines.append(f"  组合级出现率 最高: {max(blue_combo_pcts):.1f}%  最低: {min(blue_combo_pcts):.1f}%  "
                 f"均值: {sum(blue_combo_pcts)/len(blue_combo_pcts):.1f}%")
    lines.append(f"  理论概率: {1/16*100:.1f}%（每注选1个蓝球/共16个）")

    # 蓝球热度分级
    hot_blues = [b for b in range(1, 17) if blue_group_count.get(b, 0) / total_groups >= 0.20]
    warm_blues = [b for b in range(1, 17) if 0.10 <= blue_group_count.get(b, 0) / total_groups < 0.20]
    cold_blues = [b for b in range(1, 17) if blue_group_count.get(b, 0) / total_groups < 0.10]
    lines.append("")
    lines.append(f"  热号(≥20%): {' '.join(f'{b:02d}' for b in hot_blues) if hot_blues else '无'}")
    lines.append(f"  温号(10-20%): {' '.join(f'{b:02d}' for b in warm_blues) if warm_blues else '无'}")
    lines.append(f"  冷号(<10%): {' '.join(f'{b:02d}' for b in cold_blues) if cold_blues else '无'}")
    lines.append("")

    lines.append("─" * 70)
    lines.append("  说明：组级出现率反映彩民对该号码的偏好程度（人气指标），")
    lines.append("        组合级出现率反映该号码在所有展开组合中的实际覆盖比例。")
    lines.append("        偏差为正表示该号码被高估（过热），偏差为负表示被低估（偏冷）。")
    lines.append("─" * 70)

    output_text = "\n".join(lines)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        f.write(output_text + "\n")

    print(f"已输出: {OUTPUT_PATH}")
    print()
    print(output_text)


if __name__ == "__main__":
    main()

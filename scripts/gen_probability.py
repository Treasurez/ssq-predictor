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
import json
from itertools import combinations
from collections import defaultdict, Counter

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


# ===================== 蓝球综合评分（4 维度）=====================
# ① 历史频率分 40% - 近 100 期出现率（反映真实热度）
# ② 遗漏回补分 30% - 遗漏期数越长越可能回补
# ③ 复式人气分 15% - 复式出现率（彩民群体偏好）
# ④ 偏差修正分 15% - 历史频率 - 复式人气（被低估加分）

BLUE_HISTORY_WINDOW = 100  # 蓝球历史分析窗口（近 N 期）


def load_history():
    """加载历史开奖数据用于蓝球分析，返回 None 表示无历史数据"""
    history_path = os.path.join(os.path.dirname(SCRIPT_DIR), "ssq_history.json")
    if not os.path.exists(history_path):
        return None
    try:
        with open(history_path, "r", encoding="utf-8") as f:
            history = json.load(f)
        history.sort(key=lambda x: x["issue"])
        for item in history:
            item["blue_ball"] = int(item["blue_ball"])
        return history
    except Exception:
        return None


def analyze_blue_history(history, window=BLUE_HISTORY_WINDOW):
    """分析蓝球历史规律：频率、遗漏、区间、奇偶、大小"""
    n = min(window, len(history))
    recent = history[-n:]

    # 近 N 期频率
    freq = Counter(item["blue_ball"] for item in recent)
    freq_pct = {b: freq.get(b, 0) / n * 100 for b in range(1, 17)}

    # 遗漏期数（截至最新一期）
    gap = {}
    for b in range(1, 17):
        g = 0
        for item in reversed(history):
            if item["blue_ball"] == b:
                break
            g += 1
        gap[b] = g

    # 区间分布
    zones_def = [(1, 4), (5, 8), (9, 12), (13, 16)]
    zones = {}
    for lo, hi in zones_def:
        cnt = sum(1 for item in recent if lo <= item["blue_ball"] <= hi)
        zones[f"{lo:02d}-{hi:02d}"] = cnt / n * 100

    # 奇偶/大小
    odd_pct = sum(1 for item in recent if item["blue_ball"] % 2 == 1) / n * 100
    big_pct = sum(1 for item in recent if item["blue_ball"] >= 9) / n * 100

    return {
        "window": n,
        "total": len(history),
        "freq": freq,
        "freq_pct": freq_pct,
        "gap": gap,
        "zones": zones,
        "odd_pct": odd_pct,
        "big_pct": big_pct,
    }


def blue_comprehensive_score(b, hist_info, comp_pct):
    """计算蓝球综合评分（4 维度加权）

    返回 (总分 0-100, 各维度详情 dict)
    """
    if hist_info is None:
        # 无历史数据时退化为纯复式人气
        comp_score = min(comp_pct / 16 * 100, 100)
        return comp_score, {
            "freq_pct": 0, "gap": 0, "comp_pct": comp_pct, "dev": 0,
            "freq_score": 0, "gap_score": 0, "comp_score": comp_score, "dev_score": 50,
        }

    # ① 历史频率分（0-100）：近 100 期出现率，理论 6.25%，最高约 12%
    freq_pct = hist_info["freq_pct"].get(b, 0)
    freq_score = min(freq_pct / 12 * 100, 100)

    # ② 遗漏回补分（0-100）：遗漏 0 期=0 分，遗漏 50 期=100 分
    gap = hist_info["gap"].get(b, 0)
    gap_score = min(gap / 50 * 100, 100)

    # ③ 复式人气分（0-100）：复式出现率，最高约 16%
    comp_score = min(comp_pct / 16 * 100, 100)

    # ④ 偏差修正分（0-100）：历史频率 - 复式人气，正偏差（被低估）加分
    dev = freq_pct - comp_pct
    dev_score = max(0, min((dev + 10) / 20 * 100, 100))

    total = freq_score * 0.4 + gap_score * 0.3 + comp_score * 0.15 + dev_score * 0.15

    return total, {
        "freq_pct": freq_pct,
        "gap": gap,
        "comp_pct": comp_pct,
        "dev": dev,
        "freq_score": freq_score,
        "gap_score": gap_score,
        "comp_score": comp_score,
        "dev_score": dev_score,
    }


def blue_eval_label(b, info):
    """根据各维度分布给出简短评价"""
    labels = []
    if info["freq_pct"] >= 9:
        labels.append("近期热")
    elif info["freq_pct"] <= 3:
        labels.append("近期冷")
    if info["gap"] >= 40:
        labels.append("超长遗漏待回补")
    elif info["gap"] >= 20:
        labels.append("长遗漏待回补")
    if info["dev"] >= 3:
        labels.append("被彩民低估")
    elif info["dev"] <= -3:
        labels.append("被彩民高估")
    if info["comp_pct"] >= 12:
        labels.append("人气高")
    elif info["comp_pct"] <= 2:
        labels.append("人气低")
    return " + ".join(labels) if labels else "正常"


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

    # --- 蓝球综合评分（历史开奖 + 复式人气 双数据源融合）---
    history = load_history()
    hist_info = analyze_blue_history(history) if history else None

    lines.append("─" * 70)
    lines.append("【蓝球 1-16 综合评分】（历史开奖 + 复式人气 双数据源融合）")
    lines.append("─" * 70)
    lines.append("")

    if hist_info:
        lines.append(f"数据源:")
        lines.append(f"  ① 历史开奖: {hist_info['total']} 期（分析近 {hist_info['window']} 期）")
        lines.append(f"  ② 复式人气: {total_groups} 组复式")
        lines.append("")
        lines.append(f"评分维度（权重）:")
        lines.append(f"  ① 历史频率分 40% - 近 {hist_info['window']} 期出现率（反映真实热度）")
        lines.append(f"  ② 遗漏回补分 30% - 遗漏期数越长越可能回补")
        lines.append(f"  ③ 复式人气分 15% - 复式出现率（彩民群体偏好）")
        lines.append(f"  ④ 偏差修正分 15% - 历史频率 - 复式人气（被低估加分）")
    else:
        lines.append(f"⚠ 未找到 ssq_history.json，仅使用复式人气评分")
        lines.append(f"评分维度:")
        lines.append(f"  ③ 复式人气分 100% - 复式出现率（无历史数据）")
    lines.append("")

    # 计算所有蓝球综合评分
    blue_scores = {}
    for b in range(1, 17):
        comp_pct = blue_group_count.get(b, 0) / total_groups * 100
        total_score, info = blue_comprehensive_score(b, hist_info, comp_pct)
        info["total_score"] = total_score
        info["combo_count"] = blue_combo_count.get(b, 0)
        info["combo_pct"] = blue_combo_count.get(b, 0) / total_combos * 100 if total_combos > 0 else 0
        blue_scores[b] = info

    # 按综合分降序
    blue_ranked = sorted(range(1, 17), key=lambda b: -blue_scores[b]["total_score"])

    # 综合评分表
    lines.append(f"  排名 蓝球  综合分 ┃ 历史频率  遗漏  复式人气  偏差   ┃ 评价")
    lines.append(f"  ──── ──── ─────── ┃ ──────── ───── ──────── ────── ┃ ──────────────")
    for rank, b in enumerate(blue_ranked, 1):
        s = blue_scores[b]
        gap_str = f"{s['gap']:3d}期" if hist_info else "  - "
        hist_str = f"{s['freq_pct']:5.1f}%" if hist_info else "  -   "
        lines.append(
            f"  #{rank:02d}  {b:02d}   {s['total_score']:5.1f} ┃ "
            f"{hist_str}  {gap_str}  {s['comp_pct']:5.1f}%  {s['dev']:+5.1f}% ┃ "
            f"{blue_eval_label(b, s)}"
        )

    # 各维度分项（帮助理解评分来源）
    lines.append("")
    lines.append(f"  ── 各维度分项（0-100）──")
    lines.append(f"  蓝球  综合  ┃ 历史频率 遗漏回补 复式人气 偏差修正")
    lines.append(f"  ──── ───── ┃ ──────── ──────── ──────── ────────")
    for b in blue_ranked:
        s = blue_scores[b]
        lines.append(
            f"  {b:02d}   {s['total_score']:5.1f} ┃ "
            f"  {s['freq_score']:5.1f}   {s['gap_score']:5.1f}   {s['comp_score']:5.1f}   {s['dev_score']:5.1f}"
        )

    # 历史规律分析
    if hist_info:
        lines.append("")
        lines.append(f"  ── 蓝球历史规律分析（近 {hist_info['window']} 期）──")
        lines.append(f"  奇偶: 奇 {hist_info['odd_pct']:.1f}% | 偶 {100-hist_info['odd_pct']:.1f}% (理论各 50%)")
        lines.append(f"  大小: 小(1-8) {100-hist_info['big_pct']:.1f}% | 大(9-16) {hist_info['big_pct']:.1f}% (理论各 50%)")
        lines.append(f"  区间分布:")
        for zone, pct in hist_info["zones"].items():
            bar_len = int(pct / 2)
            bar = "█" * bar_len
            lines.append(f"    {zone}: {pct:5.1f}% {bar} (理论 25%)")

        # 遗漏期数表
        lines.append("")
        lines.append(f"  ── 蓝球遗漏期数表（截至最新一期）──")
        gap_sorted = sorted(range(1, 17), key=lambda b: -hist_info["gap"][b])
        for b in gap_sorted:
            gap = hist_info["gap"][b]
            bar_len = min(int(gap / 2), 40)
            bar = "▁" * bar_len
            tag = ""
            if gap >= 40:
                tag = " ←超长遗漏"
            elif gap >= 20:
                tag = " ←长遗漏"
            lines.append(f"    {b:02d}: 遗漏 {gap:3d} 期 {bar}{tag}")

    # 复式人气原始统计（保留作为参考）
    lines.append("")
    lines.append(f"  ── 蓝球复式人气原始统计（{total_groups} 组）──")
    lines.append(f"  蓝球  组级出现率          组合级出现率         理论概率    偏差")
    lines.append(f"  ──── ────────────────── ────────────────── ──────── ────────")
    for b in sorted(range(1, 17), key=lambda b: -blue_group_count.get(b, 0)):
        g_count = blue_group_count.get(b, 0)
        g_pct = g_count / total_groups * 100
        c_count = blue_combo_count.get(b, 0)
        c_pct = c_count / total_combos * 100 if total_combos > 0 else 0
        theory = 1 / 16 * 100
        deviation = c_pct - theory
        bar_len = int(g_pct / 2)
        bar = "█" * bar_len
        lines.append(
            f"  {b:02d}   {g_count:2d}/{total_groups} = {g_pct:5.1f}% {bar:<22s}"
            f"  {c_count:4d}/{total_combos} = {c_pct:5.1f}%   "
            f"{theory:5.1f}%   {deviation:+5.1f}%"
        )

    # Top 5 综合推荐
    lines.append("")
    lines.append(f"  ══ 蓝球综合推荐 Top 5 ══")
    for rank, b in enumerate(blue_ranked[:5], 1):
        s = blue_scores[b]
        lines.append(f"  #{rank}  蓝球 {b:02d}  评分 {s['total_score']:.1f}  - {blue_eval_label(b, s)}")
    lines.append("")

    lines.append("─" * 70)
    lines.append("  说明：红球部分 - 组级/组合级出现率均基于复式人气统计。")
    lines.append("        蓝球部分 - 采用 4 维度综合评分（历史频率 40% + 遗漏回补 30% +")
    lines.append("                   复式人气 15% + 偏差修正 15%），融合历史开奖与复式人气。")
    lines.append("        偏差 = 历史频率 - 复式人气，正值表示被彩民低估，负值表示被高估。")
    lines.append("─" * 70)

    output_text = "\n".join(lines)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        f.write(output_text + "\n")

    print(f"已输出: {OUTPUT_PATH}")
    print()
    print(output_text)


if __name__ == "__main__":
    main()
    #09 11 12 25 30 33 11

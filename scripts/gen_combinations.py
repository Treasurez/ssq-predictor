# -*- coding: utf-8 -*-
"""
双色球复式组合展开器
1. 从 双色球汇总.xlsx 读取 26 组复式号码
2. 展开所有可能的 6红+1蓝 单式组合
3. 去除历史开奖重复组合
4. 按出席率（被多少组复式覆盖）排序输出
"""
import json
import os
from itertools import combinations
from collections import defaultdict

import pandas as pd

# ===================== 路径配置 =====================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXCEL_PATH = os.path.join(os.path.dirname(__file__), "双色球汇总.xlsx")
HISTORY_PATH = os.path.join(BASE_DIR, "ssq_history.json")
OUTPUT_PATH = os.path.join(os.path.dirname(__file__), "双色球组合_出席率排序.txt")

# 红球全选阈值：红球数 >= 此值的组视为"全选"，不具筛选意义，跳过展开
FULL_SET_THRESHOLD = 20


# ===================== 1. 读取复式数据 =====================
def load_groups():
    """从 Excel 读取复式分组，返回 list of dict(red=set, blue=set, img=str)"""
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


def parse_nums(cell):
    if pd.isna(cell) or str(cell).strip() == "":
        return []
    return [int(p.strip()) for p in str(cell).split(",") if p.strip().isdigit()]


# ===================== 2. 读取历史开奖 =====================
def load_history():
    """读取历史开奖，返回 set of (frozenset(red), blue) 用于快速查重"""
    with open(HISTORY_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    history_set = set()
    for item in data:
        red = frozenset(int(x) for x in item["red_balls"])
        blue = int(item["blue_ball"])
        history_set.add((red, blue))
    return history_set, len(data)


# ===================== 3. 统计号码出席率 =====================
def calc_number_freq(groups):
    """统计每个红球/蓝球在多少组复式中出现"""
    red_freq = defaultdict(int)
    blue_freq = defaultdict(int)
    total = len(groups)
    for g in groups:
        for r in g["red"]:
            red_freq[r] += 1
        for b in g["blue"]:
            blue_freq[b] += 1
    return red_freq, blue_freq, total


# ===================== 4. 展开复式 → 6+1 单式组合 =====================
def expand_combinations(groups):
    """
    展开所有复式组为 6红+1蓝 单式组合。
    返回 dict: (frozenset(red_6), blue) -> set(支持的组索引)

    跳过红球数 >= FULL_SET_THRESHOLD 的组（全选组无筛选意义，且组合爆炸）。
    """
    combo_support = defaultdict(set)  # combo -> set of group indices
    skipped_groups = []
    used_groups = []

    for idx, g in enumerate(groups):
        red_list = sorted(g["red"])
        blue_list = sorted(g["blue"])

        if len(red_list) >= FULL_SET_THRESHOLD:
            skipped_groups.append((idx, g))
            continue

        used_groups.append((idx, g))

        # 生成所有 C(n,6) 红球组合 × 每个蓝球
        for red_combo in combinations(red_list, 6):
            red_fs = frozenset(red_combo)
            for blue in blue_list:
                key = (red_fs, blue)
                combo_support[key].add(idx)

    return combo_support, used_groups, skipped_groups


# ===================== 5. 主流程 =====================
def main():
    # --- 加载数据 ---
    groups = load_groups()
    history_set, history_count = load_history()
    red_freq, blue_freq, total_groups = calc_number_freq(groups)

    print(f"复式分组: {len(groups)} 组")
    print(f"历史开奖: {history_count} 条")

    # --- 号码出席率 ---
    print(f"\n红球出席率 Top10:")
    for r, cnt in sorted(red_freq.items(), key=lambda x: -x[1])[:10]:
        print(f"  {r:02d}: {cnt}/{total_groups} = {cnt/total_groups*100:.1f}%")
    print(f"\n蓝球出席率:")
    for b, cnt in sorted(blue_freq.items(), key=lambda x: -x[1]):
        print(f"  {b:02d}: {cnt}/{total_groups} = {cnt/total_groups*100:.1f}%")

    # --- 展开组合 ---
    combo_support, used_groups, skipped_groups = expand_combinations(groups)
    total_combos = len(combo_support)

    print(f"\n展开组合:")
    print(f"  参与展开的组: {len(used_groups)} 组")
    if skipped_groups:
        print(f"  跳过全选组(红球≥{FULL_SET_THRESHOLD}): {len(skipped_groups)} 组")
        for idx, g in skipped_groups:
            print(f"    - 组{idx+1} ({g['img']}): {g['red_count']}红 {g['blue_count']}蓝")
    print(f"  去重后唯一组合数: {total_combos}")

    # --- 去除历史重复 ---
    history_hits = []
    unique_combos = {}
    for combo, supporters in combo_support.items():
        if combo in history_set:
            history_hits.append((combo, supporters))
        else:
            unique_combos[combo] = supporters

    print(f"  命中历史开奖: {len(history_hits)} 组（已去除）")
    print(f"  最终有效组合: {len(unique_combos)} 组")

    # --- 排序：出席率(支持组数)降序 → 号码出席率均值降序 → 组合本身 ---
    def combo_score(item):
        combo, supporters = item
        red_fs, blue = combo
        # 号码出席率均值
        red_scores = [red_freq.get(r, 0) / total_groups for r in red_fs]
        blue_score = blue_freq.get(blue, 0) / total_groups
        avg_score = sum(red_scores) / len(red_scores)
        return (-len(supporters), -avg_score, -blue_score, sorted(red_fs))

    sorted_combos = sorted(unique_combos.items(), key=combo_score)

    # --- 输出到 txt ---
    lines = []
    lines.append("=" * 70)
    lines.append("       双色球 6+1 组合 · 出席率排序报告")
    lines.append("=" * 70)
    lines.append("")
    lines.append(f"【数据来源】")
    lines.append(f"  复式分组: {len(groups)} 组（来自 双色球汇总.xlsx）")
    lines.append(f"  历史开奖: {history_count} 条（来自 ssq_history.json）")
    lines.append(f"  参与展开: {len(used_groups)} 组")
    if skipped_groups:
        lines.append(f"  跳过全选组(红球≥{FULL_SET_THRESHOLD}): {len(skipped_groups)} 组"
                     f"（组合爆炸，无筛选意义）")
    lines.append(f"  唯一组合: {total_combos} 组")
    lines.append(f"  命中历史: {len(history_hits)} 组（已去除）")
    lines.append(f"  有效输出: {len(unique_combos)} 组")
    lines.append("")

    # --- 红球出席率 ---
    lines.append("─" * 70)
    lines.append("【红球出席率】（在多少组复式中出现）")
    lines.append("─" * 70)
    for r in range(1, 34):
        cnt = red_freq.get(r, 0)
        pct = cnt / total_groups * 100
        bar = "█" * int(pct / 5)
        lines.append(f"  {r:02d}: {cnt:2d}/{total_groups} = {pct:5.1f}% {bar}")
    lines.append("")

    # --- 蓝球出席率 ---
    lines.append("─" * 70)
    lines.append("【蓝球出席率】")
    lines.append("─" * 70)
    for b in range(1, 17):
        cnt = blue_freq.get(b, 0)
        pct = cnt / total_groups * 100
        bar = "█" * int(pct / 5)
        lines.append(f"  {b:02d}: {cnt:2d}/{total_groups} = {pct:5.1f}% {bar}")
    lines.append("")

    # --- 组合列表 ---
    lines.append("─" * 70)
    lines.append("【6+1 组合列表 · 按出席率排序】")
    lines.append("  出席率 = 该组合被多少组复式覆盖 / 总组数")
    lines.append("  号码热度 = 6个红球+1蓝球的平均出席率")
    lines.append("─" * 70)
    lines.append("")

    max_support = max(len(s) for s in unique_combos.values()) if unique_combos else 1

    for rank, (combo, supporters) in enumerate(sorted_combos, 1):
        red_fs, blue = combo
        red_sorted = sorted(red_fs)
        red_str = " ".join(f"{r:02d}" for r in red_sorted)
        blue_str = f"{blue:02d}"

        support_count = len(supporters)
        attendance_pct = support_count / total_groups * 100

        # 号码热度
        red_scores = [red_freq.get(r, 0) / total_groups for r in red_sorted]
        blue_score = blue_freq.get(blue, 0) / total_groups
        avg_hot = (sum(red_scores) + blue_score) / 7 * 100

        # 支持组来源
        support_imgs = [groups[i]["img"] for i in sorted(supporters)]
        # 去重图片名
        support_imgs_short = []
        seen = set()
        for img in support_imgs:
            if img not in seen:
                seen.add(img)
                support_imgs_short.append(img)

        lines.append(
            f"#{rank:04d}  {red_str} + {blue_str}  "
            f"| 出席率: {support_count}/{total_groups} = {attendance_pct:5.1f}%  "
            f"| 热度: {avg_hot:5.1f}%  "
            f"| 来源: {', '.join(support_imgs_short)}"
        )

    lines.append("")
    lines.append("─" * 70)
    lines.append(f"  共 {len(unique_combos)} 组有效组合（已去除 {len(history_hits)} 组历史重复）")
    lines.append("─" * 70)

    output_text = "\n".join(lines)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        f.write(output_text + "\n")

    print(f"\n已输出: {OUTPUT_PATH}")
    print(f"  有效组合: {len(unique_combos)} 组")
    print(f"  最高出席率: {max_support}/{total_groups} = {max_support/total_groups*100:.1f}%")

    # 打印前20条预览
    print(f"\n{'='*70}")
    print("前20条预览:")
    print(f"{'='*70}")
    for rank, (combo, supporters) in enumerate(sorted_combos[:20], 1):
        red_fs, blue = combo
        red_sorted = sorted(red_fs)
        red_str = " ".join(f"{r:02d}" for r in red_sorted)
        support_count = len(supporters)
        attendance_pct = support_count / total_groups * 100
        red_scores = [red_freq.get(r, 0) / total_groups for r in red_sorted]
        blue_score = blue_freq.get(blue, 0) / total_groups
        avg_hot = (sum(red_scores) + blue_score) / 7 * 100
        print(f"  #{rank:04d}  {red_str} + {blue:02d}  "
              f"| 出席率: {support_count}/{total_groups}={attendance_pct:.1f}%  "
              f"| 热度: {avg_hot:.1f}%")


if __name__ == "__main__":
    main()

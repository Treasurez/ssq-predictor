# -*- coding: utf-8 -*-
"""
双色球复式组合展开器
1. 从 最新的双色球全部号码汇总_YYYYMMDD.xlsx 读取复式号码
2. 展开所有可能的 6红+1蓝 单式组合
3. 去除历史开奖重复组合
4. 按出席率（被多少组复式覆盖）排序输出
"""
import json
import os
import sys
from itertools import combinations
from collections import defaultdict

import pandas as pd

# 导入全局配置
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import (
    get_summary_xlsx_or_exit, COMBO_ATTEND_TXT
)

# ===================== 路径配置 =====================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXCEL_PATH = get_summary_xlsx_or_exit()   # 自动定位最新汇总 xlsx
HISTORY_PATH = os.path.join(BASE_DIR, "ssq_history.json")
OUTPUT_PATH = COMBO_ATTEND_TXT

# 红球全选阈值：红球数 >= 此值的组视为"全选"，不具筛选意义，跳过展开
FULL_SET_THRESHOLD = 20

# 冷热号分级阈值（与 gen_probability.py 保持一致）
# 热: 组级出现率 >= 35%
# 温: 25% <= 组级出现率 < 35%
# 冷: 组级出现率 < 25%
HOT_THRESHOLD = 0.35
WARM_THRESHOLD = 0.25

# 冷号注入参数
COLD_INJECT_TOP_N = 500       # 从 Top N 热温组合生成冷号变体
COLD_INJECT_MAX = 2000        # 注入组合总数上限
COLD_INJECT_SEED = 42         # 随机种子（可复现）

# 算法注入组合的虚拟支持标记（用 -1 表示"算法注入"，非真实复式组）
INJECTED_MARKER = -1

# 冷号平衡加分：1-2 个冷号的组合获得此加分（叠加到号码热度均值上）
# 让 "5 热 1 冷" / "4 热 2 冷" 的混合结构能在 Top 推荐中占据合理比例
COLD_BALANCE_BONUS = 0.15


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


def classify_numbers(red_freq, total_groups):
    """根据红球组级出现率划分热/温/冷三类

    返回 (hot_set, warm_set, cold_set)，每个是红球号码的 set。
    分级阈值与 gen_probability.py 保持一致：
      热: >= 35%
      温: 25% ~ 35%
      冷: < 25%
    """
    hot, warm, cold = set(), set(), set()
    for r in range(1, 34):
        rate = red_freq.get(r, 0) / total_groups if total_groups > 0 else 0
        if rate >= HOT_THRESHOLD:
            hot.add(r)
        elif rate >= WARM_THRESHOLD:
            warm.add(r)
        else:
            cold.add(r)
    return hot, warm, cold


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


# ===================== 4.5 冷号注入：生成含冷号的变体组合 =====================
def inject_cold_combinations(unique_combos, hot_set, cold_set, history_set,
                              top_n=COLD_INJECT_TOP_N, max_injected=COLD_INJECT_MAX,
                              seed=COLD_INJECT_SEED):
    """从 Top N 热温组合生成冷号变体并注入候选池

    动机：纯按复式人气展开会系统性漏掉含冷号的真实开奖组合
    （例如本期 33 是冷号，67 组复式没人同时选 33+蓝11，导致开奖组合不在展开池中）。

    策略：
      1. 按支持组数降序取 Top N 热温组合
      2. 对每个组合，把其中 1-2 个热号替换为冷号，生成变体
      3. 蓝球保持不变（蓝球冷热预测参考价值低，不强行替换）
      4. 去除已存在组合和历史开奖组合
      5. 注入的组合用 {INJECTED_MARKER} 标记，虚拟 1 组支持

    Args:
        unique_combos: dict[(frozenset(red), blue)] -> set(支持组索引)
                       本函数会就地修改，追加注入的组合
        hot_set: 热号集合
        cold_set: 冷号集合
        history_set: 历史开奖集合，用于过滤
        top_n: 从多少个 Top 组合生成变体
        max_injected: 注入总数上限
        seed: 随机种子

    Returns:
        int: 实际注入的组合数
    """
    import random
    rng = random.Random(seed)

    # 按支持组数降序取 Top N（优先从人气最高的组合派生）
    sorted_by_support = sorted(unique_combos.items(), key=lambda x: -len(x[1]))
    top_combos = sorted_by_support[:top_n]

    cold_list = sorted(cold_set)
    injected_count = 0

    for combo, _supporters in top_combos:
        if injected_count >= max_injected:
            break

        red_fs, blue = combo
        # 找出组合中的热号（候选被替换对象）
        hot_in_combo = sorted(red_fs & hot_set)
        if not hot_in_combo:
            continue

        # 对每个组合生成 2 个变体：替换 1 个热号 + 替换 2 个热号
        for n_replace in [1, 2]:
            if injected_count >= max_injected:
                break
            if n_replace > len(hot_in_combo):
                break

            # 随机选 n_replace 个热号替换为冷号
            hot_to_replace = rng.sample(hot_in_combo, n_replace)
            available_cold = [c for c in cold_list if c not in red_fs]
            if len(available_cold) < n_replace:
                continue
            cold_replacements = rng.sample(available_cold, n_replace)

            new_reds = (red_fs - set(hot_to_replace)) | set(cold_replacements)
            if len(new_reds) != 6:
                continue
            new_combo = (frozenset(new_reds), blue)

            # 跳过已存在（原始或已注入）和历史开奖
            if new_combo in unique_combos or new_combo in history_set:
                continue

            # 注入：用 {INJECTED_MARKER} 标记，len=1 让它能参与 Top 排序
            unique_combos[new_combo] = {INJECTED_MARKER}
            injected_count += 1

    return injected_count


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

    # --- 冷热号分级 ---
    hot_set, warm_set, cold_set = classify_numbers(red_freq, total_groups)
    print(f"\n冷热号分级:")
    print(f"  热号(≥{HOT_THRESHOLD*100:.0f}%): {' '.join(f'{r:02d}' for r in sorted(hot_set))}")
    print(f"  温号({WARM_THRESHOLD*100:.0f}-{HOT_THRESHOLD*100:.0f}%): {' '.join(f'{r:02d}' for r in sorted(warm_set))}")
    print(f"  冷号(<{WARM_THRESHOLD*100:.0f}%): {' '.join(f'{r:02d}' for r in sorted(cold_set))}")

    # --- 冷号注入：从 Top 热温组合生成含冷号的变体 ---
    before_inject = len(unique_combos)
    injected_count = inject_cold_combinations(
        unique_combos, hot_set, cold_set, history_set
    )
    print(f"\n冷号注入:")
    print(f"  从 Top {COLD_INJECT_TOP_N} 热温组合派生冷号变体")
    print(f"  注入组合: {injected_count} 组 (上限 {COLD_INJECT_MAX})")
    print(f"  总有效组合: {before_inject} → {len(unique_combos)} 组")

    # --- 排序：出席率(支持组数)降序 → 号码热度+冷号平衡加分降序 → 蓝球热度 → 组合本身 ---
    # 冷号平衡：1-2 个冷号的组合获得 COLD_BALANCE_BONUS 加分
    # 让 "5 热 1 冷" / "4 热 2 冷" 的混合结构能在 Top 推荐中占据合理比例
    def combo_score(item):
        combo, supporters = item
        red_fs, blue = combo
        # 号码出席率均值
        red_scores = [red_freq.get(r, 0) / total_groups for r in red_fs]
        blue_score = blue_freq.get(blue, 0) / total_groups
        avg_score = sum(red_scores) / len(red_scores)
        # 冷号平衡加分：1-2 个冷号 +bonus，0 个或 3+ 个冷号不加分
        cold_count = len(red_fs & cold_set)
        if 1 <= cold_count <= 2:
            avg_score += COLD_BALANCE_BONUS
        return (-len(supporters), -avg_score, -blue_score, sorted(red_fs))

    sorted_combos = sorted(unique_combos.items(), key=combo_score)

    # --- 输出到 txt ---
    lines = []
    lines.append("=" * 70)
    lines.append("       双色球 6+1 组合 · 出席率排序报告")
    lines.append("=" * 70)
    lines.append("")
    lines.append(f"【数据来源】")
    lines.append(f"  复式分组: {len(groups)} 组（来自 {os.path.basename(EXCEL_PATH)}）")
    lines.append(f"  历史开奖: {history_count} 条（来自 ssq_history.json）")
    lines.append(f"  参与展开: {len(used_groups)} 组")
    if skipped_groups:
        lines.append(f"  跳过全选组(红球≥{FULL_SET_THRESHOLD}): {len(skipped_groups)} 组"
                     f"（组合爆炸，无筛选意义）")
    lines.append(f"  唯一组合: {total_combos} 组")
    lines.append(f"  命中历史: {len(history_hits)} 组（已去除）")
    lines.append(f"  冷号注入: {injected_count} 组（算法派生，标记为 [注入]）")
    lines.append(f"  有效输出: {len(unique_combos)} 组")
    lines.append("")

    # --- 冷热号分级 ---
    lines.append("─" * 70)
    lines.append("【冷热号分级】（基于复式组级出现率）")
    lines.append("─" * 70)
    lines.append(f"  热号(≥{HOT_THRESHOLD*100:.0f}%): {' '.join(f'{r:02d}' for r in sorted(hot_set))}")
    lines.append(f"  温号({WARM_THRESHOLD*100:.0f}-{HOT_THRESHOLD*100:.0f}%): {' '.join(f'{r:02d}' for r in sorted(warm_set))}")
    lines.append(f"  冷号(<{WARM_THRESHOLD*100:.0f}%): {' '.join(f'{r:02d}' for r in sorted(cold_set))}")
    lines.append(f"  注：冷号平衡加分 +{COLD_BALANCE_BONUS:.2f} 给含 1-2 个冷号的组合")
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
    lines.append("【6+1 组合列表 · 按出席率+冷号平衡排序】")
    lines.append("  出席率 = 该组合被多少组复式覆盖 / 总组数")
    lines.append("  号码热度 = 6个红球+1蓝球的平均出席率")
    lines.append("  结构   = 热号数/温号数/冷号数（基于红球）")
    lines.append("  [注入] = 算法派生的冷号变体（非复式直接展开）")
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

        # 冷热结构
        h = len(red_fs & hot_set)
        w = len(red_fs & warm_set)
        c = len(red_fs & cold_set)
        structure = f"{h}热{w}温{c}冷"

        # 来源标注：注入组合 vs 复式展开
        is_injected = INJECTED_MARKER in supporters
        if is_injected:
            source_tag = "[注入]"
        else:
            # 支持组来源
            real_supporters = [i for i in supporters if i != INJECTED_MARKER]
            support_imgs = [groups[i]["img"] for i in sorted(real_supporters)]
            # 去重图片名
            support_imgs_short = []
            seen = set()
            for img in support_imgs:
                if img not in seen:
                    seen.add(img)
                    support_imgs_short.append(img)
            source_tag = ", ".join(support_imgs_short)

        lines.append(
            f"#{rank:04d}  {red_str} + {blue_str}  "
            f"| 出席率: {support_count}/{total_groups} = {attendance_pct:5.1f}%  "
            f"| 热度: {avg_hot:5.1f}%  "
            f"| 结构: {structure}  "
            f"| 来源: {source_tag}"
        )

    lines.append("")
    lines.append("─" * 70)
    lines.append(f"  共 {len(unique_combos)} 组有效组合（已去除 {len(history_hits)} 组历史重复）")
    lines.append("─" * 70)

    output_text = "\n".join(lines)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        f.write(output_text + "\n")

    print(f"\n已输出: {OUTPUT_PATH}")
    print(f"  有效组合: {len(unique_combos)} 组 (含 {injected_count} 组注入)")
    print(f"  最高出席率: {max_support}/{total_groups} = {max_support/total_groups*100:.1f}%")

    # 统计 Top 100 冷热结构分布
    top100 = sorted_combos[:100]
    top100_with_cold = sum(1 for c, _ in top100 if len(c[0] & cold_set) > 0)
    top100_injected = sum(1 for _, s in top100 if INJECTED_MARKER in s)
    print(f"\n  Top 100 冷热结构:")
    print(f"    含冷号组合: {top100_with_cold} 个 ({top100_with_cold}%)")
    print(f"    算法注入组合: {top100_injected} 个")

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
        h = len(red_fs & hot_set)
        w = len(red_fs & warm_set)
        c = len(red_fs & cold_set)
        tag = " [注入]" if INJECTED_MARKER in supporters else ""
        print(f"  #{rank:04d}  {red_str} + {blue:02d}  "
              f"| 出席率: {support_count}/{total_groups}={attendance_pct:.1f}%  "
              f"| 热度: {avg_hot:.1f}%  "
              f"| 结构: {h}热{w}温{c}冷{tag}")


if __name__ == "__main__":
    main()

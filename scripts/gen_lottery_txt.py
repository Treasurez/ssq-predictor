# -*- coding: utf-8 -*-
"""
从 最新的双色球全部号码汇总_YYYYMMDD.xlsx 生成 txt 文本文件
格式：每行一组「红球 + 蓝球」，号码空格分隔
示例：3 7 9 11 13 14 16 20 22 28 29 31 + 11 13 15
"""
import pandas as pd
import os
import sys

# 导入全局配置
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import (
    get_summary_xlsx_or_exit, LOTTERY_SUMMARY_TXT
)

EXCEL_PATH = get_summary_xlsx_or_exit()   # 自动定位最新汇总 xlsx
TXT_PATH = LOTTERY_SUMMARY_TXT


def parse_nums(cell):
    """将 Excel 单元格中的号码字符串解析为整数列表"""
    if pd.isna(cell) or str(cell).strip() == "":
        return []
    parts = str(cell).split(",")
    nums = []
    for p in parts:
        p = p.strip()
        if p.isdigit():
            nums.append(int(p))
    return nums


def format_line(red_nums, blue_nums):
    """格式化为一行：红球 + 蓝球"""
    red_str = " ".join(str(n) for n in sorted(red_nums))
    blue_str = " ".join(str(n) for n in sorted(blue_nums))
    if blue_str:
        return f"{red_str} + {blue_str}"
    return red_str


def main():
    df = pd.read_excel(EXCEL_PATH)

    lines = []
    current_img = None
    img_idx = 0

    for _, row in df.iterrows():
        img_name = row.get("图片文件名", "")
        red_nums = parse_nums(row.get("红球", ""))
        blue_nums = parse_nums(row.get("蓝球", ""))

        # 同一张图片的分组之间加分组标题
        if img_name != current_img:
            current_img = img_name
            img_idx += 1
            lines.append(f"=== 第{img_idx}张图片: {img_name} ===")

        line = format_line(red_nums, blue_nums)
        lines.append(line)

    with open(TXT_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    print(f"已生成: {TXT_PATH}")
    print(f"共 {len(df)} 组号码，来自 {img_idx} 张图片\n")
    print("=" * 60)
    print("\n".join(lines))


if __name__ == "__main__":
    main()

"""全局配置：统一管理项目中的所有文件名和路径

所有需要引用双色球相关文件(xlsx/txt)的脚本，都应从此处导入，
确保路径/文件名一致，避免散落的硬编码。

文件名规则：
  - 彩票图片 OCR 汇总 xlsx（生成端：lottery.py / lottery_sim.py）
      命名: 双色球全部号码汇总_YYYYMMDD.xlsx   (在 PROJECT_ROOT 下)
  - 消费端（gen_combinations.py / gen_probability.py / gen_lottery_txt.py）
      自动读取 PROJECT_ROOT 下"日期戳最新"的汇总 xlsx，无需手动改名
  - 衍生输出（组合展开/概率统计等 txt）统一放 scripts/ 目录下

用法示例:
    from config import LOTTERY_SUMMARY_PATH          # 当天生成用的完整路径
    from config import find_latest_summary_xlsx()    # 消费端定位最新xlsx
    from config import (
        COMBO_ATTEND_TXT, PROBABILITY_TXT, LOTTERY_SUMMARY_TXT,
        GROUP_RESULT_XLSX, COMBO_GA_TXT, COMBO_FREQ_TXT, COMBO_SCORE_TXT
    )
"""

import os
import glob
from datetime import datetime


# ===================== 基础路径 =====================
# 项目根目录（本文件位于 scripts/，根目录是其上一级）
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
LLM_DIR = os.path.join(SCRIPTS_DIR, "llm")

# 当天日期戳，格式 YYYYMMDD
DATE_STAMP = datetime.now().strftime('%Y%m%d')


# ===================== 1. OCR 汇总 xlsx（lottery.py / lottery_sim.py 生成）=====================
# 生成端写入用的文件名（带当天日期）
LOTTERY_SUMMARY_XLSX = f'双色球全部号码汇总_{DATE_STAMP}.xlsx'
LOTTERY_SUMMARY_PATH = os.path.join(PROJECT_ROOT, LOTTERY_SUMMARY_XLSX)

# 不带日期的旧格式短名（兼容 lottery_sim.py 旧主入口风格；新代码请勿用）
_LEGACY_SHORT_XLSX = '双色球汇总.xlsx'


def find_latest_summary_xlsx(project_root=None):
    """消费端用：自动查找项目根目录下"日期戳最新"的双色球汇总 xlsx

    匹配模式: 双色球全部号码汇总_*.xlsx
    按文件名（日期戳）降序，返回第一个存在的完整路径。

    找不到带日期戳的时，回退顺序：
      1) PROJECT_ROOT/双色球汇总.xlsx        (旧短名，根目录)
      2) SCRIPTS_DIR/双色球汇总.xlsx         (旧短名，scripts 目录)

    返回:
        str  | 找到的文件完整路径
        None | 所有回退路径都不存在时
    """
    root = project_root or PROJECT_ROOT
    # 1) 带日期戳的新格式（优先）
    pattern = os.path.join(root, '双色球全部号码汇总_*.xlsx')
    matches = sorted(glob.glob(pattern), reverse=True)
    if matches:
        return matches[0]
    # 2) 旧短名回退
    for fallback in [
        os.path.join(root, _LEGACY_SHORT_XLSX),
        os.path.join(SCRIPTS_DIR, _LEGACY_SHORT_XLSX),
    ]:
        if os.path.exists(fallback):
            return fallback
    return None


def get_summary_xlsx_or_exit():
    """对 find_latest_summary_xlsx 的包装：找不到直接退出，避免下游炸空路径"""
    path = find_latest_summary_xlsx()
    if not path:
        import sys
        print("错误：找不到任何双色球汇总 xlsx 文件。")
        print("  请先运行 lottery.py 或 lottery_sim.py 生成 OCR 汇总。")
        print(f"  查找范围（新格式）: {PROJECT_ROOT}/双色球全部号码汇总_*.xlsx")
        print(f"  查找范围（旧格式）: {PROJECT_ROOT}/双色球汇总.xlsx 或 {SCRIPTS_DIR}/双色球汇总.xlsx")
        sys.exit(1)
    return path


# ===================== 2. 消费端衍生输出（都放 scripts/ 目录下）=====================

# gen_combinations.py → 组合出席率排序 txt
COMBO_ATTEND_TXT = os.path.join(SCRIPTS_DIR, f'双色球组合_出席率排序_{DATE_STAMP}.txt')

# gen_probability.py → 号码出现概率统计 txt
PROBABILITY_TXT = os.path.join(SCRIPTS_DIR, f'双色球号码出现概率_{DATE_STAMP}.txt')

# gen_lottery_txt.py → 汇总转 txt
LOTTERY_SUMMARY_TXT = os.path.join(SCRIPTS_DIR, f'双色球汇总_{DATE_STAMP}.txt')

# llm/test.py → 全部分组结果 xlsx（放项目根，方便查看）
GROUP_RESULT_XLSX = os.path.join(PROJECT_ROOT, f'双色球全部分组结果_{DATE_STAMP}.xlsx')

# llm/ga_optimize.py → GA 优化结果 txt
COMBO_GA_TXT = os.path.join(SCRIPTS_DIR, f'双色球组合_GA优化_{DATE_STAMP}.txt')

# gen_combinations.py 也生成 组合_出席率排序 → 已用 COMBO_ATTEND_TXT
# llm/ml_score.py 读取 COMBO_ATTEND_TXT，输出 COMBO_SCORE_TXT
COMBO_FREQ_TXT = COMBO_ATTEND_TXT  # 出席率和频率是同一份文件，别名兼容旧命名
COMBO_SCORE_TXT = os.path.join(SCRIPTS_DIR, f'双色球组合_统计评分_{DATE_STAMP}.txt')

# predict_combo.py: 读 COMBO_SCORE_TXT 和 COMBO_GA_TXT，输出最终推荐
def combo_final_txt():
    """最终推荐文件名：带日期戳，避免覆盖"""
    return os.path.join(SCRIPTS_DIR, f'双色球组合_最终推荐_{DATE_STAMP}.txt')

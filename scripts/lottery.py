import easyocr
import cv2
import re
import os
import sys
from collections import defaultdict
import pandas as pd

# 导入全局配置（确保能找到同目录的 config 模块）
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import LOTTERY_SUMMARY_PATH

# ===================== 1. 模型检查与初始化 =====================
# 优先使用项目目录的模型，否则使用 EasyOCR 默认路径
PROJECT_MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'easyocr_models')
DEFAULT_MODEL_DIR = os.path.expanduser('~/.EasyOCR/model')

REQUIRED_MODELS = ['craft_mlt_25k.pth', 'zh_sim_g2.pth', 'english_g2.pth']

def check_models(model_dir):
    missing = []
    if not os.path.exists(model_dir):
        return REQUIRED_MODELS
    for model_name in REQUIRED_MODELS:
        model_path = os.path.join(model_dir, model_name)
        if not os.path.exists(model_path):
            missing.append(model_name)
        elif os.path.getsize(model_path) < 500000:
            missing.append(f"{model_name} (文件太小)")
    return missing

# 先检查项目目录
missing_in_project = check_models(PROJECT_MODEL_DIR)
if not missing_in_project:
    MODEL_DIR = PROJECT_MODEL_DIR
    print(f"使用项目目录的 EasyOCR 模型: {MODEL_DIR}")
else:
    # 检查默认目录
    missing_in_default = check_models(DEFAULT_MODEL_DIR)
    if not missing_in_default:
        MODEL_DIR = DEFAULT_MODEL_DIR
        print(f"使用 EasyOCR 默认目录的模型: {MODEL_DIR}")
    else:
        # 两个目录都不完整，尝试使用默认路径触发下载
        print("检测到缺少 EasyOCR 模型文件...")
        print(f"\n项目目录 {PROJECT_MODEL_DIR} 缺少: {missing_in_project}")
        print(f"默认目录 {DEFAULT_MODEL_DIR} 缺少: {missing_in_default}")
        print("\n正在尝试通过 EasyOCR 自动下载模型...")
        print("（首次下载可能需要几分钟，取决于网络速度）\n")
        
        try:
            reader = easyocr.Reader(
                ['ch_sim','en'],
                gpu=False,
                download_enabled=True,
                verbose=True
            )
            MODEL_DIR = DEFAULT_MODEL_DIR
            print("\n模型下载完成！")
        except Exception as e:
            print(f"\n自动下载失败：{e}")
            print("\n请手动下载模型文件：")
            print("  python3 scripts/download_easyocr_models.py")
            sys.exit(1)
        
        # 下载完成后重新检查
        missing_after = check_models(DEFAULT_MODEL_DIR)
        if missing_after:
            print(f"\n下载后仍缺少: {missing_after}")
            sys.exit(1)

# 创建 Reader
reader = easyocr.Reader(
    ['ch_sim','en'],
    gpu=False,
    model_storage_directory=MODEL_DIR,
    download_enabled=False,
    verbose=False
)
print("EasyOCR 模型加载成功！")


# ===================== 2. 图片预处理：直接使用原图 =====================
def load_image(img_path):
    """加载图片，直接返回原图（BGR格式）"""
    img = cv2.imread(img_path)
    if img is None:
        return None
    return img

# ===================== 3. 单图提取所有复式红球蓝球分组 =====================
def parse_lottery_image(img_path, debug=False):
    # 直接使用原图进行OCR
    img = load_image(img_path)
    if img is None:
        return []
    
    # OCR识别全部文字（使用原图，不做二值化处理）
    text_lines = reader.readtext(img, detail=0)
    full_text = "\n".join(text_lines)
    lines = [line.strip() for line in full_text.splitlines() if line.strip()]
    
    if debug:
        print(f"  OCR识别 {len(lines)} 行文本:")
        for l in lines[:30]:  # 最多显示30行
            print(f"    {l}")
    
    # 尝试解析
    groups = try_parse_lottery_lines(lines, debug)
    
    # 如果原图识别结果不好，尝试灰度图
    if len(groups) == 0:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        text_lines2 = reader.readtext(gray, detail=0)
        full_text2 = "\n".join(text_lines2)
        lines2 = [line.strip() for line in full_text2.splitlines() if line.strip()]
        
        if debug:
            print(f"  [灰度图] OCR识别 {len(lines2)} 行文本:")
            for l in lines2[:30]:
                print(f"    {l}")
        
        groups = try_parse_lottery_lines(lines2, debug)
    
    return groups

def try_parse_lottery_lines(lines, debug=False):
    """尝试从OCR文本中解析双色球号码
    
    处理多种格式：
    1. 连续行格式: 红球:01 07 13 20 24 28 32 33
                  蓝球:10 13 14 [1倍]
    2. 分行格式:   红球:01
                  07
                  13
                  ...
                  蓝球:10
                  13
                  14
                  [1倍]
    3. 胆拖格式:   红胆:07 12 23 24
                  红拖:02 04 06 08 11 15 19
                  蓝球:01 02
                  [1倍]
    4. OCR误读格式: 红爬:02, 红皿:01, 蓝琛:09, [5倚] 等
    """
    
    # ====== 辅助函数（模块级，避免闭包变量问题） ======
    def extract_red_nums(text):
        """从文本中提取红球号码(1-33)"""
        nums = []
        for m in re.findall(r"(\d{1,2})", text):
            n = int(m)
            if 1 <= n <= 33 and n not in nums:
                nums.append(n)
        return nums
    
    def extract_blue_nums(text):
        """从文本中提取蓝球号码(1-16)"""
        nums = []
        for m in re.findall(r"(\d{1,2})", text):
            n = int(m)
            if 1 <= n <= 16 and n not in nums:
                nums.append(n)
        return nums
    
    def extract_times(text):
        """从文本中提取倍数"""
        for pat in [r"\[?(\d+)倍\]?", r"\[?(\d+)倚\]?", r"(\d+)倍", r"(\d+)倚"]:
            m = re.search(pat, text)
            if m:
                try:
                    t = int(m.group(1))
                    if 1 <= t <= 100:
                        return t
                except:
                    pass
        return None
    
    # ====== 状态变量 ======
    all_groups = []
    
    reading_red = [False]      # 是否正在读取红球
    reading_blue = [False]     # 是否正在读取蓝球
    current_red = [[]]         # 当前红球列表
    current_blue = [[]]        # 当前蓝球列表
    current_times = [1]        # 当前倍数
    current_red_dan = [[]]     # 胆拖格式-胆码
    current_red_tuo = [[]]     # 胆拖格式-拖码
    in_dantuo_mode = [False]   # 是否胆拖模式
    passed_tuo = [False]       # 胆拖模式下是否已过"红拖"标记
    # 红球队列：当多个红球组连续出现时，先缓存红球队列，等待蓝球配对
    pending_red_groups = []    # 每个元素: {"red": [...], "times": N, "dantuo": bool, "dan": [...], "tuo": [...]}
    
    def _is_red_keyword(line):
        """检测行是否以红球/红胆/红拖等关键字开头，返回类型和同行数字
        
        注意：必须从最具体的关键字开始匹配，避免"红"先匹配了"红胆/红球/红拖"等
        """
        # 红球关键字变体（从最具体到最不具体排序）
        red_keys = [
            ("红球", False),   # 必须先于"红"
            ("红胆", True),    # 胆码
            ("红拖", False),   # 拖码
            ("红爬", False),   # OCR误读
            ("红皿", False),   # OCR误读
            ("红昭", False),   # OCR误读
            ("红粑", False),   # OCR误读
            ("红", False),     # 兜底：单字红（必须带冒号以避免误匹配）
        ]
        for keyword, is_dan in red_keys:
            if keyword == "红":
                # 单字红必须带冒号，避免匹配"红胆"、"红球"等
                pat = re.compile(r"红[:：]\s*((?:\d{1,2}\s*)*)")
            else:
                pat = re.compile(re.escape(keyword) + r"[:：]?\s*((?:\d{1,2}\s*)*)")
            m = pat.match(line)
            if m:
                nums_text = m.group(1) if m.lastindex and m.lastindex >= 1 else ""
                nums = extract_red_nums(nums_text)
                return is_dan, nums
        return None, []
    
    def _is_blue_keyword(line):
        """检测行是否以蓝球关键字开头，返回同行数字
        
        注意：必须从最具体的关键字开始匹配
        """
        blue_keys = [
            "蓝球", "蓝胆", "蓝琛", "蓝",
        ]
        for keyword in blue_keys:
            if keyword == "蓝":
                # 单字蓝必须带冒号
                pat = re.compile(r"蓝[:：]\s*((?:\d{1,2}\s*)*)")
            else:
                pat = re.compile(re.escape(keyword) + r"[:：]?\s*((?:\d{1,2}\s*)*)")
            m = pat.match(line)
            if m:
                nums_text = m.group(1) if m.lastindex and m.lastindex >= 1 else ""
                nums = extract_blue_nums(nums_text)
                return nums
        # 兜底：匹配 "B球" 或 "球:" 开头（缺少"蓝"字的情况）
        for pat_str in [r"[Bb]球[:：]?\s*((?:\d{1,2}\s*)*)",
                        r"球[:：]\s*((?:\d{1,2}\s*)*)"]:
            m = re.match(pat_str, line)
            if m:
                nums_text = m.group(1) if m.lastindex and m.lastindex >= 1 else ""
                nums = extract_blue_nums(nums_text)
                return nums
        return []
    
    def _save_current_red():
        """将当前已读取的红球保存到等待队列（等待蓝球配对）"""
        if in_dantuo_mode[0]:
            total = len(current_red_dan[0]) + len(current_red_tuo[0])
            if total >= 2:
                pending_red_groups.append({
                    "dantuo": True,
                    "dan": list(current_red_dan[0]),
                    "tuo": list(current_red_tuo[0]),
                    "times": current_times[0]
                })
                # 重置红球相关状态
                current_red_dan[0] = []
                current_red_tuo[0] = []
                in_dantuo_mode[0] = False
                passed_tuo[0] = False
                current_times[0] = 1
                reading_red[0] = False
        else:
            if len(current_red[0]) >= 2:
                pending_red_groups.append({
                    "dantuo": False,
                    "red": list(current_red[0]),
                    "times": current_times[0]
                })
                current_red[0] = []
                current_times[0] = 1
                reading_red[0] = False
    
    def _try_finalize_with_blue():
        """尝试用当前蓝球与等待队列中的红球配对"""
        if pending_red_groups and len(current_blue[0]) >= 1:
            # 取出最早等待的红球组
            red_group = pending_red_groups.pop(0)
            if red_group["dantuo"]:
                merged = list(set(red_group["dan"] + red_group["tuo"]))
                merged.sort()
                if len(merged) >= 2:
                    all_groups.append({
                        "red": merged,
                        "blue": sorted(current_blue[0]),
                        "times": red_group["times"]
                    })
            else:
                if len(red_group["red"]) >= 2:
                    all_groups.append({
                        "red": sorted(red_group["red"]),
                        "blue": sorted(current_blue[0]),
                        "times": red_group["times"]
                    })
            
            # 如果还有等待的红球组，保留蓝球数据用于下一组
            # 如果没有等待的红球组，才清空蓝球数据
            if pending_red_groups:
                # 还有红球等待配对，保留蓝球数据并清空reading状态
                # 这样后续的蓝球数字可以继续添加
                # 但需要重新开启读取模式
                pass  # 保留current_blue，继续累加
            else:
                current_blue[0] = []
                reading_blue[0] = False
    
    def _finalize():
        """完成当前分组（用于非队列模式或最后一组）"""
        # 先尝试用队列模式处理所有能配对的组
        while pending_red_groups and len(current_blue[0]) >= 1:
            _try_finalize_with_blue()
        
        # 如果还有未配对的蓝球但没有更多红球等待，
        # 将蓝球与最后一个current_red（如果有）配对
        if pending_red_groups:
            # 队列中还有未配对的红球，丢弃（没有蓝球）
            # 但保留current_blue用于后续可能的红球
            pass
        
        # 处理最后一组（如果有完整红蓝球）
        if in_dantuo_mode[0]:
            merged = list(set(current_red_dan[0] + current_red_tuo[0]))
            merged.sort()
            if len(merged) >= 2 and len(current_blue[0]) >= 1:
                all_groups.append({
                    "red": merged,
                    "blue": sorted(current_blue[0]),
                    "times": current_times[0]
                })
        else:
            if len(current_red[0]) >= 2 and len(current_blue[0]) >= 1:
                all_groups.append({
                    "red": sorted(current_red[0]),
                    "blue": sorted(current_blue[0]),
                    "times": current_times[0]
                })
        
        # 处理队列中剩余未配对的红球（没有蓝球，无法组成有效组）
        pending_red_groups.clear()
        
        # 重置状态
        current_red[0] = []
        current_blue[0] = []
        current_times[0] = 1
        current_red_dan[0] = []
        current_red_tuo[0] = []
        in_dantuo_mode[0] = False
        passed_tuo[0] = False
        reading_red[0] = False
        reading_blue[0] = False
    
    def _start_new_red(nums, is_dan):
        """开始新的红球读取"""
        if is_dan:
            # 开始胆拖模式 - 红胆
            in_dantuo_mode[0] = True
            passed_tuo[0] = False
            current_red_dan[0] = list(nums)
            reading_red[0] = True
            reading_blue[0] = False
        elif in_dantuo_mode[0]:
            # 胆拖模式下遇到新的红关键字（通常是"红拖"）
            passed_tuo[0] = True
            # 将数字加入拖码（如果同行有数字）
            for n in nums:
                if n not in current_red_tuo[0]:
                    current_red_tuo[0].append(n)
            reading_red[0] = True
            reading_blue[0] = False
        else:
            # 普通模式
            current_red[0] = list(nums)
            reading_red[0] = True
            reading_blue[0] = False
    
    def _add_red_nums(nums):
        """添加红球号码到当前组"""
        if in_dantuo_mode[0]:
            if passed_tuo[0]:
                # 已过"红拖"标记，加入拖码
                for n in nums:
                    if n not in current_red_tuo[0]:
                        current_red_tuo[0].append(n)
            else:
                # 还在"红胆"阶段，加入胆码
                for n in nums:
                    if n not in current_red_dan[0]:
                        current_red_dan[0].append(n)
        else:
            for n in nums:
                if n not in current_red[0]:
                    current_red[0].append(n)
    
    def _start_new_blue(nums):
        """开始新的蓝球读取"""
        # 保留已读取的红球数据
        current_blue[0] = list(nums)
        reading_blue[0] = True
        reading_red[0] = False
    
    def _add_blue_nums(nums):
        """添加蓝球号码到当前组"""
        for n in nums:
            if n not in current_blue[0]:
                current_blue[0].append(n)
    
    # ====== 主解析循环 ======
    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        
        # 0. 检查是否是倍数行
        t = extract_times(stripped)
        if t is not None:
            current_times[0] = t
            # 尝试用队列方式完成配对
            if pending_red_groups and len(current_blue[0]) >= 1:
                _try_finalize_with_blue()
            # 同时检查当前直接模式
            if in_dantuo_mode[0]:
                total_red = len(current_red_dan[0]) + len(current_red_tuo[0])
                if total_red >= 2 and len(current_blue[0]) >= 1:
                    _finalize()
            elif len(current_red[0]) >= 2 and len(current_blue[0]) >= 1:
                _finalize()
            continue
        
        # 1. 检查是否是红球/红胆/红拖等开头
        is_dan, red_nums = _is_red_keyword(stripped)
        if is_dan is not None:
            # 如果已有红球队列或当前红球，先保存
            # 但在胆拖模式下遇到"红拖"时，不保存（只是胆拖内的状态转换）
            is_red_tuo_transition = (in_dantuo_mode[0] and not is_dan)
            if not is_red_tuo_transition:
                if in_dantuo_mode[0] or len(current_red[0]) >= 2:
                    _save_current_red()
                # 如果有蓝球等待配对，也尝试完成
                if len(current_blue[0]) >= 1:
                    _try_finalize_with_blue()
            
            _start_new_red(red_nums, is_dan)
            
            # 如果同行有足够号码，继续读下一行
            if in_dantuo_mode[0]:
                total = len(current_red_dan[0]) + len(current_red_tuo[0])
                if total >= 2:
                    continue
            else:
                if len(current_red[0]) >= 2:
                    continue
            continue
        
        # 2. 检查是否是蓝球开头
        blue_nums = _is_blue_keyword(stripped)
        if blue_nums:
            # 先把已有的红球保存到等待队列
            if in_dantuo_mode[0] or len(current_red[0]) >= 2:
                _save_current_red()
            
            _start_new_blue(blue_nums)
            # 立即尝试用蓝球与等待队列配对
            if pending_red_groups and len(current_blue[0]) >= 1:
                _try_finalize_with_blue()
            if len(current_blue[0]) >= 1:
                continue
            continue
        
        # 3. 检查是否是纯数字行（延续的号码）
        num_match = re.match(r"^\s*(\d{1,2})\s*$", stripped)
        if num_match:
            num = int(num_match.group(1))
            if reading_red[0]:
                if 1 <= num <= 33:
                    _add_red_nums([num])
            elif reading_blue[0]:
                if 1 <= num <= 16:
                    _add_blue_nums([num])
            continue
        
        # 4. 正在读取红球/蓝球时，从包含数字的行提取号码
        if reading_red[0] and re.search(r"\d", stripped):
            digit_count = len(re.findall(r"\d", stripped))
            alpha_count = len(re.findall(r"[a-zA-Z\u4e00-\u9fff]", stripped))
            if digit_count >= 1 and alpha_count <= 3:
                nums = extract_red_nums(stripped)
                if nums:
                    _add_red_nums(nums)
        elif reading_blue[0] and re.search(r"\d", stripped):
            digit_count = len(re.findall(r"\d", stripped))
            alpha_count = len(re.findall(r"[a-zA-Z\u4e00-\u9fff]", stripped))
            if digit_count >= 1 and alpha_count <= 3:
                nums = extract_blue_nums(stripped)
                if nums:
                    _add_blue_nums(nums)
    
    # 处理最后一组
    _finalize()
    
    if debug:
        print(f"  解析到 {len(all_groups)} 组复式")
        for g in all_groups:
            print(f"    红球: {g['red']}, 蓝球: {g['blue']}, 倍数: {g['times']}")
    
    return all_groups

# ===================== 4. 批量遍历文件夹所有图片，汇总全部号码 =====================
def batch_parse_images(folder_path, debug=False):
    total_groups = []
    # 遍历jpg/png图片
    for file_name in os.listdir(folder_path):
        if file_name.lower().endswith((".jpg", ".png", ".jpeg")):
            img_full_path = os.path.join(folder_path, file_name)
            if debug:
                print(f"\n{'='*60}")
                print(f"处理图片: {file_name}")
            groups = parse_lottery_image(img_full_path, debug=debug)
            for g in groups:
                g["img_name"] = file_name
                total_groups.append(g)
            print(f"【{file_name}】识别到 {len(groups)} 组复式")
            if debug and groups:
                for g in groups:
                    print(f"  红球: {g['red']}, 蓝球: {g['blue']}, 倍数: {g['times']}")
    print(f"\n全部图片解析完成，总计 {len(total_groups)} 组彩票复式")
    return total_groups

# ===================== 5. 统计红球、蓝球出现频次（加权倍数） =====================
def calc_hot_cold(all_groups):
    red_count = defaultdict(int)
    blue_count = defaultdict(int)
    for item in all_groups:
        mul = item["times"]
        # 统计红球
        for r in item["red"]:
            red_count[r] += mul
        # 统计蓝球
        for b in item["blue"]:
            blue_count[b] += mul
    # 按出现次数降序排序（热号在前）
    sorted_red = sorted(red_count.items(), key=lambda x: x[1], reverse=True)
    sorted_blue = sorted(blue_count.items(), key=lambda x: x[1], reverse=True)
    return sorted_red, sorted_blue

# ===================== 6. 根据热度推演娱乐推荐号码 =====================
def get_recommend(sorted_red, sorted_blue):
    # 取热度前12个红球作为备选池
    hot_red_pool = [num for num, cnt in sorted_red[:12]]
    # 取热度前5个蓝球作为备选池
    hot_blue_pool = [num for num, cnt in sorted_blue[:5]]

    print("\n========== 热度统计结果 ==========")
    print("红球热度排序（数字:出现次数）：")
    for num, cnt in sorted_red:
        print(f"{num:02d} : {cnt}次")

    print("\n蓝球热度排序（数字:出现次数）：")
    for num, cnt in sorted_blue:
        print(f"{num:02d} : {cnt}次")

    print("\n========== 娱乐推荐号码池 ==========")
    print(f"高频红球池(任选6个组合)：{hot_red_pool}")
    print(f"高频蓝球池(任选1个搭配)：{hot_blue_pool}")
    return hot_red_pool, hot_blue_pool

# ===================== 7. 导出所有彩票分组到Excel备查 =====================
def export_to_excel(all_groups, save_name=None):
    if save_name is None:
        save_name = LOTTERY_SUMMARY_PATH
    save_name = os.path.abspath(save_name)
    data_list = []
    for g in all_groups:
        data_list.append({
            "图片文件名": g["img_name"],
            "红球": ",".join([str(x) for x in g["red"]]),
            "蓝球": ",".join([str(x) for x in g["blue"]]),
            "投注倍数": g["times"]
        })
    df = pd.DataFrame(data_list)
    df.to_excel(save_name, index=False)
    print(f"\n所有分组已导出至：{save_name}")

# ===================== 主程序入口 =====================
if __name__ == "__main__":
    # # 普通运行
    # python3 scripts/lottery.py
    # # 调试模式（查看详细 OCR 识别结果）
    # python3 scripts/lottery.py --debug
    # # 指定图片目录
    # python3 scripts/lottery.py --folder ./your_images
    # 检查是否启用调试模式
    import argparse
    parser = argparse.ArgumentParser(description='双色球图片OCR识别')
    parser.add_argument('--debug', '-d', action='store_true', help='启用调试模式，显示详细OCR输出')
    parser.add_argument('--folder', '-f', default='../lottery_img', help='图片文件夹路径')
    args = parser.parse_args()
    
    # 1. 设置图片文件夹路径
    IMG_FOLDER = args.folder

    # 2. 批量解析所有图片
    all_lottery_groups = batch_parse_images(IMG_FOLDER, debug=args.debug)

    # 3. 导出全部号码到Excel
    export_to_excel(all_lottery_groups)

    # 4. 计算冷热号频次
    red_hot_sort, blue_hot_sort = calc_hot_cold(all_lottery_groups)

    # 5. 输出热度并生成推荐号码池
    red_pool, blue_pool = get_recommend(red_hot_sort, blue_hot_sort)
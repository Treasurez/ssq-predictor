import easyocr
import cv2
import re
import os
import sys
from collections import defaultdict
import pandas as pd

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
    """
    all_groups = []
    
    # 标记当前读取状态
    reading_red = False
    reading_blue = False
    current_red = []
    current_blue = []
    current_times = 1
    
    # 红球/蓝球开头关键字
    red_start_patterns = [
        re.compile(r"红球[:：]?\s*((?:\d{1,2}\s*)*)"),   # 红球: 或 红球
        re.compile(r"红[:：]\s*((?:\d{1,2}\s*)*)"),     # 红: 或 红
        re.compile(r"红胆[:：]\s*((?:\d{1,2}\s*)*)"),   # 红胆: 或 红胆
    ]
    
    blue_start_patterns = [
        re.compile(r"蓝球[:：]?\s*((?:\d{1,2}\s*)*)"),   # 蓝球: 或 蓝球
        re.compile(r"蓝[:：]\s*((?:\d{1,2}\s*)*)"),     # 蓝: 或 蓝
        re.compile(r"蓝胆[:：]\s*((?:\d{1,2}\s*)*)"),   # 蓝胆: 或 蓝胆
    ]
    
    # 倍数模式
    times_pattern = re.compile(r"\[?(\d+)倍\]?")
    
    # 纯数字模式
    pure_num_pattern = re.compile(r"^\s*(\d{1,2})\s*$")
    
    for line in lines:
        stripped = line.strip()
        
        # 检查是否是红球开头
        is_red_start = False
        for pattern in red_start_patterns:
            match = pattern.match(stripped)
            if match:
                is_red_start = True
                # 提取同行的数字
                red_part = match.group(1) if match.lastindex and match.lastindex >= 1 else ""
                nums = re.findall(r"\d{1,2}", red_part)
                current_red = [int(n) for n in nums if 1 <= int(n) <= 33]
                reading_red = True
                reading_blue = False
                break
        
        if is_red_start:
            # 如果同行没有完整的红球列表，继续读取下一行
            if len(current_red) < 2:
                continue
            # 否则完成红球读取
            if reading_blue and len(current_blue) >= 1:
                # 完成一组
                all_groups.append({
                    "red": sorted(current_red),
                    "blue": sorted(current_blue),
                    "times": current_times
                })
            current_red = []
            current_blue = []
            current_times = 1
            reading_red = False
            reading_blue = False
            continue
        
        # 检查是否是蓝球开头
        is_blue_start = False
        for pattern in blue_start_patterns:
            match = pattern.match(stripped)
            if match:
                is_blue_start = True
                # 先保存之前可能读取的红球
                if len(current_red) >= 2:
                    pass  # 保持当前红球
                else:
                    current_red = []
                
                # 提取同行的数字
                blue_part = match.group(1) if match.lastindex and match.lastindex >= 1 else ""
                nums = re.findall(r"\d{1,2}", blue_part)
                current_blue = [int(n) for n in nums if 1 <= int(n) <= 16]
                reading_blue = True
                reading_red = False
                break
        
        if is_blue_start:
            if len(current_blue) >= 1 and reading_red == False:
                # 如果已有红球，完成这组
                if len(current_red) >= 2:
                    all_groups.append({
                        "red": sorted(current_red),
                        "blue": sorted(current_blue),
                        "times": current_times
                    })
                    current_red = []
                    current_blue = []
                    current_times = 1
            continue
        
        # 检查是否是倍数行
        times_match = times_pattern.match(stripped)
        if times_match:
            try:
                current_times = int(times_match.group(1))
            except:
                current_times = 1
            
            # 如果已有红球和蓝球，完成这组
            if len(current_red) >= 2 and len(current_blue) >= 1:
                all_groups.append({
                    "red": sorted(current_red),
                    "blue": sorted(current_blue),
                    "times": current_times
                })
                current_red = []
                current_blue = []
                current_times = 1
                reading_red = False
                reading_blue = False
            continue
        
        # 检查是否是纯数字行（可能是延续的红球或蓝球）
        num_match = pure_num_pattern.match(stripped)
        if num_match:
            num = int(num_match.group(1))
            
            if reading_red and 1 <= num <= 33:
                # 正在读取红球
                if num not in current_red:
                    current_red.append(num)
            elif reading_blue and 1 <= num <= 16:
                # 正在读取蓝球
                if num not in current_blue:
                    current_blue.append(num)
    
    # 处理最后一组
    if len(current_red) >= 2 and len(current_blue) >= 1:
        all_groups.append({
            "red": sorted(current_red),
            "blue": sorted(current_blue),
            "times": current_times
        })
    
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
        save_name = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '双色球全部号码汇总.xlsx')
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
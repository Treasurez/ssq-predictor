import cv2
import re
import os
from collections import defaultdict
import pandas as pd
from paddleocr import PaddleOCR

# ===================== 1. 初始化PaddleOCR（全局只创建一次！） =====================
# use_angle_cls=False 关闭文字方向检测，提速；lang="ch"支持中文数字
ocr = PaddleOCR(lang="ch", use_textline_orientation=False, engine="onnxruntime")

# ===================== 2. 图片预处理：去除红色海报干扰 =====================
def preprocess_image(img_path):
    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, binary_img = cv2.threshold(gray, 130, 255, cv2.THRESH_BINARY)
    binary_color = cv2.cvtColor(binary_img, cv2.COLOR_GRAY2BGR)
    return binary_color

# ===================== 3. 单图提取所有复式红球蓝球分组 =====================
def extract_numbers(text):
    raw_nums = re.findall(r'\d+', text)
    result = []
    for num_str in raw_nums:
        if len(num_str) == 2:
            result.append(int(num_str))
        elif len(num_str) > 2:
            for i in range(0, len(num_str), 2):
                chunk = num_str[i:i+2]
                if len(chunk) == 2:
                    n = int(chunk)
                    if 1 <= n <= 33:
                        result.append(n)
    return result

def parse_lottery_image(img_path):
    img = cv2.imread(img_path)
    result = ocr.predict(img)
    text_lines = []
    for item in result:
        if isinstance(item, dict) and 'rec_texts' in item:
            text_lines.extend(item['rec_texts'])
    full_text = "\n".join(text_lines)
    lines = [line.strip() for line in full_text.splitlines() if line.strip()]
    


    all_groups = []
    current_group = None
    state = None

    for line in lines:
        red_match = re.search(r"红球[:：]\s*(.+)", line)
        red_dan_match = re.search(r"红胆[:：]\s*(.+)", line)
        red_tuo_match = re.search(r"红拖[:：]\s*(.+)", line)
        blue_match = re.search(r"蓝球[:：]\s*(.+)", line)
        
        if red_match:
            if current_group is not None:
                all_groups.append(current_group)
            nums = extract_numbers(red_match.group(1))
            current_group = {"red": nums, "blue": [], "times": 1}
            state = "collect_red"
            continue
        
        if red_dan_match:
            if current_group is not None:
                all_groups.append(current_group)
            nums = extract_numbers(red_dan_match.group(1))
            current_group = {"red": nums, "red_tuo": [], "blue": [], "times": 1}
            state = "collect_dan"
            continue
        
        if red_tuo_match:
            nums = extract_numbers(red_tuo_match.group(1))
            if current_group is not None:
                current_group["red_tuo"].extend(nums)
                state = "collect_tuo"
            continue
        
        if blue_match:
            if current_group is not None:
                blue_raw = blue_match.group(1)
                nums = extract_numbers(blue_raw)
                current_group["blue"].extend(nums)
                
                times_match = re.search(r'(\d+)\s*倍', blue_raw)
                if times_match:
                    current_group["times"] = int(times_match.group(1))
                
                if 'red_tuo' in current_group:
                    current_group["red"].extend(current_group.pop("red_tuo"))
                    current_group["red"] = sorted(list(set(current_group["red"])))
                
                all_groups.append(current_group)
                current_group = None
                state = None
            continue
        
        if state in ("collect_red", "collect_dan", "collect_tuo") and current_group is not None:
            nums = extract_numbers(line)
            if nums:
                if state == "collect_red":
                    current_group["red"].extend(nums)
                elif state == "collect_dan":
                    current_group["red"].extend(nums)
                    state = "collect_tuo"
                elif state == "collect_tuo":
                    if 'red_tuo' in current_group:
                        current_group["red_tuo"].extend(nums)
                    else:
                        current_group["red"].extend(nums)
    
    if current_group is not None:
        if 'red_tuo' in current_group:
            current_group["red"].extend(current_group.pop("red_tuo"))
            current_group["red"] = sorted(list(set(current_group["red"])))
        all_groups.append(current_group)
    
    for g in all_groups:
        g["red"] = sorted(list(set(g["red"])))
        g["blue"] = sorted(list(set(g["blue"])))
        g["red"] = [x for x in g["red"] if 1 <= x <= 33]
        g["blue"] = [x for x in g["blue"] if 1 <= x <= 16]
    
    return all_groups

# ===================== 4. 批量遍历文件夹所有图片，汇总全部号码 =====================
def batch_parse_images(folder_path):
    total_groups = []
    # 遍历jpg/png图片
    for file_name in os.listdir(folder_path):
        if file_name.lower().endswith((".jpg", ".png", ".jpeg")):
            img_full_path = os.path.join(folder_path, file_name)
            groups = parse_lottery_image(img_full_path)
            for g in groups:
                g["img_name"] = file_name
                total_groups.append(g)
            print(f"【{file_name}】识别到 {len(groups)} 组复式")
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
def export_to_excel(all_groups, save_name="双色球全部号码汇总.xlsx"):
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
    # 1. 修改为你存放彩票图片的文件夹路径
    IMG_FOLDER = r"../lottery_img"

    # 2. 批量解析所有图片
    all_lottery_groups = batch_parse_images(IMG_FOLDER)

    # 3. 导出全部号码到Excel
    export_to_excel(all_lottery_groups)

    # 4. 计算冷热号频次
    red_hot_sort, blue_hot_sort = calc_hot_cold(all_lottery_groups)

    # 5. 输出热度并生成推荐号码池
    red_pool, blue_pool = get_recommend(red_hot_sort, blue_hot_sort)
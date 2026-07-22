import easyocr
import cv2
import re
import os
from collections import defaultdict
import pandas as pd

# ===================== 1. 初始化工具 =====================
# OCR识别器，支持中文+数字
# reader = easyocr.Reader(['ch_sim', 'en'], gpu=False)
# 手动指定模型存放路径，避免重复下载
reader = easyocr.Reader(
    ['ch_sim','en'],
    gpu=False,
    model_storage_directory="./easyocr_models",  # 模型本地目录
    download_enabled=False,   # ✅ 禁止自动联网下载！模型提前下好放入文件夹
    detector=True,
    recognizer=True
)


# ===================== 2. 图片预处理：去除红色海报干扰 =====================
def preprocess_image(img_path):
    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # 二值化，过滤浅色红底水印，强化小票黑色文字
    _, binary_img = cv2.threshold(gray, 130, 255, cv2.THRESH_BINARY)
    return binary_img

# ===================== 3. 单图提取所有复式红球蓝球分组 =====================
def parse_lottery_image(img_path):
    # 预处理图片
    proc_img = preprocess_image(img_path)
    # OCR识别全部文字
    text_lines = reader.readtext(proc_img, detail=0)
    full_text = "\n".join(text_lines)
    lines = [line.strip() for line in full_text.splitlines() if line.strip()]

    all_groups = []
    temp_red = None

    # 正则匹配规则
    red_rule = re.compile(r"红球[:：]\s*((?:\d{2}\s*)+)")
    blue_rule = re.compile(r"蓝球[:：]\s*((?:\d{2}\s*)+)\s*\[(\d+)倍\]")
    num_rule = re.compile(r"\d{2}")

    for line in lines:
        # 匹配红球行
        red_match = red_rule.search(line)
        if red_match:
            red_raw = red_match.group(1)
            red_nums = sorted([int(x) for x in num_rule.findall(red_raw)])
            temp_red = red_nums
            continue
        # 匹配蓝球行，和上一行红球配对一组
        blue_match = blue_rule.search(line)
        if blue_match and temp_red is not None:
            blue_raw = blue_match.group(1)
            blue_nums = sorted([int(x) for x in num_rule.findall(blue_raw)])
            multiple = int(blue_match.group(2))
            all_groups.append({
                "red": temp_red,
                "blue": blue_nums,
                "times": multiple
            })
            temp_red = None
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
    IMG_FOLDER = r"./lottery_img"

    # 2. 批量解析所有图片
    all_lottery_groups = batch_parse_images(IMG_FOLDER)

    # 3. 导出全部号码到Excel
    export_to_excel(all_lottery_groups)

    # 4. 计算冷热号频次
    red_hot_sort, blue_hot_sort = calc_hot_cold(all_lottery_groups)

    # 5. 输出热度并生成推荐号码池
    red_pool, blue_pool = get_recommend(red_hot_sort, blue_hot_sort)
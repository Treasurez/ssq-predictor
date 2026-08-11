import cv2
import re
import os
import sys
from collections import defaultdict
import pandas as pd
from paddleocr import PaddleOCR

# 导入全局配置（确保能找到同目录的 config 模块）
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import LOTTERY_SUMMARY_PATH, LOTTERY_SUMMARY_XLSX

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
    i = 0
    while i < len(raw_nums):
        num_str = raw_nums[i]
        if len(num_str) == 1:
            n = int(num_str)
            if n == 0:
                i += 1
                continue
            if i + 1 < len(raw_nums) and len(raw_nums[i + 1]) == 1:
                next_n = int(raw_nums[i + 1])
                combined = n * 10 + next_n
                if 10 <= combined <= 33:
                    result.append(combined)
                    i += 2
                    continue
            if 1 <= n <= 9:
                result.append(n)
        elif len(num_str) == 2:
            n = int(num_str)
            result.append(n)
        elif len(num_str) > 2:
            for j in range(0, len(num_str), 2):
                chunk = num_str[j:j+2]
                if len(chunk) == 2:
                    n = int(chunk)
                    if 1 <= n <= 33:
                        result.append(n)
                elif len(chunk) == 1 and j == len(num_str) - 1:
                    n = int(chunk)
                    if 1 <= n <= 9:
                        result.append(n)
        i += 1
    return result

def is_ssq_related(text):
    ssq_keywords = ["双色球", "红球", "蓝球", "篮球", "监球", "兰球", "红胆", "红拖"]
    match_count = sum(1 for kw in ssq_keywords if kw in text)
    return match_count >= 2

def is_valid_ball_numbers(nums):
    if len(nums) == 0:
        return False
    if len(nums) > 7:
        return False
    for n in nums:
        if not (1 <= n <= 33):
            return False
    return True

def parse_region_lines(lines):
    ball_lines = []
    
    for idx, line in enumerate(lines):
        red_match = re.search(r"红球[:：]\s*(.+)", line)
        red_dan_match = re.search(r"红胆[:：]\s*(.+)", line)
        red_tuo_match = re.search(r"红拖[:：]\s*(.+)", line)
        blue_match = re.search(r"(?:蓝|篮|监|兰)球[:：]\s*(.+)", line)
        ball_match = re.search(r"球[:：]\s*(.+)", line)
        
        if red_match:
            ball_lines.append({
                'idx': idx, 'line': line,
                'type': 'RED',
                'nums': extract_numbers(red_match.group(1)),
                'has_times': False
            })
        elif red_dan_match:
            ball_lines.append({
                'idx': idx, 'line': line,
                'type': 'RED_DAN',
                'nums': extract_numbers(red_dan_match.group(1)),
                'has_times': False
            })
        elif red_tuo_match:
            ball_lines.append({
                'idx': idx, 'line': line,
                'type': 'RED_TUO',
                'nums': extract_numbers(red_tuo_match.group(1)),
                'has_times': False
            })
        elif blue_match:
            blue_raw = blue_match.group(1)
            has_times = re.search(r'[\[\(C]?(\d+)\s*倍[\]\)]?', blue_raw) is not None
            if has_times:
                blue_clean = re.sub(r'[\[\(C]?\d+\s*倍[\]\)]?', '', blue_raw)
            else:
                blue_clean = blue_raw
            ball_lines.append({
                'idx': idx, 'line': line,
                'type': 'BLUE',
                'nums': extract_numbers(blue_clean),
                'has_times': has_times
            })
        elif ball_match:
            ball_raw = ball_match.group(1)
            has_times = re.search(r'[\[\(C]?(\d+)\s*倍[\]\)]?', ball_raw) is not None
            if has_times:
                ball_clean = re.sub(r'[\[\(C]?\d+\s*倍[\]\)]?', '', ball_raw)
                ball_lines.append({
                    'idx': idx, 'line': line,
                    'type': 'BALL_BLUE',
                    'nums': extract_numbers(ball_clean),
                    'has_times': True
                })
            else:
                nums = extract_numbers(ball_raw)
                ball_lines.append({
                    'idx': idx, 'line': line,
                    'type': 'BALL_RED',
                    'nums': nums,
                    'has_times': False
                })
        elif re.match(r'^\s*[\d\s]+\s*$', line):
            nums = extract_numbers(line)
            if len(nums) >= 1 and all(1 <= n <= 33 for n in nums):
                ball_lines.append({
                    'idx': idx, 'line': line,
                    'type': 'CONTINUE',
                    'nums': nums,
                    'has_times': False
                })
    
    all_groups = []
    current_group = None
    prev_has_blue = False
    
    for bl in ball_lines:
        btype = bl['type']
        nums = bl['nums']
        
        if btype in ('RED', 'BALL_RED', 'RED_DAN'):
            if current_group is not None:
                all_groups.append(current_group)
            current_group = {"red": list(nums), "blue": [], "times": 1}
            prev_has_blue = False
        
        elif btype == 'RED_TUO':
            if current_group is not None:
                current_group['red'].extend(nums)
            else:
                current_group = {"red": list(nums), "blue": [], "times": 1}
            prev_has_blue = False
        
        elif btype in ('BLUE', 'BALL_BLUE'):
            if current_group is not None:
                current_group['blue'].extend(nums)
                current_group['times'] = 1
                all_groups.append(current_group)
                current_group = None
                prev_has_blue = True
            else:
                # Find the most recent group that doesn't have blue yet
                found_target = False
                for g in reversed(all_groups):
                    if not g.get('blue'):
                        g['blue'].extend(nums)
                        g['times'] = 1
                        found_target = True
                        break
                if not found_target:
                    # No group needs blue - create standalone
                    current_group = {"red": [], "blue": list(nums), "times": 1}
                    all_groups.append(current_group)
                    current_group = None
                prev_has_blue = True
        
        elif btype == 'CONTINUE':
            if current_group is not None:
                current_group['red'].extend(nums)
            else:
                # Find the most recent group without enough red to extend
                for g in reversed(all_groups):
                    if len(g.get('red', [])) < 7 and not g.get('blue'):
                        g['red'].extend(nums)
                        break
    
    if current_group is not None and len(current_group['red']) >= 5:
        all_groups.append(current_group)
    
    return all_groups

def is_ball_line(text):
    if re.search(r"红球[:：]", text):
        return True
    if re.search(r"(?:蓝|篮|监|兰)球[:：]", text):
        return True
    if re.search(r"红胆[:：]", text):
        return True
    if re.search(r"红拖[:：]", text):
        return True
    if re.search(r"球[:：]", text):
        return True
    if re.match(r'^\s*[\d\s]+\s*$', text):
        nums = extract_numbers(text)
        if len(nums) >= 1 and all(1 <= n <= 33 for n in nums):
            return True
    return False

def parse_lottery_image(img_path):
    img = cv2.imread(img_path)
    if img is None:
        return []
    result = ocr.predict(img)
    
    item = result[0]
    rec_texts = item['rec_texts']
    rec_polys = item['rec_polys']
    
    import numpy as np
    
    lines_with_coords = []
    for i, (text, poly) in enumerate(zip(rec_texts, rec_polys)):
        x_min = int(np.min(poly[:, 0]))
        x_max = int(np.max(poly[:, 0]))
        y_min = int(np.min(poly[:, 1]))
        y_max = int(np.max(poly[:, 1]))
        y_center = (y_min + y_max) / 2
        x_center = (x_min + x_max) / 2
        lines_with_coords.append({
            'text': text,
            'x_min': x_min,
            'x_max': x_max,
            'y_min': y_min,
            'y_max': y_max,
            'x_center': x_center,
            'y_center': y_center
        })
    
    lines_with_coords.sort(key=lambda l: (l['y_center'], l['x_min']))
    
    full_text = "\n".join([l['text'] for l in lines_with_coords])
    if not is_ssq_related(full_text):
        return []
    
    ball_lines = [l for l in lines_with_coords if is_ball_line(l['text'])]
    
    if len(ball_lines) == 0:
        return []
    
    ball_lines.sort(key=lambda l: (l['y_center'], l['x_center']))
    
    y_threshold = 80
    y_clusters = []
    current_cluster = [ball_lines[0]]
    
    for i in range(1, len(ball_lines)):
        line = ball_lines[i]
        y_center = line['y_center']
        
        cluster_max_y = max(l['y_max'] for l in current_cluster)
        if y_center - cluster_max_y <= y_threshold:
            current_cluster.append(line)
        else:
            y_clusters.append(current_cluster)
            current_cluster = [line]
    
    y_clusters.append(current_cluster)
    
    all_groups = []
    
    for y_cluster in y_clusters:
        y_cluster.sort(key=lambda l: (l['x_min'], l['y_center']))
        
        x_threshold = 200
        x_clusters = []
        current_x_cluster = [y_cluster[0]]
        
        for i in range(1, len(y_cluster)):
            line = y_cluster[i]
            # Check if this line's x-range overlaps or is close to the current cluster's range
            cluster_x_max = max(l['x_max'] for l in current_x_cluster)
            line_x_min = line['x_min']
            
            # If the new line starts before or within the cluster's x-range (with tolerance)
            if line_x_min <= cluster_x_max + x_threshold:
                current_x_cluster.append(line)
            else:
                x_clusters.append(current_x_cluster)
                current_x_cluster = [line]
        
        x_clusters.append(current_x_cluster)
        
        # 如果有多列，检查是否需要合并
        # 当某列只有红球行或只有蓝球行时，尝试合并相邻列
        if len(x_clusters) > 1:
            merged_clusters = []
            for xc in x_clusters:
                # 检查是否有明确的红球/蓝球标记
                has_red_label = any(re.search(r"红球[:：]|红胆[:：]|红拖[:：]", l['text']) for l in xc)
                has_blue_label = any(re.search(r"(?:蓝|篮|监|兰)球[:：]", l['text']) for l in xc)
                has_red_ball = any(re.search(r"球[:：]", l['text']) for l in xc)
                
                # 如果当前列没有明确的红球标记，且前一列也没有
                # 或者当前列没有明确的蓝球标记，尝试合并
                if merged_clusters:
                    prev_has_red = any(re.search(r"红球[:：]|红胆[:：]|红拖[:：]", l['text']) for l in merged_clusters[-1])
                    prev_has_blue = any(re.search(r"(?:蓝|篮|监|兰)球[:：]", l['text']) for l in merged_clusters[-1])
                    
                    # 如果前一列有红球但没有蓝球，当前列有蓝球但没有红球，尝试合并
                    if prev_has_red and not prev_has_blue and has_blue_label and not has_red_label:
                        merged_clusters[-1].extend(xc)
                        continue
                    # 如果前一列有蓝球但没有红球，当前列有红球但没有蓝球，尝试合并
                    if prev_has_blue and not prev_has_red and has_red_label and not has_blue_label:
                        merged_clusters[-1].extend(xc)
                        continue
                
                merged_clusters.append(list(xc))
            x_clusters = merged_clusters
        
        for x_cluster in x_clusters:
            x_cluster.sort(key=lambda l: l['y_center'])
            x_lines = [l['text'] for l in x_cluster]
            groups = parse_region_lines(x_lines)
            all_groups.extend(groups)
    
    for g in all_groups:
        g["red"] = sorted(list(set(g["red"])))
        g["blue"] = sorted(list(set(g["blue"])))
        g["red"] = [x for x in g["red"] if 1 <= x <= 33]
        g["blue"] = [x for x in g["blue"] if 1 <= x <= 16]
    
    return all_groups

# ===================== 4. 批量遍历文件夹所有图片，汇总全部号码 =====================
def batch_parse_images(folder_path, save_name=LOTTERY_SUMMARY_PATH, save_interval=10):
    total_groups = []
    skipped = 0
    failed = 0
    processed = 0
    
    if not os.path.exists(folder_path):
        print(f"错误：目录不存在 {folder_path}")
        return total_groups
    
    all_files = os.listdir(folder_path)
    img_files = [f for f in all_files 
                 if f.lower().endswith((".jpg", ".png", ".jpeg")) 
                 and "_thumb" not in f.lower()]
    thumb_count = len([f for f in all_files if "_thumb" in f.lower()])
    print(f"发现 {len(img_files)} 张图片（已过滤 {thumb_count} 张缩略图），开始处理...\n")
    
    for idx, file_name in enumerate(img_files, 1):
        img_full_path = os.path.join(folder_path, file_name)
        try:
            groups = parse_lottery_image(img_full_path)
            if len(groups) == 0:
                skipped += 1
            else:
                for g in groups:
                    g["img_name"] = file_name
                    total_groups.append(g)
                processed += 1
                print(f"[{idx}/{len(img_files)}] 【{file_name}】识别到 {len(groups)} 组复式")
            
            if idx % save_interval == 0 and total_groups:
                save_progress(total_groups, save_name)
                print(f"  ↳ 已保存中间结果（{len(total_groups)} 组）")
        except Exception as e:
            failed += 1
            print(f"[{idx}/{len(img_files)}] 【{file_name}】处理失败: {str(e)}")
    
    if total_groups:
        save_progress(total_groups, save_name)
    
    print(f"\n====== 处理统计 ======")
    print(f"总图片数: {len(img_files)}")
    print(f"成功识别: {processed} 张")
    print(f"非双色球图片(跳过): {skipped} 张")
    print(f"处理失败: {failed} 张")
    print(f"总计识别到 {len(total_groups)} 组彩票复式")
    print(f"结果已保存至: {save_name}")
    return total_groups

def save_progress(all_groups, save_name):
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

# ===================== 主程序入口 =====================
if __name__ == "__main__":
    # 1. 修改为你存放彩票图片的文件夹路径
    # 默认测试图片目录
    # IMG_FOLDER = r"../lottery_img"
    
    # 微信图片缓存目录（如需使用请取消注释）
    IMG_FOLDER = r"/Users/zhangzhaochao/Library/Containers/com.tencent.xinWeChat/Data/Library/Application Support/com.tencent.xinWeChat/2.0b4.0.9/dd599815ed115ac82bd4effdfadab7a5/Message/MessageTemp/9f2fe70ab6257a9a669e2b4026904633/Image"
    
    # 2. 批量解析所有图片（自动过滤非双色球图片、缩略图，支持增量保存）
    all_lottery_groups = batch_parse_images(IMG_FOLDER, save_name=LOTTERY_SUMMARY_PATH, save_interval=10)

    # 3. 计算冷热号频次
    red_hot_sort, blue_hot_sort = calc_hot_cold(all_lottery_groups)

    # 4. 输出热度并生成推荐号码池
    red_pool, blue_pool = get_recommend(red_hot_sort, blue_hot_sort)
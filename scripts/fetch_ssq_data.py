#!/usr/bin/env python3
"""双色球历史数据获取脚本 - 支持按年份批量下载全部历史数据"""

import json
import re
import urllib.request
import ssl
import time
import shutil
from pathlib import Path
from html.parser import HTMLParser
from datetime import datetime, timedelta


def validate_record(record):
    """验证单条开奖记录的数据有效性"""
    try:
        # 验证红球: 6个不重复的01-33
        red_balls = record.get('red_balls', [])
        if len(red_balls) != 6:
            return False
        if len(set(red_balls)) != 6:  # 检查重复
            return False
        for ball in red_balls:
            num = int(ball)
            if num < 1 or num > 33:
                return False

        # 验证蓝球: 01-16
        blue_ball = record.get('blue_ball', '')
        if not blue_ball:
            return False
        blue_num = int(blue_ball)
        if blue_num < 1 or blue_num > 16:
            return False

        # 验证期号
        if not record.get('issue'):
            return False

        return True
    except (ValueError, TypeError):
        return False


class SSQParser(HTMLParser):
    
    def __init__(self):
        super().__init__()
        self.results = []
        self.current_row = None
        self.current_cell = None
        self.in_cell = False
        self.cell_index = 0
        self.current_class = ""
        
    def handle_starttag(self, tag, attrs):
        attrs_dict = dict(attrs)
        class_attr = attrs_dict.get('class', '')
        
        if tag == 'tr' and class_attr and 'hgt' in class_attr:
            self.current_row = {
                'issue': '',
                'red_balls': [],
                'blue_ball': '',
                'date': ''
            }
            self.cell_index = 0
            
        elif tag == 'td' and self.current_row is not None:
            self.in_cell = True
            self.current_class = class_attr
            self.current_cell = ""
            
        elif tag == 'a' and self.current_row is not None and self.cell_index == 0:
            title = attrs_dict.get('title', '')
            if title and '开奖日期' in title:
                date_match = re.search(r'(\d{4}-\d{2}-\d{2})', title)
                if date_match:
                    self.current_row['date'] = date_match.group(1)
                    
    def handle_endtag(self, tag):
        if tag == 'td' and self.in_cell and self.current_row is not None:
            self.in_cell = False
            
            if self.cell_index == 0 and self.current_cell:
                issue_match = re.search(r'(\d+)', self.current_cell)
                if issue_match:
                    self.current_row['issue'] = issue_match.group(1)
            
            # 红球区: 列1-33对应红球01-33, 蓝球区: 列34-49对应蓝球01-16
            elif 1 <= self.cell_index <= 33:
                if self.current_class and 'redqiu' in self.current_class:
                    ball_num = str(self.cell_index).zfill(2)
                    self.current_row['red_balls'].append(ball_num)
            
            elif 34 <= self.cell_index <= 49:
                if self.current_class and 'blueqiu3' in self.current_class:
                    ball_num = str(self.cell_index - 33).zfill(2)
                    self.current_row['blue_ball'] = ball_num
            
            self.cell_index += 1
            self.current_cell = None
            self.current_class = ""
            
        elif tag == 'tr' and self.current_row is not None:
            if (self.current_row['issue'] and 
                len(self.current_row['red_balls']) == 6 and 
                self.current_row['blue_ball']):
                self.current_row['red_balls'].sort()
                self.results.append(self.current_row)
            self.current_row = None
            
    def handle_data(self, data):
        if self.in_cell and self.current_row is not None:
            self.current_cell = data.strip()


def fetch_with_retry(url, max_retries=3, retry_delay=2):
    """带重试机制的HTTP请求"""
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE

    headers = {
        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
        'Accept-Language': 'zh-CN,zh;q=0.9,en;q=0.8',
    }

    request = urllib.request.Request(url, headers=headers)

    for attempt in range(max_retries):
        try:
            with urllib.request.urlopen(request, context=ctx, timeout=30) as response:
                return response.read().decode('utf-8', errors='ignore')
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(retry_delay * (attempt + 1))  # 递增延迟
            else:
                raise e
    return None


def fetch_by_year(year):
    url = f"https://tubiao.zhcw.com/tubiao/ssqNew/ssqJsp/ssqZongHeFengBuTuAsc.jsp?kj_year={year}"

    try:
        html = fetch_with_retry(url)
        if not html:
            return []
    except Exception as e:
        print(f"  获取{year}年数据失败: {e}")
        return []

    parser = SSQParser()
    parser.feed(html)
    return parser.results


def fetch_recent(periods=100):
    url = f"https://tubiao.zhcw.com/tubiao/ssqNew/ssqInc/ssqZongHeFengBuTuAscselect={periods}.html"

    try:
        html = fetch_with_retry(url)
        if not html:
            return []
    except Exception as e:
        print(f"获取最新数据失败: {e}")
        return []

    parser = SSQParser()
    parser.feed(html)
    return parser.results


def is_data_up_to_date(latest_record):
    """检查数据是否已是最新（基于开奖日期和星期）"""
    if not latest_record or not latest_record.get('date'):
        return False

    try:
        latest_date = datetime.strptime(latest_record['date'], '%Y-%m-%d')
        today = datetime.now()

        # 如果最新数据就是今天，已是最新
        if latest_date.date() == today.date():
            return True

        # 开奖日：周二(1)、周四(3)、周日(6)
        draw_days = {1, 3, 6}
        today_weekday = today.weekday()

        # 找到上一个开奖日
        days_back = 0
        for i in range(7):
            check_day = (today_weekday - i) % 7
            if check_day in draw_days:
                days_back = i
                break

        last_draw_date = today.date() - timedelta(days=days_back)

        # 如果最新数据日期 >= 上一个开奖日，认为已是最新
        if latest_date.date() >= last_draw_date:
            return True

        return False
    except Exception:
        return False


def load_history(filepath):
    if filepath.exists():
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    return []


def merge_data(base_data, new_data):
    """合并数据，去重并验证"""
    existing_issues = {item['issue'] for item in base_data}
    added = 0
    skipped = 0

    for item in new_data:
        if item['issue'] not in existing_issues:
            if validate_record(item):
                base_data.append(item)
                existing_issues.add(item['issue'])
                added += 1
            else:
                skipped += 1

    if skipped > 0:
        print(f"  (跳过 {skipped} 条无效数据)")

    base_data.sort(key=lambda x: x['issue'])
    return base_data, added


def analyze_data(data):
    if not data:
        return {}
    
    red_count = {str(i).zfill(2): 0 for i in range(1, 34)}
    blue_count = {str(i).zfill(2): 0 for i in range(1, 17)}
    red_missing = {str(i).zfill(2): 0 for i in range(1, 34)}
    blue_missing = {str(i).zfill(2): 0 for i in range(1, 17)}
    
    for item in data:
        for ball in item['red_balls']:
            red_count[ball] += 1
        blue_count[item['blue_ball']] += 1
    
    latest_data = list(reversed(data))
    
    for i in range(1, 34):
        ball = str(i).zfill(2)
        for idx, item in enumerate(latest_data):
            if ball in item['red_balls']:
                red_missing[ball] = idx
                break
        else:
            red_missing[ball] = len(data)
            
    for i in range(1, 17):
        ball = str(i).zfill(2)
        for idx, item in enumerate(latest_data):
            if ball == item['blue_ball']:
                blue_missing[ball] = idx
                break
        else:
            blue_missing[ball] = len(data)
    
    red_hot = sorted(red_count.items(), key=lambda x: x[1], reverse=True)[:10]
    blue_hot = sorted(blue_count.items(), key=lambda x: x[1], reverse=True)[:5]
    red_cold = sorted(red_missing.items(), key=lambda x: x[1], reverse=True)[:10]
    blue_cold = sorted(blue_missing.items(), key=lambda x: x[1], reverse=True)[:5]
    
    latest = data[-1] if data else None
    first = data[0] if data else None
    
    return {
        'total_periods': len(data),
        'first': first,
        'latest': latest,
        'red_frequency': red_count,
        'blue_frequency': blue_count,
        'red_missing': red_missing,
        'blue_missing': blue_missing,
        'red_hot': [x[0] for x in red_hot],
        'blue_hot': [x[0] for x in blue_hot],
        'red_cold': [x[0] for x in red_cold],
        'blue_cold': [x[0] for x in blue_cold],
    }


def main():
    output_dir = Path(__file__).parent.parent
    history_file = output_dir / "ssq_history.json"
    backup_file = output_dir / "ssq_history.json.bak"
    analysis_file = output_dir / "ssq_analysis.json"

    print("=" * 60)
    print("         双色球全量历史数据获取系统")
    print("=" * 60)

    all_data = load_history(history_file)
    before_count = len(all_data)
    print(f"\n现有历史数据: {before_count} 期")

    # 检查数据是否已是最新
    latest_record = all_data[-1] if all_data else None
    if latest_record:
        print(f"最新一期: {latest_record['issue']} ({latest_record['date']})")

    if is_data_up_to_date(latest_record):
        print("\n✓ 数据已是最新，无需更新")
        # 仍然生成分析文件（以防丢失）
        analysis = analyze_data(all_data)
        with open(analysis_file, 'w', encoding='utf-8') as f:
            json.dump(analysis, f, ensure_ascii=False, indent=2)
        print_summary(analysis, all_data)
        return

    print("\n检测到有新数据可获取...")

    # 备份现有数据
    if history_file.exists() and before_count > 0:
        shutil.copy(history_file, backup_file)
        print(f"已备份到: {backup_file.name}")

    # 自动计算年份范围：2003年至明年
    current_year = datetime.now().year
    years_to_fetch = list(range(2003, current_year + 2))

    print(f"\n开始获取 2003-{current_year + 1} 年历史数据...")

    for year in years_to_fetch:
        print(f"  获取 {year} 年数据...", end=" ")
        year_data = fetch_by_year(year)

        if year_data:
            all_data, added = merge_data(all_data, year_data)
            print(f"获取{len(year_data)}期, 新增{added}期")
        else:
            print("无数据或获取失败")

        time.sleep(1)  # 请求间隔1秒

    print(f"\n获取最新100期数据...")
    recent_data = fetch_recent(100)
    if recent_data:
        all_data, added = merge_data(all_data, recent_data)
        print(f"新增: {added} 期")

    with open(history_file, 'w', encoding='utf-8') as f:
        json.dump(all_data, f, ensure_ascii=False, indent=2)

    after_count = len(all_data)
    print(f"\n数据更新完成!")
    print(f"  更新前: {before_count} 期")
    print(f"  更新后: {after_count} 期")
    print(f"  新增: {after_count - before_count} 期")
    print(f"  保存到: {history_file}")

    analysis = analyze_data(all_data)
    with open(analysis_file, 'w', encoding='utf-8') as f:
        json.dump(analysis, f, ensure_ascii=False, indent=2)

    print_summary(analysis, all_data)


def print_summary(analysis, all_data):
    """打印数据摘要"""
    print("\n" + "=" * 60)
    print("                 数 据 摘 要")
    print("=" * 60)

    if analysis.get('first'):
        first = analysis['first']
        print(f"\n【最早一期】 {first['issue']} ({first['date']})")

    if analysis.get('latest'):
        latest = analysis['latest']
        print(f"【最新一期】 {latest['issue']} ({latest['date']})")
        print(f"  红球: {' '.join(latest['red_balls'])}")
        print(f"  蓝球: {latest['blue_ball']}")

    print(f"\n【统计范围】 共 {len(all_data)} 期数据")

    print(f"\n【红球热号】 (全历史)")
    print(f"  {' '.join(analysis['red_hot'])}")

    print(f"\n【红球冷号】 (遗漏期数)")
    cold_with_missing = [(b, analysis['red_missing'][b]) for b in analysis['red_cold'][:5]]
    print(f"  {' '.join([f'{b}({m}期)' for b, m in cold_with_missing])}")

    print(f"\n【蓝球热号】")
    print(f"  {' '.join(analysis['blue_hot'])}")

    print(f"\n【蓝球冷号】 (遗漏期数)")
    blue_cold_with_missing = [(b, analysis['blue_missing'][b]) for b in analysis['blue_cold'][:5]]
    print(f"  {' '.join([f'{b}({m}期)' for b, m in blue_cold_with_missing])}")

    print("\n" + "=" * 60)


if __name__ == '__main__':
    main()

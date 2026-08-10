#!/usr/bin/env python3
"""双色球预测大模型调用脚本 - 支持多种主流大模型API"""

import json
import os
import sys
import argparse
from pathlib import Path

try:
    import requests
except ImportError:
    print("错误：缺少 requests 库，请先安装: pip install requests")
    sys.exit(1)


class LLMClient:
    """多平台大模型客户端封装"""
    
    def __init__(self, model_type="openai", api_key=None, base_url=None):
        self.model_type = model_type.lower()
        self.api_key = api_key or os.environ.get(f"{model_type.upper()}_API_KEY")
        self.base_url = base_url or os.environ.get(f"{model_type.upper()}_BASE_URL")
        
        if not self.api_key:
            raise ValueError(f"请设置 {model_type.upper()}_API_KEY 环境变量")
        
        self.models = {
            "openai": {
                "default_model": "gpt-4o",
                "endpoint": "/v1/chat/completions",
                "headers": {"Authorization": f"Bearer {self.api_key}"},
            },
            "anthropic": {
                "default_model": "claude-3-sonnet-20240229",
                "endpoint": "/v1/messages",
                "headers": {"x-api-key": self.api_key, "anthropic-version": "2023-06-01"},
            },
            "qwen": {
                "default_model": "qwen-turbo",
                "endpoint": "/v1/chat/completions",
                "headers": {"Authorization": f"Bearer {self.api_key}"},
            },
            "deepseek": {
                "default_model": "deepseek-chat",
                "endpoint": "/v1/chat/completions",
                "headers": {"Authorization": f"Bearer {self.api_key}"},
            },
            "zhipu": {
                "default_model": "glm-4",
                "endpoint": "/v1/chat/completions",
                "headers": {"Authorization": f"Bearer {self.api_key}"},
            },
        }
        
        if self.model_type not in self.models:
            raise ValueError(f"不支持的模型类型: {self.model_type}")
        
        self.config = self.models[self.model_type]
        
        if self.base_url:
            self.url = f"{self.base_url.rstrip('/')}{self.config['endpoint']}"
        else:
            if self.model_type == "openai":
                self.url = f"https://api.openai.com{self.config['endpoint']}"
            elif self.model_type == "anthropic":
                self.url = f"https://api.anthropic.com{self.config['endpoint']}"
            elif self.model_type == "qwen":
                self.url = f"https://dashscope.aliyuncs.com{self.config['endpoint']}"
            elif self.model_type == "deepseek":
                self.url = f"https://api.deepseek.com{self.config['endpoint']}"
            elif self.model_type == "zhipu":
                self.url = f"https://open.bigmodel.cn{self.config['endpoint']}"
    
    def build_request(self, messages, model=None):
        """构建API请求"""
        model_name = model or self.config["default_model"]
        
        if self.model_type == "anthropic":
            return {
                "model": model_name,
                "max_tokens": 4096,
                "temperature": 0.7,
                "system": messages[0]["content"] if messages[0]["role"] == "system" else "",
                "messages": [m for m in messages if m["role"] != "system"],
            }
        
        return {
            "model": model_name,
            "max_tokens": 4096,
            "temperature": 0.7,
            "messages": messages,
        }
    
    def extract_response(self, response):
        """提取响应内容"""
        if self.model_type == "anthropic":
            return response.get("content", [{}])[0].get("text", "")
        
        return response.get("choices", [{}])[0].get("message", {}).get("content", "")
    
    def chat(self, messages, model=None):
        """发送聊天请求"""
        payload = self.build_request(messages, model)
        
        try:
            response = requests.post(
                self.url,
                headers=self.config["headers"],
                json=payload,
                timeout=60,
            )
            response.raise_for_status()
            return self.extract_response(response.json())
        except requests.exceptions.RequestException as e:
            print(f"API 请求失败: {e}")
            if response:
                try:
                    print(f"响应内容: {response.text}")
                except:
                    pass
            return None


def load_data(project_root):
    """加载历史数据和分析结果"""
    history_file = project_root / "ssq_history.json"
    analysis_file = project_root / "ssq_analysis.json"
    
    if not history_file.exists():
        print("错误：ssq_history.json 不存在，请先运行 fetch_ssq_data.py")
        sys.exit(1)
    
    with open(history_file, 'r', encoding='utf-8') as f:
        history_data = json.load(f)
    
    with open(analysis_file, 'r', encoding='utf-8') as f:
        analysis_data = json.load(f)
    
    return history_data, analysis_data


def get_last_n_periods(data, n=10):
    """获取最近N期数据"""
    return data[-n:]


def format_recent_data(recent_data):
    """格式化最近数据用于提示词"""
    lines = []
    for item in reversed(recent_data):
        lines.append(f"期号: {item['issue']} | 日期: {item['date']} | 红球: {' '.join(item['red_balls'])} | 蓝球: {item['blue_ball']}")
    return "\n".join(lines)


def build_system_prompt(project_root, analysis_data):
    """构建系统提示词"""
    skill_file = project_root / "SKILL.md"
    with open(skill_file, 'r', encoding='utf-8') as f:
        skill_content = f.read()
    
    latest = analysis_data.get('latest', {})
    if latest:
        last_period = f"""
【最新一期数据】
期号：{latest.get('issue', '')}
日期：{latest.get('date', '')}
红球：{' '.join(latest.get('red_balls', []))}
蓝球：{latest.get('blue_ball', '')}
"""
    else:
        last_period = ""
    
    stats_info = f"""
【统计摘要】
总期数：{analysis_data.get('total_periods', 0)} 期
红球热号(Top10)：{' '.join(analysis_data.get('red_hot', []))}
红球冷号(遗漏最大)：{' '.join(analysis_data.get('red_cold', []))}
蓝球热号(Top5)：{' '.join(analysis_data.get('blue_hot', []))}
蓝球冷号(遗漏最大)：{' '.join(analysis_data.get('blue_cold', []))}

【遗漏详情】
红球遗漏值：{json.dumps(analysis_data.get('red_missing', {}), ensure_ascii=False)}
蓝球遗漏值：{json.dumps(analysis_data.get('blue_missing', {}), ensure_ascii=False)}

【频率统计】
红球频率：{json.dumps(analysis_data.get('red_frequency', {}), ensure_ascii=False)}
蓝球频率：{json.dumps(analysis_data.get('blue_frequency', {}), ensure_ascii=False)}
"""
    
    return f"""{skill_content}

{last_period}

{stats_info}

请基于以上数据和规则，完成双色球预测分析报告。"""


def build_user_prompt(birth_date=None, random_nums=None):
    """构建用户提示词"""
    prompt = "帮我预测下期双色球号码。"
    
    if birth_date:
        prompt += f"\n\n我的出生年月日是：{birth_date}"
    
    if random_nums:
        prompt += f"\n\n随机数字：{', '.join(random_nums)}"
    
    if not birth_date or not random_nums:
        prompt += """
如果缺少我的出生年月日和随机数字，请在报告中注明："未提供个人信息，易经玄学分析部分省略"。
"""
    
    return prompt


def main():
    parser = argparse.ArgumentParser(description='双色球预测大模型调用脚本')
    parser.add_argument('--model', type=str, default='openai', 
                        choices=['openai', 'anthropic', 'qwen', 'deepseek', 'zhipu'],
                        help='大模型类型')
    parser.add_argument('--api-key', type=str, help='API密钥（优先使用环境变量）')
    parser.add_argument('--base-url', type=str, help='自定义API基础URL')
    parser.add_argument('--birth-date', type=str, help='出生年月日（如：1990年5月15日）')
    parser.add_argument('--random-nums', type=str, help='3个随机数字，逗号分隔（如：8,16,24）')
    parser.add_argument('--auto-update', action='store_true', help='自动更新数据')
    parser.add_argument('--output', type=str, help='输出文件路径')
    args = parser.parse_args()
    
    project_root = Path(__file__).parent.parent
    
    if args.auto_update:
        print("=" * 60)
        print("         自动更新历史数据")
        print("=" * 60)
        os.system(f"python {project_root / 'scripts' / 'fetch_ssq_data.py'}")
        print()
    
    print("=" * 60)
    print("         加载数据并构建分析提示")
    print("=" * 60)
    
    history_data, analysis_data = load_data(project_root)
    
    recent_data = get_last_n_periods(history_data, 10)
    print(f"已加载 {len(history_data)} 期历史数据")
    print(f"最新期号: {analysis_data.get('latest', {}).get('issue', '')}")
    print()
    
    random_nums_list = None
    if args.random_nums:
        random_nums_list = [n.strip() for n in args.random_nums.split(',')]
    
    print("=" * 60)
    print(f"         调用 {args.model.upper()} 大模型分析")
    print("=" * 60)
    
    try:
        client = LLMClient(
            model_type=args.model,
            api_key=args.api_key,
            base_url=args.base_url,
        )
        
        system_prompt = build_system_prompt(project_root, analysis_data)
        user_prompt = build_user_prompt(args.birth_date, random_nums_list)
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        
        print("正在生成分析报告...")
        response = client.chat(messages)
        
        if response:
            print("\n" + "=" * 60)
            print("         双色球预测分析报告")
            print("=" * 60)
            print(response)
            print("\n" + "=" * 60)
            
            if args.output:
                output_path = Path(args.output)
                with open(output_path, 'w', encoding='utf-8') as f:
                    f.write(response)
                print(f"报告已保存到: {output_path}")
        else:
            print("大模型响应为空，请检查API配置")
            
    except Exception as e:
        print(f"错误: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()
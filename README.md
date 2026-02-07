# 双色球预测大师 (SSQ Predictor)

中国福利彩票双色球号码预测分析技能，融合数理统计与传统智慧。

## 功能特点

- **个人化分析**：基于出生日期推算生肖、命数、八字、喜用神
- **多维度分析**：概率统计、物理学视角、易经玄学三位一体
- **历史数据**：自动获取 2003 年至今全部开奖数据（3400+ 期）
- **智能推荐**：热号冷号、重号邻号、五行干支综合研判
- **数据智能更新**：自动检测是否需要更新，带重试机制

## 目录结构

```
ssq-predictor/
├── SKILL.md              # 技能定义文件（Agent 入口）
├── README.md             # 本文件
├── ssq_history.json      # 历史开奖数据（自动更新）
├── ssq_analysis.json     # 统计分析结果
├── scripts/
│   └── fetch_ssq_data.py # 数据获取脚本
└── references/
    ├── lottery-analysis.md    # 彩票分析术语
    ├── popular-strategies.md  # 彩民实战策略
    ├── probability-theory.md  # 概率论与统计学
    ├── physics-randomness.md  # 物理学与随机性
    ├── yijing-numerology.md   # 易经玄学（含个人化分析）
    └── betting-strategies.md  # 投注策略
```

---

## 给人类：手动安装与维护

### 前置要求

- Python 3.6+
- 支持 Agent Skill 的编辑器（Claude Code、Cursor、Windsurf、Codex CLI 等）

### 安装步骤

1. **克隆仓库**

```bash
git clone git@github.com:ma-pony/ssq-predictor.git
```

2. **复制到对应编辑器的 skill 目录**

不同编辑器的 skill 安装位置不同，请根据你使用的编辑器选择：

| 编辑器 | 全局 Skill 路径 | 项目级 Skill 路径 |
|--------|-----------------|-------------------|
| **Claude Code** | `~/.claude/skills/` | `.claude/skills/` |
| **Cursor** | 不支持全局 skill | `.cursor/rules/` |
| **Windsurf** | `~/.codeium/windsurf/memories/global_rules.md` | `.windsurf/rules/` |
| **Codex CLI** | `~/.codex/skills/` 或 `~/.agents/skills/` | `.agents/skills/` |

**Claude Code 安装示例：**

```bash
cp -r ssq-predictor ~/.claude/skills/ssq-predictor
```

**Cursor 安装示例：**

```bash
# Cursor 仅支持项目级规则，需将 SKILL.md 复制为 .mdc 规则文件
mkdir -p .cursor/rules
cp ssq-predictor/SKILL.md .cursor/rules/ssq-predictor.mdc
```

**Windsurf 安装示例：**

```bash
# 项目级安装
mkdir -p .windsurf/rules
cp ssq-predictor/SKILL.md .windsurf/rules/ssq-predictor.md
```

**Codex CLI 安装示例：**

```bash
cp -r ssq-predictor ~/.codex/skills/ssq-predictor
# 或项目级
cp -r ssq-predictor .agents/skills/ssq-predictor
```

### 日常维护

**更新历史数据：**

```bash
cd ~/.claude/skills/ssq-predictor  # 或你的安装路径
python3 scripts/fetch_ssq_data.py
```

脚本特性：
- 自动检测数据是否最新（开奖日：周二/四/日），无需重复获取
- 备份现有数据到 `ssq_history.json.bak`
- 获取 2003 年至今全部历史数据
- 验证数据有效性（红球 01-33，蓝球 01-16）
- 网络请求带重试机制（3 次重试，递增延迟）

**更新 skill 本身：**

```bash
cd ~/.claude/skills/ssq-predictor  # 或你的安装路径
git pull origin main
```

### 使用方法

在编辑器中使用以下触发词即可启动预测：

- "双色球"、"彩票预测"、"选号"、"下期号码"、"红球蓝球"

预测时需提供：
- **出生年月日**（如：1990年5月15日）
- **随机 3 个数字**（1-33 之间，用于梅花易数起卦）

---

## 给 Agent：自动安装指南

当用户要求你安装此 skill 时，请按以下步骤操作：

### 1. 克隆仓库

```bash
git clone git@github.com:ma-pony/ssq-predictor.git /tmp/ssq-predictor
```

### 2. 检测当前编辑器环境并安装到正确路径

根据你所在的运行环境，选择对应的安装方式：

**Claude Code：**

```bash
mkdir -p ~/.claude/skills
cp -r /tmp/ssq-predictor ~/.claude/skills/ssq-predictor
```

**Cursor（项目级）：**

```bash
# Cursor 使用 .mdc 格式的规则文件，不支持完整 skill 目录结构
# 需要将 SKILL.md 及引用的资源文件复制到项目中
mkdir -p .cursor/rules
cp /tmp/ssq-predictor/SKILL.md .cursor/rules/ssq-predictor.mdc
# 同时复制数据和脚本到项目可访问的位置
cp -r /tmp/ssq-predictor/scripts ./ssq-predictor-scripts
cp -r /tmp/ssq-predictor/references ./ssq-predictor-references
cp /tmp/ssq-predictor/ssq_history.json ./ssq_history.json
cp /tmp/ssq-predictor/ssq_analysis.json ./ssq_analysis.json
```

**Windsurf（项目级）：**

```bash
mkdir -p .windsurf/rules
cp /tmp/ssq-predictor/SKILL.md .windsurf/rules/ssq-predictor.md
# 同时复制数据和脚本到项目可访问的位置
cp -r /tmp/ssq-predictor/scripts ./ssq-predictor-scripts
cp -r /tmp/ssq-predictor/references ./ssq-predictor-references
cp /tmp/ssq-predictor/ssq_history.json ./ssq_history.json
cp /tmp/ssq-predictor/ssq_analysis.json ./ssq_analysis.json
```

**Codex CLI：**

```bash
mkdir -p ~/.codex/skills
cp -r /tmp/ssq-predictor ~/.codex/skills/ssq-predictor
```

### 3. 清理临时文件

```bash
rm -rf /tmp/ssq-predictor
```

### 4. 验证安装

安装完成后，尝试运行数据更新脚本确认环境正常：

```bash
python3 <skill安装路径>/scripts/fetch_ssq_data.py
```

### 环境检测提示

如果你不确定当前运行在哪个编辑器中，可以通过以下方式判断：

- 存在 `~/.claude/` 目录 → Claude Code
- 存在 `.cursor/` 目录 → Cursor
- 存在 `.windsurf/` 目录 → Windsurf
- 存在 `~/.codex/` 目录 → Codex CLI

---

## 数据来源

历史数据来自中国福利彩票官方图表网站 (tubiao.zhcw.com)。

## 免责声明

本技能仅供娱乐参考，彩票开奖完全随机，任何分析方法都无法保证中奖。请理性购彩，量力而行，切勿沉迷。

---

*小投怡情，大投伤身，佛系购彩最安心*

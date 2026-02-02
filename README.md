# AI学术助手

一款基于多智能体系统的学术论文和开源项目检索分析工具。

## 功能特点

### 🧠 智能搜索（主界面）

- 通过多轮对话了解你的研究需求
- 自动构建搜索策略并筛选结果
- 支持切换到普通搜索模式

### 🔍 普通搜索

- 直接输入关键词搜索arXiv论文和GitHub项目
- 支持时间范围和结果数量设置
- 可选择搜索来源（arXiv/GitHub）

### 🎲 闲逛热门

- 发现近3天内的热门项目和论文

### 📊 多智能体深度分析

- **arXiv论文分析**：大脑Agent + 方法理解Agent + 实验分析Agent + 审稿人Agent
- **GitHub项目分析**：大脑Agent + 架构分析Agent + 代码分析Agent + 使用分析Agent
- **相关研究分析**：自动搜索近3年相关论文，对比技术框架和实验差异

### 其他功能

- 支持多选/全选批量分析
- 支持最多2000条搜索结果（自动分批请求）
- 下载arXiv论文PDF
- 现代化Dracula深色主题界面

## 安装

```bash
conda env create -f environment.yml
conda activate arxiv_helper
```

## 使用

```bash
python src/main.py
```

或双击 `run.bat`

### 配置API

点击左上角「⚙️ 设置」按钮进行配置：

| 配置项 | 说明 |
|--------|------|
| Base URL | 大模型API地址（兼容OpenAI格式） |
| API Key | API密钥 |
| Model | 模型名称（如gpt-4, claude-3等） |
| GitHub Token | 可选，提高GitHub API限额 |

### 搜索模式切换

界面左侧提供两个搜索模式按钮：
- 「🧠 智能搜索」- 对话式搜索（默认）
- 「🔍 普通搜索」- 关键词直接搜索

### 智能搜索使用方法

1. 启动程序后默认进入智能搜索界面
2. 回答助手的问题（研究方向、目标、资源等）
3. 信息收集完成后点击「开始搜索」
4. 系统自动构建搜索策略、搜索并筛选结果

### 分析功能

- **单项分析**：选中一个结果，点击「分析选中」
- **批量分析**：Ctrl+点击多选，点击「分析选中」
- **相关研究**：选中一个结果，点击「查找相关研究」

## 打包

```bash
build.bat
```

生成的exe在 `dist\AI学术助手\` 目录

## 自定义Prompt

修改 [llm_client.py](src/llm_client.py) 中的prompt：

| Prompt | 用途 |
|--------|------|
| `BRAIN_PLAN_PROMPT` | 大脑Agent任务规划 |
| `METHOD_AGENT_PROMPT` | 方法理解分析 |
| `EXPERIMENT_AGENT_PROMPT` | 实验分析 |
| `JUDGER_AGENT_PROMPT` | 审稿人评审 |
| `INTERVIEW_PROMPT` | 智能搜索对话 |
| `FILTER_RESULTS_PROMPT` | 搜索结果筛选 |

## 项目结构

```
arxiv_helper/
├── src/
│   ├── main.py          # 主程序和GUI
│   ├── api_client.py    # arXiv/GitHub API
│   └── llm_client.py    # 多智能体系统
├── environment.yml      # Conda环境
├── run.bat              # 启动脚本
└── build.bat            # 打包脚本
```

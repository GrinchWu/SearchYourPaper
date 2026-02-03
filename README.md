# AI学术助手 - Multi-Agent System

一款基于多智能体系统的学术论文和开源项目检索分析工具，支持 arXiv、GitHub、HuggingFace、ModelScope 四大平台。

## 功能特点

### 🧠 智能搜索
- 通过多轮对话（最多3轮）了解你的研究需求
- 自动构建搜索策略并筛选结果
- 支持选择搜索数据源（arXiv/GitHub/HuggingFace/ModelScope）
- 信息收集完成后自动进入搜索就绪状态

### 🔍 普通搜索
- 直接输入关键词搜索
- 支持时间范围设置（昨天/过去一周/一个月/三个月/一年/自定义）
- 支持最多2000条搜索结果（自动分批请求）
- 可选择搜索来源组合

### 🎲 闲逛热门
- 发现近3天内的热门项目和论文
- 支持"新项目"和"活跃项目"两种模式
- 可选择数据源（arXiv/GitHub/HuggingFace/ModelScope）
- 可调整搜索数量（10-100条）

### 📊 多智能体深度分析

**arXiv论文分析系统：**
- 大脑Agent：任务规划和协调
- 方法理解Agent：分析核心方法和技术原理
- 实验分析Agent：分析实验设计和结果
- 审稿人Agent：批判性评审

**GitHub/HuggingFace/ModelScope项目分析系统：**
- 大脑Agent：任务规划和协调
- 架构分析Agent：分析项目架构
- 代码分析Agent：分析核心代码
- 使用分析Agent：分析使用方法

**相关研究分析：**
- 自动搜索近3年相关论文
- 对比技术框架和实验差异
- 生成综合分析报告

### 🖼️ 视觉分析（多模态）
- 自动从PDF论文中提取关键图片
- 从GitHub README中提取架构图
- 支持多模态模型（GPT-4o、Claude-3等）进行图片分析
- 勾选"图片分析"选项启用

### 📝 Markdown渲染
- 分析结果支持Markdown格式渲染
- 支持LaTeX数学公式显示
- 支持代码高亮和表格

### ⚡ 其他功能
- 取消搜索：搜索过程中可随时取消
- 自动续写：输出被截断时自动继续（最多3次）
- 批量分析：支持多选/全选批量分析
- PDF下载：支持下载arXiv论文PDF
- 深色主题：现代化Dracula风格界面

## 安装

### 方式一：Conda环境（推荐）

```bash
conda env create -f environment.yml
conda activate arxiv_helper
```

### 方式二：pip安装

```bash
pip install arxiv PyGithub openai httpx PyQt6 PyQt6-WebEngine requests pymupdf huggingface_hub markdown
```

## 使用

### 启动程序

```bash
python src/main.py
```

或双击 `run.bat`（Windows）

### 配置API

点击左上角「⚙️ 设置」按钮进行配置：

| 配置项 | 说明 | 示例 |
|--------|------|------|
| Base URL | 大模型API地址（兼容OpenAI格式） | `https://api.openai.com/v1` |
| API Key | API密钥 | `sk-xxx...` |
| Model | 模型名称 | `gpt-4o`, `claude-3-5-sonnet` |
| GitHub Token | 可选，提高GitHub API限额 | `ghp_xxx...` |

**支持的多模态模型（用于图片分析）：**
- OpenAI: `gpt-4-vision`, `gpt-4-turbo`, `gpt-4o`, `gpt-4o-mini`
- Anthropic: `claude-3-opus`, `claude-3-sonnet`, `claude-3-haiku`, `claude-3.5-sonnet`
- Google: `gemini-pro-vision`, `gemini-1.5-pro`, `gemini-1.5-flash`, `gemini-2`

### 界面说明

```
┌─────────────────────────────────────────────────────────────────┐
│ [⚙️设置]              类型:[新项目▼] 数量:[30] [🎲闲逛热门]      │
├─────────────────────────────────────────────────────────────────┤
│ 闲逛来源: [✓]arXiv [✓]GitHub [✓]HuggingFace [✓]ModelScope       │
├─────────────────────────────────────────────────────────────────┤
│ [🧠智能搜索] [🔍普通搜索]                                        │
├─────────────────────────────────────────────────────────────────┤
│ 搜索来源: [✓]arXiv [✓]GitHub [✓]HuggingFace [✓]ModelScope       │
├─────────────────────────────────────────────────────────────────┤
│ ┌─────────────────────────────────────────────────────────────┐ │
│ │ 智能搜索对话区域 / 普通搜索输入区域                          │ │
│ └─────────────────────────────────────────────────────────────┘ │
│ [输入框] [发送] [开始搜索] [❌取消] [重置]                       │
├─────────────────────────────────────────────────────────────────┤
│ 📋 搜索结果                                                     │
│ [全选] [取消全选]                              共 0 条结果      │
│ ┌─────────────────────────────────────────────────────────────┐ │
│ │ 结果列表                                                     │ │
│ └─────────────────────────────────────────────────────────────┘ │
│ [📊分析选中] [🖼️图片分析] [📥下载] [🔗打开]                     │
└─────────────────────────────────────────────────────────────────┘
```

### 使用流程

#### 智能搜索（推荐）

1. 启动程序后默认进入智能搜索界面
2. 选择要搜索的数据源（默认全选）
3. 回答助手的问题（研究方向、目标、资源等）
4. 最多3轮对话后自动进入搜索就绪状态
5. 点击「开始搜索」执行搜索
6. 如需取消，点击「❌ 取消」按钮

#### 普通搜索

1. 点击「🔍 普通搜索」切换模式
2. 输入搜索关键词
3. 选择时间范围和结果数量
4. 勾选要搜索的数据源
5. 点击「🔍 搜索」或按回车

#### 闲逛热门

1. 在普通搜索模式下输入关键词（可选，默认"machine learning"）
2. 选择类型：新项目（最近3天创建）或活跃项目（最近3天更新）
3. 设置搜索数量
4. 勾选要搜索的数据源
5. 点击「🎲 闲逛热门」

### 分析功能

| 操作 | 说明 |
|------|------|
| 单项分析 | 选中一个结果，点击「📊 分析选中」 |
| 批量分析 | Ctrl+点击多选，点击「📊 分析选中」 |
| 图片分析 | 勾选「🖼️ 图片分析」后再分析（需多模态模型） |
| 相关研究 | 选中一个结果，点击「🔍 查找相关研究」 |

### 结果图标说明

| 图标 | 来源 | 附加信息 |
|------|------|----------|
| 📄 | arXiv | 发布日期 |
| 📦/🔥 | GitHub | Star数量 |
| 🤗 | HuggingFace | 下载量 |
| 🔮 | ModelScope | 下载量 |

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
| `ARCHITECT_AGENT_PROMPT` | 架构分析 |
| `CODE_ANALYST_PROMPT` | 代码分析 |
| `USAGE_AGENT_PROMPT` | 使用分析 |
| `INTERVIEW_PROMPT` | 智能搜索对话 |
| `FILTER_RESULTS_PROMPT` | 搜索结果筛选 |
| `VISION_AGENT_PROMPT` | 视觉分析 |

## 项目结构

```
arxiv_helper/
├── src/
│   ├── main.py          # 主程序和GUI
│   ├── api_client.py    # arXiv/GitHub/HuggingFace/ModelScope API
│   └── llm_client.py    # 多智能体系统
├── environment.yml      # Conda环境配置
├── run.bat              # 启动脚本
├── build.bat            # 打包脚本
└── README.md            # 说明文档
```

## 依赖说明

| 依赖 | 用途 |
|------|------|
| PyQt6 | GUI框架 |
| PyQt6-WebEngine | Markdown/数学公式渲染 |
| arxiv | arXiv API |
| PyGithub | GitHub API |
| huggingface_hub | HuggingFace API |
| openai | LLM API调用 |
| pymupdf | PDF图片提取 |
| markdown | Markdown解析 |

## 常见问题

**Q: 搜索时出错怎么办？**
A: 点击「❌ 取消」按钮取消当前搜索，检查网络连接后重试。

**Q: 分析结果被截断？**
A: 系统会自动续写，最多重试3次。如仍不完整，可能是内容过长。

**Q: 图片分析不工作？**
A: 确保使用多模态模型（如gpt-4o），并勾选「🖼️ 图片分析」选项。

**Q: GitHub搜索限制？**
A: 配置GitHub Token可提高API限额。

**Q: 数学公式不显示？**
A: 需要网络连接加载MathJax库。

## License

MIT License

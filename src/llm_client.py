from openai import OpenAI
from typing import Optional
import json

# 多模态模型列表
MULTIMODAL_MODELS = [
    "gpt-4-vision", "gpt-4-turbo", "gpt-4o", "gpt-4o-mini",
    "claude-3-opus", "claude-3-sonnet", "claude-3-haiku", "claude-3.5-sonnet", "claude-3-5-sonnet",
    "gemini-pro-vision", "gemini-1.5-pro", "gemini-1.5-flash", "gemini-2"
]

def is_multimodal_model(model: str) -> bool:
    """检查模型是否支持多模态"""
    model_lower = model.lower()
    return any(m in model_lower for m in MULTIMODAL_MODELS)

class BaseAgent:
    """智能体基类 - 支持多模态"""
    def __init__(self, client: OpenAI, model: str, name: str, role: str):
        self.client, self.model, self.name, self.role = client, model, name, role
        self.supports_vision = is_multimodal_model(model)

    def think(self, prompt: str, content: str, history: list = None, images: list = None, max_retries: int = 3) -> str:
        """
        思考方法 - 支持图片输入和自动续写
        images: [{"url": "data:image/png;base64,xxx"}, ...]
        max_retries: 输出被截断时最多重试次数
        """
        messages = [{"role": "system", "content": f"你是{self.name}，{self.role}\n\n{prompt}"}]
        if history:
            messages.extend(history)

        # 构建用户消息（支持多模态）
        if images and self.supports_vision:
            user_content = [{"type": "text", "text": content}]
            for img in images:
                user_content.append({
                    "type": "image_url",
                    "image_url": {"url": img["url"], "detail": "high"}
                })
            messages.append({"role": "user", "content": user_content})
        else:
            messages.append({"role": "user", "content": content})

        full_response = ""
        for attempt in range(max_retries):
            response = self.client.chat.completions.create(model=self.model, messages=messages, temperature=0.3)
            chunk = response.choices[0].message.content or ""
            full_response += chunk

            # 检查是否被截断（finish_reason 为 length 表示达到 token 限制）
            finish_reason = response.choices[0].finish_reason
            if finish_reason != "length":
                break

            # 被截断了，添加助手回复并请求继续
            messages.append({"role": "assistant", "content": chunk})
            messages.append({"role": "user", "content": "请继续，从你上次停止的地方继续输出，不要重复已输出的内容。"})

        return full_response

class MultiAgentSystem:
    """多智能体系统基类"""
    def __init__(self, base_url: str, api_key: str, model: str):
        self.client = OpenAI(base_url=base_url, api_key=api_key)
        self.model = model
        self.supports_vision = is_multimodal_model(model)

    def create_agent(self, name: str, role: str) -> BaseAgent:
        return BaseAgent(self.client, self.model, name, role)

# ==================== 视觉分析Agent ====================

VISION_AGENT_PROMPT = """作为视觉分析Agent，你需要分析图片内容并提供专业解读。

对于架构图/流程图：
1. 识别图中的主要组件和模块
2. 解释组件之间的关系和数据流向
3. 总结整体架构设计思路

对于实验结果图/表格：
1. 识别图表类型（折线图、柱状图、表格等）
2. 提取关键数据点和趋势
3. 解读实验结论

对于其他图片：
1. 描述图片主要内容
2. 分析其在论文/项目中的作用
3. 提取关键信息

请用中文输出分析结果，结构清晰。"""

class VisionAnalysisAgent(BaseAgent):
    """视觉分析智能体"""
    def __init__(self, client: OpenAI, model: str):
        super().__init__(client, model, "视觉分析Agent",
                        "专注于分析架构图、实验结果图、表格等视觉内容的多模态专家")

    def analyze_images(self, images: list, context: str = "") -> str:
        """分析图片列表"""
        if not images or not self.supports_vision:
            return ""
        content = f"请分析以下图片。\n\n背景信息：{context}" if context else "请分析以下图片。"
        return self.think(VISION_AGENT_PROMPT, content, images=images)

# ==================== arXiv论文分析多智能体系统 ====================

class ArxivAnalysisSystem(MultiAgentSystem):
    """arXiv论文分析多智能体系统"""

    def __init__(self, base_url: str, api_key: str, model: str):
        super().__init__(base_url, api_key, model)
        self._init_agents()

    def _init_agents(self):
        self.brain = self.create_agent("大脑Agent", "负责任务规划、协调各个专家Agent、汇总结果并进行质量控制的总指挥")
        self.judger = self.create_agent("审稿人Agent", "一位严格的学术审稿人，负责批判性分析论文的优势、劣势和学术规范性")
        self.experiment = self.create_agent("实验分析Agent", "专注于分析实验设计、数据集、实验结果和资源消耗的实验专家")
        self.method = self.create_agent("方法理解Agent", "专注于理解和解释论文核心方法、技术原理的方法论专家")
        # 视觉分析Agent（仅多模态模型启用）
        self.vision = VisionAnalysisAgent(self.client, self.model) if self.supports_vision else None

    def analyze(self, paper_content: str, progress_callback=None, images: list = None) -> str:
        results = {}

        # 阶段1: 大脑Agent规划任务
        if progress_callback: progress_callback("大脑Agent正在规划分析任务...")
        plan = self.brain.think(BRAIN_PLAN_PROMPT, paper_content)

        # 阶段2: 各专家Agent并行分析
        if progress_callback: progress_callback("方法理解Agent正在分析核心方法...")
        results['method'] = self.method.think(METHOD_AGENT_PROMPT, paper_content)

        if progress_callback: progress_callback("实验分析Agent正在分析实验设计...")
        results['experiment'] = self.experiment.think(EXPERIMENT_AGENT_PROMPT, paper_content)

        if progress_callback: progress_callback("审稿人Agent正在进行批判性评审...")
        results['review'] = self.judger.think(JUDGER_AGENT_PROMPT, paper_content)

        # 视觉分析（如果有图片且支持多模态）
        if images and self.vision:
            if progress_callback: progress_callback("视觉分析Agent正在分析论文图片...")
            results['vision'] = self.vision.analyze_images(images, paper_content[:2000])

        # 阶段3: 大脑Agent汇总并质量控制
        if progress_callback: progress_callback("大脑Agent正在汇总分析结果...")
        summary_content = f"""
原始论文信息:
{paper_content}

各专家分析结果:

【方法理解Agent分析】
{results['method']}

【实验分析Agent分析】
{results['experiment']}

【审稿人Agent评审】
{results['review']}
"""
        if 'vision' in results:
            summary_content += f"\n【视觉分析Agent】\n{results['vision']}\n"

        final_result = self.brain.think(BRAIN_SUMMARY_PROMPT, summary_content)

        # 阶段4: 反思与改进（如果需要）
        if progress_callback: progress_callback("大脑Agent正在进行质量检查...")
        quality_check = self.brain.think(BRAIN_REFLECT_PROMPT, f"最终报告:\n{final_result}\n\n原始论文:\n{paper_content}")

        if "需要改进" in quality_check or "返工" in quality_check:
            if progress_callback: progress_callback("发现问题，正在改进...")
            final_result = self.brain.think(BRAIN_IMPROVE_PROMPT, f"质量检查反馈:\n{quality_check}\n\n原报告:\n{final_result}\n\n原始论文:\n{paper_content}")

        return final_result

# ==================== GitHub项目分析多智能体系统 ====================

class GithubAnalysisSystem(MultiAgentSystem):
    """GitHub项目分析多智能体系统"""

    def __init__(self, base_url: str, api_key: str, model: str):
        super().__init__(base_url, api_key, model)
        self._init_agents()

    def _init_agents(self):
        self.brain = self.create_agent("大脑Agent", "负责任务规划、协调各个专家Agent、汇总结果并进行质量控制的总指挥")
        self.architect = self.create_agent("架构分析Agent", "专注于分析项目架构、技术栈、模块设计的架构师")
        self.code_analyst = self.create_agent("代码分析Agent", "专注于分析核心代码实现、算法逻辑、代码质量的代码专家")
        self.usage = self.create_agent("使用分析Agent", "专注于分析项目使用方法、API接口、部署方式的应用专家")
        # 视觉分析Agent（仅多模态模型启用）
        self.vision = VisionAnalysisAgent(self.client, self.model) if self.supports_vision else None

    def analyze(self, project_content: str, progress_callback=None, images: list = None) -> str:
        results = {}

        # 阶段1: 大脑Agent规划
        if progress_callback: progress_callback("大脑Agent正在规划分析任务...")
        plan = self.brain.think(GITHUB_BRAIN_PLAN_PROMPT, project_content)

        # 阶段2: 各专家Agent分析
        if progress_callback: progress_callback("架构分析Agent正在分析项目架构...")
        results['architecture'] = self.architect.think(ARCHITECT_AGENT_PROMPT, project_content)

        if progress_callback: progress_callback("代码分析Agent正在分析核心代码...")
        results['code'] = self.code_analyst.think(CODE_ANALYST_PROMPT, project_content)

        if progress_callback: progress_callback("使用分析Agent正在分析使用方法...")
        results['usage'] = self.usage.think(USAGE_AGENT_PROMPT, project_content)

        # 视觉分析（如果有图片且支持多模态）
        if images and self.vision:
            if progress_callback: progress_callback("视觉分析Agent正在分析图片...")
            results['vision'] = self.vision.analyze_images(images, project_content[:2000])

        # 阶段3: 大脑Agent汇总
        if progress_callback: progress_callback("大脑Agent正在汇总分析结果...")
        summary_content = f"""
原始项目信息:
{project_content}

各专家分析结果:

【架构分析Agent】
{results['architecture']}

【代码分析Agent】
{results['code']}

【使用分析Agent】
{results['usage']}
"""
        if 'vision' in results:
            summary_content += f"\n【视觉分析Agent】\n{results['vision']}\n"

        final_result = self.brain.think(GITHUB_BRAIN_SUMMARY_PROMPT, summary_content)

        # 阶段4: 质量检查
        if progress_callback: progress_callback("大脑Agent正在进行质量检查...")
        quality_check = self.brain.think(BRAIN_REFLECT_PROMPT, f"最终报告:\n{final_result}")

        if "需要改进" in quality_check or "返工" in quality_check:
            if progress_callback: progress_callback("发现问题，正在改进...")
            final_result = self.brain.think(BRAIN_IMPROVE_PROMPT, f"质量检查反馈:\n{quality_check}\n\n原报告:\n{final_result}")

        return final_result

# ==================== arXiv分析Prompts ====================

BRAIN_PLAN_PROMPT = """作为大脑Agent，你需要规划论文分析任务。请分析这篇论文，确定需要重点关注的方面，为其他专家Agent提供指导。

请输出:
1. 论文的核心主题是什么
2. 需要方法理解Agent重点关注哪些技术点
3. 需要实验分析Agent重点关注哪些实验
4. 需要审稿人Agent重点审查哪些方面"""

METHOD_AGENT_PROMPT = """作为方法理解Agent，你需要深入理解论文的核心方法。请分析:

## 1. 核心Motivation
- 这项工作要解决什么根本问题？
- 现有方法的核心缺陷是什么？
- 作者的关键洞察(insight)是什么？

## 2. 核心方法
- 方法的核心思想用一句话概括
- 方法的关键技术创新点（不是边角改进，而是最核心的创新）
- 方法的数学原理或算法流程

## 3. 方法框架
- 用文字描述方法的整体框架（可以用ASCII图或结构化描述）
- 各个模块的作用和相互关系

## 4. 与现有方法的本质区别
- 与最相关的baseline方法相比，本质区别是什么？

请确保分析深入到方法的本质，而非表面描述。"""

EXPERIMENT_AGENT_PROMPT = """作为实验分析Agent，你需要全面分析论文的实验部分。请分析:

## 1. 实验任务
- 论文在哪些任务上进行了验证？
- 每个任务的定义和评价指标是什么？

## 2. 数据集
- 使用了哪些数据集？
- 数据集的规模和特点是什么？

## 3. 实验设置
- 主要的baseline方法有哪些？
- 实验的超参数设置如何？

## 4. 实验结果
- 主实验的结果如何？提升了多少？
- 消融实验验证了哪些设计的有效性？
- 有哪些有趣的分析实验？

## 5. 资源消耗
- 训练需要多少计算资源？
- 推理速度如何？
- 模型参数量多大？"""

JUDGER_AGENT_PROMPT = """作为审稿人Agent，你需要以严格的学术标准批判性地评审这篇论文。请从以下角度分析:

## 1. 论文优势
- 创新性：方法的创新程度如何？
- 有效性：实验是否充分证明了方法的有效性？
- 清晰度：论文写作是否清晰易懂？

## 2. 论文劣势
- 方法的局限性是什么？
- 实验设计有哪些不足？
- 论文有哪些未解决的问题？

## 3. 学术规范性
- 代码是否开源？
- 实验是否可复现？
- 与相关工作的比较是否公平？
- 结果是否合理（有无过度claim）？

## 4. 改进建议
- 如果你是审稿人，会提出哪些修改意见？

## 5. 总体评价
- 给出Accept/Weak Accept/Weak Reject/Reject的建议及理由"""

BRAIN_SUMMARY_PROMPT = """作为大脑Agent，请汇总各专家的分析结果，生成一份结构化的论文分析报告。

请按以下格式输出最终报告:

# 论文深度分析报告

## 一、核心贡献与创新
（基于方法理解Agent的分析，提炼最核心的1-3个贡献）

## 二、研究动机与洞察
（这项工作的出发点和关键insight）

## 三、方法详解
（核心方法的原理和框架）

## 四、实验验证
（关键实验结果和结论）

## 五、批判性评价
（优势、劣势、学术规范性）

## 六、研究启发
（对后续研究的启发和可能的改进方向）

请确保报告深入、准确、有洞察力，突出最核心的内容。"""

BRAIN_REFLECT_PROMPT = """作为大脑Agent，请检查最终报告的质量:

1. 是否准确反映了论文的核心贡献？
2. 是否深入分析了方法的本质，而非表面描述？
3. 是否有遗漏的重要信息？
4. 各部分是否逻辑连贯？

如果发现问题，请指出"需要改进"并说明具体问题。
如果质量合格，请回复"质量合格"。"""

BRAIN_IMPROVE_PROMPT = """作为大脑Agent，请根据质量检查的反馈改进报告。

请输出改进后的完整报告。"""

# ==================== GitHub分析Prompts ====================

GITHUB_BRAIN_PLAN_PROMPT = """作为大脑Agent，你需要规划GitHub项目分析任务。请分析这个项目，确定需要重点关注的方面。

请输出:
1. 项目的核心功能是什么
2. 需要架构分析Agent重点关注哪些模块
3. 需要代码分析Agent重点分析哪些文件
4. 需要使用分析Agent重点说明哪些使用方法"""

ARCHITECT_AGENT_PROMPT = """作为架构分析Agent，你需要深入分析项目的架构设计。请分析:

## 1. 项目定位
- 这个项目解决什么问题？
- 目标用户是谁？
- 与同类项目相比有什么优势？

## 2. 技术架构
- 整体架构设计是怎样的？
- 核心模块有哪些？各自的职责是什么？
- 模块之间如何交互？

## 3. 技术栈
- 使用了哪些编程语言/框架/库？
- 为什么选择这些技术？
- 技术选型是否合理？

## 4. 设计模式
- 使用了哪些设计模式？
- 代码组织结构是否清晰？"""

CODE_ANALYST_PROMPT = """作为代码分析Agent，你需要深入分析项目的核心代码。请分析:

## 1. 核心算法/逻辑
- 项目的核心算法是什么？
- 关键函数/类的实现思路是什么？

## 2. 代码质量
- 代码风格是否规范？
- 是否有充分的注释和文档？
- 错误处理是否完善？

## 3. 关键实现
- 最重要的几个文件/函数是什么？
- 它们是如何实现核心功能的？

## 4. 可扩展性
- 代码是否易于扩展？
- 有哪些可以改进的地方？"""

USAGE_AGENT_PROMPT = """作为使用分析Agent，你需要分析项目的使用方法。请分析:

## 1. 安装配置
- 如何安装这个项目？
- 需要哪些依赖？
- 有哪些配置选项？

## 2. 基本使用
- 最基本的使用流程是什么？
- 有哪些常用命令/API？

## 3. 高级功能
- 有哪些高级功能或配置？
- 如何进行自定义扩展？

## 4. 注意事项
- 使用时有哪些需要注意的地方？
- 常见问题和解决方案是什么？"""

GITHUB_BRAIN_SUMMARY_PROMPT = """作为大脑Agent，请汇总各专家的分析结果，生成一份结构化的项目分析报告。

请按以下格式输出最终报告:

# GitHub项目深度分析报告

## 一、项目概述
（项目定位、解决的问题、目标用户）

## 二、核心创新与价值
（项目的核心创新点和独特价值）

## 三、技术架构
（整体架构、核心模块、技术栈）

## 四、代码分析
（核心实现、代码质量、关键算法）

## 五、使用指南
（安装、基本使用、高级功能）

## 六、研究价值与应用场景
（对研究的启发、适用场景、改进方向）

请确保报告深入、准确、实用。"""

# ==================== 相关研究分析多智能体系统 ====================

class RelatedWorkSystem(MultiAgentSystem):
    """相关研究分析多智能体系统 - 搜索arXiv并深度分析相关工作"""

    def __init__(self, base_url: str, api_key: str, model: str):
        super().__init__(base_url, api_key, model)
        self._init_agents()

    def _init_agents(self):
        self.brain = self.create_agent("大脑Agent", "负责规划搜索策略、协调分析、汇总比较结果的总指挥")
        self.tech_analyst = self.create_agent("技术框架分析Agent", "专注于分析和比较不同论文的技术框架、方法论差异")
        self.experiment_analyst = self.create_agent("实验分析Agent", "专注于分析和比较不同论文的实验设置、任务、效果差异")

    def analyze(self, paper_info: str, progress_callback=None) -> str:
        # 阶段1: 大脑Agent提取关键词并规划搜索
        if progress_callback: progress_callback("大脑Agent正在分析论文并提取搜索关键词...")
        keywords_result = self.brain.think(RELATED_KEYWORD_PROMPT, paper_info)

        # 阶段2: 搜索arXiv相关论文（3年内）
        if progress_callback: progress_callback("正在搜索arXiv相关论文（近3年）...")
        from api_client import search_arxiv
        from datetime import datetime, timedelta
        end_date = datetime.now()
        start_date = end_date - timedelta(days=1095)  # 3年

        # 从关键词结果中提取搜索词
        search_terms = self._extract_search_terms(keywords_result)
        related_papers = []
        for term in search_terms[:3]:  # 最多搜索3个关键词
            try:
                papers = search_arxiv(term, start_date, end_date, max_results=10)
                related_papers.extend(papers)
            except: pass

        # 去重
        seen = set()
        unique_papers = []
        for p in related_papers:
            if p['title'] not in seen:
                seen.add(p['title'])
                unique_papers.append(p)

        if not unique_papers:
            return "未找到相关论文，请尝试其他搜索关键词。"

        # 阶段3: LLM筛选最相关的论文
        if progress_callback: progress_callback(f"大脑Agent正在从{len(unique_papers)}篇论文中筛选最相关的...")
        papers_summary = "\n".join([f"[{i+1}] {p['title']}\n摘要: {p['abstract'][:300]}..." for i, p in enumerate(unique_papers[:20])])
        filter_content = f"当前论文:\n{paper_info}\n\n候选相关论文:\n{papers_summary}"
        filtered_result = self.brain.think(RELATED_FILTER_PROMPT, filter_content)

        # 阶段4: 技术框架分析Agent分析
        if progress_callback: progress_callback("技术框架分析Agent正在分析技术差异...")
        tech_analysis = self.tech_analyst.think(RELATED_TECH_PROMPT, f"当前论文:\n{paper_info}\n\n相关论文:\n{papers_summary[:8000]}")

        # 阶段5: 实验分析Agent分析
        if progress_callback: progress_callback("实验分析Agent正在分析实验差异...")
        exp_analysis = self.experiment_analyst.think(RELATED_EXP_PROMPT, f"当前论文:\n{paper_info}\n\n相关论文:\n{papers_summary[:8000]}")

        # 阶段6: 大脑Agent汇总
        if progress_callback: progress_callback("大脑Agent正在汇总分析结果...")
        summary_content = f"""
当前论文: {paper_info}

搜索关键词分析:
{keywords_result}

筛选结果:
{filtered_result}

技术框架对比分析:
{tech_analysis}

实验对比分析:
{exp_analysis}

找到的相关论文列表:
{papers_summary[:5000]}
"""
        final_result = self.brain.think(RELATED_SUMMARY_PROMPT, summary_content)
        return final_result

    def _extract_search_terms(self, keywords_result: str) -> list:
        """从关键词结果中提取搜索词"""
        import re
        lines = keywords_result.split('\n')
        terms = []
        for line in lines:
            # 匹配常见的关键词格式
            if any(kw in line.lower() for kw in ['keyword', '关键词', 'search', '搜索']):
                # 提取引号内或冒号后的内容
                matches = re.findall(r'["\']([^"\']+)["\']|[:：]\s*(.+)', line)
                for m in matches:
                    term = m[0] or m[1]
                    if term and len(term) > 2:
                        terms.append(term.strip())
        # 如果没找到，用整行作为搜索词
        if not terms:
            for line in lines:
                line = line.strip()
                if line and len(line) > 5 and len(line) < 100:
                    terms.append(line)
        return terms[:5]

# ==================== 相关研究Prompts ====================

RELATED_KEYWORD_PROMPT = """分析以下论文，提取用于搜索相关工作的关键词。

请输出:
1. 核心技术关键词（3-5个，用于arXiv搜索）
2. 任务/应用领域关键词（2-3个）
3. 方法类别关键词（2-3个）

格式示例:
- 关键词1: "transformer attention mechanism"
- 关键词2: "large language model"
"""

RELATED_FILTER_PROMPT = """作为大脑Agent，请从候选论文中筛选出与当前论文最相关的5-10篇。

筛选标准:
1. 技术方法相似度
2. 研究任务相关性
3. 时间相近性（优先近期工作）

请输出筛选结果，格式:
[编号] 论文标题 - 相关原因（一句话）"""

RELATED_TECH_PROMPT = """作为技术框架分析Agent，请对比分析当前论文与相关论文在技术框架上的异同。

请分析:
1. 核心技术方法的异同
2. 模型架构的差异
3. 创新点的对比
4. 技术演进关系（哪些是前序工作，哪些是并行工作）"""

RELATED_EXP_PROMPT = """作为实验分析Agent，请对比分析当前论文与相关论文在实验方面的异同。

请分析:
1. 实验任务的异同
2. 使用数据集的对比
3. 评价指标的差异
4. 实验效果的对比（如果有）"""

RELATED_SUMMARY_PROMPT = """作为大脑Agent，请汇总所有分析结果，生成一份完整的相关研究分析报告。

请按以下格式输出:

# 相关研究深度分析报告

## 一、搜索关键词
（用于搜索的关键词列表）

## 二、最相关论文列表
（筛选出的5-10篇最相关论文，包含标题、链接、相关原因）

## 三、技术框架对比
（当前论文与相关工作在技术上的联系与区别）

## 四、实验任务对比
（当前论文与相关工作在实验任务、数据集、效果上的对比）

## 五、研究脉络分析
（这些工作之间的演进关系，当前论文在该领域的定位）

## 六、推荐阅读顺序
（建议的论文阅读顺序和理由）

请确保分析深入、客观、有洞察力。"""

# ==================== 其他Prompts ====================

SIMILAR_WORK_PROMPT = """基于以下论文/项目的核心方法和任务，请帮我找出可能的相关研究方向和关键词，用于搜索类似工作：

信息：
{paper_info}

请输出：
1. 相关的研究方向（3-5个）
2. 推荐的搜索关键词（5-10个）
3. 可能的相关会议/期刊/开源社区"""

# 保留简单的LLMClient用于其他功能
class LLMClient:
    def __init__(self, base_url: str, api_key: str, model: str):
        self.client = OpenAI(base_url=base_url, api_key=api_key)
        self.model = model

    def analyze(self, prompt: str, content: str) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "system", "content": prompt}, {"role": "user", "content": content}],
            temperature=0.3
        )
        return response.choices[0].message.content

# ==================== 智能搜索多智能体系统 ====================

class SmartSearchSystem(MultiAgentSystem):
    """智能搜索系统 - 通过对话收集用户意图，迭代搜索筛选"""

    def __init__(self, base_url: str, api_key: str, model: str):
        super().__init__(base_url, api_key, model)
        self.interviewer = self.create_agent("访谈Agent", "负责通过友好的对话了解用户的研究需求和背景")
        self.brain = self.create_agent("大脑Agent", "负责分析用户意图、构建搜索策略、筛选结果")
        self.user_profile = {}
        self.chat_history = []

    def get_next_question(self, user_input: str = "") -> dict:
        """获取下一个问题或生成搜索策略"""
        if user_input:
            self.chat_history.append({"role": "user", "content": user_input})

        history_text = "\n".join([f"{'用户' if m['role']=='user' else 'AI'}: {m['content']}" for m in self.chat_history])

        response = self.interviewer.think(INTERVIEW_PROMPT, f"对话历史:\n{history_text}\n\n已收集信息:\n{json.dumps(self.user_profile, ensure_ascii=False)}")
        self.chat_history.append({"role": "assistant", "content": response})

        # 检查是否收集完成
        if "【搜索就绪】" in response or "【READY】" in response:
            return {"type": "ready", "message": response, "profile": self.user_profile}

        # 提取收集到的信息
        if "【更新】" in response:
            try:
                update_start = response.find("【更新】") + 4
                update_end = response.find("【/更新】") if "【/更新】" in response else len(response)
                update_text = response[update_start:update_end].strip()
                for line in update_text.split("\n"):
                    if ":" in line or "：" in line:
                        key, val = line.replace("：", ":").split(":", 1)
                        self.user_profile[key.strip()] = val.strip()
            except: pass

        return {"type": "question", "message": response}

    def build_search_strategy(self) -> dict:
        """根据用户画像构建搜索策略"""
        profile_text = json.dumps(self.user_profile, ensure_ascii=False, indent=2)
        strategy = self.brain.think(SEARCH_STRATEGY_PROMPT, f"用户画像:\n{profile_text}")

        # 解析搜索策略
        result = {"keywords": [], "time_range": "past_year", "sources": ["arxiv", "github"], "target_count": 20}
        for line in strategy.split("\n"):
            line = line.strip()
            if "关键词" in line and ":" in line:
                keywords = line.split(":", 1)[1].strip()
                result["keywords"] = [k.strip().strip('"\'') for k in keywords.split(",")]
            elif "时间" in line and ":" in line:
                result["time_range"] = line.split(":", 1)[1].strip()
            elif "目标数量" in line and ":" in line:
                try: result["target_count"] = int(''.join(filter(str.isdigit, line.split(":", 1)[1])))
                except: pass
        return result

    def filter_results(self, results: list, user_intent: str) -> tuple[list, list]:
        """筛选搜索结果，返回(匹配的, 不匹配的)"""
        if not results:
            return [], []

        results_text = "\n".join([f"[{i+1}] {r['title']}\n摘要: {r.get('abstract', r.get('description', ''))[:200]}..."
                                   for i, r in enumerate(results[:30])])

        filter_result = self.brain.think(FILTER_RESULTS_PROMPT,
            f"用户意图:\n{user_intent}\n\n搜索结果:\n{results_text}")

        matched_indices = set()
        for line in filter_result.split("\n"):
            if "匹配" in line or "相关" in line or "推荐" in line:
                import re
                nums = re.findall(r'\[(\d+)\]', line)
                for n in nums:
                    try: matched_indices.add(int(n) - 1)
                    except: pass

        matched = [results[i] for i in range(len(results)) if i in matched_indices]
        unmatched = [results[i] for i in range(len(results)) if i not in matched_indices]
        return matched, unmatched

INTERVIEW_PROMPT = """你是一位专业的学术研究顾问，正在通过深入对话帮助用户明确他们的研究需求。

## 核心目标
你最重要的任务是**完全理解用户想要做什么项目**。很多用户可能表达不清楚，你需要：
1. 耐心引导用户一步步说出想法
2. 用自己的理解复述用户的需求，确认是否正确
3. 如果有不确定的地方，主动追问澄清
4. 帮助用户将模糊的想法具体化

## 必须了解的核心信息（按优先级）

### 第一优先级 - 项目本质（必须完全搞清楚）
1. **具体要做什么项目？**
   - 不是泛泛的"研究方向"，而是具体的项目目标
   - 例如：不是"NLP"，而是"做一个能自动总结论文的工具"
   - 追问示例："你说想做大模型相关的，能具体说说想实现什么功能吗？"

2. **应用场景是什么？**
   - 这个项目用在什么地方？解决什么实际问题？
   - 谁会使用？在什么情况下使用？
   - 追问示例："这个功能主要是给谁用的？在什么场景下会用到？"

3. **输入输出是什么？**
   - 系统接收什么输入？产出什么输出？
   - 追问示例："用户给系统什么数据，系统返回什么结果？"

### 第二优先级 - 技术约束
4. **有什么技术偏好或限制？**
   - 偏好的框架、语言、方法
   - 必须使用或避免的技术

5. **资源情况**（可选）
   - GPU、存储、API预算等

### 第三优先级 - 搜索范围
6. **关注什么时间段的研究？**
   - 最新的还是经典的？

## 对话策略

1. **先听后问**：让用户先说，然后针对性追问
2. **复述确认**：用自己的话复述理解，让用户确认
3. **具体化引导**：如果用户说得太抽象，给出具体例子引导
4. **不要假设**：不确定就问，不要自己脑补

## 对话示例

用户："我想找一些大模型相关的论文"
你："好的！大模型是个很大的领域，我想更具体地了解一下你的需求。你是想：
- 用大模型做某个具体应用？（比如对话系统、代码生成、文本分析等）
- 研究大模型本身的某个技术？（比如训练方法、推理优化、安全性等）
- 还是其他方向？

能告诉我你具体想做什么项目吗？"

## 输出格式

- 每次只问1-2个问题，保持对话自然
- 当你对用户需求有了清晰理解后，输出"【搜索就绪】"并总结：
  - 项目目标：xxx
  - 应用场景：xxx
  - 技术需求：xxx
- 每次收集到新信息时，用"【更新】key: value【/更新】"格式记录

记住：宁可多问几个问题把需求搞清楚，也不要在理解模糊的情况下就开始搜索。"""

SEARCH_STRATEGY_PROMPT = """根据用户画像，制定精准的搜索策略。

重点关注用户的：
1. 具体项目目标 - 用户想做什么
2. 应用场景 - 用在哪里，解决什么问题
3. 技术需求 - 需要什么技术/方法

请输出：
1. 搜索关键词: keyword1, keyword2, keyword3 (3-5个，用逗号分隔)
   - 关键词应该直接对应用户的项目需求
   - 包含任务类型、方法类型、应用领域
2. 时间范围: past_week/past_month/past_3months/past_year
3. 搜索来源: arxiv, github (或只选一个)
4. 目标数量: 数字

示例：
如果用户想做"论文自动总结工具"，关键词应该是：
搜索关键词: "paper summarization", "document summarization LLM", "scientific text summarization"
而不是泛泛的 "NLP", "large language model"

请确保关键词足够具体，能找到与用户项目直接相关的论文/项目。"""

FILTER_RESULTS_PROMPT = """根据用户的具体项目需求，筛选搜索结果。

用户需求分析：
- 项目目标：用户具体想做什么
- 应用场景：用在哪里
- 技术需求：需要什么方法/技术

筛选标准（按优先级）：
1. **直接相关** - 论文/项目的任务与用户项目目标高度一致
2. **方法可用** - 提出的方法可以直接用于用户的应用场景
3. **技术匹配** - 使用的技术栈与用户需求兼容

请分析每个结果，输出：
1. 匹配的结果编号（格式：[1], [3], [5]）
2. 每个匹配结果为什么对用户的项目有帮助

注意：
- 只选择真正能帮助用户完成项目的结果
- 泛泛相关的不要选，要选就选直接相关的
- 宁缺毋滥，质量优先"""

import sys
import os
import webbrowser
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QLineEdit, QComboBox, QPushButton, QTextEdit, QListWidget, QListWidgetItem,
    QGroupBox, QDateEdit, QCheckBox, QTabWidget, QProgressBar, QMessageBox, QSplitter,
    QFileDialog, QSpinBox, QAbstractItemView)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QDate
from PyQt6.QtGui import QFont, QColor, QPalette
from PyQt6.QtWebEngineWidgets import QWebEngineView
import requests
import markdown

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from api_client import search_arxiv, search_github, get_date_range, get_repo_content, search_trending, extract_images_from_pdf, search_huggingface, search_modelscope, get_huggingface_content, get_modelscope_content
from llm_client import LLMClient, ArxivAnalysisSystem, GithubAnalysisSystem, RelatedWorkSystem, SmartSearchSystem, SIMILAR_WORK_PROMPT, is_multimodal_model

def md_to_html(text):
    """将 Markdown 转换为带样式的 HTML，支持数学公式"""
    html = markdown.markdown(text, extensions=['tables', 'fenced_code'])
    return f'''<!DOCTYPE html><html><head>
    <script src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script>
    <style>
        body {{ background: #282a36; color: #f8f8f2; font-family: 'Segoe UI', 'Microsoft YaHei'; line-height: 1.6; padding: 15px; margin: 0; }}
        h1, h2, h3 {{ color: #bd93f9; }}
        code {{ background: #44475a; padding: 2px 6px; border-radius: 4px; }}
        pre {{ background: #21222c; padding: 12px; border-radius: 8px; overflow-x: auto; }}
        a {{ color: #8be9fd; }}
        table {{ border-collapse: collapse; width: 100%; }}
        th, td {{ border: 1px solid #44475a; padding: 8px; text-align: left; }}
        th {{ background: #44475a; }}
        hr {{ border: 1px solid #44475a; }}
    </style></head><body>{html}</body></html>'''

class SearchWorker(QThread):
    finished = pyqtSignal(list)
    error = pyqtSignal(str)

    def __init__(self, query, start_date, end_date, search_arxiv_flag, search_github_flag, github_token, max_results,
                 search_huggingface_flag=False, search_modelscope_flag=False):
        super().__init__()
        self.query, self.start_date, self.end_date = query, start_date, end_date
        self.search_arxiv_flag, self.search_github_flag = search_arxiv_flag, search_github_flag
        self.search_huggingface_flag, self.search_modelscope_flag = search_huggingface_flag, search_modelscope_flag
        self.github_token, self.max_results = github_token, max_results
        self._cancelled = False

    def cancel(self):
        self._cancelled = True

    def run(self):
        try:
            results = []
            if self._cancelled: return self.finished.emit(results)
            if self.search_arxiv_flag:
                results.extend(search_arxiv(self.query, self.start_date, self.end_date, self.max_results))
            if self._cancelled: return self.finished.emit(results)
            if self.search_github_flag:
                results.extend(search_github(self.query, self.start_date, self.end_date, self.github_token, self.max_results))
            if self._cancelled: return self.finished.emit(results)
            if self.search_huggingface_flag:
                results.extend(search_huggingface(self.query, self.start_date, self.end_date, self.max_results))
            if self._cancelled: return self.finished.emit(results)
            if self.search_modelscope_flag:
                results.extend(search_modelscope(self.query, self.start_date, self.end_date, self.max_results))
            self.finished.emit(results)
        except Exception as e:
            if not self._cancelled:
                self.error.emit(str(e))

class BatchAnalyzeWorker(QThread):
    finished = pyqtSignal(dict)
    progress_update = pyqtSignal(str, int, int)
    error = pyqtSignal(str)

    def __init__(self, base_url, api_key, model, papers, github_token=None, fetch_images=False):
        super().__init__()
        self.base_url, self.api_key, self.model = base_url, api_key, model
        self.papers, self.github_token = papers, github_token
        self.fetch_images = fetch_images and is_multimodal_model(model)

    def analyze_single(self, paper, idx, total):
        self.progress_update.emit(f"正在分析 ({idx+1}/{total}): {paper['title'][:40]}...", idx+1, total)
        images = []
        source = paper['source']

        if source == 'arxiv':
            system = ArxivAnalysisSystem(self.base_url, self.api_key, self.model)
            content = f"标题: {paper['title']}\n摘要: {paper['abstract']}\n作者: {', '.join(paper['authors'])}"
            if self.fetch_images:
                images = extract_images_from_pdf(paper['pdf_url'], max_images=3)
        elif source == 'github':
            system = GithubAnalysisSystem(self.base_url, self.api_key, self.model)
            repo_content = get_repo_content(paper['title'], self.github_token, fetch_images=self.fetch_images)
            content = f"# 项目: {paper['title']}\n## 描述\n{paper['description']}\n## README\n{repo_content['readme'][:15000]}\n"
            content += f"## 项目结构\n" + "\n".join(repo_content['structure'][:50]) + "\n## 关键代码文件\n"
            for f in repo_content['key_files'][:5]:
                content += f"\n### {f['name']}\n```\n{f['content'][:3000]}\n```\n"
            images = repo_content.get('images', [])
        elif source == 'huggingface':
            system = GithubAnalysisSystem(self.base_url, self.api_key, self.model)
            hf_content = get_huggingface_content(paper['title'])
            content = f"# HuggingFace模型: {paper['title']}\n{hf_content['model_info']}\n## README\n{hf_content['readme']}\n"
            content += f"## 文件列表\n" + "\n".join(hf_content['files'][:30])
        elif source == 'modelscope':
            system = GithubAnalysisSystem(self.base_url, self.api_key, self.model)
            # 从URL中提取模型路径
            model_path = paper['url'].replace('https://modelscope.cn/models/', '')
            ms_content = get_modelscope_content(model_path)
            content = f"# ModelScope模型: {paper['title']}\n{ms_content['model_info']}\n## README\n{ms_content['readme']}\n"
            content += f"## 文件列表\n" + "\n".join(ms_content['files'][:30])
        else:
            # 默认使用 GitHub 分析系统
            system = GithubAnalysisSystem(self.base_url, self.api_key, self.model)
            content = f"# 项目: {paper['title']}\n## 描述\n{paper.get('description', '')}\n"

        return paper['title'], system.analyze(content, images=images)

    def run(self):
        try:
            results = {}
            total = len(self.papers)
            for idx, paper in enumerate(self.papers):
                try:
                    title, result = self.analyze_single(paper, idx, total)
                    results[title] = result
                except Exception as e:
                    results[paper['title']] = f"❌ 分析失败: {str(e)}"
            self.finished.emit(results)
        except Exception as e:
            import traceback
            self.error.emit(f"{str(e)}\n\n详细信息:\n{traceback.format_exc()}")

class RelatedWorkWorker(QThread):
    finished = pyqtSignal(str)
    progress_update = pyqtSignal(str)
    error = pyqtSignal(str)

    def __init__(self, base_url, api_key, model, paper):
        super().__init__()
        self.base_url, self.api_key, self.model, self.paper = base_url, api_key, model, paper

    def run(self):
        try:
            system = RelatedWorkSystem(self.base_url, self.api_key, self.model)
            paper_info = f"标题: {self.paper['title']}\n摘要: {self.paper.get('abstract', self.paper.get('description', ''))}"
            result = system.analyze(paper_info, lambda msg: self.progress_update.emit(msg))
            self.finished.emit(result)
        except Exception as e:
            self.error.emit(str(e))

class ExploreWorker(QThread):
    finished = pyqtSignal(list)
    error = pyqtSignal(str)

    def __init__(self, query, github_token, max_results=30, search_new=True,
                 search_arxiv=True, search_github=True, search_hf=True, search_ms=True):
        super().__init__()
        self.query, self.github_token = query, github_token
        self.max_results, self.search_new = max_results, search_new
        self.search_arxiv_flag = search_arxiv
        self.search_github_flag = search_github
        self.search_hf_flag = search_hf
        self.search_ms_flag = search_ms

    def run(self):
        try:
            results = []
            per_source = max(5, self.max_results // 4)
            from datetime import datetime, timedelta
            end_date = datetime.now()
            start_date = end_date - timedelta(days=3)

            if self.search_github_flag:
                try:
                    github_results = search_trending(self.query, self.github_token,
                                         max_results=per_source, search_new=self.search_new)
                    results.extend(github_results)
                except: pass
            if self.search_arxiv_flag:
                try:
                    arxiv_results = search_arxiv(self.query, start_date, end_date, per_source)
                    results.extend(arxiv_results)
                except: pass
            if self.search_hf_flag:
                try:
                    hf_results = search_huggingface(self.query, start_date, end_date, per_source)
                    results.extend(hf_results)
                except: pass
            if self.search_ms_flag:
                try:
                    ms_results = search_modelscope(self.query, start_date, end_date, per_source)
                    results.extend(ms_results)
                except: pass

            self.finished.emit(results)
        except Exception as e:
            self.error.emit(str(e))

class SmartSearchWorker(QThread):
    question_ready = pyqtSignal(dict)
    search_progress = pyqtSignal(str)
    results_ready = pyqtSignal(list)
    error = pyqtSignal(str)

    def __init__(self, base_url, api_key, model, action, user_input="", github_token=None,
                 sources=None, max_questions=3):
        super().__init__()
        self.base_url, self.api_key, self.model = base_url, api_key, model
        self.action, self.user_input, self.github_token = action, user_input, github_token
        self.sources = sources or ['arxiv', 'github', 'huggingface', 'modelscope']
        self.max_questions = max_questions
        self.system = None

    def run(self):
        try:
            if not hasattr(SmartSearchWorker, '_system') or SmartSearchWorker._system is None:
                SmartSearchWorker._system = SmartSearchSystem(self.base_url, self.api_key, self.model)
                SmartSearchWorker._question_count = 0
            self.system = SmartSearchWorker._system

            if self.action == "ask":
                SmartSearchWorker._question_count += 1
                result = self.system.get_next_question(self.user_input)
                # 强制在3个问题后进入搜索就绪状态
                if SmartSearchWorker._question_count >= self.max_questions and result['type'] != 'ready':
                    result['type'] = 'ready'
                    result['message'] += "\n\n【搜索就绪】已收集足够信息，可以开始搜索了。"
                self.question_ready.emit(result)
            elif self.action == "search":
                self.search_progress.emit("正在构建搜索策略...")
                strategy = self.system.build_search_strategy()

                self.search_progress.emit(f"搜索关键词: {', '.join(strategy['keywords'])}")
                all_results = []
                time_map = {"past_week": 7, "past_month": 30, "past_3months": 90, "past_year": 365}
                days = time_map.get(strategy.get('time_range', 'past_year'), 365)
                from datetime import datetime, timedelta
                end_date, start_date = datetime.now(), datetime.now() - timedelta(days=days)

                for kw in strategy['keywords'][:3]:
                    self.search_progress.emit(f"搜索: {kw}...")
                    if 'arxiv' in self.sources:
                        try:
                            results = search_arxiv(kw, start_date, end_date, 20)
                            all_results.extend(results)
                        except: pass
                    if 'github' in self.sources:
                        try:
                            results = search_github(kw, start_date, end_date, self.github_token, 10)
                            all_results.extend(results)
                        except: pass
                    if 'huggingface' in self.sources:
                        try:
                            results = search_huggingface(kw, start_date, end_date, 10)
                            all_results.extend(results)
                        except: pass
                    if 'modelscope' in self.sources:
                        try:
                            results = search_modelscope(kw, start_date, end_date, 10)
                            all_results.extend(results)
                        except: pass

                # 去重
                seen, unique = set(), []
                for r in all_results:
                    if r['title'] not in seen:
                        seen.add(r['title'])
                        unique.append(r)

                self.search_progress.emit(f"筛选 {len(unique)} 条结果...")
                user_intent = "\n".join([f"{k}: {v}" for k, v in self.system.user_profile.items()])
                matched, _ = self.system.filter_results(unique, user_intent)

                self.results_ready.emit(matched if matched else unique[:strategy['target_count']])
            elif self.action == "reset":
                SmartSearchWorker._system = SmartSearchSystem(self.base_url, self.api_key, self.model)
        except Exception as e:
            self.error.emit(str(e))

class SettingsDialog(QWidget):
    """设置对话框"""
    def __init__(self, parent=None):
        super().__init__(parent, Qt.WindowType.Window)
        self.setWindowTitle("设置")
        self.setFixedSize(450, 280)
        self.parent_window = parent
        self.setup_ui()
        self.apply_style()

    def setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(15)
        layout.setContentsMargins(20, 20, 20, 20)

        # API设置
        self.base_url = QLineEdit("https://api.openai.com/v1")
        self.api_key = QLineEdit()
        self.api_key.setEchoMode(QLineEdit.EchoMode.Password)
        self.model_name = QLineEdit("gpt-4")
        self.github_token = QLineEdit()
        self.github_token.setEchoMode(QLineEdit.EchoMode.Password)

        for label, widget in [("Base URL:", self.base_url), ("API Key:", self.api_key),
                              ("Model:", self.model_name), ("GitHub Token:", self.github_token)]:
            row = QHBoxLayout()
            lbl = QLabel(label)
            lbl.setFixedWidth(100)
            row.addWidget(lbl)
            row.addWidget(widget)
            layout.addLayout(row)

        layout.addStretch()
        save_btn = QPushButton("保存")
        save_btn.clicked.connect(self.close)
        layout.addWidget(save_btn)

    def apply_style(self):
        self.setStyleSheet("""
            QWidget { background: #282a36; color: #f8f8f2; font-family: 'Segoe UI', 'Microsoft YaHei'; }
            QLineEdit { padding: 8px; border: 2px solid #44475a; border-radius: 6px; background: #21222c; }
            QLineEdit:focus { border-color: #bd93f9; }
            QPushButton { background: #bd93f9; color: #282a36; border: none; padding: 10px 20px; border-radius: 6px; font-weight: bold; }
            QPushButton:hover { background: #ff79c6; }
            QLabel { color: #f8f8f2; }
        """)

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("AI学术助手 - Multi-Agent System")
        self.setMinimumSize(1400, 900)
        self.results, self.analysis_results = [], {}
        self.search_mode = "smart"  # "smart" or "normal"
        self.settings_dialog = SettingsDialog(self)
        self.setup_ui()
        self.apply_style()

    @property
    def base_url(self): return self.settings_dialog.base_url
    @property
    def api_key(self): return self.settings_dialog.api_key
    @property
    def model_name(self): return self.settings_dialog.model_name
    @property
    def github_token(self): return self.settings_dialog.github_token

    def setup_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        layout = QHBoxLayout(central)
        layout.setSpacing(15)
        layout.setContentsMargins(15, 15, 15, 15)

        # 左侧面板
        left = QWidget()
        left_layout = QVBoxLayout(left)
        left_layout.setSpacing(12)

        # 顶部工具栏：设置按钮 + 闲逛热门
        toolbar = QHBoxLayout()
        self.settings_btn = QPushButton("⚙️ 设置")
        self.settings_btn.clicked.connect(lambda: self.settings_dialog.show())

        # 闲逛热门区域
        self.explore_type_combo = QComboBox()
        self.explore_type_combo.addItems(["新项目", "活跃项目"])
        self.explore_type_combo.setToolTip("新项目: 最近3天创建\n活跃项目: 最近3天有更新")
        self.explore_type_combo.setFixedWidth(90)

        self.explore_count_spin = QSpinBox()
        self.explore_count_spin.setRange(10, 100)
        self.explore_count_spin.setValue(30)
        self.explore_count_spin.setToolTip("搜索数量")
        self.explore_count_spin.setMinimumWidth(80)

        self.explore_btn = QPushButton("🎲 闲逛热门")
        self.explore_btn.clicked.connect(self.do_explore)

        toolbar.addWidget(self.settings_btn)
        toolbar.addStretch()
        toolbar.addWidget(QLabel("类型:"))
        toolbar.addWidget(self.explore_type_combo)
        toolbar.addWidget(QLabel("数量:"))
        toolbar.addWidget(self.explore_count_spin)
        toolbar.addWidget(self.explore_btn)
        left_layout.addLayout(toolbar)

        # 闲逛数据源选择
        explore_source_layout = QHBoxLayout()
        explore_source_layout.addWidget(QLabel("闲逛来源:"))
        self.explore_arxiv_check = QCheckBox("arXiv")
        self.explore_arxiv_check.setChecked(True)
        self.explore_github_check = QCheckBox("GitHub")
        self.explore_github_check.setChecked(True)
        self.explore_hf_check = QCheckBox("HuggingFace")
        self.explore_hf_check.setChecked(True)
        self.explore_ms_check = QCheckBox("ModelScope")
        self.explore_ms_check.setChecked(True)
        explore_source_layout.addWidget(self.explore_arxiv_check)
        explore_source_layout.addWidget(self.explore_github_check)
        explore_source_layout.addWidget(self.explore_hf_check)
        explore_source_layout.addWidget(self.explore_ms_check)
        explore_source_layout.addStretch()
        left_layout.addLayout(explore_source_layout)

        # 搜索模式切换
        mode_layout = QHBoxLayout()
        self.smart_mode_btn = QPushButton("🧠 智能搜索")
        self.smart_mode_btn.setCheckable(True)
        self.smart_mode_btn.setChecked(True)
        self.smart_mode_btn.clicked.connect(lambda: self.switch_search_mode("smart"))
        self.normal_mode_btn = QPushButton("🔍 普通搜索")
        self.normal_mode_btn.setCheckable(True)
        self.normal_mode_btn.clicked.connect(lambda: self.switch_search_mode("normal"))
        mode_layout.addWidget(self.smart_mode_btn)
        mode_layout.addWidget(self.normal_mode_btn)
        mode_layout.addStretch()
        left_layout.addLayout(mode_layout)

        # 搜索区域容器（用于切换智能/普通搜索）
        self.search_stack = QWidget()
        search_stack_layout = QVBoxLayout(self.search_stack)
        search_stack_layout.setContentsMargins(0, 0, 0, 0)

        # 智能搜索对话区域
        self.smart_search_widget = QWidget()
        smart_layout = QVBoxLayout(self.smart_search_widget)
        smart_layout.setContentsMargins(0, 0, 0, 0)

        # 智能搜索数据源选择
        smart_source_layout = QHBoxLayout()
        smart_source_layout.addWidget(QLabel("搜索来源:"))
        self.smart_arxiv_check = QCheckBox("arXiv")
        self.smart_arxiv_check.setChecked(True)
        self.smart_github_check = QCheckBox("GitHub")
        self.smart_github_check.setChecked(True)
        self.smart_hf_check = QCheckBox("HuggingFace")
        self.smart_hf_check.setChecked(True)
        self.smart_ms_check = QCheckBox("ModelScope")
        self.smart_ms_check.setChecked(True)
        smart_source_layout.addWidget(self.smart_arxiv_check)
        smart_source_layout.addWidget(self.smart_github_check)
        smart_source_layout.addWidget(self.smart_hf_check)
        smart_source_layout.addWidget(self.smart_ms_check)
        smart_source_layout.addStretch()
        smart_layout.addLayout(smart_source_layout)

        self.chat_display = QTextEdit()
        self.chat_display.setReadOnly(True)
        self.chat_display.setPlaceholderText("开始智能搜索对话...")
        self.chat_display.setMinimumHeight(280)
        smart_layout.addWidget(self.chat_display, 1)

        chat_input_layout = QHBoxLayout()
        self.chat_input = QLineEdit()
        self.chat_input.setPlaceholderText("输入你的回答...")
        self.chat_input.returnPressed.connect(self.send_chat_message)
        self.chat_send_btn = QPushButton("发送")
        self.chat_send_btn.clicked.connect(self.send_chat_message)
        self.chat_search_btn = QPushButton("开始搜索")
        self.chat_search_btn.clicked.connect(self.execute_smart_search)
        self.chat_search_btn.setEnabled(False)
        self.chat_reset_btn = QPushButton("重置")
        self.chat_reset_btn.clicked.connect(self.reset_smart_search)
        chat_input_layout.addWidget(self.chat_input)
        chat_input_layout.addWidget(self.chat_send_btn)
        chat_input_layout.addWidget(self.chat_search_btn)
        self.chat_cancel_btn = QPushButton("❌ 取消")
        self.chat_cancel_btn.clicked.connect(self.cancel_search)
        self.chat_cancel_btn.setVisible(False)
        chat_input_layout.addWidget(self.chat_cancel_btn)
        chat_input_layout.addWidget(self.chat_reset_btn)
        smart_layout.addLayout(chat_input_layout)
        search_stack_layout.addWidget(self.smart_search_widget)

        # 普通搜索区域
        self.normal_search_widget = QWidget()
        normal_layout = QVBoxLayout(self.normal_search_widget)
        normal_layout.setContentsMargins(0, 0, 0, 0)
        self.query_input = QLineEdit()
        self.query_input.setPlaceholderText("输入搜索关键词，如: large language model, transformer...")
        self.query_input.returnPressed.connect(self.do_search)
        normal_layout.addWidget(self.query_input)

        # 时间范围
        date_layout = QHBoxLayout()
        self.period_combo = QComboBox()
        self.period_combo.addItems(["昨天", "过去一周", "过去一个月", "过去三个月", "过去一年", "自定义"])
        self.period_combo.currentTextChanged.connect(self.on_period_changed)
        self.start_date = QDateEdit(QDate.currentDate().addDays(-7))
        self.end_date = QDateEdit(QDate.currentDate())
        self.start_date.setEnabled(False)
        self.end_date.setEnabled(False)
        date_layout.addWidget(QLabel("时间:"))
        date_layout.addWidget(self.period_combo)
        date_layout.addWidget(self.start_date)
        date_layout.addWidget(self.end_date)
        normal_layout.addLayout(date_layout)

        # 结果数量和来源
        options_layout = QHBoxLayout()
        self.max_results_spin = QSpinBox()
        self.max_results_spin.setRange(10, 2000)
        self.max_results_spin.setValue(50)
        self.max_results_spin.setSingleStep(10)
        self.arxiv_check = QCheckBox("arXiv")
        self.arxiv_check.setChecked(True)
        self.github_check = QCheckBox("GitHub")
        self.github_check.setChecked(True)
        self.huggingface_check = QCheckBox("HuggingFace")
        self.huggingface_check.setToolTip("搜索 Hugging Face 模型")
        self.modelscope_check = QCheckBox("ModelScope")
        self.modelscope_check.setToolTip("搜索 ModelScope 模型（国内）")
        options_layout.addWidget(QLabel("数量:"))
        options_layout.addWidget(self.max_results_spin)
        options_layout.addWidget(self.arxiv_check)
        options_layout.addWidget(self.github_check)
        options_layout.addWidget(self.huggingface_check)
        options_layout.addWidget(self.modelscope_check)
        options_layout.addStretch()
        normal_layout.addLayout(options_layout)

        search_btn_layout = QHBoxLayout()
        self.search_btn = QPushButton("🔍 搜索")
        self.search_btn.clicked.connect(self.do_search)
        self.cancel_search_btn = QPushButton("❌ 取消")
        self.cancel_search_btn.clicked.connect(self.cancel_search)
        self.cancel_search_btn.setVisible(False)
        search_btn_layout.addWidget(self.search_btn)
        search_btn_layout.addWidget(self.cancel_search_btn)
        normal_layout.addLayout(search_btn_layout)
        self.normal_search_widget.setVisible(False)
        search_stack_layout.addWidget(self.normal_search_widget)

        left_layout.addWidget(self.search_stack)

        # 结果列表
        result_group = QGroupBox("📋 搜索结果")
        result_layout = QVBoxLayout(result_group)

        select_layout = QHBoxLayout()
        self.select_all_btn = QPushButton("全选")
        self.select_all_btn.clicked.connect(self.select_all)
        self.deselect_all_btn = QPushButton("取消全选")
        self.deselect_all_btn.clicked.connect(self.deselect_all)
        self.result_count_label = QLabel("共 0 条结果")
        select_layout.addWidget(self.select_all_btn)
        select_layout.addWidget(self.deselect_all_btn)
        select_layout.addStretch()
        select_layout.addWidget(self.result_count_label)
        result_layout.addLayout(select_layout)

        self.result_list = QListWidget()
        self.result_list.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self.result_list.itemClicked.connect(self.on_item_clicked)
        result_layout.addWidget(self.result_list, 1)

        btn_layout = QHBoxLayout()
        self.analyze_btn = QPushButton("📊 分析选中")
        self.analyze_btn.clicked.connect(self.analyze_selected)
        self.vision_check = QCheckBox("🖼️ 图片分析")
        self.vision_check.setToolTip("启用后将提取并分析图片（需要多模态模型如gpt-4o）")
        self.download_btn = QPushButton("📥 下载")
        self.download_btn.clicked.connect(self.download_selected)
        self.open_btn = QPushButton("🔗 打开")
        self.open_btn.clicked.connect(self.open_selected)
        btn_layout.addWidget(self.analyze_btn)
        btn_layout.addWidget(self.vision_check)
        btn_layout.addWidget(self.download_btn)
        btn_layout.addWidget(self.open_btn)
        result_layout.addLayout(btn_layout)

        self.progress = QProgressBar()
        self.progress.setVisible(False)
        result_layout.addWidget(self.progress)
        self.progress_label = QLabel("")
        result_layout.addWidget(self.progress_label)
        left_layout.addWidget(result_group, 1)

        # 右侧分析结果
        right = QWidget()
        right_layout = QVBoxLayout(right)
        right_layout.setSpacing(12)

        self.analysis_tabs = QTabWidget()
        self.analysis_text = QWebEngineView()
        self.similar_text = QWebEngineView()
        self.batch_text = QWebEngineView()

        # 设置默认深色背景
        default_html = '<!DOCTYPE html><html><body style="background:#282a36;margin:0;"></body></html>'
        self.analysis_text.setHtml(default_html)
        self.similar_text.setHtml(default_html)
        self.batch_text.setHtml(default_html)

        self.analysis_tabs.addTab(self.analysis_text, "📝 详情/分析")
        self.analysis_tabs.addTab(self.similar_text, "🔗 相关研究")
        self.analysis_tabs.addTab(self.batch_text, "📊 批量分析结果")
        right_layout.addWidget(self.analysis_tabs)

        self.find_similar_btn = QPushButton("🔍 查找相关研究 (多智能体深度分析)")
        self.find_similar_btn.clicked.connect(self.find_similar)
        right_layout.addWidget(self.find_similar_btn)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(left)
        splitter.addWidget(right)
        splitter.setSizes([750, 650])
        layout.addWidget(splitter)

        # 初始化智能搜索
        self.init_smart_search()

    def switch_search_mode(self, mode):
        """切换搜索模式"""
        self.search_mode = mode
        self.smart_mode_btn.setChecked(mode == "smart")
        self.normal_mode_btn.setChecked(mode == "normal")
        self.smart_search_widget.setVisible(mode == "smart")
        self.normal_search_widget.setVisible(mode == "normal")

    def init_smart_search(self):
        """初始化智能搜索"""
        self.chat_display.clear()
        self.chat_display.append("🧠 **智能搜索助手**\n\n我会通过几个问题了解你的研究需求，然后为你精准搜索相关论文和项目。\n\n---\n")
        if self.api_key.text():
            SmartSearchWorker._system = None
            self._smart_search_ask("")

    def apply_style(self):
        # 现代深色主题配色 - Dracula风格
        self.setStyleSheet("""
            QMainWindow { background: qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 #1a1a2e, stop:1 #16213e); }
            QWidget { color: #f8f8f2; font-family: 'Segoe UI', 'Microsoft YaHei'; }
            QGroupBox {
                font-weight: bold; font-size: 13px;
                border: 2px solid #44475a; border-radius: 10px;
                margin-top: 12px; padding: 15px; padding-top: 25px;
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #282a36, stop:1 #21222c);
            }
            QGroupBox::title {
                subcontrol-origin: margin; left: 15px; padding: 0 8px;
                color: #bd93f9;
            }
            QPushButton {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #bd93f9, stop:1 #9580ff);
                color: #282a36; border: none; padding: 10px 20px;
                border-radius: 6px; font-weight: bold; font-size: 12px;
            }
            QPushButton:hover {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #ff79c6, stop:1 #ff6bcb);
            }
            QPushButton:pressed { background: #6272a4; }
            QPushButton:disabled { background: #44475a; color: #6272a4; }
            QLineEdit, QComboBox, QDateEdit, QSpinBox {
                padding: 8px 12px; border: 2px solid #44475a; border-radius: 6px;
                background: #282a36; color: #f8f8f2; selection-background-color: #44475a;
            }
            QLineEdit:focus, QComboBox:focus, QSpinBox:focus { border-color: #bd93f9; }
            QComboBox::drop-down { border: none; padding-right: 10px; }
            QComboBox QAbstractItemView { background: #282a36; border: 2px solid #44475a; selection-background-color: #44475a; }
            QListWidget {
                border: 2px solid #44475a; border-radius: 8px;
                background: #282a36; alternate-background-color: #21222c;
            }
            QListWidget::item { padding: 10px; border-bottom: 1px solid #44475a; }
            QListWidget::item:selected { background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #44475a, stop:1 #6272a4); }
            QListWidget::item:hover { background: #383a59; }
            QTextEdit {
                border: 2px solid #44475a; border-radius: 8px;
                padding: 12px; background: #282a36; color: #f8f8f2;
                selection-background-color: #44475a;
            }
            QTabWidget::pane { border: 2px solid #44475a; border-radius: 8px; background: #282a36; }
            QTabBar::tab {
                background: #21222c; color: #6272a4; padding: 10px 20px;
                border-top-left-radius: 6px; border-top-right-radius: 6px;
                margin-right: 2px;
            }
            QTabBar::tab:selected { background: #282a36; color: #50fa7b; border-bottom: 2px solid #50fa7b; }
            QTabBar::tab:hover { color: #f8f8f2; }
            QProgressBar {
                border: 2px solid #44475a; border-radius: 6px;
                background: #21222c; text-align: center; color: #f8f8f2;
            }
            QProgressBar::chunk { background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #50fa7b, stop:1 #8be9fd); border-radius: 4px; }
            QLabel { color: #f8f8f2; }
            QCheckBox { color: #f8f8f2; spacing: 8px; }
            QCheckBox::indicator { width: 18px; height: 18px; border-radius: 4px; border: 2px solid #44475a; background: #282a36; }
            QCheckBox::indicator:checked { background: #50fa7b; border-color: #50fa7b; }
            QSplitter::handle { background: #44475a; width: 3px; }
            QScrollBar:vertical { background: #21222c; width: 12px; border-radius: 6px; }
            QScrollBar::handle:vertical { background: #44475a; border-radius: 6px; min-height: 30px; }
            QPushButton:checked { background: #50fa7b; color: #282a36; }
            QScrollBar::handle:vertical:hover { background: #6272a4; }
        """)

    def select_all(self):
        self.result_list.selectAll()

    def deselect_all(self):
        self.result_list.clearSelection()

    def on_period_changed(self, text):
        custom = text == "自定义"
        self.start_date.setEnabled(custom)
        self.end_date.setEnabled(custom)

    def do_search(self):
        query = self.query_input.text().strip()
        if not query:
            QMessageBox.warning(self, "错误", "请输入搜索关键词")
            return

        period_map = {"昨天": "yesterday", "过去一周": "past_week", "过去一个月": "past_month",
                      "过去三个月": "past_3months", "过去一年": "past_year"}
        if self.period_combo.currentText() == "自定义":
            start = self.start_date.date().toPyDate()
            end = self.end_date.date().toPyDate()
            start_dt, end_dt = datetime.combine(start, datetime.min.time()), datetime.combine(end, datetime.max.time())
        else:
            start_dt, end_dt = get_date_range(period_map[self.period_combo.currentText()])

        self.progress.setVisible(True)
        self.progress.setRange(0, 0)
        self.search_btn.setEnabled(False)
        self.cancel_search_btn.setVisible(True)
        self.progress_label.setText("正在搜索...")

        self.worker = SearchWorker(query, start_dt, end_dt, self.arxiv_check.isChecked(),
                                   self.github_check.isChecked(), self.github_token.text() or None,
                                   self.max_results_spin.value(),
                                   self.huggingface_check.isChecked(), self.modelscope_check.isChecked())
        self.worker.finished.connect(self.on_search_finished)
        self.worker.error.connect(self.on_search_error)
        self.worker.start()

    def on_search_finished(self, results):
        self.progress.setVisible(False)
        self.search_btn.setEnabled(True)
        self.cancel_search_btn.setVisible(False)
        self.progress_label.setText("")
        self.results = results
        self.result_list.clear()
        for r in results:
            source = r['source']
            if source == 'arxiv':
                icon, extra = "📄", r['published']
            elif source == 'github':
                icon, extra = "📦", f"⭐{r['stars']}"
            elif source == 'huggingface':
                icon, extra = "🤗", f"⬇{r.get('downloads', 0)}"
            elif source == 'modelscope':
                icon, extra = "🔮", f"⬇{r.get('downloads', 0)}"
            else:
                icon, extra = "📋", ""
            item = QListWidgetItem(f"{icon} [{extra}] {r['title'][:55]}...")
            item.setData(Qt.ItemDataRole.UserRole, r)
            self.result_list.addItem(item)
        self.result_count_label.setText(f"共 {len(results)} 条结果")
        self.analysis_text.setHtml(md_to_html(f"✅ 找到 {len(results)} 个结果\n\n点击查看详情，或多选后点击'分析选中'进行批量深度分析。\n\n支持 Ctrl+点击 多选，Shift+点击 范围选择。"))

    def on_search_error(self, error):
        self.progress.setVisible(False)
        self.search_btn.setEnabled(True)
        self.cancel_search_btn.setVisible(False)
        self.progress_label.setText("")
        QMessageBox.critical(self, "搜索错误", error)

    def cancel_search(self):
        """取消当前搜索"""
        if hasattr(self, 'worker') and self.worker.isRunning():
            self.worker.cancel()
            self.progress_label.setText("正在取消...")
        if hasattr(self, 'explore_worker') and self.explore_worker.isRunning():
            self.explore_worker.terminate()
        if hasattr(self, 'search_worker') and self.search_worker.isRunning():
            self.search_worker.terminate()
        self.progress.setVisible(False)
        self.search_btn.setEnabled(True)
        self.cancel_search_btn.setVisible(False)
        self.chat_cancel_btn.setVisible(False)
        self.explore_btn.setEnabled(True)
        self.progress_label.setText("已取消")

    def on_item_clicked(self, item):
        data = item.data(Qt.ItemDataRole.UserRole)
        source = data['source']
        if source == 'arxiv':
            info = f"# 📄 {data['title']}\n\n**👥 作者:** {', '.join(data['authors'])}\n\n**📅 发布:** {data['published']}\n\n**🏷️ 分类:** {', '.join(data['categories'])}\n\n## 📝 摘要\n{data['abstract']}\n\n**🔗 链接:** {data['url']}\n\n**📥 PDF:** {data['pdf_url']}"
        elif source == 'github':
            info = f"# 📦 {data['title']}\n\n**📝 描述:** {data['description']}\n\n**💻 语言:** {data['language']}\n\n**⭐ Stars:** {data['stars']}\n\n**📅 更新:** {data['updated']}\n\n**🏷️ Topics:** {', '.join(data['topics'])}\n\n**🔗 仓库:** {data['url']}"
        elif source == 'huggingface':
            info = f"# 🤗 {data['title']}\n\n**📝 描述:** {data.get('description', '')}\n\n**⬇️ 下载:** {data.get('downloads', 0)}\n\n**❤️ 点赞:** {data.get('likes', 0)}\n\n**📅 更新:** {data.get('updated', '')}\n\n**🏷️ 标签:** {', '.join(data.get('tags', []))}\n\n**🔗 链接:** {data['url']}"
        elif source == 'modelscope':
            info = f"# 🔮 {data['title']}\n\n**📝 描述:** {data.get('description', '')}\n\n**⬇️ 下载:** {data.get('downloads', 0)}\n\n**❤️ 点赞:** {data.get('likes', 0)}\n\n**📅 更新:** {data.get('updated', '')}\n\n**🏷️ 标签:** {', '.join(data.get('tags', []))}\n\n**🔗 链接:** {data['url']}"
        else:
            info = f"# 📋 {data['title']}\n\n**🔗 链接:** {data['url']}"
        self.analysis_text.setHtml(md_to_html(info))

    def analyze_selected(self):
        selected = self.result_list.selectedItems()
        if not selected:
            QMessageBox.warning(self, "提示", "请先选择要分析的项目")
            return
        if not self.api_key.text():
            QMessageBox.warning(self, "错误", "请输入API Key")
            return

        papers = [item.data(Qt.ItemDataRole.UserRole) for item in selected]

        self.progress.setVisible(True)
        self.progress.setRange(0, len(papers))
        self.progress.setValue(0)
        self.analyze_btn.setEnabled(False)
        self.progress_label.setText(f"准备分析 {len(papers)} 个项目...")

        self.batch_worker = BatchAnalyzeWorker(
            self.base_url.text(), self.api_key.text(), self.model_name.text(),
            papers, self.github_token.text() or None, fetch_images=self.vision_check.isChecked()
        )
        self.batch_worker.finished.connect(self.on_batch_finished)
        self.batch_worker.progress_update.connect(self.on_batch_progress)
        self.batch_worker.error.connect(self.on_batch_error)
        self.batch_worker.start()

    def on_batch_progress(self, msg, current, total):
        self.progress.setValue(current)
        self.progress_label.setText(msg)

    def on_batch_finished(self, results):
        self.progress.setVisible(False)
        self.analyze_btn.setEnabled(True)
        self.progress_label.setText(f"✅ 完成 {len(results)} 个项目的分析")
        self.analysis_results.update(results)

        # 显示批量结果
        output = "# 📊 批量分析结果\n\n"
        for title, result in results.items():
            output += f"---\n## 📄 {title[:60]}...\n\n{result}\n\n"
        self.batch_text.setHtml(md_to_html(output))
        self.analysis_tabs.setCurrentIndex(2)

    def on_batch_error(self, error):
        self.progress.setVisible(False)
        self.analyze_btn.setEnabled(True)
        self.progress_label.setText("")
        QMessageBox.critical(self, "分析错误", error)

    def find_similar(self):
        item = self.result_list.currentItem()
        if not item:
            QMessageBox.warning(self, "提示", "请先选择一个项目")
            return
        if not self.api_key.text():
            QMessageBox.warning(self, "错误", "请输入API Key")
            return

        self.progress.setVisible(True)
        self.progress.setRange(0, 0)
        self.find_similar_btn.setEnabled(False)
        self.progress_label.setText("多智能体系统正在搜索和分析相关研究...")

        self.related_worker = RelatedWorkWorker(
            self.base_url.text(), self.api_key.text(), self.model_name.text(),
            item.data(Qt.ItemDataRole.UserRole)
        )
        self.related_worker.finished.connect(self.on_related_finished)
        self.related_worker.progress_update.connect(lambda msg: self.progress_label.setText(msg))
        self.related_worker.error.connect(self.on_related_error)
        self.related_worker.start()

    def on_related_finished(self, result):
        self.progress.setVisible(False)
        self.find_similar_btn.setEnabled(True)
        self.progress_label.setText("")
        self.similar_text.setHtml(md_to_html(result))
        self.analysis_tabs.setCurrentIndex(1)

    def on_related_error(self, error):
        self.progress.setVisible(False)
        self.find_similar_btn.setEnabled(True)
        self.progress_label.setText("")
        QMessageBox.critical(self, "错误", error)

    def download_selected(self):
        selected = self.result_list.selectedItems()
        if not selected:
            QMessageBox.warning(self, "提示", "请先选择要下载的项目")
            return

        folder = QFileDialog.getExistingDirectory(self, "选择保存目录")
        if not folder:
            return

        downloaded = 0
        for item in selected:
            data = item.data(Qt.ItemDataRole.UserRole)
            if data['source'] == 'arxiv':
                try:
                    filename = f"{data['title'][:50].replace('/', '_')}.pdf"
                    response = requests.get(data['pdf_url'])
                    with open(os.path.join(folder, filename), 'wb') as f:
                        f.write(response.content)
                    downloaded += 1
                except: pass

        QMessageBox.information(self, "完成", f"已下载 {downloaded} 个PDF文件到:\n{folder}")

    def open_selected(self):
        selected = self.result_list.selectedItems()
        for item in selected[:5]:  # 最多打开5个
            webbrowser.open(item.data(Qt.ItemDataRole.UserRole)['url'])

    def do_explore(self):
        """闲逛功能：搜索热门项目"""
        query = self.query_input.text().strip()
        if not query:
            query = "machine learning"  # 默认关键词

        search_new = self.explore_type_combo.currentText() == "新项目"
        max_results = self.explore_count_spin.value()

        self.progress.setVisible(True)
        self.progress.setRange(0, 0)
        self.search_btn.setEnabled(False)
        self.explore_btn.setEnabled(False)

        type_text = "新创建" if search_new else "活跃"
        self.progress_label.setText(f"🎲 正在搜索近3天{type_text}的热门项目...")

        self.explore_worker = ExploreWorker(query, self.github_token.text() or None,
                                           max_results=max_results, search_new=search_new,
                                           search_arxiv=self.explore_arxiv_check.isChecked(),
                                           search_github=self.explore_github_check.isChecked(),
                                           search_hf=self.explore_hf_check.isChecked(),
                                           search_ms=self.explore_ms_check.isChecked())
        self.explore_worker.finished.connect(self.on_explore_finished)
        self.explore_worker.error.connect(self.on_search_error)
        self.explore_worker.start()

    def on_explore_finished(self, results):
        self.progress.setVisible(False)
        self.search_btn.setEnabled(True)
        self.explore_btn.setEnabled(True)
        self.progress_label.setText("")
        self.results = results
        self.result_list.clear()
        for r in results:
            source = r['source']
            if source == 'arxiv':
                icon, extra = "📄", r['published']
            elif source == 'github':
                icon, extra = "🔥", f"⭐{r['stars']}"
            elif source == 'huggingface':
                icon, extra = "🤗", f"⬇{r.get('downloads', 0)}"
            elif source == 'modelscope':
                icon, extra = "🔮", f"⬇{r.get('downloads', 0)}"
            else:
                icon, extra = "📋", ""
            item = QListWidgetItem(f"{icon} [{extra}] {r['title'][:55]}...")
            item.setData(Qt.ItemDataRole.UserRole, r)
            self.result_list.addItem(item)
        self.result_count_label.setText(f"共 {len(results)} 条热门结果")
        self.analysis_text.setHtml(md_to_html(f"🔥 找到 {len(results)} 个近3天热门项目\n\n选择后可进行深度分析。"))

    # ==================== 智能搜索功能 ====================
    def _smart_search_ask(self, user_input):
        """发送问题给智能搜索系统"""
        self.chat_send_btn.setEnabled(False)
        self.smart_worker = SmartSearchWorker(
            self.base_url.text(), self.api_key.text(), self.model_name.text(),
            "ask", user_input, self.github_token.text() or None
        )
        self.smart_worker.question_ready.connect(self.on_smart_question)
        self.smart_worker.error.connect(self.on_smart_error)
        self.smart_worker.start()

    def send_chat_message(self):
        """发送用户消息"""
        msg = self.chat_input.text().strip()
        if not msg:
            return
        self.chat_display.append(f"**你:** {msg}\n")
        self.chat_input.clear()
        self._smart_search_ask(msg)

    def on_smart_question(self, result):
        """处理智能搜索返回的问题"""
        self.chat_send_btn.setEnabled(True)
        # 清理消息中的更新标记
        msg = result['message']
        if "【更新】" in msg:
            msg = msg[:msg.find("【更新】")] + msg[msg.find("【/更新】")+5:] if "【/更新】" in msg else msg[:msg.find("【更新】")]
        msg = msg.replace("【搜索就绪】", "").replace("【READY】", "").strip()
        self.chat_display.append(f"**助手:** {msg}\n")

        if result['type'] == 'ready':
            self.chat_search_btn.setEnabled(True)
            self.chat_display.append("\n✅ **信息收集完成！** 点击「开始搜索」按钮开始智能搜索。\n")

    def on_smart_error(self, error):
        self.chat_send_btn.setEnabled(True)
        self.chat_display.append(f"\n❌ **错误:** {error}\n")

    def execute_smart_search(self):
        """执行智能搜索"""
        self.chat_search_btn.setEnabled(False)
        self.chat_cancel_btn.setVisible(True)
        self.chat_display.append("\n🔍 **开始智能搜索...**\n")
        self.progress.setVisible(True)
        self.progress.setRange(0, 0)

        # 获取选中的数据源
        sources = []
        if self.smart_arxiv_check.isChecked(): sources.append('arxiv')
        if self.smart_github_check.isChecked(): sources.append('github')
        if self.smart_hf_check.isChecked(): sources.append('huggingface')
        if self.smart_ms_check.isChecked(): sources.append('modelscope')

        self.search_worker = SmartSearchWorker(
            self.base_url.text(), self.api_key.text(), self.model_name.text(),
            "search", "", self.github_token.text() or None, sources=sources
        )
        self.search_worker.search_progress.connect(lambda msg: self.chat_display.append(f"  → {msg}\n"))
        self.search_worker.results_ready.connect(self.on_smart_results)
        self.search_worker.error.connect(self.on_smart_error)
        self.search_worker.start()

    def on_smart_results(self, results):
        """处理智能搜索结果"""
        self.progress.setVisible(False)
        self.chat_cancel_btn.setVisible(False)
        self.chat_display.append(f"\n✅ **搜索完成！** 找到 {len(results)} 个匹配结果。\n")
        self.results = results
        self.result_list.clear()
        for r in results:
            source = r['source']
            if source == 'arxiv':
                icon, extra = "📄", r['published']
            elif source == 'github':
                icon, extra = "📦", f"⭐{r['stars']}"
            elif source == 'huggingface':
                icon, extra = "🤗", f"⬇{r.get('downloads', 0)}"
            elif source == 'modelscope':
                icon, extra = "🔮", f"⬇{r.get('downloads', 0)}"
            else:
                icon, extra = "📋", ""
            item = QListWidgetItem(f"{icon} [{extra}] {r['title'][:55]}...")
            item.setData(Qt.ItemDataRole.UserRole, r)
            self.result_list.addItem(item)
        self.result_count_label.setText(f"共 {len(results)} 条智能筛选结果")

    def reset_smart_search(self):
        """重置智能搜索"""
        SmartSearchWorker._system = None
        self.chat_search_btn.setEnabled(False)
        self.init_smart_search()

def main():
    app = QApplication(sys.argv)
    app.setFont(QFont("Segoe UI", 10))
    window = MainWindow()
    window.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()

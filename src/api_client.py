import arxiv
from github import Github
from datetime import datetime, timedelta
from typing import Optional
import re
import base64

def get_date_range(period: str) -> tuple[datetime, datetime]:
    """根据时间段返回日期范围"""
    end = datetime.now()
    periods = {
        "yesterday": 1, "past_week": 7, "past_month": 30,
        "past_3months": 90, "past_year": 365
    }
    days = periods.get(period, 7)
    return end - timedelta(days=days), end

def search_arxiv(query: str, start_date: datetime, end_date: datetime, max_results: int = 20) -> list[dict]:
    """搜索arXiv论文，支持分批搜索"""
    client = arxiv.Client()
    results = []
    batch_size = 50
    offset = 0

    while len(results) < max_results:
        remaining = max_results - len(results)
        current_batch = min(batch_size, remaining)
        search = arxiv.Search(query=query, max_results=current_batch, sort_by=arxiv.SortCriterion.SubmittedDate)
        search.start = offset

        batch_results = []
        for paper in client.results(search):
            if start_date <= paper.published.replace(tzinfo=None) <= end_date:
                batch_results.append({
                    "title": paper.title, "authors": [a.name for a in paper.authors],
                    "abstract": paper.summary, "url": paper.entry_id,
                    "pdf_url": paper.pdf_url, "published": paper.published.strftime("%Y-%m-%d"),
                    "categories": paper.categories, "source": "arxiv"
                })

        if not batch_results:
            break
        results.extend(batch_results)
        offset += current_batch
        if len(batch_results) < current_batch:
            break

    return results[:max_results]

def search_github(query: str, start_date: datetime, end_date: datetime, token: Optional[str] = None, max_results: int = 20) -> list[dict]:
    """搜索GitHub项目，支持分批搜索"""
    g = Github(token) if token else Github()
    date_query = f"{query} pushed:{start_date.strftime('%Y-%m-%d')}..{end_date.strftime('%Y-%m-%d')}"
    results = []
    batch_size = 30
    page = 0

    while len(results) < max_results:
        remaining = max_results - len(results)
        repos = g.search_repositories(date_query, sort="stars", order="desc")
        batch = list(repos.get_page(page))
        if not batch:
            break
        for repo in batch[:remaining]:
            results.append({
                "title": repo.full_name, "description": repo.description or "",
                "url": repo.html_url, "stars": repo.stargazers_count,
                "language": repo.language, "updated": repo.updated_at.strftime("%Y-%m-%d"),
                "topics": repo.get_topics(), "source": "github"
            })
            if len(results) >= max_results:
                break
        page += 1
        if len(batch) < batch_size:
            break

    return results

def search_trending(query: str, token: Optional[str] = None, max_results: int = 20) -> list[dict]:
    """搜索过去3天内热门项目（按stars排序）"""
    end_date = datetime.now()
    start_date = end_date - timedelta(days=3)
    results = []

    # 搜索GitHub热门项目
    g = Github(token) if token else Github()
    date_query = f"{query} pushed:{start_date.strftime('%Y-%m-%d')}..{end_date.strftime('%Y-%m-%d')}"
    repos = g.search_repositories(date_query, sort="stars", order="desc")
    for repo in repos[:max_results // 2]:
        results.append({
            "title": repo.full_name, "description": repo.description or "",
            "url": repo.html_url, "stars": repo.stargazers_count,
            "language": repo.language, "updated": repo.updated_at.strftime("%Y-%m-%d"),
            "topics": repo.get_topics(), "source": "github"
        })

    # 搜索arXiv最新论文
    arxiv_results = search_arxiv(query, start_date, end_date, max_results // 2)
    results.extend(arxiv_results)

    # 按热度排序（GitHub按stars，arXiv按时间）
    results.sort(key=lambda x: x.get('stars', 0) if x['source'] == 'github' else 0, reverse=True)
    return results

def get_repo_content(repo_name: str, token: Optional[str] = None) -> dict:
    """获取GitHub仓库的详细内容用于深度分析"""
    g = Github(token) if token else Github()
    repo = g.get_repo(repo_name)
    content = {"readme": "", "structure": [], "key_files": []}

    # 获取README
    for name in ["README.md", "README.rst", "README.txt", "README"]:
        try:
            readme = repo.get_contents(name)
            content["readme"] = base64.b64decode(readme.content).decode('utf-8', errors='ignore')
            break
        except: pass

    # 获取项目结构和关键文件
    code_exts = {'.py', '.js', '.ts', '.java', '.go', '.rs', '.cpp', '.c', '.h'}
    doc_exts = {'.md', '.rst', '.txt'}

    def scan_dir(path="", depth=0):
        if depth > 2: return
        try:
            items = repo.get_contents(path)
            for item in items:
                if item.type == "dir" and not item.name.startswith('.'):
                    content["structure"].append(f"{'  '*depth}📁 {item.name}/")
                    scan_dir(item.path, depth + 1)
                elif item.type == "file":
                    ext = '.' + item.name.split('.')[-1] if '.' in item.name else ''
                    content["structure"].append(f"{'  '*depth}📄 {item.name}")
                    # 读取关键文件（限制大小和数量）
                    if len(content["key_files"]) < 10 and item.size < 50000:
                        if ext in code_exts or ext in doc_exts or item.name in ['setup.py', 'requirements.txt', 'package.json']:
                            try:
                                file_content = base64.b64decode(repo.get_contents(item.path).content).decode('utf-8', errors='ignore')
                                content["key_files"].append({"name": item.path, "content": file_content[:8000]})
                            except: pass
        except: pass

    scan_dir()
    return content

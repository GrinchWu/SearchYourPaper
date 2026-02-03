import arxiv
from github import Github
from datetime import datetime, timedelta
from typing import Optional
import re
import base64
import requests

def extract_images_from_readme(readme_content: str, repo_name: str, token: str = None) -> list:
    """从README中提取图片URL并下载为base64"""
    images = []
    # 匹配 Markdown 图片: ![alt](url) 和 HTML img: <img src="url">
    patterns = [
        r'!\[([^\]]*)\]\(([^)]+)\)',  # Markdown
        r'<img[^>]+src=["\']([^"\']+)["\'][^>]*>',  # HTML
    ]

    urls = []
    for pattern in patterns:
        matches = re.findall(pattern, readme_content)
        for m in matches:
            url = m[1] if isinstance(m, tuple) and len(m) > 1 else m
            if url and not url.startswith('http'):
                # 相对路径转绝对路径
                url = f"https://raw.githubusercontent.com/{repo_name}/main/{url}"
            if url:
                urls.append(url)

    # 下载图片（限制数量和大小）
    headers = {"Authorization": f"token {token}"} if token else {}
    for url in urls[:5]:  # 最多5张图
        try:
            if any(ext in url.lower() for ext in ['.png', '.jpg', '.jpeg', '.gif', '.webp', '.svg']):
                resp = requests.get(url, headers=headers, timeout=10)
                if resp.status_code == 200 and len(resp.content) < 5 * 1024 * 1024:  # <5MB
                    b64 = base64.b64encode(resp.content).decode()
                    ext = url.split('.')[-1].split('?')[0].lower()
                    mime = {'png': 'image/png', 'jpg': 'image/jpeg', 'jpeg': 'image/jpeg',
                            'gif': 'image/gif', 'webp': 'image/webp', 'svg': 'image/svg+xml'}.get(ext, 'image/png')
                    images.append({"url": f"data:{mime};base64,{b64}", "source_url": url})
        except:
            pass
    return images

def extract_images_from_pdf(pdf_url: str, max_images: int = 5) -> list:
    """从PDF中提取图片（需要PyMuPDF）"""
    images = []
    try:
        import fitz  # PyMuPDF

        # 下载PDF
        resp = requests.get(pdf_url, timeout=30)
        if resp.status_code != 200:
            return images

        doc = fitz.open(stream=resp.content, filetype="pdf")
        img_count = 0

        # 优先提取前几页的大图（通常是架构图、流程图）
        for page_num in range(min(len(doc), 15)):  # 扫描前15页
            page = doc[page_num]

            # 方法1: 提取嵌入图片
            image_list = page.get_images()
            for img_index, img in enumerate(image_list):
                if img_count >= max_images:
                    break
                try:
                    xref = img[0]
                    base_image = doc.extract_image(xref)
                    image_bytes = base_image["image"]
                    image_ext = base_image["ext"]

                    # 降低阈值，保留更多图片（>5KB）
                    if len(image_bytes) < 5000:
                        continue

                    b64 = base64.b64encode(image_bytes).decode()
                    mime = f"image/{image_ext}"
                    images.append({
                        "url": f"data:{mime};base64,{b64}",
                        "page": page_num + 1,
                        "description": f"Page {page_num + 1}, Image {img_index + 1}"
                    })
                    img_count += 1
                except:
                    pass

            # 方法2: 如果嵌入图片不够，渲染页面为图片（捕获图表）
            if img_count < max_images and page_num < 5:
                try:
                    # 渲染页面为图片
                    mat = fitz.Matrix(2, 2)  # 2x缩放
                    pix = page.get_pixmap(matrix=mat)
                    if pix.width > 200 and pix.height > 200:
                        img_bytes = pix.tobytes("png")
                        if len(img_bytes) > 10000:  # >10KB
                            b64 = base64.b64encode(img_bytes).decode()
                            images.append({
                                "url": f"data:image/png;base64,{b64}",
                                "page": page_num + 1,
                                "description": f"Page {page_num + 1} (rendered)"
                            })
                            img_count += 1
                except:
                    pass

            if img_count >= max_images:
                break
        doc.close()
    except ImportError:
        pass  # PyMuPDF 未安装
    except:
        pass
    return images

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

def search_trending(query: str, token: Optional[str] = None, max_results: int = 20, search_new: bool = True) -> list[dict]:
    """搜索热门项目

    Args:
        query: 搜索关键词
        token: GitHub token
        max_results: 最大结果数
        search_new: True=搜索新创建的项目(created:), False=搜索有更新的项目(pushed:)
    """
    end_date = datetime.now()
    start_date = end_date - timedelta(days=3)
    results = []

    # 搜索GitHub热门项目
    g = Github(token) if token else Github()
    date_param = "created" if search_new else "pushed"
    date_query = f"{query} {date_param}:{start_date.strftime('%Y-%m-%d')}..{end_date.strftime('%Y-%m-%d')}"
    repos = g.search_repositories(date_query, sort="stars", order="desc")
    for repo in repos[:max_results]:
        results.append({
            "title": repo.full_name, "description": repo.description or "",
            "url": repo.html_url, "stars": repo.stargazers_count,
            "language": repo.language, "updated": repo.updated_at.strftime("%Y-%m-%d"),
            "created": repo.created_at.strftime("%Y-%m-%d"),
            "topics": repo.get_topics(), "source": "github"
        })

    # 按热度排序
    results.sort(key=lambda x: x.get('stars', 0), reverse=True)
    return results

def get_repo_content(repo_name: str, token: Optional[str] = None, fetch_images: bool = False) -> dict:
    """获取GitHub仓库的详细内容用于深度分析"""
    g = Github(token) if token else Github()
    repo = g.get_repo(repo_name)
    content = {"readme": "", "structure": [], "key_files": [], "images": []}

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

    # 提取图片（如果启用）
    if fetch_images and content["readme"]:
        content["images"] = extract_images_from_readme(content["readme"], repo_name, token)

    return content

# ==================== Hugging Face 搜索 ====================

def search_huggingface(query: str, start_date: datetime = None, end_date: datetime = None,
                       max_results: int = 20, search_type: str = "models") -> list[dict]:
    """搜索 Hugging Face 模型或数据集

    Args:
        query: 搜索关键词
        start_date: 开始日期（可选）
        end_date: 结束日期（可选）
        max_results: 最大结果数
        search_type: "models" 或 "datasets"
    """
    results = []
    try:
        from huggingface_hub import HfApi
        api = HfApi()

        if search_type == "models":
            items = api.list_models(
                search=query,
                sort="lastModified",
                direction=-1,
                limit=max_results
            )
        else:
            items = api.list_datasets(
                search=query,
                sort="lastModified",
                direction=-1,
                limit=max_results
            )

        for item in items:
            # 时间过滤
            if hasattr(item, 'lastModified') and item.lastModified:
                modified = item.lastModified.replace(tzinfo=None) if hasattr(item.lastModified, 'replace') else None
                if modified and start_date and end_date:
                    if not (start_date <= modified <= end_date):
                        continue

            results.append({
                "title": item.id,
                "description": getattr(item, 'description', '') or '',
                "url": f"https://huggingface.co/{item.id}",
                "downloads": getattr(item, 'downloads', 0) or 0,
                "likes": getattr(item, 'likes', 0) or 0,
                "updated": item.lastModified.strftime("%Y-%m-%d") if hasattr(item, 'lastModified') and item.lastModified else "",
                "tags": list(getattr(item, 'tags', []) or [])[:5],
                "source": "huggingface",
                "type": search_type
            })

            if len(results) >= max_results:
                break

    except ImportError:
        pass  # huggingface_hub 未安装
    except Exception as e:
        print(f"Hugging Face search error: {e}")

    return results

# ==================== ModelScope 搜索 ====================

def search_modelscope(query: str, start_date: datetime = None, end_date: datetime = None,
                      max_results: int = 20) -> list[dict]:
    """搜索 ModelScope 模型

    Args:
        query: 搜索关键词
        start_date: 开始日期（可选）
        end_date: 结束日期（可选）
        max_results: 最大结果数
    """
    results = []
    try:
        # 使用 ModelScope API（REST方式，不需要安装SDK）
        url = "https://modelscope.cn/api/v1/models"
        params = {
            "Query": query,
            "PageSize": max_results,
            "SortBy": "gmt_modified"
        }
        resp = requests.get(url, params=params, timeout=30)
        if resp.status_code == 200:
            data = resp.json()
            models = data.get("Data", {}).get("Models", [])

            for model in models:
                results.append({
                    "title": model.get("Name", ""),
                    "description": model.get("ChineseDescription", "") or model.get("Description", ""),
                    "url": f"https://modelscope.cn/models/{model.get('Path', '')}",
                    "downloads": model.get("Downloads", 0),
                    "likes": model.get("Likes", 0),
                    "updated": model.get("LastUpdatedTime", "")[:10] if model.get("LastUpdatedTime") else "",
                    "tags": model.get("Tags", [])[:5] if model.get("Tags") else [],
                    "source": "modelscope",
                    "type": "models"
                })

                if len(results) >= max_results:
                    break

    except Exception as e:
        print(f"ModelScope search error: {e}")

    return results

def get_huggingface_content(model_id: str) -> dict:
    """获取 Hugging Face 模型的详细内容用于深度分析"""
    content = {"readme": "", "model_info": "", "files": []}
    try:
        from huggingface_hub import HfApi, hf_hub_download
        api = HfApi()

        # 获取模型信息
        model_info = api.model_info(model_id)
        content["model_info"] = f"""
模型ID: {model_id}
作者: {getattr(model_info, 'author', '')}
下载量: {getattr(model_info, 'downloads', 0)}
点赞数: {getattr(model_info, 'likes', 0)}
标签: {', '.join(getattr(model_info, 'tags', []) or [])}
Pipeline: {getattr(model_info, 'pipeline_tag', '')}
库: {getattr(model_info, 'library_name', '')}
"""

        # 获取 README
        try:
            readme_path = hf_hub_download(repo_id=model_id, filename="README.md")
            with open(readme_path, 'r', encoding='utf-8', errors='ignore') as f:
                content["readme"] = f.read()[:15000]
        except:
            pass

        # 获取文件列表
        try:
            files = api.list_repo_files(model_id)
            content["files"] = files[:50]
        except:
            pass

    except Exception as e:
        print(f"Get HuggingFace content error: {e}")

    return content

def get_modelscope_content(model_path: str) -> dict:
    """获取 ModelScope 模型的详细内容用于深度分析"""
    content = {"readme": "", "model_info": "", "files": []}
    try:
        # 获取模型详情
        url = f"https://modelscope.cn/api/v1/models/{model_path}"
        resp = requests.get(url, timeout=30)
        if resp.status_code == 200:
            data = resp.json().get("Data", {})
            content["model_info"] = f"""
模型名称: {data.get('Name', '')}
描述: {data.get('ChineseDescription', '') or data.get('Description', '')}
下载量: {data.get('Downloads', 0)}
标签: {', '.join(data.get('Tags', []) or [])}
任务: {data.get('Task', '')}
"""
            content["readme"] = data.get("ReadmeContent", "")[:15000]

        # 获取文件列表
        files_url = f"https://modelscope.cn/api/v1/models/{model_path}/repo/files"
        files_resp = requests.get(files_url, timeout=30)
        if files_resp.status_code == 200:
            files_data = files_resp.json().get("Data", {}).get("Files", [])
            content["files"] = [f.get("Name", "") for f in files_data[:50]]

    except Exception as e:
        print(f"Get ModelScope content error: {e}")

    return content

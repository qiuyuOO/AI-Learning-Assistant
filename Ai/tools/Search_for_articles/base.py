from dataclasses import dataclass, field
from datetime import datetime
from typing import Any


@dataclass
class BaseArticle:
    
    id: str
    # 文章唯一 ID

    title: str
    # 文章标题

    url: str
    # 原文链接

    source: str
    # 信息来源，例如 OpenAI、Anthropic

    source_type: str
    # 数据来源类型，例如 rss、api、github

    published_at: datetime
    # 文章发布时间

    first_seen_at: datetime
    # 系统第一次发现文章的时间

    description: str | None = None
    # 文章摘要

    category: str | None = None
    # 文章分类，例如 Agent、LLM、RAG

    tags: list[str] = field(default_factory=list)
    # 文章标签

    raw_data: dict[str, Any] = field(default_factory=dict)
    # 数据源返回的原始数据
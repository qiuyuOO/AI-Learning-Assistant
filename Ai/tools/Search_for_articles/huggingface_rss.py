"""Hugging Face Blog RSS 查询工具。

★ 这个模块对外只暴露两样东西：
    HUGGINGFACE_RSS_SPEC —— 喂给模型的 ToolSpec（build_tools 用）
    HUGGINGFACE_RSS_TOOL —— 完整的 Tool（tool_registry.ALL_TOOLS 注册用）

  handler / validate 是模型不该看见的实现细节，统一收在 Tool 里，
  由 tool_registry.execute 负责“查表 → 角色 → validate → authorize → handler”五步编排。

★ handler 的契约（见 tool_registry.execute 第 5 步）：
    签名必须是 (day, *, session)，因为 execute 用 handler(**clean, session=session) 调；
    返回值必须是 ToolResult，而不是 list[BaseArticle] —— loop 只认 ToolResult。
"""
import json
import logging
import uuid
from datetime import datetime, timedelta, timezone
from email.utils import parsedate_to_datetime
from zoneinfo import ZoneInfo

import feedparser

from llm_types import ToolSpec
from tool_types import Session, Tool, ToolResult, ToolValidationError
from tools.Search_for_articles.base import BaseArticle

log = logging.getLogger("tool.huggingface_rss")

# ★ 常量集中放这里，便于测试时替换 / mock
SOURCE = "huggingface"
SOURCE_TYPE = "rss"
RSS_URL = "https://huggingface.co/blog/feed.xml"

DAY_MIN = 1
DAY_MAX = 7
DAY_DEFAULT = 1


HUGGINGFACE_RSS_SPEC = ToolSpec(
    name="huggingface_rss",
    description=(
        "查询 Hugging Face Blog 的最新文章。"
        "根据指定的天数，返回最近一段时间内发布的文章。"
        "文章包含标题、链接、发布时间、摘要等基本信息。\n\n"
        "使用场景："
        "查询 Hugging Face 最近发布了哪些文章、"
        "获取最近 1 天或几天的 AI 相关动态、"
        "获取 Hugging Face Blog 的最新资讯。\n\n"
        "注意："
        "本工具只能查询 Hugging Face Blog RSS 中已经发布的文章，"
        "不负责对文章内容进行详细分析。"
        "如果需要了解文章的完整内容，应进一步访问文章链接。"
    ),
    parameters={
        "type": "object",
        "properties": {
            "day": {
                "type": "integer",
                "minimum": DAY_MIN,
                "maximum": DAY_MAX,
                "default": DAY_DEFAULT,
                "description": (
                    "查询最近多少天发布的文章。"
                    "例如 day=1 表示查询最近 24 小时内发布的文章，"
                    "day=3 表示查询最近 3 天内发布的文章，"
                    "day=7 表示查询最近 7 天内发布的文章。"
                ),
            },
        },
        "required": [],
        "additionalProperties": False,
    },
)


# ---------------------------------------------------------------------------
# validate：参数级校验（tool_registry.execute 第 3 步调用）
# ---------------------------------------------------------------------------
def _validate(args: dict) -> dict:
    """把模型传来的 args 规范化成 handler 能直接吃的 clean dict。

    ★ 只做“强制转换 + 边界拒绝”，不做业务逻辑。
    ★ 任何非法都抛 ToolValidationError —— execute 会把它转成 ToolResult，
      不会冒到 agent loop。retryable=False：同样的参数重试结果不变。
    """
    day = args.get("day", DAY_DEFAULT)

    # ---- coerce：模型经常把整数传成字符串 "3" ----
    try:
        day = int(day)
    except (TypeError, ValueError):
        raise ToolValidationError(
            "INVALID_ARGUMENT",
            f"参数 day 必须是整数，收到 {args.get('day')!r}",
            f"请把 day 设为 {DAY_MIN}~{DAY_MAX} 之间的整数后重试。",
        ) from None

    # ---- reject：越界 ----
    if not (DAY_MIN <= day <= DAY_MAX):
        raise ToolValidationError(
            "INVALID_ARGUMENT",
            f"参数 day={day} 超出范围",
            f"day 只能取 {DAY_MIN}~{DAY_MAX} 之间的整数。请修正后重试。",
        )

    return {"day": day}


# ---------------------------------------------------------------------------
# 数据抓取：纯函数，返回 list[BaseArticle]，不关心 ToolResult
# ---------------------------------------------------------------------------
def _parse_published(entry) -> datetime | None:
    """解析单条 entry 的发布时间，统一转成 UTC。

    ★ 解析失败返回 None（跳过这条），而不是抛异常整批失败。
    ★ RSS 的时间可能没有时区信息，此时按 UTC 处理，避免用本机时区误判。
    """
    raw = entry.get("published") or entry.get("updated")
    if not raw:
        return None
    try:
        dt = parsedate_to_datetime(raw)
    except (TypeError, ValueError):
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _fetch_articles(day: int) -> list[BaseArticle]:
    """抓取最近 day 天内发布的 HF Blog 文章。"""
    feed = feedparser.parse(RSS_URL)

    # ★ feedparser 网络失败不抛异常，只置 bozo + 空 entries。
    #   必须在这里区分“抓不到”和“抓到但没新文章”，否则会把失败误报成 ok。
    if feed.bozo and not feed.entries:
        raise RuntimeError(feed.bozo_exception or "RSS 解析失败且无任何条目")

    now_utc = datetime.now(ZoneInfo("Asia/Shanghai")).astimezone(timezone.utc)
    cutoff = now_utc - timedelta(days=day)

    articles: list[BaseArticle] = []
    for entry in feed.entries:
        published_at = _parse_published(entry)
        # ★ 解析不出时间 or 早于 cutoff：跳过（continue 而非 break）。
        #   feed 未必严格倒序，break 会漏掉后面更新的文章。
        if published_at is None or published_at < cutoff:
            continue

        articles.append(BaseArticle(
            id=uuid.uuid4().hex,
            title=entry.get("title", "").strip(),
            url=entry.get("link", ""),
            source=SOURCE,
            source_type=SOURCE_TYPE,
            published_at=published_at,
            first_seen_at=now_utc,
            description=entry.get("description", ""),
            raw_data=dict(entry),
        ))

    return articles


def _serialize(articles: list[BaseArticle]) -> list[dict]:
    """BaseArticle → 给模型看的精简 dict。

    ★ 丢掉 raw_data：里面是整篇 HTML，塞进上下文既费 token 又干扰模型。
    """
    return [
        {
            "title": a.title,
            "url": a.url,
            "source": a.source,
            "source_type": a.source_type,
            "published_at": a.published_at.isoformat(),
            "description": a.description,
        }
        for a in articles
    ]


# ---------------------------------------------------------------------------
# handler：tool_registry.execute 第 5 步调用
# ---------------------------------------------------------------------------
def _handle(day: int, *, session: Session) -> ToolResult:
    """执行一次查询，返回 ToolResult。

    ★ 签名 (day, *, session)：execute 用 handler(**clean, session=session) 调。
    ★ 网络 / 解析异常在这里兜成 ToolResult，绝不让异常冒到 loop。
      （execute 也有兜底，但工具自己兜能给出更贴切的 error_code 和 hint。）
    """
    try:
        articles = _fetch_articles(day)
    except Exception as e:  # noqa: BLE001 —— 工具边界，必须全兜
        log.exception("huggingface_rss 抓取失败 day=%d", day)
        return ToolResult(
            ok=False,
            error_code="RSS_FETCH_ERROR",
            content=f"抓取 Hugging Face Blog RSS 失败：{e}",
            hint="网络或数据源暂时不可用，可稍后重试；若持续失败请告知坐席。",
            retryable=True,   # ★ 网络类错误重试有意义
        )

    if not articles:
        # ★ 空结果是“成功但没数据”，不是失败 —— 用 ok=True，避免模型误判后重试
        return ToolResult(
            ok=True,
            content=json.dumps(
                {"day": day, "count": 0, "articles": []},
                ensure_ascii=False,
            ),
            hint=f"最近 {day} 天内 Hugging Face Blog 没有新文章。"
                 "可尝试增大 day（最大 7），或直接告知用户暂无更新。",
        )

    return ToolResult(
        ok=True,
        content=json.dumps(
            {
                "day": day,
                "count": len(articles),
                "articles": _serialize(articles),
            },
            ensure_ascii=False,
        ),
    )


# ---------------------------------------------------------------------------
# 对外注册对象
# ---------------------------------------------------------------------------
# ★ required_scopes 留空：这是公开资讯查询，任何角色都能用。
HUGGINGFACE_RSS_TOOL = Tool(
    spec=HUGGINGFACE_RSS_SPEC,
    handler=_handle,
    required_scopes=frozenset(),
    validate=_validate,
    authorize=None,
)

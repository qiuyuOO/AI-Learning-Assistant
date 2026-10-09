"""arXiv 多分类 RSS 聚合查询工具。

★ 对外只暴露：
    ARXIV_RSS_SPEC —— 喂给模型的 ToolSpec（build_tools 用）
    ARXIV_RSS_TOOL —— 完整的 Tool（tool_registry.ALL_TOOLS 注册用）

  一个工具聚合抓取全部 arXiv 分类并按 URL 去重，模型只需给 day。
  handler / validate / 各分类抓取函数都是实现细节，封装在 Tool 里。

★ handler 契约（见 tool_registry.execute 第 5 步）：
    签名 (day, *, session)，因为 execute 用 handler(**clean, session=session) 调；
    返回 ToolResult，而不是 list[BaseArticle] —— loop 只认 ToolResult。
"""
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from email.utils import parsedate_to_datetime

import feedparser
import json
import logging
import uuid
import requests
from llm_types import ToolSpec
from tool_types import Session, Tool, ToolResult, ToolValidationError
from tools.Search_for_articles.base import BaseArticle
from memory import get_default_store

log = logging.getLogger("tool.arxiv_rss")

SOURCE_TYPE = "rss"

# ★ 分类 → RSS 地址。新增分类只改这里。
ARXIV_FEEDS = {
    "AI": "https://rss.arxiv.org/rss/cs.AI",
    "LG": "https://rss.arxiv.org/rss/cs.LG",
    "CL": "https://rss.arxiv.org/rss/cs.CL",
    "MA": "https://rss.arxiv.org/rss/cs.MA",
    "IR": "https://rss.arxiv.org/rss/cs.IR",
    "ML": "https://rss.arxiv.org/rss/stat.ML",
    "CV": "https://rss.arxiv.org/rss/cs.CV",
}

DAY_MIN = 1
DAY_MAX = 7
DAY_DEFAULT = 1

# ---- Jev 二次筛选配置 ----
# ★ Jev 是 Tool 内部实现细节，模型（Agent）不需要知道它的存在，
#   因此这些配置不进 ARXIV_RSS_SPEC。换机器 / 换端口只改 JEV_URL。
JEV_URL = "http://127.0.0.1:8017/v1/systemone"
JEV_MODEL = "jevos-v4"   # 请求体 model 字段（实测服务端要求）
JEV_TOP_K = 8          # 筛选后最多返回多少篇
JEV_TIMEOUT = 30       # 单次 Jev 请求超时（秒）
JEV_MAX_WORKERS = 1    # ★ 实测本地 Jev 是串行推理：持续并发会 thrash
                       #   （60 条并发 132s 且间歇超时），串行反而稳且快（~0.5s/条）
JEV_SCORE_MAX = 40     # 最多给最近多少篇打分（约束 Jev 调用次数/时延）
JEV_DEGRADE_MAX = 20   # Jev 降级时最多回多少篇（防全量摘要撑爆上下文）

JEV_PROMPT_HEADER = """你是一个 AI 技术论文筛选器。

请根据论文标题和摘要，判断这篇 arXiv 论文对用户的价值。"""

# 无用户记忆时的默认关注方向（行为等同改造前）
JEV_DEFAULT_FOCUS = """用户主要关注：

1. AI Agent
2. LLM
3. RAG
4. MCP
5. Agent Engineering
6. LLM Engineering
7. Multi-Agent
8. AI Infra
9. AI 工程实践
10. 对 AI 学习、项目开发和实习有帮助的技术"""

# 评分标准对默认/个性化两种情况都通用，故抽出来复用
JEV_RUBRIC = """评分标准：

0.9 - 1.0：
高度相关，值得重点关注

0.7 - 0.89：
明显相关，有较高学习价值

0.5 - 0.69：
有一定相关性

0.3 - 0.49：
相关性较低

0.0 - 0.29：
基本无关

只根据标题和摘要判断。

不要因为论文来自热门机构就提高评分。

重点判断：
- 是否与上述用户关注方向相关
- 是否具有工程实践价值
- 是否能够帮助用户学习相关技术
- 是否具有较强的新颖性或实际应用价值"""


def _assemble_jev_prompt(focus: str) -> str:
    """把 头部 + 关注方向 + 评分标准 拼成完整的 Jev 提示词。"""
    return f"{JEV_PROMPT_HEADER}\n\n{focus}\n\n{JEV_RUBRIC}"


# ★ 默认提示词：无用户记忆时用，等价于改造前写死的那份
JEV_PROMPT = _assemble_jev_prompt(JEV_DEFAULT_FOCUS)

# 从记忆里取"关注方向"时的置信度门槛：低于它的不算稳定兴趣，不进提示词
JEV_MEMORY_MIN_CONFIDENCE = 0.5


ARXIV_RSS_SPEC = ToolSpec(
    name="arxiv_rss",
    description=(
        "查询 arXiv 上与你关注方向最相关的最新论文。"
        "工具会聚合抓取 cs.AI / cs.LG / cs.CL / cs.MA / cs.IR / stat.ML / cs.CV "
        "等多个分类的 RSS 并按链接去重，再从最近发布的论文中，"
        "按与用户关注方向（AI Agent、LLM、RAG、MCP、Multi-Agent、AI Infra、"
        "AI/LLM 工程实践，以及对 AI 学习、项目开发和实习有帮助的技术）"
        "的相关性排序，筛选出最值得关注的少量论文（最多约 8 篇）。\n\n"
        "返回的每篇论文包含标题、链接、发布时间、摘要等基本信息，"
        "结果数量有界、已按相关性排序，可直接用于向用户汇总近期学术动态。\n\n"
        "使用场景："
        "查询 arXiv 最近与 AI Agent / LLM / RAG / MCP 等方向相关的高价值论文、"
        "获取过去一天到过去一周(7天)最值得关注的学术动态、"
        "追踪某个方向的热门新论文。\n\n"
        "注意："
        "本工具只做相关性筛选与信息返回，不负责下载 PDF，也不做论文的深入解读。"
        "如需了解论文完整内容，应进一步访问返回的论文链接。"
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
                    "查询最近多少天发布的论文。"
                    "例如 day=1 表示最近 24 小时，day=3 表示最近 3 天，"
                    "day=7 表示最近 7 天。"
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
    """把模型传来的 args 规范化成 handler 能直接吃的 clean dict。"""
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
    """解析单条 entry 的发布时间，统一转 UTC；失败返回 None（跳过该条）。"""
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


def _fetch_one(source_name: str, rss_url: str, cutoff: datetime,
               now_utc: datetime) -> tuple[list[BaseArticle], bool]:
    """抓单个分类。返回 (文章列表, 该 feed 是否抓取成功)。

    ★ 单分类失败不拖垮整体：返回 ([], False)，由调用方决定是否全部失败。
    """
    try:
        feed = feedparser.parse(rss_url)
    except Exception:  # noqa: BLE001
        log.warning("arXiv feed 抓取异常 source=%s", source_name, exc_info=True)
        return [], False

    # feedparser 网络失败不抛异常，只置 bozo + 空 entries → 视为该 feed 失败
    if feed.bozo and not feed.entries:
        log.warning("arXiv feed 解析失败 source=%s: %s",
                    source_name, feed.bozo_exception)
        return [], False

    articles: list[BaseArticle] = []
    for entry in feed.entries:
        published_at = _parse_published(entry)
        # ★ 解析不出时间 or 早于 cutoff：跳过（continue 而非 break，feed 未必严格倒序）
        if published_at is None or published_at < cutoff:
            continue

        articles.append(BaseArticle(
            id=uuid.uuid4().hex,
            title=entry.get("title", "").strip(),
            url=entry.get("link", ""),
            source=f"arxiv_{source_name}",
            source_type=SOURCE_TYPE,
            published_at=published_at,
            first_seen_at=now_utc,
            description=entry.get("description", ""),
            raw_data=dict(entry),
        ))

    return articles, True


def _fetch_articles(day: int) -> list[BaseArticle]:
    """聚合抓取全部分类，按 URL 去重后返回。"""
    now_utc = datetime.now(timezone.utc)
    cutoff = now_utc - timedelta(days=day)

    collected: list[BaseArticle] = []
    ok_feeds = 0
    for source_name, rss_url in ARXIV_FEEDS.items():
        articles, ok = _fetch_one(source_name, rss_url, cutoff, now_utc)
        ok_feeds += int(ok)
        collected.extend(articles)

    # ★ 所有 feed 都抓不到才算失败；只要有一个成功就当正常（可能确实没新论文）
    if ok_feeds == 0:
        raise RuntimeError("全部 arXiv feed 抓取失败（网络或数据源不可用）")

    # 按 URL 去重（同一篇论文可能出现在多个分类里）
    seen: set[str] = set()
    unique: list[BaseArticle] = []
    for article in collected:
        if article.url and article.url in seen:
            continue
        seen.add(article.url)
        unique.append(article)

    return unique

# ---------------------------------------------------------------------------
# Jev 二次筛选层：位于 _fetch_articles() 与 _serialize() 之间
#   _build_jev_items   —— 构造给 Jev 的精简输入
#   _post_jev/_parse   —— 与 Jev 通信 + 解析返回
#   _jev_filter_articles —— 排序 + Top K + 失败降级
# ---------------------------------------------------------------------------
def _focus_from_memory(memory) -> str | None:
    """把 UserMemory 里的兴趣/偏好/厌恶/决策标准渲染成 Jev 的"关注方向"段。

    ★ 只取稳定信号：兴趣要 status=active 且 confidence 达标；厌恶单独列出，
      让 Jev 对这些方向显著降分。返回 None 表示这个用户还没有可用画像。
    """
    lines: list[str] = []

    interests = [str(i.value) for i in memory.interests
                 if getattr(i, "status", "active") == "active"
                 and i.confidence >= JEV_MEMORY_MIN_CONFIDENCE and i.value]
    if interests:
        lines.append("用户长期关注方向：" + "、".join(dict.fromkeys(interests)))

    likes = [p.subject for p in memory.preferences if p.subject and p.preference > 0]
    if likes:
        lines.append("用户偏好：" + "、".join(likes))

    dislikes = [str(d.value) for d in memory.dislikes
                if getattr(d, "status", "active") == "active" and d.value]
    if dislikes:
        lines.append("用户不喜欢（命中应显著降低评分）：" + "、".join(dislikes))

    criteria = [c.name for c in
                sorted(memory.decision_criteria, key=lambda c: c.weight, reverse=True)
                if c.name]
    if criteria:
        lines.append("用户评价技术时看重（由重到轻）：" + " > ".join(criteria))

    if not lines:
        return None
    return ("请依据以下这位用户的长期画像来判断论文价值：\n"
            + "\n".join(lines))


def _build_jev_prompt(session: Session) -> str:
    """按 session 对应用户的记忆动态生成 Jev 提示词。

    ★ 记忆缺失 / 加载失败 / 画像为空 —— 一律回退到默认 JEV_PROMPT，
      绝不让记忆问题影响 arxiv 工具本身可用性。
    """
    try:
        memory = get_default_store().load(session.operator_id)
        focus = _focus_from_memory(memory)
    except Exception:  # noqa: BLE001 —— 记忆是增强项，失败必须能降级
        log.exception("加载用户记忆失败，Jev 提示词回退默认 operator_id=%s",
                      getattr(session, "operator_id", None))
        return JEV_PROMPT
    return _assemble_jev_prompt(focus) if focus else JEV_PROMPT


def _build_jev_items(articles: list[BaseArticle],
                     prompt: str = JEV_PROMPT) -> list[str]:
    """把“评判标准 + 论文关键信息”拼成 Jev 的 state 字符串（一篇一个）。

    ★ 实测结论：评判标准必须放进 state 才有区分度；questions 的 noul 规格
      是 extra_forbidden 的，标准塞不进 question 里。
    ★ prompt 由调用方按用户记忆动态生成后传入（默认回退到 JEV_PROMPT）。
    ★ 只给标题/摘要/时间，不塞 raw_data / RSS XML。
    """
    return [
        prompt
        + "\n\n论文标题：" + article.title
        + "\n摘要：" + (article.description or "")
        + ("\n发布时间：" + article.published_at.isoformat()
           if article.published_at else "")
        for article in articles
    ]


def _extract_score(answer) -> float | None:
    """从单个 answer 里取出分数。

    ★ Jev 实际返回形如：
        {"type": "noul", "noul": 0.7202...}
      即分数放在 answer["type"] 所指示的那个键下。这里按 type 取，
      取不到再兜底找第一个数值，避免硬编码 "noul"。
    """
    if not isinstance(answer, dict):
        return None
    key = answer.get("type")
    val = answer.get(key) if isinstance(key, str) else None
    if not isinstance(val, (int, float)) or isinstance(val, bool):
        for v in answer.values():
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                val = v
                break
    return float(val) if isinstance(val, (int, float)) else None


def _post_jev(state: str) -> dict:
    """对单个 state 发一次 Jev 请求，返回解析后的 JSON。

    ★ 请求体是实测探测出来的真实 schema：{state, model, questions}。
      方案里猜的 {prompt, items} 会被服务端 422 拒绝。
    ★ 网络/HTTP 错误向上抛（触发降级）。
    """
    response = requests.post(
        JEV_URL,
        json={
            "state": state,
            "model": JEV_MODEL,
            "questions": {"valuable": {"type": "noul"}},
        },
        timeout=JEV_TIMEOUT,
    )
    response.raise_for_status()
    return response.json()


def _score_one(state: str) -> float:
    """给单个 state 打分，返回 0.0~1.0 的分数。"""
    data = _post_jev(state)
    answer = (data.get("answers") or {}).get("valuable")
    score = _extract_score(answer)
    if score is None:
        raise ValueError(f"无法解析 Jev 评分返回: {data!r}")
    return score


def _call_jev(states: list[str]) -> list[float]:
    """返回与 states 等长的分数列表。

    ★ 该 API 一个请求只评一个 state（answers 只按问题名键控，无批量），
      所以用线程池并发逐条打分控制总时延。
    ★ 任何一条失败即抛异常 —— 由 _jev_filter_articles 接住降级。
    """
    with ThreadPoolExecutor(max_workers=JEV_MAX_WORKERS) as ex:
        return list(ex.map(_score_one, states))


def _most_recent(articles: list[BaseArticle], limit: int) -> list[BaseArticle]:
    """按发布时间倒序取前 limit 篇（降级兜底用，防全量摘要撑爆上下文）。"""
    epoch = datetime(1970, 1, 1, tzinfo=timezone.utc)
    return sorted(articles,
                  key=lambda a: a.published_at or epoch,
                  reverse=True)[:limit]


def _jev_filter_articles(articles: list[BaseArticle],
                         prompt: str = JEV_PROMPT) -> list[BaseArticle]:
    """Jev 评分 → 降序排序 → Top K。

    ★ prompt 是按用户记忆动态生成的评判标准，由 _handle 传入。
    ★ Jev 失败（挂了/超时/422/格式不对）不拖垮工具：降级返回“最近
      JEV_DEGRADE_MAX 篇”而不是全量 —— 全量上千篇完整摘要会把上下文
      撑到百万 token 级，直接 ContextLengthExceeded 崩掉 loop。
    """
    if not articles:
        return []

    # ★ 先按时间取最近 JEV_SCORE_MAX 篇再打分：该 API 无批量、只能逐条，
    #   不限制的话 day=7 要打两千多次 Jev，工具会卡好几分钟。
    candidates = _most_recent(articles, JEV_SCORE_MAX)

    try:
        states = _build_jev_items(candidates, prompt)
        scores = _call_jev(states)
    except Exception:  # noqa: BLE001 —— Jev 是增强项，失败必须能降级
        log.exception("Jev 筛选失败，降级为最近 %d 篇原始 RSS", JEV_DEGRADE_MAX)
        return _most_recent(articles, JEV_DEGRADE_MAX)

    scored = sorted(zip(scores, candidates), key=lambda x: x[0], reverse=True)
    return [article for _, article in scored[:JEV_TOP_K]]


def _serialize(articles: list[BaseArticle]) -> list[dict]:
    """BaseArticle → 给模型看的精简 dict（丢掉 raw_data 里的整篇摘要 HTML，省 token）。"""
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
    """执行一次查询，返回 ToolResult。异常在此兜住，绝不冒到 agent loop。"""
    try:
        articles = _fetch_articles(day)
        # ★ 按当前用户的长期记忆动态生成 Jev 评判标准（无记忆则回退默认），
        #   让"值得看"因人而异。记忆加载失败也在 _build_jev_prompt 内降级。
        prompt = _build_jev_prompt(session)
        # Jev 二次筛选（Jev 自身失败会在 _jev_filter_articles 内降级为原始 RSS）
        articles = _jev_filter_articles(articles, prompt)
    except Exception as e:  # noqa: BLE001 —— 工具边界，必须全兜
        log.exception("arxiv_rss 抓取失败 day=%d", day)
        return ToolResult(
            ok=False,
            error_code="RSS_FETCH_ERROR",
            content=f"抓取 arXiv RSS 失败：{e}",
            hint="网络或数据源暂时不可用，可稍后重试；若持续失败请告知坐席。",
            retryable=True,   # ★ 网络类错误重试有意义
        )

    if not articles:
        # ★ 空结果是“成功但没数据”，不是失败 —— ok=True，避免模型误判后重试
        return ToolResult(
            ok=True,
            content=json.dumps(
                {"day": day, "count": 0, "articles": []},
                ensure_ascii=False,
            ),
            hint=f"最近 {day} 天内 arXiv 相关分类没有新论文。"
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
# ★ required_scopes 留空：公开资讯查询，任何角色都能用。
ARXIV_RSS_TOOL = Tool(
    spec=ARXIV_RSS_SPEC,
    handler=_handle,
    required_scopes=frozenset(),
    validate=_validate,
    authorize=None,
)

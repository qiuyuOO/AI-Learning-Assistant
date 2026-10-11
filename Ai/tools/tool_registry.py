"""工具注册与执行编排。

★ 这个文件是 agent loop 和散装工具函数之间唯一的接口。
  loop 只认三个东西：build_tools / execute / ToolResult。
"""
import logging
import os
from typing import Any

from llm_types import ToolSpec
from tool_types import (Session, Tool, ToolResult,
                                          ToolValidationError)
from tools.Search_for_articles.huggingface_rss import HUGGINGFACE_RSS_TOOL
from tools.Search_for_articles.openai_rss import OPENAI_RSS_TOOL
from tools.Search_for_articles.arxiv import ARXIV_RSS_TOOL
from tools.remember_preference import REMEMBER_PREFERENCE_TOOL
from MCP.mcpclient import MCPClient

log = logging.getLogger("tool_registry")


# ★ 用 list 不用 dict：顺序稳定 → 喂给模型的 tools schema 顺序固定
#   → prompt 前缀稳定 → 缓存命中率高
# ★ 新工具追加到末尾：保持既有前缀不变，最大化 prompt 缓存命中
ALL_TOOLS: list[Tool] = [
    HUGGINGFACE_RSS_TOOL,
    OPENAI_RSS_TOOL,
    ARXIV_RSS_TOOL,
    REMEMBER_PREFERENCE_TOOL,
]
ZHIHU_MCP_URL = "https://developer.zhihu.com/api/mcp/v1"

# ★ 模块级持有 MCPClient：工具 handler 靠闭包 self 用它的连接发调用，
#   它的生命周期必须覆盖整个进程；shutdown 时也靠这个引用去 close。
_mcp_client: "MCPClient | None" = None


def register_mcp_tools(tools: list[Tool]) -> None:
    """把 MCP 工具追加到 ALL_TOOLS 末尾（低层原语）。"""
    ALL_TOOLS.extend(tools)     # ★ extend 不是 append；追加末尾保持本地工具前缀不变（缓存稳定）


def bootstrap_mcp(url: str = ZHIHU_MCP_URL, secret: str | None = None) -> int:
    """启动时调一次：建全局 MCPClient → connect → 把工具注册进 ALL_TOOLS。

    ★ 幂等：已初始化则直接返回 0，避免重复连接 / 重复 extend 出重复工具。
    ★ secret 缺省从环境变量 ZHIHU_API_KEY 读；调用方也可显式传。
    ★ 连接失败会向上抛异常 —— 由调用方（loop / server）决定"降级为仅本地工具"，
      本函数不吞异常，好让入口能记录真实原因。
    返回本次注册的工具数量。
    """
    global _mcp_client
    if _mcp_client is not None:
        return 0
    if secret is None:
        secret = os.getenv("ZHIHU_API_KEY")
    client = MCPClient()
    client.connect(url, secret)
    tools = client.get_tools()
    register_mcp_tools(tools)
    _mcp_client = client          # 留住引用：handler 要用它的连接，shutdown 要 close 它
    return len(tools)


def shutdown_mcp() -> None:
    """退出时调一次：关闭 MCP 连接与后台 loop，并释放引用。"""
    global _mcp_client
    if _mcp_client is not None:
        _mcp_client.close()
        _mcp_client = None

def build_tools(role: str) -> list[ToolSpec]:
    """返回该角色可见的工具声明，直接喂给 client.chat(tools=...)。

    ★ 返回 ToolSpec 而不是 Tool —— handler / validate 是模型不该看见的。
    ★ 无权限的工具不进 schema：模型看不见 → 省 token，也减少被注入诱导的面。
      但这不是权限底线 —— 底线在 execute 的第 2 步。
    """
    return [t.spec for t in ALL_TOOLS
            if not t.required_scopes or role in t.required_scopes]


def _fail(code: str, detail: str, hint: str) -> ToolResult:
    """构造失败结果。统一 retryable=False —— 见 execute 的说明。"""
    return ToolResult(ok=False, error_code=code, content=detail,
                      hint=hint, retryable=False)


def execute(name: str, args: dict[str, Any], session: Session) -> ToolResult:
    """执行一次工具调用。

    ★ 五步顺序不能换：
      1. 查表      —— 工具不存在和参数错是两回事，错误码必须分开
      2. 角色检查  —— 工具级权限（这个人能不能用这个工具）
      3. validate  —— 参数级（coerce / reject）
      4. authorize —— 数据级（这个数据在不在他的范围里）
      5. handler

    ★ 任何 ToolValidationError 都在这里转成 ToolResult，不冒到 agent loop。
      loop 只处理 ToolResult，不处理异常 —— 它的代码里不需要 try/except。
    """
    # ---- 1. 查表 ----
    tool = next((t for t in ALL_TOOLS if t.spec.name == name), None)
    if tool is None:
        available = ", ".join(t.spec.name for t in ALL_TOOLS)
        return _fail("TOOL_NOT_FOUND",
                     f"工具 {name!r} 不存在",
                     f"可用工具：{available}。请勿重试。")

    # ---- 2. 角色检查（工具级权限）----
    # ★ build_tools 里已经过滤过一次，这里必须再挡一次：
    #   模型看不见 ≠ 模型不能传。prompt injection 可以让它调没进 schema 的工具名。
    if tool.required_scopes and session.role not in tool.required_scopes:
        return _fail("TOOL_NOT_ALLOWED",
                     f"角色 {session.role!r} 无权使用 {name}",
                     "你当前的角色无权使用该工具。请勿重试，告知坐席。")

    # ---- 3 + 4. 参数校验 → 数据级鉴权 ----
    try:
        clean = tool.validate(args) if tool.validate else args
        if tool.authorize:
            # ★ 必须传 clean，不是 args：
            #   鉴权要拿规范化后的 user_id 和 session.scopes 比对
            tool.authorize(clean, session)
    except ToolValidationError as e:
        return ToolResult(ok=False, error_code=e.error_code, content=e.detail,
                          hint=e.hint, retryable=e.retryable)

    # ---- 5. 执行 ----
    try:
        # ★ handler 的签名是 (session, **clean)
        return tool.handler(**clean, session=session)
    except Exception:
        # ★ 兜底：一个工具挂掉不能让整个 agent loop 崩。
        #   log.exception 必须打 —— 否则堆栈丢了，线上无法定位。
        log.exception("工具 %s 执行异常", name)
        return _fail("TOOL_INTERNAL_ERROR",
                     "工具内部错误",
                     "工具内部错误，请勿重试，告知坐席已记录该问题。")

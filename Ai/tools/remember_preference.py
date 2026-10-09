"""显式记忆工具：把用户明确要求"记住"的偏好立刻写进长期记忆。

★ 为什么单独做一个工具，而不是靠 loop 的攒批 observe：
  observe 是"每 N 次对话后台悄悄更新一次"，用户说完偏好可能要聊很久才生效，
  对 arxiv 的 Jev 个性化也要等攒够批次。当用户明说"记住我喜欢 X"时，
  应当当场落盘、当场能确认 —— 这就是本工具存在的意义。

★ 对外只暴露两样东西（和其它工具一致的约定）：
    REMEMBER_PREFERENCE_SPEC —— 喂给模型的 ToolSpec（build_tools 用）
    REMEMBER_PREFERENCE_TOOL —— 完整的 Tool（tool_registry.ALL_TOOLS 注册用）

★ handler 契约（见 tool_registry.execute 第 5 步）：
    签名 (content, *, session)，因为 execute 用 handler(**clean, session=session) 调；
    返回 ToolResult。记忆写入走 memory.get_default_manager() 单例 —— 和 loop
    攒批写用的是同一个 manager / 同一把锁 / 同一个 store，不会互相覆盖。
"""
import json
import logging
import uuid

from llm_types import ToolSpec
from memory import Interaction, get_default_manager, now_iso
from tool_types import Session, Tool, ToolResult, ToolValidationError

log = logging.getLogger("tool.remember_preference")

# content 长度上限：偏好应是一句话级别的描述，挡住把整篇文章塞进来的误用
CONTENT_MAX = 500


REMEMBER_PREFERENCE_SPEC = ToolSpec(
    name="remember_preference",
    description=(
        "把当前用户明确要求记住的长期偏好写入其个人记忆，立即生效。"
        "适用于用户直接表达希望被长期记住的信息，例如："
        "『记住我只看 RAG 相关的论文』『以后别给我推纯理论的文章』"
        "『我评价方案时最看重落地成本』『我偏好中文回复』。\n\n"
        "使用场景："
        "用户说出稳定的兴趣 / 喜好 / 厌恶 / 决策标准 / 沟通偏好，"
        "并且期望它在以后的对话里持续生效时，调用本工具把它固化下来。\n\n"
        "注意："
        "只记『长期、可复用』的偏好，不要把一次性的临时问题当偏好记；"
        "content 用一句自然语言把偏好说清楚即可，系统会自动归类到"
        "兴趣 / 偏好 / 厌恶 / 决策标准等类别。"
        "本工具只写当前这位用户自己的记忆，不涉及他人。"
    ),
    parameters={
        "type": "object",
        "properties": {
            "content": {
                "type": "string",
                "description": (
                    "要长期记住的偏好，用一句自然语言描述。"
                    "例如：『只关注 RAG 与向量检索方向的论文』、"
                    "『不喜欢纯理论、无实验的内容』、"
                    "『评价技术方案时优先看工程落地成本』。"
                ),
            },
        },
        "required": ["content"],
        "additionalProperties": False,
    },
)


# ---------------------------------------------------------------------------
# validate：参数级校验（tool_registry.execute 第 3 步调用）
# ---------------------------------------------------------------------------
def _validate(args: dict) -> dict:
    """把模型传来的 args 规范化成 handler 能直接吃的 clean dict。

    ★ 只做"强制转换 + 边界拒绝"，不做业务判断。
    ★ 非法一律抛 ToolValidationError —— execute 会转成 ToolResult，不冒到 loop。
    """
    content = args.get("content")

    if content is None:
        raise ToolValidationError(
            "INVALID_ARGUMENT",
            "缺少参数 content",
            "请把要记住的偏好写进 content 字段后重试。",
        )

    # coerce：模型偶尔把内容包成非字符串（数字 / 列表），统一转成 str
    if not isinstance(content, str):
        content = str(content)
    content = content.strip()

    if not content:
        raise ToolValidationError(
            "INVALID_ARGUMENT",
            "参数 content 为空",
            "content 必须是非空文本，用来描述要记住的偏好。",
        )

    if len(content) > CONTENT_MAX:
        raise ToolValidationError(
            "INVALID_ARGUMENT",
            f"content 过长（{len(content)} > {CONTENT_MAX} 字符）",
            f"请把偏好精简成一句话（不超过 {CONTENT_MAX} 字符）后重试。",
        )

    return {"content": content}


# ---------------------------------------------------------------------------
# handler：tool_registry.execute 第 5 步调用
# ---------------------------------------------------------------------------
def _handle(content: str, *, session: Session) -> ToolResult:
    """把 content 作为一条"用户显式要求记住"的交互，立即写入长期记忆。

    ★ 复用 memory 的 Interaction + manager.remember：
      remember 内部会 take_pending 把已缓冲的对话一起结算并清空，
      然后同步跑一次 LLM 更新，返回 (ok, output)。
    ★ 任何异常都在这里兜成 ToolResult，绝不冒到 agent loop。
    """
    try:
        manager = get_default_manager()
        interaction = Interaction(
            interaction_id=uuid.uuid4().hex,
            session_id=session.session_id,   # 溯源：这条显式偏好来自哪一次会话
            timestamp=now_iso(),
            user_message=content,
            # agent_message 只是给记忆管理器一点上下文，标明这是显式指令
            agent_message="（用户显式要求长期记住以上偏好）",
        )
        ok, output = manager.remember(session.operator_id, interaction)
    except Exception as e:  # noqa: BLE001 —— 工具边界，必须全兜
        log.exception("remember_preference 写入失败 operator_id=%s",
                      session.operator_id)
        return ToolResult(
            ok=False,
            error_code="MEMORY_UPDATE_ERROR",
            content=f"写入用户记忆时发生异常：{e}",
            hint="记忆写入暂时不可用，可稍后重试；该偏好不会丢失。",
            retryable=True,
        )

    if not ok:
        # LLM 调用/解析失败：remember 已把这批回填缓冲，稍后会自动重试。
        # retryable=False —— 立刻重试大概率还是同样的失败，别让模型空转。
        return ToolResult(
            ok=False,
            error_code="MEMORY_UPDATE_ERROR",
            content="记忆更新未成功（模型调用或解析失败），该偏好已放回待处理缓冲，将在稍后自动重试。",
            hint="不要立即重复调用。可以告诉用户『已记录，稍后生效』。",
            retryable=False,
        )

    changes = [
        {"action": c.action, "memory_type": c.memory_type, "reason": c.reason}
        for c in output.changes
    ]
    return ToolResult(
        ok=True,
        content=json.dumps(
            {"remembered": content, "changes": changes},
            ensure_ascii=False,
        ),
        hint="已写入该用户的长期记忆。可简短向用户确认『已记住』，不必复述内部结构。",
    )


# ---------------------------------------------------------------------------
# 对外注册对象
# ---------------------------------------------------------------------------
# ★ required_scopes 留空：任何角色都能记自己的偏好。
# ★ authorize=None：handler 只写 session.operator_id 自己的记忆，
#   无法触及他人数据，故不需要数据级鉴权。
REMEMBER_PREFERENCE_TOOL = Tool(
    spec=REMEMBER_PREFERENCE_SPEC,
    handler=_handle,
    required_scopes=frozenset(),
    validate=_validate,
    authorize=None,
)

"""工具执行层的契约。

★ 这一层不允许出现任何业务语义。
  "客户"、"订单"、"坐席" 这些词只能出现在具体工具里。
  通用层用 scopes 这种不解释内容的容器承载权限信息。

与 llm_types 的分工：
  llm_types  —— 怎么跟模型通信（Message / Part / ToolSpec / LLMResponse）
  tool_types —— 怎么执行工具（Session / ToolResult / Tool / ToolValidationError）
两者正交：可以只用其中一个。
"""
from collections.abc import Callable
from dataclasses import dataclass

from llm_types import ToolSpec


@dataclass(frozen=True)
class Session:
    """调用者身份。

    ★ scopes 是通用容器，本层不解释其含义：
      客服场景 = 负责的客户 ID 集合
      知识库场景 = 可访问的部门 / 密级
      运维场景 = 可操作的主机分组
    由各工具的 authorize() 决定怎么用。

    ★ operator_id vs session_id（两个不同层级的身份，别混）：
      operator_id —— "谁在用"，用户级身份。长期记忆按它聚合，
                     本就该跨会话共享（同一用户不同对话共用一份记忆）。
      session_id  —— "这是哪一次对话"，会话级身份。用于区分同一用户的
                     不同会话：会话持久化按它分文件，记忆 Interaction 也带上它，
                     好让"某条偏好来自哪次对话"可溯源。默认空串（不参与会话管理时）。
    """
    operator_id: str
    role: str
    scopes: frozenset[str] = frozenset()
    session_id: str = ""


# tool_types.py
@dataclass
class ToolResult:
    ok: bool
    content: str = ""
    error_code: str | None = None
    hint: str | None = None
    retryable: bool = False

    @property
    def model_text(self) -> str:
        """给模型看的文本。★ hint 是指令，必须一起给模型 ——
        否则像 DUPLICATE_CALL 这种 content 为空的结果，
        模型只会收到一个空洞的失败然后原样重试。"""
        if self.content and self.hint:
            return f"{self.content}\n\n[HINT] {self.hint}"
        return self.content or self.hint or self.error_code or "（工具未返回内容）"


class ToolValidationError(Exception):
    def __init__(self, error_code: str, detail: str, hint: str,
                 retryable: bool = False):
        super().__init__(detail)
        self.error_code = error_code
        self.detail = detail
        self.hint = hint
        self.retryable = retryable


@dataclass
class Tool:
    spec: ToolSpec
    handler: Callable[..., ToolResult]
    required_scopes: frozenset[str] = frozenset()   # 空集 = 无需特定权限
    validate: Callable[[dict], dict] | None = None
    authorize: Callable[[dict, Session], None] | None = None
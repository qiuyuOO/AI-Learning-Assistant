from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable

class Role (str, Enum):
    TOOL = "tool"
    USER = "user"
    ASSISTANT="assistant"

@dataclass
class TextPart:
    text: str

@dataclass
class ToolSpec:
    name: str
    description: str
    parameters: dict[str, Any]        # JSON Schema

@dataclass
class ToolCallPart:

    id: str
    name: str
    arguments: dict[str, Any]


@dataclass
class ToolResultPart:

    tool_call_id: str
    content: str
    is_error: bool = False


@dataclass
class Message:
    role: Role
    parts: list[TextPart | ToolCallPart | ToolResultPart ]


class StopReason(str, Enum):
    END_TURN = "end_turn"  # 正常结束（模型认为回答完毕）
    MAX_TOKENS = "max_tokens"  # 达到 max_tokens 上限被截断
    STOP_SEQUENCE = "stop_sequence"  # 命中自定义 stop_sequences
    TOOL_USE = "tool_use"  # 模型要调用工具（tool_use 块）
    PAUSE_TURN = "pause_turn"  # 模型主动暂停（等待外部输入后继续）
    REFUSAL = "refusal"  # 因安全策略拒绝回答
    MODEL_CONTEXT_WINDOW_EXCEEDED = "model_context_window_exceeded"  # 上下文超限
    UNKNOWN = "unknown"


@dataclass
class LLMResponse:
    id : str
    model : str
    created_at: float  # unix 时间戳

    text: str
    content_blocks: list[dict] = field(default_factory=list) #取证用
    tool_calls: list[ToolCallPart] = field(default_factory=list)

    stop_reason: StopReason = StopReason.UNKNOWN

    input_tokens : int = 0
    output_tokens :int = 0
    cache_read_tokens : int =0
    cache_creation_tokens: int = 0

    service_tier: str = "standard"

    connect_ms: float = 0.0
    ttft_ms: float = 0.0
    total_ms: float = 0.0

@dataclass
class UsageRecord:
    # 调用用量记账类
    ts: float
    session_id: str
    provider: str
    model: str
    input_tokens: int
    output_tokens: int
    cache_read_tokens: int = 0
    cache_creation_tokens: int = 0
    service_tier: str = "standard"
    latency_ms: float = 0.0
    success: bool = True         # ★ 失败的调用也要记账（它可能已经计费了）

# ---- 辅助构造函数：把"role 与 part 的搭配规则"固化下来 ----
def user_message(text: str) -> Message:
    return Message(role=Role.USER, parts=[TextPart(text)])


def assistant_message(resp: LLMResponse) -> Message:
    """LLMResponse → canonical Message（回填历史用）。

    ★ 只提取 text / tool_use 两类 part。
      thinking block 不转换 —— 它留在 content_blocks 里作取证。
    """
    parts: list[Any] = []
    for block in resp.content_blocks:
        t = block.get("type")
        if t == "text":
            parts.append(TextPart(block["text"]))
        elif t == "tool_use":
            parts.append(ToolCallPart(id=block["id"], name=block["name"],
                                      arguments=block["input"]))
    return Message(role=Role.ASSISTANT, parts=parts)


def tool_message(tool_call_id: str, content: str, is_error: bool = False) -> Message:
    return Message(role=Role.TOOL,
                   parts=[ToolResultPart(tool_call_id, content, is_error)])




# 观测钩子：客户端每次调用后交出一条 UsageRecord
UsageSink = Callable[[UsageRecord], None]
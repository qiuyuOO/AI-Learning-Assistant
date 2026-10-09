from typing import Any, NoReturn
import logging
import time
import anthropic
from anthropic import Anthropic

from adapters.base import ProviderAdapter
from adapters.registry import register
from llm_types import LLMResponse,Message,StopReason, ToolSpec, Role, TextPart, ToolCallPart, ToolResultPart

from error_type import (
    AuthenticationFailed,
    BadRequest,
    ContextLengthExceeded,
    ModelUnavailable,
    RateLimited, _STATUS_TO_DOMAIN,
)

log = logging.getLogger("DeepseekAdapter")


def _to_domain_error(e: BaseException) ->NoReturn:
    """anthropic 异常 → 域异常。

    ★ 四段，顺序即语义：
      1. 查表      —— 所有带状态码的例外都在 _STATUS_TO_DOMAIN 里（429/401/403/413/402…）
      2. 超时      —— 必须在 3 之前：APITimeoutError 是 APIConnectionError 的子类
      3. 连接      —— 没有状态码
      4. 默认      —— 表没命中的状态码，按 5xx/4xx 分流
    ★ 每个分支都必须 from e —— 否则日志里只剩域异常，看不到根因。
    """
    # ---- 1. 表查（显式配置优先于通用规则）----
    code = getattr(e, "status_code", None)
    hit = _STATUS_TO_DOMAIN.get(code) if code is not None else None
    if hit is not None:
        exc_cls, msg, budget = hit
        raise exc_cls(msg, budget) from e

    # ---- 2. 超时 ----
    if isinstance(e, anthropic.APITimeoutError):
        raise ModelUnavailable() from e

    # ---- 3. 连接失败 ----
    if isinstance(e, anthropic.APIConnectionError):
        raise ModelUnavailable() from e

    # ---- 4. 默认：表没命中的状态码 ----
    # 用 status_code 而不是列举类名，SDK 新增 5xx 类时这里自动正确
    if isinstance(e, anthropic.APIStatusError):
        if code is None or code >= 500:
            raise ModelUnavailable() from e
        raise BadRequest() from e

    # ---- 兜底：APIError 是所有 SDK 错误的基类 ----
    # 含 APIResponseValidationError 这种没有 status_code 的
    if isinstance(e, anthropic.APIError):
        raise ModelUnavailable() from e

    # 不是 provider 异常 —— 调用方传错了，原样抛出
    raise e

@register
class DeepseekAdapter(ProviderAdapter) :
        name = "deepseek"
        def __init__(self, api_key: str, base_url: str | None = None, timeout=None):
                kwargs = {"api_key": api_key, "max_retries": 0}
                if base_url:
                        kwargs["base_url"] = base_url
                if timeout is not None:
                        kwargs["timeout"] = timeout
                self._client=Anthropic(**kwargs)

        def to_request(self,model : str, messages: list[Message], system: str | None = None,
                       tools: list[ToolSpec] | None = None,
                       max_tokens: int | None = None) -> dict:
                req: dict[str, Any] = {
                        "model": model,
                        "messages": self.to_native_messages(messages),
                        # ★ Anthropic 必填。这里只兜底，真正的默认值由 client 按窗口预算算
                        "max_tokens": max_tokens if max_tokens is not None else 1000000,
                }
                if system:
                        req["system"] = system  # ★ 顶层参数，不进 messages
                if tools:
                        req["tools"] = self.to_native_tools(tools)
                return req

        def to_native_messages(self, messages: list[Message]) -> list[dict]:
                """canonical messages → Anthropic native messages。

                三个必须处理的转换：
                1. ToolCallPart  → {"type": "tool_use", ...}
                2. ToolResultPart → 包进 {"role": "user", "content": [{"type": "tool_result"}]}
                                    ★ Anthropic 没有 tool 角色，工具结果伪装成 user
                3. 同一 role 的连续消息 → 合并成一条（Anthropic 要求 role 交替）
                """
                out: list[dict] = []
                for msg in messages:
                        native_role = "user" if msg.role in (Role.USER, Role.TOOL) else "assistant"
                        blocks = [self._part_to_block(p) for p in msg.parts]

                        if out and out[-1]["role"] == native_role:
                                # ★ 合并。不合并的话，"工具结果消息紧跟在用户消息后面"
                                #   会变成两条连续 user 消息 → Anthropic 直接 400
                                out[-1]["content"].extend(blocks)
                        else:
                                out.append({"role": native_role, "content": blocks})
                return out

        def _part_to_block(self, part) -> dict:
                if isinstance(part, TextPart):
                        return {"type": "text", "text": part.text}
                if isinstance(part, ToolCallPart):
                        return {"type": "tool_use", "id": part.id,
                                "name": part.name, "input": part.arguments}
                if isinstance(part, ToolResultPart):
                        block: dict[str, Any] = {
                                "type": "tool_result",
                                "tool_use_id": part.tool_call_id,  # ★ 靠它配对回 tool_use
                                "content": part.content,
                        }
                        if part.is_error:
                                block["is_error"] = True
                        return block
                raise TypeError(f"未知 part 类型: {type(part).__name__}")

        def to_native_tools(self, tools: list[ToolSpec]) -> list[dict]:
                """canonical ToolSpec → Anthropic tools 格式。

                ★ 差异点：Anthropic 叫 input_schema。
                  OpenAI 叫 parameters，且要多包一层 {"type":"function","function":{...}}
                """
                return [
                        {"name": t.name, "description": t.description, "input_schema": t.parameters}
                        for t in tools
                ]

        # ---------- 调用与响应侧 ----------

        def call(self, **native_kwargs) -> LLMResponse:
            """发一次请求，把 provider 异常归一化成域异常后向上抛。"""
            try:
                response = self._client.messages.create(**native_kwargs)
            except anthropic.APIError as e:
                raise _to_domain_error(e) from e
            return self.to_canonical(response)

        def to_canonical(self, raw) -> LLMResponse:
                blocks = [b.model_dump() for b in raw.content]

                text = "".join(b["text"] for b in blocks if b.get("type") == "text")
                tool_calls = [
                        ToolCallPart(id=b["id"], name=b["name"], arguments=b["input"])
                        for b in blocks if b.get("type") == "tool_use"
                ]

                raw_reason = raw.stop_reason

                try:
                        stop_reason = StopReason(raw_reason)
                except ValueError:
                        # 未知值 —— 记录原始值，归类 UNKNOWN，不抛异常
                        log.warning("未知 stop_reason: %r model=%s id=%s",
                                    raw_reason, raw.model, raw.id)
                        stop_reason = StopReason.UNKNOWN

                usage = raw.usage
                return LLMResponse(
                        id=raw.id,
                        model=raw.model,
                        created_at=time.time(),
                        text=text,
                        tool_calls=tool_calls,
                        content_blocks=blocks,  # 取证用，不回填
                        stop_reason=stop_reason,
                        input_tokens=getattr(usage, "input_tokens", 0) or 0,
                        output_tokens=getattr(usage, "output_tokens", 0) or 0,
                        cache_read_tokens=getattr(usage, "cache_read_input_tokens", 0) or 0,
                        cache_creation_tokens=getattr(usage, "cache_creation_input_tokens", 0) or 0,
                        service_tier=getattr(usage, "service_tier", "standard") or "standard",
                )
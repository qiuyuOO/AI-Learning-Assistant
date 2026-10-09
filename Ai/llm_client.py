import os
import json
import logging
import time
from pathlib import Path
from dataclasses import asdict, replace
from context import compress

from adapters.registry import get_adapter_class
from error_type import ContextLengthExceeded
from llm_types import (LLMResponse, TextPart, ToolCallPart,
                                         ToolResultPart, UsageRecord)
from model_specs import resolve_context_window
from retry import call_with_retry


_DEFAULT_USAGE_LOG = str(Path(__file__).parent / "usage_pending.jsonl")

log = logging.getLogger("llm_client")

HARD_CAP_MAX_TOKENS = 8_192      # 单次输出上限，防账单和延迟失控
SAFETY_MARGIN = 1_024            # 吸收 token 估算误差
MIN_OUTPUT_TOKENS = 256

# 字符 → token 的折算比例。中文为主的实测约 1.9 字符/token，取 1.6 偏保守
# （估出的 token 偏多）—— 符合"宁可少输出，也不要 400"的原则。
_CHARS_PER_TOKEN = 1.6


def _estimate_tokens(messages, system=None, tools=None) -> int:
    """粗略估算"这次请求的输入有多少 token"。

    ★ 有意不精确 —— 它只是首轮的兜底（第二轮起用上一轮的实测 usage）。
      不数的地方：Message 的结构开销、图片/多模态、嵌套 block 的额外括号。
      因为估算偏保守（宁可高估），漏数这些是可接受的。

    ★ 为什么必须数工具 schema：实测两个工具的 schema ≈ 896 token，
      占了小窗口的三分之一。漏了它就会严重低估。
    """
    chars = 0

    for m in messages:
        for p in m.parts:
            if isinstance(p, TextPart):
                chars += len(p.text)
            elif isinstance(p, ToolCallPart):
                # 参数发出去时是 JSON，按序列化后的长度数
                chars += len(json.dumps(p.arguments, ensure_ascii=False))
                chars += len(p.name) + len(p.id)
            elif isinstance(p, ToolResultPart):
                chars += len(p.content)

    if system:
        chars += len(system)

    for t in (tools or ()):
        chars += (len(t.name) + len(t.description)
                  + len(json.dumps(t.parameters, ensure_ascii=False)))

    return int(chars / _CHARS_PER_TOKEN)

class LLMClient:
    def __init__(self, provider: str, model: str, api_key: str,
                 context_window: int | None = None,
                 usage_sink=None,
                 usage_log_path: str = _DEFAULT_USAGE_LOG,
                 session_id: str = "default",
                 **adapter_config):
        """★ 配置分两组，按归属命名清楚 —— 避免 usage_sink 被误传给 adapter：

        client 自己用：model / context_window / usage_sink / session_id
        adapter 要用： api_key / **adapter_config（base_url / timeout ...）
        """
        self.provider = provider
        self.model = model
        self.context_window = resolve_context_window(model, context_window)
        self.session_id = session_id
        self.usage_sink = usage_sink
        self.usage_log_path = usage_log_path

        self._last_usage: LLMResponse | None = None
        self._usage_loss_count = 0
        self._pending_fallback = False      # 见 _fallback_write 的注释

        self.adapter = get_adapter_class(provider).create(
            api_key=api_key, **adapter_config)

    def _decide_max_tokens(self, messages, system=None, tools=None) -> int:
        # ★ 本地粗估：基于这次要发的真实内容。
        #   第二轮起和上一轮的实测 usage 取 max —— 两个都可能低估：
        #     本地估偏低（英文/代码多时比例不是 1.6）
        #     实测值偏低（历史在长，上一轮的输入比这一轮少）
        local = _estimate_tokens(messages, system, tools)
        measured = self._last_usage.input_tokens if self._last_usage else 0
        estimated_input = max(local, measured)

        available = self.context_window - estimated_input - SAFETY_MARGIN
        if available < MIN_OUTPUT_TOKENS:
            # ★ 报错带上"两个估算值"：排障时要能看出是哪个撑爆的
            #   （之前只打一个常量，导致查不到根因）
            raise ContextLengthExceeded(
                f"上下文剩余空间不足：window={self.context_window} "
                f"estimated_input={estimated_input}"
                f"（本地估 {local}，上轮实测 {measured}）")

        return min(HARD_CAP_MAX_TOKENS, available)

    def reset_usage_baseline(self) -> None:
        """压缩/裁剪上下文后调用。

        ★ _last_usage.input_tokens 是"上一次请求的输入"，它代表压缩前的上下文。
          上下文变了之后它就成了过期的估计，必须清掉，
          否则 _decide_max_tokens 会一直用旧的（更大的）那个值。
        """
        self._last_usage = None


    def _emit_usage(self, resp: LLMResponse) -> None:
        """记账。

        ★ 主流程已经成功，记账失败不允许影响返回值。
          但失败必须可见 —— 可以丢数据，不可以不知道丢了。
        """
        record = UsageRecord(
            ts=resp.created_at,
            session_id=self.session_id,
            provider=self.provider,
            model=resp.model,
            input_tokens=resp.input_tokens,
            output_tokens=resp.output_tokens,
            cache_read_tokens=resp.cache_read_tokens,
            cache_creation_tokens=resp.cache_creation_tokens,
            service_tier=resp.service_tier,
            latency_ms=resp.total_ms,
            success=True,
        )

        if self.usage_sink is not None:
            try:
                self.usage_sink(record)
                return
            except Exception:
                log.exception("usage_sink 失败，降级写本地文件")

        self._fallback_write(record)

    def _fallback_write(self, record: UsageRecord) -> None:
        """三级降级：文件 → ERROR 日志（含完整数据）→ 计数告警。"""
        payload = json.dumps(asdict(record), ensure_ascii=False)

        try:
            with open(self.usage_log_path, "a", encoding="utf-8") as f:
                f.write(payload + "\n")
            self._pending_fallback = True
            return
        except Exception:
            log.exception("usage 文件写入失败，降级到日志")

        # ★ 必须打完整内容 —— 只打一句"写入失败"等于把数据丢了
        log.error("USAGE_LOST %s", payload)

        self._usage_loss_count += 1
        if self._usage_loss_count % 100 == 1:
            log.critical("usage 记账已丢失 %s 条（监控告警入口）",
                         self._usage_loss_count)

    def _flush_pending(self) -> None:
        """补推 pending 记录。

        ★ 为什么用"重命名 + 只追未完成部分"，而不是"推完再清空"：
        "推送到 sink"和"清空文件"是两个独立操作，中间崩溃就会重复推送。
          重命名是原子的 —— 要么完全成功，要么完全没发生，没有中间态。

        投递语义：at-least-once（至少一次）。
          崩溃恢复后不会重复推已完成的记录；只有"推送中崩溃"的那一批会重推。
          绝对不重复需要接收端幂等，那是另一层的事。
    """
        if not self._pending_fallback or self.usage_sink is None:
            return

        inflight = self.usage_log_path + ".inflight"
        try:
            os.replace(self.usage_log_path, inflight)  # ★ 原子
        except FileNotFoundError:
            self._pending_fallback = False  # 没有待推记录
            return
        except OSError:
            log.exception("pending 文件重命名失败，保留待下次")
            return
        try:
            with open(inflight, "r", encoding="utf-8") as f:
                records = [json.loads(line) for line in f if line.strip()]
        except Exception:
            log.exception("读取 inflight 失败")
            return

        pushed = 0
        for r in records:
            try:
                self.usage_sink(UsageRecord(**r))
                pushed += 1
            except Exception:
                log.exception("补推第 %s 条失败，剩余 %s 条留待下次",
                              pushed, len(records) - pushed)
                break

        # 只把"没推成功的"写回去 —— 这就是避免重复的关键
        remaining = records[pushed:]
        if remaining:
            with open(self.usage_log_path, "a", encoding="utf-8") as f:
                for r in remaining:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")

        os.remove(inflight)
        self._pending_fallback = bool(remaining)

    def chat(self, messages, tools=None, system=None, max_tokens=None) -> LLMResponse:
        self._flush_pending()  # 上次失败留下的记录先补写

        # ① 发请求前算预算：本轮输入无法预知，只能拿上一轮实测值 + 余量
        if max_tokens is None:
            max_tokens = self._decide_max_tokens(messages, system, tools)

        def _do_call():
            native = self.adapter.to_request(
                self.model ,
                messages, system=system, tools=tools, max_tokens=max_tokens)
            # 异常从这里抛给 retry 层，client 不认识具体异常类型
            return self.adapter.call(**native)

        # ② 计时包住整个重试过程（所以 total_ms 里含退避等待时间）
        t0 = time.perf_counter()
        resp = call_with_retry(_do_call)
        resp.total_ms = (time.perf_counter() - t0) * 1000
        # ttft_ms 保持 0：非流式调用测不到首 token 时刻。
        #   0 表示"本次未测量"，不是"为零"。流式改造后才会填真值。

        # ③ 记账 + 为下一轮留基线
        self._emit_usage(resp)
        self._last_usage = resp

        return resp




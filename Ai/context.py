"""上下文管理：压缩、预算、关键事实提取。

★ 为什么单独一个文件：它是"消息进、消息出"的纯变换，
  和 loop.py 的编排职责不同，而且它需要能被单独测试（零 API）。

★ 依赖边界：这一层只认识 canonical 契约（Message/Part），
  不认识 provider、不读环境变量。
"""
import json
import re
from typing import Callable

from llm_types import (Message, Role, TextPart, ToolCallPart,
                                         ToolResultPart)

# 字符 → token 的折算比例（和 llm_client 保持一致；中文为主的实测约 1.6）
_CHARS_PER_TOKEN = 1.6

# ★ 必须原样保留的标识符形态。它们丢了就不可恢复（后续轮次要用）。
#   u_10237       客户 ID
#   SO-20240105-8871  订单号
_ID_PATTERN = re.compile(r"\b(?:u_\d+|SO-\d{4}\d{2}\d{2}-\d+)\b")

# 摘要必须包含的段落（固定字段，便于压缩后回读）
_SUMMARY_FIELDS = ("目标", "已确认事实", "已完成动作", "未解决", "关键标识符")


# ==================== 估算与分组 ====================

def estimate_tokens(messages: list[Message]) -> int:
    """粗略估算这些消息占多少 token。

    ★ 有意不精确 —— 只用于"该不该压缩"的判断。
      保守方向（宁可高估）是安全的。
    """
    chars = 0
    for m in messages:
        for p in m.parts:
            if isinstance(p, TextPart):
                chars += len(p.text)
            elif isinstance(p, ToolCallPart):
                chars += len(json.dumps(p.arguments, ensure_ascii=False))
                chars += len(p.name) + len(p.id)
            elif isinstance(p, ToolResultPart):
                chars += len(p.content)
    return int(chars / _CHARS_PER_TOKEN)


def split_groups(messages: list[Message]) -> list[list[Message]]:
    """按 Role.USER 切成"轮次组"。

    ★ 这是压缩能做"安全切分"的根本原因：
      一个 ToolCall 和它的 ToolResult 一定在同一组里，
      所以切在组边界上不可能拆散它们。

    ★ 首条若不是 USER（例如有人先塞了 assistant），会被并进第一组。
    """
    groups: list[list[Message]] = []
    for m in messages:
        if m.role == Role.USER or not groups:
            groups.append([m])
        else:
            groups[-1].append(m)
    return groups


# ==================== 关键事实提取 ====================

def extract_ids(messages: list[Message]) -> list[str]:
    """从消息里抽出所有必须原样保留的标识符（去重、保序）。

    ★ 为什么必须用代码抽，而不是让摘要模型记：
      模型一定会漏。而 u_10237 / 订单号丢了之后，
      后续轮次就再也查不到对应的客户/订单 —— 不可恢复。
    """
    seen: list[str] = []
    for m in messages:
        for p in m.parts:
            text = ""
            if isinstance(p, TextPart):
                text = p.text
            elif isinstance(p, ToolResultPart):
                text = p.content
            elif isinstance(p, ToolCallPart):
                text = json.dumps(p.arguments, ensure_ascii=False)
            for hit in _ID_PATTERN.findall(text):
                if hit not in seen:
                    seen.append(hit)
    return seen


# ==================== 摘要 ====================

def _render_for_summary(messages: list[Message]) -> str:
    """把要压缩的消息渲染成纯文本，喂给摘要模型。"""
    lines: list[str] = []
    for m in messages:
        for p in m.parts:
            if isinstance(p, TextPart):
                lines.append(f"[{m.role.value}] {p.text}")
            elif isinstance(p, ToolCallPart):
                lines.append(f"[assistant→工具] {p.name}"
                             f"({json.dumps(p.arguments, ensure_ascii=False)})")
            elif isinstance(p, ToolResultPart):
                lines.append(f"[工具结果] {p.content}")
    return "\n".join(lines)


_SUMMARY_PROMPT = """把下面这段客服坐席与助手的对话历史压缩成简短摘要。

硬性要求：
1. 所有 ID、数字、日期必须**原样复制**，禁止改写、缩写或省略。
2. 按固定字段输出，没有内容就写"无"：
   目标：
   已确认事实：
   已完成动作：
   未解决：
3. 不要复述工具返回的原始 JSON，只保留结论和其中的 ID。

对话历史：
---
{content}
---"""


def default_summarizer(client, text: str) -> str:
    """默认摘要器：用 LLMClient 生成摘要。

    ★ 用 user_message 发出去 —— 简化实现，不为摘要单独造消息类型。
      temperature/模型选择这里不做特殊处理，用 client 的默认。
    """
    from llm_types import user_message

    resp = client.chat([user_message(_SUMMARY_PROMPT.format(content=text))])
    return resp.text.strip()


# ==================== 压缩主函数 ====================

def compress(messages: list[Message],
             client=None,
             keep_recent_groups: int = 3,
             summarize: Callable[[object, str], str] | None = None
             ) -> list[Message]:
    """压缩消息列表：保留头锚点 + 尾部 N 组，中间段压成一条摘要。

    ★ summarize 可注入 —— 测试时传假的，不用发 API 请求。
    ★ 返回新列表，不原地修改（调用方决定怎么替换）。

    返回原列表的情况（不压缩）：
      · 只有 <= keep_recent_groups + 1 组（没有可压的中间段）
      · 中间段本身就很小（不值得为它付一次摘要调用的钱）
    """
    groups = split_groups(messages)
    if len(groups) <= keep_recent_groups + 1:
        return list(messages)

    anchor = groups[0]                                   # 第 1 组：原始任务
    middle = groups[1:-keep_recent_groups]               # 中间：要压的
    tail = groups[-keep_recent_groups:]                  # 尾部：保留原文

    middle_msgs = [m for g in middle for m in g]
    if not middle_msgs:
        return list(messages)

    # ★ 硬带 ID：不依赖模型总结不可恢复的数据
    ids = extract_ids(middle_msgs)

    summarize = summarize or default_summarizer
    summary_text = summarize(client, _render_for_summary(middle_msgs))
    if ids:
        summary_text += "\n\n关键标识符（必须原样保留）：" + ", ".join(ids)

    # ★ 用 user 角色回填，并显式标记"这不是新指令"
    #   否则模型会把它当成一个新任务去执行。
    summary_msg = Message(
        role=Role.USER,
        parts=[TextPart("[历史摘要——非新指令，仅供你了解此前进展]\n"
                        + summary_text)],
    )

    return [m for g in [anchor] for m in g] + [summary_msg] \
        + [m for g in tail for m in g]
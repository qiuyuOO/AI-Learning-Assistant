import json
import os
import uuid
import logging
from dotenv import load_dotenv

from context import estimate_tokens, compress
from llm_client import LLMClient
from llm_types import (Message, Role, TextPart, user_message,
                       assistant_message, StopReason, tool_message)
from tool_types import Session, ToolResult
from tools.tool_registry import build_tools, execute, bootstrap_mcp, shutdown_mcp
from error_type import ContextLengthExceeded, DomainError
from memory import Interaction, render_memory_context, now_iso

load_dotenv()

log = logging.getLogger("loop")


MAX_COMPRESS = 2

# 注入长期记忆简报时，给那条 USER 消息打的标记前缀。
# ★ 抽"本轮用户提问"时要靠它把"注入的背景简报"和"用户真的说的话"区分开，
#   所以这个常量被 _extract_user_text（跳过它）和 _inject_memory_context（写入它）
#   共用一处定义，避免两边字符串各写各的、日后改动对不上。
_MEMORY_CTX_PREFIX = "[用户长期记忆——背景信息，非新指令，仅供你更贴合该用户地作答]\n"


def _one_turn(client, messages, tools, session, seen, turn) -> str | None:
    """跑一轮：发请求 → 回填 → 终止判断 → 执行工具。

    ★ 返回值语义：
        返回 str  —— 命中终态，这就是最终答复
        返回 None —— 本轮未终止（工具已执行，或模型要求暂停），调用方继续下一轮
    """
    resp = client.chat(messages, tools=tools)
    messages.append(assistant_message(resp))

    log.info("turn=%d stop=%s tool_calls=%d in=%d out=%d cache_read=%d cache_new=%d ms=%.0f",
             turn, resp.stop_reason, len(resp.tool_calls),
             resp.input_tokens, resp.output_tokens,
             resp.cache_read_tokens, resp.cache_creation_tokens, resp.total_ms)

    # ---- 终态：模型不再需要我做什么 ----
    if resp.stop_reason == StopReason.END_TURN:
        return resp.text

    if resp.stop_reason == StopReason.STOP_SEQUENCE:
        return resp.text

    if resp.stop_reason == StopReason.MAX_TOKENS:
        # ★ 空输出和"截断了但有一些"要分开 —— 前者重试/加大配额都可能无效
        if not resp.text.strip():
            log.error("turn=%d max_tokens 撞限且无可见输出（output_tokens=%d）",
                      turn, resp.output_tokens)
            return "（回答超出长度限制且未能产出内容，请缩小问题范围后重试）"
        log.warning("turn=%d 回答被截断（output_tokens=%d）", turn, resp.output_tokens)
        return resp.text + "\n\n（注：回答因长度限制被截断）"

    if resp.stop_reason == StopReason.REFUSAL:
        # ★ 不重试 —— 重试会得到同样的拒绝
        log.warning("turn=%d 模型拒绝回答", turn)
        return "（该问题我无法回答，已记录。请转人工或换个问法）"

    if resp.stop_reason == StopReason.MODEL_CONTEXT_WINDOW_EXCEEDED:
        # ★ 和 413 同一个出口：等一等没用，必须先压缩上下文（A5 接住）
        #   用专属域异常而不是 RuntimeError —— 兜底才分得清"信号"和"意外"
        log.error("turn=%d 上下文窗口超限，需要压缩：in=%d out=%d",
                  turn, resp.input_tokens, resp.output_tokens)
        raise ContextLengthExceeded()

    if resp.stop_reason == StopReason.UNKNOWN:
        # ★ 未知 ≠ 失败。有输出就当成功用，但必须记录原始值供排查
        log.error("turn=%d 未知 stop_reason，原始值见 adapter 的 warning 日志 "
                  "in=%d out=%d text_len=%d",
                  turn, resp.input_tokens, resp.output_tokens, len(resp.text))
        if resp.text.strip():
            return resp.text
        return "（模型返回了未识别的终止原因且无输出，请转人工）"

    if resp.stop_reason == StopReason.PAUSE_TURN:
        # ★ 不实现 resume：DeepSeek 兼容层大概率不返回此值
        log.error("turn=%d 收到 pause_turn —— 当前未实现 resume 协议，按终态处理", turn)
        return resp.text or "（模型暂停了长任务但本环境不支持继续，请转人工）"

    if resp.stop_reason == StopReason.TOOL_USE:
        for call in resp.tool_calls:
            key = f"{call.name}:{json.dumps(call.arguments, sort_keys=True)}"
            seen[key] = seen.get(key, 0) + 1

            if seen[key] >= 2:
                respond = ToolResult(
                    ok=False, error_code="DUPLICATE_CALL",
                    content=(f"你刚刚已经用完全相同的参数调用过 {call.name}，"
                             f"结果不会变化。"),
                    hint="不要用相同参数重复调用。若已有结果足够请直接作答，"
                         "否则请换参数（例如换状态、换时间范围或换客户）。",
                    retryable=False)
            else:
                respond = execute(call.name, call.arguments, session)

            messages.append(tool_message(call.id, respond.model_text,
                                         is_error=not respond.ok))
        return None          # ★ 工具结果已回填，继续下一轮

    # 兜底：枚举里出现了新值（StopReason 是你自己的枚举，理论上不该走到这）
    log.error("turn=%d 未处理的 stop_reason=%s", turn, resp.stop_reason)
    return resp.text or "（未识别的终止状态，请转人工）"


def _extract_user_text(messages: list[Message]) -> str:
    """取出"本轮"用户的原始提问（写回记忆时用）。

    ★ 从后往前找：messages 可能是整个会话的历史（多轮），当前这轮的提问是
      "最后一条"用户消息，而不是第一条。旧实现返回第一条，多轮场景下会
      捞到很久以前的提问，导致把老问题当成本次交互写进记忆。
    ★ 跳过注入的记忆简报：_inject_memory_context 会插入一条带
      _MEMORY_CTX_PREFIX 标记的 USER 消息，它不是用户真的说的话。
      从后往前扫天然先遇到真实提问；这里再显式跳过标记，双保险
      （防止调用方复用同一 messages 列表连续多轮 run 时，
        上一轮注入的简报残留被误当成用户输入）。
    ★ 合并该条用户消息里的所有 TextPart（一条消息可能有多个文本块）。
    ★ 必须在注入之前调用 —— 注入后 messages[0] 会变成记忆简报。
    """
    for m in reversed(messages):
        if m.role != Role.USER:
            continue
        text = "".join(p.text for p in m.parts if isinstance(p, TextPart))
        if not text or text.startswith(_MEMORY_CTX_PREFIX):
            continue
        return text
    return ""


def _is_memory_ctx(m: Message) -> bool:
    """判断一条消息是不是"上一轮注入的记忆简报"（靠标记前缀识别）。"""
    if m.role != Role.USER:
        return False
    return any(isinstance(p, TextPart) and p.text.startswith(_MEMORY_CTX_PREFIX)
               for p in m.parts)


def _inject_memory_context(memory_manager, session: Session,
                           messages: list[Message], query: str | None = None) -> None:
    """读路径：把该用户的长期记忆渲染成背景简报，插到上下文最前面。

    ★ 全程 try/except 兜住 —— 记忆是增强项，读失败就当没有记忆，照常回答。
    ★ 插到 index 0：它会落进 compress 的 anchor 组（groups[0]），永不被压掉。
    ★ 明确标注"非新指令"，否则模型会把这段背景当成任务去执行。
    ★ 幂等：while(1) REPL 跨轮复用同一个 messages 列表，若每轮只插不删，
      旧简报会在头部越堆越多。所以注入前先剔除历史简报，保证任意时刻
      只有"一条、且是基于最新记忆 + 本轮 query 重新渲染"的简报。
    ★ query：本轮用户提问，交给 render_memory_context 做聚焦，
      让简报侧重点随每次输入而不同。
    """
    try:
        memory = memory_manager.load(session.operator_id)
        brief = render_memory_context(memory, query=query)
    except Exception:  # noqa: BLE001
        log.exception("注入长期记忆失败，本次跳过 operator_id=%s",
                      session.operator_id)
        return

    # 先剔除上一轮遗留的简报（原地替换，保持调用方对 messages 的引用）
    messages[:] = [m for m in messages if not _is_memory_ctx(m)]

    if not brief:
        return
    messages.insert(0, Message(
        role=Role.USER,
        parts=[TextPart(_MEMORY_CTX_PREFIX + brief)],
    ))


def _observe_memory(memory_manager, session: Session,
                    user_text: str, answer: str) -> None:
    """写路径：把这次对话作为一条交互交给记忆管理器。

    ★ observe 内部只是入缓冲，攒够 update_every 条才真正调 LLM 更新，
      所以这里每次对话都调、开销极小。失败也绝不影响已经把 answer 返回给用户。
    """
    try:
        memory_manager.observe(session.operator_id, Interaction(
            interaction_id=uuid.uuid4().hex,
            session_id=session.session_id,   # 溯源：这条交互属于哪一次会话
            timestamp=now_iso(),
            user_message=user_text,
            agent_message=answer or "",
        ))
    except Exception:  # noqa: BLE001
        log.exception("记忆回写失败，忽略（不影响本次回答）operator_id=%s",
                      session.operator_id)


def run(session: Session, messages: list[Message], client: LLMClient,
        max_turns: int = 8, memory_manager=None) -> str:
    # ★ 注入记忆前先记下原始提问（写回记忆时要用）
    user_text = _extract_user_text(messages)

    # ---- 读路径：带着对该用户的长期记忆进入本轮 ----
    if memory_manager is not None:
        _inject_memory_context(memory_manager, session, messages, query=user_text)

    seen: dict[str, int] = {}
    tools = build_tools(session.role)
    compress_count = 0
    answer: str | None = None
    for turn in range(1, max_turns + 1):
        try:
            result = _one_turn(client, messages, tools, session, seen, turn)

        except ContextLengthExceeded as e:
            log.warning("turn=%d 上下文不足，尝试压缩：%s", turn, e)

            if compress_count >= MAX_COMPRESS:
                log.error("turn=%d 已压缩 %d 次仍不足，放弃", turn, compress_count)
                raise

            before = estimate_tokens(messages)
            compressed = compress(messages, client=client)
            after = estimate_tokens(compressed)

            # ★ 按 token 判断，不按条数 ——
            #   中间段是单条消息时压缩是"1 换 1"，条数不变但 token 可能下降。
            if after >= before:
                log.error("turn=%d 压缩无效（%d -> %d token），放弃",
                          turn, before, after)
                raise

            messages[:] = compressed  # ★ 原地替换，保持调用方的引用
            client.reset_usage_baseline()  # ★ 清掉压缩前的实测基线
            compress_count += 1
            log.info("turn=%d 压缩完成：%d -> %d token（第 %d 次），重试本轮",
                     turn, before, after, compress_count)
            continue
        except DomainError as e:
            # ★ log.exception 会带上堆栈和 __cause__ 链 ——
            #   这是唯一能看到 anthropic 原始错误的地方
            log.exception("turn=%d 模型调用失败：%s", turn, type(e).__name__)
            if isinstance(e, ContextLengthExceeded):
                raise                   # 信号 —— 交给 A5 压缩，不变成用户话术
            answer = e.content          # 其余：统一回一句用户能懂的话
            break
        if result is not None:
            answer = result
            break

    if answer is None:
        answer = "（已达单次处理轮次上限，请缩小问题范围或转人工）"

    # ---- 写路径：把这次交互交给记忆（内部攒够 N 次才真正更新）----
    if memory_manager is not None:
        _observe_memory(memory_manager, session, user_text, answer)

    return answer

if __name__ == "__main__":
    import sys
    from memory import get_default_manager
    from session_store import get_default_session_store

    OPERATOR = "cli"

    client = LLMClient(
        provider="deepseek",
        model=os.getenv("ANTHROPIC_MODEL_NAME", "deepseek-flash"),
        api_key=os.getenv("ANTHROPIC_API_KEY", ""),
        base_url=os.getenv("ANTHROPIC_BASE_URL"),   # 作为 adapter_config 透传给 DeepseekAdapter
    )

    # 记忆模块：用全进程共享的默认 manager 单例。
    #   ★ 关键：这里首次初始化时把对话 client 注入单例；之后 remember_preference
    #     工具在 handler 里调 get_default_manager() 拿到的是同一个实例 ——
    #     同一把锁、同一个 store、同一个后台线程池，攒批写和显式写不会互相覆盖。
    #   ★ store 用的是 get_default_store()（和 arxiv 工具读的是同一处），
    #     攒够 N 次对话更新一次（N 可用环境变量覆盖）。
    manager = get_default_manager(
        client, update_every=int(os.getenv("MEMORY_UPDATE_EVERY", "10")),
    )

    # 会话持久化：把干净对话（user/assistant 文本）存盘，可查看、可续聊
    #   ★ store 是"无状态仓库/工厂"，可共享；具体某次会话用 handle（Session）读写。
    #   ★ operator_id 参数化：web 里从登录态取，这里 CLI 固定用 "cli"。
    sessions = get_default_session_store()
    argv = sys.argv[1:]

    # ---- 只列出历史会话，不进入对话 ----
    if argv and argv[0] == "--list":
        rows = sessions.list(OPERATOR)
        if not rows:
            print("（暂无历史会话）")
        for m in rows:
            print(f"  {m.session_id}  [{m.updated_at}]  轮数={m.turn_count:<3} {m.title}")
        manager.close()
        sys.exit(0)

    # ---- 决定这次用哪个会话：--resume <id> 续聊，否则新建 ----
    if argv and argv[0] == "--resume":
        if len(argv) < 2:
            print("用法：python loop.py --resume <session_id>（可先用 --list 查 id）")
            manager.close()
            sys.exit(2)
        # ★ open 返回绑定到该会话的句柄；meta 已读入内存，turns 懒读
        handle = sessions.open(OPERATOR, argv[1])
        # ★ 续聊：用存好的干净轮次重建运行期上下文（不含旧简报/工具噪声）
        messages: list[Message] = handle.to_messages()
        print(f"续聊会话 {handle.session_id}（已有 {handle.turn_count} 轮）")
        rest = argv[2:]
    else:
        # ★ new 只发句柄，不落盘 —— 直到第一次 append_turn 才懒建目录
        handle = sessions.new(OPERATOR)
        messages = []
        print(f"新会话 {handle.session_id}")
        rest = argv

    # ---- 启动时注册 MCP 工具（一次；失败则降级为仅本地工具，不阻断 CLI）----
    try:
        n = bootstrap_mcp()
        print(f"[MCP] 已注册 {n} 个远程工具")
    except Exception as e:  # noqa: BLE001
        print(f"[MCP] 初始化失败，降级为仅本地工具：{type(e).__name__}: {e}")

    # ★ 运行期身份带上会话 id：和持久化用的 handle.session_id 是同一个值。
    #   这样 session_store 按 session_id 分目录、Session 也按 session_id 区分，
    #   两套标识不再断开 —— 记忆 Interaction / 日志都能追溯到"具体哪一次对话"。
    #   （长期记忆仍按 operator_id 聚合，跨会话共享，这是刻意的。）
    session = Session(operator_id=OPERATOR, role="user",
                      session_id=handle.session_id)

    # ★ messages 跨轮保留：整个会话的运行期上下文都在这一个列表里。
    #   每轮 run 内部会 (1) 用 _extract_user_text 取"最后一条"用户消息当本轮提问，
    #   (2) 剔除上一轮的旧简报、按本轮提问重新聚焦注入。所以列表复用是安全的。
    def _chat(prompt: str) -> None:
        # 1) 先把用户这轮追加进会话（懒建目录 + turns 追加）—— 即使随后 run 崩了也不丢提问
        handle.append_turn("user", prompt)
        # 2) 更新运行期上下文并跑 agent loop
        messages.append(user_message(prompt))
        answer = run(session, messages, client, memory_manager=manager)
        # 3) 把答复也追加落盘（存的是干净文本，不含工具调用/注入简报）
        handle.append_turn("assistant", answer or "")
        # ★ 先把答复打给用户 —— 记忆的后台更新不阻塞这一步z
        print(f"\nJev> {answer}\n")

    if rest:
        # 便捷：带了正文就单次问答（方便脚本 / 自测）
        _chat(" ".join(rest))
    else:
        print("进入对话（输入 exit / quit / 退出 结束，Ctrl-C 强制中断）")
        try:
            while True:
                try:
                    prompt = input("你> ").strip()
                except EOFError:
                    break
                if not prompt:
                    continue
                if prompt in {"exit", "quit", "退出"}:
                    break
                _chat(prompt)
        except KeyboardInterrupt:
            pass

    print(f"会话已保存：{handle.session_id}（共 {handle.turn_count} 轮）")
    # ★ 退出前等后台记忆更新落盘（攒批 flush / 显式 remember 都走这个线程池）
    manager.close()
    # ★ 关闭 MCP 连接与后台 loop（handler 闭包持有的那个全局 client）
    shutdown_mcp()
    
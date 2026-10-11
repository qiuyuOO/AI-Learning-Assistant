"""MCP 客户端 / 桥接层（bridge）。

★ 这个类就是我们一直说的 "bridge"：把 async 的 MCP 世界关进一个常驻后台事件
  循环线程，对外只暴露【同步】方法（connect / call_tool / get_tools / close）。
  agent loop、tool_registry.execute、各 handler 全程同步，不必碰 async。

★ 关键架构：单常驻 task（_session_runner）
  anyio 的 cancel scope 要求"在哪个 task 进入，就必须在同一个 task 退出"。
  若像早期版本那样用 AsyncExitStack：connect 在 task A 里 enter，close 在
  task B 里 aclose，就会抛
      RuntimeError: Attempted to exit cancel scope in a different task ...
  所以这里改成：后台跑【一个】长驻 runner 协程，它 async with 打开传输层与
  ClientSession（进入 scope），然后在同一个 task 里循环处理命令队列；收到
  shutdown 时跳出循环，async with 在【同一个 task】里退出（退出 scope）——
  进/出同 task，问题解决。call_tool / list_tools 都通过队列投递给这个 runner，
  用 concurrent.futures.Future 把结果同步取回。

★ 单 server 版本：只持有一个 session（当前接知乎 MCP）。call_tool 只需
  remote_name + args，不需要 server 参数。多 server 时把 runner 扩成字典即可。

★ 命名区分：
    self._session —— MCP 的 ClientSession（到远程 server 的连接，仅 runner 内用）
    handler(*, session) 里的 session —— 调用者身份 tool_types.Session
"""
import os
import asyncio
import threading
from concurrent.futures import Future
from datetime import timedelta
from typing import Optional

from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client
from dotenv import load_dotenv

from llm_types import ToolSpec
from tool_types import Tool, ToolResult

load_dotenv()


# ---------------------------------------------------------------------------
# 模块级：把 MCP 返回的 content 块列表拍平成给模型看的字符串
# ★ 放模块级（不是类方法）：handler 里直接当普通函数调，避免 self 纠缠。
# ---------------------------------------------------------------------------
def _serialize_content(content) -> str:
    parts = []
    for block in content:
        t = getattr(block, "type", None)
        if t == "text":
            parts.append(block.text)                                   # 文本直接取
        elif t == "image":
            parts.append(f"[图片 {getattr(block, 'mimeType', '')}，二进制已省略]")  # base64 别塞给模型
        elif t == "resource":
            res = getattr(block, "resource", None)
            parts.append(getattr(res, "text", None) or f"[资源 {getattr(res, 'uri', '')}]")
        else:
            parts.append(f"[未识别内容块 type={t}]")
    return "\n\n".join(p for p in parts if p)


class MCPClient:
    def __init__(self):
        # ---- 常驻后台事件循环 ----
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self._loop.run_forever, daemon=True)
        self._thread.start()

        # ---- 命令队列 + runner 状态（都活在后台 loop 上）----
        self._cmd_q: Optional[asyncio.Queue] = None   # 命令队列，loop 线程操作
        self._ready: Optional[Future] = None          # 连接是否就绪（线程安全 Future）
        self._runner: Optional[Future] = None         # runner 协程的 future（run_coroutine_threadsafe 返回）
        self._session: Optional[ClientSession] = None  # 仅供判空，实际操作都在 runner task 内

    # ==================================================================
    # 连接：启动 runner 长驻 task，等它就绪
    # ==================================================================
    def connect(self, server_url: str, access_secret: str):
        if not access_secret:
            raise ValueError("未读取到知乎 Access Secret，请检查 .env 中的 ZHIHU_API_KEY")

        self._ready = Future()
        # asyncio.Queue 在 loop 线程上创建，避免跨线程绑错 loop
        self._cmd_q = asyncio.run_coroutine_threadsafe(
            self._make_queue(), self._loop).result(timeout=10)

        # 启动 runner —— 不 .result()（它长驻，直到 shutdown 才结束）
        self._runner = asyncio.run_coroutine_threadsafe(
            self._session_runner(server_url, access_secret), self._loop)

        # 阻塞等待"连接 + initialize 完成"；连接失败时这里会抛出异常
        self._ready.result(timeout=60)

    async def _make_queue(self) -> asyncio.Queue:
        return asyncio.Queue()

    async def _session_runner(self, server_url: str, access_secret: str):
        """长驻 task：打开连接 → 服务命令 → 收到 shutdown 后在同一 task 内干净退出。"""
        headers = {"Authorization": f"Bearer {access_secret}"}
        try:
            # ★ async with 的进入与退出都发生在本 task，满足 anyio cancel scope 约束
            async with streamablehttp_client(
                url=server_url,
                headers=headers,
                timeout=timedelta(seconds=30),
                sse_read_timeout=timedelta(seconds=300),
            ) as transport:
                read_stream, write_stream = transport[0], transport[1]
                async with ClientSession(read_stream, write_stream) as session:
                    await session.initialize()
                    self._session = session
                    self._ready.set_result(True)   # 通知 connect() 已就绪

                    # ---- 命令循环：所有 session 操作都在本 task 内 ----
                    while True:
                        cmd = await self._cmd_q.get()
                        kind = cmd[0]

                        if kind == "shutdown":
                            cmd[-1].set_result(True)   # 先回执，再跳出 → async with 退出
                            break

                        if kind == "call_tool":
                            _, name, args, fut = cmd
                            try:
                                fut.set_result(await session.call_tool(name, args))
                            except BaseException as e:   # noqa: BLE001
                                fut.set_exception(e)

                        elif kind == "list_tools":
                            _, fut = cmd
                            try:
                                fut.set_result(await session.list_tools())
                            except BaseException as e:   # noqa: BLE001
                                fut.set_exception(e)
        except BaseException as e:   # noqa: BLE001
            # 连接阶段失败：把异常交给 connect()；就绪后崩溃：留待后续 _submit 超时暴露
            if self._ready is not None and not self._ready.done():
                self._ready.set_exception(e)
        finally:
            self._session = None

    # ==================================================================
    # 同步投递命令给 runner，阻塞取结果
    # ==================================================================
    def _submit(self, cmd: tuple, timeout: float):
        fut = Future()
        item = cmd + (fut,)
        # asyncio.Queue 非线程安全 → 用 call_soon_threadsafe 在 loop 线程入队
        self._loop.call_soon_threadsafe(self._cmd_q.put_nowait, item)
        return fut.result(timeout=timeout)

    def call_tool(self, remote_name: str, args: dict):
        """同步调用远程工具，返回 mcp 的 CallToolResult。"""
        if self._session is None:
            raise RuntimeError("MCP 尚未连接，请先调用 connect()")
        return self._submit(("call_tool", remote_name, args), timeout=120)

    def _list_tools(self):
        if self._session is None:
            raise RuntimeError("MCP 尚未连接，请先调用 connect()")
        return self._submit(("list_tools",), timeout=60)

    # ==================================================================
    # 列出远程工具 → 翻译成本项目的 Tool（同步）
    # ==================================================================
    def get_tools(self) -> list[Tool]:
        response = self._list_tools()
        tools: list[Tool] = []
        for t in response.tools:
            spec = ToolSpec(
                name=f"mcp__zhihu__{t.name}",   # ★ 加前缀防冲突
                description=t.description,
                parameters=t.inputSchema,        # ★ inputSchema 本身就是 JSON Schema，直接用
            )
            tools.append(Tool(
                spec=spec,
                handler=self.make_mcp_handler(t.name),  # ★ 原始工具名被闭包记住
                required_scopes=frozenset(),
                validate=None,                   # 透传，参数校验交给远程 server（返回 isError）
                authorize=None,
            ))
        return tools

    # ==================================================================
    # handler 工厂：一个 MCP 工具 → 一个同步 handler
    # ==================================================================
    def make_mcp_handler(self, remote_name: str):
        def handler(*, session, **args):
            # ① 远程调用本身可能抛异常（超时/断连），必须兜住，绝不冒到 agent loop
            try:
                result = self.call_tool(remote_name, args)
            except Exception as e:  # noqa: BLE001 —— 工具边界，全兜
                return ToolResult(
                    ok=False, error_code="MCP_CALL_ERROR",
                    content=f"调用 MCP 工具 {remote_name} 失败：{e}",
                    hint="远程工具暂不可用，可稍后重试。",
                    retryable=True,
                )

            text = _serialize_content(result.content)

            # ② isError=True → 远程报告执行失败
            if result.isError:
                return ToolResult(
                    ok=False, error_code="MCP_TOOL_ERROR",
                    content=text or "MCP 工具执行失败",
                    hint="远程工具报告失败，请勿原样重试，可换参数或告知用户。",
                    retryable=False,
                )

            # ③ 成功
            return ToolResult(ok=True, content=text)

        return handler

    # ==================================================================
    # 清理：通知 runner 退出（在同一 task 内收尾）→ 停后台 loop
    # ==================================================================
    def close(self):
        try:
            if self._cmd_q is not None and self._runner is not None and not self._runner.done():
                self._submit(("shutdown",), timeout=30)
                self._runner.result(timeout=30)   # 等 runner 真正退出（async with 清理完成）
        except Exception:  # noqa: BLE001 —— 清理阶段不再抛，尽力而为
            pass
        finally:
            self._loop.call_soon_threadsafe(self._loop.stop)
            self._thread.join(timeout=5)



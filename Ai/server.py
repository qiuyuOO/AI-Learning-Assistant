"""FastAPI 后端：把已有的 CLI/库能力（loop.run + session_store + memory）暴露成 REST API，供前端调用。

★ 设计原则
  · 无状态服务：每个请求都从磁盘（session_store）重建对话上下文，服务进程本身
    不长期持有某个会话的运行期 messages。天然支持多会话切换、多标签页、重启不丢。
  · operator_id 参数化：web 里应从登录态取；当前无鉴权，用请求里的 operator_id
    字段（缺省 "web"）。session_store 会做白名单校验，非法值直接 400。
  · 阻塞调用交给线程池：run() 会同步调 LLM，端点用普通 def（非 async），
    FastAPI 自动把它放进线程池，不会卡住事件循环。session_store 每会话一把锁，
    并发写同一会话安全。
  · 只存干净对话：和 CLI 一致，落盘的是 user/assistant 文本；记忆简报由 run()
    每轮临时注入到运行期 messages，不进 turns.jsonl。

★ 运行（在 Ai/ 目录下）：
    uvicorn server:app --reload --port 8000
  接口文档：http://127.0.0.1:8000/docs
"""
from __future__ import annotations

import logging
import os

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from llm_client import LLMClient
from loop import run
from memory import (Interaction, get_default_manager, now_iso,
                    render_memory_context)
from session_store import get_default_session_store
from tool_types import Session

load_dotenv()

log = logging.getLogger("server")
logging.basicConfig(level=logging.INFO)

# ===========================================================================
# 进程级单例：对话 client / 记忆 manager / 会话仓库。
#   ★ manager 用 get_default_manager 单例 —— 和 loop、remember_preference 工具
#     共享同一把锁/store/后台线程池，攒批写和显式写不会互相覆盖。
#   ★ store 用 get_default_session_store 单例（无状态工厂，可共享）。
# ===========================================================================
DEFAULT_OPERATOR = os.getenv("WEB_OPERATOR_ID", "web")

_client = LLMClient(
    provider="deepseek",
    model=os.getenv("ANTHROPIC_MODEL_NAME", "deepseek-flash"),
    api_key=os.getenv("ANTHROPIC_API_KEY", ""),
    base_url=os.getenv("ANTHROPIC_BASE_URL"),
)
_manager = get_default_manager(
    _client, update_every=int(os.getenv("MEMORY_UPDATE_EVERY", "10")),
)
_store = get_default_session_store()

app = FastAPI(title="Agent API", version="1.0.0")

# 允许 Vite 开发服务器跨域访问（默认 5173，可用环境变量追加）
_CORS_ORIGINS = [o.strip() for o in os.getenv(
    "WEB_CORS_ORIGINS",
    "http://localhost:5173,http://127.0.0.1:5173",
).split(",") if o.strip()]
app.add_middleware(
    CORSMiddleware,
    allow_origins=_CORS_ORIGINS,
    allow_origin_regex=r"http://(localhost|127\.0\.0\.1):\d+",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# 请求体模型
# ---------------------------------------------------------------------------
class NewSessionIn(BaseModel):
    operator_id: str = DEFAULT_OPERATOR


class ChatIn(BaseModel):
    operator_id: str = DEFAULT_OPERATOR
    text: str = Field(default="")


class RememberIn(BaseModel):
    operator_id: str = DEFAULT_OPERATOR
    content: str = Field(default="")


# ---------------------------------------------------------------------------
# 小工具
# ---------------------------------------------------------------------------
def _meta_dict(handle) -> dict:
    m = handle.meta
    return {
        "session_id": m.session_id,
        "operator_id": m.operator_id,
        "title": m.title,
        "created_at": m.created_at,
        "updated_at": m.updated_at,
        "turn_count": m.turn_count,
    }


def _turns_list(handle) -> list[dict]:
    return [{"role": t.role, "text": t.text, "ts": t.ts} for t in handle.turns()]


# ---------------------------------------------------------------------------
# 端点
# ---------------------------------------------------------------------------
@app.get("/api/health")
def health() -> dict:
    return {"ok": True, "default_operator": DEFAULT_OPERATOR,
            "model": os.getenv("ANTHROPIC_MODEL_NAME", "deepseek-flash")}


@app.get("/api/sessions")
def list_sessions(operator_id: str = DEFAULT_OPERATOR) -> dict:
    """列出某用户所有会话（只读 meta.json，按更新时间倒序）。"""
    try:
        metas = _store.list(operator_id)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    return {"operator_id": operator_id,
            "sessions": [m.model_dump(mode="json") for m in metas]}


@app.post("/api/sessions")
def new_session(body: NewSessionIn) -> dict:
    """新建一个会话：只发句柄、返回 id，不落盘（等第一条消息）。"""
    try:
        handle = _store.new(body.operator_id)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    return _meta_dict(handle)


@app.get("/api/sessions/{session_id}")
def get_session(session_id: str, operator_id: str = DEFAULT_OPERATOR) -> dict:
    """读一个会话：元信息 + 全部干净轮次（用于前端渲染历史）。"""
    try:
        handle = _store.open(operator_id, session_id)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    return {"meta": _meta_dict(handle), "turns": _turns_list(handle)}


@app.post("/api/sessions/{session_id}/messages")
def post_message(session_id: str, body: ChatIn) -> dict:
    """发一条消息：追加用户轮 → 跑 agent loop → 追加助手轮 → 返回答复。

    ★ 用户轮先落盘（即使随后 run 崩了也不丢提问），和 CLI 的 _chat 一致。
    ★ messages 从磁盘干净轮次重建；run() 内部按最新记忆 + 本轮提问注入简报。
    """
    text = (body.text or "").strip()
    if not text:
        raise HTTPException(status_code=400, detail="消息不能为空")

    try:
        handle = _store.open(body.operator_id, session_id)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e

    handle.append_turn("user", text)

    session = Session(operator_id=body.operator_id, role="user",
                      session_id=session_id)
    messages = handle.to_messages()   # 含刚追加的这条用户消息
    try:
        answer = run(session, messages, _client, memory_manager=_manager)
    except Exception as e:  # noqa: BLE001 —— 不让单次失败拖垮服务
        log.exception("run 失败 operator=%s session=%s",
                      body.operator_id, session_id)
        raise HTTPException(status_code=500, detail=f"agent 处理失败：{e}") from e

    handle.append_turn("assistant", answer or "")
    return {"answer": answer or "", "meta": _meta_dict(handle),
            "turns": _turns_list(handle)}


@app.get("/api/memory")
def get_memory(operator_id: str = DEFAULT_OPERATOR) -> dict:
    """读某用户的长期记忆：结构化画像 + 渲染后的简报文本（供面板展示）。"""
    try:
        memory = _manager.load(operator_id)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    brief = render_memory_context(memory, query=None) or ""
    return {
        "operator_id": operator_id,
        "memory": memory.model_dump(mode="json"),
        "brief": brief,
        "updated_at": memory.updated_at,
    }


@app.post("/api/memory/remember")
def remember(body: RememberIn) -> dict:
    """显式记住一条偏好：立即同步更新长期记忆（等价 remember_preference 工具）。"""
    content = (body.content or "").strip()
    if not content:
        raise HTTPException(status_code=400, detail="content 不能为空")
    if len(content) > 500:
        raise HTTPException(status_code=400, detail="content 过长（<=500 字）")

    interaction = Interaction(
        interaction_id=os.urandom(16).hex(),
        session_id="",
        timestamp=now_iso(),
        user_message=content,
        agent_message="（用户显式要求长期记住以上偏好）",
    )
    try:
        ok, output = _manager.remember(body.operator_id, interaction)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except Exception as e:  # noqa: BLE001
        log.exception("remember 失败 operator=%s", body.operator_id)
        raise HTTPException(status_code=500, detail=f"记忆更新失败：{e}") from e

    if not ok:
        return {"ok": False, "changes": [],
                "message": "记忆更新未成功，已回填缓冲待下次重试"}
    return {
        "ok": True,
        "changes": [c.model_dump(mode="json") for c in output.changes],
        "message": f"已记住，共 {len(output.changes)} 条变更",
    }


@app.on_event("shutdown")
def _shutdown() -> None:
    """进程退出前，等后台记忆更新落盘。"""
    try:
        _manager.close()
    except Exception:  # noqa: BLE001
        log.exception("关闭记忆 manager 时出错（忽略）")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="127.0.0.1",
                port=int(os.getenv("WEB_PORT", "8000")), reload=False)

"""会话持久化：把一次对话（若干轮 user/assistant 文本）存到磁盘，供后续查看与续聊。

★ 只存"干净对话"：真实的用户提问 + 助手答复文本。
  不存工具调用/工具结果/注入的记忆简报 —— 那些是运行期噪声：
    · 记忆简报每轮会按最新记忆重新渲染注入，存下来反而会过期；
    · 工具调用细节对"回看对话"没价值，还会让文件又大又乱。
  续聊时用 to_messages() 把干净轮次重建成 list[Message]，
  run() 会在此基础上重新注入当轮简报 —— 复原的是"对话语义"，不是"字节级快照"。

★ 组织方式：一个会话一个"目录"，按 operator_id 再分一层：
    Ai/data/sessions/{operator_id}/{session_id}/
        ├── meta.json     会话元信息（id/标题/时间戳/轮数），小，每次追加整体原子重写
        └── turns.jsonl   逐轮追加，一行一轮（append-only）
  为什么拆两个文件？
    · turns.jsonl 追加写：长会话每轮只 append 一行，不必把整段历史重写一遍
      （旧的"单 JSON 文件整体覆盖"随轮数增长会越来越贵）；
    · meta.json 单独放：列表页只想看标题/时间/轮数，读一个几百字节的小文件即可，
      不必把成千上万行的 turns 全 parse 一遍。

★ web-ready 的两个概念，别混：
    · SessionStore —— 无状态的"仓库/工厂"。负责算路径、校验 id、列目录、发句柄。
      本身不绑定任何一次具体会话，可被所有请求共享（进程级单例即可）。
    · Session      —— 绑定到 (operator_id, session_id) 的"会话句柄"。
      读写某个具体会话都通过它。web 里"切换会话"= 换一个 Session 句柄。
  operator_id 一律由调用方传入（web 从登录态取，CLI 传固定值），库里不写死。

★ 目录懒创建：构造 SessionStore / 拿到 Session 句柄都不碰磁盘。
  只有真正第一次 append_turn 时才 makedirs 建这个会话的目录 ——
  "新建一个还没说话的会话"不会在磁盘上留下空目录。

★ 必须守住的坑（和 memory/store.py 同一套骨架）：
    1. operator_id / session_id 拼路径 → 白名单校验，防路径穿越
    2. 落盘 → 临时文件 + os.fsync + os.replace 原子写；turns 追加也 flush+fsync
    3. 读失败/文件损坏 → 降级为空，绝不抛异常拖垮调用方
    4. 并发安全 → 同一会话目录一把进程内锁，串行化 meta 重写与 turns 追加
"""
from __future__ import annotations

import json
import logging
import os
import re
import threading
import uuid
from datetime import datetime, timezone

from llm_types import Message, Role, TextPart
from pydantic import BaseModel, ConfigDict

log = logging.getLogger("session_store")

# ★ 坑1：只允许字母数字下划线连字符做 id，挡住 ../ 这类路径穿越
_ID_PATTERN = re.compile(r"^[A-Za-z0-9_-]+$")

# 列表里给会话起的默认标题：取首条用户消息的前 N 字
_TITLE_MAX = 30

_META_FILE = "meta.json"
_TURNS_FILE = "turns.jsonl"


def _now_iso() -> str:
    """统一 UTC ISO 时间戳（和 memory 一致：存 UTC 不存本地时区）。"""
    return datetime.now(timezone.utc).isoformat()


class _Base(BaseModel):
    """忽略未知字段：日后加字段时，老文件仍能被载入。"""
    model_config = ConfigDict(extra="ignore")


class Turn(_Base):
    """一轮发言。role 只取 'user' | 'assistant'。"""
    role: str = ""
    text: str = ""
    ts: str = ""


class SessionMeta(_Base):
    """会话元信息（列表页只需要这些，不必读 turns）。"""
    session_id: str = ""
    operator_id: str = ""
    title: str = ""                       # 便于在列表里辨认，默认取首条用户消息
    created_at: str = ""
    updated_at: str = ""
    turn_count: int = 0


class Session:
    """绑定到某个具体会话 (operator_id, session_id) 的句柄。

    ★ 由 SessionStore.new/open/switch 产出，不要自己直接构造（构造也不落盘，
      但拿不到 store 的锁注册表，写并发就不安全了）。
    ★ 句柄很轻：只记路径 + 共享锁，元信息懒读，磁盘目录懒建。
    """

    def __init__(self, store: "SessionStore", meta: SessionMeta,
                 lock: threading.Lock):
        self._store = store
        self._meta = meta
        self._lock = lock          # 同一会话目录全进程共享这一把锁（store 发的）

    # ---- 只读属性 ----
    @property
    def operator_id(self) -> str:
        return self._meta.operator_id

    @property
    def session_id(self) -> str:
        return self._meta.session_id

    @property
    def title(self) -> str:
        return self._meta.title

    @property
    def turn_count(self) -> int:
        return self._meta.turn_count

    @property
    def meta(self) -> SessionMeta:
        """当前已知的元信息（内存里的最新副本）。"""
        return self._meta

    @property
    def dir(self) -> str:
        return self._store._session_dir(self.operator_id, self.session_id)

    def exists(self) -> bool:
        """这个会话在磁盘上是否已经真正落过盘（建过目录 + 写过 meta）。"""
        return os.path.isfile(os.path.join(self.dir, _META_FILE))

    # ---- 读：任何异常都降级为空，绝不崩（坑3）----
    def turns(self) -> list[Turn]:
        """读 turns.jsonl，逐行 parse；坏行跳过并告警，不整篇作废。"""
        path = os.path.join(self.dir, _TURNS_FILE)
        if not os.path.isfile(path):
            return []
        out: list[Turn] = []
        try:
            with open(path, "r", encoding="utf-8") as f:
                for lineno, line in enumerate(f, 1):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        out.append(Turn.model_validate(json.loads(line)))
                    except Exception:  # noqa: BLE001 —— 单行坏不牵连全局
                        log.warning("turns.jsonl 第 %d 行损坏，跳过 %s",
                                    lineno, path)
        except Exception:  # noqa: BLE001 —— 整个文件读不了就当没有
            log.warning("读取 turns 失败，返回空 %s", path, exc_info=True)
            return []
        return out

    def to_messages(self) -> list[Message]:
        """把干净轮次重建成 list[Message] 喂回 run()（只含 user/assistant 文本）。

        ★ 不含记忆简报：run() 每轮会按最新记忆 + 当轮提问重新注入。
        ★ 不含工具调用：干净对话的选择就是丢掉这些运行期细节。
        """
        msgs: list[Message] = []
        for t in self.turns():
            role = Role.USER if t.role == "user" else Role.ASSISTANT
            msgs.append(Message(role=role, parts=[TextPart(t.text)]))
        return msgs

    # ---- 写：追加一轮（懒建目录 + turns 追加 + meta 原子重写），全程持锁 ----
    def append_turn(self, role: str, text: str) -> None:
        if role not in ("user", "assistant"):
            raise ValueError(f"非法 role: {role!r}（只支持 user/assistant）")
        with self._lock:                     # 坑4：串行化对同一会话的并发写
            self._store._ensure_dir(self.dir)  # 懒创建：第一次说话才建目录
            now = _now_iso()

            # 1) turns.jsonl 追加一行（append-only，长会话不必重写历史）
            turn = Turn(role=role, text=text, ts=now)
            tpath = os.path.join(self.dir, _TURNS_FILE)
            with open(tpath, "a", encoding="utf-8") as f:
                f.write(json.dumps(turn.model_dump(mode="json"),
                                   ensure_ascii=False) + "\n")
                f.flush()
                os.fsync(f.fileno())         # 坑2：确保落到物理磁盘

            # 2) 更新内存里的 meta，再原子重写 meta.json
            if not self._meta.title and role == "user" and text.strip():
                self._meta.title = text.strip()[:_TITLE_MAX]
            self._meta.turn_count += 1
            self._meta.updated_at = now
            self._store._write_meta(self.dir, self._meta)

    # ---- web 切换会话：拿同一 operator 下另一个会话的句柄 ----
    def switch(self, session_id: str) -> "Session":
        """切到同一用户的另一个会话（不存在则以该 id 新建一个空句柄）。"""
        return self._store.switch(self.operator_id, session_id)


class SessionStore:
    """无状态的会话仓库/工厂：算路径、校验 id、列目录、发句柄。

    ★ 不绑定任何一次具体会话，可被所有请求共享（进程级单例即可）。
    ★ 构造时"不"建任何目录（懒创建）：base_dir 也等到第一次真正写才建。
    """

    def __init__(self, base_dir: str):
        self._base_dir = base_dir
        # 每个会话目录一把锁；用一张注册表让"同一会话的多个句柄"共享同一把锁
        self._locks: dict[str, threading.Lock] = {}
        self._locks_guard = threading.Lock()

    # ---- 私有：安全路径（坑1）----
    def _check_id(self, value: str, label: str) -> str:
        if not isinstance(value, str) or not _ID_PATTERN.match(value):
            raise ValueError(f"非法 {label}: {value!r}")
        return value

    def _user_dir(self, operator_id: str) -> str:
        self._check_id(operator_id, "operator_id")
        return os.path.join(self._base_dir, operator_id)

    def _session_dir(self, operator_id: str, session_id: str) -> str:
        self._check_id(operator_id, "operator_id")
        self._check_id(session_id, "session_id")
        return os.path.join(self._user_dir(operator_id), session_id)

    def _lock_for(self, session_dir: str) -> threading.Lock:
        with self._locks_guard:
            lk = self._locks.get(session_dir)
            if lk is None:
                lk = threading.Lock()
                self._locks[session_dir] = lk
            return lk

    # ---- 私有：落盘原语（坑2）----
    def _ensure_dir(self, session_dir: str) -> None:
        os.makedirs(session_dir, exist_ok=True)

    def _write_meta(self, session_dir: str, meta: SessionMeta) -> None:
        path = os.path.join(session_dir, _META_FILE)
        payload = meta.model_dump_json(indent=2)
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            f.write(payload)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)               # 原子替换：要么旧要么新，不会半截

    def _read_meta(self, operator_id: str, session_id: str) -> SessionMeta:
        """读 meta.json；缺失/损坏降级为一份带正确 id 的空元信息（坑3）。"""
        path = os.path.join(self._session_dir(operator_id, session_id), _META_FILE)
        if not os.path.isfile(path):
            now = _now_iso()
            return SessionMeta(session_id=session_id, operator_id=operator_id,
                               created_at=now, updated_at=now)
        try:
            with open(path, "r", encoding="utf-8") as f:
                meta = SessionMeta.model_validate(json.load(f))
        except Exception:  # noqa: BLE001
            log.warning("meta.json 损坏，降级为空 %s", path, exc_info=True)
            now = _now_iso()
            return SessionMeta(session_id=session_id, operator_id=operator_id,
                               created_at=now, updated_at=now)
        # ★ 以调用方给的 id 为准，强制校正，防止文件内字段被改串
        meta.operator_id = operator_id
        meta.session_id = session_id
        return meta

    # ---- 会话 id：时间戳(本地可读) + 短 uuid(防同秒碰撞) ----
    def new_session_id(self) -> str:
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        return f"{stamp}-{uuid.uuid4().hex[:6]}"

    def _make_handle(self, meta: SessionMeta) -> Session:
        session_dir = self._session_dir(meta.operator_id, meta.session_id)
        return Session(self, meta, self._lock_for(session_dir))

    # =====================================================================
    # 工厂 API：new / open / switch / list
    # =====================================================================
    def new(self, operator_id: str, session_id: str | None = None) -> Session:
        """开一个全新会话：返回句柄，但"不"落盘（等第一次 append_turn）。"""
        self._check_id(operator_id, "operator_id")
        sid = session_id or self.new_session_id()
        self._check_id(sid, "session_id")
        now = _now_iso()
        meta = SessionMeta(session_id=sid, operator_id=operator_id,
                           created_at=now, updated_at=now)
        return self._make_handle(meta)

    def open(self, operator_id: str, session_id: str) -> Session:
        """打开一个已存在的会话：读它的 meta 返回句柄。

        ★ 若目录/ meta 不存在或损坏，不报错 —— 降级成"以该 id 为准的空会话"，
          后续 append_turn 会把它当新会话建起来。web 里更常见的是先 list 再 open。
        """
        meta = self._read_meta(operator_id, session_id)
        return self._make_handle(meta)

    def switch(self, operator_id: str, session_id: str) -> Session:
        """web "切换会话"入口：语义等同 open（不存在则以该 id 新建空句柄）。

        单列一个名字是为了让调用方意图清晰：open 偏"读一个已知会话"，
        switch 偏"用户在侧边栏点了另一个会话"。行为一致，便于日后各自演化。
        """
        return self.open(operator_id, session_id)

    def list(self, operator_id: str) -> list[SessionMeta]:
        """列出某用户的所有会话元信息（只读 meta.json，按更新时间倒序）。"""
        user_dir = self._user_dir(operator_id)
        if not os.path.isdir(user_dir):
            return []
        metas: list[SessionMeta] = []
        for name in os.listdir(user_dir):
            sess_dir = os.path.join(user_dir, name)
            # 只认"目录 + 里面有 meta.json"的条目，散落的临时文件忽略
            if not os.path.isdir(sess_dir):
                continue
            if not _ID_PATTERN.match(name):
                continue
            if not os.path.isfile(os.path.join(sess_dir, _META_FILE)):
                continue
            metas.append(self._read_meta(operator_id, name))
        metas.sort(key=lambda m: m.updated_at, reverse=True)
        return metas


# ===========================================================================
# 进程级默认 store：和 memory.get_default_store 一个套路。
#   ★ 路径基于本文件位置算出 Ai/data/sessions，不依赖进程 cwd。
#   ★ 单例的是"仓库"（无状态、可共享），不是某一次会话 —— 会话句柄按需 new/open。
#   ★ 构造不建目录：base_dir 也等第一次写才懒建。
# ===========================================================================
_AI_ROOT = os.path.dirname(os.path.abspath(__file__))
DEFAULT_SESSION_DIR = os.path.join(_AI_ROOT, "data", "sessions")

_default_store: "SessionStore | None" = None
_default_guard = threading.Lock()


def get_default_session_store() -> SessionStore:
    """返回全进程共用的默认 SessionStore（懒加载单例，双检锁）。"""
    global _default_store
    if _default_store is None:
        with _default_guard:
            if _default_store is None:
                _default_store = SessionStore(DEFAULT_SESSION_DIR)
    return _default_store

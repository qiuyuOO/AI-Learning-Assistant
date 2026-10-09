"""长期记忆的持久化：一个用户一个 JSON 文件，按 user_id 隔离。

★ 这一层只干"存/取"，不认识 LLM、不认识 prompt、不认识 loop。
  它和你之前写的 PreferenceStore 是同一套骨架，只是存的东西从
  "扁平偏好 dict"升级成了完整的 UserMemory 对象。

★ 三个必须守住的坑（和之前一样）：
  1. user_id 拼文件名 → 白名单校验，防路径穿越
  2. 落盘 → 临时文件 + os.replace 原子写
  3. 读失败/文件损坏 → 降级为空记忆，绝不抛异常拖垮调用方
"""
import json
import logging
import os
import re

from memory.schema import Interaction, UserMemory

log = logging.getLogger("memory.store")

# ★ 坑1：只允许字母数字下划线连字符做 user_id，挡住 ../ 这类穿越
_ID_PATTERN = re.compile(r"^[A-Za-z0-9_-]+$")


class MemoryStore:
    def __init__(self, base_dir: str):
        self._base_dir = base_dir
        os.makedirs(base_dir, exist_ok=True)

    # ---- 私有：user_id -> 安全路径 ----
    def _path(self, user_id: str) -> str:
        if not isinstance(user_id, str) or not _ID_PATTERN.match(user_id):
            raise ValueError(f"非法 user_id: {user_id!r}")
        return os.path.join(self._base_dir, f"{user_id}.json")

    # ---- 读：任何异常都降级为"空记忆"，绝不崩 ----
    def load(self, user_id: str) -> UserMemory:
        path = self._path(user_id)
        if not os.path.exists(path):
            return UserMemory.empty(user_id)

        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            memory = UserMemory.model_validate(data)
        except Exception:  # noqa: BLE001 —— JSON 坏 / 字段类型不对，都降级
            log.warning("记忆文件损坏，降级为空记忆 user_id=%s", user_id,
                        exc_info=True)
            return UserMemory.empty(user_id)

        # ★ 以文件名为准强制校正 user_id：防止 A 的文件里写着 B 的 id 造成串号
        memory.user_id = user_id
        memory.profile.user_id = user_id
        return memory

    # ---- 写：整体覆盖，原子落盘 ----
    def save(self, user_id: str, memory: UserMemory) -> None:
        path = self._path(user_id)
        # 同样以调用方给的 user_id 为准，模型填的不算数
        memory.user_id = user_id
        memory.profile.user_id = user_id

        payload = memory.model_dump_json(indent=2)  # pydantic v2：对象 → JSON 串

        # ★ 坑2：先写 .tmp 再 os.replace 覆盖，写一半崩了也不会损坏原文件
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            f.write(payload)
            f.flush()
            os.fsync(f.fileno())   # 确保内容真的落到磁盘，再改名
        os.replace(tmp, path)      # 同分区原子操作

    # =======================================================================
    # 待处理交互缓冲区：支撑"攒够 N 次对话再更新记忆"
    #   为什么落盘：run() 每次是独立调用、进程可能重启，内存计数会丢；
    #   而且批量更新本来就要把这 N 条交互一起喂给 LLM，必须存下来。
    #   形态：每个用户一个 JSONL 追加文件，一行一条 Interaction。
    # =======================================================================
    def _pending_path(self, user_id: str) -> str:
        if not isinstance(user_id, str) or not _ID_PATTERN.match(user_id):
            raise ValueError(f"非法 user_id: {user_id!r}")
        return os.path.join(self._base_dir, f"{user_id}.pending.jsonl")

    def append_interaction(self, user_id: str, interaction: Interaction) -> None:
        """把一条交互追加到缓冲区。追加写，不读全文，便宜。"""
        path = self._pending_path(user_id)
        # model_dump_json() 默认紧凑单行，正好当 JSONL 的一行
        with open(path, "a", encoding="utf-8") as f:
            f.write(interaction.model_dump_json() + "\n")

    def peek_pending(self, user_id: str) -> list[Interaction]:
        """读出当前所有待处理交互，但不删除（更新成功后才由 clear_pending 清）。"""
        path = self._pending_path(user_id)
        if not os.path.exists(path):
            return []
        items: list[Interaction] = []
        try:
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    items.append(Interaction.model_validate(json.loads(line)))
        except Exception:  # noqa: BLE001 —— 缓冲损坏不该拖垮主流程
            log.warning("待处理缓冲损坏，本次忽略 user_id=%s", user_id,
                        exc_info=True)
            return []
        return items

    def clear_pending(self, user_id: str) -> None:
        """清空缓冲区。★ 只在记忆真正更新成功后调用。

        不用 os.remove（避免永久删除语义）：写一个空文件再原子覆盖，
        效果等同清空，且保持"要么全清要么不清"的原子性。
        """
        path = self._pending_path(user_id)
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)

    def take_pending(self, user_id: str) -> list[Interaction]:
        """原子"认领"当前所有待处理交互：读出后立即清空缓冲并返回。

        ★ 用于后台更新前的交接：一旦 take 走，这批就从 pending 移除，
          后续对话再 append 的是新的一批，不会被这次的清空误删。
        ★ 读+清之间不能与并发 append 交错 —— 由 MemoryManager 的锁保证
          （append / take / 失败回填都在同一把锁下）。
        """
        items = self.peek_pending(user_id)
        if items:
            self.clear_pending(user_id)
        return items


# ===========================================================================
# 进程级默认 store：让 loop（写记忆）和各工具（读记忆，如 arxiv 的 Jev 提示词）
# 一定指向同一个目录，否则 cwd 不同就会读写到两处、记忆对不上。
#   ★ 路径基于本文件位置算出 Ai/data/memory，不依赖进程 cwd。
# ===========================================================================
_AI_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_MEMORY_DIR = os.path.join(_AI_ROOT, "data", "memory")

_default_store: "MemoryStore | None" = None


def get_default_store() -> MemoryStore:
    """返回全进程共用的默认 MemoryStore（懒加载单例）。"""
    global _default_store
    if _default_store is None:
        _default_store = MemoryStore(DEFAULT_MEMORY_DIR)
    return _default_store

"""记忆管理器：编排 prompt / schema / store，并实现"攒够 N 次对话再更新"。

★ 对 loop 暴露两个入口，一读一写，loop 保持很傻：
    load(user_id)            —— 读路径：拿当前记忆去渲染背景简报
    observe(user_id, interaction) —— 写路径：每次对话丢一条进来，
                                     内部攒够 update_every 条才真正调 LLM 更新

★ 为什么攒批更新：提示词强调"重复证据 > 单次证据"。一次喂 10 条交互，
  LLM 更能识别稳定模式、给出可靠 confidence；同时把 LLM 调用从
  "每次对话一次"降到"每 N 次对话一次"，省钱省时。

★ 铁律：LLM 调用/解析失败绝不拖垮主流程，也绝不把坏数据写盘、
  更绝不丢掉缓冲（失败时保留 pending，下次攒够再重试）。
"""
import json
import logging
import re
import threading
from concurrent.futures import ThreadPoolExecutor

from llm_types import user_message
from memory.prompt import MEMORY_MANAGER_SYSTEM_PROMPT
from memory.schema import (Interaction, ManagerOutput, MemoryChange,
                           UserMemory, now_iso)
from memory.store import MemoryStore, get_default_store

log = logging.getLogger("memory.manager")

# 模型有时会把 JSON 包在 ```json ... ``` 围栏里
_FENCE = re.compile(r"```(?:json)?\s*(\{.*\})\s*```", re.S)

DEFAULT_UPDATE_EVERY = 10   # 默认攒够 10 次对话更新一次
MIN_CONFIDENCE = 0.5        # 渲染背景简报时的置信度门槛（低于它不进上下文）


def _extract_json(text: str) -> dict:
    """从模型输出里抠出第一个 JSON 对象（三级兜底：直接 parse → 去围栏 → 截 {}）。"""
    text = (text or "").strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    m = _FENCE.search(text)
    if m:
        return json.loads(m.group(1))

    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end > start:
        return json.loads(text[start:end + 1])

    raise ValueError(f"无法从模型输出解析 JSON：{text[:200]!r}")


# ---------------------------------------------------------------------------
# 轻量相关性打分：让记忆简报能"按当前提问聚焦"
#   ★ 无第三方依赖、纯字符串处理：中文按 unigram+bigram，英文/数字按词。
#     只是启发式排序用，不追求语言学正确 —— 够用、可解释，失败也只是退回全量。
# ---------------------------------------------------------------------------
_ASCII_TOKEN = re.compile(r"[a-z0-9]+")
_CJK_RUN = re.compile(r"[\u4e00-\u9fff]+")


def _tokens(text: str) -> set[str]:
    """文本 → 归一化 token 集合（小写 ASCII 词 + 中文相邻二字 bigram）。

    ★ 只取中文 bigram、不取单字：单个汉字重合度太高（"量子"和"向量"都含"量"），
      会造成大量伪相关。二字组合才更能代表词义。
    """
    text = (text or "").lower()
    toks: set[str] = set(_ASCII_TOKEN.findall(text))
    for run in _CJK_RUN.findall(text):
        for i in range(len(run) - 1):        # 中文 bigram
            toks.add(run[i:i + 2])
    return toks


def _relevance(text: str, query: str) -> float:
    """一条记忆文本与本轮提问的相关度，>=0；0 表示无关。

    ★ 命中 token 里，多字符（中文 bigram / 英文词）比单个字符更 specific，权重更高。
    ★ 用 query 的 token 规模开方归一，避免"长文本天然更容易命中"的偏置。
    """
    a, b = _tokens(text), _tokens(query)
    if not a or not b:
        return 0.0
    inter = a & b
    if not inter:
        return 0.0
    score = sum(2.0 if len(t) >= 2 else 0.5 for t in inter)
    return score / (len(b) ** 0.5)


# ---------------------------------------------------------------------------
# 读路径渲染：把结构化 UserMemory 压成一小段给模型看的自然语言简报
# ---------------------------------------------------------------------------
_CAP_PER_CATEGORY = 5   # 每个类别最多渲染几条，防止简报过长撑大上下文


def render_memory_context(memory: UserMemory, query: str | None = None,
                          min_confidence: float = MIN_CONFIDENCE) -> str | None:
    """UserMemory → 精简背景简报。记忆为空/无高置信条目时返回 None（不注入）。

    ★ 不直接丢整个 JSON：太大太噪，还会撑大每一轮上下文。
      只挑 status=active 且置信度达标的条目，渲染成人话。
    ★ query 给定时"按提问聚焦"：每个类别只保留与本轮提问相关的条目，
      并按（相关度, 置信度）从高到低排、截断到前 _CAP_PER_CATEGORY 条。
      某类别全不相关就整类省略 —— 简报侧重点因此随每次输入而不同。
    ★ 聚焦后若整份简报都空了（本轮提问很新、记忆里没有相关内容），
      优雅退回"不带 query 的高置信全量简报"，保证个性化不会突然消失。
    """
    def _active(item) -> bool:
        return getattr(item, "status", "active") == "active"

    def _pick(cands: list[tuple[str, float]]) -> list[str]:
        """cands: [(文本, 置信度/权重)] → 该类别最终要渲染的文本列表。"""
        cands = [(t, c) for (t, c) in cands if t]
        if not cands:
            return []
        if query:
            scored = [(t, c, _relevance(t, query)) for (t, c) in cands]
            relevant = [x for x in scored if x[2] > 0]
            if not relevant:
                return []                       # 本类别与提问无关 → 省略
            relevant.sort(key=lambda x: (x[2], x[1]), reverse=True)
            return [t for (t, _c, _s) in relevant[:_CAP_PER_CATEGORY]]
        cands.sort(key=lambda x: x[1], reverse=True)   # 无 query：按置信度取前几
        return [t for (t, _c) in cands[:_CAP_PER_CATEGORY]]

    lines: list[str] = []

    interests = _pick([(str(it.value), it.confidence) for it in memory.interests
                       if _active(it) and it.confidence >= min_confidence and it.value])
    if interests:
        lines.append("长期关注：" + "、".join(dict.fromkeys(interests)))

    likes = _pick([(p.subject, p.strength or abs(p.preference))
                   for p in memory.preferences if p.subject and p.preference > 0])
    if likes:
        lines.append("偏好：" + "、".join(likes))

    dislikes = _pick([(str(d.value), d.confidence) for d in memory.dislikes
                      if _active(d) and d.value])
    if dislikes:
        lines.append("不喜欢：" + "、".join(dislikes))

    criteria = _pick([(c.name, c.weight) for c in memory.decision_criteria if c.name])
    if criteria:
        lines.append("评价时看重：" + " > ".join(criteria))

    rules = _pick([(r.conclusion_pattern, r.confidence) for r in memory.decision_rules
                   if r.conclusion_pattern and r.confidence >= min_confidence])
    if rules:
        lines.append("决策习惯：" + "；".join(rules))

    beliefs = _pick([(b.proposition, b.confidence) for b in memory.beliefs
                     if b.proposition and b.confidence >= min_confidence])
    if beliefs:
        lines.append("用户相信：" + "；".join(beliefs))

    habits = _pick([(h.behavior, h.confidence) for h in memory.habits
                    if h.behavior and h.confidence >= min_confidence])
    if habits:
        lines.append("行为习惯：" + "；".join(habits))

    kept = [ln for ln in lines if ln.split("：", 1)[-1].strip()]
    if not kept:
        # 聚焦后全空 → 退回不带 query 的全量简报（仅在本来带 query 时递归一次）
        if query:
            return render_memory_context(memory, query=None,
                                         min_confidence=min_confidence)
        return None
    return "\n".join("· " + ln for ln in kept)


class MemoryManager:
    def __init__(self, client, store: MemoryStore,
                 update_every: int = DEFAULT_UPDATE_EVERY):
        # ★ client 只要有 .chat(messages, system=...) 即可（鸭子类型，便于假测）
        self._client = client
        self._store = store
        self._update_every = max(1, int(update_every))

        # 后台更新：昂贵的 LLM 调用放到单 worker 线程，observe 立刻返回，
        # 不阻塞用户回答。max_workers=1 保证同一进程内记忆更新串行，
        # 避免两次更新并发读改写同一份 UserMemory 互相覆盖。
        self._lock = threading.Lock()
        self._executor = ThreadPoolExecutor(max_workers=1,
                                            thread_name_prefix="mem-flush")

    # ---- 读路径入口 ----
    def load(self, user_id: str) -> UserMemory:
        return self._store.load(user_id)

    # ---- 写路径入口：每次对话调这个，攒够才真正更新（更新在后台） ----
    def observe(self, user_id: str, interaction: Interaction | dict) -> None:
        if isinstance(interaction, dict):
            interaction = Interaction.model_validate(interaction)

        batch: list[Interaction] = []
        # ★ append / 计数 / 认领 都在同一把锁下，和后台的失败回填互斥，
        #   保证"认领走的那批"和"新 append 的那批"不会互相踩。
        with self._lock:
            self._store.append_interaction(user_id, interaction)
            if len(self._store.peek_pending(user_id)) >= self._update_every:
                batch = self._store.take_pending(user_id)   # 主线程原子认领（快）

        # 认领到一批才丢后台跑 LLM；这一步在主线程外，不阻塞返回。
        if batch:
            self._executor.submit(self._flush, user_id, batch)

    def _flush(self, user_id: str, interactions: list[Interaction]) -> None:
        """后台线程里执行：调 LLM 更新记忆。

        ★ 这批已经被 take_pending 从缓冲移走了：
          成功 → 记忆已落盘，无需再动缓冲；
          失败 → 把这批"回填"到缓冲，下次攒够再重试，绝不丢交互。
        """
        ok, output = self._update(user_id, interactions)
        if ok:
            log.info("记忆已更新（后台）user_id=%s 批量=%d 变更=%d",
                     user_id, len(interactions), len(output.changes))
            return
        with self._lock:
            for it in interactions:
                self._store.append_interaction(user_id, it)
        log.warning("记忆更新失败，已回填 %d 条缓冲待重试 user_id=%s",
                    len(interactions), user_id)

    def close(self, wait: bool = True) -> None:
        """关闭后台线程池。wait=True 时等已提交的更新跑完（进程退出前调用）。"""
        self._executor.shutdown(wait=wait)

    # ---- 写路径入口（显式）：用户明确要求"记住"时用，立即同步更新 ----
    def remember(self, user_id: str,
                 interaction: Interaction | dict) -> tuple[bool, ManagerOutput]:
        """显式记忆：立刻更新（同步返回结果），并顺带结算、清空待处理缓冲。

        ★ 和 observe 的三点区别：
          1. 同步：工具要当场拿到"记没记住"的结果回复用户，不能丢后台。
          2. 立即：不攒批 —— 用户都明说"记住"了，就该马上写进长期记忆。
          3. 连带 flush：take_pending 把已缓冲的对话认领走（缓冲即被清空），
             和这条显式偏好合成一批一起更新。既不丢历史交互，也避免这批
             缓冲稍后又触发一次重复更新。

        ★ 仍走同一个单 worker 线程池：与后台 flush 串行，
          杜绝"显式更新"和"后台更新"并发 load-modify-save 同一份记忆文件。
        ★ 返回 (ok, output)：ok=False 表示 LLM 调用/解析失败，
          这批（含显式偏好）已回填缓冲待重试 —— 工具据此给用户不同话术。
        """
        if isinstance(interaction, dict):
            interaction = Interaction.model_validate(interaction)
        # submit(...).result()：排队到后台线程执行并阻塞等结果，
        # 从而和已有的后台 flush 天然串行（不会两个 _update 同时写盘）。
        return self._executor.submit(self._remember, user_id, interaction).result()

    def _remember(self, user_id: str, interaction: Interaction) -> tuple[bool, ManagerOutput]:
        # 认领并清空缓冲（锁内），把显式偏好放在这批的最后
        with self._lock:
            batch = self._store.take_pending(user_id)
            batch.append(interaction)

        ok, output = self._update(user_id, batch)
        if ok:
            log.info("显式记忆已更新 user_id=%s 合并交互=%d 变更=%d",
                     user_id, len(batch), len(output.changes))
            return True, output

        # 失败：把这批（含显式偏好）回填缓冲，下次攒够或再次 remember 时重试，
        # 绝不丢用户明确要求记住的东西。
        with self._lock:
            for it in batch:
                self._store.append_interaction(user_id, it)
        log.warning("显式记忆更新失败，已回填 %d 条缓冲待重试 user_id=%s",
                    len(batch), user_id)
        return False, output

    # ---- 单条/批量更新（向后兼容 + 手动触发用）----
    def update(self, user_id: str,
               interaction: Interaction | dict) -> ManagerOutput:
        """立即用单条交互更新（不走缓冲）。保留给测试/手动场景。"""
        if isinstance(interaction, dict):
            interaction = Interaction.model_validate(interaction)
        _, output = self._update(user_id, [interaction])
        return output

    def update_batch(self, user_id: str, interactions: list) -> ManagerOutput:
        """立即用一批交互更新（不走缓冲）。"""
        items = [Interaction.model_validate(i) if isinstance(i, dict) else i
                 for i in interactions]
        _, output = self._update(user_id, items)
        return output

    # ---- 真正的 LLM 调用核心：返回 (是否成功, 输出) ----
    def _update(self, user_id: str,
                interactions: list[Interaction]) -> tuple[bool, ManagerOutput]:
        current = self._store.load(user_id)
        payload = {
            "current_user_memory": current.model_dump(mode="json"),
            "interactions": [i.model_dump(mode="json") for i in interactions],
        }

        try:
            resp = self._client.chat(
                [user_message(json.dumps(payload, ensure_ascii=False))],
                system=MEMORY_MANAGER_SYSTEM_PROMPT,
            )
            output = ManagerOutput.model_validate(_extract_json(resp.text))
        except Exception:  # noqa: BLE001 —— 记忆是增强项，失败必须能降级
            log.exception("记忆更新失败，保持原样 user_id=%s", user_id)
            return False, ManagerOutput(
                updated_memory=current,
                changes=[MemoryChange(action="NO_CHANGE",
                                      reason="记忆管理器调用/解析失败，未改动")],
            )

        # ★ 身份/时间戳以本地为准，不信模型填的
        memory = output.updated_memory
        memory.user_id = user_id
        memory.profile.user_id = user_id
        memory.updated_at = now_iso()

        self._store.save(user_id, memory)
        return True, output


# ===========================================================================
# 进程级默认 manager：让"显式记忆工具"和 loop 共用同一个 MemoryManager。
#   ★ 为什么必须要单例：工具 handler 的契约是 (…, *, session)，拿不到 loop
#     在 __main__ 里 new 出来的那个 manager。若工具自己再 new 一个，就会有
#     两个实例、两把锁、两个线程池，却指向同一份记忆文件 —— 并发 load-modify-save
#     会互相覆盖。用单例把它们收敛到同一把锁 / 同一个 store / 同一个 executor。
#   ★ 首次调用一般由 loop.__main__ 传入对话 client 完成初始化；之后（工具触发）
#     直接拿同一实例。若从未初始化又没传 client，就按 .env 自建一个专用 client。
# ===========================================================================
_default_manager: "MemoryManager | None" = None
_manager_lock = threading.Lock()


def _build_client_from_env():
    """没有现成 client 时（例如工具独立触发、脚本直跑），按 .env 自建一个。

    ★ 懒加载 import：memory 包导入期不拉 llm_client / dotenv 这些重依赖，
      也避免潜在循环导入（llm_client 依赖链很长）。
    """
    import os
    from dotenv import load_dotenv
    from llm_client import LLMClient

    load_dotenv()
    return LLMClient(
        provider="deepseek",
        model=os.getenv("ANTHROPIC_MODEL_NAME", "deepseek-flash"),
        api_key=os.getenv("ANTHROPIC_API_KEY", ""),
        base_url=os.getenv("ANTHROPIC_BASE_URL"),
    )


def get_default_manager(client=None,
                        update_every: int = DEFAULT_UPDATE_EVERY) -> MemoryManager:
    """返回全进程共用的默认 MemoryManager（懒加载单例，双检锁）。

    ★ 只有"第一次创建"时 client / update_every 才生效；之后调用忽略入参、
      返回同一实例。这样 loop 先用对话 client 建好，工具随后拿到的是同一个。
    """
    global _default_manager
    with _manager_lock:
        if _default_manager is None:
            if client is None:
                client = _build_client_from_env()
            _default_manager = MemoryManager(
                client, get_default_store(), update_every=update_every)
        return _default_manager


if __name__ == "__main__":
    # 不连真实 API 的自测：假 client + update_every=3，验证"攒够才更新 + 后台异步"。
    import os
    import shutil
    import time

    class _FakeClient:
        def __init__(self):
            self.calls = 0

        def chat(self, messages, system=None, **kw):
            self.calls += 1
            time.sleep(0.3)   # 模拟 LLM 延迟，好观察"observe 不阻塞"
            n = len(json.loads(messages[0].parts[0].text)["interactions"])

            class _R:
                text = ("```json\n" + json.dumps({
                    "updated_memory": {
                        "user_id": "cli",
                        "profile": {"user_id": "cli", "language": "zh"},
                        "interests": [{
                            "id": "int_1", "value": "RAG", "confidence": 0.8,
                            "source": "user_explicit", "status": "active",
                            "evidence_refs": [f"i_{n}"],
                        }],
                    },
                    "changes": [{"action": "ADD", "memory_type": "interest",
                                 "memory_id": "int_1",
                                 "reason": f"综合 {n} 条交互确认对 RAG 的持续兴趣"}],
                }, ensure_ascii=False) + "\n```")
            return _R()

    DIR = os.path.join("data", "_demo")
    shutil.rmtree(DIR, ignore_errors=True)
    store = MemoryStore(DIR)
    fake = _FakeClient()
    manager = MemoryManager(fake, store, update_every=3)

    for i in range(1, 6):
        t0 = time.perf_counter()
        manager.observe("cli", {
            "interaction_id": f"i_{i}", "timestamp": now_iso(),
            "user_message": f"第 {i} 次：多给我推 RAG 的论文",
            "agent_message": "好的。",
        })
        # ★ observe 应当几乎瞬间返回（即便这一条触发了后台更新）
        ms = (time.perf_counter() - t0) * 1000
        print(f"observe #{i}: 耗时={ms:.1f}ms 缓冲待处理={len(store.peek_pending('cli'))}")

    print("close 前 LLM调用次数（后台可能还没跑完）:", fake.calls)
    manager.close(wait=True)   # 进程退出前等后台更新落盘
    print("close 后 LLM调用次数:", fake.calls)
    print("最终记忆 interests:", store.load("cli").interests)
    shutil.rmtree(DIR, ignore_errors=True)

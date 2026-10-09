"""UserMemory 的数据结构（严格对齐 prompt.py 里的规范）。

★ 为什么用 pydantic 而不是 dataclass：
  记忆的"生产者"是 LLM，输出是 JSON 字符串。pydantic 能一步完成
  "JSON → 校验 → 对象"，字段类型不对会明确报错，省去手写一堆校验。

★ 为什么所有字段都给默认值、且 extra="ignore"：
  LLM 不保证每次都吐完整、规整的 JSON。宽松校验（缺字段补默认、
  多字段忽略）能让偶尔不完美的输出仍然可用，而不是整次更新失败。
  这是"面向 LLM 的数据结构"和"面向人的数据结构"最大的区别。
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Annotated, Any, Optional

from pydantic import BaseModel, BeforeValidator, ConfigDict, Field


def now_iso() -> str:
    """统一的 UTC ISO 时间戳。★ 存 UTC 不存本地时区，跨机器/跨时区不打架。"""
    return datetime.now(timezone.utc).isoformat()


class _Base(BaseModel):
    """所有记忆模型的公共基类：忽略 LLM 多吐的未知字段。"""
    model_config = ConfigDict(extra="ignore")


# ---------------------------------------------------------------------------
# 宽松的 float 类型：吸收 LLM 对数值字段的"不规矩"输出。
#   ★ 为什么需要：LLM 有时把 strength / confidence 这类数值字段写成
#     分类词（"high"/"medium"/"low"）或别的字符串。pydantic 默认严格模式下，
#     只要一个字段解析不了，整个 ManagerOutput 校验就失败 —— 一次记忆更新
#     会因为一个脏字段全盘回滚。这与本模块"面向 LLM 宽松校验"的原则相悖。
#   ★ 做法：BeforeValidator 先把常见分类词映射成数值、能转的字符串转成 float，
#     实在解析不了就兜底 0.0。单个坏字段降级为一个默认值，不牵连整批更新。
# ---------------------------------------------------------------------------
_CATALOG_FLOAT = {
    "very high": 1.0, "veryhigh": 1.0, "high": 1.0, "strong": 1.0,
    "强": 1.0, "很高": 1.0, "高": 1.0,
    "medium": 0.5, "moderate": 0.5, "mid": 0.5, "normal": 0.5,
    "中": 0.5, "中等": 0.5,
    "low": 0.2, "weak": 0.2, "弱": 0.2, "低": 0.2,
    "none": 0.0, "无": 0.0, "": 0.0,
}


def _coerce_float(v: Any) -> float:
    """把任意输入尽量转成 float；转不动就返回 0.0（绝不抛异常）。"""
    if isinstance(v, bool):           # bool 是 int 子类，先挡掉，免得 True->1.0 语义怪
        return 1.0 if v else 0.0
    if isinstance(v, (int, float)):
        return float(v)
    if isinstance(v, str):
        s = v.strip()
        low = s.lower()
        if low in _CATALOG_FLOAT:
            return _CATALOG_FLOAT[low]
        if s in _CATALOG_FLOAT:       # 中文键不在 lower() 影响范围，直接查原串
            return _CATALOG_FLOAT[s]
        try:
            return float(s)
        except ValueError:
            return 0.0
    return 0.0


LenientFloat = Annotated[float, BeforeValidator(_coerce_float)]


# ---------------------------------------------------------------------------
# 通用记忆条目：用于 interests / dislikes（prompt 没给它们专门的模型）
# ---------------------------------------------------------------------------
class MemoryItem(_Base):
    id: str = ""
    value: Any = None
    confidence: LenientFloat = 0.0
    source: str = "agent_inference"   # user_explicit | user_behavior | agent_inference
    status: str = "active"            # active | superseded | deprecated
    first_seen_at: str = ""
    last_seen_at: str = ""
    update_count: int = 0
    evidence_refs: list[str] = Field(default_factory=list)
    tags: list[str] = Field(default_factory=list)


class UserProfile(_Base):
    user_id: str = ""
    preferred_name: Optional[str] = None
    language: Optional[str] = None
    timezone: Optional[str] = None
    occupation: Optional[str] = None
    expertise_level: dict[str, LenientFloat] = Field(default_factory=dict)


class Preference(_Base):
    subject: str = ""
    preference: LenientFloat = 0.0    # -1.0 强烈不喜欢 ~ +1.0 强烈喜欢
    strength: LenientFloat = 0.0
    context: Optional[str] = None


class DecisionCriterion(_Base):
    name: str = ""
    weight: LenientFloat = 0.0
    context: str = ""
    evidence_refs: list[str] = Field(default_factory=list)


class DecisionRule(_Base):
    context: str = ""
    condition: str = ""
    reasoning_steps: list[str] = Field(default_factory=list)
    conclusion_pattern: str = ""
    confidence: LenientFloat = 0.0
    exceptions: list[str] = Field(default_factory=list)
    evidence_refs: list[str] = Field(default_factory=list)


class Belief(_Base):
    topic: str = ""
    proposition: str = ""
    confidence: LenientFloat = 0.0
    last_updated_at: str = ""
    evidence_refs: list[str] = Field(default_factory=list)


class Habit(_Base):
    context: str = ""
    behavior: str = ""
    frequency: LenientFloat = 0.0
    confidence: LenientFloat = 0.0
    evidence_refs: list[str] = Field(default_factory=list)


class BeliefUpdatePattern(_Base):
    topic: str = ""
    current_position: str = ""
    trigger: str = ""
    update_direction: str = ""
    probability: LenientFloat = 0.0
    evidence_refs: list[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# 聚合：一个用户的完整长期记忆
# ---------------------------------------------------------------------------
class UserMemory(_Base):
    user_id: str = ""
    profile: UserProfile = Field(default_factory=UserProfile)

    interests: list[MemoryItem] = Field(default_factory=list)
    preferences: list[Preference] = Field(default_factory=list)
    decision_criteria: list[DecisionCriterion] = Field(default_factory=list)
    decision_rules: list[DecisionRule] = Field(default_factory=list)
    beliefs: list[Belief] = Field(default_factory=list)
    habits: list[Habit] = Field(default_factory=list)
    belief_update_patterns: list[BeliefUpdatePattern] = Field(default_factory=list)
    dislikes: list[MemoryItem] = Field(default_factory=list)

    updated_at: str = ""

    @classmethod
    def empty(cls, user_id: str) -> "UserMemory":
        """新用户的初始记忆：全空，但身份和时间戳已就位。"""
        return cls(user_id=user_id,
                   profile=UserProfile(user_id=user_id),
                   updated_at=now_iso())


# ---------------------------------------------------------------------------
# 管理器 I/O 契约
# ---------------------------------------------------------------------------
class Interaction(_Base):
    """一次交互的输入。字段对齐 prompt 的 Input 段。"""
    interaction_id: str = ""
    session_id: str = ""      # 这条交互来自哪一次会话（溯源用；记忆仍按 user 聚合）
    timestamp: str = ""
    user_message: str = ""
    agent_message: str = ""
    user_feedback: str = ""
    user_choice: Any = None


class MemoryChange(_Base):
    """管理器报告的单条变更，供上层审计/日志用。"""
    action: str = "NO_CHANGE"   # ADD | UPDATE | MERGE | SUPERSEDE | NO_CHANGE
    memory_type: str = ""
    memory_id: str = ""
    reason: str = ""


class ManagerOutput(_Base):
    """管理器的最终输出：更新后的完整记忆 + 这次做了哪些变更。"""
    updated_memory: UserMemory = Field(default_factory=UserMemory)
    changes: list[MemoryChange] = Field(default_factory=list)

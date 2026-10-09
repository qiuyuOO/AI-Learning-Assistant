"""长期用户记忆包。

对外只暴露这几个名字（和 adapters 包一样的做法）：
    UserMemory / Interaction / ManagerOutput / MemoryChange  —— 数据结构
    MemoryStore   —— 持久化
    MemoryManager —— LLM 驱动的更新流水线

★ flat import：在 Ai/ 目录下 `from memory import MemoryManager, MemoryStore`。
"""
from memory.schema import (BeliefUpdatePattern, Belief, DecisionCriterion,
                           DecisionRule, Habit, Interaction, ManagerOutput,
                           MemoryChange, MemoryItem, Preference, UserMemory,
                           UserProfile, now_iso)
from memory.store import MemoryStore, get_default_store, DEFAULT_MEMORY_DIR
from memory.manager import (DEFAULT_UPDATE_EVERY, MemoryManager,
                            get_default_manager, render_memory_context)

__all__ = [
    "UserMemory", "UserProfile", "MemoryItem", "Preference",
    "DecisionCriterion", "DecisionRule", "Belief", "Habit",
    "BeliefUpdatePattern", "Interaction", "MemoryChange", "ManagerOutput",
    "MemoryStore", "get_default_store", "DEFAULT_MEMORY_DIR",
    "MemoryManager", "get_default_manager", "render_memory_context", "now_iso",
    "DEFAULT_UPDATE_EVERY",
]

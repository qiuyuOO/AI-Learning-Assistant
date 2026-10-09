"""模型静态规格表。

★ 窗口是模型规格，不随部署变 —— 和价格表不同（价格会按合同变）。
  所以它属于 runtime 可用，而价格属于离线聚合。

★ 它不进 adapter：adapter 是无状态的格式翻译器，
  而"预算"是编排决策，依赖上一轮实测 usage。
"""

_SPECS: dict[str, int] = {
    "deepseek-flash": 1_000_000,
    "deepseek-v4-pro": 1_000_000,
}

# 查不到时保守取值：宁可少输出，也不要 400
_DEFAULT_WINDOW = 32_000


def resolve_context_window(model: str, override: int | None = None) -> int:
    """三级来源，优先级从高到低：

    1. 显式 override（客户给了非公开模型时必须能覆盖）
    2. 内置规格表
    3. 保守默认值

    这种"精确来源 → 近似来源 → 安全兜底"是配置解析的通用形状，
    和 max_tokens 预算、异常处理、usage_sink 降级是同一个模式。
    """
    if override:
        return override
    return _SPECS.get(model, _DEFAULT_WINDOW)
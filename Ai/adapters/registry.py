"""provider 注册表。

依赖方向必须单向，否则循环导入：
    adapters/__init__.py → 具体 adapter → registry → base → llm_types

★ 这个文件不 import 任何具体 adapter。
  "触发注册"由 adapters/__init__.py 负责。
"""
from typing import Type

from adapters.base import ProviderAdapter

_REGISTRY: dict[str, Type[ProviderAdapter]] = {}


def register(cls: Type[ProviderAdapter]) -> Type[ProviderAdapter]:
    """装饰器注册。

    key 取自类上的 name 属性，不是从类名推导 —— 否则用户 API 会
    变成 "deepseekanthropicadapter" 这种由内部类名决定的字符串。
    """
    key = getattr(cls, "name", "")
    if not key:
        raise ValueError(f"{cls.__name__} 必须定义 name 属性")
    if key in _REGISTRY:
        raise ValueError(f"provider 重复注册: {key!r}（{cls.__name__}）")
    _REGISTRY[key] = cls
    return cls


def get_adapter_class(provider: str) -> Type[ProviderAdapter]:
    try:
        return _REGISTRY[provider.lower()]
    except KeyError:
        raise ValueError(
            f"未知 provider: {provider!r}，已注册: {sorted(_REGISTRY)}"
        ) from None


def available_providers() -> list[str]:
    return sorted(_REGISTRY)

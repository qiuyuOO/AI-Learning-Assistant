from abc import ABC, abstractmethod
from typing import Any

from llm_types import LLMResponse, Message, ToolSpec


class ProviderAdapter(ABC):
    """所有 provider 适配器的契约。

    新增一个 provider = 新增一个子类 + 在 registry 里注册一行。
    不允许在 client 或 agent loop 里出现 if provider == "..." 这种分支。
    """

    name: str          # provider 标识，注册表用它做 key

    def __init__(self, api_key: str, **config: Any):
        self.api_key = api_key
        self.config = config

    @abstractmethod
    def to_request(self, model: str,messages: list[Message], system: str | None = None,
                   tools: list[ToolSpec] | None = None,
                   max_tokens: int | None = None) -> dict:
        """canonical 入参 → provider 原生请求参数。"""

    @abstractmethod
    def to_native_messages(self, messages: list[Message]) -> list:
        """canonical 消息 → provider 原生消息数组。"""

    @abstractmethod
    def to_native_tools(self, tools: list[ToolSpec]) -> list:
        """canonical 工具声明 → provider 原生工具格式。"""

    @abstractmethod
    def to_canonical(self, raw) -> LLMResponse:
        """provider 原始响应 → LLMResponse。"""

    @abstractmethod
    def call(self, **native_kwargs) -> LLMResponse:
        """发一次请求。异常向上抛，由 retry 层接管。"""

    @classmethod
    def create(cls, api_key: str, **config: Any) -> "ProviderAdapter":
        """统一的构造入口。

        ★ 默认实现对所有 provider 通用。
          子类只有在"需要校验/补默认值/字段名不同"时才覆盖。

        之所以是 classmethod 而不是 staticmethod：
        cls 让这一个方法自动适配所有子类。
        """
        return cls(api_key=api_key, **config)
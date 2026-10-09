class DomainError(Exception):
    """所有业务领域异常的基类。

        类属性 = 类型级的默认（子类覆盖）：
          content       默认话术
          retry_budget  允许重试几次；0 = 不重试
        两者都能被构造参数覆盖（表驱动映射靠这个）。
        """

    content: str = "模型调用失败。"
    retry_budget: int = 0          # ★ 保守默认：没声明的类型不重试

    def __init__(self, content: str | None = None,
                 retry_budget: int | None = None):
        self.content = content or type(self).content
        self.retry_budget = (retry_budget if retry_budget is not None
                             else type(self).retry_budget)
        super().__init__(self.content)

class RateLimited(DomainError):
    content = "当前请求量过大，服务繁忙，请稍后重试。"
    retry_budget =  4


class ModelUnavailable(DomainError):
    content = "模型服务暂时不可用，请稍后重试或转人工。"
    retry_budget = 2

class AuthenticationFailed(DomainError):
    content = "模型服务认证失败（API Key 无效或过期），请联系管理员。"

class BadRequest(DomainError):
    content = "请求参数不合法。"


class ContextLengthExceeded(DomainError):
    content = "对话上下文超出模型窗口，需要压缩。"

_STATUS_TO_DOMAIN: dict[int, tuple[type[DomainError], str , int]] = {
    401: (AuthenticationFailed, None , None),
    402: (BadRequest, "模型服务账户余额不足，请联系管理员充值。" , None),
    403: (AuthenticationFailed, None , None),
    413: (ContextLengthExceeded, None , None),
    429: (RateLimited, None , None),
}
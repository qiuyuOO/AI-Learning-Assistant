"""注入真实的 SDK 异常类型，让生产代码的 except 完全不用改。

用法：
    from fake_api_error import make_rate_limit_error
    raise make_rate_limit_error(429, "模拟限流")
"""
import httpx2
import anthropic

_FAKE_REQUEST_URL = "https://api.deepseek.com/anthropic/v1/messages"

_STATUS_TO_EXC = {
    400: anthropic.BadRequestError,
    401: anthropic.AuthenticationError,
    403: anthropic.PermissionDeniedError,
    404: anthropic.NotFoundError,
    413: anthropic.RequestTooLargeError,
    422: anthropic.UnprocessableEntityError,
    429: anthropic.RateLimitError,
    500: anthropic.InternalServerError,
    503: anthropic.ServiceUnavailableError,
    529: anthropic.OverloadedError,
}


def make_status_error(status_code: int, message: str = "模拟错误"):
    """按状态码造出对应的真实 anthropic 异常实例（不发任何网络请求）。"""
    exc_cls = _STATUS_TO_EXC.get(status_code, anthropic.APIStatusError)
    request = httpx2.Request("POST", _FAKE_REQUEST_URL)
    response = httpx2.Response(status_code, request=request)
    return exc_cls(
        message,
        response=response,
        body={"error": {"type": "fake_error", "message": message}},
    )


def make_timeout_error():
    """造 APITimeoutError（注意它的签名只要 request，不要 message）。"""
    request = httpx2.Request("POST", _FAKE_REQUEST_URL)
    return anthropic.APITimeoutError(request=request)

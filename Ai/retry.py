
import logging
import time
from random import random
from error_type import DomainError
log = logging.getLogger("retry")

INITIAL_DELAY = 0.5
MAX_DELAY = 8


def calculate_backoff(attempt: int) -> float:
    base = min(INITIAL_DELAY * 2 ** attempt, MAX_DELAY)
    jitter = 1 - 0.25 * random()
    return base * jitter


def call_with_retry(call_fn):
    """call_fn: 无参可调用对象。成功返回结果，失败抛异常。"""
    attempt = 0
    while True:
        try:
            return call_fn()
        except DomainError as e:

            if e.retry_budget <= 0:
                log.error("不可重试 | %s: %s", type(e).__name__, e)
                raise

            if attempt >= e.retry_budget:
                log.error("重试次数达到上限（%s 次），放弃 %s 异常信息：%s", e.retry_budget , type(e).__name__,e.content)
                raise
            attempt += 1
            delay = calculate_backoff(attempt)
            log.info("第%s次重试 | 等%.3fs | 原因%s", attempt, delay, type(e).__name__)
            time.sleep(delay)







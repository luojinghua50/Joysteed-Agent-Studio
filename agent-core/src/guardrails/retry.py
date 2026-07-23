import asyncio
from enum import Enum
from typing import Any, Callable, Coroutine


class RetryCategory(Enum):
    RETRYABLE = "retryable"
    NON_RETRYABLE = "non_retryable"
    FALLBACK = "fallback"


def classify_error(error: Exception) -> RetryCategory:
    """Classify an error into retry categories."""
    error_name = type(error).__name__

    retryable_types = ("TimeoutError", "ConnectionError", "ServerError", "RateLimitError")
    # 结构化输出解析/校验失败：re-roll 常能修复，故归为可重试（由 output_parse_retries 控制次数）。
    # 按类型名匹配而非 isinstance，避免把普通 ValueError（ValidationError 的父类）也卷进来。
    parse_error_types = ("OutputParserException", "ValidationError")
    fallback_types = ("ModelUnavailableError",)

    if error_name in retryable_types or isinstance(error, (TimeoutError, ConnectionError)):
        return RetryCategory.RETRYABLE
    if error_name in parse_error_types:
        return RetryCategory.RETRYABLE
    if error_name in fallback_types:
        return RetryCategory.FALLBACK
    return RetryCategory.NON_RETRYABLE


async def retry_with_backoff(
    fn: Callable[[], Coroutine],
    max_retries: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 10.0,
    fallback_fn: Callable[[], Coroutine] | None = None,
) -> Any:
    """Execute with exponential backoff retry."""
    last_error: Exception | None = None

    for attempt in range(max_retries + 1):
        try:
            return await fn()
        except Exception as e:
            last_error = e
            category = classify_error(e)

            if category == RetryCategory.NON_RETRYABLE:
                raise
            if category == RetryCategory.FALLBACK and fallback_fn:
                return await fallback_fn()

            if attempt < max_retries:
                delay = min(base_delay * (2 ** attempt), max_delay)
                await asyncio.sleep(delay)

    raise last_error  # type: ignore

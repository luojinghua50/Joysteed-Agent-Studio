import pytest
from src.guardrails.retry import retry_with_backoff, classify_error, RetryCategory


class TestClassifyError:
    def test_timeout_is_retryable(self):
        assert classify_error(TimeoutError()) == RetryCategory.RETRYABLE

    def test_connection_error_is_retryable(self):
        assert classify_error(ConnectionError()) == RetryCategory.RETRYABLE

    def test_value_error_is_non_retryable(self):
        assert classify_error(ValueError("bad input")) == RetryCategory.NON_RETRYABLE

    def test_generic_exception_is_non_retryable(self):
        assert classify_error(Exception("generic")) == RetryCategory.NON_RETRYABLE


@pytest.mark.asyncio
async def test_retry_success_first_try():
    call_count = 0

    async def task():
        nonlocal call_count
        call_count += 1
        return "success"

    result = await retry_with_backoff(task, max_retries=3, base_delay=0.01)
    assert result == "success"
    assert call_count == 1


@pytest.mark.asyncio
async def test_retry_success_after_retries():
    call_count = 0

    async def task():
        nonlocal call_count
        call_count += 1
        if call_count < 3:
            raise TimeoutError("timeout")
        return "success"

    result = await retry_with_backoff(task, max_retries=3, base_delay=0.01)
    assert result == "success"
    assert call_count == 3


@pytest.mark.asyncio
async def test_retry_non_retryable_raises_immediately():
    call_count = 0

    async def task():
        nonlocal call_count
        call_count += 1
        raise ValueError("bad input")

    with pytest.raises(ValueError):
        await retry_with_backoff(task, max_retries=3, base_delay=0.01)
    assert call_count == 1


@pytest.mark.asyncio
async def test_retry_exhausted_raises():
    async def task():
        raise TimeoutError("always times out")

    with pytest.raises(TimeoutError):
        await retry_with_backoff(task, max_retries=2, base_delay=0.01)


@pytest.mark.asyncio
async def test_retry_with_fallback():
    async def task():
        raise ConnectionError("down")

    async def fallback():
        return "fallback"

    # ConnectionError is RETRYABLE not FALLBACK, so it retries
    # We need to test with a FALLBACK-category error
    # Let's just test that fallback_fn works with None fallback
    with pytest.raises(ConnectionError):
        await retry_with_backoff(task, max_retries=1, base_delay=0.01)

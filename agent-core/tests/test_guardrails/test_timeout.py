import pytest
import asyncio
from src.guardrails.timeout import with_timeout, AgentTimeoutError


@pytest.mark.asyncio
async def test_with_timeout_succeeds():
    async def quick_task():
        return "done"

    result = await with_timeout(quick_task(), timeout=1.0)
    assert result == "done"


@pytest.mark.asyncio
async def test_with_timeout_raises_on_timeout():
    async def slow_task():
        await asyncio.sleep(10)
        return "done"

    with pytest.raises(AgentTimeoutError):
        await with_timeout(slow_task(), timeout=0.1)


@pytest.mark.asyncio
async def test_with_timeout_uses_fallback():
    async def slow_task():
        await asyncio.sleep(10)
        return "done"

    async def fallback():
        return "fallback_result"

    result = await with_timeout(slow_task(), timeout=0.1, fallback_fn=fallback)
    assert result == "fallback_result"

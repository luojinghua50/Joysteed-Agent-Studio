import asyncio
from typing import Any, Callable, Coroutine


class AgentTimeoutError(Exception):
    """Raised when an agent operation times out."""
    pass


async def with_timeout(
    coro: Coroutine,
    timeout: float,
    fallback_fn: Callable[[], Coroutine] | None = None,
) -> Any:
    """Execute a coroutine with a timeout, optionally falling back."""
    try:
        return await asyncio.wait_for(coro, timeout=timeout)
    except asyncio.TimeoutError:
        if fallback_fn:
            return await fallback_fn()
        raise AgentTimeoutError(f"Operation timed out after {timeout}s")

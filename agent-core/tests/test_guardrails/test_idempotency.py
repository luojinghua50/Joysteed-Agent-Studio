import pytest
from src.guardrails.idempotency import IdempotencyGuard
from src.config import StabilityConfig


@pytest.mark.asyncio
async def test_non_protected_tool_not_checked():
    guard = IdempotencyGuard()
    is_dup, cached = await guard.check_and_mark("s1", "query_order", {"order_id": "ORD-001"})
    assert is_dup is False
    assert cached is None


@pytest.mark.asyncio
async def test_protected_tool_first_call():
    guard = IdempotencyGuard()
    is_dup, cached = await guard.check_and_mark(
        "s1", "apply_refund", {"order_id": "ORD-001", "amount": 100}
    )
    assert is_dup is False
    assert cached is None


@pytest.mark.asyncio
async def test_protected_tool_duplicate_detected():
    guard = IdempotencyGuard()
    args = {"order_id": "ORD-001", "amount": 100}

    await guard.store_result("s1", "apply_refund", args, {"refund_id": "RF-123"})

    is_dup, cached = await guard.check_and_mark("s1", "apply_refund", args)
    assert is_dup is True
    assert cached == {"refund_id": "RF-123"}


@pytest.mark.asyncio
async def test_different_args_not_duplicate():
    guard = IdempotencyGuard()
    args1 = {"order_id": "ORD-001", "amount": 100}
    args2 = {"order_id": "ORD-001", "amount": 200}

    await guard.store_result("s1", "apply_refund", args1, {"refund_id": "RF-123"})

    is_dup, cached = await guard.check_and_mark("s1", "apply_refund", args2)
    assert is_dup is False


@pytest.mark.asyncio
async def test_disabled_always_allows():
    config = StabilityConfig(idempotency_enabled=False)
    guard = IdempotencyGuard(config=config)

    await guard.store_result("s1", "apply_refund", {"order_id": "ORD-001"}, {"result": "ok"})
    is_dup, cached = await guard.check_and_mark("s1", "apply_refund", {"order_id": "ORD-001"})
    assert is_dup is False

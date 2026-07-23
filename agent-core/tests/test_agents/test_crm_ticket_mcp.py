import pytest
import pytest_asyncio
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", "agent-tools"))

from crm_server.server import get_customer_info, update_customer_tag, get_customer_history
from ticket_server.server import create_ticket, query_ticket, update_ticket, apply_compensation


@pytest_asyncio.fixture(scope="module", loop_scope="module")
async def ticket_db():
    """守卫：ticket 工单工具依赖真实 PostgreSQL（进程内直连），DB 不可达就 skip。

    默认连接串用的是 compose 内部主机名 `postgres`，在宿主机裸跑 pytest 时该
    名字解析不到数据库、建连会挂死成 Timeout。本 fixture 用短超时探一次，连不上
    则把这些集成测试标记为 skipped（而非 5 分钟超时失败）。本机要真跑，设
    TICKET_DATABASE_URL=postgresql+asyncpg://postgres:postgres@localhost:5432/agent_core。
    """
    from ticket_server.store import TicketStore
    from ticket_server.server import store as _server_store

    probe = TicketStore(os.getenv(
        "TICKET_DATABASE_URL",
        "postgresql+asyncpg://postgres:postgres@postgres:5432/agent_core",
    ))
    try:
        async with probe._engine.connect():
            pass
    except Exception as e:
        pytest.skip(f"ticket DB 不可达，跳过集成测试: {type(e).__name__}: {str(e)[:120]}")
    finally:
        await probe._engine.dispose()
    return _server_store


@pytest.mark.asyncio
async def test_get_customer_info_exists():
    result = await get_customer_info("C001")
    assert result["customer_id"] == "C001"
    assert result["name"] == "张三"
    assert result["vip_level"] == 2


@pytest.mark.asyncio
async def test_get_customer_info_not_exists():
    result = await get_customer_info("C999")
    assert "error" in result


@pytest.mark.asyncio
async def test_update_customer_tag_add():
    result = await update_customer_tag("C002", "活跃用户", "add")
    assert result["success"] is True


@pytest.mark.asyncio
async def test_update_customer_tag_remove():
    result = await update_customer_tag("C002", "新用户", "remove")
    assert result["success"] is True


@pytest.mark.asyncio
async def test_get_customer_history():
    result = await get_customer_history("C001")
    assert result["customer_id"] == "C001"
    assert len(result["history"]) > 0


@pytest.mark.asyncio(loop_scope="module")
async def test_create_ticket(ticket_db):
    result = await create_ticket("C001", "测试工单", "这是一个测试", "high")
    assert "ticket_id" in result
    assert result["ticket_id"].startswith("TK-")


@pytest.mark.asyncio(loop_scope="module")
async def test_query_ticket_exists(ticket_db):
    result = await query_ticket("TK-001")
    assert result["ticket_id"] == "TK-001"
    assert result["status"] == "open"


@pytest.mark.asyncio(loop_scope="module")
async def test_query_ticket_not_exists(ticket_db):
    result = await query_ticket("TK-999")
    assert "error" in result


@pytest.mark.asyncio(loop_scope="module")
async def test_update_ticket_status(ticket_db):
    result = await update_ticket("TK-001", status="in_progress")
    assert result["success"] is True


@pytest.mark.asyncio
async def test_update_ticket_invalid_status():
    result = await update_ticket("TK-001", status="invalid_status")
    assert "error" in result


@pytest.mark.asyncio
async def test_apply_compensation_within_limit():
    result = await apply_compensation("C001", "ORD-001", 100.0, "配送延迟")
    assert result["status"] == "approved"
    assert result["amount"] == 100.0


@pytest.mark.asyncio
async def test_apply_compensation_exceeds_limit():
    result = await apply_compensation("C001", "ORD-001", 600.0, "配送延迟")
    assert "error" in result
    assert "上限" in result["error"]

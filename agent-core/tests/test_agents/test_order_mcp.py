import pytest
import pytest_asyncio
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", "agent-tools"))

from order_server.server import (
    query_order, apply_refund, check_refund_eligibility,
    track_shipping, modify_order, urge_shipping,
)


@pytest_asyncio.fixture(scope="module", loop_scope="module")
async def order_db():
    """守卫：order 工具已落库（进程内直连 PG），DB 不可达就 skip。

    默认连接串用 compose 内部主机名 `postgres`，宿主机裸跑 pytest 解析不到 →
    asyncpg 建连挂死成 Timeout。短超时探一次，连不上则 skip（仿 ticket 测试）。
    本机真跑设 ORDER_DATABASE_URL=...@localhost:5432/agent_core。
    模块级事件循环：order_server.store 是单例 store，连接池绑事件循环，跨循环
    复用会触发 asyncpg `another operation is in progress`。
    """
    from order_server.store import OrderStore
    from order_server.server import store as _server_store

    probe = OrderStore(os.getenv(
        "ORDER_DATABASE_URL",
        "postgresql+asyncpg://postgres:postgres@postgres:5432/agent_core",
    ))
    try:
        async with probe._engine.connect():
            pass
    except Exception as e:
        pytest.skip(f"order DB 不可达，跳过集成测试: {type(e).__name__}: {str(e)[:120]}")
    finally:
        await probe._engine.dispose()
    return _server_store


@pytest.mark.asyncio(loop_scope="module")
async def test_query_order_exists(order_db):
    result = await query_order("ORD-001")
    assert result["order_id"] == "ORD-001"
    assert result["status"] == "shipped"
    assert result["amount"] == 299.00


@pytest.mark.asyncio(loop_scope="module")
async def test_query_order_not_exists(order_db):
    result = await query_order("ORD-999")
    assert "error" in result


@pytest.mark.asyncio(loop_scope="module")
async def test_refund_flow_on_ord002(order_db):
    """退款状态机完整闭环（自包含于 ORD-002，不污染其它测试读的订单）。

    覆盖：7天无理由资格（验证相对时间 seed 修掉穿帮）→ 部分退款 →
    累计超额拒绝 → 补足全额转 refunded → 全额后锁定再退被拒。
    ORD-002: delivered（2天前）, amount 1599。
    """
    # 落库后写操作持久，本测试会把 ORD-002 改成 refunded → 必须先重置成干净
    # 状态，否则二次运行（表已 seeded）首断言即挂。删退款记录 + 还原 delivered。
    from order_server.store import OrderModel, RefundModel
    from sqlalchemy import delete
    async with order_db._session() as db:
        await db.execute(delete(RefundModel).where(RefundModel.order_id == "ORD-002"))
        o = await db.get(OrderModel, "ORD-002")
        o.status = "delivered"
        await db.commit()

    # 1) 7天无理由期内可退（相对时间 seed 后此分支可命中，不再穿帮）
    elig = await check_refund_eligibility("ORD-002")
    assert elig["eligible"] is True
    assert "7天" in elig["reason"]

    # 2) 部分退款 100 → 落库 completed，订单转 partial_refunded
    r1 = await apply_refund("ORD-002", 100.0, "部分退款")
    assert "refund_id" in r1
    assert r1["status"] == "completed"
    assert r1["amount"] == 100.0
    assert r1["order_status"] == "partial_refunded"

    # 3) 累计超额：已退 100 + 2000 > 1599 → 拒绝，不写库
    r2 = await apply_refund("ORD-002", 2000.0, "超额")
    assert "error" in r2
    assert "超过" in r2["error"]

    # 4) 补足剩余 1499 → 累计 1599 = 订单金额 → 全额 refunded
    r3 = await apply_refund("ORD-002", 1499.0, "补足全额")
    assert r3["status"] == "completed"
    assert r3["order_status"] == "refunded"

    # 5) 已全额退款后再退 → 资格判定拒绝
    r4 = await apply_refund("ORD-002", 10.0, "重复退款")
    assert "error" in r4


@pytest.mark.asyncio(loop_scope="module")
async def test_apply_refund_exceeds_amount(order_db):
    # ORD-001 shipped(299)，单次 999 > 299 → 拒绝（不写库，ORD-001 保持 shipped）
    result = await apply_refund("ORD-001", 999.0, "退款")
    assert "error" in result
    assert "超过" in result["error"]


@pytest.mark.asyncio(loop_scope="module")
async def test_check_refund_eligibility_pending(order_db):
    result = await check_refund_eligibility("ORD-003")
    assert result["eligible"] is True
    assert "未发货" in result["reason"]


@pytest.mark.asyncio(loop_scope="module")
async def test_check_refund_eligibility_not_exists(order_db):
    result = await check_refund_eligibility("ORD-999")
    assert result["eligible"] is False


@pytest.mark.asyncio(loop_scope="module")
async def test_track_shipping_with_data(order_db):
    result = await track_shipping("ORD-001")
    assert result["status"] == "in_transit"
    assert result["carrier"] == "顺丰速运"
    assert len(result["tracks"]) > 0


@pytest.mark.asyncio(loop_scope="module")
async def test_track_shipping_pending_order(order_db):
    result = await track_shipping("ORD-003")
    assert result["status"] == "not_shipped"


@pytest.mark.asyncio(loop_scope="module")
async def test_modify_order_pending(order_db):
    result = await modify_order("ORD-003", "shipping_address", "北京市朝阳区")
    assert result["success"] is True


@pytest.mark.asyncio(loop_scope="module")
async def test_modify_order_shipped_fails(order_db):
    result = await modify_order("ORD-001", "shipping_address", "新地址")
    assert "error" in result
    assert "已发货" in result["error"]


@pytest.mark.asyncio(loop_scope="module")
async def test_modify_order_invalid_field(order_db):
    result = await modify_order("ORD-003", "price", "100")
    assert "error" in result


@pytest.mark.asyncio(loop_scope="module")
async def test_urge_shipping_pending(order_db):
    result = await urge_shipping("ORD-003")
    assert result["success"] is True
    assert "加急" in result["message"]


@pytest.mark.asyncio(loop_scope="module")
async def test_urge_shipping_already_shipped(order_db):
    result = await urge_shipping("ORD-001")
    assert result["success"] is True
    assert "催促" in result["message"]

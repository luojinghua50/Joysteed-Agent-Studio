"""Order MCP Server — durable (PostgreSQL) via order_server.store.OrderStore.

Orders/shipping/refunds are persisted; the refund flow is a real state machine
(apply_refund checks eligibility + cumulative-amount cap, then transitions the
order to refunded/partial_refunded). Tool signatures and return shapes are
unchanged from the old mock version, so agent-core needs no changes.
"""
import os

from mcp.server.fastmcp import FastMCP

from order_server.store import OrderStore

mcp = FastMCP("order-service")

DATABASE_URL = os.getenv(
    "ORDER_DATABASE_URL",
    "postgresql+asyncpg://postgres:postgres@postgres:5432/agent_core",
)
store = OrderStore(DATABASE_URL)


@mcp.tool()
async def query_order(order_id: str) -> dict:
    """查询订单详情，包括状态、金额、商品信息。"""
    order = await store.get_order(order_id)
    if not order:
        return {"error": f"订单 {order_id} 不存在"}
    return order


@mcp.tool()
async def apply_refund(order_id: str, amount: float, reason: str) -> dict:
    """申请退款。校验资格与累计退款上限后落库，返回退款单号和到账时间。"""
    return await store.apply_refund(order_id, amount, reason)


@mcp.tool()
async def check_refund_eligibility(order_id: str) -> dict:
    """检查订单是否符合退款条件。"""
    return await store.check_eligibility(order_id)


@mcp.tool()
async def track_shipping(order_id: str) -> dict:
    """查询物流追踪信息。"""
    shipping = await store.get_shipping(order_id)
    if shipping:
        return shipping
    order = await store.get_order(order_id)
    if not order:
        return {"error": f"订单 {order_id} 不存在"}
    if order["status"] == "pending":
        return {"status": "not_shipped", "message": "订单尚未发货"}
    return {"status": "unknown", "message": "暂无物流信息"}


@mcp.tool()
async def modify_order(order_id: str, field: str, value: str) -> dict:
    """修改订单信息（仅限未发货订单）。"""
    return await store.modify_order(order_id, field, value)


@mcp.tool()
async def urge_shipping(order_id: str) -> dict:
    """催促物流发货/配送。"""
    return await store.urge_shipping(order_id)


if __name__ == "__main__":
    import uvicorn
    app = mcp.streamable_http_app()
    uvicorn.run(app, host="0.0.0.0", port=8002)

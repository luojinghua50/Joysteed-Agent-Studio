from src.mcp_client.client import MCPClientManager


async def query_order_detail(order_id: str, mcp: MCPClientManager | None = None) -> str:
    """查询订单完整信息，包括状态、物流、支付详情。"""
    if mcp is None:
        mcp = MCPClientManager()

    order = await mcp.call_tool("order", "query_order", {"order_id": order_id})
    shipping = await mcp.call_tool("order", "track_shipping", {"order_id": order_id})

    if order.get("error"):
        return f"查询订单失败: {order['error']}"

    shipping_info = ""
    if shipping and not shipping.get("error"):
        shipping_info = f"\n- 物流: {shipping.get('status', '未知')} - {shipping.get('location', '')}"

    return (
        f"订单 {order_id}：\n"
        f"- 状态：{order.get('status', '未知')}\n"
        f"- 金额：¥{order.get('amount', 0)}\n"
        f"- 下单时间：{order.get('created_at', '未知')}"
        f"{shipping_info}"
    )

from src.mcp_client.client import MCPClientManager


async def track_shipping_detail(order_id: str, mcp: MCPClientManager | None = None) -> str:
    """查询物流追踪信息。"""
    if mcp is None:
        mcp = MCPClientManager()

    shipping = await mcp.call_tool("order", "track_shipping", {"order_id": order_id})

    if shipping.get("error"):
        return f"查询物流失败: {shipping['error']}"

    tracks = shipping.get("tracks", [])
    if not tracks:
        return f"订单 {order_id} 暂无物流信息"

    result = f"订单 {order_id} 物流追踪：\n"
    result += f"- 状态：{shipping.get('status', '未知')}\n"
    result += f"- 承运商：{shipping.get('carrier', '未知')}\n"
    result += f"- 运单号：{shipping.get('tracking_number', '未知')}\n"
    result += "- 物流记录：\n"
    for track in tracks[-5:]:
        result += f"  [{track.get('time', '')}] {track.get('description', '')}\n"

    return result

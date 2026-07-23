"""CRM MCP Server with mock data."""
from datetime import datetime

from mcp.server.fastmcp import FastMCP

mcp = FastMCP("crm-service")

MOCK_CUSTOMERS = {
    "C001": {
        "customer_id": "C001",
        "name": "张三",
        "phone": "138****1234",
        "email": "zhang***@example.com",
        "vip_level": 2,
        "total_orders": 15,
        "total_spend": 8500.00,
        "registered_at": "2023-06-15",
        "tags": ["高价值", "电子产品爱好者"],
        "preferred_channel": "在线客服",
    },
    "C002": {
        "customer_id": "C002",
        "name": "李四",
        "phone": "139****5678",
        "email": "li***@example.com",
        "vip_level": 0,
        "total_orders": 3,
        "total_spend": 450.00,
        "registered_at": "2024-01-05",
        "tags": ["新用户"],
        "preferred_channel": "电话",
    },
    "C003": {
        "customer_id": "C003",
        "name": "王五",
        "phone": "137****9012",
        "email": "wang***@example.com",
        "vip_level": 3,
        "total_orders": 45,
        "total_spend": 32000.00,
        "registered_at": "2022-03-20",
        "tags": ["VIP", "高价值", "敏感客户"],
        "preferred_channel": "专属客服",
    },
}

MOCK_HISTORY = {
    "C001": [
        {"date": "2024-01-10", "type": "order", "summary": "购买智能手表，已签收"},
        {"date": "2024-01-05", "type": "inquiry", "summary": "咨询退换货政策"},
        {"date": "2023-12-20", "type": "complaint", "summary": "配送延迟，已补偿优惠券"},
    ],
    "C002": [
        {"date": "2024-01-20", "type": "order", "summary": "购买手机壳，待发货"},
    ],
}


@mcp.tool()
async def get_customer_info(customer_id: str) -> dict:
    """获取客户基本信息，包括VIP等级、消费记录等。"""
    customer = MOCK_CUSTOMERS.get(customer_id)
    if not customer:
        return {"error": f"客户 {customer_id} 不存在"}
    return customer


@mcp.tool()
async def update_customer_tag(customer_id: str, tag: str, action: str = "add") -> dict:
    """更新客户标签。action: add/remove。"""
    customer = MOCK_CUSTOMERS.get(customer_id)
    if not customer:
        return {"error": f"客户 {customer_id} 不存在"}

    if action == "add":
        if tag not in customer["tags"]:
            customer["tags"].append(tag)
        return {"success": True, "message": f"已添加标签: {tag}"}
    elif action == "remove":
        if tag in customer["tags"]:
            customer["tags"].remove(tag)
        return {"success": True, "message": f"已移除标签: {tag}"}
    else:
        return {"error": f"无效操作: {action}，可用: add/remove"}


@mcp.tool()
async def get_customer_history(customer_id: str, limit: int = 5) -> dict:
    """获取客户历史交互记录。"""
    history = MOCK_HISTORY.get(customer_id, [])
    return {
        "customer_id": customer_id,
        "history": history[:limit],
        "total": len(history),
    }


if __name__ == "__main__":
    import uvicorn
    app = mcp.streamable_http_app()
    uvicorn.run(app, host="0.0.0.0", port=8004)

"""Ticket MCP Server — durable (PostgreSQL) + REST API for the agent-desk workbench.

Two faces over one store (ticket_server.store.TicketStore):
- MCP tools (JSONRPC) — used by the LLM agents in agent-core.
- REST API (/api/tickets/*) — used by the agent-desk browser UI.
Both share the same persistence, so a ticket created by the AI shows up in the
seat workbench and vice versa.
"""
import os

from mcp.server.fastmcp import FastMCP
from starlette.requests import Request
from starlette.responses import JSONResponse

from ticket_server.store import TicketStore, VALID_STATUSES, VALID_PRIORITIES, AGENT_POOL

mcp = FastMCP("ticket-service")

DATABASE_URL = os.getenv(
    "TICKET_DATABASE_URL",
    "postgresql+asyncpg://postgres:postgres@postgres:5432/agent_core",
)
store = TicketStore(DATABASE_URL)


# ===== MCP tools (for LLM agents) =====

@mcp.tool()
async def create_ticket(
    customer_id: str, title: str, description: str, priority: str = "medium"
) -> dict:
    """创建工单。返回工单号。"""
    t = await store.create(customer_id, title, description, priority)
    return {"ticket_id": t["ticket_id"], "message": f"工单已创建，编号: {t['ticket_id']}"}


@mcp.tool()
async def query_ticket(ticket_id: str) -> dict:
    """查询工单详情。"""
    t = await store.get(ticket_id)
    return t or {"error": f"工单 {ticket_id} 不存在"}


@mcp.tool()
async def list_tickets(
    status: str | None = None, assigned_to: str | None = None, priority: str | None = None
) -> dict:
    """列出工单，可按状态/坐席/优先级筛选。"""
    tickets = await store.list(status, assigned_to, priority)
    return {"tickets": tickets, "total": len(tickets)}


@mcp.tool()
async def update_ticket(ticket_id: str, status: str | None = None, note: str | None = None) -> dict:
    """更新工单状态。note 会作为一条系统评论留痕。"""
    if status and status not in VALID_STATUSES:
        return {"error": f"无效状态: {status}，可用: {VALID_STATUSES}"}
    t = await store.update(ticket_id, status)
    if not t:
        return {"error": f"工单 {ticket_id} 不存在"}
    if note:
        await store.add_comment(ticket_id, "system", note)
    return {"success": True, "message": f"工单 {ticket_id} 已更新"}


@mcp.tool()
async def reassign_ticket(ticket_id: str, new_agent: str) -> dict:
    """转派工单给其他坐席。"""
    t = await store.reassign(ticket_id, new_agent)
    if not t:
        return {"error": f"工单 {ticket_id} 不存在"}
    return {"success": True, "message": f"工单 {ticket_id} 已转派给 {new_agent}"}


@mcp.tool()
async def add_ticket_comment(ticket_id: str, author: str, comment: str) -> dict:
    """给工单添加评论/处理记录。"""
    c = await store.add_comment(ticket_id, author, comment)
    if not c:
        return {"error": f"工单 {ticket_id} 不存在"}
    return {"success": True, "comment_id": c["id"]}


@mcp.tool()
async def apply_compensation(
    customer_id: str, order_id: str, amount: float, reason: str
) -> dict:
    """申请赔偿/补偿。"""
    if amount > 500:
        return {"error": "赔偿金额超过上限(500元)，需要主管审批"}
    import random
    comp_id = f"COMP-{random.randint(10000, 99999)}"
    return {
        "compensation_id": comp_id, "customer_id": customer_id, "order_id": order_id,
        "amount": amount, "status": "approved",
        "message": f"赔偿已审批通过，¥{amount} 将以优惠券形式发放",
    }


# ===== REST API (for agent-desk browser UI) =====
# FastMCP returns a Starlette app; custom_route mounts plain HTTP routes alongside
# the MCP transport. CORS is handled by the nginx proxy in agent-desk.

@mcp.custom_route("/api/health", methods=["GET"])
async def rest_health(_request: Request) -> JSONResponse:
    return JSONResponse({"status": "ok", "service": "ticket-mcp"})


@mcp.custom_route("/api/agents", methods=["GET"])
async def rest_agents(_request: Request) -> JSONResponse:
    return JSONResponse({"agents": AGENT_POOL})


@mcp.custom_route("/api/tickets", methods=["GET"])
async def rest_list_tickets(request: Request) -> JSONResponse:
    q = request.query_params
    tickets = await store.list(
        status=q.get("status"), assigned_to=q.get("assigned_to"), priority=q.get("priority")
    )
    return JSONResponse({"tickets": tickets, "total": len(tickets)})


@mcp.custom_route("/api/tickets", methods=["POST"])
async def rest_create_ticket(request: Request) -> JSONResponse:
    body = await request.json()
    if not body.get("title") or not body.get("customer_id"):
        return JSONResponse({"error": "customer_id 和 title 必填"}, status_code=400)
    priority = body.get("priority", "medium")
    if priority not in VALID_PRIORITIES:
        return JSONResponse({"error": f"无效优先级: {priority}"}, status_code=400)
    t = await store.create(
        body["customer_id"], body["title"], body.get("description", ""),
        priority, body.get("assigned_to"),
    )
    return JSONResponse(t, status_code=201)


@mcp.custom_route("/api/tickets/{ticket_id}", methods=["GET"])
async def rest_get_ticket(request: Request) -> JSONResponse:
    ticket_id = request.path_params["ticket_id"]
    t = await store.get(ticket_id)
    if not t:
        return JSONResponse({"error": "工单不存在"}, status_code=404)
    t["comments"] = await store.list_comments(ticket_id)
    return JSONResponse(t)


@mcp.custom_route("/api/tickets/{ticket_id}/status", methods=["POST"])
async def rest_update_status(request: Request) -> JSONResponse:
    ticket_id = request.path_params["ticket_id"]
    body = await request.json()
    status = body.get("status")
    if status not in VALID_STATUSES:
        return JSONResponse({"error": f"无效状态: {status}，可用: {VALID_STATUSES}"}, status_code=400)
    t = await store.update(ticket_id, status)
    if not t:
        return JSONResponse({"error": "工单不存在"}, status_code=404)
    return JSONResponse(t)


@mcp.custom_route("/api/tickets/{ticket_id}/reassign", methods=["POST"])
async def rest_reassign(request: Request) -> JSONResponse:
    ticket_id = request.path_params["ticket_id"]
    body = await request.json()
    new_agent = body.get("new_agent")
    if not new_agent:
        return JSONResponse({"error": "new_agent 必填"}, status_code=400)
    t = await store.reassign(ticket_id, new_agent)
    if not t:
        return JSONResponse({"error": "工单不存在"}, status_code=404)
    return JSONResponse(t)


@mcp.custom_route("/api/tickets/{ticket_id}/comments", methods=["POST"])
async def rest_add_comment(request: Request) -> JSONResponse:
    ticket_id = request.path_params["ticket_id"]
    body = await request.json()
    author = body.get("author", "agent")
    comment = body.get("comment", "").strip()
    if not comment:
        return JSONResponse({"error": "comment 必填"}, status_code=400)
    c = await store.add_comment(ticket_id, author, comment)
    if not c:
        return JSONResponse({"error": "工单不存在"}, status_code=404)
    return JSONResponse(c, status_code=201)


if __name__ == "__main__":
    import uvicorn
    app = mcp.streamable_http_app()
    uvicorn.run(app, host="0.0.0.0", port=8003)

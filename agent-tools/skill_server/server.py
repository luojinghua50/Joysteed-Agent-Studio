"""Skill MCP Server — 编排型 server：把多步业务动作封装成单个原子工具。

与 order/knowledge 等 leaf server 不同，skill_server 自己持有一个 MCPClient，
去调其它 leaf server 的工具，把"有固定 SOP 的多步流程"塌缩成一次工具调用。
agent-core 把这里的工具当普通 MCP 工具发现/调用，核心逻辑零改动。

设计约束（重要）：
- 本 server 的 skill 一律【只读】。写操作（如 create_ticket）绝不在此发生 ——
  否则会绕过 agent-core executor 的审批闸（executor 只拦截它直接看到的 tool_call
  名字，看不见 skill_server 进程内的调用）。skill 只产出诊断结论与"建单草稿 +
  need_ticket 标志"，真正的建单由 agent 拿到结论后自行调用 create_ticket，正常
  触发审批。这样审批/幂等两条不变量完全不受影响。
"""
import asyncio

from mcp.server.fastmcp import FastMCP

from skill_server.mcp_client import SkillMCPClient

mcp = FastMCP("skill-service")

# skill_server 自己作为 MCP 客户端去调 leaf server。进程级单例，复用 session。
_client = SkillMCPClient()

# 判定 KB 是否命中"可用方案"的相关度阈值。低于此值视为无有效方案 → 建议建单。
KB_SCORE_THRESHOLD = 0.5


@mcp.tool()
async def diagnose_fault(
    symptom: str,
    order_id: str | None = None,
    top_k: int = 3,
) -> dict:
    """故障诊断（只读）：结合订单/产品状态检索技术方案，给出结论与建单建议。

    编排 query_order(可选) + search_knowledge，塌缩为一次调用，替代 LLM 逐步编排。
    本工具不建单、不做任何写操作；若知识库无可用方案，返回 need_ticket=true 及
    ticket_draft，由上层 agent 自行调用 create_ticket（走审批）。

    Args:
        symptom: 用户描述的故障现象原文（必填）。
        order_id: 可选。提供则查订单/产品状态辅助诊断；查询失败不阻断诊断。
        top_k: 检索方案条数，默认 3。

    Returns:
        {
          symptom, order_context, solutions, resolved_by_kb,
          need_ticket, ticket_draft, source
        }
        - order_context: {product, order_status} 或 null（未传 order_id / 查询失败）
        - solutions: [{title, content, score}, ...]
        - resolved_by_kb: 是否命中可用方案（有 score >= 阈值 的结果）
        - need_ticket: KB 无可用方案时为 true，提示上层建单
        - ticket_draft: need_ticket=true 时给出 {title, description, priority}，否则 null
                        注意：不含 customer_id，由 agent 从会话上下文补全
    """
    # 1) 并发拉取：查订单(可选) + 搜方案。两者无依赖，可并行以省延迟。
    async def _fetch_order() -> dict | None:
        if not order_id:
            return None
        res = await _client.call_tool("order", "query_order", {"order_id": order_id})
        if not isinstance(res, dict) or res.get("error"):
            # 查询失败/订单不存在 → 降级为无订单上下文，不阻断诊断
            return None
        return res

    order_task = asyncio.create_task(_fetch_order())
    kb_task = asyncio.create_task(
        _client.call_tool("knowledge", "search_knowledge",
                          {"query": symptom, "top_k": top_k})
    )
    order_raw, kb_raw = await asyncio.gather(order_task, kb_task)

    # 2) 归一化订单上下文。order store 的订单 dict 无独立 product 字段，
    #    商品信息在 items=[{name, qty, price}, ...] 里，故从 items 提取商品名。
    order_context = None
    if order_raw:
        items = order_raw.get("items") or []
        product = "、".join(
            i.get("name", "") for i in items if isinstance(i, dict) and i.get("name")
        ) or None
        order_context = {
            "product": product,
            "order_status": order_raw.get("status"),
        }

    # 3) 归一化方案 + 判定是否命中可用方案
    solutions: list[dict] = []
    if isinstance(kb_raw, dict) and not kb_raw.get("error"):
        for item in kb_raw.get("results", []) or []:
            solutions.append({
                "title": item.get("title", "无标题"),
                "content": item.get("content", ""),
                "score": item.get("score", 0),
            })

    resolved_by_kb = any(s.get("score", 0) >= KB_SCORE_THRESHOLD for s in solutions)
    need_ticket = not resolved_by_kb

    # 4) 无可用方案 → 拼建单草稿（不含 customer_id，交给 agent 补全并落地）
    ticket_draft = None
    if need_ticket:
        ctx_line = ""
        if order_context:
            ctx_line = (f"\n关联订单：{order_id}"
                        f"（商品：{order_context.get('product') or '未知'}，"
                        f"状态：{order_context.get('order_status') or '未知'}）")
        ticket_draft = {
            "title": f"技术故障-{symptom[:20]}",
            "description": (f"用户反馈故障现象：{symptom}{ctx_line}\n"
                            f"知识库检索未命中可用解决方案，转技术支持处理。"),
            "priority": "medium",
        }

    return {
        "symptom": symptom,
        "order_context": order_context,
        "solutions": solutions,
        "resolved_by_kb": resolved_by_kb,
        "need_ticket": need_ticket,
        "ticket_draft": ticket_draft,
        "source": "skill:diagnose_fault",
    }


if __name__ == "__main__":
    import uvicorn
    app = mcp.streamable_http_app()
    uvicorn.run(app, host="0.0.0.0", port=8005)

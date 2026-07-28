"""写操作审批闸（Layer 2）。

agent 检测到 LLM 想调敏感写工具（create_ticket/apply_refund）时不执行，把待办
写调用存进 state，路由到本模块的两个节点：

- approval_node：仅调 `interrupt` 等人工确认，**不调 LLM**。这是方案 B 的正确性
  关键——LangGraph 恢复时整节点 replay，若在此调 LLM 会重放并可能 divergence；
  本节点无 LLM，replay 无副作用。
- execute_node：按人工决定分支。批准 → 执行写工具（IdempotencyGuard 幂等去重）
  + 据结果生成回复；拒绝 → 生成取消回复，写副作用绝不落地。

两种路径共用这两个节点：
- 单意图：pending_write（单 agent）→ approval → execute → END。
- 多意图批量栅栏：多个并行子 agent 各自命中敏感写，累积到 pending_writes（按 agent
  名分桶）。dispatch 收敛到**单个** approval 节点（单 task/单 interrupt，绕开并发
  interrupt 的 resume-map），一次 interrupt 亮出所有待批写，人工**逐条**批/拒；
  execute 逐 agent 落地被批准的写、生成回复写入 agent_results（该 agent 转 done），
  再回 dispatch 评估剩余子意图。resume 决定按写调用 id 逐条分发。
"""
import structlog
from langchain_core.messages import AIMessage
from langgraph.types import interrupt

from src.agents.state import CustomerState
from src.tools.executor import execute_pending_writes
from src.tools.mcp_adapter import get_agent_tools
from src.guardrails.idempotency import IdempotencyGuard

logger = structlog.get_logger()

# 第一步窄范围：只给这两个高危写工具上闸。后续可扩到 IdempotencyGuard.PROTECTED_TOOLS 全集。
APPROVAL_REQUIRED_TOOLS: set[str] = {"create_ticket", "apply_refund"}


def _pending_by_agent(state: CustomerState) -> dict:
    """多意图待审批写：pending_writes 中尚未 done（未写 agent_results）的 agent 桶。"""
    done = set((state.get("agent_results") or {}).keys())
    return {a: pw for a, pw in (state.get("pending_writes") or {}).items() if a not in done}


def _resolve_decision(decision, call_id: str) -> bool:
    """把 resume 的决定解析成「某条写调用 id 是否批准」。

    约定 decision 为 {"approved": bool, "reason": str, "decisions": {call_id: bool}}：
    - 有 decisions 且含该 id → 用逐条决定（批量栅栏逐条批/拒）。
    - 否则退回整批 approved（向后兼容单意图 / 整批一刀切）。
    也兼容裸 bool。
    """
    if isinstance(decision, dict):
        per = decision.get("decisions")
        if isinstance(per, dict) and call_id in per:
            return bool(per[call_id])
        return bool(decision.get("approved"))
    return bool(decision)


async def _judge_refund_reply(reflection, agent: str, user_msg: str,
                              results: dict, resp: AIMessage) -> AIMessage:
    """B 路径：退款成功时对确认文案做 L2 仲裁，失败走确定性模板。

    仅当 apply_refund 真正执行且返回成功（无 error 键）才触发；工具返回 error 时
    不改文案（保留模型对失败的说明，人工按需介入）。无 reflection 注入则原样返回。
    """
    if reflection is None:
        return resp
    refund_result = (results or {}).get("apply_refund")
    if not isinstance(refund_result, dict) or refund_result.get("error"):
        return resp  # 未退款/退款失败：不走质量闸

    from src.reflection.loop import judge_postwrite, REFUND_SKILL

    return await judge_postwrite(
        reflection.judge, reflection.error_store,
        agent=agent, skill=REFUND_SKILL, user_msg=user_msg,
        tool_result=refund_result, response=resp,
    )


async def approval_node(state: CustomerState) -> dict:
    """人工确认闸：interrupt 暂停图，等 /approve 带决定 resume。不调 LLM。

    单意图看 pending_write，多意图看 pending_writes（未 done 的桶）。多意图把所有
    待批写聚合成**单个** interrupt 一次性亮出（批量栅栏），故 approval 始终是单
    interrupt——绕开并发 interrupt 的 resume-map。决定存进 state，供 execute 消费。
    """
    pending = _pending_by_agent(state)

    if pending:
        # 多意图批量栅栏：聚合所有未 done agent 的待批写为一个列表（每条带 id + agent）
        calls = [
            {"id": c["id"], "agent": agent, "name": c["name"], "args": c["args"]}
            for agent, pw in pending.items()
            for c in pw.get("pending_calls", [])
        ]
        logger.info("approval_interrupt_batch", agents=list(pending), count=len(calls))
        decision = interrupt({
            "type": "batch_tool_approval",
            "calls": calls,
            "message": f"即将执行 {len(calls)} 项敏感写操作，请逐条确认。",
        })
        return {"approval_decision": decision}

    # 单意图
    pw = state.get("pending_write") or {}
    calls = pw.get("pending_calls", [])
    logger.info("approval_interrupt", agent=pw.get("agent"), tools=[c["name"] for c in calls])
    decision = interrupt({
        "type": "tool_approval",
        "agent": pw.get("agent"),
        "calls": [{"name": c["name"], "args": c["args"]} for c in calls],
        "message": f"即将执行敏感写操作 {[c['name'] for c in calls]}，需要确认。",
    })
    approved = decision.get("approved") if isinstance(decision, dict) else bool(decision)
    return {"approval_result": "approved" if approved else "rejected"}


async def _execute_one_agent(
    agent: str, pw: dict, decision, *, make_llm, mcp, session_id: str,
    reflection=None, user_msg: str = "",
) -> AIMessage:
    """逐 agent 落地被批准的写并生成回复（多意图批量栅栏用）。

    approved 的写幂等去重后执行；被拒/去重的写作为 stub 补占位 ToolMessage，
    保证 tool_call/ToolMessage 配对约束。返回该 agent 的最终回复。

    B 路径：若本 agent 执行了 apply_refund（写已不可逆），对生成的退款确认文案做
    L2 仲裁——不合格走确定性模板兜底，绝不重写、不重跑写。
    """
    calls = pw.get("pending_calls", [])
    working_messages = pw.get("working_messages", [])
    guard = IdempotencyGuard()
    tools = await get_agent_tools(agent, mcp)
    tool_map = {t.name: t for t in tools}

    executable, stubs = [], []
    for c in calls:
        if not _resolve_decision(decision, c["id"]):
            stubs.append(c)  # 被拒
            continue
        is_dup, _ = await guard.check_and_mark(session_id, c["name"], c["args"])
        if is_dup:
            logger.warning("approval_execute_dedup_skip", agent=agent, tool=c["name"])
            stubs.append(c)  # 去重：不执行但要占位
            continue
        executable.append(c)
        await guard.store_result(session_id, c["name"], c["args"], {"approved": True})

    llm = make_llm(agent)
    resp, results = await execute_pending_writes(
        llm, tool_map, list(working_messages), executable, stub_calls=stubs,
        return_results=True,
    )
    logger.info(
        "approval_executed_agent", agent=agent,
        approved=[c["name"] for c in executable], rejected=[c["name"] for c in stubs],
    )
    return await _judge_refund_reply(reflection, agent, user_msg, results, resp)


def _last_user_msg(state: CustomerState) -> str:
    """取最近一条用户消息（B 路径 judge 评估用）。"""
    for msg in reversed(state.get("messages") or []):
        if getattr(msg, "type", None) == "human":
            return getattr(msg, "content", "")
    return ""


async def execute_node(state: CustomerState, *, make_llm, mcp, prompts=None,
                       reflection=None, memory=None) -> dict:
    """审批后落地：批准则执行写工具 + 生成回复；拒绝则取消。

    多意图批量栅栏：逐 agent 落地被批准的写、生成回复写入 agent_results（该 agent
    转 done），不清 pending_writes（靠 done 过滤），返回后回 dispatch 评估剩余子意图。
    单意图：沿用原路径，清空 pending_write，→ END。

    B 路径：apply_refund 执行后的退款确认文案经 _judge_refund_reply 做 L2 仲裁。
    memory 非空时从写工具结果（apply_refund 的 refund_id/amount 等）抽实体写工作记忆。
    """
    pending = _pending_by_agent(state)
    user_msg = _last_user_msg(state)
    if pending:
        session_id = state.get("session_id", "")
        decision = state.get("approval_decision")
        results = {}
        for agent, pw in pending.items():
            resp = await _execute_one_agent(
                agent, pw, decision, make_llm=make_llm, mcp=mcp, session_id=session_id,
                reflection=reflection, user_msg=user_msg,
            )
            results[agent] = {"message": resp}
        # 写入 agent_results → 这些 agent 转 done，dispatch 据此推进剩余波次
        return {
            "agent_results": results,
            "is_multi_intent": True,
            "approval_decision": None,
        }

    # —— 单意图 ——
    pw = state.get("pending_write") or {}
    agent = pw.get("agent", "unknown")
    calls = pw.get("pending_calls", [])
    working_messages = pw.get("working_messages", [])

    # 拒绝：不碰任何写工具，直接给取消回复
    if state.get("approval_result") != "approved":
        logger.info("approval_rejected", agent=agent, tools=[c["name"] for c in calls])
        return {
            "messages": [AIMessage(content="您请求的操作需要人工确认，本次未获批准，已取消。如需继续请重新发起或联系人工客服。")],
            "resolved": True,
            "current_agent": agent,
            "pending_write": None,
        }

    # 批准：幂等去重后执行写工具，再据结果生成回复
    guard = IdempotencyGuard()
    tools = await get_agent_tools(agent, mcp)
    tool_map = {t.name: t for t in tools}
    session_id = state.get("session_id", "")

    executable = []
    for c in calls:
        is_dup, _ = await guard.check_and_mark(session_id, c["name"], c["args"])
        if is_dup:
            logger.warning("approval_execute_dedup_skip", tool=c["name"])
            continue
        executable.append(c)
        await guard.store_result(session_id, c["name"], c["args"], {"approved": True})

    llm = make_llm(agent)
    response, results = await execute_pending_writes(
        llm, tool_map, list(working_messages), executable, return_results=True
    )
    logger.info("approval_executed", agent=agent, tools=[c["name"] for c in executable])
    response = await _judge_refund_reply(reflection, agent, user_msg, results, response)
    # 工作记忆：从写工具结果（apply_refund 的 refund_id/amount 等）抽实体
    from src.agents.generic import _write_entities
    await _write_entities(memory, session_id, results)
    return {
        "messages": [response],
        "resolved": True,
        "current_agent": agent,
        "pending_write": None,
    }

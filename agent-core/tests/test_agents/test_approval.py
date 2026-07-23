"""写操作审批闸（Layer 2 第一步）测试。

覆盖：executor 对写工具的拦截（读工具照跑）、execute_node 批准/拒绝分支、
整图 interrupt→resume 闭环（经真实 LangGraph + checkpointer）。全程 mock LLM/工具，
不触 MCP / live API。
"""
import pytest
from functools import partial
from unittest.mock import patch

from langchain_core.messages import AIMessage, HumanMessage
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command

from src.tools.executor import run_agent_with_tools, ApprovalRequired, execute_pending_writes
from src.agents.approval import approval_node, execute_node, APPROVAL_REQUIRED_TOOLS
from src.agents.dispatch import dispatch_plan
from src.agents.state import CustomerState


class _FakeTool:
    def __init__(self, name, result="ok-result"):
        self.name = name
        self._result = result
        self.calls = 0

    async def ainvoke(self, args):
        self.calls += 1
        return f"{self.name}:{self._result}"


class _ToolThenTextLLM:
    """第 1 次 ainvoke 吐指定 tool_calls；之后吐纯文本（模拟 executor 的工具循环）。"""
    def __init__(self, tool_calls, final_text="已完成 T123"):
        self._tool_calls = tool_calls
        self._final = final_text
        self.n = 0

    def bind_tools(self, tools):
        return self

    async def ainvoke(self, messages):
        self.n += 1
        if self.n == 1 and self._tool_calls:
            return AIMessage(content="", tool_calls=self._tool_calls)
        return AIMessage(content=self._final)


class _TextLLM:
    """纯文本 LLM：execute 阶段只需据工具结果生成一句回复，不再发起工具调用。"""
    def __init__(self, text="工单 T123 已建"):
        self._text = text

    def bind_tools(self, tools):
        return self

    async def ainvoke(self, messages):
        return AIMessage(content=self._text)


class TestExecutorInterception:
    @pytest.mark.asyncio
    async def test_write_tool_raises_approval_required(self):
        llm = _ToolThenTextLLM([{"name": "create_ticket", "args": {"t": "x"}, "id": "c1"}])
        with pytest.raises(ApprovalRequired) as ei:
            await run_agent_with_tools(
                llm=llm, tools=[_FakeTool("create_ticket")], system_prompt="p",
                messages=[], protected_tools={"create_ticket"},
            )
        ar = ei.value
        assert [c["name"] for c in ar.pending_calls] == ["create_ticket"]
        # working_messages 带上了发起 tool_calls 的 AIMessage，供 execute 恢复上下文
        assert any(isinstance(m, AIMessage) for m in ar.working_messages)

    @pytest.mark.asyncio
    async def test_read_tool_runs_write_tool_held(self):
        # 同一轮既有读(search)又有写(create_ticket)：读跑掉，写触发中断
        read_tool = _FakeTool("search_faq")
        llm = _ToolThenTextLLM([
            {"name": "search_faq", "args": {"q": "x"}, "id": "r1"},
            {"name": "create_ticket", "args": {"t": "y"}, "id": "w1"},
        ])
        with pytest.raises(ApprovalRequired) as ei:
            await run_agent_with_tools(
                llm=llm, tools=[read_tool, _FakeTool("create_ticket")], system_prompt="p",
                messages=[], protected_tools={"create_ticket"},
            )
        assert read_tool.calls == 1                       # 读工具执行了
        assert [c["name"] for c in ei.value.pending_calls] == ["create_ticket"]

    @pytest.mark.asyncio
    async def test_no_protected_tools_behaves_as_before(self):
        # protected 为空（审批关闭）：写工具照常执行，不中断
        tool = _FakeTool("create_ticket")
        llm = _ToolThenTextLLM([{"name": "create_ticket", "args": {"t": "x"}, "id": "c1"}])
        resp = await run_agent_with_tools(
            llm=llm, tools=[tool], system_prompt="p", messages=[], protected_tools=set(),
        )
        assert tool.calls == 1
        assert resp.content == "已完成 T123"

    def test_default_protected_set(self):
        assert APPROVAL_REQUIRED_TOOLS == {"create_ticket", "apply_refund"}


def _pending_state(agent="complaint"):
    return {
        "messages": [HumanMessage(content="投诉")],
        "agent_results": {}, "current_agent": "", "session_id": "s1",
        "needs_approval": True, "approval_result": None,
        "pending_write": {
            "agent": agent,
            "pending_calls": [{"name": "create_ticket", "args": {"t": "x"}, "id": "c1"}],
            "working_messages": [AIMessage(content="", tool_calls=[
                {"name": "create_ticket", "args": {"t": "x"}, "id": "c1"}])],
        },
    }


class TestExecuteNode:
    @pytest.mark.asyncio
    async def test_approved_executes_and_replies(self):
        tool = _FakeTool("create_ticket", result="T123")
        async def fake_tools(agent, mcp): return [tool]
        with patch("src.agents.approval.get_agent_tools", fake_tools):
            state = {**_pending_state(), "approval_result": "approved"}
            out = await execute_node(state, make_llm=lambda a: _TextLLM("工单 T123 已建"), mcp=None)
        assert tool.calls == 1                            # 批准后写工具执行
        assert "T123" in out["messages"][-1].content
        assert out["resolved"] is True
        assert out["pending_write"] is None               # 清空

    @pytest.mark.asyncio
    async def test_rejected_skips_execution(self):
        tool = _FakeTool("create_ticket")
        async def fake_tools(agent, mcp): return [tool]
        with patch("src.agents.approval.get_agent_tools", fake_tools):
            state = {**_pending_state(), "approval_result": "rejected"}
            out = await execute_node(state, make_llm=lambda a: _TextLLM(), mcp=None)
        assert tool.calls == 0                            # 拒绝：写工具绝不执行
        assert "未获批准" in out["messages"][-1].content
        assert out["resolved"] is True
        assert out["pending_write"] is None


class TestApprovalGraphLoop:
    """整图 interrupt→resume：approval 中断 → /approve resume → execute 落地。"""

    def _build(self, make_llm, tool):
        async def fake_tools(agent, mcp): return [tool]
        patcher = patch("src.agents.approval.get_agent_tools", fake_tools)
        patcher.start()

        async def seed(state):
            return _pending_state()  # 模拟 agent 拦截后置好 pending_write 的状态

        g = StateGraph(CustomerState)
        g.add_node("seed", seed)
        g.add_node("approval", approval_node)
        g.add_node("execute", partial(execute_node, make_llm=make_llm, mcp=None))
        g.set_entry_point("seed")
        g.add_edge("seed", "approval")
        g.add_edge("approval", "execute")
        g.add_edge("execute", END)
        return g.compile(checkpointer=MemorySaver()), patcher

    @pytest.mark.asyncio
    async def test_interrupt_then_approve(self):
        tool = _FakeTool("create_ticket", result="T123")
        app, patcher = self._build(lambda a: _TextLLM("工单 T123 已建"), tool)
        try:
            cfg = {"configurable": {"thread_id": "t-approve"}}
            r1 = await app.ainvoke({}, config=cfg)
            assert r1.get("__interrupt__"), "应在 approval 节点中断"
            payload = r1["__interrupt__"][0].value
            assert payload["type"] == "tool_approval"
            assert payload["calls"][0]["name"] == "create_ticket"
            assert tool.calls == 0                        # 中断时写工具尚未执行

            r2 = await app.ainvoke(Command(resume={"approved": True}), config=cfg)
            assert tool.calls == 1                        # resume 批准后才执行
            assert "T123" in r2["messages"][-1].content
        finally:
            patcher.stop()

    @pytest.mark.asyncio
    async def test_interrupt_then_reject(self):
        tool = _FakeTool("create_ticket")
        app, patcher = self._build(lambda a: _TextLLM(), tool)
        try:
            cfg = {"configurable": {"thread_id": "t-reject"}}
            await app.ainvoke({}, config=cfg)
            r = await app.ainvoke(Command(resume={"approved": False}), config=cfg)
            assert tool.calls == 0                        # 拒绝：写副作用绝不落地
            assert "未获批准" in r["messages"][-1].content
        finally:
            patcher.stop()


# ————————————————————————————————————————————————————————————————
# 多意图批量栅栏审批（Layer 2 多意图路径）
# ————————————————————————————————————————————————————————————————

def _pw_bucket(agent, tool, call_id):
    """构造 pending_writes[agent] 桶：带发起 tool_calls 的 AIMessage（供 execute 恢复）。"""
    calls = [{"name": tool, "args": {"x": call_id}, "id": call_id}]
    return {
        "pending_calls": calls,
        "working_messages": [AIMessage(content="", tool_calls=calls)],
    }


class TestDispatchApprovalGate:
    """dispatch 三出口：本波跑完后，有待审批写 → approval；否则 synthesize。"""

    def test_parked_write_routes_to_approval(self):
        # order 已 done（只读），complaint 命中写累积到 pending_writes 未 done → 进 approval
        state = {
            "plan": [
                {"agent": "order", "query": "查单", "depends_on": []},
                {"agent": "complaint", "query": "投诉", "depends_on": []},
            ],
            "agent_results": {"order": {"message": AIMessage(content="订单已查")}},
            "pending_writes": {"complaint": _pw_bucket("complaint", "create_ticket", "c1")},
        }
        assert dispatch_plan(state) == "approval"

    def test_parked_agent_not_refanned_out(self):
        # complaint 已 parked（在 pending_writes 未 done），不能再被扇出重复跑 LLM
        state = {
            "plan": [{"agent": "complaint", "query": "投诉", "depends_on": []}],
            "agent_results": {},
            "pending_writes": {"complaint": _pw_bucket("complaint", "create_ticket", "c1")},
        }
        assert dispatch_plan(state) == "approval"   # 不是 Send 列表

    def test_done_agent_in_pending_writes_ignored(self):
        # complaint 已执行完（写进 agent_results=done），残留的 pending_writes 桶不再触发 approval
        state = {
            "plan": [{"agent": "complaint", "query": "投诉", "depends_on": []}],
            "agent_results": {"complaint": {"message": AIMessage(content="已建单")}},
            "pending_writes": {"complaint": _pw_bucket("complaint", "create_ticket", "c1")},
        }
        assert dispatch_plan(state) == "synthesize"


class TestBatchApprovalNode:
    """批量栅栏：多个 parked agent 聚合成单个 interrupt，决定透传给 execute。"""

    @pytest.mark.asyncio
    async def test_aggregates_all_pending_into_single_interrupt(self):
        state = {
            "agent_results": {},
            "pending_writes": {
                "complaint": _pw_bucket("complaint", "create_ticket", "c1"),
                "refund": _pw_bucket("refund", "apply_refund", "r1"),
            },
        }
        with patch("src.agents.approval.interrupt", lambda payload: payload) as _:
            out = await approval_node(state)
        # interrupt 被 mock 成 identity，approval_decision 即 payload
        payload = out["approval_decision"]
        assert payload["type"] == "batch_tool_approval"
        ids = {c["id"] for c in payload["calls"]}
        assert ids == {"c1", "r1"}                  # 两个 agent 的写聚合进一个 interrupt

    @pytest.mark.asyncio
    async def test_done_agent_excluded_from_batch(self):
        state = {
            "agent_results": {"complaint": {"message": AIMessage(content="done")}},
            "pending_writes": {
                "complaint": _pw_bucket("complaint", "create_ticket", "c1"),
                "refund": _pw_bucket("refund", "apply_refund", "r1"),
            },
        }
        with patch("src.agents.approval.interrupt", lambda payload: payload):
            out = await approval_node(state)
        ids = {c["id"] for c in out["approval_decision"]["calls"]}
        assert ids == {"r1"}                        # 已 done 的 complaint 不再出现


class TestBatchExecuteNode:
    """execute 逐 agent 落地被批准的写、写入 agent_results（转 done）、回 dispatch。"""

    def _multi_state(self, decision):
        return {
            "session_id": "s1",
            "is_multi_intent": True,
            "agent_results": {},
            "pending_writes": {
                "complaint": _pw_bucket("complaint", "create_ticket", "c1"),
                "refund": _pw_bucket("refund", "apply_refund", "r1"),
            },
            "approval_decision": decision,
        }

    @pytest.mark.asyncio
    async def test_per_call_approve_and_reject(self):
        # c1 批准执行，r1 拒绝 → 只有 create_ticket 落地
        ticket = _FakeTool("create_ticket", result="T1")
        refund = _FakeTool("apply_refund", result="R1")
        tool_by_agent = {"complaint": [ticket], "refund": [refund]}
        async def fake_tools(agent, mcp): return tool_by_agent[agent]
        with patch("src.agents.approval.get_agent_tools", fake_tools):
            state = self._multi_state({"decisions": {"c1": True, "r1": False}})
            out = await execute_node(
                state, make_llm=lambda a: _TextLLM(f"{a}-done"), mcp=None
            )
        assert ticket.calls == 1                    # 批准的写执行
        assert refund.calls == 0                    # 拒绝的写绝不执行
        # 两个 agent 都写进 agent_results → 都转 done，dispatch 才能推进
        assert set(out["agent_results"].keys()) == {"complaint", "refund"}
        assert out["is_multi_intent"] is True

    @pytest.mark.asyncio
    async def test_result_marks_done_so_dispatch_advances(self):
        ticket = _FakeTool("create_ticket", result="T1")
        refund = _FakeTool("apply_refund", result="R1")
        tool_by_agent = {"complaint": [ticket], "refund": [refund]}
        async def fake_tools(agent, mcp): return tool_by_agent[agent]
        with patch("src.agents.approval.get_agent_tools", fake_tools):
            state = self._multi_state({"approved": True})   # 整批批准
            out = await execute_node(state, make_llm=lambda a: _TextLLM("ok"), mcp=None)
        assert ticket.calls == 1 and refund.calls == 1
        # 把 execute 产出并回 dispatch：两个 agent 都 done → synthesize
        merged = {**state, "agent_results": out["agent_results"],
                  "plan": [{"agent": "complaint", "query": "x", "depends_on": []},
                           {"agent": "refund", "query": "y", "depends_on": []}]}
        assert dispatch_plan(merged) == "synthesize"


class TestBatchApprovalGraphLoop:
    """整图闭环：并行两写 → 单 interrupt 批量栅栏 → 逐条 resume → execute 回 dispatch → synthesize。"""

    def _build(self):
        ticket = _FakeTool("create_ticket", result="T1")
        refund = _FakeTool("apply_refund", result="R1")
        tool_by_agent = {"complaint": [ticket], "refund": [refund]}
        async def fake_tools(agent, mcp): return tool_by_agent[agent]
        patcher = patch("src.agents.approval.get_agent_tools", fake_tools)
        patcher.start()

        async def seed(state):
            # 模拟一波并行子 agent 跑完：两个都命中写、累积到 pending_writes（未 done）
            return {
                "is_multi_intent": True,
                "plan": [
                    {"agent": "complaint", "query": "投诉", "depends_on": []},
                    {"agent": "refund", "query": "退款", "depends_on": []},
                ],
                "agent_results": {},
                "pending_writes": {
                    "complaint": _pw_bucket("complaint", "create_ticket", "c1"),
                    "refund": _pw_bucket("refund", "apply_refund", "r1"),
                },
                "session_id": "s1",
            }

        async def synth(state):
            return {"messages": [AIMessage(content="融合完成")], "resolved": True}

        g = StateGraph(CustomerState)
        g.add_node("seed", seed)
        g.add_node("dispatch", lambda s: {})
        g.add_node("approval", approval_node)
        g.add_node("execute", partial(execute_node, make_llm=lambda a: _TextLLM(f"{a}-ok"), mcp=None))
        g.add_node("synthesize", synth)
        g.set_entry_point("seed")
        g.add_edge("seed", "dispatch")
        g.add_conditional_edges("dispatch", dispatch_plan,
                                {"complaint": "synthesize", "refund": "synthesize",
                                 "approval": "approval", "synthesize": "synthesize"})
        g.add_edge("approval", "execute")
        g.add_conditional_edges("execute", lambda s: "dispatch",
                                {"dispatch": "dispatch"})
        g.add_edge("synthesize", END)
        return g.compile(checkpointer=MemorySaver()), ticket, refund, patcher

    @pytest.mark.asyncio
    async def test_batch_gate_partial_approve_then_advance(self):
        app, ticket, refund, patcher = self._build()
        try:
            cfg = {"configurable": {"thread_id": "t-batch"}}
            r1 = await app.ainvoke({}, config=cfg)
            assert r1.get("__interrupt__"), "应在批量 approval 节点中断"
            payload = r1["__interrupt__"][0].value
            assert payload["type"] == "batch_tool_approval"
            assert {c["id"] for c in payload["calls"]} == {"c1", "r1"}
            assert ticket.calls == 0 and refund.calls == 0   # 中断时都未落地

            # 逐条：批准 create_ticket，拒绝 apply_refund
            r2 = await app.ainvoke(
                Command(resume={"decisions": {"c1": True, "r1": False}}), config=cfg
            )
            assert ticket.calls == 1                          # 批准的落地
            assert refund.calls == 0                          # 拒绝的绝不落地
            assert r2["messages"][-1].content == "融合完成"    # 回 dispatch→全 done→synthesize
        finally:
            patcher.stop()

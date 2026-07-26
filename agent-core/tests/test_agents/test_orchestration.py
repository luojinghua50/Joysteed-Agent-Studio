"""多意图编排相关纯函数的单元测试（不依赖 LLM / MCP）。

覆盖：merge_results reducer、detect_handoff 白名单、哑路由封顶、
dispatch 分波（依赖感知）、supervisor 计划归一化、synthesizer 退化与融合。
"""
import pytest
from langchain_core.messages import AIMessage, HumanMessage

from src.agents.state import merge_results, take_latest
from src.agents.handoff import detect_handoff
from src.agents.router import route_after_agent, bump_routing_count
from src.agents.dispatch import dispatch_plan
from src.agents.supervisor import _normalize_plan, IntentPlan, SubIntent
from src.agents.synthesizer import synthesize_node
from src.agents.generic import agent_node
from src.agents.registry import AgentSpec


class TestMergeResults:
    def test_merges_disjoint(self):
        assert merge_results({"order": 1}, {"faq": 2}) == {"order": 1, "faq": 2}

    def test_idempotent_overwrite(self):
        # 同一 agent 重试时覆盖而非重复，避免重复副作用
        assert merge_results({"order": {"v": 1}}, {"order": {"v": 2}}) == {"order": {"v": 2}}

    def test_handles_none(self):
        assert merge_results(None, {"a": 1}) == {"a": 1}
        assert merge_results({"a": 1}, None) == {"a": 1}


class TestTakeLatest:
    """current_agent 的 last-write-wins reducer：让多意图并行扇出时的并发写合法。"""

    def test_takes_right(self):
        assert take_latest("order", "faq") == "faq"

    def test_falls_back_to_left_when_right_empty(self):
        assert take_latest("order", "") == "order"
        assert take_latest("order", None) == "order"


class TestParallelFanOutNoConcurrentUpdate:
    """回归：多意图同一 super-step 内多个 Agent 并发写 current_agent，
    曾触发 INVALID_CONCURRENT_GRAPH_UPDATE。current_agent 带 take_latest
    reducer 后并发写合法化，图能正常跑完。"""

    @pytest.mark.asyncio
    async def test_parallel_writes_to_current_agent_do_not_crash(self):
        from langgraph.graph import StateGraph, END
        from langgraph.types import Send
        from src.agents.state import CustomerState

        # 两个业务节点：都在同一步写 current_agent（复现并发写冲突的最小条件）
        async def order_node(state):
            return {"agent_results": {"order": {"message": "order done"}},
                    "current_agent": "order"}

        async def faq_node(state):
            return {"agent_results": {"faq": {"message": "faq done"}},
                    "current_agent": "faq"}

        def fan_out(state):
            return [Send("order", {**state}), Send("faq", {**state})]

        g = StateGraph(CustomerState)
        g.add_node("dispatch", lambda s: {})
        g.add_node("order", order_node)
        g.add_node("faq", faq_node)
        g.set_entry_point("dispatch")
        g.add_conditional_edges("dispatch", fan_out, {"order": "order", "faq": "faq"})
        g.add_edge("order", END)
        g.add_edge("faq", END)
        app = g.compile()

        initial = {"messages": [HumanMessage(content="查订单是否含耳机 + 怎么退换货")],
                   "agent_results": {}, "is_multi_intent": True, "current_agent": ""}

        # 修复前此处会抛 INVALID_CONCURRENT_GRAPH_UPDATE；修复后正常返回
        result = await app.ainvoke(initial)

        # 两个并行 Agent 的产出都通过 reducer 归并保留
        assert set(result["agent_results"].keys()) == {"order", "faq"}
        # current_agent 取到其中一个并发写入值（last-write-wins，无副作用）
        assert result["current_agent"] in {"order", "faq"}


class TestDetectHandoff:
    def test_extracts_allowed_target(self):
        assert detect_handoff("好的 [HANDOFF:complaint]", ["complaint", "order"]) == "complaint"

    def test_rejects_out_of_whitelist(self):
        assert detect_handoff("[HANDOFF:order]", ["complaint"]) is None

    def test_no_marker(self):
        assert detect_handoff("普通回复，没有交接", ["order"]) is None

    def test_empty_and_none(self):
        assert detect_handoff("", ["order"]) is None
        assert detect_handoff(None, ["order"]) is None


class TestRouteAfterAgent:
    def test_resolved_goes_end(self):
        from langgraph.graph import END
        assert route_after_agent({"resolved": True}) == END

    def test_handoff_routes_through_bump_chokepoint(self):
        # 有效交接不直接跳目标，而是先进 handoff 节点（+1 计数）再路由
        state = {"resolved": False, "handoff_target": "complaint", "routing_count": 0}
        assert route_after_agent(state) == "handoff"

    def test_caps_to_human_when_over_limit(self):
        # routing_count 达到上限 → 兜底转人工，防死循环
        state = {"resolved": False, "handoff_target": "complaint", "routing_count": 99}
        assert route_after_agent(state) == "human_handoff"

    def test_no_target_falls_back_human(self):
        state = {"resolved": False, "handoff_target": None, "routing_count": 0}
        assert route_after_agent(state) == "human_handoff"

    def test_bump_increments(self):
        out = bump_routing_count({"routing_count": 1, "handoff_target": "order"})
        assert out["routing_count"] == 2
        assert out["intent"] == "order"

    def test_route_to_target_after_bump(self):
        # bump 节点出口按 handoff_target 路由到真正目标 agent
        from src.agents.router import route_to_target
        assert route_to_target({"handoff_target": "order"}) == "order"


class TestDispatchPlan:
    def test_parallel_when_no_deps(self):
        state = {
            "plan": [
                {"agent": "order", "query": "查订单", "depends_on": []},
                {"agent": "faq", "query": "问政策", "depends_on": []},
            ],
            "agent_results": {},
        }
        sends = dispatch_plan(state)
        assert isinstance(sends, list)
        assert {s.node for s in sends} == {"order", "faq"}
        # 每个 Send 带有改写后的子查询且标记多意图
        for s in sends:
            assert s.arg["is_multi_intent"] is True
            assert "_sub_query" in s.arg

    def test_serial_holds_back_dependent(self):
        # complaint 依赖 order，首波只应派发 order
        state = {
            "plan": [
                {"agent": "order", "query": "查订单", "depends_on": []},
                {"agent": "complaint", "query": "赔偿", "depends_on": ["order"]},
            ],
            "agent_results": {},
        }
        sends = dispatch_plan(state)
        assert {s.node for s in sends} == {"order"}

    def test_dependent_released_after_dep_done(self):
        state = {
            "plan": [
                {"agent": "order", "query": "查订单", "depends_on": []},
                {"agent": "complaint", "query": "赔偿", "depends_on": ["order"]},
            ],
            "agent_results": {"order": {"message": AIMessage(content="订单已查")}},
        }
        sends = dispatch_plan(state)
        assert {s.node for s in sends} == {"complaint"}

    def test_all_done_goes_synthesize(self):
        state = {
            "plan": [{"agent": "order", "query": "x", "depends_on": []}],
            "agent_results": {"order": {"message": AIMessage(content="done")}},
        }
        assert dispatch_plan(state) == "synthesize"


class TestNormalizePlan:
    def test_filters_invalid_agent(self):
        plan = IntentPlan(is_multi_intent=True, sub_intents=[
            SubIntent(agent="order", query="a"),
            SubIntent(agent="not_an_agent", query="b"),
        ])
        out = _normalize_plan(plan)
        assert [s["agent"] for s in out] == ["order"]

    def test_drops_deps_outside_plan(self):
        plan = IntentPlan(is_multi_intent=True, sub_intents=[
            SubIntent(agent="complaint", query="赔偿", depends_on=["order", "ghost"]),
            SubIntent(agent="order", query="查单"),
        ])
        out = _normalize_plan(plan)
        comp = next(s for s in out if s["agent"] == "complaint")
        assert comp["depends_on"] == ["order"]  # ghost 被剔除


class _StubLLM:
    """记录是否被调用的假 LLM。"""
    def __init__(self):
        self.called = False

    async def ainvoke(self, messages):
        self.called = True
        return AIMessage(content="融合后的连贯回复")


class TestSynthesizer:
    @pytest.mark.asyncio
    async def test_single_result_skips_llm(self):
        llm = _StubLLM()
        state = {
            "agent_results": {"order": {"message": AIMessage(content="订单结果")}},
            "messages": [HumanMessage(content="查订单")],
        }
        out = await synthesize_node(state, llm)
        assert llm.called is False  # 退化为单结果，省一次 LLM
        assert out["messages"][0].content == "订单结果"
        assert out["resolved"] is True

    @pytest.mark.asyncio
    async def test_multi_result_calls_llm(self):
        llm = _StubLLM()
        state = {
            "agent_results": {
                "order": {"message": AIMessage(content="订单已查")},
                "complaint": {"message": AIMessage(content="已记录投诉")},
            },
            "messages": [HumanMessage(content="查订单并投诉")],
        }
        out = await synthesize_node(state, llm)
        assert llm.called is True
        assert out["messages"][0].content == "融合后的连贯回复"

    @pytest.mark.asyncio
    async def test_empty_results_graceful(self):
        llm = _StubLLM()
        out = await synthesize_node({"agent_results": {}, "messages": []}, llm)
        assert llm.called is False
        assert out["resolved"] is True


# —— supervisor 生产级三层降级 ——
from langchain_core.messages import SystemMessage
from src.agents.supervisor import supervisor_node, IntentPlan, SubIntent


class _StructuredLLM:
    """模拟经 litellm 中转的 LLM：所有调用都走 ainvoke 返回文本。

    supervisor 已不再用 with_structured_output（litellm 下它会退化成纯文本 +
    代码围栏），L1 结构化分解与 L2 单意图分类都通过 ainvoke 取文本。这里按
    system prompt 区分两类调用：
    - 含「意图分解器」（SUPERVISOR_PLAN_PROMPT 的特征串）→ L1 分解，按
      decompose_outcomes 依次返回：IntentPlan → 序列化成 JSON 文本供 supervisor
      自解析；Exception → 抛出（驱动有界重试 / 降级）。
    - 否则 → L2 单意图纯文本分类，返回 classify_text（或抛出）。
    """
    def __init__(self, decompose_outcomes, classify_text="order", critic_verdict=None):
        self._decompose_outcomes = list(decompose_outcomes)
        self._classify_text = classify_text
        # critic 裁决 JSON 文本；None => 默认无误派（rejected 为空），plan 原样放行
        self._critic_verdict = critic_verdict
        self.decompose_calls = 0
        self.classify_calls = 0
        self.critic_calls = 0

    async def ainvoke(self, messages):
        system = messages[0].content if messages else ""
        if "意图分解器" in system:          # SUPERVISOR_PLAN_PROMPT => L1 分解
            i = self.decompose_calls
            self.decompose_calls += 1
            outcome = self._decompose_outcomes[min(i, len(self._decompose_outcomes) - 1)]
            if isinstance(outcome, Exception):
                raise outcome
            return AIMessage(content=outcome.model_dump_json())
        if "分派校验器" in system:          # CRITIC_PROMPT => 派发前 plan 校验
            self.critic_calls += 1
            return AIMessage(content=self._critic_verdict or '{"rejected": []}')
        # 否则是 L2 单意图分类（SUPERVISOR_SYSTEM_PROMPT）
        self.classify_calls += 1
        if isinstance(self._classify_text, Exception):
            raise self._classify_text
        return AIMessage(content=self._classify_text)


def _msg_state():
    return {"messages": [HumanMessage(content="查订单 12345")], "memory_context": ""}


class TestSupervisorDegradation:
    @pytest.mark.asyncio
    async def test_l1_structured_success(self):
        # L1：结构化分解直接成功（单意图）
        plan = IntentPlan(is_multi_intent=False, sub_intents=[SubIntent(agent="order", query="查订单")])
        llm = _StructuredLLM([plan])
        out = await supervisor_node(_msg_state(), llm)
        assert out["intent"] == "order"
        assert out["is_multi_intent"] is False
        assert llm.classify_calls == 0  # 没走降级

    @pytest.mark.asyncio
    async def test_l1_multi_intent_plan(self):
        plan = IntentPlan(is_multi_intent=True, sub_intents=[
            SubIntent(agent="order", query="查订单"),
            SubIntent(agent="complaint", query="投诉", depends_on=["order"]),
        ])
        llm = _StructuredLLM([plan])
        out = await supervisor_node(_msg_state(), llm)
        assert out["is_multi_intent"] is True
        assert [s["agent"] for s in out["plan"]] == ["order", "complaint"]

    @pytest.mark.asyncio
    async def test_l1_retries_parse_error_then_succeeds(self):
        # 解析错误归为可重试 → re-roll 后成功，不降级
        from pydantic import ValidationError
        try:
            IntentPlan(is_multi_intent="not_a_bool", sub_intents="bad")  # 触发真实 ValidationError
        except ValidationError as ve:
            parse_err = ve
        plan = IntentPlan(is_multi_intent=False, sub_intents=[SubIntent(agent="faq", query="问政策")])
        llm = _StructuredLLM([parse_err, plan])
        out = await supervisor_node(_msg_state(), llm)
        assert out["intent"] == "faq"
        assert llm.decompose_calls == 2   # 重试了一次
        assert llm.classify_calls == 0    # 未降级到 L2

    @pytest.mark.asyncio
    async def test_l2_degrades_to_single_classify(self):
        # L1 彻底失败（持续解析错误）→ 降级到 L2 单意图分类
        from pydantic import ValidationError
        try:
            IntentPlan(is_multi_intent="x", sub_intents="y")
        except ValidationError as ve:
            parse_err = ve
        llm = _StructuredLLM([parse_err], classify_text="complaint")
        out = await supervisor_node(_msg_state(), llm)
        assert out["intent"] == "complaint"
        assert out["is_multi_intent"] is False
        assert llm.classify_calls >= 1    # 走了 L2

    @pytest.mark.asyncio
    async def test_l3_routes_human_when_all_fail(self):
        # L1 和 L2 都失败（LLM 全挂）→ 兜底转人工，不抛异常
        llm = _StructuredLLM([RuntimeError("model down")], classify_text=RuntimeError("model down"))
        out = await supervisor_node(_msg_state(), llm)
        assert out["intent"] == "human"
        assert out["current_agent"] == "human_handoff"
        assert out["is_multi_intent"] is False

    @pytest.mark.asyncio
    async def test_empty_messages_routes_human(self):
        llm = _StructuredLLM([RuntimeError("should not be called")])
        out = await supervisor_node({"messages": [], "memory_context": ""}, llm)
        assert out["current_agent"] == "human_handoff"
        assert llm.decompose_calls == 0



class TestAgentNodeHandoffRuleInjection:
    """交接规则仅在单意图模式注入 prompt；多意图模式不注入（否则泄漏 [HANDOFF:x]）。"""

    _SPEC = AgentSpec(
        name="order", prompt_id="order.v1",
        can_handoff_to=["complaint", "faq"],
    )

    async def _capture_prompt(self, monkeypatch, state):
        """跑一次 agent_node，返回传给 run_agent_with_tools 的 system_prompt。"""
        captured = {}

        async def _fake_run(*, llm, tools, system_prompt, messages,
                            protected_tools=None, return_transcript=False):
            captured["prompt"] = system_prompt
            return AIMessage(content="ok")

        async def _fake_tools(name, mcp):
            return []

        monkeypatch.setattr("src.agents.generic.run_agent_with_tools", _fake_run)
        monkeypatch.setattr("src.agents.generic.get_agent_tools", _fake_tools)
        await agent_node(state, spec=self._SPEC, llm=object(), mcp=object(), prompts=None)
        return captured["prompt"]

    @pytest.mark.asyncio
    async def test_single_intent_injects_handoff_rule(self, monkeypatch):
        state = {"messages": [HumanMessage(content="查订单")], "is_multi_intent": False}
        prompt = await self._capture_prompt(monkeypatch, state)
        assert "交接规则" in prompt
        assert "[HANDOFF:" in prompt

    @pytest.mark.asyncio
    async def test_multi_intent_omits_handoff_rule(self, monkeypatch):
        # 多意图：子意图由 dispatch 用 _sub_query 扇出，prompt 不含交接规则
        state = {
            "messages": [HumanMessage(content="查订单并投诉")],
            "is_multi_intent": True, "_sub_query": "查订单",
        }
        prompt = await self._capture_prompt(monkeypatch, state)
        assert "交接规则" not in prompt
        assert "[HANDOFF:" not in prompt


# —— Layer 1：派发前 plan 校验（plan_critic） ——
from src.agents.plan_critic import (
    critique_plan, plan_needs_critique, high_risk_agents,
    _apply_verdict, PlanVerdict, RejectedIntent,
)
from src.config import ReflectionConfig

_RC = ReflectionConfig()  # 默认 complaint=judge（高危），其余非高危


class TestPlanCriticPure:
    """纯函数：高危集合、触发门、裁决应用（不调 LLM）。"""

    def test_high_risk_from_policies(self):
        assert "complaint" in high_risk_agents(_RC)
        assert "faq" not in high_risk_agents(_RC)

    def test_needs_critique_only_when_high_risk_present(self):
        assert plan_needs_critique([{"agent": "complaint", "query": "x"}], _RC) is True
        assert plan_needs_critique([{"agent": "order", "query": "x"},
                                    {"agent": "faq", "query": "y"}], _RC) is False

    def test_needs_critique_off_when_disabled(self):
        cfg = ReflectionConfig(enabled=False)
        assert plan_needs_critique([{"agent": "complaint", "query": "x"}], cfg) is False

    def test_apply_verdict_reassigns_high_risk_to_safe(self):
        subs = [{"agent": "order", "query": "查单", "depends_on": []},
                {"agent": "complaint", "query": "退货政策", "depends_on": []}]
        verdict = PlanVerdict(rejected=[RejectedIntent(agent="complaint", reason="无投诉信号", reassign_to="faq")])
        out = _apply_verdict(subs, verdict, _RC)
        assert {s["agent"] for s in out} == {"order", "faq"}

    def test_apply_verdict_drops_when_reassign_invalid(self):
        # reassign 目标也是高危 → 不采纳，直接剔除（绝不把误派改成另一个高危分派）
        subs = [{"agent": "complaint", "query": "退货政策", "depends_on": []}]
        verdict = PlanVerdict(rejected=[RejectedIntent(agent="complaint", reason="无信号", reassign_to="complaint")])
        out = _apply_verdict(subs, verdict, _RC)
        assert out == []

    def test_apply_verdict_ignores_rejection_of_non_high_risk(self):
        # critic 若误拒非高危 agent，一律忽略（保守，防误伤正常分派）
        subs = [{"agent": "faq", "query": "问政策", "depends_on": []}]
        verdict = PlanVerdict(rejected=[RejectedIntent(agent="faq", reason="乱拒")])
        out = _apply_verdict(subs, verdict, _RC)
        assert [s["agent"] for s in out] == ["faq"]

    def test_apply_verdict_cleans_dangling_deps(self):
        # complaint 被剔除后，依赖它的子意图的 depends_on 要清掉悬空引用
        subs = [{"agent": "complaint", "query": "投诉", "depends_on": []},
                {"agent": "order", "query": "查单", "depends_on": ["complaint"]}]
        verdict = PlanVerdict(rejected=[RejectedIntent(agent="complaint", reason="无信号")])
        out = _apply_verdict(subs, verdict, _RC)
        order = next(s for s in out if s["agent"] == "order")
        assert order["depends_on"] == []


class TestCritiquePlanFailOpen:
    @pytest.mark.asyncio
    async def test_fail_open_on_llm_error(self):
        class _BoomLLM:
            async def ainvoke(self, messages):
                raise RuntimeError("judge down")
        subs = [{"agent": "complaint", "query": "退货政策", "depends_on": []}]
        out = await critique_plan(_BoomLLM(), [HumanMessage(content="退货政策")], subs, _RC)
        assert out == subs  # 校验失败 → 原样放行，绝不崩

    @pytest.mark.asyncio
    async def test_fail_open_on_bad_json(self):
        class _JunkLLM:
            async def ainvoke(self, messages):
                return AIMessage(content="这不是JSON")
        subs = [{"agent": "complaint", "query": "退货政策", "depends_on": []}]
        out = await critique_plan(_JunkLLM(), [HumanMessage(content="退货政策")], subs, _RC)
        assert out == subs


class TestSupervisorPlanCriticE2E:
    """端到端：核心场景「问退货政策被误派 complaint」经 supervisor 被拦下。"""

    def _state(self):
        return {"messages": [HumanMessage(content="你们的退货政策是什么，另外帮我查下订单 123")],
                "memory_context": ""}

    @pytest.mark.asyncio
    async def test_misrouted_complaint_reassigned_to_faq(self):
        plan = IntentPlan(is_multi_intent=True, sub_intents=[
            SubIntent(agent="order", query="查订单 123"),
            SubIntent(agent="complaint", query="退货政策"),   # ← 误派
        ])
        verdict = '{"rejected": [{"agent": "complaint", "reason": "仅询问退货政策，无投诉信号", "reassign_to": "faq"}]}'
        llm = _StructuredLLM([plan], critic_verdict=verdict)
        out = await supervisor_node(self._state(), llm)
        assert llm.critic_calls == 1                       # 高危 plan 触发了校验
        assert out["is_multi_intent"] is True
        assert set(s["agent"] for s in out["plan"]) == {"order", "faq"}  # complaint 已被改派

    @pytest.mark.asyncio
    async def test_misroute_dropped_collapses_to_single_intent(self):
        # 误派被剔除且无合法改派 → 只剩 order → collapse 成单意图
        plan = IntentPlan(is_multi_intent=True, sub_intents=[
            SubIntent(agent="order", query="查订单 123"),
            SubIntent(agent="complaint", query="退货政策"),
        ])
        verdict = '{"rejected": [{"agent": "complaint", "reason": "无投诉信号"}]}'
        llm = _StructuredLLM([plan], critic_verdict=verdict)
        out = await supervisor_node(self._state(), llm)
        assert llm.critic_calls == 1
        assert out["is_multi_intent"] is False
        assert out["intent"] == "order"

    @pytest.mark.asyncio
    async def test_no_high_risk_skips_critic(self):
        # plan 不含高危 agent → 不触发校验，省一次 LLM
        plan = IntentPlan(is_multi_intent=True, sub_intents=[
            SubIntent(agent="order", query="查单"),
            SubIntent(agent="faq", query="问政策"),
        ])
        llm = _StructuredLLM([plan])
        out = await supervisor_node(self._state(), llm)
        assert llm.critic_calls == 0
        assert out["is_multi_intent"] is True

    @pytest.mark.asyncio
    async def test_legit_complaint_survives_critic(self):
        # 真实投诉 → critic 无误派（rejected 空）→ plan 原样保留
        plan = IntentPlan(is_multi_intent=True, sub_intents=[
            SubIntent(agent="order", query="查订单"),
            SubIntent(agent="complaint", query="态度太差要投诉赔偿"),
        ])
        llm = _StructuredLLM([plan])  # 默认裁决 rejected 为空
        out = await supervisor_node(self._state(), llm)
        assert llm.critic_calls == 1
        assert set(s["agent"] for s in out["plan"]) == {"order", "complaint"}

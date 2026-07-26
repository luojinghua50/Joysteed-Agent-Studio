import pytest
from langchain_core.messages import AIMessage
from src.reflection.judge import JudgeReflector, JudgeResult
from src.reflection.error_memory import ErrorMemoryStore, ErrorRecord, SqlErrorMemoryStore
from src.reflection.loop import (
    judge_prewrite, judge_postwrite, judge_synthesize, build_refund_template,
)
from datetime import datetime


class _StubJudge:
    """按预设序列返回 JudgeResult 的假 judge，用于驱动重试分支。"""

    def __init__(self, results):
        self._results = list(results)
        self.calls = 0

    async def evaluate(self, *, user_message, tool_results, agent_response):
        self.calls += 1
        return self._results.pop(0) if self._results else self._results_default()

    @staticmethod
    def _results_default():
        return JudgeResult(score=9.0, passed=True, issues=[], suggestion="")


def _pass(score=9.0):
    return JudgeResult(score=score, passed=True, issues=[], suggestion="")


def _fail(issues, suggestion="修正"):
    return JudgeResult(score=3.0, passed=False, issues=issues, suggestion=suggestion)


class TestJudgeReflector:
    def test_rule_based_good_response(self):
        judge = JudgeReflector()
        result = judge._rule_based_evaluate(
            "我的订单到哪了",
            "您的订单ORD-001目前在杭州转运中心，预计明天送达。如有其他问题请随时告诉我。"
        )
        assert result.passed is True
        assert result.score >= 7.0
        assert result.issues == []

    def test_rule_based_too_short_response(self):
        judge = JudgeReflector()
        result = judge._rule_based_evaluate(
            "我的订单到哪了",
            "不知道"
        )
        assert result.passed is False
        assert result.score < 7.0
        assert any("过短" in issue for issue in result.issues)

    def test_rule_based_internal_info_leak(self):
        judge = JudgeReflector()
        result = judge._rule_based_evaluate(
            "查订单",
            "抱歉，系统出现了 traceback 错误，请稍后重试。Exception: NullPointerError"
        )
        assert result.passed is False
        assert any("内部信息" in issue for issue in result.issues)

    def test_rule_based_inappropriate_promise(self):
        judge = JudgeReflector()
        result = judge._rule_based_evaluate(
            "退款什么时候到",
            "我保证24小时内一定到账，肯定能在明天前解决。"
        )
        assert result.passed is False
        assert any("承诺" in issue for issue in result.issues)


@pytest.mark.asyncio
async def test_judge_evaluate_without_llm():
    judge = JudgeReflector()
    result = await judge.evaluate(
        user_message="查订单",
        tool_results=[],
        agent_response="您好，我来帮您查询订单信息。请提供您的订单号。"
    )
    assert isinstance(result, JudgeResult)
    assert result.passed is True


class TestErrorMemoryStore:
    @pytest.mark.asyncio
    async def test_add_and_get_errors(self):
        store = ErrorMemoryStore(max_size=10)
        record = ErrorRecord(
            agent="order",
            user_message="退款",
            failed_response="我保证一定退款",
            issues=["不当承诺"],
            suggestion="不要承诺具体时间",
        )
        await store.add_error(record)

        context = await store.get_error_context("order")
        assert "不当承诺" in context
        assert "不要承诺具体时间" in context

    @pytest.mark.asyncio
    async def test_get_error_context_empty(self):
        store = ErrorMemoryStore()
        context = await store.get_error_context("unknown_agent")
        assert context == ""

    @pytest.mark.asyncio
    async def test_max_size_enforced(self):
        store = ErrorMemoryStore(max_size=3)
        for i in range(5):
            await store.add_error(ErrorRecord(
                agent="order",
                user_message=f"msg-{i}",
                failed_response=f"resp-{i}",
                issues=[f"issue-{i}"],
                suggestion=f"fix-{i}",
            ))
        assert len(store._store["order"]) == 3

    @pytest.mark.asyncio
    async def test_forbidden_patterns(self):
        store = ErrorMemoryStore()
        for _ in range(3):
            await store.add_error(ErrorRecord(
                agent="order",
                user_message="test",
                failed_response="test",
                issues=["repeated_issue", "unique_issue"],
                suggestion="fix",
            ))

        patterns = await store.get_forbidden_patterns("order")
        assert "repeated_issue" in patterns

    @pytest.mark.asyncio
    async def test_clear(self):
        store = ErrorMemoryStore()
        await store.add_error(ErrorRecord(
            agent="order",
            user_message="test",
            failed_response="test",
            issues=["issue"],
            suggestion="fix",
        ))
        await store.clear("order")
        context = await store.get_error_context("order")
        assert context == ""


async def _sql_store(max_size=20):
    """基于 sqlite in-memory 的持久化错误记忆（须复用同一 engine 才能跨调用可见）。"""
    from src.database import init_db
    factory = await init_db("sqlite+aiosqlite:///:memory:")
    return SqlErrorMemoryStore(factory, max_size=max_size)


class TestSqlErrorMemoryStore:
    @pytest.mark.asyncio
    async def test_add_and_get_context_persisted(self):
        store = await _sql_store()
        await store.add_error(ErrorRecord(
            agent="complaint", user_message="投诉物流",
            failed_response="我保证明天到", issues=["不当承诺"],
            suggestion="不要承诺具体时间",
        ))
        context = await store.get_error_context("complaint")
        assert "不当承诺" in context
        assert "不要承诺具体时间" in context
        # 其他 agent 隔离
        assert await store.get_error_context("order") == ""

    @pytest.mark.asyncio
    async def test_get_context_latest_five(self):
        store = await _sql_store()
        for i in range(7):
            await store.add_error(ErrorRecord(
                agent="complaint", user_message=f"m{i}",
                failed_response=f"r{i}", issues=[f"issue-{i}"], suggestion="fix",
            ))
        context = await store.get_error_context("complaint")
        # 只保留最近 5 条：issue-0/issue-1 被挤出
        assert "issue-6" in context and "issue-2" in context
        assert "issue-0" not in context and "issue-1" not in context

    @pytest.mark.asyncio
    async def test_forbidden_patterns_threshold(self):
        store = await _sql_store()
        for _ in range(3):
            await store.add_error(ErrorRecord(
                agent="complaint", user_message="t", failed_response="t",
                issues=["repeated", "unique"], suggestion="fix",
            ))
        patterns = await store.get_forbidden_patterns("complaint")
        assert "repeated" in patterns  # 出现 3 次 ≥ 阈值 2

    @pytest.mark.asyncio
    async def test_clear(self):
        store = await _sql_store()
        await store.add_error(ErrorRecord(
            agent="complaint", user_message="t", failed_response="t",
            issues=["issue"], suggestion="fix",
        ))
        await store.clear("complaint")
        assert await store.get_error_context("complaint") == ""


class _StubLLM:
    """记录被调用并按序列返回 AIMessage 的假模型（用于 bounded 重写/重融合）。"""

    def __init__(self, replies):
        self._replies = list(replies)
        self.calls = 0

    async def ainvoke(self, messages):
        self.calls += 1
        content = self._replies.pop(0) if self._replies else "兜底"
        return AIMessage(content=content)


class TestJudgePrewrite:
    """A 路径：单意图 complaint，失败 bounded 重写、不重跑工具。"""

    @pytest.mark.asyncio
    async def test_pass_returns_immediately_no_reword(self):
        judge = _StubJudge([_pass()])
        reword = _StubLLM(["不该被调用"])
        original = AIMessage(content="您好，已为您登记投诉，专员将在24小时内联系您。")
        result = await judge_prewrite(
            judge, ErrorMemoryStore(), agent="complaint", skill=None,
            user_msg="投诉", tool_results=[], working_messages=[original],
            reword_llm=reword, max_retries=2,
        )
        assert result is original
        assert reword.calls == 0  # 通过则不重写

    @pytest.mark.asyncio
    async def test_fail_then_reword_and_pass(self):
        judge = _StubJudge([_fail(["不当承诺"]), _pass()])
        reword = _StubLLM(["修正后的稳妥回复"])
        store = ErrorMemoryStore()
        original = AIMessage(content="我保证一定退款")
        result = await judge_prewrite(
            judge, store, agent="complaint", skill=None, user_msg="投诉",
            tool_results=[], working_messages=[original],
            reword_llm=reword, max_retries=2,
        )
        assert result.content == "修正后的稳妥回复"
        assert reword.calls == 1  # bounded 重写一次
        # 失败被记入错误记忆
        assert "不当承诺" in await store.get_error_context("complaint")

    @pytest.mark.asyncio
    async def test_exhausted_returns_last_version(self):
        judge = _StubJudge([_fail(["坏"]), _fail(["还坏"]), _fail(["仍坏"])])
        reword = _StubLLM(["v2", "v3"])
        result = await judge_prewrite(
            judge, ErrorMemoryStore(), agent="complaint", skill=None,
            user_msg="投诉", tool_results=[], working_messages=[AIMessage(content="v1")],
            reword_llm=reword, max_retries=2,
        )
        # max_retries=2 → 3 次评估、2 次重写，返回最后一版
        assert result.content == "v3"
        assert reword.calls == 2


class TestJudgePostwrite:
    """B 路径：退款已落地，失败走确定性模板，绝不重写。"""

    @pytest.mark.asyncio
    async def test_pass_returns_model_reply(self):
        judge = _StubJudge([_pass()])
        resp = AIMessage(content="您的退款 ¥299 已提交，预计 3 天内到账。")
        result = await judge_postwrite(
            judge, ErrorMemoryStore(), agent="order", skill="refund",
            user_msg="退款", tool_result={"amount": 299, "refund_id": "RF-1", "eta": "2026-08-01"},
            response=resp,
        )
        assert result is resp

    @pytest.mark.asyncio
    async def test_fail_uses_deterministic_template(self):
        judge = _StubJudge([_fail(["金额说错"])])
        store = ErrorMemoryStore()
        bad = AIMessage(content="您的退款 ¥999 已到账")  # 乱报金额
        result = await judge_postwrite(
            judge, store, agent="order", skill="refund", user_msg="退款",
            tool_result={"amount": 299, "refund_id": "RF-1", "eta": "2026-08-01",
                         "order_status": "refunded"},
            response=bad,
        )
        # 文案来自工具真实返回，不含错误金额
        assert "299" in result.content
        assert "RF-1" in result.content
        assert "999" not in result.content
        assert "金额说错" in await store.get_error_context("order")


class TestJudgeSynthesize:
    """C 路径：多意图融合，失败重跑融合（零副作用）。"""

    @pytest.mark.asyncio
    async def test_pass_returns_summary(self):
        judge = _StubJudge([_pass()])
        summary = AIMessage(content="融合回复")
        calls = {"n": 0}

        async def _resynth(_fb):
            calls["n"] += 1
            return AIMessage(content="重融合")

        result = await judge_synthesize(
            judge, ErrorMemoryStore(), user_msg="查订单并投诉",
            agent_results={"complaint": {"message": AIMessage(content="投诉已登记")}},
            response=summary, resynth_fn=_resynth, max_retries=2,
        )
        assert result is summary
        assert calls["n"] == 0

    @pytest.mark.asyncio
    async def test_fail_then_resynth(self):
        judge = _StubJudge([_fail(["篡改金额"]), _pass()])
        store = ErrorMemoryStore()

        async def _resynth(_fb):
            return AIMessage(content="纠正后的融合回复")

        result = await judge_synthesize(
            judge, store, user_msg="查订单并退款",
            agent_results={"order": {"message": AIMessage(content="退款 ¥299")}},
            response=AIMessage(content="退款 ¥888"), resynth_fn=_resynth, max_retries=2,
        )
        assert result.content == "纠正后的融合回复"
        assert "篡改金额" in await store.get_error_context("synthesizer")


class TestRefundTemplate:
    def test_full_refund(self):
        text = build_refund_template({
            "amount": 299, "refund_id": "RF-9", "eta": "2026-08-01",
            "order_status": "refunded",
        })
        assert "299" in text and "RF-9" in text and "2026-08-01" in text
        assert "全额退款" in text

    def test_partial_refund_minimal_fields(self):
        text = build_refund_template({"amount": 50, "order_status": "partial_refunded"})
        assert "50" in text
        assert "全额" not in text

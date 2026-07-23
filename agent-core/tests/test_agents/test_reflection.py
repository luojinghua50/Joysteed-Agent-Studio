import pytest
from src.reflection.judge import JudgeReflector, JudgeResult
from src.reflection.error_memory import ErrorMemoryStore, ErrorRecord
from datetime import datetime


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

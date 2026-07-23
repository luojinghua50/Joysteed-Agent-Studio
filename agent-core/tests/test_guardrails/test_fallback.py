import pytest
from src.guardrails.fallback import FallbackHandler, FALLBACK_TEMPLATES


class TestFallbackTemplates:
    def test_all_intents_have_templates(self):
        expected = {"faq", "order", "complaint", "tech_support", "default"}
        assert set(FALLBACK_TEMPLATES.keys()) == expected

    def test_templates_are_nonempty(self):
        for intent, template in FALLBACK_TEMPLATES.items():
            assert len(template) > 0


@pytest.mark.asyncio
async def test_fallback_handler_returns_response():
    handler = FallbackHandler()
    state = {"intent": "order", "current_agent": "order", "failure_count": 0}
    result = await handler.handle_failure(state, Exception("test error"), "test")

    assert "messages" in result
    assert len(result["messages"]) > 0
    assert "订单" in result["messages"][0].content


@pytest.mark.asyncio
async def test_fallback_handler_complaint_forces_handoff():
    handler = FallbackHandler()
    state = {"intent": "complaint", "current_agent": "complaint", "failure_count": 0}
    result = await handler.handle_failure(state, Exception("test"), "test")

    assert result["current_agent"] == "human_handoff"
    assert "人工客服" in result["messages"][0].content


@pytest.mark.asyncio
async def test_fallback_handler_max_failures_triggers_handoff():
    handler = FallbackHandler()
    state = {"intent": "faq", "current_agent": "faq", "failure_count": 3}
    result = await handler.handle_failure(state, Exception("test"), "test")

    assert result["current_agent"] == "human_handoff"


@pytest.mark.asyncio
async def test_fallback_default_template():
    handler = FallbackHandler()
    state = {"intent": "unknown_intent", "current_agent": "unknown", "failure_count": 0}
    result = await handler.handle_failure(state, Exception("test"), "test")

    assert "messages" in result
    assert FALLBACK_TEMPLATES["default"] in result["messages"][0].content

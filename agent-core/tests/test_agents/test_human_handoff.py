import pytest
from src.agents.human_handoff import human_handoff_node


@pytest.mark.asyncio
async def test_human_handoff_returns_message():
    state = {
        "messages": [],
        "intent": "human",
        "customer_id": "C001",
        "session_id": "s001",
    }
    result = await human_handoff_node(state)
    assert "messages" in result
    assert result["resolved"] is True
    assert result["current_agent"] == "human_handoff"
    assert "人工客服" in result["messages"][0].content

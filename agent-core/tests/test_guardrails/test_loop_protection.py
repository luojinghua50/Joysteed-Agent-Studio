import pytest
from src.guardrails.loop_protection import LoopGuard


class TestLoopGuard:
    def test_initial_state(self):
        guard = LoopGuard()
        assert guard.tool_call_count == 0
        assert guard.routing_count == 0
        assert guard.agent_hop_count == 0

    def test_check_tool_call_within_limit(self):
        guard = LoopGuard(max_tool_calls=3)
        assert guard.check_tool_call() is True
        assert guard.check_tool_call() is True
        assert guard.check_tool_call() is True

    def test_check_tool_call_exceeds_limit(self):
        guard = LoopGuard(max_tool_calls=2)
        assert guard.check_tool_call() is True
        assert guard.check_tool_call() is True
        assert guard.check_tool_call() is False

    def test_check_routing_within_limit(self):
        guard = LoopGuard(max_routing_loops=2)
        assert guard.check_routing() is True
        assert guard.check_routing() is True

    def test_check_routing_exceeds_limit(self):
        guard = LoopGuard(max_routing_loops=1)
        assert guard.check_routing() is True
        assert guard.check_routing() is False

    def test_check_agent_hop(self):
        guard = LoopGuard(max_agent_hops=2)
        assert guard.check_agent_hop() is True
        assert guard.check_agent_hop() is True
        assert guard.check_agent_hop() is False

    def test_should_force_end_not_triggered(self):
        guard = LoopGuard(max_tool_calls=10)
        guard.check_tool_call()
        should_end, reason = guard.should_force_end()
        assert should_end is False
        assert reason == ""

    def test_should_force_end_tool_calls(self):
        guard = LoopGuard(max_tool_calls=2)
        for _ in range(3):
            guard.check_tool_call()
        should_end, reason = guard.should_force_end()
        assert should_end is True
        assert reason == "tool_call_limit"

    def test_should_force_end_routing(self):
        guard = LoopGuard(max_routing_loops=1)
        guard.check_routing()
        guard.check_routing()
        should_end, reason = guard.should_force_end()
        assert should_end is True
        assert reason == "routing_loop"

    def test_reset(self):
        guard = LoopGuard()
        guard.check_tool_call()
        guard.check_routing()
        guard.check_agent_hop()
        guard.reset()
        assert guard.tool_call_count == 0
        assert guard.routing_count == 0
        assert guard.agent_hop_count == 0

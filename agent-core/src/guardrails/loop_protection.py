from dataclasses import dataclass, field


@dataclass
class LoopGuard:
    """Loop protection: prevents infinite loops in agent execution."""

    max_tool_calls: int = 10
    max_routing_loops: int = 3
    max_reflection_retries: int = 2
    max_agent_hops: int = 5
    max_skill_steps: int = 8

    tool_call_count: int = field(default=0, init=False)
    routing_count: int = field(default=0, init=False)
    agent_hop_count: int = field(default=0, init=False)

    def check_tool_call(self) -> bool:
        """Returns True if within limit."""
        self.tool_call_count += 1
        return self.tool_call_count <= self.max_tool_calls

    def check_routing(self) -> bool:
        """Returns True if within limit."""
        self.routing_count += 1
        return self.routing_count <= self.max_routing_loops

    def check_agent_hop(self) -> bool:
        """Returns True if within limit."""
        self.agent_hop_count += 1
        return self.agent_hop_count <= self.max_agent_hops

    def should_force_end(self) -> tuple[bool, str]:
        """Check if any loop limit has been exceeded."""
        if self.tool_call_count > self.max_tool_calls:
            return True, "tool_call_limit"
        if self.routing_count > self.max_routing_loops:
            return True, "routing_loop"
        if self.agent_hop_count > self.max_agent_hops:
            return True, "agent_hop_limit"
        return False, ""

    def reset(self):
        """Reset all counters."""
        self.tool_call_count = 0
        self.routing_count = 0
        self.agent_hop_count = 0

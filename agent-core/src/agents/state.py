from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph.message import add_messages


def merge_results(left: dict, right: dict) -> dict:
    """agent_results 的 reducer：按 agent 名幂等归并。

    重试同一个 agent 时覆盖而非重复，避免重复退款/建单等副作用。
    """
    return {**(left or {}), **(right or {})}


def take_latest(left: str, right: str) -> str:
    """current_agent 的 reducer：取最新一次写入（last-write-wins）。

    多意图并行扇出时，同一 super-step 内多个 agent 会各自写 current_agent，
    裸键只允许一个值会触发 INVALID_CONCURRENT_GRAPH_UPDATE。该 reducer 让
    并行写入合法化——current_agent 仅作溯源用途，并行场景下取哪个无副作用。
    """
    return right or left


class CustomerState(TypedDict):
    """Shared state across all agents in the graph."""

    messages: Annotated[list, add_messages]
    intent: str | None
    confidence: float
    customer_id: str
    session_id: str
    customer_info: dict | None
    current_agent: Annotated[str, take_latest]
    needs_approval: bool
    approval_result: str | None
    resolved: bool
    memory_context: str
    failure_count: int
    routing_count: int
    # —— 会话内点对点交接（Level 2 handoff） ——
    handoff_target: str | None
    # —— 多意图编排（orchestrator）相关 ——
    plan: list[dict] | None
    agent_results: Annotated[dict, merge_results]
    is_multi_intent: bool
    # —— 写操作审批闸（Layer 2，单意图路径）——
    # agent 检测到 LLM 想调敏感写工具（create_ticket/apply_refund）时不执行，把待办
    # 写调用 + 已累积的对话上下文存这里，路由到 approval 节点 interrupt 等人工确认。
    # 结构：{"agent": str, "pending_calls": [{"name","args","id"}], "working_messages": list}
    pending_write: dict | None
    # —— 写操作审批闸（Layer 2，多意图批量栅栏路径）——
    # 多意图并行扇出时，多个子 agent 可能各自命中敏感写。它们不能各自 interrupt
    # （并发 interrupt 需 resume-map，复杂），而是把待办写按 agent 名累积到这里，
    # 由 dispatch 收敛到单个 approval 节点一次性交人工（单 interrupt）。带 merge_results
    # reducer：并行子 agent 写不同 key 不撞车（裸键会 INVALID_CONCURRENT_GRAPH_UPDATE）。
    # 结构：{agent_name: {"pending_calls": [{"name","args","id"}], "working_messages": list}}
    # 已处理的 agent 不从本键删除，靠「已写入 agent_results（done）」在消费侧过滤。
    pending_writes: Annotated[dict, merge_results]
    # 多意图批量栅栏 approval_node interrupt 的 resume 决定，透传给 execute_node 逐条分发。
    # 结构：{"approved": bool, "reason": str, "decisions": {call_id: bool}}，也兼容裸 bool。
    approval_decision: dict | bool | None

"""哑路由 chokepoint：不调 LLM 的统一交接出口。

所有会话内交接都穿过这里：校验目标合法、routing_count 封顶防死循环、
记审计日志。这是从「回 supervisor 重判」方案里抽出的唯一有价值部分
（集中 chokepoint），但不付重判那次 LLM 的成本。
"""
import structlog
from langgraph.graph import END

from src.agents.state import CustomerState
from src.config import StabilityConfig

logger = structlog.get_logger()
_stability = StabilityConfig()


def route_after_agent(state: CustomerState) -> str:
    """Agent 执行后的统一出口：决定结束 / 交接 / 兜底转人工。"""
    if state.get("resolved", True):
        return END

    target = state.get("handoff_target")
    hops = state.get("routing_count", 0)

    # 防死循环：交接次数封顶后兜底转人工（复用 StabilityConfig）
    if hops >= _stability.max_routing_loops or not target:
        logger.warning("handoff_capped", hops=hops, target=target)
        return "human_handoff"

    logger.info("agent_handoff", to=target, hops=hops + 1)  # 审计 chokepoint
    return "handoff"  # 先过 bump 节点 +1，再由它路由到 target（边回调不能写 state）


def bump_routing_count(state: CustomerState) -> dict:
    """交接落地前 +1，供 route_after_agent 封顶判断。"""
    return {"routing_count": state.get("routing_count", 0) + 1,
            "intent": state.get("handoff_target")}


def route_to_target(state: CustomerState) -> str:
    """bump 节点出口：+1 后路由到真正的交接目标 agent。"""
    return state.get("handoff_target")

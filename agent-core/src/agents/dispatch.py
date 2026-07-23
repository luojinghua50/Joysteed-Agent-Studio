"""多意图执行编排（确定性骨架，LLM 只在 Agent 内）。

dispatch_plan 作为条件边，在每波结束后重新评估剩余子意图：无依赖的并行、
有依赖的等前序完成后再派发，全部完成则进入汇总。编排逻辑不交给 LLM，
避免「编排本身也会判错」。
"""
import structlog
from langgraph.types import Send

from src.agents.state import CustomerState

logger = structlog.get_logger()


def dispatch_plan(state: CustomerState):
    """按依赖关系分波次派发子意图：无依赖的并行，有依赖的串行。

    用 LangGraph Send API 动态扇出到泛型 Agent node。三个出口：
    - 有 ready 子意图 → Send 扇出（并行/串行）
    - 无 ready 但有待审批的写（pending_writes 未 done）→ "approval" 批量栅栏
    - 无 ready 且无待审批 → "synthesize" 汇总
    """
    plan = state.get("plan") or []
    done = set((state.get("agent_results") or {}).keys())
    # parked：命中敏感写、已累积待审批、但尚未执行（未写 agent_results=未 done）的 agent。
    # 它们不能再被扇出（会重复跑 LLM），也不算 done（写还没落地）。
    parked = {a for a in (state.get("pending_writes") or {}) if a not in done}

    ready = [
        s for s in plan
        if s["agent"] not in done
        and s["agent"] not in parked
        and all(dep in done for dep in s.get("depends_on", []))
    ]

    if ready:
        logger.info("dispatch_wave", ready=[s["agent"] for s in ready], done=list(done))
        # 同一波次内的独立子意图并行执行（各自带改写后的 query）
        return [
            Send(s["agent"], {**state, "_sub_query": s["query"], "is_multi_intent": True})
            for s in ready
        ]

    if parked:
        # 本波已跑完、有待审批的写 → 收敛到单个 approval 节点做批量栅栏审批
        logger.info("dispatch_approval_gate", parked=list(parked), done=list(done))
        return "approval"

    return "synthesize"   # 全部完成（或无可推进项）→ 汇总

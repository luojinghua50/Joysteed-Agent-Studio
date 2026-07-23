"""汇总节点（Synthesizer）：将多意图各子结果融合为一段连贯回复。

多意图各子意图由各自 Agent 处理（已过各自护栏），汇总节点用一次 LLM 把
它们融合成连贯、不重复、有优先级的单条回复。单意图不经此节点。
"""
import structlog
from langchain_core.messages import AIMessage, SystemMessage, HumanMessage

from src.agents.state import CustomerState

logger = structlog.get_logger()


SYNTHESIZE_PROMPT = """你是智能客服的回复融合器。下面是针对用户一条多诉求消息、各专业 Agent 分别处理后的结果。

请把它们融合成一段**连贯、不重复、有优先级**的回复：
1. 合并重复信息，去掉各部分的重复寒暄。
2. 按紧急度/重要性排序（如退款、投诉等敏感诉求优先）。
3. 统一口吻，像一位客服一次性回答，而非拼接多段。
4. 不要新增或编造任何各部分结果里没有的信息。
"""


def _extract_text(result: dict) -> str:
    msg = result.get("message") if isinstance(result, dict) else None
    if msg is None:
        return ""
    return getattr(msg, "content", str(msg))


async def synthesize_node(state: CustomerState, llm) -> dict:
    """将多个子意图的 Agent 产出融合为一段连贯回复。"""
    results = state.get("agent_results") or {}

    if not results:
        return {"messages": [AIMessage(content="抱歉，没能处理您的请求，请稍后再试。")],
                "resolved": True, "current_agent": "synthesizer"}

    if len(results) == 1:  # 退化为单结果，直接用，省一次 LLM
        only = next(iter(results.values()))
        msg = only.get("message") if isinstance(only, dict) else None
        if msg is None:
            msg = AIMessage(content=_extract_text(only))
        return {"messages": [msg], "resolved": True, "current_agent": "synthesizer"}

    parts = "\n\n".join(
        f"【{agent}】{_extract_text(r)}" for agent, r in results.items()
    )
    user_msg = ""
    if state.get("messages"):
        user_msg = getattr(state["messages"][-1], "content", "")

    summary = await llm.ainvoke([
        SystemMessage(content=SYNTHESIZE_PROMPT),
        HumanMessage(content=f"用户原始诉求：{user_msg}\n\n各部分处理结果：\n{parts}"),
    ])
    logger.info("synthesize_done", parts=list(results.keys()))
    return {"messages": [summary], "resolved": True, "current_agent": "synthesizer"}

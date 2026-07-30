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


def _has_high_risk(results: dict, reflection) -> bool:
    """本轮融合是否有高危子意图参与（agent 策略为 judge，如 complaint）。

    仅高危参与才触发 C 路径 judge，保住"只在高风险场景花 Opus"。退款文案已在
    execute_node（B 路径）做过确定性兜底，此处 judge 只防融合篡改。
    """
    if reflection is None:
        return False
    from src.reflection.loop import is_high_risk_agent

    return any(is_high_risk_agent(reflection.config, name) for name in results)


async def synthesize_node(state: CustomerState, llm, reflection=None) -> dict:
    """将多个子意图的 Agent 产出融合为一段连贯回复。

    C 路径：多意图的终端回复在此融合产生。若有高危子意图参与且注入了 reflection，
    对融合结果做 L2 仲裁 + 重跑融合（synthesize 纯融合、零写副作用，重试安全）。
    """
    results = state.get("agent_results") or {}

    if not results:
        return {"messages": [AIMessage(content="抱歉，没能处理您的请求，请稍后再试。")],
                "resolved": True, "current_agent": "synthesizer"}

    user_msg = ""
    if state.get("messages"):
        user_msg = getattr(state["messages"][-1], "content", "")

    if len(results) == 1:  # 退化为单结果，直接用，省一次 LLM
        only = next(iter(results.values()))
        msg = only.get("message") if isinstance(only, dict) else None
        if msg is None:
            msg = AIMessage(content=_extract_text(only))
        # 单结果无融合，但若来自高危 agent 仍过一次质量闸。无法重写（无融合可重跑），
        # resynth 原样返回：等价于对该子结果做一次评估+记忆，不改文案。
        if _has_high_risk(results, reflection):
            async def _passthrough(_fb: str) -> AIMessage:
                return msg
            msg = await _judge_fusion(reflection, user_msg, results, msg, resynth=_passthrough)
        return {"messages": [msg], "resolved": True, "current_agent": "synthesizer"}

    parts = "\n\n".join(
        f"【{agent}】{_extract_text(r)}" for agent, r in results.items()
    )

    async def _fuse(extra: str = "") -> AIMessage:
        sys = SYNTHESIZE_PROMPT + (f"\n\n## 上一轮融合被质检拒绝\n{extra}" if extra else "")
        return await llm.ainvoke([
            SystemMessage(content=sys),
            HumanMessage(content=f"用户原始诉求：{user_msg}\n\n各部分处理结果：\n{parts}"),
        ])

    summary = await _fuse()
    logger.info("synthesize_done", parts=list(results.keys()))

    if _has_high_risk(results, reflection):
        summary = await _judge_fusion(reflection, user_msg, results, summary, resynth=_fuse)

    return {"messages": [summary], "resolved": True, "current_agent": "synthesizer"}


async def _judge_fusion(reflection, user_msg, results, summary, *, resynth) -> AIMessage:
    """调 C 路径 judge：失败带反馈重跑 resynth（融合/透传）。"""
    from src.reflection.loop import judge_synthesize

    return await judge_synthesize(
        reflection.judge, reflection.error_store,
        user_msg=user_msg, agent_results=results, response=summary,
        resynth_fn=resynth, max_retries=reflection.max_retries,
    )

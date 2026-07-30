"""L2 自我反思 helper：judge 仲裁 + 错误记忆 + 分路径兜底。

三条路径对应终端回复的三个诞生地（详见 plan）：
- A. 单意图 complaint 回复（写未落地）→ judge_prewrite：失败可 bounded 重写。
- B. 单意图退款确认文案（写已不可逆）→ judge_postwrite：失败走确定性模板，绝不重写。
- C. 多意图融合回复 → judge_synthesize：失败重跑 synthesize（纯融合、零副作用）。

核心约束：重写只发生在不可逆写操作之前；写之后只评估、只记忆、失败走确定性
兜底。judge（Opus）只评估、从不生成用户可见文案；重写/重融合用对应节点自己的模型。
"""
from dataclasses import dataclass
from datetime import datetime

import structlog
from langchain_core.messages import AIMessage, SystemMessage

from src.reflection.judge import JudgeReflector, JudgeResult
from src.reflection.error_memory import ErrorRecord

logger = structlog.get_logger()


# 退款 skill 名（skill_policies 里标为 judge 的高危写场景），用于识别退款路径。
REFUND_SKILL = "refund"


@dataclass
class ReflectionContext:
    """注入图节点的反思依赖包：judge 仲裁器 + 错误记忆 + 配置。

    graph 构建时创建一份，partial 注入 agent_node/synthesize_node/execute_node。
    error_store 可为内存版或 SqlErrorMemoryStore；judge 持有 Opus LLM。
    """

    judge: JudgeReflector
    error_store: object
    config: object

    @property
    def max_retries(self) -> int:
        return getattr(self.config, "max_retries", 2)


def is_high_risk_agent(config, agent_name: str) -> bool:
    """agent 级反思策略是否为 judge（高危）。"""
    policies = getattr(config, "agent_policies", {}) or {}
    return policies.get(agent_name) == "judge"


def build_refund_template(tool_result: dict) -> str:
    """退款确认的确定性文案：金额/单号/到账日全部取自 apply_refund 的真实返回，
    零模型生成、零漂移。调用方应先确认 tool_result 不含 error 键。"""
    amount = tool_result.get("amount")
    refund_id = tool_result.get("refund_id", "")
    eta = tool_result.get("eta", "")
    order_status = tool_result.get("order_status", "")
    kind = "全额退款" if order_status == "refunded" else "退款"
    parts = [f"您的{kind}"]
    if amount is not None:
        parts.append(f" ¥{amount}")
    parts.append("已提交")
    if refund_id:
        parts.append(f"（退款单号 {refund_id}）")
    if eta:
        parts.append(f"，预计 {eta} 前到账")
    parts.append("。如有疑问可随时联系我们。")
    return "".join(parts)


def _feedback_message(judge_result: JudgeResult) -> SystemMessage:
    """把 judge 的问题+建议包成一条 SystemMessage，供重写/重融合参考。"""
    return SystemMessage(content=(
        f"上一轮回复被质量检查拒绝。\n"
        f"问题：{'; '.join(judge_result.issues)}\n"
        f"修正建议：{judge_result.suggestion}\n"
        f"请据此重新生成回复，不要引入工具结果里没有的信息。"
    ))


async def _record_error(error_store, *, agent, skill, user_msg, response_text,
                        judge_result: JudgeResult, attempt: int):
    """持久化一条被拒回复（错误记忆）。error_store 缺失时静默跳过。"""
    if error_store is None:
        return
    try:
        await error_store.add_error(ErrorRecord(
            timestamp=datetime.now(),
            agent=agent,
            skill=skill,
            user_message=user_msg,
            failed_response=response_text,
            issues=judge_result.issues,
            suggestion=judge_result.suggestion,
            retry_count=attempt,
        ))
    except Exception as e:  # 记忆持久化失败绝不阻断主回复
        logger.warning("error_memory_persist_failed", agent=agent, error=str(e))


async def judge_prewrite(
    judge: JudgeReflector,
    error_store,
    *,
    agent: str,
    skill: str | None,
    user_msg: str,
    tool_results: list[dict],
    working_messages: list,
    reword_llm,
    max_retries: int = 2,
) -> AIMessage:
    """A 路径（单意图 complaint，写未落地）：evaluate → 失败则 bounded 重写。

    重写不重跑工具（complaint 的 update_customer_tag 已在 executor 内执行，重跑会
    重复打标签）：只在 working_messages（已含工具结果）尾部追加 judge 反馈，用
    reword_llm **不 bind 工具** 重新措辞。循环至通过或耗尽，耗尽返回最后一版。
    """
    messages = list(working_messages)
    response = messages[-1] if messages and isinstance(messages[-1], AIMessage) else None

    for attempt in range(max_retries + 1):
        text = getattr(response, "content", "") if response is not None else ""
        result = await judge.evaluate(
            user_message=user_msg, tool_results=tool_results, agent_response=text
        )
        if result.passed:
            if attempt:
                logger.info("judge_prewrite_recovered", agent=agent, attempts=attempt)
            return response if response is not None else AIMessage(content=text)

        await _record_error(
            error_store, agent=agent, skill=skill, user_msg=user_msg,
            response_text=text, judge_result=result, attempt=attempt,
        )
        if attempt == max_retries:
            logger.warning("judge_prewrite_exhausted", agent=agent, issues=result.issues)
            return response if response is not None else AIMessage(content=text)

        messages = messages + [_feedback_message(result)]
        response = await reword_llm.ainvoke(messages)
        messages = messages + [response]

    return response if response is not None else AIMessage(content="")


async def judge_postwrite(
    judge: JudgeReflector,
    error_store,
    *,
    agent: str,
    skill: str | None,
    user_msg: str,
    tool_result: dict,
    response: AIMessage,
) -> AIMessage:
    """B 路径（单意图退款，写已不可逆）：evaluate → 失败走确定性模板兜底。

    绝不重写、不重跑写。tool_result 含 error 键交给调用方（走 human_handoff），
    此处只在成功退款上做质量闸：judge 不通过则丢弃模型文案，用 apply_refund 的
    真实返回值拼确定性模板（金额/单号来自工具，零漂移）。
    """
    text = getattr(response, "content", "") or ""
    result = await judge.evaluate(
        user_message=user_msg, tool_results=[tool_result], agent_response=text
    )
    if result.passed:
        return response

    await _record_error(
        error_store, agent=agent, skill=skill, user_msg=user_msg,
        response_text=text, judge_result=result, attempt=0,
    )
    logger.warning("judge_postwrite_fallback_template", agent=agent, issues=result.issues)
    return AIMessage(content=build_refund_template(tool_result))


async def judge_synthesize(
    judge: JudgeReflector,
    error_store,
    *,
    user_msg: str,
    agent_results: dict,
    response: AIMessage,
    resynth_fn,
    max_retries: int = 2,
) -> AIMessage:
    """C 路径（多意图融合）：evaluate 融合结果 → 失败则重跑 synthesize。

    基准 = 各子结果原文（检查融合有无新增/篡改，尤其退款金额/单号）。synthesize
    纯融合、零写副作用，重试完全安全。resynth_fn(feedback_text) -> AIMessage 由
    synthesize_node 提供（带 judge 反馈重跑融合）。
    """
    def _subresults_text() -> list[dict]:
        out = []
        for name, r in (agent_results or {}).items():
            msg = r.get("message") if isinstance(r, dict) else None
            out.append({"agent": name, "content": getattr(msg, "content", str(msg))})
        return out

    subresults = _subresults_text()
    for attempt in range(max_retries + 1):
        text = getattr(response, "content", "") or ""
        result = await judge.evaluate(
            user_message=user_msg, tool_results=subresults, agent_response=text
        )
        if result.passed:
            if attempt:
                logger.info("judge_synthesize_recovered", attempts=attempt)
            return response

        await _record_error(
            error_store, agent="synthesizer", skill=None, user_msg=user_msg,
            response_text=text, judge_result=result, attempt=attempt,
        )
        if attempt == max_retries:
            logger.warning("judge_synthesize_exhausted", issues=result.issues)
            return response

        response = await resynth_fn(_feedback_message(result).content)

    return response

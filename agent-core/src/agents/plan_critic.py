"""派发前 plan 校验（Layer 1）：dispatch 扇出前拦截高危误派。

多意图 plan 由 supervisor LLM 一次性拆出、dispatch 确定性照单派发，中途无
自愈能力（见 technical-design §6.2.5）。规划器把「问退货政策」这类 benign
诉求误派给高危 agent（complaint 会打客户标签 / 建投诉工单 / 理赔）时，工具会
在 agent 执行中直接落地真实写副作用，事后要人工清理。

本模块在派发前用一次轻量 LLM 复核分派依据：高危 agent 被分派时，用户消息
里必须有对应信号（投诉 / 不满 / 理赔），否则剔除该子意图或重派到安全 agent。

约束：
- 仅在 plan 含高危 agent 时触发（ReflectionConfig.agent_policies == "judge"），
  benign 多意图不付这次成本。
- 任何失败一律 fail-open（保留原 plan）：critic 是概率性降误派器，不是硬闸；
  硬保证由写操作审批闸（Layer 2）负责，不能让校验本身崩掉正常路由。
- 重派只允许指向【非高危】agent，绝不把误派「修正」成另一个高危分派。
"""
import json
import re

import structlog
from langchain_core.messages import SystemMessage
from pydantic import BaseModel, Field

from src.agents.registry import AGENT_REGISTRY
from src.config import ReflectionConfig

logger = structlog.get_logger()

_VALID_AGENTS = set(AGENT_REGISTRY.keys())
_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL)


class RejectedIntent(BaseModel):
    agent: str
    reason: str = ""
    reassign_to: str | None = None  # 建议改派的目标；仅当为非高危合法 agent 时采纳


class PlanVerdict(BaseModel):
    rejected: list[RejectedIntent] = Field(default_factory=list)


CRITIC_PROMPT = """你是多意图客服路由的【分派校验器】。下面是用户原始消息，以及系统已拆分出的子意图分派计划。

你的唯一任务：检查每个子意图分派的 agent 是否有充分依据，找出【误派】。

## 重点审查（高危 agent，会产生真实副作用：打客户标签 / 建工单 / 理赔）
- complaint（投诉 / 理赔）：用户消息必须含明确的不满、投诉、要求赔偿、情绪激烈等信号。
  仅仅是询问政策、咨询流程、查询信息 —— 不构成投诉，把它分派给 complaint 属误派。

## 判定原则
- 宁可放过，不可误伤：只有当分派明显缺乏依据时才拒绝。
- 不确定 → 不拒绝。
- 若该诉求本该由别的 agent 处理，可在 reassign_to 给出建议目标（faq / order / tech_support）。

## 输入
用户原始消息：{user_message}
子意图计划：{plan}

## 输出 JSON（只列被拒绝的误派，无误派则 rejected 为空）
{{"rejected": [{{"agent": "complaint", "reason": "用户仅询问退货政策，无任何投诉/理赔信号", "reassign_to": "faq"}}]}}
"""


def high_risk_agents(config: ReflectionConfig) -> set[str]:
    """高危 agent = reflection 策略被标为 judge 的（默认 {complaint}）。"""
    return {a for a, p in (config.agent_policies or {}).items() if p == "judge"}


def plan_needs_critique(sub_intents: list[dict], config: ReflectionConfig) -> bool:
    """仅当校验启用且 plan 含高危 agent 时才值得付这次 LLM 成本。"""
    if not config.enabled:
        return False
    hi = high_risk_agents(config)
    return any(s["agent"] in hi for s in sub_intents)


def _extract_json(text: str) -> str:
    """从 LLM 文本响应中提取 JSON 串（剥代码围栏 / 截取首尾花括号）。"""
    if not text:
        return ""
    m = _FENCE_RE.search(text)
    if m:
        return m.group(1).strip()
    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end != -1 and end > start:
        return text[start : end + 1].strip()
    return text.strip()


def _user_message(context_messages: list) -> str:
    """取最近一条用户消息作为校验依据。"""
    for m in reversed(context_messages or []):
        if m.__class__.__name__ == "HumanMessage" or getattr(m, "type", "") == "human":
            return getattr(m, "content", "")
    return getattr(context_messages[-1], "content", "") if context_messages else ""


def _apply_verdict(sub_intents: list[dict], verdict: PlanVerdict, config: ReflectionConfig) -> list[dict]:
    """按裁决剔除 / 重派高危误派；非高危的拒绝一律忽略（保守，防 critic 误伤）。"""
    hi = high_risk_agents(config)
    rejected = {r.agent: r for r in verdict.rejected}
    out: list[dict] = []
    taken: set[str] = set()

    for s in sub_intents:
        agent = s["agent"]
        r = rejected.get(agent)
        # 未被拒 / 被拒但非高危 → 一律保留（只对高危误派动手）
        if r is None or agent not in hi:
            out.append(s)
            taken.add(agent)
            continue

        target = r.reassign_to
        # 重派仅允许指向非高危、合法、未占用的 agent；否则直接剔除
        if target and target in _VALID_AGENTS and target not in hi and target not in taken:
            logger.warning("plan_critic_reassign", frm=agent, to=target, reason=r.reason[:120])
            out.append({**s, "agent": target})
            taken.add(target)
        else:
            logger.warning("plan_critic_drop", agent=agent, reason=r.reason[:120])
            # 剔除：宁可漏答一个子意图（用户下一轮可重提），不可留下错误写副作用

    # 清理指向已移除 agent 的悬空依赖
    final = {s["agent"] for s in out}
    return [{**s, "depends_on": [d for d in s.get("depends_on", []) if d in final]} for s in out]


async def critique_plan(llm, context_messages: list, sub_intents: list[dict], config: ReflectionConfig) -> list[dict]:
    """派发前校验；任何异常 / 解析失败一律 fail-open 返回原 plan。"""
    try:
        plan_repr = [{"agent": s["agent"], "query": s.get("query", "")} for s in sub_intents]
        prompt = CRITIC_PROMPT.format(
            user_message=_user_message(context_messages),
            plan=json.dumps(plan_repr, ensure_ascii=False),
        )
        resp = await llm.ainvoke([SystemMessage(content=prompt)])
        verdict = PlanVerdict.model_validate_json(_extract_json(resp.content))
        if not verdict.rejected:
            return sub_intents
        result = _apply_verdict(sub_intents, verdict, config)
        logger.info("plan_critic_applied", before=len(sub_intents), after=len(result))
        return result
    except Exception as e:
        logger.warning("plan_critic_failed_open", error_type=type(e).__name__, error=str(e)[:200])
        return sub_intents

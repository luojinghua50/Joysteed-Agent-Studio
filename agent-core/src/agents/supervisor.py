"""Supervisor 节点：同一节点按意图复杂度承担 Router / Orchestrator 两种职能。

- 单意图请求：只做 Router（分类一次 → 选一个目标 → 退场）。
- 多意图请求：承担 Orchestrator 职能，分解为带依赖关系的子意图计划，
  交给 dispatch（确定性骨架）分波派发，最后由 synthesizer 融合。

术语界定见 technical-design.md §4.7.0。
"""

import re

import structlog
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import SystemMessage
from pydantic import BaseModel, Field

from src.agents.registry import AGENT_REGISTRY
from src.agents.state import CustomerState
from src.agents.plan_critic import critique_plan, plan_needs_critique
from src.config import StabilityConfig, ReflectionConfig
from src.guardrails.retry import retry_with_backoff

logger = structlog.get_logger()

_VALID_AGENTS = set(AGENT_REGISTRY.keys())
_stability = StabilityConfig()
_reflection = ReflectionConfig()


SUPERVISOR_SYSTEM_PROMPT = """你是智能客服路由器。根据用户消息判断意图并选择最合适的处理 Agent。

## 路由规则
- **faq**：产品功能、使用方法、政策咨询、闲聊、打招呼、一般性提问（包括天气、时间等无法回答的问题）
- **order**：订单相关（查询、修改、退款、取消、物流追踪）
- **complaint**：投诉、不满、要求赔偿、情绪激动
- **tech_support**：产品故障、技术问题、报错信息、操作异常
- **human**：用户明确要求人工客服（如"转人工"/"找经理"），或涉及法律纠纷等严重问题

## 判断要点
1. 优先识别用户的核心诉求
2. 闲聊、打招呼、问好 → faq（不要转人工）
3. 如果消息包含情绪词（"太差了"/"气死了"/"投诉"）→ complaint
4. 包含订单号或明确提到订单操作 → order
5. 如果有对话历史，结合上下文判断（比如用户在回复上一轮的追问）→ 保持上一轮的意图
6. 只有用户明确说"转人工"/"找人工客服"时才选 human

## 输出格式
只输出意图标签，不要输出其他内容。可选值：faq, order, complaint, tech_support, human
"""


SUPERVISOR_PLAN_PROMPT = """你是智能客服的意图分解器。判断用户单条消息里包含几个独立诉求，并分解为子意图计划。

## 可用 Agent
- faq：产品功能、使用方法、政策咨询
- order：订单查询、修改、退款、物流追踪
- complaint：投诉、不满、要求赔偿
- tech_support：产品故障、技术问题
- human：明确要求人工或严重纠纷

## 判断规则
1. 单一诉求 → is_multi_intent=false，sub_intents 只含一个。
2. 多个独立诉求（如"查物流 + 投诉态度 + 问退货政策"）→ is_multi_intent=true，逐个拆成 sub_intent。
3. 每个 sub_intent 的 query 改写成可独立处理的完整诉求（补全指代）。
4. depends_on 填该子意图依赖的其他子意图的 agent 名；无依赖留空。
   - 例："赔偿"依赖先查到的"订单"信息 → complaint.depends_on=["order"]。
5. agent 字段只能取：faq, order, complaint, tech_support, human。
"""


class SubIntent(BaseModel):
    agent: str = Field(description="目标 Agent，取值：faq/order/complaint/tech_support/human")
    query: str = Field(description="该子意图对应的、改写后的独立诉求")
    depends_on: list[str] = Field(
        default_factory=list, description="依赖的其他子意图 agent 名（空=可并行）"
    )


class IntentPlan(BaseModel):
    is_multi_intent: bool
    sub_intents: list[SubIntent] = Field(default_factory=list)


# 匹配 ```json ... ``` 或 ``` ... ``` 代码围栏，捕获其中的内容
_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL)


def _extract_json(text: str) -> str:
    """从 LLM 文本响应中提取 JSON 串。

    litellm 中转下结构化输出有时退化成纯文本，模型会把 JSON 裹进 markdown 代码
    围栏（```json ... ```）。langchain 的 with_structured_output 直接 json.loads
    这段带围栏的字符串会抛 json_invalid。这里先剥围栏，再退而求其次截取第一个
    `{` 到最后一个 `}`，最大化解析成功率。
    """
    if not text:
        return ""
    m = _FENCE_RE.search(text)
    if m:
        return m.group(1).strip()
    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end != -1 and end > start:
        return text[start : end + 1].strip()
    return text.strip()


def _parse_intent(content: str) -> str:
    """Parse the intent from LLM response."""
    content = (content or "").strip().lower()
    valid_intents = {"faq", "order", "complaint", "tech_support", "human"}
    for intent in valid_intents:
        if intent in content:
            return intent
    return "human"


def _intent_to_agent(intent: str) -> str:
    """Map intent label to agent node name."""
    mapping = {
        "faq": "faq",
        "order": "order",
        "complaint": "complaint",
        "tech_support": "tech_support",
        "human": "human_handoff",
    }
    return mapping.get(intent, "human_handoff")


def route_by_intent(state: CustomerState) -> str:
    """Conditional edge: route to agent node based on single intent."""
    return _intent_to_agent(state.get("intent"))


def _normalize_plan(plan: IntentPlan) -> list[dict]:
    """过滤非法 agent，保证 depends_on 只引用计划内的合法 agent。"""
    valid = [s for s in plan.sub_intents if s.agent in _VALID_AGENTS]
    names = {s.agent for s in valid}
    out = []
    for s in valid:
        out.append(
            {
                "agent": s.agent,
                "query": s.query,
                "depends_on": [d for d in s.depends_on if d in names],
            }
        )
    return out


def _single_intent_result(intent: str) -> dict:
    return {
        "intent": intent,
        "confidence": 0.9 if intent != "human" else 0.5,
        "current_agent": _intent_to_agent(intent),
        "is_multi_intent": False,
    }


async def _decompose(llm: BaseChatModel, context_messages: list, memory_context: str) -> IntentPlan:
    """结构化分解（承担 Orchestrator 职能的前置判断）。

    不用 with_structured_output：litellm 中转下它常退化成纯文本 + 代码围栏，
    内部 json.loads 直接抛 json_invalid。改为自取文本 → 剥围栏 → pydantic 校验，
    解析/校验失败仍抛 ValidationError，由上层有界重试 + 三层降级接管。
    """
    prompt = SUPERVISOR_PLAN_PROMPT + (
        "\n\n## 输出格式\n严格输出单个 JSON 对象，不要任何解释或代码围栏。"
        '\n形如：{"is_multi_intent": false, "sub_intents": '
        '[{"agent": "order", "query": "...", "depends_on": []}]}'
    )
    if memory_context:
        prompt += f"\n\n## 用户背景信息\n{memory_context}"
    response = await llm.ainvoke([SystemMessage(content=prompt), *context_messages])
    return IntentPlan.model_validate_json(_extract_json(response.content))


async def _classify_single(llm: BaseChatModel, context_messages: list, memory_context: str) -> dict:
    """单意图纯文本分类（Router 职能 / 结构化分解失败时的功能降级）。"""
    prompt = SUPERVISOR_SYSTEM_PROMPT
    if memory_context:
        prompt += f"\n\n## 用户背景信息\n{memory_context}"
    response = await llm.ainvoke([SystemMessage(content=prompt), *context_messages])
    return _single_intent_result(_parse_intent(response.content))


def _result_from_sub_intents(sub_intents: list[dict], is_multi: bool) -> dict:
    """把（已归一化、已过 critic 的）子意图列表收口为 state 更新。

    单意图或 critic 剔除后只剩 <=1 个 → collapse 成单意图直接路由；否则写多意图计划。
    """
    if not is_multi or len(sub_intents) <= 1:
        only = sub_intents[0]["agent"] if sub_intents else "human"
        return _single_intent_result(only)

    logger.info(
        "supervisor_multi_intent", count=len(sub_intents), agents=[s["agent"] for s in sub_intents]
    )
    return {
        "is_multi_intent": True,
        "plan": sub_intents,
        "intent": "multi",
        "current_agent": "supervisor",
    }


def _plan_to_result(plan: IntentPlan) -> dict:
    """把结构化计划归一化为 state 更新（单意图直接路由 / 多意图写计划）。"""
    return _result_from_sub_intents(_normalize_plan(plan), plan.is_multi_intent)


async def supervisor_node(state: CustomerState, llm: BaseChatModel) -> dict:
    """单意图→直接路由；多意图→分解为带依赖的子意图计划。

    生产级三层降级（编排判错是多 Agent 系统最大失败源，supervisor 又是 live
    路径上的安全网，故绝不向图抛异常）：
      L1 结构化分解，瞬时/解析错误按 StabilityConfig 有界重试（指数退避）。
      L2 分解彻底失败 → 降级到单意图纯文本分类（功能完整，同样有界重试）。
      L3 连分类都失败（LLM 全挂）→ 兜底路由人工，保证 turn 不崩。
    """
    messages = state["messages"]
    if not messages:
        return {
            "intent": "human",
            "confidence": 0.0,
            "current_agent": "human_handoff",
            "is_multi_intent": False,
        }

    context_messages = messages[-6:]
    memory_context = state.get("memory_context", "")

    retry_enabled = _stability.retry_enabled
    max_parse_retries = _stability.output_parse_retries if retry_enabled else 0
    max_llm_retries = _stability.llm_max_retries if retry_enabled else 0
    base_delay = _stability.llm_retry_base_delay

    # —— L1：结构化分解（带有界重试；解析失败 re-roll 常能修复） ——
    try:
        plan: IntentPlan = await retry_with_backoff(
            lambda: _decompose(llm, context_messages, memory_context),
            max_retries=max_parse_retries,
            base_delay=base_delay,
        )
        sub_intents = _normalize_plan(plan)
        # 派发前 plan 校验（Layer 1）：仅多意图且含高危 agent 时触发，拦截无
        # 依据的高危误派（如「问退货政策」被派给 complaint）。fail-open，不抛。
        if plan.is_multi_intent and plan_needs_critique(sub_intents, _reflection):
            sub_intents = await critique_plan(llm, context_messages, sub_intents, _reflection)
        return _result_from_sub_intents(sub_intents, plan.is_multi_intent)
    except Exception as e:
        logger.warning(
            "supervisor_decompose_failed", error_type=type(e).__name__, error=str(e)[:300]
        )

    # —— L2：降级到单意图纯文本分类（功能降级，非错误模板；带有界重试） ——
    try:
        return await retry_with_backoff(
            lambda: _classify_single(llm, context_messages, memory_context),
            max_retries=max_llm_retries,
            base_delay=base_delay,
        )
    except Exception as e:
        # —— L3：LLM 整体不可用 → 兜底转人工，绝不抛异常崩掉整个 turn ——
        logger.error(
            "supervisor_classify_failed_route_human",
            error_type=type(e).__name__,
            error=str(e)[:300],
        )
        return _single_intent_result("human")

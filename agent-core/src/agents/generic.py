"""泛型 Agent Node：所有业务 Agent 共用的执行体。

取代 faq/order/complaint/tech_support 四个近乎克隆的 node 文件——行为差异
全部来自数据（AgentSpec）。human_handoff（无 LLM/无工具）与 supervisor
（纯 LLM 路由）是真正的特例，不走泛型 node。

两种执行模式：
- 单意图：直接路由进来，产出写 `messages`，出口走哑路由 chokepoint。
- 多意图：经 dispatch 用 Send 扇出（带改写后的 `_sub_query`），产出写
  `agent_results[spec.name]`，供 synthesizer 融合；不直接污染 `messages`。
"""
import structlog
from langchain_core.messages import AIMessage, HumanMessage

from src.agents.state import CustomerState
from src.agents.registry import AgentSpec
from src.agents.handoff import detect_handoff
from src.agents.approval import APPROVAL_REQUIRED_TOOLS
from src.mcp_client.client import MCPClientManager
from src.tools.mcp_adapter import get_agent_tools
from src.tools.executor import run_agent_with_tools, ApprovalRequired

logger = structlog.get_logger()


async def _resolve_prompt(spec: AgentSpec, prompts) -> str:
    """从 PromptRegistry 取 prompt；取不到时退回最小占位，保证可运行。"""
    if prompts is not None:
        try:
            return await prompts.get(spec.prompt_id)
        except Exception as e:
            logger.warning("prompt_load_failed", prompt_id=spec.prompt_id, error=str(e))
    return f"你是{spec.name}客服，请专业、简洁地处理用户的请求。"


async def agent_node(state: CustomerState, *, spec: AgentSpec, llm, mcp: MCPClientManager,
                     prompts=None, approval_enabled: bool = False) -> dict:
    """所有业务 Agent 共用的执行体，行为差异全部来自 AgentSpec 数据。"""
    is_multi = state.get("is_multi_intent", False)
    sub_query = state.get("_sub_query")

    # 多意图模式下用该子意图改写后的独立诉求；单意图沿用历史窗口
    if is_multi and sub_query:
        messages = [HumanMessage(content=sub_query)]
    else:
        messages = state["messages"][-spec.history_window:]

    try:
        tools = await get_agent_tools(spec.name, mcp)
    except Exception:
        tools = []

    # prompt 来自 PromptRegistry（不再内联常量），支持版本化与热更新
    prompt = await _resolve_prompt(spec, prompts)
    if state.get("memory_context"):
        prompt += f"\n\n## 用户记忆上下文\n{state['memory_context']}"
    # 交接规则仅在单意图模式注入：多意图的路由已由 supervisor 规划器在扇出前
    # 定完，子意图 agent 无交接决策权，注入该规则只会诱导它输出无人消费的
    # [HANDOFF:x] 标记，泄漏进 synthesizer 融合结果污染最终回复。
    if spec.can_handoff_to and not is_multi:
        prompt += (
            f"\n\n## 交接规则\n如果用户诉求完全不属于你的职责，只回一行 "
            f"`[HANDOFF:目标]`，目标可选：{', '.join(spec.can_handoff_to)}。"
        )

    # 写操作审批闸在单意图与多意图路径均生效。关闭 approval 时 protected 为空集，
    # executor 行为与改造前完全一致。多意图下命中敏感写不各自 interrupt，而是累积到
    # pending_writes（见下），由 dispatch 收敛到单个 approval 节点做批量栅栏审批。
    protected = APPROVAL_REQUIRED_TOOLS if approval_enabled else set()

    try:
        response: AIMessage = await run_agent_with_tools(
            llm=llm, tools=tools, system_prompt=prompt, messages=messages,
            protected_tools=protected,
        )
    except ApprovalRequired as ar:
        # 敏感写工具待确认：写副作用尚未落地——execute 节点在批准后才执行。
        if is_multi:
            # 多意图：累积到 pending_writes[agent]（带 reducer，并行写不撞车），
            # 不设 needs_approval、不写 agent_results（该 agent 未 done）。dispatch
            # 会在本波跑完后收敛到单个 approval 节点做批量栅栏审批。
            return {
                "pending_writes": {
                    spec.name: {
                        "pending_calls": ar.pending_calls,
                        "working_messages": ar.working_messages,
                    }
                },
                "current_agent": spec.name,
            }
        # 单意图：存 pending_write，路由到 approval 节点 interrupt。
        return {
            "needs_approval": True,
            "pending_write": {
                "agent": spec.name,
                "pending_calls": ar.pending_calls,
                "working_messages": ar.working_messages,
            },
            "current_agent": spec.name,
        }

    # 多意图：产出写入 agent_results，交给 synthesizer 融合（不污染 messages）
    if is_multi:
        return {
            "agent_results": {spec.name: {"message": response}},
            "current_agent": spec.name,
        }

    # 单意图：检测交接意图 → 置 resolved=False，由哑路由 chokepoint 接管
    target = detect_handoff(response.content, allowed=spec.can_handoff_to)
    if target:
        return {"resolved": False, "handoff_target": target, "current_agent": spec.name}

    return {"messages": [response], "resolved": True, "current_agent": spec.name}

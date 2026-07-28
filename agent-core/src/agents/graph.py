"""主编排图：数据驱动建图 + 多意图编排接线。

- 业务 Agent 全部由 AGENT_REGISTRY 循环生成（新增一行 spec 即接入，无需改本文件）。
- supervisor 出口：单意图走 route_by_intent；多意图走 dispatch 扇出。
- 业务 Agent 出口：多意图回 dispatch 评估下一波；单意图走哑路由 chokepoint。
"""

from functools import partial

from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph

from src.agents.approval import approval_node, execute_node
from src.agents.dispatch import dispatch_plan
from src.agents.generic import agent_node
from src.agents.human_handoff import human_handoff_node
from src.agents.registry import AGENT_REGISTRY
from src.agents.router import bump_routing_count, route_after_agent, route_to_target
from src.agents.state import CustomerState
from src.agents.supervisor import route_by_intent, supervisor_node
from src.agents.synthesizer import synthesize_node
from src.config import Settings


def create_llm(settings: Settings, model_name: str | None = None) -> ChatOpenAI:
    """Create an LLM instance configured to use LiteLLM proxy."""
    return ChatOpenAI(
        model=model_name or settings.model_main,
        base_url=settings.litellm_base_url,
        api_key=settings.litellm_api_key,
        temperature=0.1,
    )


def build_graph(
    settings=None, memory_manager=None, prompts=None, mcp=None,
    approval_enabled: bool | None = None, error_store=None,
) -> StateGraph:
    """Build the main agent orchestration graph (data-driven + multi-intent).

    error_store 非空时启用 L2 自我反思：构建 judge（Opus，judge_model）+
    ReflectionContext，注入 agent_node（A）/synthesize_node（C）/execute_node（B）。
    为空则三节点行为与接入前完全一致（reflection=None，一行不变）。
    """
    if settings is None:
        settings = Settings()
    if mcp is None:
        from src.mcp_client.client import MCPClientManager

        mcp = MCPClientManager(settings)
    approval_enabled = bool(
        getattr(settings, "approval_enabled", False)
        if approval_enabled is None
        else approval_enabled
    )

    # 工作记忆句柄：从 memory_manager 取 WorkingMemory，注入业务/执行节点写实体。
    # 为空则节点不写工作记忆（行为与接入前一致）。
    working = getattr(memory_manager, "working", None) if memory_manager is not None else None

    # —— L2 反思上下文（仅当注入 error_store 时启用）——
    reflection = None
    if error_store is not None:
        from src.config import ReflectionConfig
        from src.reflection.judge import JudgeReflector
        from src.reflection.loop import ReflectionContext

        reflection_cfg = ReflectionConfig()
        if reflection_cfg.enabled:
            judge_llm = create_llm(settings, reflection_cfg.judge_model)
            reflection = ReflectionContext(
                judge=JudgeReflector(config=reflection_cfg, llm=judge_llm),
                error_store=error_store,
                config=reflection_cfg,
            )

    graph = StateGraph(CustomerState)

    # —— 特例节点：supervisor（纯 LLM 路由/分解）、human_handoff（无 LLM） ——
    graph.add_node(
        "supervisor", partial(supervisor_node, llm=create_llm(settings, settings.model_main))
    )
    graph.add_node("human_handoff", human_handoff_node)

    # —— 多意图编排节点 ——
    graph.add_node(
        "synthesize",
        partial(synthesize_node, llm=create_llm(settings, settings.model_main), reflection=reflection),
    )

    # —— 单意图交接 chokepoint：+1 计数 + 审计，再路由到目标 agent ——
    # 边回调不能写 state，故把 routing_count +1 抽成独立节点，穿过它才落地交接。
    graph.add_node("handoff", bump_routing_count)

    # —— 写操作审批闸（Layer 2）：approval interrupt 等确认；execute 批准后落地 ——
    # execute_node 用 make_llm 按 agent 名建对应 model_key 的 llm（与业务节点一致）。
    def make_llm(agent_name: str):
        spec = AGENT_REGISTRY.get(agent_name)
        model = (
            getattr(settings, spec.model_key, settings.model_main) if spec else settings.model_main
        )
        return create_llm(settings, model)

    graph.add_node("approval", approval_node)
    graph.add_node(
        "execute",
        partial(execute_node, make_llm=make_llm, mcp=mcp, prompts=prompts,
                reflection=reflection, memory=working),
    )

    # 业务 Agent 全部由 registry 循环生成 —— 新增一行 spec 即接入，无需改本函数
    for name, spec in AGENT_REGISTRY.items():
        llm = create_llm(settings, getattr(settings, spec.model_key, settings.model_main))
        graph.add_node(
            name,
            partial(
                agent_node,
                spec=spec,
                llm=llm,
                mcp=mcp,
                prompts=prompts,
                approval_enabled=approval_enabled,
                reflection=reflection,
                memory=working,
            ),
        )

    graph.set_entry_point("supervisor")

    # supervisor 出口：单意图走 route_by_intent；多意图走 dispatch 扇出锚点
    # intent_map = {"faq": "faq","order": "order","complaint": "complaint","tech_support": "tech_support","human_handoff": "human_handoff",}
    intent_map = {n: n for n in AGENT_REGISTRY} | {"human_handoff": "human_handoff"}

    def route_from_supervisor(state) -> str:
        return "dispatch" if state.get("is_multi_intent") else route_by_intent(state)

    graph.add_conditional_edges(
        "supervisor",
        route_from_supervisor,  # route_from_supervisor(state) 来决定下一跳
        {**intent_map, "dispatch": "dispatch"},
    )

    # dispatch 作为动态扇出边：用 Send 并行/串行派发，全部完成后进 synthesize。
    # dispatch_plan 既是该锚点节点的条件函数，节点本身是 pass-through（不改 state）。
    graph.add_node("dispatch", lambda state: {})
    graph.add_conditional_edges(
        "dispatch",
        dispatch_plan,
        # dispatch 三出口：扇出到业务 agent / 批量栅栏 approval / 汇总 synthesize
        {**{n: n for n in AGENT_REGISTRY}, "approval": "approval", "synthesize": "synthesize"},
    )

    # 业务 Agent 出口：待审批 → approval 闸；多意图回 dispatch；单意图走哑路由 chokepoint
    # 单意图交接不直接跳目标，而是先进 handoff 节点 +1，故出口映射到 "handoff"。
    handoff_map = {
        END: END,
        "human_handoff": "human_handoff",
        "handoff": "handoff",
        "approval": "approval",
    }

    def route_after_agent_v2(state) -> str:
        if state.get("needs_approval") and state.get("pending_write"):
            return "approval"  # 敏感写待确认，先进审批闸
        if state.get("is_multi_intent"):
            return "dispatch"  # 回编排器评估剩余子意图
        return route_after_agent(state)  # 单意图：哑路由 chokepoint

    # 针对每一个 agent 节点，添加出口路由规则：route_after_agent_v2
    for name in AGENT_REGISTRY:
        graph.add_conditional_edges(
            name,
            route_after_agent_v2,
            {**handoff_map, "dispatch": "dispatch"},
        )

    # handoff 节点 +1 后，按 handoff_target 路由到真正的目标 agent
    graph.add_conditional_edges(
        "handoff",
        route_to_target,
        {n: n for n in AGENT_REGISTRY},
    )

    # 审批闸：approval（interrupt 等确认）→ execute（批准执行/拒绝取消）
    graph.add_edge("approval", "execute")

    # execute 出口：多意图批量栅栏落地后回 dispatch 评估剩余子意图；单意图 → END。
    def route_after_execute(state) -> str:
        return "dispatch" if state.get("is_multi_intent") else END

    graph.add_conditional_edges(
        "execute",
        route_after_execute,
        {"dispatch": "dispatch", END: END},
    )

    graph.add_edge("human_handoff", END)
    graph.add_edge("synthesize", END)
    return graph


def compile_graph(
    settings: Settings | None = None, memory_manager=None, prompts=None, mcp=None,
    checkpointer=None, error_store=None,
):
    """Compile the graph for execution.

    审批闸依赖 checkpointer 才能 interrupt/resume（暂停后按 thread_id 恢复）。
    未显式传入时，若 approval_enabled 则默认用进程内 MemorySaver（生产使用持久化
    saver，如Postgres，以便跨进程恢复）。

    error_store 透传给 build_graph 启用 L2 自我反思（见 build_graph）。
    """
    if settings is None:
        settings = Settings()
    if checkpointer is None and getattr(settings, "approval_enabled", False):
        from langgraph.checkpoint.memory import MemorySaver

        checkpointer = MemorySaver()
    graph = build_graph(settings, memory_manager, prompts=prompts, mcp=mcp, error_store=error_store)
    return graph.compile(checkpointer=checkpointer)

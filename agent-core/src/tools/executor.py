"""Tool calling loop: executes LLM with tools and handles iterative tool calls."""
import structlog
from langchain_core.messages import AIMessage, ToolMessage, SystemMessage
from langchain_core.language_models import BaseChatModel
from langchain_core.tools import StructuredTool

logger = structlog.get_logger()

MAX_TOOL_ITERATIONS = 3


class ApprovalRequired(Exception):
    """敏感写工具执行前需人工确认（Layer 2 审批闸）。

    在工具真正 ainvoke 之前抛出，携带待办写调用 + 当前已累积的对话上下文
    （含带 tool_calls 的 AIMessage 与本轮已执行的读工具 ToolMessage），交给
    approval/execute 节点接管：approval 节点 interrupt 等人工确认，execute
    节点批准后据此执行写工具并生成回复。写副作用绝不在确认前落地。
    """

    def __init__(self, pending_calls: list[dict], working_messages: list):
        self.pending_calls = pending_calls        # [{"name","args","id"}]
        self.working_messages = working_messages  # 供 execute 节点恢复上下文
        super().__init__(f"approval required: {[c['name'] for c in pending_calls]}")


async def _run_one_tool(tool_map: dict, tool_call: dict) -> ToolMessage:
    """执行单个工具调用，包成 ToolMessage（未知/异常都降级为文本结果，不抛）。"""
    name, args, call_id = tool_call["name"], tool_call["args"], tool_call["id"]
    logger.info("tool_call", tool=name, args=args)
    if name not in tool_map:
        result = f"未知工具: {name}"
    else:
        try:
            result = await tool_map[name].ainvoke(args)
        except Exception as e:
            logger.error("tool_execution_error", tool=name, error=str(e))
            result = f"工具执行失败: {str(e)}"
    logger.info("tool_result", tool=name, result_preview=str(result)[:200])
    return ToolMessage(content=str(result), tool_call_id=call_id)


async def execute_pending_writes(
    llm: BaseChatModel,
    tool_map: dict,
    working_messages: list,
    pending_calls: list[dict],
    stub_calls: list[dict] | None = None,
) -> AIMessage:
    """execute 节点用：批准后执行待办写工具，据结果生成最终回复。

    working_messages 里的 AIMessage 带着**全部**待办写的 tool_calls，但只有 approved
    的写会真正 ainvoke。被拒/去重（未执行）的写必须补一条取消 ToolMessage，否则该
    tool_call 无配对结果，末尾 llm.ainvoke 会因 API 的 tool_call/ToolMessage 配对约束
    报错。stub_calls 即这些「不执行但要占位」的写调用。

    末尾的 llm.ainvoke 不绑定工具（bounded：只让它基于工具结果措辞，不再发起新
    工具调用），因此无 replay、无 divergence、LLM 调用次数与正常 loop 一致。
    """
    for call in pending_calls:
        working_messages.append(await _run_one_tool(tool_map, call))
    for call in stub_calls or []:
        working_messages.append(
            ToolMessage(
                content="[未执行] 该操作未获批准或已去重，已跳过。",
                tool_call_id=call["id"],
            )
        )
    return await llm.ainvoke(working_messages)


async def run_agent_with_tools(
    llm: BaseChatModel,
    tools: list[StructuredTool],
    system_prompt: str,
    messages: list,
    max_iterations: int = MAX_TOOL_ITERATIONS,
    protected_tools: set[str] | None = None,
) -> AIMessage:
    if tools:
        llm_with_tools = llm.bind_tools(tools)
    else:
        llm_with_tools = llm

    protected = protected_tools or set()
    tool_map = {tool.name: tool for tool in tools}
    working_messages = [SystemMessage(content=system_prompt), *messages]

    logger.info("llm_invoke_start", tool_count=len(tools), message_count=len(messages))

    for iteration in range(max_iterations + 1):
        response: AIMessage = await llm_with_tools.ainvoke(working_messages)

        logger.info(
            "llm_response",
            iteration=iteration,
            has_tool_calls=bool(response.tool_calls),
            content_preview=response.content[:200] if response.content else "",
        )

        if not response.tool_calls:
            return response

        if iteration >= max_iterations:
            logger.warning("tool_loop_max_iterations", iterations=iteration)
            working_messages.append(response)
            working_messages.append(
                ToolMessage(
                    content="[系统] 已达到工具调用上限，请直接基于已有信息回复用户。",
                    tool_call_id=response.tool_calls[-1]["id"],
                )
            )
            final = await llm.ainvoke(working_messages)
            return final

        working_messages.append(response)

        # 先跑本轮的安全（读）工具，把待审批的写工具挑出来
        pending_writes = [tc for tc in response.tool_calls if tc["name"] in protected]
        for tool_call in response.tool_calls:
            if tool_call["name"] in protected:
                continue  # 写工具留给审批闸，绝不在确认前执行
            working_messages.append(await _run_one_tool(tool_map, tool_call))

        # 有写工具 → 在其落地前中断，交审批闸。working_messages 此刻已含带全部
        # tool_calls 的 AIMessage + 读工具的 ToolMessage；execute 节点补齐写工具
        # 的 ToolMessage 后，每个 tool_call 都有对应结果，满足 API 的配对约束。
        if pending_writes:
            raise ApprovalRequired(pending_calls=pending_writes, working_messages=working_messages)

    return AIMessage(content="抱歉，处理过程中遇到了问题，请稍后再试。")

"""MCP-to-LangChain Tool adapter: converts MCP tool schemas to StructuredTool objects."""
import json
from typing import Any

import structlog
from pydantic import create_model, Field
from langchain_core.tools import StructuredTool

from src.mcp_client.client import MCPClientManager
from src.tools.registry import TOOL_SERVER_MAP
from src.agents.registry import AGENT_REGISTRY, AgentSpec

logger = structlog.get_logger()

_TYPE_MAP = {
    "string": str,
    "integer": int,
    "number": float,
    "boolean": bool,
    "object": dict,
    "array": list,
}

_agent_tools_cache: dict[str, list[StructuredTool]] = {}


def _build_pydantic_model(tool_name: str, input_schema: dict):
    properties = input_schema.get("properties", {})
    required = set(input_schema.get("required", []))
    fields: dict[str, Any] = {}
    for name, prop in properties.items():
        py_type = _TYPE_MAP.get(prop.get("type", "string"), str)
        description = prop.get("description", "")
        if name in required:
            fields[name] = (py_type, Field(description=description))
        else:
            fields[name] = (py_type | None, Field(default=None, description=description))
    if not fields:
        # 无参工具：pydantic v2 禁止下划线开头的字段名，故用 "placeholder"
        fields["placeholder"] = (str | None, Field(default=None, description="unused"))
    return create_model(f"{tool_name}_Input", **fields)


def _create_mcp_tool(
    tool_name: str,
    description: str,
    input_schema: dict,
    server: str,
    mcp: MCPClientManager,
) -> StructuredTool:
    args_model = _build_pydantic_model(tool_name, input_schema)

    async def _invoke(**kwargs: Any) -> str:
        kwargs.pop("placeholder", None)
        result = await mcp.call_tool(server, tool_name, kwargs)
        if isinstance(result, dict) and "error" in result:
            return f"工具调用失败: {result['error']}"
        return json.dumps(result, ensure_ascii=False, default=str)

    return StructuredTool.from_function(
        func=None,
        coroutine=_invoke,
        name=tool_name,
        description=description,
        args_schema=args_model,
    )


async def get_agent_tools(agent_name: str, mcp: MCPClientManager) -> list[StructuredTool]:
    if agent_name in _agent_tools_cache:
        return _agent_tools_cache[agent_name]

    spec: AgentSpec | None = AGENT_REGISTRY.get(agent_name)
    if spec is None:
        return []

    # 需要查询的 server：server_grants 直接全量；tool_grants 经 TOOL_SERVER_MAP 反查归属
    servers_needed = set(spec.server_grants)
    for t in spec.tool_grants:
        if t in TOOL_SERVER_MAP:
            servers_needed.add(TOOL_SERVER_MAP[t])

    if not servers_needed:
        return []

    # 动态发现各 server 的工具清单（schema/描述全部来自运行时，不写死）。
    # discovery_failed：任一授权 server 发现失败即置位。只有全部成功才写缓存，
    # 否则本次返回已发现的部分工具但不缓存，下次请求重试 —— MCP 恢复后自动自愈，
    # 不再依赖重启 agent-core（见 memory: restart-agent-core-after-mcp-restart）。
    all_tool_schemas: dict[str, dict] = {}
    server_to_tools: dict[str, list[str]] = {}
    discovery_failed = False
    for server in servers_needed:
        try:
            tools_list = await mcp.list_tools(server)
        except Exception as e:
            logger.warning("mcp_list_tools_failed", server=server, error=str(e))
            discovery_failed = True
            continue
        for tool_def in tools_list:
            name = tool_def["name"]
            all_tool_schemas[name] = {
                "description": tool_def.get("description", ""),
                "inputSchema": tool_def.get("inputSchema", {}),
                "server": server,
            }
            server_to_tools.setdefault(server, []).append(name)

    # 按授权解析为最终允许的工具名（最小权限）：
    #   server_grants → 该 server 发现到的全部工具（含未来新增，零改动）
    #   tool_grants   → 仅这些具体工具
    allowed: list[str] = []
    for server in spec.server_grants:
        allowed.extend(server_to_tools.get(server, []))
    allowed.extend(spec.tool_grants)

    # 去重并保持稳定顺序；只保留确实在 MCP 上发现到的工具
    seen = set()
    langchain_tools = []
    for tool_name in allowed:
        if tool_name in seen or tool_name not in all_tool_schemas:
            continue
        seen.add(tool_name)
        schema = all_tool_schemas[tool_name]
        langchain_tools.append(_create_mcp_tool(
            tool_name=tool_name,
            description=schema["description"],
            input_schema=schema["inputSchema"],
            server=schema["server"],
            mcp=mcp,
        ))

    # 仅当所有授权 server 都发现成功才缓存，避免把"MCP 不可达导致的空/残缺工具"
    # 永久钉死。发现失败则不缓存，下次请求重试 → MCP 恢复后自动自愈。
    if discovery_failed:
        logger.warning(
            "agent_tools_partial_not_cached",
            agent=agent_name,
            discovered=len(langchain_tools),
        )
        return langchain_tools

    _agent_tools_cache[agent_name] = langchain_tools
    return langchain_tools


async def prewarm_agent_tools(mcp: MCPClientManager) -> dict[str, int]:
    """启动预热：并发把每个 agent 的工具清单拉满缓存，将冷启动延迟从用户请求
    路径挪到启动阶段。单个 agent 失败只记 warning、绝不 raise —— 保留 lazy 的
    启动韧性（agent-tools 未就绪时 agent-core 仍能启动，下次请求再自愈）。

    返回 {"ok": 命中数, "failed": 失败数} 供启动日志观测。
    """
    import asyncio

    async def _one(name: str) -> bool:
        try:
            await get_agent_tools(name, mcp)
        except Exception as e:
            logger.warning("prewarm_agent_failed", agent=name, error=str(e))
            return False
        # 命中缓存即视为预热成功：get_agent_tools 只在全部授权 server 发现成功时
        # 写缓存；无 tool_grants 的 agent 走早返回（本就无 MCP 依赖），不计失败。
        spec = AGENT_REGISTRY.get(name)
        no_grants = spec is None or (not spec.server_grants and not spec.tool_grants)
        return no_grants or name in _agent_tools_cache

    names = list(AGENT_REGISTRY.keys())
    results = await asyncio.gather(*(_one(n) for n in names))
    ok = sum(1 for r in results if r is True)
    return {"ok": ok, "failed": len(names) - ok}

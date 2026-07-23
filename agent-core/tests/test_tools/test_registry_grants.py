"""registry.py (YAML 加载 + grant 解析) 与 mcp_adapter 授权解析的单元测试。

覆盖 B 方案核心诉求：
- YAML 加载为 AgentSpec
- "server:" / "tool:" / 裸名 三种授权写法解析
- server 授权自动包含未来新增工具（零改动）
- tool 授权严格最小权限，不越权
"""
import textwrap

import pytest

from src.agents.registry import load_registry, _parse_grants, AgentSpec
from src.tools import mcp_adapter
from src.tools.mcp_adapter import get_agent_tools, prewarm_agent_tools


class TestParseGrants:
    def test_server_prefix(self):
        servers, tools = _parse_grants(["server:order"])
        assert servers == frozenset({"order"}) and tools == frozenset()

    def test_tool_prefix(self):
        servers, tools = _parse_grants(["tool:apply_refund"])
        assert servers == frozenset() and tools == frozenset({"apply_refund"})

    def test_bare_name_treated_as_tool(self):
        # 向后兼容：裸工具名当作细粒度工具授权
        servers, tools = _parse_grants(["create_ticket"])
        assert tools == frozenset({"create_ticket"})

    def test_mixed(self):
        servers, tools = _parse_grants(["server:ticket", "tool:get_customer_info"])
        assert servers == frozenset({"ticket"})
        assert tools == frozenset({"get_customer_info"})


class TestLoadRegistry:
    def test_loads_yaml(self, tmp_path):
        p = tmp_path / "agents.yaml"
        p.write_text(textwrap.dedent("""
            agents:
              demo:
                prompt_id: agents/demo
                model_key: model_fast
                history_window: 7
                tools:
                  - server:order
                  - tool:create_ticket
                can_handoff_to: [order]
                reflection: judge
        """), encoding="utf-8")
        reg = load_registry(p)
        assert set(reg) == {"demo"}
        spec = reg["demo"]
        assert isinstance(spec, AgentSpec)
        assert spec.prompt_id == "agents/demo"
        assert spec.model_key == "model_fast"
        assert spec.history_window == 7
        assert spec.server_grants == frozenset({"order"})
        assert spec.tool_grants == frozenset({"create_ticket"})
        assert spec.can_handoff_to == ["order"]
        assert spec.reflection == "judge"

    def test_missing_file_returns_empty(self, tmp_path):
        assert load_registry(tmp_path / "nope.yaml") == {}

    def test_defaults_applied(self, tmp_path):
        p = tmp_path / "a.yaml"
        p.write_text("agents:\n  x:\n    prompt_id: agents/x\n", encoding="utf-8")
        spec = load_registry(p)["x"]
        assert spec.model_key == "model_main"
        assert spec.history_window == 10
        assert spec.reflection == "off"
        assert spec.server_grants == frozenset()


_SCHEMA = {"type": "object", "properties": {"q": {"type": "string"}}}


class _FakeMCP:
    """假 MCP：按 server 返回工具清单。catalog 可注入"未来新增工具"。"""
    def __init__(self, catalog):
        self.catalog = catalog

    async def list_tools(self, server):
        return self.catalog.get(server, [])


def _tool(name):
    return {"name": name, "description": "d", "inputSchema": _SCHEMA}


@pytest.fixture
def patch_registry(monkeypatch):
    """用受控的 AgentSpec 替换全局 AGENT_REGISTRY，并清工具缓存。"""
    def _apply(specs: dict):
        monkeypatch.setattr(mcp_adapter, "AGENT_REGISTRY", specs)
        mcp_adapter._agent_tools_cache.clear()
    yield _apply
    mcp_adapter._agent_tools_cache.clear()


class TestGetAgentToolsGrants:
    @pytest.mark.asyncio
    async def test_server_grant_includes_future_tool(self, patch_registry):
        # server 授权 → 自动拿到该 server 未来新增的工具，零改动
        patch_registry({"a": AgentSpec(name="a", prompt_id="p",
                                       server_grants=frozenset({"knowledge"}))})
        mcp = _FakeMCP({"knowledge": [_tool("search_faq"), _tool("NEW_tool")]})
        tools = await get_agent_tools("a", mcp)
        assert sorted(t.name for t in tools) == ["NEW_tool", "search_faq"]

    @pytest.mark.asyncio
    async def test_tool_grant_is_least_privilege(self, patch_registry):
        # tool 细粒度授权：同 server 其它工具不会被拿到
        patch_registry({"a": AgentSpec(name="a", prompt_id="p",
                                       tool_grants=frozenset({"create_ticket"}))})
        mcp = _FakeMCP({"ticket": [_tool("create_ticket"), _tool("apply_compensation")]})
        tools = await get_agent_tools("a", mcp)
        assert [t.name for t in tools] == ["create_ticket"]

    @pytest.mark.asyncio
    async def test_mixed_grants(self, patch_registry):
        patch_registry({"a": AgentSpec(name="a", prompt_id="p",
                                       server_grants=frozenset({"knowledge"}),
                                       tool_grants=frozenset({"create_ticket"}))})
        mcp = _FakeMCP({
            "knowledge": [_tool("search_faq")],
            "ticket": [_tool("create_ticket"), _tool("apply_compensation")],
        })
        tools = await get_agent_tools("a", mcp)
        assert sorted(t.name for t in tools) == ["create_ticket", "search_faq"]

    @pytest.mark.asyncio
    async def test_unknown_agent_returns_empty(self, patch_registry):
        patch_registry({})
        assert await get_agent_tools("ghost", _FakeMCP({})) == []

    @pytest.mark.asyncio
    async def test_no_grants_returns_empty(self, patch_registry):
        patch_registry({"a": AgentSpec(name="a", prompt_id="p")})
        assert await get_agent_tools("a", _FakeMCP({"x": [_tool("t")]})) == []


class _FlakyMCP:
    """先失败后成功的 MCP：模拟 agent-tools 启动晚于 agent-core，之后恢复。"""
    def __init__(self, catalog):
        self.catalog = catalog
        self.fail = True

    async def list_tools(self, server):
        if self.fail:
            raise ConnectionError(f"MCP down: {server}")
        return self.catalog.get(server, [])


class TestDiscoveryFailureSelfHeal:
    @pytest.mark.asyncio
    async def test_failure_not_cached_and_self_heals(self, patch_registry):
        # 发现失败不写缓存；MCP 恢复后下次请求自动拿到工具，无需重启。
        patch_registry({"a": AgentSpec(name="a", prompt_id="p",
                                       server_grants=frozenset({"knowledge"}))})
        mcp = _FlakyMCP({"knowledge": [_tool("search_faq")]})

        first = await get_agent_tools("a", mcp)
        assert first == []                                  # 失败 → 空
        assert "a" not in mcp_adapter._agent_tools_cache    # 关键：未钉死缓存

        mcp.fail = False                                    # agent-tools 恢复
        second = await get_agent_tools("a", mcp)
        assert [t.name for t in second] == ["search_faq"]   # 自愈
        assert "a" in mcp_adapter._agent_tools_cache        # 成功后才缓存

    @pytest.mark.asyncio
    async def test_partial_failure_not_cached(self, patch_registry):
        # 多 server 中任一失败即不缓存（避免残缺工具集被钉死）。
        patch_registry({"a": AgentSpec(name="a", prompt_id="p",
                                       server_grants=frozenset({"knowledge", "order"}))})

        class _HalfMCP:
            async def list_tools(self, server):
                if server == "order":
                    raise ConnectionError("order down")
                return [_tool("search_faq")]

        tools = await get_agent_tools("a", _HalfMCP())
        assert [t.name for t in tools] == ["search_faq"]    # 返回已发现的部分
        assert "a" not in mcp_adapter._agent_tools_cache    # 但不缓存


class TestPrewarm:
    @pytest.mark.asyncio
    async def test_prewarm_fills_cache(self, patch_registry):
        patch_registry({
            "a": AgentSpec(name="a", prompt_id="p", server_grants=frozenset({"knowledge"})),
            "b": AgentSpec(name="b", prompt_id="p"),  # 无授权，走早返回、不计失败
        })
        mcp = _FakeMCP({"knowledge": [_tool("search_faq")]})
        stats = await prewarm_agent_tools(mcp)
        assert stats == {"ok": 2, "failed": 0}
        assert "a" in mcp_adapter._agent_tools_cache

    @pytest.mark.asyncio
    async def test_prewarm_counts_failure_but_does_not_raise(self, patch_registry):
        patch_registry({"a": AgentSpec(name="a", prompt_id="p",
                                       server_grants=frozenset({"knowledge"}))})
        mcp = _FlakyMCP({"knowledge": [_tool("search_faq")]})  # 一直失败
        stats = await prewarm_agent_tools(mcp)                 # 不抛
        assert stats == {"ok": 0, "failed": 1}
        assert "a" not in mcp_adapter._agent_tools_cache


class TestNoArgTool:
    """无参工具（空 inputSchema）的回归测试。

    pydantic v2 禁止下划线开头字段名，曾导致无参工具建模型时抛 NameError。
    """
    def test_empty_schema_builds_model(self):
        from src.tools.mcp_adapter import _build_pydantic_model
        model = _build_pydantic_model("ping", {})  # 不应抛 NameError
        assert "placeholder" in model.model_fields

    @pytest.mark.asyncio
    async def test_noarg_tool_resolves_and_strips_placeholder(self, patch_registry):
        captured = {}

        class _CapturingMCP:
            async def list_tools(self, server):
                return [{"name": "ping", "description": "no args", "inputSchema": {}}]

            async def call_tool(self, server, tool_name, arguments):
                captured["args"] = arguments
                return {"ok": True}

        patch_registry({"a": AgentSpec(name="a", prompt_id="p",
                                       server_grants=frozenset({"misc"}))})
        tools = await get_agent_tools("a", _CapturingMCP())
        assert [t.name for t in tools] == ["ping"]

        # 调用无参工具：placeholder 不应泄漏给 MCP server
        await tools[0].ainvoke({})
        assert captured["args"] == {}

"""MCPClientManager 的 404 失效 session 自愈重连测试。

场景：MCP server 重启后，agent-core 仍持旧 session ID，POST /mcp → 404。
期望：client 自动丢弃旧 session、强制重握手拿新 session、用新 session 重试一次。
覆盖 memory: restart-agent-core-after-mcp-restart 里 (2) session 缓存这条路径。
"""
import httpx
import pytest

from src.config import Settings
from src.mcp_client.client import MCPClientManager


def _resp(status: int, *, text: str = "", session_id: str | None = None) -> httpx.Response:
    headers = {}
    if session_id is not None:
        headers["mcp-session-id"] = session_id
    req = httpx.Request("POST", "http://order-mcp:8002/mcp")
    return httpx.Response(status, headers=headers, text=text, request=req)


class _ScriptedClient:
    """按预设脚本依次返回响应，记录每次请求携带的 Mcp-Session-Id。"""
    def __init__(self, script):
        self.script = list(script)
        self.calls = []      # 每次 post 携带的 session header
        self.is_closed = False

    async def post(self, url, json=None, headers=None):
        self.calls.append((json.get("method"), (headers or {}).get("Mcp-Session-Id")))
        return self.script.pop(0)


@pytest.fixture
def mcp():
    m = MCPClientManager(Settings())
    return m


@pytest.mark.asyncio
async def test_call_tool_reconnects_on_404(mcp, monkeypatch):
    # 脚本：initialize→session s1 / tools/call→404 / re-initialize→session s2 / tools/call→200
    ok_body = 'data: {"jsonrpc":"2.0","id":2,"result":{"content":[{"type":"text","text":"{\\"ok\\":true}"}]}}'
    client = _ScriptedClient([
        _resp(200, session_id="s1"),          # 首次 initialize
        _resp(404, text="Not Found"),          # 旧 session 失效
        _resp(200, session_id="s2"),          # 强制重握手
        _resp(200, text=ok_body),              # 用新 session 重试成功
    ])
    monkeypatch.setattr(mcp, "_get_client", lambda: _await(client))

    result = await mcp.call_tool("order", "get_order", {"id": "1"})

    assert result == {"ok": True}
    assert mcp._sessions["order"] == "s2"                 # 新 session 已落地
    # 重试请求应带新 session s2，而非失效的 s1
    methods = [m for m, _ in client.calls]
    assert methods == ["initialize", "tools/call", "initialize", "tools/call"]
    assert client.calls[-1] == ("tools/call", "s2")


@pytest.mark.asyncio
async def test_list_tools_reconnects_on_404(mcp, monkeypatch):
    tools_body = 'data: {"jsonrpc":"2.0","id":3,"result":{"tools":[{"name":"get_order"}]}}'
    client = _ScriptedClient([
        _resp(200, session_id="s1"),          # initialize
        _resp(404, text="Not Found"),          # 失效
        _resp(200, session_id="s2"),          # 重握手
        _resp(200, text=tools_body),           # 重试成功
    ])
    monkeypatch.setattr(mcp, "_get_client", lambda: _await(client))

    tools = await mcp.list_tools("order")

    assert [t["name"] for t in tools] == ["get_order"]
    assert mcp._sessions["order"] == "s2"


@pytest.mark.asyncio
async def test_list_tools_raises_when_reconnect_still_404(mcp, monkeypatch):
    # 重握手后仍 404（server 真挂）→ 抛异常，供 get_agent_tools 判定不缓存
    client = _ScriptedClient([
        _resp(200, session_id="s1"),
        _resp(404, text="Not Found"),
        _resp(200, session_id="s2"),
        _resp(404, text="Not Found"),
    ])
    monkeypatch.setattr(mcp, "_get_client", lambda: _await(client))

    with pytest.raises(httpx.HTTPStatusError):
        await mcp.list_tools("order")


def _await(value):
    """把同步值包装成可 await 的协程（_get_client 是 async）。"""
    async def _coro():
        return value
    return _coro()

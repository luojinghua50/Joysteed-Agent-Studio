"""Skill server 专用的 MCP client。

复制自 agent-core/src/mcp_client/client.py 并精简：skill_server 是编排型 server，
自己作为 MCP 客户端去调 leaf server（knowledge/order）。保留原样的 SSE 解析 +
session 失效(404)重握手自愈逻辑；仅裁掉 skill 用不到的 server（ticket/crm）。

MVP 阶段选择"复制一份"而非抽共享包（见设计决策 1）；跑通后再看是否抽包。
"""
import os

import httpx
import json
import structlog

logger = structlog.get_logger()


class SkillMCPClient:
    """Minimal MCP client for skill orchestration — talks to knowledge/order leaf servers."""

    def __init__(self) -> None:
        self.servers = {
            "knowledge": os.environ.get(
                "KNOWLEDGE_MCP_URL", "http://localhost:8001/mcp"
            ),
            "order": os.environ.get("ORDER_MCP_URL", "http://localhost:8002/mcp"),
        }
        self._http_client: httpx.AsyncClient | None = None
        self._sessions: dict[str, str] = {}

    async def _get_client(self) -> httpx.AsyncClient:
        if self._http_client is None or self._http_client.is_closed:
            self._http_client = httpx.AsyncClient(timeout=15.0)
        return self._http_client

    def _parse_sse_response(self, text: str) -> dict:
        """Parse SSE response format from MCP server."""
        for line in text.strip().split("\n"):
            if line.startswith("data: "):
                return json.loads(line[6:])
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            return {"error": f"Cannot parse response: {text[:200]}"}

    async def _ensure_initialized(self, server: str, force: bool = False) -> bool:
        """Initialize MCP session if not already done. force=True 强制重握手（404 自愈）。"""
        if server in self._sessions and not force:
            return True
        if force:
            self._sessions.pop(server, None)

        url = self.servers.get(server)
        if not url:
            return False

        try:
            client = await self._get_client()
            host = url.split("://")[-1].split("/")[0]
            headers = {
                "Accept": "application/json, text/event-stream",
                "Content-Type": "application/json",
                "Host": host.replace(host.split(":")[0], "localhost"),
            }
            response = await client.post(url, json={
                "jsonrpc": "2.0",
                "method": "initialize",
                "params": {
                    "protocolVersion": "2024-11-05",
                    "capabilities": {},
                    "clientInfo": {"name": "skill-server", "version": "1.0"},
                },
                "id": 1,
            }, headers=headers)
            response.raise_for_status()

            session_id = response.headers.get("mcp-session-id", "")
            if session_id:
                self._sessions[server] = session_id
                logger.info("mcp_session_initialized", server=server)
                return True

            result = self._parse_sse_response(response.text)
            if "result" in result:
                session_id = response.headers.get("mcp-session-id", "")
                if session_id:
                    self._sessions[server] = session_id
                return True
            return False
        except Exception as e:
            logger.error("mcp_initialize_error", server=server, error=str(e))
            return False

    async def _rpc(self, server: str, payload: dict) -> dict:
        """发送单个 JSON-RPC 请求，遇 404(session 失效)自动重握手并重试一次。"""
        url = self.servers[server]
        if not await self._ensure_initialized(server):
            raise ConnectionError(f"MCP initialize failed: {server}")

        client = await self._get_client()
        host = url.split("://")[-1].split("/")[0]

        def _headers() -> dict:
            h = {
                "Accept": "application/json, text/event-stream",
                "Content-Type": "application/json",
                "Host": host.replace(host.split(":")[0], "localhost"),
            }
            if server in self._sessions:
                h["Mcp-Session-Id"] = self._sessions[server]
            return h

        response = await client.post(url, json=payload, headers=_headers())
        if response.status_code == 404:
            logger.warning("mcp_session_stale_reconnect", server=server)
            if await self._ensure_initialized(server, force=True):
                response = await client.post(url, json=payload, headers=_headers())
        response.raise_for_status()
        return self._parse_sse_response(response.text)

    async def call_tool(self, server: str, tool_name: str, arguments: dict) -> dict:
        """Call a tool on a leaf MCP server. 失败返回 {"error": ...}，不抛。"""
        url = self.servers.get(server)
        if not url:
            return {"error": f"Unknown MCP server: {server}"}

        try:
            result = await self._rpc(server, {
                "jsonrpc": "2.0",
                "method": "tools/call",
                "params": {"name": tool_name, "arguments": arguments},
                "id": 2,
            })

            if "error" in result:
                return {"error": result["error"].get("message", str(result["error"]))}

            mcp_result = result.get("result", {})
            if isinstance(mcp_result, dict) and "content" in mcp_result:
                contents = mcp_result["content"]
                if isinstance(contents, list) and contents:
                    text_parts = [c.get("text", "") for c in contents if c.get("type") == "text"]
                    if text_parts:
                        combined = "\n".join(text_parts)
                        try:
                            return json.loads(combined)
                        except json.JSONDecodeError:
                            return {"text": combined}
            return mcp_result

        except httpx.TimeoutException:
            logger.warning("mcp_tool_timeout", server=server, tool=tool_name)
            return {"error": f"Tool {tool_name} timed out"}
        except httpx.HTTPError as e:
            logger.error("mcp_tool_error", server=server, tool=tool_name, error=str(e))
            return {"error": f"Tool {tool_name} call failed: {str(e)}"}

    async def close(self) -> None:
        if self._http_client and not self._http_client.is_closed:
            await self._http_client.aclose()

import httpx
import json
import structlog
from src.config import Settings

logger = structlog.get_logger()


class MCPClientManager:
    """Manages connections to multiple MCP servers using Streamable HTTP transport."""

    def __init__(self, settings: Settings | None = None):
        if settings is None:
            settings = Settings()

        self.servers = {
            "knowledge": settings.knowledge_mcp_url,
            "order": settings.order_mcp_url,
            "ticket": settings.ticket_mcp_url,
            "crm": settings.crm_mcp_url,
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

    def _get_headers(self, server: str) -> dict:
        """Get required headers for MCP Streamable HTTP."""
        url = self.servers.get(server, "")
        host = url.split("://")[-1].split("/")[0] if url else "localhost"
        headers = {
            "Accept": "application/json, text/event-stream",
            "Content-Type": "application/json",
            "Host": host.replace(server.replace("_", "-") + "-mcp", "localhost"),
        }
        if server in self._sessions:
            headers["Mcp-Session-Id"] = self._sessions[server]
        return headers

    async def _ensure_initialized(self, server: str, force: bool = False) -> bool:
        """Initialize MCP session if not already done.

        force=True 强制重新握手（忽略已有 session）：用于 MCP server 重启后旧
        session 失效（POST /mcp → 404）的重连自愈，见 call_tool/list_tools 的重试。
        """
        if server in self._sessions and not force:
            return True
        if force:
            self._sessions.pop(server, None)  # 丢弃失效 session，下面重新握手

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
                    "clientInfo": {"name": "agent-core", "version": "1.0"},
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
        """发送单个 JSON-RPC 请求，遇 404 自动重握手并重试一次。

        404 是 MCP server 重启后旧 session 失效的特征信号（≠406：406 是端点活着
        但缺 SSE 头）。命中 404 时强制重握手拿新 session 再重试一次；仍失败则抛
        HTTPStatusError 交由调用方处理。初始化失败（拿不到 session）直接抛
        ConnectionError。调用方须自行确认 server 已在 self.servers 中。
        """
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
            # session 失效：强制重握手，用新 session 重试一次
            logger.warning("mcp_session_stale_reconnect", server=server)
            if await self._ensure_initialized(server, force=True):
                response = await client.post(url, json=payload, headers=_headers())
        response.raise_for_status()
        return self._parse_sse_response(response.text)

    async def call_tool(self, server: str, tool_name: str, arguments: dict) -> dict:
        """Call a tool on a specific MCP server."""
        url = self.servers.get(server)
        if not url:
            return {"error": f"Unknown MCP server: {server}"}

        try:
            result = await self._rpc(server, {
                "jsonrpc": "2.0",
                "method": "tools/call",
                "params": {
                    "name": tool_name,
                    "arguments": arguments,
                },
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

    async def list_tools(self, server: str) -> list[dict]:
        """List available tools on a specific MCP server.

        失败即抛：网络错误 / 初始化失败 / 未知 server 都会 raise，由调用方
        （get_agent_tools）据此区分"MCP 不可达"与"该 server 无工具"，从而决定
        是否写缓存。切勿把失败吞成 []，否则空结果会被永久缓存、MCP 恢复也不自愈。
        """
        if server not in self.servers:
            raise ValueError(f"Unknown MCP server: {server}")

        # 经 _rpc：含 404 失效 session 的重握手重试；网络/HTTP 错误在此上抛，
        # 由 get_agent_tools 据此判定"MCP 不可达"从而不缓存空结果。
        result = await self._rpc(server, {
            "jsonrpc": "2.0",
            "method": "tools/list",
            "params": {},
            "id": 3,
        })
        return result.get("result", {}).get("tools", [])

    async def close(self):
        """Close the HTTP client."""
        if self._http_client and not self._http_client.is_closed:
            await self._http_client.aclose()

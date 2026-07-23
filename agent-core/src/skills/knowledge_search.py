from src.mcp_client.client import MCPClientManager


async def search_knowledge(query: str, top_k: int = 3, mcp: MCPClientManager | None = None) -> str:
    """搜索知识库获取答案：走统一聚合检索入口（跨库路由 + faq 短路 + 加权融合）。"""
    if mcp is None:
        mcp = MCPClientManager()

    results = await mcp.call_tool("knowledge", "search_knowledge", {
        "query": query,
        "top_k": top_k,
    })

    if results.get("error"):
        return f"知识库检索失败: {results['error']}"

    items = results.get("results", [])
    if not items:
        return "未找到相关知识内容，建议联系人工客服获取帮助。"

    output = "相关知识库内容：\n"
    for i, item in enumerate(items, 1):
        output += f"\n{i}. {item.get('title', '无标题')}\n"
        output += f"   {item.get('content', '')}\n"
        score = item.get("score", 0)
        if score:
            output += f"   (相关度: {score:.2f})\n"

    return output

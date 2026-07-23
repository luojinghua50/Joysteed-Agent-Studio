"""Knowledge MCP Server: bridges agent-core to agent-rag service with mock fallback.

统一入口 search_knowledge 走 agent-rag 的 /api/route-search（跨库路由 + faq 短路 +
加权 RRF 融合），Agent 无需关心库的物理 kb_id —— 路由层按 scope/kb_form 自动选库。
get_knowledge_filters 暴露各库可过滤的元数据字段，供 Agent 的 LLM 推断 filters 后回传。
search_faq / search_docs 保留向后兼容，动态解析真实 kb_id 后查单库。
"""
import os

import httpx
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("knowledge-service")

RAG_SERVICE_URL = os.environ.get("RAG_SERVICE_URL", "http://localhost:8010")
# agent-rag 的 kb_id 是动态生成的（如 fca1957d），不是固定的 "faq"/"docs"。
# 按知识库名称解析出真实 id；名称可配，默认指向客服知识库。search_faq 与
# search_docs 当前都查这同一个 KB（现状只有一个客服库）。
KNOWLEDGE_KB_NAME = os.environ.get("KNOWLEDGE_KB_NAME", "客服知识库")

# 解析结果缓存：避免每次检索都打一次 /api/knowledge-bases。
_kb_id_cache: str | None = None


async def _resolve_kb_id(client: httpx.AsyncClient) -> str | None:
    """从 agent-rag 列出知识库，按名称解析真实 kb_id；解析不到返回 None。

    优先匹配 KNOWLEDGE_KB_NAME；匹配不到时退而用列表中的第一个库（单库部署
    下即客服库），保证只要 rag 里有数据就能检索到。结果缓存。
    """
    global _kb_id_cache
    if _kb_id_cache is not None:
        return _kb_id_cache
    try:
        resp = await client.get(f"{RAG_SERVICE_URL}/api/knowledge-bases")
        if resp.status_code != 200:
            return None
        kbs = resp.json() or []
        if not kbs:
            return None
        match = next((kb for kb in kbs if kb.get("name") == KNOWLEDGE_KB_NAME), kbs[0])
        _kb_id_cache = match.get("id")
        return _kb_id_cache
    except Exception:
        return None

MOCK_FAQ = [
    {
        "id": "faq-001",
        "title": "退换货政策",
        "content": "我们提供7天无理由退换货服务。商品签收后7天内，保持原包装完好即可申请退换。退款将在审核通过后3个工作日内原路退回。",
        "category": "售后",
    },
    {
        "id": "faq-002",
        "title": "配送时效",
        "content": "标准配送：3-5个工作日送达。加急配送：1-2个工作日送达（需额外付费）。偏远地区可能延迟1-2天。",
        "category": "物流",
    },
    {
        "id": "faq-003",
        "title": "会员权益",
        "content": "VIP会员享受：1. 全场95折 2. 优先客服通道 3. 专属优惠券 4. 生日礼品 5. 免费加急配送。年度消费满5000元自动升级。",
        "category": "会员",
    },
    {
        "id": "faq-004",
        "title": "支付方式",
        "content": "支持支付宝、微信支付、银行卡（借记卡/信用卡）、花呗分期。分期免息活动请关注首页活动页。",
        "category": "支付",
    },
    {
        "id": "faq-005",
        "title": "发票开具",
        "content": "订单完成后可在\"我的订单-申请发票\"中开具电子发票。支持个人和企业发票，电子发票将在申请后24小时内发送到您的邮箱。",
        "category": "发票",
    },
    {
        "id": "faq-006",
        "title": "账号安全",
        "content": "建议定期修改密码，开启手机验证码登录。如发现账号异常，请立即联系客服冻结账号。不要向他人透露验证码。",
        "category": "安全",
    },
]

MOCK_DOCS = [
    {
        "id": "doc-001",
        "title": "产品使用指南 - 无线耳机",
        "content": "配对方法：1. 打开耳机盒 2. 长按配对键3秒 3. 在手机蓝牙中搜索设备 4. 点击连接。重置方法：同时按住两只耳机10秒，指示灯闪红后松开。",
        "category": "产品指南",
    },
    {
        "id": "doc-002",
        "title": "产品使用指南 - 智能手表",
        "content": "首次使用：1. 充电至少30分钟 2. 长按右侧按钮开机 3. 下载APP扫码绑定 4. 允许权限后同步数据。常见问题：心率不准请佩戴紧贴手腕。",
        "category": "产品指南",
    },
]


def _simple_search(query: str, items: list[dict], top_k: int = 3) -> list[dict]:
    """Simple keyword-based search (fallback when RAG service unavailable)."""
    results = []
    keywords = [k for k in query.lower().split() if len(k) >= 2]
    if not keywords:
        keywords = [query.lower()] if len(query) >= 2 else []

    for item in items:
        score = 0.0
        text = f"{item.get('title', '')} {item.get('content', '')}".lower()
        for keyword in keywords:
            if keyword in text:
                score += 0.3

        if score > 0:
            results.append({**item, "score": min(score, 1.0)})

    results.sort(key=lambda x: x["score"], reverse=True)
    return results[:top_k]


def _format_results(results: list[dict]) -> list[dict]:
    """把 agent-rag 的 SearchResult 规整成工具返回形态。"""
    return [
        {
            "id": r.get("chunk_id", ""),
            "title": r.get("source", ""),
            "content": r.get("text", ""),
            "score": r.get("score", 0.0),
            "kb_id": r.get("kb_id", ""),
        }
        for r in results
    ]


async def _rag_search(query: str, kb_id: str, top_k: int = 3) -> list[dict] | None:
    """Call agent-rag 单库检索 /api/search. Returns None on failure."""
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.post(
                f"{RAG_SERVICE_URL}/api/search",
                json={"query": query, "kb_id": kb_id, "top_k": top_k},
            )
            if resp.status_code == 200:
                results = resp.json().get("results", [])
                if results:
                    return _format_results(results)
    except Exception:
        pass
    return None


async def _rag_route_search(query: str, top_k: int, scope: list[str] | None,
                            filters: list[dict] | None) -> dict | None:
    """Call agent-rag 聚合检索 /api/route-search（跨库路由 + 短路 + 融合）。

    返回 {results, total, shortcut, routed_kbs} 或失败时 None。
    """
    payload: dict = {"query": query, "top_k": top_k}
    if scope:
        payload["scope"] = scope
    if filters:
        payload["filters"] = filters
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.post(f"{RAG_SERVICE_URL}/api/route-search", json=payload)
            if resp.status_code == 200:
                data = resp.json()
                return {
                    "results": _format_results(data.get("results", [])),
                    "total": data.get("total", 0),
                    "shortcut": data.get("shortcut", False),
                    "routed_kbs": data.get("routed_kbs", []),
                }
    except Exception:
        pass
    return None


@mcp.tool()
async def search_knowledge(query: str, top_k: int = 3,
                           scope: list[str] | None = None,
                           filters: list[dict] | None = None) -> dict:
    """统一知识检索入口：跨知识库路由检索，自动选库、faq 高置信短路、加权融合。

    Args:
        query: 用户问题原文。
        top_k: 返回条数。
        scope: 可选，限定参与的知识库（kb_id 或 kb_form 如 "faq"/"standard"/"temporal"）；
               不传则检索租户下全部库。
        filters: 可选，库内元数据过滤条件列表，每项形如
                 {"field": "category", "op": "eq", "value": "耳机"}。
                 可先调 get_knowledge_filters 查看各库可用字段后再推断填入。

    Returns:
        {results, total, shortcut, routed_kbs, source}；shortcut=True 表示命中 FAQ 直答。
    """
    routed = await _rag_route_search(query, top_k, scope, filters)
    if routed is not None:
        return {**routed, "source": "rag"}

    # 降级：rag 不可用时用内置 mock（合并 FAQ+文档库做关键词检索）
    results = _simple_search(query, MOCK_FAQ + MOCK_DOCS, top_k)
    return {"results": results, "total": len(results),
            "shortcut": False, "routed_kbs": [], "source": "mock"}


@mcp.tool()
async def get_knowledge_filters() -> dict:
    """列出各知识库可用于过滤的元数据字段，供推断 search_knowledge 的 filters 参数。

    Returns:
        {kbs: [{kb_id, name, kb_form, fields: [{name, field_type}]}], source}。
        field_type 为 string|number|time。检索时把要过滤的字段编成 filters 回传。
    """
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.get(f"{RAG_SERVICE_URL}/api/knowledge-bases")
            if resp.status_code != 200:
                return {"kbs": [], "source": "unavailable"}
            kbs = resp.json() or []
            out = []
            for kb in kbs:
                fresp = await client.get(
                    f"{RAG_SERVICE_URL}/api/knowledge-bases/{kb['id']}/metadata-fields"
                )
                fields = fresp.json() if fresp.status_code == 200 else []
                out.append({
                    "kb_id": kb.get("id", ""), "name": kb.get("name", ""),
                    "kb_form": kb.get("kb_form", "standard"),
                    "fields": [{"name": f.get("name"), "field_type": f.get("field_type")}
                               for f in fields],
                })
            return {"kbs": out, "source": "rag"}
    except Exception:
        return {"kbs": [], "source": "unavailable"}


@mcp.tool()
async def search_faq(query: str, top_k: int = 3) -> dict:
    """搜索FAQ知识库，返回最相关的常见问题解答。

    向后兼容入口；新接入建议用 search_knowledge。动态解析真实 kb_id 后查单库。
    """
    async with httpx.AsyncClient(timeout=10.0) as client:
        kb_id = await _resolve_kb_id(client)
    if kb_id:
        rag_results = await _rag_search(query, kb_id, top_k)
        if rag_results is not None:
            return {"results": rag_results, "total": len(rag_results), "source": "rag"}

    results = _simple_search(query, MOCK_FAQ, top_k)
    return {"results": results, "total": len(results), "source": "mock"}


@mcp.tool()
async def search_docs(query: str, top_k: int = 3) -> dict:
    """搜索产品文档库，返回相关的使用指南和技术文档。

    向后兼容入口；新接入建议用 search_knowledge。动态解析真实 kb_id 后查单库。
    """
    async with httpx.AsyncClient(timeout=10.0) as client:
        kb_id = await _resolve_kb_id(client)
    if kb_id:
        rag_results = await _rag_search(query, kb_id, top_k)
        if rag_results is not None:
            return {"results": rag_results, "total": len(rag_results), "source": "rag"}

    results = _simple_search(query, MOCK_DOCS, top_k)
    return {"results": results, "total": len(results), "source": "mock"}


if __name__ == "__main__":
    import uvicorn
    app = mcp.streamable_http_app()
    uvicorn.run(app, host="0.0.0.0", port=8001)

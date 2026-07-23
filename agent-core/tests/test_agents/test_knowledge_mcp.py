import pytest
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", "agent-tools"))

from knowledge_server import server
from knowledge_server.server import search_faq, search_docs, search_knowledge


@pytest.fixture
def force_mock(monkeypatch):
    """把 RAG 指向不可达地址并清缓存，强制走内置 mock 路径（测试 mock 契约用）。

    否则本地若起了 agent-rag(8010)，search_* 会命中真实库，断言失配。
    """
    monkeypatch.setattr(server, "RAG_SERVICE_URL", "http://127.0.0.1:9")
    monkeypatch.setattr(server, "_kb_id_cache", None)


@pytest.mark.asyncio
async def test_search_faq_returns_results(force_mock):
    result = await search_faq("退换货")
    assert "results" in result
    assert result["source"] == "mock"
    assert len(result["results"]) > 0
    assert any("退换货" in r.get("title", "") for r in result["results"])


@pytest.mark.asyncio
async def test_search_faq_returns_score(force_mock):
    result = await search_faq("退换货")
    for item in result["results"]:
        assert "score" in item
        assert item["score"] > 0


@pytest.mark.asyncio
async def test_search_faq_no_results(force_mock):
    result = await search_faq("xyznonexistent123")
    assert result["total"] == 0


@pytest.mark.asyncio
async def test_search_docs_returns_results(force_mock):
    result = await search_docs("无线耳机")
    assert "results" in result
    assert len(result["results"]) > 0


@pytest.mark.asyncio
async def test_search_faq_top_k(force_mock):
    result = await search_faq("退换货", top_k=1)
    assert len(result["results"]) <= 1


@pytest.mark.asyncio
async def test_search_knowledge_mock_fallback(force_mock):
    """统一入口在 RAG 不可达时降级 mock，返回带 shortcut/routed_kbs 的统一结构。"""
    result = await search_knowledge("退换货")
    assert result["source"] == "mock"
    assert result["shortcut"] is False
    assert "routed_kbs" in result
    assert len(result["results"]) > 0


@pytest.mark.asyncio
async def test_search_knowledge_top_k(force_mock):
    result = await search_knowledge("配送", top_k=1)
    assert len(result["results"]) <= 1

import json

import pytest
from httpx import AsyncClient, ASGITransport
from src.api import create_app
from src.config import RAGSettings
from src.db import init_db
from src.pipeline import SmartSplitter
from src.retrieval import (
    RetrievalEngine, MemoryRetriever, reciprocal_rank_fusion, resolve_mode,
    MODE_VECTOR, MODE_FULLTEXT, MODE_HYBRID,
)
from src.routing import SearchRouter, KbPlan, apply_temporal_filters as _apply_temporal
from src.models import SearchRequest, ChunkingStrategy, SearchResult as _SR


@pytest.fixture
async def app():
    # In-memory SQLite + memory retrieval/object-store so tests need no external deps.
    settings = RAGSettings(database_url="sqlite+aiosqlite:///:memory:")
    application = create_app(settings)
    # ASGITransport does not run lifespan, so init the DB factory explicitly.
    application.state.db_session_factory = await init_db(settings.database_url)
    return application


@pytest.fixture
async def client(app):
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        yield c


@pytest.mark.asyncio
async def test_health(client):
    resp = await client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


@pytest.mark.asyncio
async def test_create_and_list_kb(client):
    resp = await client.post("/api/knowledge-bases", params={"name": "测试知识库", "description": "用于测试"})
    assert resp.status_code == 200
    kb = resp.json()
    assert kb["name"] == "测试知识库"
    kb_id = kb["id"]

    resp = await client.get("/api/knowledge-bases")
    assert resp.status_code == 200
    assert len(resp.json()) == 1

    resp = await client.get(f"/api/knowledge-bases/{kb_id}")
    assert resp.status_code == 200
    assert resp.json()["id"] == kb_id


@pytest.mark.asyncio
async def test_delete_kb(client):
    resp = await client.post("/api/knowledge-bases", params={"name": "to_delete"})
    kb_id = resp.json()["id"]

    resp = await client.delete(f"/api/knowledge-bases/{kb_id}")
    assert resp.status_code == 200

    resp = await client.get(f"/api/knowledge-bases/{kb_id}")
    assert resp.status_code == 404


@pytest.mark.asyncio
async def test_update_shortcut_threshold(client):
    # faq 库默认阈值 0.70；可被 PATCH 改到合法值并持久化。
    resp = await client.post("/api/knowledge-bases", params={"name": "faq_thr", "kb_form": "faq"})
    kb = resp.json()
    kb_id = kb["id"]
    assert kb["shortcut_threshold"] == 0.70

    resp = await client.patch(f"/api/knowledge-bases/{kb_id}",
                              params={"shortcut_threshold": 0.8})
    assert resp.status_code == 200
    assert resp.json()["shortcut_threshold"] == 0.8
    # 持久化校验
    assert (await client.get(f"/api/knowledge-bases/{kb_id}")).json()["shortcut_threshold"] == 0.8


@pytest.mark.asyncio
async def test_update_shortcut_threshold_validation(client):
    # 值域 [0,1] 之外 → 400
    faq = (await client.post("/api/knowledge-bases",
                             params={"name": "faq_v", "kb_form": "faq"})).json()
    assert (await client.patch(f"/api/knowledge-bases/{faq['id']}",
                               params={"shortcut_threshold": 1.5})).status_code == 400
    assert (await client.patch(f"/api/knowledge-bases/{faq['id']}",
                               params={"shortcut_threshold": -0.1})).status_code == 400
    # 空 body（无可更新字段）→ 400
    assert (await client.patch(f"/api/knowledge-bases/{faq['id']}")).status_code == 400

    # 非 faq 库不允许改阈值 → 400
    std = (await client.post("/api/knowledge-bases",
                             params={"name": "std_v", "kb_form": "standard"})).json()
    assert (await client.patch(f"/api/knowledge-bases/{std['id']}",
                               params={"shortcut_threshold": 0.5})).status_code == 400

    # 不存在的库 → 404
    assert (await client.patch("/api/knowledge-bases/nope",
                               params={"shortcut_threshold": 0.5})).status_code == 404


@pytest.mark.asyncio
async def test_upload_and_search(client):
    resp = await client.post("/api/knowledge-bases", params={"name": "faq_kb"})
    kb_id = resp.json()["id"]

    content = "# 退换货政策\n\n我们提供7天无理由退换货服务。\n\n# 配送时效\n\n标准配送3-5个工作日送达。"
    resp = await client.post(
        f"/api/knowledge-bases/{kb_id}/documents",
        files={"file": ("faq.md", content.encode(), "text/markdown")},
    )
    assert resp.status_code == 200
    doc = resp.json()
    assert doc["status"] == "active"
    assert doc["chunk_count"] > 0

    resp = await client.post("/api/search", json={"query": "退换货", "kb_id": kb_id, "top_k": 3})
    assert resp.status_code == 200
    data = resp.json()
    assert data["total"] > 0
    assert "退换货" in data["results"][0]["text"]


@pytest.mark.asyncio
async def test_persistence_across_factory(app, client):
    # Data written through the API is in the DB, not process dict: a fresh
    # session sees it.
    resp = await client.post("/api/knowledge-bases", params={"name": "persist_kb"})
    kb_id = resp.json()["id"]

    from src.db import KnowledgeBaseModel
    async with app.state.db_session_factory() as db:
        kb = await db.get(KnowledgeBaseModel, kb_id)
        assert kb is not None
        assert kb.name == "persist_kb"


@pytest.mark.asyncio
async def test_delete_document(client):
    resp = await client.post("/api/knowledge-bases", params={"name": "doc_test"})
    kb_id = resp.json()["id"]

    resp = await client.post(
        f"/api/knowledge-bases/{kb_id}/documents",
        files={"file": ("test.txt", "测试文档内容".encode(), "text/plain")},
    )
    doc_id = resp.json()["id"]

    resp = await client.delete(f"/api/documents/{doc_id}")
    assert resp.status_code == 200

    resp = await client.get(f"/api/documents/{doc_id}")
    assert resp.status_code == 404


async def _make_kb(client, name="ver_kb"):
    resp = await client.post("/api/knowledge-bases", params={"name": name})
    return resp.json()["id"]


async def _upload(client, kb_id, filename, content):
    return await client.post(
        f"/api/knowledge-bases/{kb_id}/documents",
        files={"file": (filename, content.encode(), "text/markdown")},
    )


@pytest.mark.asyncio
async def test_reupload_same_filename_creates_new_version(client):
    kb_id = await _make_kb(client)
    r1 = await _upload(client, kb_id, "policy.md", "# 政策\n\n版本一内容退换货7天")
    doc_id = r1.json()["id"]
    assert r1.json()["version_no"] == 1

    r2 = await _upload(client, kb_id, "policy.md", "# 政策\n\n版本二内容退换货15天")
    assert r2.json()["id"] == doc_id          # same logical document
    assert r2.json()["version_no"] == 2       # new version

    versions = (await client.get(f"/api/documents/{doc_id}/versions")).json()
    assert len(versions) == 2
    current = [v for v in versions if v["is_current"]][0]
    assert current["version_no"] == 2


@pytest.mark.asyncio
async def test_search_sees_only_current_version(client):
    kb_id = await _make_kb(client)
    await _upload(client, kb_id, "policy.md", "# 政策\n\n旧版本说退货7天")
    await _upload(client, kb_id, "policy.md", "# 政策\n\n新版本说退货15天")

    data = (await client.post("/api/search", json={"query": "15天", "kb_id": kb_id, "top_k": 5})).json()
    assert data["total"] > 0
    assert "15天" in data["results"][0]["text"]

    # old version content no longer retrievable
    old = (await client.post("/api/search", json={"query": "7天", "kb_id": kb_id, "top_k": 5})).json()
    assert all("7天" not in r["text"] for r in old["results"])


@pytest.mark.asyncio
async def test_idempotent_same_content(client):
    kb_id = await _make_kb(client)
    await _upload(client, kb_id, "policy.md", "# 同样内容")
    await _upload(client, kb_id, "policy.md", "# 同样内容")  # identical → no new version
    doc_id = (await client.get(f"/api/knowledge-bases/{kb_id}/documents")).json()[0]["id"]
    versions = (await client.get(f"/api/documents/{doc_id}/versions")).json()
    assert len(versions) == 1


@pytest.mark.asyncio
async def test_rollback(client):
    kb_id = await _make_kb(client)
    r1 = await _upload(client, kb_id, "policy.md", "# 政策\n\n版本一退货7天")
    doc_id = r1.json()["id"]
    await _upload(client, kb_id, "policy.md", "# 政策\n\n版本二退货15天")

    # roll back to version 1
    resp = await client.post(f"/api/documents/{doc_id}/rollback", params={"target_version_no": 1})
    assert resp.status_code == 200

    data = (await client.post("/api/search", json={"query": "7天", "kb_id": kb_id, "top_k": 5})).json()
    assert data["total"] > 0
    assert "7天" in data["results"][0]["text"]


@pytest.mark.asyncio
async def test_tenant_isolation(client):
    # KB created under tenant A is invisible to tenant B.
    resp = await client.post("/api/knowledge-bases", params={"name": "a_kb"},
                             headers={"X-Tenant-Id": "tenantA"})
    kb_id = resp.json()["id"]

    assert (await client.get(f"/api/knowledge-bases/{kb_id}",
                             headers={"X-Tenant-Id": "tenantB"})).status_code == 404
    assert (await client.get(f"/api/knowledge-bases/{kb_id}",
                             headers={"X-Tenant-Id": "tenantA"})).status_code == 200
    listed = (await client.get("/api/knowledge-bases", headers={"X-Tenant-Id": "tenantB"})).json()
    assert listed == []


class TestSmartSplitter:
    def test_split_short_text(self):
        splitter = SmartSplitter(chunk_size=1000)
        chunks = splitter.split("短文本", "txt")
        assert len(chunks) == 1
        assert chunks[0].text == "短文本"

    def test_split_by_heading(self):
        text = "# 标题一\n\n内容一\n\n# 标题二\n\n内容二"
        splitter = SmartSplitter(chunk_size=500)
        chunks = splitter.split(text, "md", ChunkingStrategy.HEADING)
        assert len(chunks) >= 2

    def test_split_recursive(self):
        text = "段落一。\n\n段落二。\n\n段落三。"
        splitter = SmartSplitter(chunk_size=10)
        chunks = splitter.split(text, "txt", ChunkingStrategy.RECURSIVE)
        assert len(chunks) >= 2

    def test_split_qa_pair(self):
        text = "Q：什么是退货政策？\nA：7天无理由退货。\n\nQ：配送多久？\nA：3-5天。"
        splitter = SmartSplitter(chunk_size=500)
        chunks = splitter.split(text, "txt", ChunkingStrategy.QA_PAIR)
        assert len(chunks) >= 2


class TestRetrievalEngine:
    @pytest.mark.asyncio
    async def test_index_and_search(self):
        engine = RetrievalEngine()
        await engine.index_chunks("kb1", [
            {"id": "c1", "doc_id": "d1", "text": "退换货政策说明", "keywords": ["退换货"], "metadata": {}},
            {"id": "c2", "doc_id": "d1", "text": "配送时效说明", "keywords": ["配送"], "metadata": {}},
        ])
        results = await engine.search(SearchRequest(query="退换货", kb_id="kb1", top_k=5))
        assert len(results) > 0
        assert "退换货" in results[0].text

    @pytest.mark.asyncio
    async def test_search_empty_kb(self):
        engine = RetrievalEngine()
        results = await engine.search(SearchRequest(query="test", kb_id="nonexistent", top_k=5))
        assert results == []


# ===== 步骤3: 元数据过滤 =====


async def _upload_meta(client, kb_id, filename, content, metadata=None):
    files = {"file": (filename, content.encode("utf-8"), "text/plain")}
    data = {}
    if metadata is not None:
        data["metadata"] = json.dumps(metadata)
    return await client.post(f"/api/knowledge-bases/{kb_id}/documents", files=files, data=data)


@pytest.mark.asyncio
async def test_metadata_field_crud(client):
    kb_id = (await client.post("/api/knowledge-bases", params={"name": "meta_kb"})).json()["id"]
    # 创建字段
    r = await client.post(f"/api/knowledge-bases/{kb_id}/metadata-fields",
                          params={"name": "category", "field_type": "string"})
    assert r.status_code == 200
    field_id = r.json()["id"]
    # 重复创建 → 409
    assert (await client.post(f"/api/knowledge-bases/{kb_id}/metadata-fields",
                              params={"name": "category"})).status_code == 409
    # 非法类型 → 400
    assert (await client.post(f"/api/knowledge-bases/{kb_id}/metadata-fields",
                              params={"name": "x", "field_type": "bogus"})).status_code == 400
    # 列出
    listed = (await client.get(f"/api/knowledge-bases/{kb_id}/metadata-fields")).json()
    assert len(listed) == 1 and listed[0]["name"] == "category"
    # 删除
    assert (await client.delete(f"/api/knowledge-bases/{kb_id}/metadata-fields/{field_id}")).status_code == 200
    assert (await client.get(f"/api/knowledge-bases/{kb_id}/metadata-fields")).json() == []


@pytest.mark.asyncio
async def test_upload_rejects_undefined_metadata(client):
    kb_id = (await client.post("/api/knowledge-bases", params={"name": "strict_kb"})).json()["id"]
    # 未定义任何字段，带 metadata 上传 → 拒绝
    r = await _upload_meta(client, kb_id, "a.txt", "退换货政策", {"category": "耳机"})
    assert r.status_code == 400


@pytest.mark.asyncio
async def test_upload_validates_and_filters_by_metadata(client):
    kb_id = (await client.post("/api/knowledge-bases", params={"name": "filter_kb"})).json()["id"]
    await client.post(f"/api/knowledge-bases/{kb_id}/metadata-fields",
                      params={"name": "category", "field_type": "string"})
    # 合法元数据上传两个文档
    await _upload_meta(client, kb_id, "earphone.txt", "无线耳机退换货说明", {"category": "耳机"})
    await _upload_meta(client, kb_id, "phone.txt", "手机退换货说明", {"category": "手机"})
    # 不过滤 → 两个都命中
    all_hits = (await client.post("/api/search", json={"query": "退换货", "kb_id": kb_id, "top_k": 5})).json()
    assert all_hits["total"] == 2
    # 过滤 category==耳机 → 只命中耳机
    filtered = (await client.post("/api/search", json={
        "query": "退换货", "kb_id": kb_id, "top_k": 5,
        "filters": [{"field": "category", "op": "eq", "value": "耳机"}],
    })).json()
    assert filtered["total"] == 1
    assert "耳机" in filtered["results"][0]["text"]


@pytest.mark.asyncio
async def test_upload_rejects_bad_metadata_value(client):
    kb_id = (await client.post("/api/knowledge-bases", params={"name": "num_kb"})).json()["id"]
    await client.post(f"/api/knowledge-bases/{kb_id}/metadata-fields",
                      params={"name": "battery", "field_type": "number"})
    # number 字段给非数字 → 400
    r = await _upload_meta(client, kb_id, "x.txt", "内容", {"battery": "不是数字"})
    assert r.status_code == 400


@pytest.mark.asyncio
async def test_temporal_kb_filters_expired_docs(client):
    kb_id = (await client.post("/api/knowledge-bases",
                               params={"name": "promo", "kb_form": "temporal"})).json()["id"]
    await client.post(f"/api/knowledge-bases/{kb_id}/metadata-fields",
                      params={"name": "effective_ts", "field_type": "time"})
    await client.post(f"/api/knowledge-bases/{kb_id}/metadata-fields",
                      params={"name": "expire_ts", "field_type": "time"})
    # 有效活动：生效在过去、失效在未来
    await _upload_meta(client, kb_id, "active.txt", "双十一促销活动满300减50",
                       {"effective_ts": 1000000000, "expire_ts": 4000000000})
    # 过期活动：已失效（expire_ts 在过去）
    await _upload_meta(client, kb_id, "expired.txt", "双十一促销活动满200减30",
                       {"effective_ts": 1000000000, "expire_ts": 1000000100})
    # temporal 库自动注入有效期过滤 → 只命中未过期的
    hits = (await client.post("/api/search", json={"query": "双十一促销活动", "kb_id": kb_id, "top_k": 5})).json()
    assert hits["total"] == 1
    assert "满300减50" in hits["results"][0]["text"]


# ===== 步骤4: 混合检索（三模式 + RRF 融合） =====


def test_resolve_mode():
    # 请求级覆盖库级
    assert resolve_mode("vector", "hybrid") == MODE_VECTOR
    assert resolve_mode(None, "fulltext") == MODE_FULLTEXT
    # 都缺省 → hybrid
    assert resolve_mode(None, None) == MODE_HYBRID
    # 非法值兜底 hybrid
    assert resolve_mode("bogus", None) == MODE_HYBRID
    assert resolve_mode(None, "garbage") == MODE_HYBRID


def test_rrf_fusion_accumulates_and_orders():
    a = [_SR(chunk_id="c1", doc_id="d", text="t1", score=0.9),
         _SR(chunk_id="c2", doc_id="d", text="t2", score=0.8)]
    b = [_SR(chunk_id="c2", doc_id="d", text="t2", score=0.7),
         _SR(chunk_id="c3", doc_id="d", text="t3", score=0.6)]
    fused = reciprocal_rank_fusion([a, b])
    ids = [r.chunk_id for r in fused]
    # c2 两路命中（a 第2 + b 第1）应排首位
    assert ids[0] == "c2"
    assert set(ids) == {"c1", "c2", "c3"}
    # 融合分写回 score，且 c2 分最高
    assert fused[0].score == max(r.score for r in fused)


def test_rrf_fusion_weights():
    a = [_SR(chunk_id="x", doc_id="d", text="t", score=1.0)]
    b = [_SR(chunk_id="y", doc_id="d", text="t", score=1.0)]
    # b 权重更高 → y 排前
    fused = reciprocal_rank_fusion([a, b], weights=[0.1, 0.9])
    assert fused[0].chunk_id == "y"


@pytest.mark.asyncio
async def test_memory_three_modes():
    r = MemoryRetriever()
    await r.index_chunks("kb1", [
        {"id": "c1", "doc_id": "d1", "kb_id": "kb1", "text": "无线蓝牙耳机续航说明",
         "keywords": ["耳机", "续航"], "metadata": {}},
        {"id": "c2", "doc_id": "d1", "kb_id": "kb1", "text": "有线耳机降噪说明",
         "keywords": ["耳机", "降噪"], "metadata": {}},
    ])
    req = SearchRequest(query="耳机续航", kb_id="kb1", top_k=5)
    # 三模式都应能召回，且不报错
    for mode in (MODE_VECTOR, MODE_FULLTEXT, MODE_HYBRID):
        out = await r.search(req, kb_mode=mode)
        assert len(out) > 0
    # hybrid 下与续航相关的 c1 应排前
    hybrid = await r.search(req, kb_mode=MODE_HYBRID)
    assert hybrid[0].chunk_id == "c1"


@pytest.mark.asyncio
async def test_request_mode_overrides_kb_mode():
    r = MemoryRetriever()
    await r.index_chunks("kb1", [
        {"id": "c1", "doc_id": "d1", "kb_id": "kb1", "text": "退换货政策七天无理由",
         "keywords": ["退换货"], "metadata": {}},
    ])
    # 库级 hybrid，请求级强制 fulltext
    req = SearchRequest(query="退换货", kb_id="kb1", top_k=5, mode="fulltext")
    out = await r.search(req, kb_mode="hybrid")
    assert len(out) == 1 and out[0].chunk_id == "c1"


@pytest.mark.asyncio
async def test_search_endpoint_respects_request_mode(client):
    kb_id = (await client.post("/api/knowledge-bases",
                               params={"name": "mode_kb"})).json()["id"]
    await _upload(client, kb_id, "policy.md", "# 政策\n\n退换货七天无理由退货")
    # 显式 mode=vector 走端到端不报错
    data = (await client.post("/api/search", json={
        "query": "退换货", "kb_id": kb_id, "top_k": 5, "mode": "vector",
    })).json()
    assert data["total"] >= 1


# ===== 步骤5: 检索路由层（faq 短路 + 跨库加权 RRF） =====


@pytest.mark.asyncio
async def test_router_faq_shortcut():
    r = MemoryRetriever()
    # faq 库：完全匹配的 QA → 高分触发短路
    await r.index_chunks("faq_kb", [
        {"id": "f1", "doc_id": "d", "kb_id": "faq_kb", "text": "退换货政策",
         "keywords": ["退换货"], "metadata": {}},
    ])
    await r.index_chunks("std_kb", [
        {"id": "s1", "doc_id": "d", "kb_id": "std_kb", "text": "退换货的详细说明文档",
         "keywords": ["退换货"], "metadata": {}},
    ])
    router = SearchRouter(r)
    plans = [
        KbPlan(kb_id="faq_kb", kb_form="faq", retrieval_mode="fulltext",
               priority_weight=1.0, shortcut_threshold=0.5,
               visible_version_ids=None),
        KbPlan(kb_id="std_kb", kb_form="standard", retrieval_mode="hybrid",
               priority_weight=0.7, visible_version_ids=None),
    ]
    resp = await router.route("退换货政策", plans, top_k=3)
    # 整串命中 faq → 短路，只返回 faq 库
    assert resp.shortcut is True
    assert resp.routed_kbs == ["faq_kb"]
    assert resp.results[0].chunk_id == "f1"


@pytest.mark.asyncio
async def test_router_cross_kb_weighted_rrf():
    r = MemoryRetriever()
    await r.index_chunks("kb_a", [
        {"id": "a1", "doc_id": "d", "kb_id": "kb_a", "text": "配送时效三到五天",
         "keywords": ["配送"], "metadata": {}},
    ])
    await r.index_chunks("kb_b", [
        {"id": "b1", "doc_id": "d", "kb_id": "kb_b", "text": "配送范围全国覆盖",
         "keywords": ["配送"], "metadata": {}},
    ])
    router = SearchRouter(r)
    # kb_b 权重远高 → b1 应排前
    plans = [
        KbPlan(kb_id="kb_a", priority_weight=0.1, visible_version_ids=None),
        KbPlan(kb_id="kb_b", priority_weight=0.9, visible_version_ids=None),
    ]
    resp = await router.route("配送", plans, top_k=5)
    assert resp.shortcut is False
    assert resp.results[0].chunk_id == "b1"
    assert set(resp.routed_kbs) == {"kb_a", "kb_b"}


@pytest.mark.asyncio
async def test_router_empty_plans():
    router = SearchRouter(MemoryRetriever())
    resp = await router.route("任意", [], top_k=5)
    assert resp.total == 0 and resp.results == []


# ===== rerank 精排（接法 A：RRF 粗排 → cross-encoder 精排） =====


class _FakeReranker:
    """测试用 reranker：工作时反转 RRF 顺序，作为确定性的"精排改变了排序"信号。"""

    class _S:
        rerank_candidate_n = 20

    def __init__(self, enabled=True, fail=False):
        self._enabled = enabled
        self._fail = fail
        self.settings = self._S()

    @property
    def enabled(self):
        return self._enabled

    async def rerank(self, query, results, top_k):
        if self._fail:
            # 模拟模型异常时调用方应保留 RRF 顺序
            return results[:top_k], False
        # 反转顺序，验证精排确实改写了 RRF 排序
        ranked = list(reversed(results))
        for i, r in enumerate(ranked[:top_k]):
            r.score = round(1.0 - i * 0.01, 6)
        return ranked[:top_k], True


async def _two_kb_router(reranker):
    # 两库 chunk 都含"配送"，同一 query 都能召回 → 权重才真正决定 RRF 粗排顺序
    r = MemoryRetriever()
    await r.index_chunks("kb_a", [
        {"id": "a1", "doc_id": "d", "kb_id": "kb_a", "text": "配送范围全国覆盖",
         "keywords": ["配送"], "metadata": {}},
    ])
    await r.index_chunks("kb_b", [
        {"id": "b1", "doc_id": "d", "kb_id": "kb_b", "text": "配送时效三到五天",
         "keywords": ["配送"], "metadata": {}},
    ])
    return SearchRouter(r, reranker=reranker)


@pytest.mark.asyncio
async def test_router_rerank_reorders_and_flags():
    # kb_a 权重高 → RRF 粗排 a1 在前；reranker 反转 → 精排后 b1 顶到第一
    router = await _two_kb_router(_FakeReranker(enabled=True))
    plans = [
        KbPlan(kb_id="kb_a", priority_weight=0.9, visible_version_ids=None),
        KbPlan(kb_id="kb_b", priority_weight=0.1, visible_version_ids=None),
    ]
    resp = await router.route("配送", plans, top_k=5)
    assert resp.shortcut is False
    assert resp.reranked is True
    assert resp.results[0].chunk_id == "b1"  # 精排改写了 RRF 顺序


@pytest.mark.asyncio
async def test_router_rerank_disabled_keeps_rrf():
    # reranker.enabled=False → 退回 RRF 顺序，reranked=False
    router = await _two_kb_router(_FakeReranker(enabled=False))
    plans = [
        KbPlan(kb_id="kb_a", priority_weight=0.9, visible_version_ids=None),
        KbPlan(kb_id="kb_b", priority_weight=0.1, visible_version_ids=None),
    ]
    resp = await router.route("配送", plans, top_k=5)
    assert resp.reranked is False
    assert resp.results[0].chunk_id == "a1"  # 仍是 RRF 粗排顺序


@pytest.mark.asyncio
async def test_router_rerank_failure_falls_back():
    # rerank 抛错路径由 fake 的 fail 模拟 → reranked=False、保留 RRF 顺序
    router = await _two_kb_router(_FakeReranker(enabled=True, fail=True))
    plans = [
        KbPlan(kb_id="kb_a", priority_weight=0.9, visible_version_ids=None),
        KbPlan(kb_id="kb_b", priority_weight=0.1, visible_version_ids=None),
    ]
    resp = await router.route("配送", plans, top_k=5)
    assert resp.reranked is False
    assert resp.results[0].chunk_id == "a1"


@pytest.mark.asyncio
async def test_reranker_disabled_provider_is_noop():
    # Reranker(provider=disabled) 是 no-op：enabled=False、rerank 原样返回
    from src.rerank import Reranker
    from src.config import RAGSettings

    rr = Reranker(RAGSettings(rerank_provider="disabled"))
    assert rr.enabled is False
    items = ["x", "y", "z"]

    class _R:
        def __init__(self, t):
            self.text = t
            self.score = 0.0

    rs = [_R(t) for t in items]
    out, flag = await rr.rerank("q", rs, top_k=2)
    assert flag is False
    assert [r.text for r in out] == ["x", "y"]  # 原序、截断到 top_k


def test_apply_temporal_filters_only_for_temporal():
    field_defs = {"effective_ts": "time", "expire_ts": "time"}
    # 非 temporal 库不注入
    assert _apply_temporal("standard", field_defs, []) == []
    # temporal 库注入两个边界过滤
    out = _apply_temporal("temporal", field_defs, [])
    fields = {f.field for f in out}
    assert fields == {"effective_ts", "expire_ts"}


@pytest.mark.asyncio
async def test_route_search_endpoint(client):
    # faq 库（短路）+ standard 库
    faq_id = (await client.post("/api/knowledge-bases",
                                params={"name": "faq", "kb_form": "faq"})).json()["id"]
    std_id = (await client.post("/api/knowledge-bases",
                                params={"name": "std", "kb_form": "standard"})).json()["id"]
    await _upload(client, faq_id, "faq.txt", "Q：退换货政策？\nA：七天无理由。")
    await _upload(client, std_id, "doc.txt", "# 售后\n\n退换货的完整流程说明")
    # 不限定 scope → 两库都参与
    resp = (await client.post("/api/route-search",
                              json={"query": "退换货", "top_k": 5})).json()
    assert resp["total"] >= 1
    assert len(resp["routed_kbs"]) >= 1
    # scope 限定只查 standard 库
    scoped = (await client.post("/api/route-search",
                                json={"query": "退换货", "top_k": 5, "scope": [std_id]})).json()
    assert scoped["routed_kbs"] == [std_id]


@pytest.mark.asyncio
async def test_route_search_scope_by_kb_form(client):
    await client.post("/api/knowledge-bases", params={"name": "f1", "kb_form": "faq"})
    std_id = (await client.post("/api/knowledge-bases",
                                params={"name": "s1", "kb_form": "standard"})).json()["id"]
    await _upload(client, std_id, "doc.txt", "# 政策\n\n保修一年的说明")
    # scope 用 kb_form="standard" 限定
    resp = (await client.post("/api/route-search",
                              json={"query": "保修", "top_k": 5, "scope": ["standard"]})).json()
    assert resp["routed_kbs"] == [std_id]

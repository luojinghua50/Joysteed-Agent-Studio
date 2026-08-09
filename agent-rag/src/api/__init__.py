import json
from datetime import datetime, timezone

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Header, Depends
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from sqlalchemy import select

from src.config import RAGSettings
from src.models import ChunkingStrategy, SearchRequest, SearchResponse, RouteSearchRequest, RouteSearchResponse
from src.pipeline import SmartSplitter
from src.retrieval import create_retriever
from src.routing import SearchRouter, KbPlan, apply_temporal_filters
from src.rerank import Reranker
from src.storage import ObjectStore
from src.db import (
    init_db, KnowledgeBaseModel, DocumentModel, DocumentVersionModel,
    KbMetadataFieldModel,
)
from src.versioning import VersionManager, visible_version_ids, short_id, _audit


def _tenant(x_tenant_id: str = Header(default="default")) -> str:
    """Tenant from header. In production this is derived from auth.

    See docs/auth-user-design.md — replace this with token-derived tenant.
    """
    return x_tenant_id


VALID_KB_FORMS = ("faq", "standard", "temporal", "multimodal")
VALID_RETRIEVAL_MODES = ("vector", "fulltext", "hybrid")


def _form_defaults(kb_form: str) -> dict:
    """知识形态模板 → 默认配置。模板只预填，显式参数可覆盖。"""
    table = {
        "faq":        {"chunking_strategy": "qa_pair", "retrieval_mode": "hybrid",
                       "priority_weight": 1.0, "vector_weight": 0.7, "keyword_weight": 0.3,
                       "score_threshold": 0.0, "shortcut_threshold": 0.70},
        "standard":   {"chunking_strategy": "heading", "retrieval_mode": "hybrid",
                       "priority_weight": 0.7, "vector_weight": 0.6, "keyword_weight": 0.4,
                       "score_threshold": 0.0, "shortcut_threshold": 0.0},
        "temporal":   {"chunking_strategy": "auto", "retrieval_mode": "hybrid",
                       "priority_weight": 0.5, "vector_weight": 0.6, "keyword_weight": 0.4,
                       "score_threshold": 0.0, "shortcut_threshold": 0.0},
        "multimodal": {"chunking_strategy": "auto", "retrieval_mode": "vector",
                       "priority_weight": 0.3, "vector_weight": 1.0, "keyword_weight": 0.0,
                       "score_threshold": 0.0, "shortcut_threshold": 0.0},
    }
    return table.get(kb_form, table["standard"])


def _validate_ratio(name: str, value: float):
    if not 0.0 <= value <= 1.0:
        raise HTTPException(status_code=400, detail=f"{name} 必须在 [0, 1] 之间")


def _validate_chunk_params(chunk_size: int, chunk_overlap: int):
    if chunk_size < 100 or chunk_size > 4000:
        raise HTTPException(status_code=400, detail="chunk_size 必须在 [100, 4000] 之间")
    if chunk_overlap < 0 or chunk_overlap >= chunk_size:
        raise HTTPException(status_code=400, detail="chunk_overlap 必须 >=0 且小于 chunk_size")


def _validate_top_k(top_k: int):
    if top_k < 1 or top_k > 50:
        raise HTTPException(status_code=400, detail="top_k 必须在 [1, 50] 之间")


def _validate_hybrid_weights(vector_weight: float, keyword_weight: float):
    if abs((vector_weight + keyword_weight) - 1.0) > 0.001:
        raise HTTPException(status_code=400, detail="vector_weight + keyword_weight 必须等于 1")


def _chunking_value(strategy: str) -> str:
    try:
        return ChunkingStrategy(strategy).value
    except ValueError:
        raise HTTPException(status_code=400, detail=f"invalid chunking_strategy: {strategy}")


def _apply_kb_search_config(request: SearchRequest, kb: KnowledgeBaseModel) -> SearchRequest:
    request.top_k = request.top_k or kb.top_k
    request.embedding_model = request.embedding_model or kb.embedding_model
    request.vector_weight = kb.vector_weight
    request.keyword_weight = kb.keyword_weight
    request.score_threshold = kb.score_threshold
    return request


def _kb_dict(kb: KnowledgeBaseModel) -> dict:
    return {
        "id": kb.id, "tenant_id": kb.tenant_id, "name": kb.name,
        "description": kb.description, "chunking_strategy": kb.chunking_strategy,
        "chunk_size": kb.chunk_size, "chunk_overlap": kb.chunk_overlap,
        "embedding_model": kb.embedding_model, "rerank_model": kb.rerank_model,
        "document_count": kb.document_count,
        "kb_form": kb.kb_form, "retrieval_mode": kb.retrieval_mode,
        "top_k": kb.top_k,
        "priority_weight": kb.priority_weight, "vector_weight": kb.vector_weight,
        "keyword_weight": kb.keyword_weight, "score_threshold": kb.score_threshold,
        "shortcut_threshold": kb.shortcut_threshold,
        "created_at": kb.created_at.isoformat() if kb.created_at else None,
    }


def _doc_dict(doc: DocumentModel, ver: DocumentVersionModel | None = None) -> dict:
    d = {
        "id": doc.id, "tenant_id": doc.tenant_id, "kb_id": doc.kb_id,
        "filename": doc.filename, "current_version_id": doc.current_version_id,
    }
    if ver is not None:
        d.update({
            "version_no": ver.version_no, "status": ver.status,
            "file_type": ver.file_type, "file_size": ver.file_size,
            "chunk_count": ver.chunk_count,
        })
    return d


async def _require_kb(db, kb_id: str, tenant_id: str) -> KnowledgeBaseModel:
    kb = await db.get(KnowledgeBaseModel, kb_id)
    if not kb or kb.tenant_id != tenant_id:
        raise HTTPException(status_code=404, detail="Knowledge base not found")
    return kb


async def _require_doc(db, doc_id: str, tenant_id: str) -> DocumentModel:
    doc = await db.get(DocumentModel, doc_id)
    if not doc or doc.tenant_id != tenant_id:
        raise HTTPException(status_code=404, detail="Document not found")
    return doc


async def _kb_field_defs(db, kb_id: str) -> dict[str, str]:
    """库的元数据字段定义 {name: field_type}。无定义时返回空（不强校验）。"""
    rows = (await db.execute(
        select(KbMetadataFieldModel).where(KbMetadataFieldModel.kb_id == kb_id)
    )).scalars().all()
    return {r.name: r.field_type for r in rows}


def _coerce_meta_value(field_type: str, raw):
    """按字段类型把上传的元数据值规整化；非法值抛 ValueError。

    string→str，number→float/int，time→epoch 秒(int)。时间接受 epoch 数字或 ISO 字符串。
    """
    if raw is None:
        return None
    if field_type == "string":
        return str(raw)
    if field_type == "number":
        if isinstance(raw, bool):
            raise ValueError("number 字段不接受布尔值")
        return float(raw) if isinstance(raw, str) and "." in raw else (raw if isinstance(raw, (int, float)) else float(raw))
    if field_type == "time":
        if isinstance(raw, (int, float)) and not isinstance(raw, bool):
            return int(raw)
        if isinstance(raw, str):
            s = raw.strip()
            if s.isdigit():
                return int(s)
            dt = datetime.fromisoformat(s.replace("Z", "+00:00"))
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return int(dt.timestamp())
        raise ValueError(f"无法解析 time 值: {raw!r}")
    raise ValueError(f"未知字段类型: {field_type}")


def _validate_metadata(raw: dict, field_defs: dict[str, str]) -> dict:
    """按库的字段定义校验+规整上传元数据。未定义字段直接拒绝，避免脏字段污染索引。"""
    if not raw:
        return {}
    out = {}
    for k, v in raw.items():
        if k not in field_defs:
            raise HTTPException(status_code=400, detail=f"未定义的元数据字段: {k}")
        try:
            out[k] = _coerce_meta_value(field_defs[k], v)
        except (ValueError, TypeError) as e:
            raise HTTPException(status_code=400, detail=f"元数据字段 {k} 值非法: {e}")
    return out


def create_app(settings: RAGSettings | None = None) -> FastAPI:
    if settings is None:
        settings = RAGSettings()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.db_session_factory = await init_db(settings.database_url)
        yield

    app = FastAPI(
        title="Agent RAG - 知识库服务",
        description="RAG Knowledge Base Service (persistent + versioned)",
        version="0.2.0",
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.state.settings = settings
    app.state.retriever = create_retriever(settings)
    app.state.store = ObjectStore(settings)
    app.state.splitter = SmartSplitter(
        chunk_size=settings.default_chunk_size,
        chunk_overlap=settings.default_chunk_overlap,
    )
    app.state.vm = VersionManager(settings, app.state.retriever, app.state.store)
    app.state.reranker = Reranker(settings)
    app.state.router = SearchRouter(app.state.retriever, reranker=app.state.reranker)

    def db_factory():
        return app.state.db_session_factory()

    @app.get("/health")
    async def health():
        return {"status": "ok", "service": "agent-rag", "version": "0.2.0",
                "retrieval_backend": settings.retrieval_backend}

    # ===== Knowledge Base CRUD =====
    @app.post("/api/knowledge-bases")
    async def create_kb(name: str, description: str = "", strategy: str | None = None,
                        kb_form: str = "standard", chunk_size: int | None = None,
                        chunk_overlap: int | None = None, retrieval_mode: str | None = None,
                        top_k: int | None = None,
                        priority_weight: float | None = None, vector_weight: float | None = None,
                        keyword_weight: float | None = None, score_threshold: float | None = None,
                        shortcut_threshold: float | None = None, embedding_model: str | None = None,
                        rerank_model: str | None = None, tenant_id: str = Depends(_tenant)):
        """建库：模板预填默认值，显式参数覆盖，最终配置落库。"""
        if kb_form not in VALID_KB_FORMS:
            raise HTTPException(status_code=400, detail=f"invalid kb_form: {kb_form}")
        defaults = _form_defaults(kb_form)
        chunking = _chunking_value(strategy) if strategy else defaults["chunking_strategy"]
        final_chunk_size = chunk_size if chunk_size is not None else settings.default_chunk_size
        final_chunk_overlap = chunk_overlap if chunk_overlap is not None else settings.default_chunk_overlap
        _validate_chunk_params(final_chunk_size, final_chunk_overlap)
        final_top_k = top_k if top_k is not None else 5
        _validate_top_k(final_top_k)
        final_retrieval_mode = retrieval_mode or defaults["retrieval_mode"]
        if final_retrieval_mode not in VALID_RETRIEVAL_MODES:
            raise HTTPException(status_code=400, detail=f"invalid retrieval_mode: {final_retrieval_mode}")
        final_priority_weight = defaults["priority_weight"] if priority_weight is None else priority_weight
        final_vector_weight = defaults["vector_weight"] if vector_weight is None else vector_weight
        final_keyword_weight = defaults["keyword_weight"] if keyword_weight is None else keyword_weight
        final_score_threshold = defaults["score_threshold"] if score_threshold is None else score_threshold
        final_shortcut_threshold = (
            defaults["shortcut_threshold"] if shortcut_threshold is None else shortcut_threshold
        )
        for field, value in (
            ("priority_weight", final_priority_weight),
            ("vector_weight", final_vector_weight),
            ("keyword_weight", final_keyword_weight),
            ("score_threshold", final_score_threshold),
            ("shortcut_threshold", final_shortcut_threshold),
        ):
            _validate_ratio(field, value)
        _validate_hybrid_weights(final_vector_weight, final_keyword_weight)
        kb_id = short_id()
        async with db_factory() as db:
            kb = KnowledgeBaseModel(
                id=kb_id, tenant_id=tenant_id, name=name, description=description,
                chunking_strategy=chunking, chunk_size=final_chunk_size,
                chunk_overlap=final_chunk_overlap,
                embedding_model=embedding_model or settings.embedding_model,
                rerank_model=rerank_model or settings.rerank_model,
                kb_form=kb_form,
                collection_name=f"kb_{kb_id}",
                retrieval_mode=final_retrieval_mode,
                top_k=final_top_k,
                priority_weight=final_priority_weight,
                vector_weight=final_vector_weight,
                keyword_weight=final_keyword_weight,
                score_threshold=final_score_threshold,
                shortcut_threshold=final_shortcut_threshold,
            )
            db.add(kb)
            await db.commit()
        return _kb_dict(kb)

    @app.get("/api/knowledge-bases")
    async def list_kbs(tenant_id: str = Depends(_tenant)):
        async with db_factory() as db:
            rows = (await db.execute(
                select(KnowledgeBaseModel).where(KnowledgeBaseModel.tenant_id == tenant_id)
            )).scalars().all()
            return [_kb_dict(kb) for kb in rows]

    @app.get("/api/knowledge-bases/{kb_id}")
    async def get_kb(kb_id: str, tenant_id: str = Depends(_tenant)):
        async with db_factory() as db:
            kb = await _require_kb(db, kb_id, tenant_id)
            return _kb_dict(kb)

    @app.patch("/api/knowledge-bases/{kb_id}")
    async def update_kb(kb_id: str, name: str | None = None, description: str | None = None,
                        kb_form: str | None = None, chunking_strategy: str | None = None,
                        chunk_size: int | None = None, chunk_overlap: int | None = None,
                        retrieval_mode: str | None = None, top_k: int | None = None,
                        priority_weight: float | None = None,
                        vector_weight: float | None = None, keyword_weight: float | None = None,
                        score_threshold: float | None = None, shortcut_threshold: float | None = None,
                        embedding_model: str | None = None, rerank_model: str | None = None,
                        tenant_id: str = Depends(_tenant)):
        """更新知识库配置。

        检索参数可热更新；chunking_strategy/chunk_size/chunk_overlap/embedding_model
        影响后续新版本构建，已有文档需重新上传/重建索引后才完全生效。
        """
        if all(v is None for v in (
            name, description, kb_form, chunking_strategy, chunk_size, chunk_overlap,
            retrieval_mode, top_k, priority_weight, vector_weight, keyword_weight,
            score_threshold, shortcut_threshold,
            embedding_model, rerank_model,
        )):
            raise HTTPException(status_code=400, detail="无可更新字段")
        async with db_factory() as db:
            kb = await _require_kb(db, kb_id, tenant_id)
            before = _kb_dict(kb)
            if name is not None:
                next_name = name.strip()
                if not next_name:
                    raise HTTPException(status_code=400, detail="name cannot be empty")
                kb.name = next_name
            if description is not None:
                kb.description = description
            if kb_form is not None:
                if kb_form not in VALID_KB_FORMS:
                    raise HTTPException(status_code=400, detail=f"invalid kb_form: {kb_form}")
                kb.kb_form = kb_form
            if chunking_strategy is not None:
                kb.chunking_strategy = _chunking_value(chunking_strategy)
            next_chunk_size = kb.chunk_size if chunk_size is None else chunk_size
            next_chunk_overlap = kb.chunk_overlap if chunk_overlap is None else chunk_overlap
            if chunk_size is not None or chunk_overlap is not None:
                _validate_chunk_params(next_chunk_size, next_chunk_overlap)
                kb.chunk_size = next_chunk_size
                kb.chunk_overlap = next_chunk_overlap
            if retrieval_mode is not None:
                if retrieval_mode not in VALID_RETRIEVAL_MODES:
                    raise HTTPException(status_code=400, detail=f"invalid retrieval_mode: {retrieval_mode}")
                kb.retrieval_mode = retrieval_mode
            if top_k is not None:
                _validate_top_k(top_k)
                kb.top_k = top_k
            for field, value in (
                ("priority_weight", priority_weight),
                ("vector_weight", vector_weight),
                ("keyword_weight", keyword_weight),
                ("score_threshold", score_threshold),
                ("shortcut_threshold", shortcut_threshold),
            ):
                if value is not None:
                    _validate_ratio(field, value)
                    setattr(kb, field, value)
            if embedding_model is not None:
                kb.embedding_model = embedding_model.strip()
            if rerank_model is not None:
                kb.rerank_model = rerank_model.strip()
            _validate_hybrid_weights(kb.vector_weight, kb.keyword_weight)
            await _audit(db, tenant_id, "admin", "update_kb_config", "knowledge_base", kb.id,
                         {"from": before, "to": _kb_dict(kb)})
            await db.commit()
            await db.refresh(kb)
            return _kb_dict(kb)

    @app.delete("/api/knowledge-bases/{kb_id}")
    async def delete_kb(kb_id: str, tenant_id: str = Depends(_tenant)):
        async with db_factory() as db:
            kb = await _require_kb(db, kb_id, tenant_id)
            await db.delete(kb)
            await db.commit()
        await app.state.retriever.delete_kb(kb_id)
        await app.state.store.delete_prefix(f"{tenant_id}/{kb_id}/")
        return {"status": "deleted", "kb_id": kb_id}

    # ===== Metadata Field Definitions (库内过滤字段) =====
    @app.get("/api/knowledge-bases/{kb_id}/metadata-fields")
    async def list_metadata_fields(kb_id: str, tenant_id: str = Depends(_tenant)):
        async with db_factory() as db:
            await _require_kb(db, kb_id, tenant_id)
            rows = (await db.execute(
                select(KbMetadataFieldModel).where(KbMetadataFieldModel.kb_id == kb_id)
            )).scalars().all()
            return [{"id": r.id, "name": r.name, "field_type": r.field_type} for r in rows]

    @app.post("/api/knowledge-bases/{kb_id}/metadata-fields")
    async def create_metadata_field(kb_id: str, name: str, field_type: str = "string",
                                    tenant_id: str = Depends(_tenant)):
        """定义库内可过滤字段。string|number|time。time 用于 temporal 库的有效期过滤。"""
        if field_type not in ("string", "number", "time"):
            raise HTTPException(status_code=400, detail=f"invalid field_type: {field_type}")
        async with db_factory() as db:
            await _require_kb(db, kb_id, tenant_id)
            exists = (await db.execute(
                select(KbMetadataFieldModel).where(
                    KbMetadataFieldModel.kb_id == kb_id, KbMetadataFieldModel.name == name
                )
            )).scalar_one_or_none()
            if exists:
                raise HTTPException(status_code=409, detail=f"字段已存在: {name}")
            field = KbMetadataFieldModel(
                id=short_id(), tenant_id=tenant_id, kb_id=kb_id,
                name=name, field_type=field_type,
            )
            db.add(field)
            await db.commit()
        # 给该字段建标量索引（Milvus 实现；memory 后端为 no-op，collection 未建则跳过）
        await app.state.retriever.ensure_scalar_index(kb_id, name, field_type)
        return {"id": field.id, "name": field.name, "field_type": field.field_type}

    @app.delete("/api/knowledge-bases/{kb_id}/metadata-fields/{field_id}")
    async def delete_metadata_field(kb_id: str, field_id: str, tenant_id: str = Depends(_tenant)):
        async with db_factory() as db:
            await _require_kb(db, kb_id, tenant_id)
            field = await db.get(KbMetadataFieldModel, field_id)
            if not field or field.kb_id != kb_id or field.tenant_id != tenant_id:
                raise HTTPException(status_code=404, detail="metadata field not found")
            await db.delete(field)
            await db.commit()
        return {"status": "deleted", "field_id": field_id}

    # ===== Document + Version Management =====
    @app.post("/api/knowledge-bases/{kb_id}/documents")
    async def upload_document(kb_id: str, file: UploadFile = File(...),
                             metadata: str | None = Form(default=None),
                             tenant_id: str = Depends(_tenant)):
        content = await file.read()
        file_type = file.filename.rsplit(".", 1)[-1] if file.filename and "." in file.filename else "txt"
        filename = file.filename or "unknown"

        # 解析上传的元数据 JSON（运营在上传时给文档打 category/effective_ts 等标签）
        raw_meta = {}
        if metadata:
            try:
                raw_meta = json.loads(metadata)
            except (json.JSONDecodeError, ValueError):
                raise HTTPException(status_code=400, detail="metadata 必须是合法 JSON 对象")
            if not isinstance(raw_meta, dict):
                raise HTTPException(status_code=400, detail="metadata 必须是 JSON 对象")

        async with db_factory() as db:
            kb = await _require_kb(db, kb_id, tenant_id)
            field_defs = await _kb_field_defs(db, kb_id)
            clean_meta = _validate_metadata(raw_meta, field_defs)

            # find existing logical document by filename, else create one
            doc = (await db.execute(
                select(DocumentModel).where(
                    DocumentModel.kb_id == kb_id, DocumentModel.filename == filename
                )
            )).scalar_one_or_none()
            is_new_doc = doc is None
            if is_new_doc:
                doc = DocumentModel(id=short_id(), tenant_id=tenant_id, kb_id=kb_id,
                                    filename=filename, doc_metadata=clean_meta)
                db.add(doc)
                await db.commit()
            elif clean_meta:
                # 重传带新元数据：合并更新（新值覆盖旧值），供下一版本下沉
                doc.doc_metadata = {**(doc.doc_metadata or {}), **clean_meta}
                await db.commit()

            ver = await app.state.vm.add_version(
                db, kb, doc, content, filename, file_type,
                SmartSplitter(chunk_size=kb.chunk_size, chunk_overlap=kb.chunk_overlap),
            )

            if is_new_doc:
                kb.document_count = (kb.document_count or 0) + 1
                await db.commit()

            await db.refresh(doc)
            return _doc_dict(doc, ver)

    @app.get("/api/knowledge-bases/{kb_id}/documents")
    async def list_documents(kb_id: str, tenant_id: str = Depends(_tenant)):
        async with db_factory() as db:
            await _require_kb(db, kb_id, tenant_id)
            docs = (await db.execute(
                select(DocumentModel).where(DocumentModel.kb_id == kb_id)
            )).scalars().all()
            out = []
            for doc in docs:
                ver = await db.get(DocumentVersionModel, doc.current_version_id) if doc.current_version_id else None
                out.append(_doc_dict(doc, ver))
            return out

    @app.get("/api/documents/{doc_id}")
    async def get_document(doc_id: str, tenant_id: str = Depends(_tenant)):
        async with db_factory() as db:
            doc = await _require_doc(db, doc_id, tenant_id)
            ver = await db.get(DocumentVersionModel, doc.current_version_id) if doc.current_version_id else None
            return _doc_dict(doc, ver)

    @app.get("/api/documents/{doc_id}/versions")
    async def list_versions(doc_id: str, tenant_id: str = Depends(_tenant)):
        async with db_factory() as db:
            doc = await _require_doc(db, doc_id, tenant_id)
            vers = (await db.execute(
                select(DocumentVersionModel).where(DocumentVersionModel.doc_id == doc.id)
                .order_by(DocumentVersionModel.version_no.desc())
            )).scalars().all()
            return [
                {
                    "id": v.id, "version_no": v.version_no, "status": v.status,
                    "file_hash": v.file_hash, "file_size": v.file_size,
                    "chunk_count": v.chunk_count, "created_by": v.created_by,
                    "is_current": v.id == doc.current_version_id,
                }
                for v in vers
            ]

    @app.post("/api/documents/{doc_id}/rollback")
    async def rollback_document(doc_id: str, target_version_no: int,
                               tenant_id: str = Depends(_tenant)):
        async with db_factory() as db:
            doc = await _require_doc(db, doc_id, tenant_id)
            try:
                result = await app.state.vm.rollback(db, doc, target_version_no)
            except ValueError as e:
                raise HTTPException(status_code=400, detail=str(e))
            return result

    @app.delete("/api/documents/{doc_id}")
    async def delete_document(doc_id: str, tenant_id: str = Depends(_tenant)):
        async with db_factory() as db:
            doc = await _require_doc(db, doc_id, tenant_id)
            kb_id = doc.kb_id
            vers = (await db.execute(
                select(DocumentVersionModel).where(DocumentVersionModel.doc_id == doc.id)
            )).scalars().all()
            for v in vers:
                await app.state.retriever.delete_by_version(kb_id, v.id)
            await app.state.store.delete_prefix(f"{tenant_id}/{kb_id}/{doc.id}/")
            kb = await db.get(KnowledgeBaseModel, kb_id)
            if kb:
                kb.document_count = max(0, (kb.document_count or 0) - 1)
            await db.delete(doc)
            await db.commit()
        return {"status": "deleted", "doc_id": doc_id}

    # ===== Search =====
    @app.post("/api/search", response_model=SearchResponse)
    async def search(request: SearchRequest, tenant_id: str = Depends(_tenant)):
        async with db_factory() as db:
            kb = await _require_kb(db, request.kb_id, tenant_id)
            visible = await visible_version_ids(db, request.kb_id)
            field_defs = await _kb_field_defs(db, request.kb_id)
        # temporal 库：自动注入有效期过滤（仅当库定义了对应时间字段时）
        request = _apply_kb_search_config(request, kb)
        _validate_top_k(request.top_k)
        request.filters = apply_temporal_filters(kb.kb_form, field_defs, request.filters)
        results = await app.state.retriever.search(
            request, visible_version_ids=visible, kb_mode=kb.retrieval_mode,
        )
        return SearchResponse(query=request.query, results=results, total=len(results))

    @app.post("/api/route-search", response_model=RouteSearchResponse)
    async def route_search(request: RouteSearchRequest, tenant_id: str = Depends(_tenant)):
        """聚合检索（Agent 主用）：跨库路由 + faq 短路 + 加权 RRF 融合。

        scope 限定参与库（kb_id 或 kb_form）；None=租户下全部库。
        """
        async with db_factory() as db:
            kbs = (await db.execute(
                select(KnowledgeBaseModel).where(KnowledgeBaseModel.tenant_id == tenant_id)
            )).scalars().all()
            # scope 过滤：命中 kb_id 或 kb_form 任一即参与
            if request.scope:
                scope = set(request.scope)
                kbs = [kb for kb in kbs if kb.id in scope or kb.kb_form in scope]
            plans = []
            for kb in kbs:
                visible = await visible_version_ids(db, kb.id)
                if not visible:
                    continue  # 空库（无已激活版本）不参与
                plans.append(KbPlan(
                    kb_id=kb.id, kb_form=kb.kb_form,
                    retrieval_mode=kb.retrieval_mode,
                    top_k=kb.top_k,
                    priority_weight=kb.priority_weight,
                    embedding_model=kb.embedding_model,
                    rerank_model=kb.rerank_model,
                    vector_weight=kb.vector_weight,
                    keyword_weight=kb.keyword_weight,
                    score_threshold=kb.score_threshold,
                    shortcut_threshold=kb.shortcut_threshold,
                    visible_version_ids=visible,
                    field_defs=await _kb_field_defs(db, kb.id),
                ))
        if request.top_k is not None:
            _validate_top_k(request.top_k)
        return await app.state.router.route(
            request.query, plans, top_k=request.top_k, base_filters=request.filters,
        )

    return app

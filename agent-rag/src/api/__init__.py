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


def _form_defaults(kb_form: str) -> dict:
    """知识形态 → 切片/检索默认配置（运营只选形态，不碰内部参数）。"""
    table = {
        "faq":        {"chunking_strategy": "qa_pair", "retrieval_mode": "hybrid",
                       "priority_weight": 1.0, "shortcut_threshold": 0.70},
        "standard":   {"chunking_strategy": "heading", "retrieval_mode": "hybrid",
                       "priority_weight": 0.7, "shortcut_threshold": 0.0},
        "temporal":   {"chunking_strategy": "auto", "retrieval_mode": "hybrid",
                       "priority_weight": 0.5, "shortcut_threshold": 0.0},
        "multimodal": {"chunking_strategy": "auto", "retrieval_mode": "vector",
                       "priority_weight": 0.3, "shortcut_threshold": 0.0},
    }
    return table.get(kb_form, table["standard"])


def _kb_dict(kb: KnowledgeBaseModel) -> dict:
    return {
        "id": kb.id, "tenant_id": kb.tenant_id, "name": kb.name,
        "description": kb.description, "chunking_strategy": kb.chunking_strategy,
        "chunk_size": kb.chunk_size, "chunk_overlap": kb.chunk_overlap,
        "document_count": kb.document_count,
        "kb_form": kb.kb_form, "retrieval_mode": kb.retrieval_mode,
        "priority_weight": kb.priority_weight,
        "shortcut_threshold": kb.shortcut_threshold,
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
                        kb_form: str = "standard", tenant_id: str = Depends(_tenant)):
        """建库：选知识形态(kb_form)即绑定默认切片/检索配置；strategy 可显式覆盖。"""
        if kb_form not in ("faq", "standard", "temporal", "multimodal"):
            raise HTTPException(status_code=400, detail=f"invalid kb_form: {kb_form}")
        defaults = _form_defaults(kb_form)
        chunking = ChunkingStrategy(strategy).value if strategy else defaults["chunking_strategy"]
        kb_id = short_id()
        async with db_factory() as db:
            kb = KnowledgeBaseModel(
                id=kb_id, tenant_id=tenant_id, name=name, description=description,
                chunking_strategy=chunking,
                kb_form=kb_form,
                collection_name=f"kb_{kb_id}",
                retrieval_mode=defaults["retrieval_mode"],
                priority_weight=defaults["priority_weight"],
                shortcut_threshold=defaults["shortcut_threshold"],
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
    async def update_kb(kb_id: str, shortcut_threshold: float | None = None,
                        tenant_id: str = Depends(_tenant)):
        """更新库的可调参数。目前仅开放 faq 库的 shortcut_threshold(高置信短路阈值)。

        阈值是把危险旋钮：过低则误短路答错，过高则永不短路。仅 faq 库可改、
        值域 [0,1]、改动写审计。换 embedding 模型后需据真实校准重设此值。
        """
        if shortcut_threshold is None:
            raise HTTPException(status_code=400, detail="无可更新字段")
        if not 0.0 <= shortcut_threshold <= 1.0:
            raise HTTPException(status_code=400, detail="shortcut_threshold 必须在 [0, 1] 之间")
        async with db_factory() as db:
            kb = await _require_kb(db, kb_id, tenant_id)
            if kb.kb_form != "faq":
                raise HTTPException(status_code=400,
                                    detail="shortcut_threshold 仅对 faq 库有效")
            old = kb.shortcut_threshold
            kb.shortcut_threshold = shortcut_threshold
            await _audit(db, tenant_id, "admin", "update_threshold", "knowledge_base", kb.id,
                         {"from": old, "to": shortcut_threshold})
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
                db, kb, doc, content, filename, file_type, app.state.splitter,
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
                    priority_weight=kb.priority_weight,
                    shortcut_threshold=kb.shortcut_threshold,
                    visible_version_ids=visible,
                    field_defs=await _kb_field_defs(db, kb.id),
                ))
        return await app.state.router.route(
            request.query, plans, top_k=request.top_k, base_filters=request.filters,
        )

    return app

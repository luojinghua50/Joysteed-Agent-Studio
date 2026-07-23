"""Version management: shadow build, atomic switch, rollback, retention.

A logical document (documents) has many physical versions (document_versions).
A new upload creates a shadow version that is invisible to search until it is
atomically activated by flipping documents.current_version_id. This gives
zero-downtime updates and instant rollback.
"""
import hashlib
import uuid
from datetime import datetime, UTC

import structlog
from sqlalchemy import select, func

from src.db import (
    KnowledgeBaseModel, DocumentModel, DocumentVersionModel, ChunkModel, AuditLogModel,
)

logger = structlog.get_logger()


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def short_id() -> str:
    return uuid.uuid4().hex[:8]


def version_id() -> str:
    return uuid.uuid4().hex[:24]


def _utcnow() -> datetime:
    """Naive UTC timestamp matching the TIMESTAMP WITHOUT TIME ZONE columns.

    asyncpg rejects tz-aware values for naive columns, so strip tzinfo to stay
    consistent with the server_default=func.now() columns on the same tables.
    """
    return datetime.now(UTC).replace(tzinfo=None)


async def visible_version_ids(db, kb_id: str) -> list[str]:
    """current_version_id of every document in the kb — the search filter set."""
    stmt = select(DocumentModel.current_version_id).where(
        DocumentModel.kb_id == kb_id,
        DocumentModel.current_version_id.is_not(None),
    )
    rows = (await db.execute(stmt)).scalars().all()
    return [r for r in rows if r]


async def _audit(db, tenant_id: str, actor: str, action: str, target_type: str,
                 target_id: str, detail: dict):
    db.add(AuditLogModel(
        tenant_id=tenant_id, actor=actor, action=action,
        target_type=target_type, target_id=target_id, detail=detail,
    ))


class VersionManager:
    """Coordinates PG metadata, object store, and the retrieval index."""

    def __init__(self, settings, retriever, object_store):
        self.settings = settings
        self.retriever = retriever
        self.store = object_store

    async def add_version(self, db, kb: KnowledgeBaseModel, doc: DocumentModel,
                          content: bytes, filename: str, file_type: str,
                          splitter, actor: str = "system") -> DocumentVersionModel:
        """Create + process + activate a new version for an existing document."""
        file_hash = sha256(content)

        # Idempotency: identical content as current version → no-op
        if doc.current_version_id:
            cur = await db.get(DocumentVersionModel, doc.current_version_id)
            if cur and cur.file_hash == file_hash:
                logger.info("version_unchanged_skip", doc_id=doc.id)
                return cur

        # next version_no
        max_no = (await db.execute(
            select(func.max(DocumentVersionModel.version_no)).where(DocumentVersionModel.doc_id == doc.id)
        )).scalar() or 0
        ver_no = max_no + 1
        vid = version_id()

        minio_key = self.store.build_key(doc.tenant_id, kb.id, doc.id, ver_no, filename)
        await self.store.put(minio_key, content, content_type="application/octet-stream")

        # shadow version row — invisible to search (current pointer not moved)
        ver = DocumentVersionModel(
            id=vid, tenant_id=doc.tenant_id, doc_id=doc.id, kb_id=kb.id,
            version_no=ver_no, file_hash=file_hash, file_type=file_type,
            file_size=len(content), minio_key=minio_key, status="processing",
            heartbeat_at=_utcnow(),
        )
        db.add(ver)
        await db.commit()

        # build: chunk → persist chunks → index
        try:
            text = content.decode("utf-8", errors="ignore")
            chunks = splitter.split(text, file_type, kb.chunking_strategy)
            # 文档级元数据（category/effective_ts...）下沉到每个 chunk，供库内过滤
            doc_meta = dict(doc.doc_metadata or {})
            chunk_dicts = []
            for i, ch in enumerate(chunks):
                cid = f"{vid}-{i:04d}"
                # chunk 自身 metadata 优先，文档级补位
                merged_meta = {**doc_meta, **(ch.metadata or {})}
                db.add(ChunkModel(
                    id=cid, tenant_id=doc.tenant_id, version_id=vid, doc_id=doc.id,
                    kb_id=kb.id, chunk_index=i, text=ch.text,
                    chunk_hash=sha256(ch.text.encode("utf-8")),
                    context_header=ch.context_header, keywords=ch.keywords,
                    token_count=ch.token_count, meta=merged_meta,
                ))
                chunk_dicts.append({
                    "id": cid, "version_id": vid, "doc_id": doc.id, "kb_id": kb.id,
                    "text": ch.text, "keywords": ch.keywords,
                    "context_header": ch.context_header, "metadata": merged_meta,
                })
            await self.retriever.index_chunks(kb.id, chunk_dicts)
            ver.chunk_count = len(chunks)
            ver.status = "ready"
            ver.completed_at = _utcnow()
            await db.commit()
        except Exception as e:
            ver.status = "failed"
            ver.error = str(e)
            await db.commit()
            logger.error("version_build_failed", doc_id=doc.id, version_id=vid, error=str(e))
            raise

        await self.activate(db, doc, vid, actor=actor)
        return ver

    async def activate(self, db, doc: DocumentModel, new_version_id: str, actor: str = "system"):
        """Atomically point the document at a ready version."""
        new_ver = await db.get(DocumentVersionModel, new_version_id)
        if not new_ver or new_ver.status not in ("ready", "archived", "active"):
            raise ValueError(f"version {new_version_id} not activatable")

        old_version_id = doc.current_version_id
        doc.current_version_id = new_version_id
        new_ver.status = "active"
        if old_version_id and old_version_id != new_version_id:
            old = await db.get(DocumentVersionModel, old_version_id)
            if old:
                old.status = "archived"
        await _audit(db, doc.tenant_id, actor, "activate", "document", doc.id,
                     {"from": old_version_id, "to": new_version_id})
        await db.commit()
        logger.info("version_activated", doc_id=doc.id, version_id=new_version_id, prev=old_version_id)
        return {"doc_id": doc.id, "active_version": new_version_id, "previous": old_version_id}

    async def rollback(self, db, doc: DocumentModel, target_version_no: int, actor: str = "system"):
        """Roll back to a historical version by version_no (must not be GC'd)."""
        ver = (await db.execute(
            select(DocumentVersionModel).where(
                DocumentVersionModel.doc_id == doc.id,
                DocumentVersionModel.version_no == target_version_no,
            )
        )).scalar_one_or_none()
        if not ver:
            raise ValueError(f"version {target_version_no} not found")
        if ver.status not in ("ready", "archived", "active"):
            raise ValueError(f"version {target_version_no} not restorable (status={ver.status})")
        result = await self.activate(db, doc, ver.id, actor=actor)
        await _audit(db, doc.tenant_id, actor, "rollback", "document", doc.id,
                     {"to_version_no": target_version_no})
        await db.commit()
        return result

    async def apply_retention(self, db, doc: DocumentModel, actor: str = "system") -> int:
        """GC: keep current + last N versions + recent ones; purge the rest."""
        versions = (await db.execute(
            select(DocumentVersionModel).where(DocumentVersionModel.doc_id == doc.id)
            .order_by(DocumentVersionModel.version_no.desc())
        )).scalars().all()

        keep_n = self.settings.keep_last_n_versions
        purged = 0
        for idx, ver in enumerate(versions):
            is_current = ver.id == doc.current_version_id
            within_keep = idx < keep_n
            if is_current or within_keep or ver.status == "purged":
                continue
            # purge chunks + index + object
            await self.retriever.delete_by_version(doc.kb_id, ver.id)
            await db.execute(
                ChunkModel.__table__.delete().where(ChunkModel.version_id == ver.id)
            )
            await self.store.delete_prefix(ver.minio_key)
            ver.status = "purged"
            purged += 1
        if purged:
            await _audit(db, doc.tenant_id, actor, "gc", "document", doc.id, {"purged": purged})
        await db.commit()
        return purged

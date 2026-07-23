"""Database layer: SQLAlchemy async models, engine init, and repositories.

Uses portable column types (String/Text/JSON/DateTime) so the same models run
on PostgreSQL (production) and SQLite (tests).
"""
from datetime import datetime

from sqlalchemy import String, Text, DateTime, Integer, BigInteger, Float, JSON, ForeignKey, func, Index, UniqueConstraint
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker, AsyncSession
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    pass


class KnowledgeBaseModel(Base):
    __tablename__ = "knowledge_bases"

    id: Mapped[str] = mapped_column(String(16), primary_key=True)
    tenant_id: Mapped[str] = mapped_column(String(32), index=True, default="default")
    name: Mapped[str] = mapped_column(String(128))
    description: Mapped[str] = mapped_column(Text, default="")
    chunking_strategy: Mapped[str] = mapped_column(String(32), default="auto")
    chunk_size: Mapped[int] = mapped_column(Integer, default=512)
    chunk_overlap: Mapped[int] = mapped_column(Integer, default=50)
    embedding_model: Mapped[str] = mapped_column(String(64), default="text-embedding-3-small")
    document_count: Mapped[int] = mapped_column(Integer, default=0)
    # 知识形态与检索配置（多 collection 方案）
    kb_form: Mapped[str] = mapped_column(String(24), default="standard")  # faq|standard|temporal|multimodal
    collection_name: Mapped[str] = mapped_column(String(64), default="")  # 物理 Milvus collection 名
    retrieval_mode: Mapped[str] = mapped_column(String(16), default="hybrid")  # vector|fulltext|hybrid
    priority_weight: Mapped[float] = mapped_column(Float, default=0.7)   # 跨库融合的库级权重
    vector_weight: Mapped[float] = mapped_column(Float, default=0.6)     # 库内 hybrid 向量配比
    keyword_weight: Mapped[float] = mapped_column(Float, default=0.4)    # 库内 hybrid 关键词配比
    score_threshold: Mapped[float] = mapped_column(Float, default=0.0)
    shortcut_threshold: Mapped[float] = mapped_column(Float, default=0.0)  # 仅 faq：高置信短路阈值
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now(), onupdate=func.now())


class DocumentModel(Base):
    __tablename__ = "documents"

    id: Mapped[str] = mapped_column(String(16), primary_key=True)
    tenant_id: Mapped[str] = mapped_column(String(32), default="default")
    kb_id: Mapped[str] = mapped_column(String(16), ForeignKey("knowledge_bases.id", ondelete="CASCADE"), index=True)
    filename: Mapped[str] = mapped_column(String(512))
    current_version_id: Mapped[str | None] = mapped_column(String(24), nullable=True)
    doc_metadata: Mapped[dict] = mapped_column(JSON, default=dict)  # 元数据字段值 {category:"耳机",...}
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now(), onupdate=func.now())

    __table_args__ = (Index("idx_documents_tenant", "tenant_id", "kb_id"),)


class DocumentVersionModel(Base):
    __tablename__ = "document_versions"

    id: Mapped[str] = mapped_column(String(24), primary_key=True)
    tenant_id: Mapped[str] = mapped_column(String(32), default="default")
    doc_id: Mapped[str] = mapped_column(String(16), ForeignKey("documents.id", ondelete="CASCADE"), index=True)
    kb_id: Mapped[str] = mapped_column(String(16), index=True)
    version_no: Mapped[int] = mapped_column(Integer)
    file_hash: Mapped[str] = mapped_column(String(64))
    file_type: Mapped[str] = mapped_column(String(16))
    file_size: Mapped[int] = mapped_column(BigInteger, default=0)
    minio_key: Mapped[str] = mapped_column(String(1024), default="")
    status: Mapped[str] = mapped_column(String(16), default="pending", index=True)
    heartbeat_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    chunk_count: Mapped[int] = mapped_column(Integer, default=0)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_by: Mapped[str] = mapped_column(String(64), default="system")
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())
    completed_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)

    __table_args__ = (Index("idx_versions_doc", "tenant_id", "doc_id", "version_no"),)


class ChunkModel(Base):
    __tablename__ = "chunks"

    id: Mapped[str] = mapped_column(String(32), primary_key=True)
    tenant_id: Mapped[str] = mapped_column(String(32), default="default")
    version_id: Mapped[str] = mapped_column(String(24), ForeignKey("document_versions.id", ondelete="CASCADE"), index=True)
    doc_id: Mapped[str] = mapped_column(String(16), index=True)
    kb_id: Mapped[str] = mapped_column(String(16), index=True)
    chunk_index: Mapped[int] = mapped_column(Integer, default=0)
    text: Mapped[str] = mapped_column(Text)
    chunk_hash: Mapped[str] = mapped_column(String(64))
    context_header: Mapped[str] = mapped_column(String(512), default="")
    keywords: Mapped[list] = mapped_column(JSON, default=list)
    token_count: Mapped[int] = mapped_column(Integer, default=0)
    meta: Mapped[dict] = mapped_column(JSON, default=dict)


class KbMetadataFieldModel(Base):
    """知识库可自定义的元数据字段定义（库内过滤用）。"""
    __tablename__ = "kb_metadata_fields"

    id: Mapped[str] = mapped_column(String(16), primary_key=True)
    tenant_id: Mapped[str] = mapped_column(String(32), index=True, default="default")
    kb_id: Mapped[str] = mapped_column(String(16), ForeignKey("knowledge_bases.id", ondelete="CASCADE"), index=True)
    name: Mapped[str] = mapped_column(String(64))         # 字段名，如 category / effective_ts
    field_type: Mapped[str] = mapped_column(String(16))   # string | number | time
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())

    __table_args__ = (UniqueConstraint("kb_id", "name", name="uq_kb_field_name"),)


class AuditLogModel(Base):
    __tablename__ = "audit_log"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    tenant_id: Mapped[str] = mapped_column(String(32), index=True, default="default")
    actor: Mapped[str] = mapped_column(String(64), default="system")
    action: Mapped[str] = mapped_column(String(32))
    target_type: Mapped[str] = mapped_column(String(16))
    target_id: Mapped[str] = mapped_column(String(24))
    detail: Mapped[dict] = mapped_column(JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())


async def init_db(database_url: str) -> async_sessionmaker[AsyncSession]:
    """Create engine, ensure tables exist, return session factory."""
    engine = create_async_engine(database_url, echo=False)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    return async_sessionmaker(engine, expire_on_commit=False)

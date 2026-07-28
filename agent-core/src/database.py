"""Database layer: SQLAlchemy async models and engine initialization."""
from datetime import datetime

from sqlalchemy import String, Text, DateTime, ForeignKey, Integer, Float, JSON, UniqueConstraint, func
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker, AsyncSession
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    pass


class UserModel(Base):
    """A user account. ``id`` is the ``customer_id`` carried through sessions,
    memory and (future) RAG tenancy, so identity semantics stay stable as the
    auth story evolves from self-reported -> token-derived."""
    __tablename__ = "users"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    username: Mapped[str] = mapped_column(String(128), unique=True, index=True)
    password_hash: Mapped[str | None] = mapped_column(String(255), nullable=True)
    display_name: Mapped[str | None] = mapped_column(String(128), nullable=True)
    roles: Mapped[str] = mapped_column(String(255), default="customer")
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())


class SessionModel(Base):
    __tablename__ = "sessions"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    customer_id: Mapped[str] = mapped_column(String(64), index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now(), onupdate=func.now())

    # 短期记忆滚动摘要：summary 存已压缩的老消息摘要，summary_upto_id 记摘要已覆盖到的
    # 最大 message id（0=无摘要）。加载时只取 id>summary_upto_id 的原文 + 该摘要，无 gap。
    # TODO(生产): init_db 的 create_all 只建不改列，已存在的 sessions 表需 migration 补这两列。
    summary: Mapped[str | None] = mapped_column(Text, nullable=True)
    summary_upto_id: Mapped[int] = mapped_column(Integer, default=0)

    messages: Mapped[list["MessageModel"]] = relationship(back_populates="session", order_by="MessageModel.timestamp")


class MessageModel(Base):
    __tablename__ = "messages"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    session_id: Mapped[str] = mapped_column(String(64), ForeignKey("sessions.id"), index=True)
    role: Mapped[str] = mapped_column(String(16))
    content: Mapped[str] = mapped_column(Text)
    agent: Mapped[str | None] = mapped_column(String(64), nullable=True)
    timestamp: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())

    session: Mapped["SessionModel"] = relationship(back_populates="messages")


class ErrorRecordModel(Base):
    """A judge-rejected agent response, persisted so the reflection loop learns
    across process restarts (the in-memory ``ErrorMemoryStore`` loses these on
    restart). Injected back into high-risk agents' prompts as 历史错误/禁止事项.

    ``issues`` is stored as a newline-joined string (SQLite/Postgres portable);
    the store splits it back into a list on read."""
    __tablename__ = "error_records"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    agent: Mapped[str] = mapped_column(String(64), index=True)
    skill: Mapped[str | None] = mapped_column(String(64), nullable=True)
    user_message: Mapped[str] = mapped_column(Text)
    failed_response: Mapped[str] = mapped_column(Text)
    issues: Mapped[str] = mapped_column(Text, default="")
    suggestion: Mapped[str] = mapped_column(Text, default="")
    retry_count: Mapped[int] = mapped_column(Integer, default=0)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now(), index=True)


class ProfileModel(Base):
    """长期记忆-用户画像：跨会话累积的用户特征（VIP/沟通风格/敏感点等）。

    以 customer_id 为主键（一人一档）。list/dict 字段用 JSON 列存，SQLite/PG 通用。
    从会话结束时 LLM 推断 + CRM 冷启动增量更新。"""
    __tablename__ = "profiles"

    customer_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    vip_level: Mapped[int] = mapped_column(Integer, default=0)
    preferred_channel: Mapped[str | None] = mapped_column(String(32), nullable=True)
    communication_style: Mapped[str | None] = mapped_column(String(32), nullable=True)
    sensitive_points: Mapped[list] = mapped_column(JSON, default=list)
    frequent_categories: Mapped[list] = mapped_column(JSON, default=list)
    tags: Mapped[dict] = mapped_column(JSON, default=dict)
    last_updated: Mapped[datetime] = mapped_column(DateTime, server_default=func.now(), onupdate=func.now())


class FactModel(Base):
    """长期记忆-事实知识：用户相关的持久事实（常用地址/偏好等），KV 结构。

    带 source/confidence/updated_at：读取时按 updated_at 距今天数做置信度衰减
    （ConfidenceManager），低置信度在 prompt 里标注"待核实"。(customer_id,key) 唯一。"""
    __tablename__ = "facts"
    __table_args__ = (UniqueConstraint("customer_id", "key", name="uq_fact_customer_key"),)

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    customer_id: Mapped[str] = mapped_column(String(64), index=True)
    key: Mapped[str] = mapped_column(String(64))
    value: Mapped[str] = mapped_column(Text)
    source: Mapped[str] = mapped_column(String(32), default="agent_inferred")
    confidence: Mapped[float] = mapped_column(Float, default=0.6)
    updated_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now(), onupdate=func.now())


class EpisodeModel(Base):
    """长期记忆-历史交互摘要：每次会话结束的 LLM 摘要（结构化真相源）。

    向量化后另存 Milvus 供语义检索；本表按 id 精确载全量、审计。created_at 供时间衰减。"""
    __tablename__ = "episodes"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    session_id: Mapped[str] = mapped_column(String(64), index=True)
    customer_id: Mapped[str] = mapped_column(String(64), index=True)
    summary: Mapped[str] = mapped_column(Text)
    intent: Mapped[str] = mapped_column(String(64), default="general")
    resolution: Mapped[str] = mapped_column(String(32), default="resolved")
    key_entities: Mapped[dict] = mapped_column(JSON, default=dict)
    satisfaction: Mapped[float | None] = mapped_column(Float, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now(), index=True)


async def init_db(database_url: str) -> async_sessionmaker[AsyncSession]:
    """Create engine, ensure tables exist, return session factory."""
    engine = create_async_engine(database_url, echo=False)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    return async_sessionmaker(engine, expire_on_commit=False)

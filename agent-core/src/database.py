"""Database layer: SQLAlchemy async models and engine initialization."""
from datetime import datetime

from sqlalchemy import String, Text, DateTime, ForeignKey, Integer, func
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


async def init_db(database_url: str) -> async_sessionmaker[AsyncSession]:
    """Create engine, ensure tables exist, return session factory."""
    engine = create_async_engine(database_url, echo=False)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    return async_sessionmaker(engine, expire_on_commit=False)

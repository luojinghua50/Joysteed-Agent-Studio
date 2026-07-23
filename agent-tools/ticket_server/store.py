"""Ticket persistence layer — shared by MCP tools and the REST API.

Tickets used to live in an in-process dict (lost on restart, no list query).
This module backs them with PostgreSQL via SQLAlchemy async, so the agent-desk
workbench and the LLM tools operate on the same durable store.
"""
from __future__ import annotations  # lazy annotations: the `list` method below shadows builtin list

from datetime import datetime, UTC

from sqlalchemy import String, Text, DateTime, Integer, ForeignKey, func, select
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker, AsyncSession
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

VALID_STATUSES = ["open", "in_progress", "resolved", "closed"]
VALID_PRIORITIES = ["low", "medium", "high"]
# Simulated seat pool. Assignment picks the least-loaded seat (see pick_agent).
AGENT_POOL = [f"agent-{i:03d}" for i in range(1, 6)]


class Base(DeclarativeBase):
    pass


def _utcnow() -> datetime:
    """Naive UTC to match TIMESTAMP WITHOUT TIME ZONE columns (asyncpg rejects tz-aware)."""
    return datetime.now(UTC).replace(tzinfo=None)


class TicketModel(Base):
    __tablename__ = "tickets"

    ticket_id: Mapped[str] = mapped_column(String(16), primary_key=True)
    customer_id: Mapped[str] = mapped_column(String(32), index=True)
    title: Mapped[str] = mapped_column(String(256))
    description: Mapped[str] = mapped_column(Text, default="")
    status: Mapped[str] = mapped_column(String(16), default="open", index=True)
    priority: Mapped[str] = mapped_column(String(16), default="medium", index=True)
    assigned_to: Mapped[str] = mapped_column(String(32), default="", index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now(), onupdate=func.now())


class TicketCommentModel(Base):
    __tablename__ = "ticket_comments"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    ticket_id: Mapped[str] = mapped_column(String(16), ForeignKey("tickets.ticket_id", ondelete="CASCADE"), index=True)
    author: Mapped[str] = mapped_column(String(64))
    comment: Mapped[str] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())


class TicketStore:
    """Async ticket store. One instance per process, initialized at startup."""

    def __init__(self, database_url: str):
        # connect_args.timeout 给 asyncpg 建连设上限：DB 不可达时秒级抛错而非
        # 无限挂起（连接黑洞会拖垮工具调用、也曾让本机 pytest 卡到超时）。
        self._engine = create_async_engine(
            database_url, echo=False, connect_args={"timeout": 5},
        )
        self._session = async_sessionmaker(self._engine, expire_on_commit=False)
        self._initialized = False

    async def init(self) -> None:
        async with self._engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        await self._seed()

    async def ensure_init(self) -> None:
        """Idempotent lazy init — FastMCP manages its own lifespan, so we can't
        hook a startup event cleanly; init on first tool/route call instead."""
        if self._initialized:
            return
        await self.init()
        self._initialized = True

    async def _seed(self) -> None:
        """Seed the two demo tickets once, so the workbench isn't empty on first run."""
        async with self._session() as db:
            existing = (await db.execute(select(func.count()).select_from(TicketModel))).scalar()
            if existing:
                return
            db.add_all([
                TicketModel(
                    ticket_id="TK-001", customer_id="C001", title="订单配送延迟投诉",
                    description="订单ORD-001已超过预计配送时间3天未到", status="open",
                    priority="high", assigned_to="agent-003",
                ),
                TicketModel(
                    ticket_id="TK-002", customer_id="C002", title="商品质量问题",
                    description="收到的商品有划痕，要求换货", status="resolved",
                    priority="medium", assigned_to="agent-001",
                ),
            ])
            await db.commit()

    async def _next_id(self, db: AsyncSession) -> str:
        count = (await db.execute(select(func.count()).select_from(TicketModel))).scalar() or 0
        return f"TK-{count + 1:03d}"

    async def pick_agent(self, db: AsyncSession) -> str:
        """Least-loaded seat by open/in_progress ticket count (replaces random assignment)."""
        loads: dict[str, int] = {a: 0 for a in AGENT_POOL}
        rows = (await db.execute(
            select(TicketModel.assigned_to, func.count())
            .where(TicketModel.status.in_(["open", "in_progress"]))
            .group_by(TicketModel.assigned_to)
        )).all()
        for agent, cnt in rows:
            if agent in loads:
                loads[agent] = cnt
        return min(loads, key=loads.get)

    @staticmethod
    def to_dict(t: TicketModel) -> dict:
        return {
            "ticket_id": t.ticket_id, "customer_id": t.customer_id, "title": t.title,
            "description": t.description, "status": t.status, "priority": t.priority,
            "assigned_to": t.assigned_to,
            "created_at": t.created_at.strftime("%Y-%m-%d %H:%M:%S") if t.created_at else "",
            "updated_at": t.updated_at.strftime("%Y-%m-%d %H:%M:%S") if t.updated_at else "",
        }

    async def create(self, customer_id: str, title: str, description: str,
                     priority: str = "medium", assigned_to: str | None = None) -> dict:
        await self.ensure_init()
        async with self._session() as db:
            ticket_id = await self._next_id(db)
            agent = assigned_to or await self.pick_agent(db)
            t = TicketModel(
                ticket_id=ticket_id, customer_id=customer_id, title=title,
                description=description, status="open", priority=priority, assigned_to=agent,
            )
            db.add(t)
            await db.commit()
            await db.refresh(t)
            return self.to_dict(t)

    async def get(self, ticket_id: str) -> dict | None:
        await self.ensure_init()
        async with self._session() as db:
            t = await db.get(TicketModel, ticket_id)
            return self.to_dict(t) if t else None

    async def list(self, status: str | None = None, assigned_to: str | None = None,
                   priority: str | None = None) -> list[dict]:
        await self.ensure_init()
        async with self._session() as db:
            stmt = select(TicketModel)
            if status:
                stmt = stmt.where(TicketModel.status == status)
            if assigned_to:
                stmt = stmt.where(TicketModel.assigned_to == assigned_to)
            if priority:
                stmt = stmt.where(TicketModel.priority == priority)
            stmt = stmt.order_by(TicketModel.created_at.desc())
            rows = (await db.execute(stmt)).scalars().all()
            return [self.to_dict(t) for t in rows]

    async def update(self, ticket_id: str, status: str | None = None) -> dict | None:
        await self.ensure_init()
        async with self._session() as db:
            t = await db.get(TicketModel, ticket_id)
            if not t:
                return None
            if status:
                t.status = status
            t.updated_at = _utcnow()
            await db.commit()
            await db.refresh(t)
            return self.to_dict(t)

    async def reassign(self, ticket_id: str, new_agent: str) -> dict | None:
        await self.ensure_init()
        async with self._session() as db:
            t = await db.get(TicketModel, ticket_id)
            if not t:
                return None
            t.assigned_to = new_agent
            t.updated_at = _utcnow()
            await db.commit()
            await db.refresh(t)
            return self.to_dict(t)

    async def add_comment(self, ticket_id: str, author: str, comment: str) -> dict | None:
        await self.ensure_init()
        async with self._session() as db:
            t = await db.get(TicketModel, ticket_id)
            if not t:
                return None
            c = TicketCommentModel(ticket_id=ticket_id, author=author, comment=comment)
            db.add(c)
            t.updated_at = _utcnow()
            await db.commit()
            await db.refresh(c)
            return {"id": c.id, "ticket_id": ticket_id, "author": author,
                    "comment": comment,
                    "created_at": c.created_at.strftime("%Y-%m-%d %H:%M:%S") if c.created_at else ""}

    async def list_comments(self, ticket_id: str) -> list[dict]:
        await self.ensure_init()
        async with self._session() as db:
            rows = (await db.execute(
                select(TicketCommentModel).where(TicketCommentModel.ticket_id == ticket_id)
                .order_by(TicketCommentModel.created_at.asc())
            )).scalars().all()
            return [{"id": c.id, "ticket_id": c.ticket_id, "author": c.author,
                     "comment": c.comment,
                     "created_at": c.created_at.strftime("%Y-%m-%d %H:%M:%S") if c.created_at else ""}
                    for c in rows]

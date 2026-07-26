from datetime import datetime
from pydantic import BaseModel, Field
from collections import Counter


class ErrorRecord(BaseModel):
    """A single error record from reflection."""

    timestamp: datetime = Field(default_factory=datetime.now)
    agent: str
    skill: str | None = None
    user_message: str
    failed_response: str
    issues: list[str]
    suggestion: str
    retry_count: int = 0


class ErrorMemoryStore:
    """Stores error records for learning from past mistakes."""

    def __init__(self, max_size: int = 20):
        self.max_size = max_size
        self._store: dict[str, list[ErrorRecord]] = {}

    async def add_error(self, record: ErrorRecord):
        """Add an error record."""
        agent = record.agent
        if agent not in self._store:
            self._store[agent] = []
        self._store[agent].append(record)
        if len(self._store[agent]) > self.max_size:
            self._store[agent] = self._store[agent][-self.max_size:]

    async def get_error_context(self, agent: str) -> str:
        """Get historical error summary for prompt injection."""
        records = self._store.get(agent, [])[-5:]
        if not records:
            return ""
        context = "## 历史错误记录（避免重复）\n"
        for r in records:
            if r.issues:
                context += f"- 问题：{r.issues[0]} | 修正建议：{r.suggestion}\n"
        return context

    async def get_forbidden_patterns(self, agent: str) -> list[str]:
        """Extract high-frequency error patterns as forbidden rules."""
        records = self._store.get(agent, [])[-20:]
        all_issues = [issue for r in records for issue in r.issues]
        common = Counter(all_issues).most_common(5)
        return [issue for issue, count in common if count >= 2]

    async def clear(self, agent: str | None = None):
        """Clear error records."""
        if agent:
            self._store.pop(agent, None)
        else:
            self._store.clear()


class SqlErrorMemoryStore:
    """SQLAlchemy-backed error memory: same async interface as ``ErrorMemoryStore``
    but persists across process restarts (the in-memory store loses records on
    restart). Backed by ``ErrorRecordModel``; ``issues`` is stored newline-joined
    and split back into a list on read.

    Drop-in for ``ErrorMemoryStore``: the reflection loop depends only on
    ``add_error`` / ``get_error_context`` / ``get_forbidden_patterns`` / ``clear``.
    """

    def __init__(self, session_factory, max_size: int = 20):
        self.session_factory = session_factory
        self.max_size = max_size

    async def add_error(self, record: ErrorRecord):
        """Persist an error record."""
        from src.database import ErrorRecordModel

        async with self.session_factory() as db:
            db.add(ErrorRecordModel(
                agent=record.agent,
                skill=record.skill,
                user_message=record.user_message,
                failed_response=record.failed_response,
                issues="\n".join(record.issues),
                suggestion=record.suggestion,
                retry_count=record.retry_count,
            ))
            await db.commit()

    async def _recent(self, agent: str, limit: int) -> list[ErrorRecord]:
        """Load the most recent ``limit`` records for an agent (newest first)."""
        from sqlalchemy import select
        from src.database import ErrorRecordModel

        async with self.session_factory() as db:
            stmt = (
                select(ErrorRecordModel)
                .where(ErrorRecordModel.agent == agent)
                .order_by(ErrorRecordModel.created_at.desc(), ErrorRecordModel.id.desc())
                .limit(limit)
            )
            rows = (await db.execute(stmt)).scalars().all()
        return [
            ErrorRecord(
                timestamp=r.created_at or datetime.now(),
                agent=r.agent,
                skill=r.skill,
                user_message=r.user_message,
                failed_response=r.failed_response,
                issues=r.issues.split("\n") if r.issues else [],
                suggestion=r.suggestion,
                retry_count=r.retry_count,
            )
            for r in rows
        ]

    async def get_error_context(self, agent: str) -> str:
        """Get historical error summary for prompt injection (latest 5)."""
        records = list(reversed(await self._recent(agent, 5)))
        if not records:
            return ""
        context = "## 历史错误记录（避免重复）\n"
        for r in records:
            if r.issues:
                context += f"- 问题：{r.issues[0]} | 修正建议：{r.suggestion}\n"
        return context

    async def get_forbidden_patterns(self, agent: str) -> list[str]:
        """Extract high-frequency error patterns as forbidden rules (latest 20)."""
        records = await self._recent(agent, 20)
        all_issues = [issue for r in records for issue in r.issues]
        common = Counter(all_issues).most_common(5)
        return [issue for issue, count in common if count >= 2]

    async def clear(self, agent: str | None = None):
        """Delete error records (all, or for one agent)."""
        from sqlalchemy import delete
        from src.database import ErrorRecordModel

        async with self.session_factory() as db:
            stmt = delete(ErrorRecordModel)
            if agent:
                stmt = stmt.where(ErrorRecordModel.agent == agent)
            await db.execute(stmt)
            await db.commit()

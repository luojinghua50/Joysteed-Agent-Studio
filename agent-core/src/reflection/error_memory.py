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

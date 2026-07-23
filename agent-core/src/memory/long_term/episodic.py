from pydantic import BaseModel, Field
from datetime import datetime


class EpisodeRecord(BaseModel):
    """A single episodic memory entry."""

    session_id: str
    customer_id: str
    timestamp: datetime = Field(default_factory=datetime.now)
    summary: str
    intent: str
    resolution: str = "resolved"
    key_entities: dict = Field(default_factory=dict)
    satisfaction: float | None = None


class EpisodicMemory:
    """Episodic memory: stores conversation summaries for semantic retrieval."""

    def __init__(self):
        self._store: dict[str, list[EpisodeRecord]] = {}

    async def search(
        self, query: str, customer_id: str, top_k: int = 3
    ) -> list[EpisodeRecord]:
        """Search relevant historical interactions (mock: return most recent)."""
        episodes = self._store.get(customer_id, [])
        return episodes[-top_k:]

    async def save_episode(
        self, session_id: str, customer_id: str, messages: list
    ):
        """Save a conversation episode."""
        summary = self._generate_simple_summary(messages)
        record = EpisodeRecord(
            session_id=session_id,
            customer_id=customer_id,
            summary=summary,
            intent="general",
            resolution="resolved",
        )
        if customer_id not in self._store:
            self._store[customer_id] = []
        self._store[customer_id].append(record)

    async def delete_all(self, customer_id: str):
        """Delete all episodes for a customer."""
        self._store.pop(customer_id, None)

    async def count(self) -> int:
        return sum(len(v) for v in self._store.values())

    def _generate_simple_summary(self, messages: list) -> str:
        """Generate a simple summary from messages (mock implementation)."""
        if not messages:
            return "空会话"
        msg_texts = []
        for m in messages[:5]:
            content = getattr(m, "content", str(m))
            msg_texts.append(content[:50])
        return "会话摘要: " + " | ".join(msg_texts)

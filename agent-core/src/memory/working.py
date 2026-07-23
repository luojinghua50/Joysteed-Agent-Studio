import json
from typing import Any


class WorkingMemory:
    """Working memory: stores session-level entities in Redis (or in-memory fallback)."""

    def __init__(self, redis_client=None, ttl: int = 3600):
        self.redis = redis_client
        self.ttl = ttl
        self._local_store: dict[str, dict[str, Any]] = {}

    async def get(self, session_id: str) -> dict:
        """Get all entities for a session."""
        if self.redis:
            data = await self.redis.hgetall(f"working:{session_id}")
            return {k: json.loads(v) for k, v in data.items()} if data else {}
        return self._local_store.get(session_id, {})

    async def set_entity(self, session_id: str, key: str, value: Any):
        """Set an entity in working memory."""
        if self.redis:
            await self.redis.hset(f"working:{session_id}", key, json.dumps(value))
            await self.redis.expire(f"working:{session_id}", self.ttl)
        else:
            if session_id not in self._local_store:
                self._local_store[session_id] = {}
            self._local_store[session_id][key] = value

    async def get_entity(self, session_id: str, key: str) -> Any | None:
        """Get a specific entity."""
        if self.redis:
            val = await self.redis.hget(f"working:{session_id}", key)
            return json.loads(val) if val else None
        return self._local_store.get(session_id, {}).get(key)

    async def clear(self, session_id: str):
        """Clear all entities for a session."""
        if self.redis:
            await self.redis.delete(f"working:{session_id}")
        else:
            self._local_store.pop(session_id, None)

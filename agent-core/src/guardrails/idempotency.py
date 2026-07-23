import hashlib
import json
from datetime import timedelta

from src.config import StabilityConfig


class IdempotencyGuard:
    """Prevents duplicate execution of side-effect operations."""

    PROTECTED_TOOLS = {"apply_refund", "cancel_order", "create_ticket", "modify_order"}

    def __init__(self, redis_client=None, config: StabilityConfig | None = None):
        self.redis = redis_client
        self.config = config or StabilityConfig()
        self.ttl = timedelta(seconds=self.config.idempotency_ttl_seconds)
        self._local_cache: dict[str, dict] = {}

    def _make_key(self, session_id: str, tool_name: str, args: dict) -> str:
        content = f"{session_id}:{tool_name}:{json.dumps(args, sort_keys=True)}"
        return f"idempotency:{hashlib.sha256(content.encode()).hexdigest()}"

    async def check_and_mark(
        self, session_id: str, tool_name: str, args: dict
    ) -> tuple[bool, dict | None]:
        """Check if this is a duplicate call. Returns (is_duplicate, cached_result)."""
        if not self.config.idempotency_enabled:
            return False, None

        if tool_name not in self.PROTECTED_TOOLS:
            return False, None

        key = self._make_key(session_id, tool_name, args)

        if self.redis:
            cached = await self.redis.get(key)
            if cached:
                return True, json.loads(cached)
        else:
            if key in self._local_cache:
                return True, self._local_cache[key]

        return False, None

    async def store_result(
        self, session_id: str, tool_name: str, args: dict, result: dict
    ):
        """Store execution result for deduplication."""
        if not self.config.idempotency_enabled:
            return
        if tool_name not in self.PROTECTED_TOOLS:
            return

        key = self._make_key(session_id, tool_name, args)

        if self.redis:
            await self.redis.setex(key, int(self.ttl.total_seconds()), json.dumps(result))
        else:
            self._local_cache[key] = result

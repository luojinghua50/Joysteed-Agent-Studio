class SemanticMemory:
    """Semantic memory: stores persistent user facts (KV structure)."""

    def __init__(self):
        self._store: dict[str, dict[str, str]] = {}

    async def get_facts(self, customer_id: str) -> dict[str, str]:
        """Get all facts for a customer."""
        return self._store.get(customer_id, {})

    async def set_fact(self, customer_id: str, key: str, value: str):
        """Set a single fact."""
        if customer_id not in self._store:
            self._store[customer_id] = {}
        self._store[customer_id][key] = value

    async def get_fact(self, customer_id: str, key: str) -> str | None:
        """Get a single fact."""
        return self._store.get(customer_id, {}).get(key)

    async def delete_all(self, customer_id: str):
        """Delete all facts for a customer."""
        self._store.pop(customer_id, None)

    async def count(self) -> int:
        return sum(len(v) for v in self._store.values())

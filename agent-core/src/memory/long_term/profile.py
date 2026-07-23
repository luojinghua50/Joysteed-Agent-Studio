from pydantic import BaseModel, Field
from datetime import datetime


class UserProfile(BaseModel):
    """User profile stored in long-term memory."""

    customer_id: str
    vip_level: int = 0
    preferred_channel: str | None = None
    communication_style: str | None = None
    sensitive_points: list[str] = Field(default_factory=list)
    frequent_categories: list[str] = Field(default_factory=list)
    tags: dict[str, str] = Field(default_factory=dict)
    last_updated: datetime | None = None


class ProfileMemory:
    """User profile memory: manages user profiles with in-memory fallback."""

    def __init__(self, redis_client=None):
        self.redis = redis_client
        self._store: dict[str, UserProfile] = {}

    async def get(self, customer_id: str) -> UserProfile:
        """Get user profile, creating a default if not found."""
        if customer_id in self._store:
            return self._store[customer_id]

        profile = UserProfile(customer_id=customer_id)
        self._store[customer_id] = profile
        return profile

    async def save(self, profile: UserProfile):
        """Save/update user profile."""
        profile.last_updated = datetime.now()
        self._store[profile.customer_id] = profile

    async def update_from_conversation(self, customer_id: str, extracted: dict):
        """Update profile from conversation-extracted info."""
        profile = await self.get(customer_id)
        if extracted.get("communication_style"):
            profile.communication_style = extracted["communication_style"]
        if extracted.get("sensitive_points"):
            profile.sensitive_points = list(set(
                profile.sensitive_points + extracted["sensitive_points"]
            ))
        if extracted.get("vip_level"):
            profile.vip_level = extracted["vip_level"]
        profile.last_updated = datetime.now()
        self._store[customer_id] = profile

    async def delete(self, customer_id: str):
        """Delete a user profile (GDPR compliance)."""
        self._store.pop(customer_id, None)

    async def count(self) -> int:
        return len(self._store)

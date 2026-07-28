from pydantic import BaseModel, Field
from datetime import datetime

import structlog

logger = structlog.get_logger()


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
    """User profile memory.

    session_factory 非空 → 走 SQLAlchemy（``ProfileModel`` 表，跨重启持久化）；
    为空 → 内存 dict fallback（测试/降级，与改造前行为一致）。
    """

    def __init__(self, redis_client=None, session_factory=None):
        self.redis = redis_client
        self.session_factory = session_factory
        self._store: dict[str, UserProfile] = {}

    # —— 内存 ↔ ORM 互转 ——
    @staticmethod
    def _to_profile(row) -> UserProfile:
        return UserProfile(
            customer_id=row.customer_id,
            vip_level=row.vip_level,
            preferred_channel=row.preferred_channel,
            communication_style=row.communication_style,
            sensitive_points=list(row.sensitive_points or []),
            frequent_categories=list(row.frequent_categories or []),
            tags=dict(row.tags or {}),
            last_updated=row.last_updated,
        )

    async def get(self, customer_id: str) -> UserProfile:
        """Get user profile, creating a default if not found."""
        if self.session_factory is None:
            if customer_id in self._store:
                return self._store[customer_id]
            profile = UserProfile(customer_id=customer_id)
            self._store[customer_id] = profile
            return profile

        from src.database import ProfileModel
        async with self.session_factory() as db:
            row = await db.get(ProfileModel, customer_id)
            if row is None:
                return UserProfile(customer_id=customer_id)
            return self._to_profile(row)

    async def save(self, profile: UserProfile):
        """Save/update user profile."""
        profile.last_updated = datetime.now()
        if self.session_factory is None:
            self._store[profile.customer_id] = profile
            return

        from src.database import ProfileModel
        async with self.session_factory() as db:
            row = await db.get(ProfileModel, profile.customer_id)
            if row is None:
                row = ProfileModel(customer_id=profile.customer_id)
                db.add(row)
            row.vip_level = profile.vip_level
            row.preferred_channel = profile.preferred_channel
            row.communication_style = profile.communication_style
            row.sensitive_points = list(profile.sensitive_points)
            row.frequent_categories = list(profile.frequent_categories)
            row.tags = dict(profile.tags)
            await db.commit()

    async def update_from_conversation(self, customer_id: str, extracted: dict):
        """Update profile from conversation-extracted info (merge semantics)."""
        profile = await self.get(customer_id)
        if extracted.get("communication_style"):
            profile.communication_style = extracted["communication_style"]
        if extracted.get("sensitive_points"):
            profile.sensitive_points = sorted(set(
                profile.sensitive_points + list(extracted["sensitive_points"])
            ))
        if extracted.get("frequent_categories"):
            profile.frequent_categories = sorted(set(
                profile.frequent_categories + list(extracted["frequent_categories"])
            ))
        if extracted.get("vip_level"):
            profile.vip_level = extracted["vip_level"]
        await self.save(profile)

    async def delete(self, customer_id: str):
        """Delete a user profile (GDPR compliance)."""
        if self.session_factory is None:
            self._store.pop(customer_id, None)
            return

        from sqlalchemy import delete
        from src.database import ProfileModel
        async with self.session_factory() as db:
            await db.execute(delete(ProfileModel).where(ProfileModel.customer_id == customer_id))
            await db.commit()

    async def count(self) -> int:
        if self.session_factory is None:
            return len(self._store)
        from sqlalchemy import func, select
        from src.database import ProfileModel
        async with self.session_factory() as db:
            return (await db.execute(select(func.count()).select_from(ProfileModel))).scalar() or 0

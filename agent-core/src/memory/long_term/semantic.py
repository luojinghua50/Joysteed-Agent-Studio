from datetime import datetime

import structlog

from src.memory.decay import ConfidenceManager

logger = structlog.get_logger()

_confidence = ConfidenceManager()


class SemanticMemory:
    """Semantic memory: persistent user facts (KV structure) with confidence decay.

    session_factory 非空 → SQLAlchemy（``FactModel``，含 source/confidence/updated_at）；
    为空 → 内存 dict fallback（测试/降级）。get_facts_scored 返回按 updated_at 衰减后的
    当前置信度，供 prompt 分档标注；get_facts 保持原 KV 形态兼容现有调用方。
    """

    def __init__(self, session_factory=None):
        self.session_factory = session_factory
        self._store: dict[str, dict[str, str]] = {}
        # 内存 fallback 下记录 source/时间，保证衰减语义一致
        self._meta: dict[str, dict[str, tuple[str, datetime]]] = {}

    async def get_facts(self, customer_id: str) -> dict[str, str]:
        """Get all facts for a customer (plain key->value)."""
        if self.session_factory is None:
            return dict(self._store.get(customer_id, {}))

        from sqlalchemy import select
        from src.database import FactModel
        async with self.session_factory() as db:
            rows = (await db.execute(
                select(FactModel).where(FactModel.customer_id == customer_id)
            )).scalars().all()
            return {r.key: r.value for r in rows}

    async def get_facts_scored(self, customer_id: str) -> list[dict]:
        """Get facts with time-decayed current confidence.

        Returns [{key, value, source, confidence}], confidence 已按 updated_at
        距今天数衰减（ConfidenceManager.current_confidence）。
        """
        out: list[dict] = []
        if self.session_factory is None:
            for key, value in self._store.get(customer_id, {}).items():
                source, ts = self._meta.get(customer_id, {}).get(key, ("agent_inferred", datetime.now()))
                initial = _confidence.initial_confidence(source)
                days = max((datetime.now() - ts).days, 0)
                out.append({"key": key, "value": value, "source": source,
                            "confidence": _confidence.current_confidence(initial, days)})
            return out

        from sqlalchemy import select
        from src.database import FactModel
        async with self.session_factory() as db:
            rows = (await db.execute(
                select(FactModel).where(FactModel.customer_id == customer_id)
            )).scalars().all()
        for r in rows:
            days = max((datetime.now() - (r.updated_at or datetime.now())).days, 0)
            out.append({"key": r.key, "value": r.value, "source": r.source,
                        "confidence": _confidence.current_confidence(r.confidence, days)})
        return out

    async def set_fact(self, customer_id: str, key: str, value: str,
                       source: str = "agent_inferred"):
        """Set a single fact. Initial confidence derived from source."""
        if self.session_factory is None:
            self._store.setdefault(customer_id, {})[key] = value
            self._meta.setdefault(customer_id, {})[key] = (source, datetime.now())
            return

        from sqlalchemy import select
        from src.database import FactModel
        async with self.session_factory() as db:
            row = (await db.execute(
                select(FactModel).where(
                    FactModel.customer_id == customer_id, FactModel.key == key
                )
            )).scalar_one_or_none()
            if row is None:
                row = FactModel(customer_id=customer_id, key=key)
                db.add(row)
            row.value = value
            row.source = source
            row.confidence = _confidence.initial_confidence(source)
            await db.commit()

    async def get_fact(self, customer_id: str, key: str) -> str | None:
        """Get a single fact value."""
        facts = await self.get_facts(customer_id)
        return facts.get(key)

    async def delete_all(self, customer_id: str):
        """Delete all facts for a customer."""
        if self.session_factory is None:
            self._store.pop(customer_id, None)
            self._meta.pop(customer_id, None)
            return

        from sqlalchemy import delete
        from src.database import FactModel
        async with self.session_factory() as db:
            await db.execute(delete(FactModel).where(FactModel.customer_id == customer_id))
            await db.commit()

    async def count(self) -> int:
        if self.session_factory is None:
            return sum(len(v) for v in self._store.values())
        from sqlalchemy import func, select
        from src.database import FactModel
        async with self.session_factory() as db:
            return (await db.execute(select(func.count()).select_from(FactModel))).scalar() or 0

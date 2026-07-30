"""长期记忆-历史交互摘要（Episodic Memory）。

- 结构化真相源落 SQLAlchemy ``EpisodeModel``（审计、按 id 精确载全量）。
- 摘要向量另存 Milvus 供语义检索；检索 = 向量召回 × 时间衰减重排（MemoryDecay）。
- 三重降级：无 session_factory → 内存 list；无 embedder/milvus → DB 最近 N 条
  （等价改造前的"假搜索"，但数据来自持久层）。摘要由 MemoryManager 传入（LLM 生成），
  本类不持 LLM。
"""
from datetime import datetime

import structlog
from pydantic import BaseModel, Field

from src.memory.decay import MemoryDecay

logger = structlog.get_logger()

_decay = MemoryDecay()


class EpisodeRecord(BaseModel):
    """A single episodic memory entry."""

    session_id: str
    customer_id: str
    timestamp: datetime = Field(default_factory=datetime.now)
    summary: str
    intent: str = "general"
    resolution: str = "resolved"
    key_entities: dict = Field(default_factory=dict)
    satisfaction: float | None = None
    id: int | None = None


class EpisodicMemory:
    """Episodic memory: conversation summaries with vector semantic retrieval."""

    def __init__(self, session_factory=None, embedder=None, milvus_client=None,
                 collection: str = "memory_episodes", dim: int = 512):
        self.session_factory = session_factory
        self.embedder = embedder
        self.milvus = milvus_client
        self.collection = collection
        self.dim = dim
        self._store: dict[str, list[EpisodeRecord]] = {}  # 内存 fallback
        self._collection_ready = False

    # ————————————————————— 保存 —————————————————————
    async def save_episode(
        self, session_id: str, customer_id: str, messages: list,
        summary: str | None = None, intent: str = "general",
        resolution: str = "resolved", key_entities: dict | None = None,
        satisfaction: float | None = None,
    ):
        """保存一次会话摘要。summary 由调用方（MemoryManager）用 LLM 生成；缺省时
        退回极简拼接（仅无 LLM 的降级路径用）。"""
        if summary is None:
            summary = self._fallback_summary(messages)
        record = EpisodeRecord(
            session_id=session_id, customer_id=customer_id, summary=summary,
            intent=intent, resolution=resolution,
            key_entities=key_entities or {}, satisfaction=satisfaction,
        )

        if self.session_factory is None:
            self._store.setdefault(customer_id, []).append(record)
            return

        from src.database import EpisodeModel
        async with self.session_factory() as db:
            row = EpisodeModel(
                session_id=session_id, customer_id=customer_id, summary=summary,
                intent=intent, resolution=resolution,
                key_entities=key_entities or {}, satisfaction=satisfaction,
            )
            db.add(row)
            await db.commit()
            await db.refresh(row)
            episode_id = row.id
            created = row.created_at or datetime.now()

        # 向量另存 Milvus（失败不阻断——DB 已是真相源，检索可降级）
        await self._index_vector(episode_id, customer_id, summary, created)

    def _fallback_summary(self, messages: list) -> str:
        if not messages:
            return "空会话"
        texts = [getattr(m, "content", str(m))[:50] for m in messages[:5]]
        return "会话摘要: " + " | ".join(texts)

    # ————————————————————— Milvus 向量 —————————————————————
    def _ensure_collection(self):
        """建 collection（幂等，参 agent-rag _ensure_collection）。仅在有 milvus 时调。"""
        if self._collection_ready or self.milvus is None:
            return
        from pymilvus import DataType

        if not self.milvus.has_collection(self.collection):
            schema = self.milvus.create_schema(auto_id=False, enable_dynamic_field=True)
            schema.add_field("episode_id", DataType.INT64, is_primary=True)
            schema.add_field("customer_id", DataType.VARCHAR, max_length=64)
            schema.add_field("created_ts", DataType.INT64)
            schema.add_field("dense", DataType.FLOAT_VECTOR, dim=self.dim)
            index_params = self.milvus.prepare_index_params()
            index_params.add_index(field_name="dense", index_type="AUTOINDEX",
                                   metric_type="COSINE")
            self.milvus.create_collection(self.collection, schema=schema,
                                          index_params=index_params)
            logger.info("milvus_collection_created", collection=self.collection)
        self._collection_ready = True

    async def _index_vector(self, episode_id, customer_id, summary, created):
        """把摘要向量写入 Milvus。embedder/milvus 缺失或异常时静默跳过（检索会降级）。"""
        if self.embedder is None or self.milvus is None:
            return
        try:
            self._ensure_collection()
            vec = await self.embedder.embed_one(summary)
            self.milvus.insert(collection_name=self.collection, data=[{
                "episode_id": int(episode_id),
                "customer_id": customer_id,
                "created_ts": int(created.timestamp()),
                "dense": vec,
            }])
        except Exception as e:
            logger.warning("milvus_index_failed", error=str(e))

    # ————————————————————— 检索 —————————————————————
    async def search(
        self, query: str, customer_id: str, top_k: int = 3
    ) -> list[EpisodeRecord]:
        """语义检索：向量召回 × 时间衰减重排。任一环节不可用则降级 DB/内存最近 N 条。"""
        # 有向量栈 → 真检索
        if self.embedder is not None and self.milvus is not None and self.session_factory is not None:
            try:
                return await self._vector_search(query, customer_id, top_k)
            except Exception as e:
                logger.warning("episodic_vector_search_degraded", error=str(e))
        # 降级：最近 N 条
        return await self._recent(customer_id, top_k)

    async def _vector_search(self, query, customer_id, top_k) -> list[EpisodeRecord]:
        self._ensure_collection()
        qvec = await self.embedder.embed_one(query)
        # consistency_level="Strong"：MilvusClient 默认 Bounded 一致性，会话结束刚归档
        # 的历史若立即检索（同/近会话）可能看不到，读到过时状态导致召回错乱。记忆检索
        # 对准确性要求高于毫秒级延迟，故用强一致性确保"写完即可查"。
        hits = self.milvus.search(
            collection_name=self.collection, data=[qvec], anns_field="dense",
            filter=f'customer_id == "{customer_id}"', limit=max(top_k * 3, top_k),
            output_fields=["episode_id"], search_params={"metric_type": "COSINE"},
            consistency_level="Strong",
        )
        # hits: [[{id, distance, entity}]]
        scored: list[tuple[int, float]] = []
        for h in (hits[0] if hits else []):
            eid = h.get("entity", {}).get("episode_id") or h.get("id")
            scored.append((int(eid), float(h.get("distance", 0.0))))
        if not scored:
            return await self._recent(customer_id, top_k)

        from sqlalchemy import select
        from src.database import EpisodeModel
        async with self.session_factory() as db:
            rows = {r.id: r for r in (
                await db.execute(
                    select(EpisodeModel).where(EpisodeModel.id.in_([e for e, _ in scored]))
                )
            ).scalars().all()}

        # 时间衰减重排：语义分 × exp(-rate·days_ago)
        now = datetime.now()
        ranked = []
        for eid, score in scored:
            row = rows.get(eid)
            if row is None:
                continue
            days = max((now - (row.created_at or now)).days, 0)
            ranked.append((_decay.episodic_relevance(score, days), row))
        ranked.sort(key=lambda x: x[0], reverse=True)
        return [self._to_record(r) for _, r in ranked[:top_k]]

    async def _recent(self, customer_id: str, top_k: int) -> list[EpisodeRecord]:
        """降级路径：返回最近 N 条（数据来自持久层或内存）。"""
        if self.session_factory is None:
            return self._store.get(customer_id, [])[-top_k:]
        from sqlalchemy import select
        from src.database import EpisodeModel
        async with self.session_factory() as db:
            rows = (await db.execute(
                select(EpisodeModel).where(EpisodeModel.customer_id == customer_id)
                .order_by(EpisodeModel.created_at.desc()).limit(top_k)
            )).scalars().all()
        return [self._to_record(r) for r in reversed(rows)]

    @staticmethod
    def _to_record(row) -> EpisodeRecord:
        return EpisodeRecord(
            id=row.id, session_id=row.session_id, customer_id=row.customer_id,
            timestamp=row.created_at or datetime.now(), summary=row.summary,
            intent=row.intent, resolution=row.resolution,
            key_entities=dict(row.key_entities or {}), satisfaction=row.satisfaction,
        )

    async def delete_all(self, customer_id: str):
        """删除某客户全部历史（GDPR）：DB + Milvus 向量。"""
        if self.session_factory is None:
            self._store.pop(customer_id, None)
            return
        from sqlalchemy import delete
        from src.database import EpisodeModel
        async with self.session_factory() as db:
            await db.execute(delete(EpisodeModel).where(EpisodeModel.customer_id == customer_id))
            await db.commit()
        if self.milvus is not None:
            try:
                self._ensure_collection()
                self.milvus.delete(collection_name=self.collection,
                                   filter=f'customer_id == "{customer_id}"')
            except Exception as e:
                logger.warning("milvus_delete_failed", error=str(e))

    async def count(self) -> int:
        if self.session_factory is None:
            return sum(len(v) for v in self._store.values())
        from sqlalchemy import func, select
        from src.database import EpisodeModel
        async with self.session_factory() as db:
            return (await db.execute(select(func.count()).select_from(EpisodeModel))).scalar() or 0

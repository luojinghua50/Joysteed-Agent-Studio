"""Retrieval engine with pluggable backends.

- MemoryRetriever: keyword matching, no external deps. Default so the service
  runs without Milvus/embedding.
- MilvusRetriever: dense + sparse(BM25) hybrid search. Ready but requires Milvus
  and embedding to be configured.

Both filter by version_id so shadow (not-yet-activated) versions are invisible.
"""
from src.models import SearchResult, SearchRequest


# 库内三种检索模式
MODE_VECTOR = "vector"      # 仅稠密向量（语义）
MODE_FULLTEXT = "fulltext"  # 仅稀疏 BM25（关键词）
MODE_HYBRID = "hybrid"      # 稠密+稀疏，库内 RRF 融合


def resolve_mode(request_mode: str | None, kb_mode: str | None) -> str:
    """请求级 mode 覆盖库级 retrieval_mode；都缺省则 hybrid。"""
    mode = (request_mode or kb_mode or MODE_HYBRID).lower()
    return mode if mode in (MODE_VECTOR, MODE_FULLTEXT, MODE_HYBRID) else MODE_HYBRID


def reciprocal_rank_fusion(
    result_lists: list[list[SearchResult]],
    k: int = 60,
    weights: list[float] | None = None,
) -> list[SearchResult]:
    """RRF 融合多路检索结果：score = Σ weight / (k + rank)。

    rank 从 1 起算。同一 chunk 在多路命中则分数累加，融合分写回 result.score。
    """
    if weights is None:
        weights = [1.0] * len(result_lists)
    scores: dict[str, float] = {}
    chunk_map: dict[str, SearchResult] = {}
    for weight, results in zip(weights, result_lists):
        for rank, result in enumerate(results, 1):
            cid = result.chunk_id
            scores[cid] = scores.get(cid, 0.0) + weight / (k + rank)
            chunk_map[cid] = result
    ordered = sorted(scores.keys(), key=lambda c: scores[c], reverse=True)
    fused = []
    for cid in ordered:
        r = chunk_map[cid]
        r.score = round(scores[cid], 6)
        fused.append(r)
    return fused


def match_filters(meta: dict, filters: list) -> bool:
    """库内元数据过滤：所有条件 AND 成立才保留。meta 取不到的字段视为不匹配。"""
    if not filters:
        return True
    for f in filters:
        field = getattr(f, "field", None) if not isinstance(f, dict) else f.get("field")
        op = getattr(f, "op", None) if not isinstance(f, dict) else f.get("op")
        value = getattr(f, "value", None) if not isinstance(f, dict) else f.get("value")
        if field not in (meta or {}):
            return False
        actual = meta[field]
        try:
            if op == "eq" and not (actual == value):
                return False
            elif op == "in" and actual not in (value or []):
                return False
            elif op == "gt" and not (actual > value):
                return False
            elif op == "gte" and not (actual >= value):
                return False
            elif op == "lt" and not (actual < value):
                return False
            elif op == "lte" and not (actual <= value):
                return False
        except TypeError:
            return False
    return True


class BaseRetriever:
    async def index_chunks(self, kb_id: str, chunks: list[dict]):
        raise NotImplementedError

    async def search(self, request: SearchRequest, visible_version_ids: list[str] | None = None,
                     kb_mode: str | None = None) -> list[SearchResult]:
        raise NotImplementedError

    async def delete_by_version(self, kb_id: str, version_id: str):
        raise NotImplementedError

    async def delete_kb(self, kb_id: str):
        raise NotImplementedError

    async def ensure_scalar_index(self, kb_id: str, field_name: str, field_type: str):
        """为库内可过滤的元数据字段建标量索引（默认 no-op，仅 Milvus 实现）。"""
        return None


class MemoryRetriever(BaseRetriever):
    """In-memory retriever. Default backend, no external dependencies.

    无 Milvus 时模拟库内三模式：fulltext 走关键词词项匹配（BM25 代理），
    vector 走字符 n-gram Jaccard（语义代理），hybrid 在库内 RRF 融合两路。
    """

    def __init__(self):
        self._chunks: dict[str, list[dict]] = {}

    async def index_chunks(self, kb_id: str, chunks: list[dict]):
        self._chunks.setdefault(kb_id, []).extend(chunks)

    @staticmethod
    def _ngrams(text: str, n: int = 2) -> set[str]:
        t = text.lower().strip()
        if len(t) < n:
            return {t} if t else set()
        return {t[i:i + n] for i in range(len(t) - n + 1)}

    def _result_of(self, chunk: dict, kb_id: str, score: float) -> SearchResult:
        return SearchResult(
            chunk_id=chunk.get("id", ""),
            doc_id=chunk.get("doc_id", ""),
            text=chunk.get("text", ""),
            score=score,
            metadata=chunk.get("metadata", {}) or chunk.get("meta", {}),
            source=chunk.get("context_header", ""),
            kb_id=chunk.get("kb_id", kb_id),
            version_no=chunk.get("version_no"),
        )

    def _fulltext_rank(self, chunks: list[dict], request: SearchRequest) -> list[SearchResult]:
        """关键词词项匹配（BM25 代理）。"""
        query_lower = request.query.lower()
        query_terms = [t for t in query_lower.split() if len(t) >= 2] or [query_lower]
        out = []
        for chunk in chunks:
            text_lower = chunk.get("text", "").lower()
            keywords = " ".join(chunk.get("keywords", [])).lower()
            combined = text_lower + " " + keywords
            score = 0.0
            for term in query_terms:
                if term in combined:
                    score += 0.3
                    if term in chunk.get("context_header", "").lower():
                        score += 0.2
            if query_lower in text_lower:  # 整串包含（中文无空格分词兜底）
                score += 0.4
            if score > 0:
                out.append(self._result_of(chunk, request.kb_id, min(score, 1.0)))
        out.sort(key=lambda r: r.score, reverse=True)
        return out

    def _vector_rank(self, chunks: list[dict], request: SearchRequest) -> list[SearchResult]:
        """字符 n-gram Jaccard（稠密语义代理，无需 embedding 模型）。"""
        q_grams = self._ngrams(request.query)
        if not q_grams:
            return []
        out = []
        for chunk in chunks:
            c_grams = self._ngrams(chunk.get("text", ""))
            if not c_grams:
                continue
            inter = len(q_grams & c_grams)
            if inter == 0:
                continue
            sim = inter / len(q_grams | c_grams)
            out.append(self._result_of(chunk, request.kb_id, round(sim, 6)))
        out.sort(key=lambda r: r.score, reverse=True)
        return out

    async def search(self, request: SearchRequest, visible_version_ids: list[str] | None = None,
                     kb_mode: str | None = None) -> list[SearchResult]:
        kb_chunks = self._chunks.get(request.kb_id, [])
        if not kb_chunks:
            return []

        if visible_version_ids is not None:
            visible = set(visible_version_ids)
            kb_chunks = [c for c in kb_chunks if c.get("version_id") in visible]

        # 库内元数据过滤（filters AND 组合）
        filters = getattr(request, "filters", None) or []
        if filters:
            kb_chunks = [c for c in kb_chunks if match_filters(c.get("metadata", {}) or c.get("meta", {}), filters)]

        mode = resolve_mode(getattr(request, "mode", None), kb_mode)
        recall = max(request.top_k, 20)
        if mode == MODE_VECTOR:
            results = self._vector_rank(kb_chunks, request)
        elif mode == MODE_FULLTEXT:
            results = self._fulltext_rank(kb_chunks, request)
        else:  # hybrid：库内 RRF 融合稠密+稀疏两路
            dense = self._vector_rank(kb_chunks, request)[:recall]
            sparse = self._fulltext_rank(kb_chunks, request)[:recall]
            results = reciprocal_rank_fusion([dense, sparse])
        return results[:request.top_k]

    async def delete_by_version(self, kb_id: str, version_id: str):
        if kb_id in self._chunks:
            self._chunks[kb_id] = [c for c in self._chunks[kb_id] if c.get("version_id") != version_id]

    async def delete_kb(self, kb_id: str):
        self._chunks.pop(kb_id, None)

    async def count(self, kb_id: str) -> int:
        return len(self._chunks.get(kb_id, []))


def create_retriever(settings) -> BaseRetriever:
    """Factory: pick retrieval backend by config. Falls back to memory."""
    if settings.retrieval_backend == "milvus":
        try:
            from src.retrieval.milvus_backend import MilvusRetriever

            return MilvusRetriever(settings)
        except Exception:  # pragma: no cover - depends on external service
            import structlog

            structlog.get_logger().warning("milvus_unavailable_fallback_memory")
            return MemoryRetriever()
    return MemoryRetriever()


# Backwards-compatible alias: existing tests import RetrievalEngine.
class RetrievalEngine(MemoryRetriever):
    """Deprecated alias kept for compatibility. Adds legacy delete_by_doc."""

    async def delete_by_doc(self, kb_id: str, doc_id: str):
        if kb_id in self._chunks:
            self._chunks[kb_id] = [c for c in self._chunks[kb_id] if c.get("doc_id") != doc_id]

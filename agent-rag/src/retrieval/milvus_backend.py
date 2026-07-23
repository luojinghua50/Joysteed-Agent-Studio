"""Milvus retrieval backend: collection-per-kb hybrid (dense + sparse BM25) search.

Active when retrieval_backend="milvus". Each knowledge base maps to its own
Milvus collection (name derived as kb_<kb_id>), so different knowledge forms can
hold their own schema/index without cross-kb noise. Filters by version_id so
only the currently-active version of each document is searched (shadow/archived
versions stay indexed but invisible).

Each collection carries a dense FLOAT_VECTOR (semantic) and a SPARSE_FLOAT_VECTOR
fed by Milvus' built-in BM25 function over the analyzed text field. Search runs
in one of three modes (vector / fulltext / hybrid); hybrid fuses dense + sparse
in-collection via RRFRanker through a single hybrid_search call. Metadata scalar
filtering compiles into the same boolean expression applied to every mode.
"""
import re

import structlog

from src.models import SearchResult, SearchRequest
from src.retrieval import BaseRetriever
from src.embedding import Embedder

logger = structlog.get_logger()


def collection_name_for(kb_id: str) -> str:
    """Derive a valid Milvus collection name from kb_id.

    Milvus names must start with a letter/underscore and contain only
    letters/digits/underscores. kb ids are short hex-ish ids, so prefix + sanitize.
    """
    safe = re.sub(r"[^0-9a-zA-Z_]", "_", kb_id or "")
    return f"kb_{safe}"


def _lit(value) -> str:
    """把过滤值编译成 Milvus 表达式字面量。"""
    if isinstance(value, str):
        escaped = value.replace('"', '\\"')
        return f'"{escaped}"'
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def compile_filters(filters: list) -> str:
    """把 MetaFilter 列表编译成 Milvus boolean 表达式（AND 组合）。"""
    parts = []
    for f in filters or []:
        field = getattr(f, "field", None) if not isinstance(f, dict) else f.get("field")
        op = getattr(f, "op", None) if not isinstance(f, dict) else f.get("op")
        value = getattr(f, "value", None) if not isinstance(f, dict) else f.get("value")
        if not field or not op:
            continue
        if op == "eq":
            parts.append(f'{field} == {_lit(value)}')
        elif op == "in":
            vals = ", ".join(_lit(v) for v in (value or []))
            parts.append(f'{field} in [{vals}]')
        elif op == "gt":
            parts.append(f'{field} > {_lit(value)}')
        elif op == "gte":
            parts.append(f'{field} >= {_lit(value)}')
        elif op == "lt":
            parts.append(f'{field} < {_lit(value)}')
        elif op == "lte":
            parts.append(f'{field} <= {_lit(value)}')
    return " && ".join(parts)


class MilvusRetriever(BaseRetriever):
    def __init__(self, settings):
        from pymilvus import MilvusClient

        self.settings = settings
        self.embedder = Embedder(settings)
        self.client = MilvusClient(uri=f"http://{settings.milvus_host}:{settings.milvus_port}")
        # 已确保存在的 collection 名缓存，避免每次 has_collection RPC
        self._ensured: set[str] = set()

    def _ensure_collection(self, collection: str):
        from pymilvus import DataType, Function, FunctionType

        if collection in self._ensured:
            return
        if self.client.has_collection(collection):
            self._ensured.add(collection)
            return
        schema = self.client.create_schema(auto_id=False, enable_dynamic_field=True)
        schema.add_field("chunk_id", DataType.VARCHAR, is_primary=True, max_length=32)
        schema.add_field("version_id", DataType.VARCHAR, max_length=24)
        schema.add_field("doc_id", DataType.VARCHAR, max_length=16)
        schema.add_field("kb_id", DataType.VARCHAR, max_length=16)
        # text 开 analyzer，供 BM25 function 切词；中文用 jieba 分析器（步骤7 验证调优）
        schema.add_field("text", DataType.VARCHAR, max_length=8192,
                         enable_analyzer=True, analyzer_params={"type": "chinese"})
        schema.add_field("dense", DataType.FLOAT_VECTOR, dim=self.settings.embedding_dim)
        # 稀疏 BM25 向量：由 Milvus 内建 BM25 function 从 text 自动生成
        schema.add_field("sparse", DataType.SPARSE_FLOAT_VECTOR)
        schema.add_function(Function(
            name="text_bm25", function_type=FunctionType.BM25,
            input_field_names=["text"], output_field_names=["sparse"],
        ))

        index_params = self.client.prepare_index_params()
        index_params.add_index("dense", index_type="HNSW", metric_type="IP",
                               params={"M": 16, "efConstruction": 200})
        index_params.add_index("sparse", index_type="SPARSE_INVERTED_INDEX",
                               metric_type="BM25")
        index_params.add_index("version_id", index_type="INVERTED")
        self.client.create_collection(collection, schema=schema, index_params=index_params)
        self._ensured.add(collection)
        logger.info("milvus_collection_created", collection=collection)

    async def index_chunks(self, kb_id: str, chunks: list[dict]):
        if not chunks:
            return
        collection = collection_name_for(kb_id)
        self._ensure_collection(collection)
        texts = [c["text"] for c in chunks]
        vectors = await self.embedder.embed_batch(texts)
        rows = []
        for c, vec in zip(chunks, vectors):
            row = {
                "chunk_id": c["id"], "version_id": c.get("version_id", ""),
                "doc_id": c.get("doc_id", ""), "kb_id": kb_id,
                "text": c["text"][:8192], "dense": vec,
            }
            # 元数据字段下沉为动态字段（步骤3 会对要过滤的字段建标量索引）
            for k, v in (c.get("metadata") or {}).items():
                if k not in row:
                    row[k] = v
            rows.append(row)
        self.client.insert(collection_name=collection, data=rows)
        self.client.flush(collection)  # make rows queryable immediately

    async def search(self, request: SearchRequest, visible_version_ids: list[str] | None = None,
                     kb_mode: str | None = None) -> list[SearchResult]:
        from src.retrieval import resolve_mode, MODE_VECTOR, MODE_FULLTEXT

        collection = collection_name_for(request.kb_id)
        if not self.client.has_collection(collection):
            return []

        # 版本可见性 + 元数据过滤合并成一个 Milvus 表达式
        exprs = []
        if visible_version_ids is not None:
            if not visible_version_ids:
                return []
            quoted = ", ".join(f'"{v}"' for v in visible_version_ids)
            exprs.append(f"version_id in [{quoted}]")
        meta_expr = compile_filters(getattr(request, "filters", None) or [])
        if meta_expr:
            exprs.append(meta_expr)
        expr = " && ".join(exprs)

        mode = resolve_mode(getattr(request, "mode", None), kb_mode)
        self.client.load_collection(collection)
        out_fields = ["chunk_id", "doc_id", "text"]

        if mode == MODE_FULLTEXT:
            hits = self.client.search(
                collection_name=collection, data=[request.query], anns_field="sparse",
                filter=expr, limit=request.top_k, output_fields=out_fields,
            )
            return self._to_results(hits, request.kb_id)

        query_vec = (await self.embedder.embed_batch([request.query]))[0]
        if mode == MODE_VECTOR:
            hits = self.client.search(
                collection_name=collection, data=[query_vec], anns_field="dense",
                filter=expr, limit=request.top_k, output_fields=out_fields,
            )
            return self._to_results(hits, request.kb_id)

        # hybrid：稠密 + 稀疏(BM25) 库内 RRF 融合，一次 hybrid_search 完成
        return self._hybrid_search(collection, request, query_vec, expr, out_fields)

    def _hybrid_search(self, collection, request, query_vec, expr, out_fields):
        from pymilvus import AnnSearchRequest, RRFRanker

        recall = max(request.top_k, 20)
        dense_req = AnnSearchRequest(
            data=[query_vec], anns_field="dense",
            param={"metric_type": "IP"}, limit=recall, expr=expr or None,
        )
        sparse_req = AnnSearchRequest(
            data=[request.query], anns_field="sparse",
            param={"metric_type": "BM25"}, limit=recall, expr=expr or None,
        )
        hits = self.client.hybrid_search(
            collection_name=collection,
            reqs=[dense_req, sparse_req],
            ranker=RRFRanker(60),
            limit=request.top_k, output_fields=out_fields,
        )
        return self._to_results(hits, request.kb_id)

    @staticmethod
    def _to_results(hits, kb_id: str) -> list[SearchResult]:
        results = []
        for hit in (hits[0] if hits else []):
            entity = hit.get("entity", {})
            results.append(SearchResult(
                chunk_id=entity.get("chunk_id", ""), doc_id=entity.get("doc_id", ""),
                text=entity.get("text", ""), score=float(hit.get("distance", 0.0)),
                kb_id=kb_id,
            ))
        return results

    async def delete_by_version(self, kb_id: str, version_id: str):
        collection = collection_name_for(kb_id)
        if not self.client.has_collection(collection):
            return
        self.client.delete(collection_name=collection, filter=f'version_id == "{version_id}"')
        self.client.flush(collection)

    async def delete_kb(self, kb_id: str):
        collection = collection_name_for(kb_id)
        if self.client.has_collection(collection):
            self.client.drop_collection(collection)  # 整库即整 collection，直接 drop 干净
            self._ensured.discard(collection)

    async def ensure_scalar_index(self, kb_id: str, field_name: str, field_type: str):
        """给库内可过滤字段建标量索引，加速 filter 表达式。

        字段以动态字段（JSON $meta 内）存储，Milvus 支持对动态字段建 INVERTED 索引。
        collection 尚不存在（建库后未上传）时跳过，待 index_chunks 建表后下次定义触发。
        失败不阻断字段定义（best-effort），仅记录告警。
        """
        collection = collection_name_for(kb_id)
        if not self.client.has_collection(collection):
            return
        try:
            self.client.release_collection(collection)
            index_params = self.client.prepare_index_params()
            index_params.add_index(field_name, index_type="INVERTED")
            self.client.create_index(collection, index_params=index_params)
            logger.info("milvus_scalar_index_created",
                        collection=collection, field=field_name, field_type=field_type)
        except Exception as e:  # pragma: no cover - depends on external Milvus
            logger.warning("milvus_scalar_index_failed",
                           collection=collection, field=field_name, error=str(e))


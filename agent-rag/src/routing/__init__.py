"""检索路由层：跨多库聚合检索（Agent 主用入口）。

单库 /api/search 仍保留供调试。生产链路走 /api/route-search：
  1. faq 先探针：命中高置信 FAQ 直接返回单一答案，跳过全量检索
  2. 多路并行：对参与的每个库并发检索（各库用自己的 retrieval_mode）
  3. 跨库加权 RRF：按库级 priority_weight 融合多库结果

SearchRouter 不依赖 DB —— 端点负责把参与库解析成 KbPlan（含 visible
versions、字段定义），router 只做检索编排与融合，便于单测。
"""
import asyncio
from dataclasses import dataclass, field
from datetime import datetime, timezone

from src.models import SearchRequest, RouteSearchResponse
from src.retrieval import reciprocal_rank_fusion

FAQ_FORM = "faq"
TEMPORAL_FORM = "temporal"

# temporal 库有效期字段约定（epoch 秒）
TEMPORAL_EFFECTIVE_FIELD = "effective_ts"
TEMPORAL_EXPIRE_FIELD = "expire_ts"


def apply_temporal_filters(kb_form: str, field_defs: dict, filters: list) -> list:
    """temporal 库自动追加"当前有效"过滤：effective_ts <= now <= expire_ts。

    仅当库定义了对应 time 字段、且调用方未显式过滤该字段时注入。
    """
    if kb_form != TEMPORAL_FORM:
        return filters
    from src.models import MetaFilter

    now = int(datetime.now(timezone.utc).timestamp())
    existing = {getattr(f, "field", None) for f in filters}
    out = list(filters)
    if field_defs.get(TEMPORAL_EFFECTIVE_FIELD) == "time" and TEMPORAL_EFFECTIVE_FIELD not in existing:
        out.append(MetaFilter(field=TEMPORAL_EFFECTIVE_FIELD, op="lte", value=now))
    if field_defs.get(TEMPORAL_EXPIRE_FIELD) == "time" and TEMPORAL_EXPIRE_FIELD not in existing:
        out.append(MetaFilter(field=TEMPORAL_EXPIRE_FIELD, op="gte", value=now))
    return out


@dataclass
class KbPlan:
    """单个参与库的检索计划（DB 解析产物，喂给 router）。"""
    kb_id: str
    kb_form: str = "standard"
    retrieval_mode: str = "hybrid"
    top_k: int = 5
    priority_weight: float = 0.7
    embedding_model: str | None = None
    rerank_model: str | None = None
    vector_weight: float = 0.6
    keyword_weight: float = 0.4
    score_threshold: float = 0.0
    shortcut_threshold: float = 0.0
    visible_version_ids: list[str] = field(default_factory=list)
    field_defs: dict = field(default_factory=dict)


class SearchRouter:
    """跨库检索编排器（无状态，包一个 retriever 和可选 reranker）。"""

    def __init__(self, retriever, reranker=None):
        self.retriever = retriever
        self.reranker = reranker

    async def _search_one(self, plan: KbPlan, query: str, top_k: int, base_filters: list):
        filters = apply_temporal_filters(plan.kb_form, plan.field_defs, base_filters)
        req = SearchRequest(
            query=query, kb_id=plan.kb_id, top_k=top_k, filters=filters,
            embedding_model=plan.embedding_model,
            vector_weight=plan.vector_weight, keyword_weight=plan.keyword_weight,
            score_threshold=plan.score_threshold,
        )
        return await self.retriever.search(
            req, visible_version_ids=plan.visible_version_ids, kb_mode=plan.retrieval_mode,
        )

    async def _vector_probe_score(self, plan: KbPlan, query: str, base_filters: list) -> float:
        """faq 短路的置信信号：用 vector 模式单独探一刀，取稠密语义相似度 top 分。

        库级检索模式可能是 hybrid（融合分是基于排名的 RRF，不反映匹配质量、尺度也不对），
        不能拿来当"高置信"判断。这里强制走 vector，分数落在归一化语义相似度尺度上
        （经校准：精确问法 ~0.72，同义 ~0.66，无关 ~0.37），阈值才有意义。
        """
        filters = apply_temporal_filters(plan.kb_form, plan.field_defs, base_filters)
        req = SearchRequest(
            query=query, kb_id=plan.kb_id, top_k=1, filters=filters,
            embedding_model=plan.embedding_model,
        )
        res = await self.retriever.search(
            req, visible_version_ids=plan.visible_version_ids, kb_mode="vector",
        )
        return res[0].score if res else 0.0

    async def _faq_shortcut_plan(self, plans: list[KbPlan], query: str, base_filters: list) -> KbPlan | None:
        faq_plans = [p for p in plans if p.kb_form == FAQ_FORM and p.shortcut_threshold > 0]
        if not faq_plans:
            return None
        probes = await asyncio.gather(
            *[self._vector_probe_score(p, query, base_filters) for p in faq_plans]
        )
        for plan, probe in zip(faq_plans, probes):
            if probe >= plan.shortcut_threshold:
                return plan
        return None

    async def route(self, query: str, plans: list[KbPlan], top_k: int | None = None,
                    base_filters: list | None = None) -> RouteSearchResponse:
        base_filters = base_filters or []
        if not plans:
            return RouteSearchResponse(query=query, results=[], total=0)
        final_top_k = top_k or max((p.top_k for p in plans), default=5)

        shortcut_plan = await self._faq_shortcut_plan(plans, query, base_filters)
        if shortcut_plan is not None:
            out = await self._search_one(shortcut_plan, query, final_top_k, base_filters)
            out = out[:final_top_k]
            return RouteSearchResponse(
                query=query, results=out, total=len(out),
                shortcut=True, routed_kbs=[shortcut_plan.kb_id],
            )

        # 多路并行：每库各自检索（含 temporal 自动过滤）
        searches = await asyncio.gather(
            *[self._search_one(p, query, final_top_k, base_filters) for p in plans]
        )

        # 跨库加权 RRF：库级 priority_weight 作为各路权重（粗排，决定候选入围）
        weights = [p.priority_weight for p in plans]
        fused = reciprocal_rank_fusion(list(searches), weights=weights)
        routed = [p.kb_id for p, r in zip(plans, searches) if r]

        # rerank 精排：对 RRF top-N 候选用 cross-encoder 按语义重排，取 top_k。
        # reranker 缺省/禁用/异常时退回 RRF 顺序（reranked=False），不阻断检索。
        reranked = False
        if self.reranker is not None and self.reranker.enabled:
            import inspect

            candidate_n = getattr(self.reranker.settings, "rerank_candidate_n", 20)
            candidates = fused[:candidate_n]
            plan_by_kb = {p.kb_id: p for p in plans}
            rerank_model = None
            if candidates:
                candidate_plan = plan_by_kb.get(candidates[0].kb_id)
                rerank_model = candidate_plan.rerank_model if candidate_plan else None
            if "model_name" in inspect.signature(self.reranker.rerank).parameters:
                out, reranked = await self.reranker.rerank(
                    query, candidates, final_top_k, model_name=rerank_model,
                )
            else:
                out, reranked = await self.reranker.rerank(query, candidates, final_top_k)
        else:
            out = fused[:final_top_k]

        return RouteSearchResponse(
            query=query, results=out, total=len(out),
            shortcut=False, reranked=reranked, routed_kbs=routed,
        )

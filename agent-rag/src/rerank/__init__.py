"""Reranking: cross-encoder 精排，串在跨库 RRF 粗排之后。

权重粗排（priority_weight 决定哪些候选入围）→ rerank 精排（语义分定最终序）。
provider:
- disabled: no-op，返回原序（默认，单测/无依赖）。
- fastembed: 本地 ONNX cross-encoder（如 BAAI/bge-reranker-base），无 torch、离线。

rerank 不可用时调用方应降级回 RRF 顺序，不阻断检索。
"""
import structlog

logger = structlog.get_logger()


class Reranker:
    """跨库候选精排器。无状态，包一个 cross-encoder 模型。"""

    def __init__(self, settings):
        self.settings = settings
        self.provider = getattr(settings, "rerank_provider", "disabled")
        self._models: dict[str, object] = {}

    def _ensure_fastembed(self, model_name: str) -> bool:
        if model_name in self._models:
            return True
        try:
            from fastembed.rerank.cross_encoder import TextCrossEncoder

            cache_dir = getattr(self.settings, "fastembed_cache_path", None)
            self._models[model_name] = TextCrossEncoder(
                model_name=model_name,
                cache_dir=cache_dir,
                lazy_load=True,
                local_files_only=True,
            )
            logger.info("reranker_ready", model=model_name, cache_dir=cache_dir)
            return True
        except Exception as e:  # pragma: no cover - depends on model download
            logger.warning("reranker_init_failed_fallback_disabled", model=model_name, error=str(e))
            return False

    @property
    def enabled(self) -> bool:
        return self.provider == "fastembed"

    @staticmethod
    def _score_value(score) -> float:
        value = getattr(score, "score", score)
        try:
            return float(value)
        except (TypeError, ValueError):
            return 0.0

    async def rerank(
        self, query: str, results: list, top_k: int, model_name: str | None = None
    ) -> tuple[list, bool]:
        """对候选按 query 语义相关度精排，取 top_k。

        返回 (排序后的结果, reranked 标志)。disabled/异常/空输入时原样返回、标志 False，
        让调用方保留 RRF 粗排顺序而非报错。
        """
        if not self.enabled or not results:
            return results[:top_k], False
        model = model_name or self.settings.rerank_model
        if not self._ensure_fastembed(model):
            return results[:top_k], False
        try:
            import asyncio
            import math

            texts = [r.text for r in results]
            # fastembed 是同步/CPU 实现，丢线程避免阻塞事件循环。
            scores = await asyncio.to_thread(
                lambda: list(self._models[model].rerank(query, texts))
            )
            ranked = sorted(
                zip(results, scores), key=lambda rs: self._score_value(rs[1]), reverse=True
            )
            out = []
            for r, score in ranked[:top_k]:
                # cross-encoder 输出原始 logit（无界、可为负）；套 sigmoid 归一化为
                # 0~1 相关概率，便于展示，且单调不改变排序。
                out.append(r.model_copy(update={
                    "score": round(1.0 / (1.0 + math.exp(-self._score_value(score))), 6)
                }))
            return out, True
        except Exception as e:
            logger.warning("rerank_failed_fallback_rrf", error=str(e))
            return results[:top_k], False

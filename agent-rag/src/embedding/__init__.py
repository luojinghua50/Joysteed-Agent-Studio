"""Embedding client with pluggable providers.

- pseudo:    deterministic hash vectors, no deps (tests/dev). Default.
- fastembed: local ONNX model (e.g. BAAI/bge-small-zh-v1.5), no torch, offline.
- openai:    OpenAI-compatible endpoint via base_url (needs an embedding model).
"""
import hashlib
import structlog

logger = structlog.get_logger()


class Embedder:
    def __init__(self, settings):
        self.settings = settings
        self.provider = settings.embedding_provider
        self.dim = settings.embedding_dim
        self._fastembed = None
        self._openai = None
        if self.provider == "openai":
            self._init_openai()

    def _ensure_fastembed(self) -> bool:
        if self._fastembed is not None:
            return True
        try:
            from fastembed import TextEmbedding

            cache_dir = getattr(self.settings, "fastembed_cache_path", None)
            self._fastembed = TextEmbedding(
                model_name=self.settings.embedding_model,
                cache_dir=cache_dir,
                lazy_load=True,
                local_files_only=True,
            )
            logger.info("fastembed_ready", model=self.settings.embedding_model, cache_dir=cache_dir)
            return True
        except Exception as e:  # pragma: no cover - depends on model download
            logger.warning("fastembed_init_failed_fallback_pseudo", error=str(e))
            self.provider = "pseudo"
            self._fastembed = None
            return False

    def _init_openai(self):
        try:
            from openai import AsyncOpenAI

            self._openai = AsyncOpenAI(
                base_url=self.settings.embedding_base_url,
                api_key=self.settings.embedding_api_key,
            )
        except Exception as e:  # pragma: no cover
            logger.warning("openai_embed_init_failed_fallback_pseudo", error=str(e))
            self.provider = "pseudo"

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        if self.provider == "fastembed" and self._ensure_fastembed():
            # fastembed is sync/CPU; run in a thread to avoid blocking the loop.
            import asyncio

            try:
                vecs = await asyncio.to_thread(lambda: list(self._fastembed.embed(texts)))
                return [v.tolist() for v in vecs]
            except Exception as e:  # pragma: no cover - depends on model/runtime
                logger.warning("fastembed_embed_failed_fallback_pseudo", error=str(e))
                self.provider = "pseudo"
                self._fastembed = None
        if self.provider == "openai" and self._openai:
            resp = await self._openai.embeddings.create(
                model=self.settings.embedding_model, input=texts,
            )
            return [d.embedding for d in resp.data]
        return [self._pseudo_vector(t) for t in texts]

    def _pseudo_vector(self, text: str) -> list[float]:
        """Deterministic fallback vector so the pipeline runs without a model."""
        seed = hashlib.sha256(text.encode("utf-8")).digest()
        vals = [(b / 255.0) for b in seed]  # 32 floats
        return [vals[i % len(vals)] for i in range(self.dim)]

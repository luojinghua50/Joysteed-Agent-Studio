"""Download and verify local fastembed models.

This is intentionally separate from the API request path: production requests use
local_files_only and degrade quickly when the cache is missing. Run this module
after deployment, or manually when model cache volume is empty.
"""

from src.config import RAGSettings


def main() -> None:
    settings = RAGSettings()
    cache_dir = settings.fastembed_cache_path

    if settings.embedding_provider == "fastembed":
        from fastembed import TextEmbedding

        embedder = TextEmbedding(
            model_name=settings.embedding_model,
            cache_dir=cache_dir,
            lazy_load=False,
        )
        # Force ONNX session load and a real inference, not just metadata download.
        list(embedder.embed(["模型预热"]))
        print(f"embedding_ready model={settings.embedding_model} cache_dir={cache_dir}")

    if settings.rerank_provider == "fastembed":
        from fastembed.rerank.cross_encoder import TextCrossEncoder

        reranker = TextCrossEncoder(
            model_name=settings.rerank_model,
            cache_dir=cache_dir,
            lazy_load=False,
        )
        list(reranker.rerank("退款多久到账", ["退款审核通过后，款项将在3到5个工作日内原路退回。"]))
        print(f"reranker_ready model={settings.rerank_model} cache_dir={cache_dir}")


if __name__ == "__main__":
    main()

"""Object storage for raw uploaded files.

Backed by MinIO/S3 in production. When disabled (no MinIO available, e.g. tests
or local dev), falls back to an in-memory store so the service still runs.
"""
import structlog

logger = structlog.get_logger()


class ObjectStore:
    def __init__(self, settings):
        self.settings = settings
        self.enabled = settings.minio_enabled
        self._client = None
        self._mem: dict[str, bytes] = {}
        if self.enabled:
            self._init_client()

    def _init_client(self):
        try:
            from minio import Minio

            self._client = Minio(
                self.settings.minio_endpoint,
                access_key=self.settings.minio_access_key,
                secret_key=self.settings.minio_secret_key,
                secure=self.settings.minio_secure,
            )
            bucket = self.settings.minio_bucket
            if not self._client.bucket_exists(bucket):
                self._client.make_bucket(bucket)
        except Exception as e:  # pragma: no cover - depends on external service
            logger.warning("minio_init_failed_fallback_memory", error=str(e))
            self.enabled = False
            self._client = None

    @staticmethod
    def build_key(tenant_id: str, kb_id: str, doc_id: str, version_no: int, filename: str) -> str:
        return f"{tenant_id}/{kb_id}/{doc_id}/v{version_no}/{filename}"

    async def put(self, key: str, data: bytes, content_type: str = "application/octet-stream") -> str:
        if self.enabled and self._client:
            import io

            self._client.put_object(
                self.settings.minio_bucket, key, io.BytesIO(data), length=len(data),
                content_type=content_type,
            )
        else:
            self._mem[key] = data
        return key

    async def get(self, key: str) -> bytes | None:
        if self.enabled and self._client:
            try:
                resp = self._client.get_object(self.settings.minio_bucket, key)
                return resp.read()
            except Exception:  # pragma: no cover
                return None
        return self._mem.get(key)

    async def delete_prefix(self, prefix: str) -> int:
        """Delete all objects under a prefix (e.g. a doc's all versions)."""
        if self.enabled and self._client:
            from minio.deleteobjects import DeleteObject

            objs = self._client.list_objects(self.settings.minio_bucket, prefix=prefix, recursive=True)
            to_delete = [DeleteObject(o.object_name) for o in objs]
            errors = list(self._client.remove_objects(self.settings.minio_bucket, to_delete))
            return len(to_delete) - len(errors)
        removed = [k for k in self._mem if k.startswith(prefix)]
        for k in removed:
            del self._mem[k]
        return len(removed)

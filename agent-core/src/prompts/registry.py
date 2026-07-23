from pathlib import Path

from jinja2 import Template
from src.prompts.models import PromptTemplate, PromptVersion


class PromptRegistry:
    """Centralized prompt management registry."""

    def __init__(self):
        self._store: dict[str, PromptTemplate] = {}
        self._versions: dict[str, list[PromptVersion]] = {}

    # 默认渲染变量：保证未提供时 {% if x %} 块干净折叠、{{ x }} 不以字面量泄漏
    _DEFAULT_VARS = {"company_name": "我们", "memory_context": "", "error_context": ""}

    async def get(self, prompt_id: str, variables: dict | None = None) -> str:
        """Get a prompt by ID, rendering Jinja variables with safe defaults."""
        prompt = self._store.get(prompt_id)
        if not prompt:
            raise KeyError(f"Prompt '{prompt_id}' not found")

        render_vars = {**self._DEFAULT_VARS, **(variables or {})}
        return self._render(prompt.content, render_vars)

    async def load_dir(self, base_dir: str | Path) -> int:
        """从目录批量加载 .md prompt，prompt_id = 相对路径去掉后缀。

        例：prompts/agents/faq_system.md → "agents/faq_system"。
        返回成功加载的数量。已存在的同名 prompt 会被覆盖。
        """
        return self.load_dir_sync(base_dir)

    def load_dir_sync(self, base_dir: str | Path) -> int:
        """load_dir 的同步版本，供同步上下文（如 create_app）使用。"""
        base = Path(base_dir)
        if not base.is_dir():
            return 0

        count = 0
        for path in sorted(base.rglob("*.md")):
            prompt_id = path.relative_to(base).with_suffix("").as_posix()
            content = path.read_text(encoding="utf-8")
            category = prompt_id.split("/")[0] if "/" in prompt_id else "general"
            self._register_sync(PromptTemplate(
                id=prompt_id,
                name=prompt_id,
                category=category,
                content=content,
            ))
            count += 1
        return count

    def _register_sync(self, prompt: PromptTemplate):
        self._store[prompt.id] = prompt
        self._versions[prompt.id] = [PromptVersion(
            prompt_id=prompt.id,
            version=prompt.version,
            content=prompt.content,
            change_reason="initial registration",
        )]

    async def register(self, prompt: PromptTemplate):
        """Register a new prompt."""
        self._register_sync(prompt)

    async def update(self, prompt_id: str, content: str, reason: str, author: str = "system"):
        """Update a prompt, creating a new version."""
        prompt = self._store.get(prompt_id)
        if not prompt:
            raise KeyError(f"Prompt '{prompt_id}' not found")

        prompt.version += 1
        prompt.content = content
        from datetime import datetime
        prompt.updated_at = datetime.now()

        version = PromptVersion(
            prompt_id=prompt_id,
            version=prompt.version,
            content=content,
            change_reason=reason,
            author=author,
        )
        if prompt_id not in self._versions:
            self._versions[prompt_id] = []
        self._versions[prompt_id].append(version)

    async def rollback(self, prompt_id: str, target_version: int):
        """Rollback to a specific version."""
        versions = self._versions.get(prompt_id, [])
        for v in versions:
            if v.version == target_version:
                prompt = self._store.get(prompt_id)
                if prompt:
                    prompt.content = v.content
                    prompt.version = target_version
                return
        raise ValueError(f"Version {target_version} not found for prompt '{prompt_id}'")

    async def list_all(self) -> list[PromptTemplate]:
        """List all registered prompts."""
        return list(self._store.values())

    def _render(self, template: str, variables: dict) -> str:
        """Render Jinja2 template variables."""
        return Template(template).render(**variables)

from typing import Any


class SkillResult:
    """Standard result container for skill execution."""

    def __init__(self, success: bool, data: Any = None, error: str | None = None):
        self.success = success
        self.data = data
        self.error = error

    def to_str(self) -> str:
        if self.success:
            return str(self.data)
        return f"操作失败: {self.error}"


class BaseSkill:
    """Base class for all skills."""

    name: str = "base_skill"
    description: str = ""

    async def execute(self, **kwargs) -> SkillResult:
        raise NotImplementedError

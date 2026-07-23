from pydantic import BaseModel, Field
from datetime import datetime


class PromptTemplate(BaseModel):
    """Prompt template data model."""

    id: str
    name: str
    category: str
    agent: str | None = None
    content: str
    version: int = 1
    variables: list[str] = Field(default_factory=list)
    metadata: dict = Field(default_factory=dict)
    status: str = "active"
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)


class PromptVersion(BaseModel):
    """Versioned prompt record."""

    prompt_id: str
    version: int
    content: str
    change_reason: str = ""
    author: str = "system"
    eval_score: float | None = None
    created_at: datetime = Field(default_factory=datetime.now)

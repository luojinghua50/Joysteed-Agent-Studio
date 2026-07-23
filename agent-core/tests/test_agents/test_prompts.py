import pytest
from src.prompts.registry import PromptRegistry
from src.prompts.models import PromptTemplate


@pytest.mark.asyncio
async def test_register_and_get():
    registry = PromptRegistry()
    prompt = PromptTemplate(
        id="test_prompt",
        name="Test Prompt",
        category="system",
        content="Hello {{ name }}, welcome!",
        variables=["name"],
    )
    await registry.register(prompt)

    result = await registry.get("test_prompt", {"name": "张三"})
    assert result == "Hello 张三, welcome!"


@pytest.mark.asyncio
async def test_get_without_variables():
    registry = PromptRegistry()
    prompt = PromptTemplate(
        id="simple",
        name="Simple",
        category="system",
        content="You are a helpful assistant.",
    )
    await registry.register(prompt)

    result = await registry.get("simple")
    assert result == "You are a helpful assistant."


@pytest.mark.asyncio
async def test_get_nonexistent_raises():
    registry = PromptRegistry()
    with pytest.raises(KeyError):
        await registry.get("nonexistent")


@pytest.mark.asyncio
async def test_update_creates_new_version():
    registry = PromptRegistry()
    prompt = PromptTemplate(
        id="versioned",
        name="Versioned Prompt",
        category="system",
        content="Version 1",
    )
    await registry.register(prompt)

    await registry.update("versioned", "Version 2", "improved wording", "dev")
    result = await registry.get("versioned")
    assert result == "Version 2"


@pytest.mark.asyncio
async def test_rollback():
    registry = PromptRegistry()
    prompt = PromptTemplate(
        id="rollback_test",
        name="Rollback Test",
        category="system",
        content="Original",
        version=1,
    )
    await registry.register(prompt)
    await registry.update("rollback_test", "Updated", "update", "dev")

    await registry.rollback("rollback_test", 1)
    result = await registry.get("rollback_test")
    assert result == "Original"


@pytest.mark.asyncio
async def test_list_all():
    registry = PromptRegistry()
    await registry.register(PromptTemplate(
        id="p1", name="P1", category="system", content="content1"
    ))
    await registry.register(PromptTemplate(
        id="p2", name="P2", category="skill", content="content2"
    ))

    all_prompts = await registry.list_all()
    assert len(all_prompts) == 2
    ids = {p.id for p in all_prompts}
    assert ids == {"p1", "p2"}

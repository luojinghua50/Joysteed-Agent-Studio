import pytest


@pytest.fixture
def settings():
    from src.config import Settings
    return Settings()


@pytest.fixture
def stability_config():
    from src.config import StabilityConfig
    return StabilityConfig()


@pytest.fixture
def reflection_config():
    from src.config import ReflectionConfig
    return ReflectionConfig()


@pytest.fixture
def memory_manager():
    from src.memory.manager import MemoryManager
    return MemoryManager()


@pytest.fixture
def working_memory():
    from src.memory.working import WorkingMemory
    return WorkingMemory()


@pytest.fixture
def prompt_registry():
    from src.prompts.registry import PromptRegistry
    return PromptRegistry()

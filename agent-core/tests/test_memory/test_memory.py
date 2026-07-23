import pytest
from src.memory.working import WorkingMemory
from src.memory.manager import MemoryManager, format_memory_for_prompt
from src.memory.long_term.profile import ProfileMemory, UserProfile
from src.memory.long_term.episodic import EpisodicMemory
from src.memory.long_term.semantic import SemanticMemory


@pytest.mark.asyncio
async def test_working_memory_set_and_get():
    wm = WorkingMemory()
    await wm.set_entity("session1", "order_id", "ORD-001")
    await wm.set_entity("session1", "amount", 299.0)

    entities = await wm.get("session1")
    assert entities["order_id"] == "ORD-001"
    assert entities["amount"] == 299.0


@pytest.mark.asyncio
async def test_working_memory_get_entity():
    wm = WorkingMemory()
    await wm.set_entity("session1", "phone", "13800001234")

    result = await wm.get_entity("session1", "phone")
    assert result == "13800001234"

    result = await wm.get_entity("session1", "nonexistent")
    assert result is None


@pytest.mark.asyncio
async def test_working_memory_clear():
    wm = WorkingMemory()
    await wm.set_entity("session1", "key", "value")
    await wm.clear("session1")

    entities = await wm.get("session1")
    assert entities == {}


@pytest.mark.asyncio
async def test_working_memory_isolated_sessions():
    wm = WorkingMemory()
    await wm.set_entity("s1", "key", "value1")
    await wm.set_entity("s2", "key", "value2")

    assert (await wm.get("s1"))["key"] == "value1"
    assert (await wm.get("s2"))["key"] == "value2"


@pytest.mark.asyncio
async def test_profile_memory_get_creates_default():
    pm = ProfileMemory()
    profile = await pm.get("new_customer")
    assert profile.customer_id == "new_customer"
    assert profile.vip_level == 0


@pytest.mark.asyncio
async def test_profile_memory_save_and_get():
    pm = ProfileMemory()
    profile = UserProfile(customer_id="C001", vip_level=2, communication_style="简洁")
    await pm.save(profile)

    loaded = await pm.get("C001")
    assert loaded.vip_level == 2
    assert loaded.communication_style == "简洁"


@pytest.mark.asyncio
async def test_profile_memory_update_from_conversation():
    pm = ProfileMemory()
    await pm.save(UserProfile(customer_id="C001"))

    await pm.update_from_conversation("C001", {
        "communication_style": "详细",
        "sensitive_points": ["配送延迟"],
    })

    profile = await pm.get("C001")
    assert profile.communication_style == "详细"
    assert "配送延迟" in profile.sensitive_points


@pytest.mark.asyncio
async def test_episodic_memory_save_and_search():
    em = EpisodicMemory()
    await em.save_episode("s1", "C001", [])
    await em.save_episode("s2", "C001", [])

    results = await em.search("query", "C001", top_k=5)
    assert len(results) == 2


@pytest.mark.asyncio
async def test_semantic_memory_facts():
    sm = SemanticMemory()
    await sm.set_fact("C001", "address", "北京市朝阳区")
    await sm.set_fact("C001", "phone", "13800001234")

    facts = await sm.get_facts("C001")
    assert facts["address"] == "北京市朝阳区"
    assert facts["phone"] == "13800001234"


@pytest.mark.asyncio
async def test_semantic_memory_delete_all():
    sm = SemanticMemory()
    await sm.set_fact("C001", "key", "value")
    await sm.delete_all("C001")
    facts = await sm.get_facts("C001")
    assert facts == {}


@pytest.mark.asyncio
async def test_memory_manager_load_context():
    mm = MemoryManager()
    context = await mm.load_context("C001", "session1", "查询订单")
    assert "profile" in context
    assert "relevant_history" in context
    assert "facts" in context
    assert "entities" in context


@pytest.mark.asyncio
async def test_memory_manager_session_end():
    mm = MemoryManager()
    await mm.working.set_entity("session1", "order_id", "ORD-001")
    await mm.on_session_end("C001", "session1", [])

    entities = await mm.working.get("session1")
    assert entities == {}

    episodes = await mm.episodic.search("", "C001")
    assert len(episodes) == 1


@pytest.mark.asyncio
async def test_memory_manager_purge():
    mm = MemoryManager()
    await mm.profile.save(UserProfile(customer_id="C001", vip_level=3))
    await mm.semantic.set_fact("C001", "key", "value")
    await mm.episodic.save_episode("s1", "C001", [])

    await mm.purge_customer_memory("C001")

    profile = await mm.profile.get("C001")
    assert profile.vip_level == 0  # reset to default
    facts = await mm.semantic.get_facts("C001")
    assert facts == {}


def test_format_memory_for_prompt_empty():
    memory = {"profile": None, "relevant_history": [], "facts": {}, "entities": {}}
    result = format_memory_for_prompt(memory)
    assert result == ""


def test_format_memory_for_prompt_with_profile():
    profile = UserProfile(customer_id="C001", vip_level=2, communication_style="简洁")
    memory = {"profile": profile, "relevant_history": [], "facts": {}, "entities": {}}
    result = format_memory_for_prompt(memory)
    assert "VIP等级: 2" in result
    assert "简洁" in result

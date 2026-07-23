import json

from src.memory.working import WorkingMemory
from src.memory.long_term.profile import ProfileMemory, UserProfile
from src.memory.long_term.episodic import EpisodicMemory
from src.memory.long_term.semantic import SemanticMemory


class MemoryManager:
    """Unified memory management entry point."""

    def __init__(
        self,
        working: WorkingMemory | None = None,
        profile: ProfileMemory | None = None,
        episodic: EpisodicMemory | None = None,
        semantic: SemanticMemory | None = None,
    ):
        self.working = working or WorkingMemory()
        self.profile = profile or ProfileMemory()
        self.episodic = episodic or EpisodicMemory()
        self.semantic = semantic or SemanticMemory()

    async def load_context(
        self, customer_id: str, session_id: str, current_message: str
    ) -> dict:
        """Load all relevant memory context before agent execution."""
        profile = await self.profile.get(customer_id)
        history = await self.episodic.search(current_message, customer_id, top_k=3)
        facts = await self.semantic.get_facts(customer_id)
        entities = await self.working.get(session_id)

        return {
            "profile": profile,
            "relevant_history": history,
            "facts": facts,
            "entities": entities,
        }

    async def on_session_end(
        self, customer_id: str, session_id: str, messages: list
    ):
        """Archive session to long-term memory on session end."""
        await self.episodic.save_episode(session_id, customer_id, messages)
        await self.working.clear(session_id)

    async def update_fact(
        self, customer_id: str, field: str, value: str, source: str = "agent_inferred"
    ):
        """Update a fact in semantic memory."""
        await self.semantic.set_fact(customer_id, field, value)

    async def purge_customer_memory(self, customer_id: str):
        """GDPR compliance: delete all memory for a customer."""
        await self.profile.delete(customer_id)
        await self.episodic.delete_all(customer_id)
        await self.semantic.delete_all(customer_id)


def format_memory_for_prompt(memory: dict) -> str:
    """Format memory context as a prompt fragment for agent consumption."""
    parts = []

    profile = memory.get("profile")
    if profile:
        parts.append(
            f"## 用户信息\n"
            f"VIP等级: {profile.vip_level}, "
            f"沟通风格: {profile.communication_style or '未知'}"
        )
        if profile.sensitive_points:
            parts.append(f"注意事项: {', '.join(profile.sensitive_points)}")

    history = memory.get("relevant_history", [])
    if history:
        parts.append("## 相关历史")
        for h in history:
            ts = h.timestamp.strftime("%m-%d") if h.timestamp else "未知"
            parts.append(f"- [{ts}] {h.summary} (结果: {h.resolution})")

    entities = memory.get("entities", {})
    if entities:
        parts.append(f"## 当前会话实体\n{json.dumps(entities, ensure_ascii=False)}")

    return "\n".join(parts)

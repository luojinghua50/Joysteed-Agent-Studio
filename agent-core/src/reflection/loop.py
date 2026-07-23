from langchain_core.messages import SystemMessage

from src.reflection.judge import JudgeReflector, JudgeResult
from src.reflection.error_memory import ErrorMemoryStore, ErrorRecord
from src.config import ReflectionConfig
from datetime import datetime


class ReflectiveAgentLoop:
    """Agent execution loop with optional reflection (self-check or judge)."""

    def __init__(
        self,
        config: ReflectionConfig | None = None,
        error_memory: ErrorMemoryStore | None = None,
        judge: JudgeReflector | None = None,
    ):
        self.config = config or ReflectionConfig()
        self.error_memory = error_memory or ErrorMemoryStore(
            max_size=self.config.error_memory_size
        )
        self.judge = judge or JudgeReflector(config=self.config)

    async def execute(
        self,
        agent_name: str,
        skill_name: str | None,
        agent_executor,
        state: dict,
    ) -> dict:
        """Execute agent with reflection based on configured policy."""
        policy = self._get_policy(agent_name, skill_name)

        if policy == "off":
            return await agent_executor.ainvoke(state)

        for attempt in range(self.config.max_retries + 1):
            enriched_state = await self._enrich_with_error_memory(state, agent_name)
            result = await agent_executor.ainvoke(enriched_state)
            agent_response = self._extract_response(result)

            if policy == "self_check":
                return result

            if policy == "judge":
                judge_result = await self.judge.evaluate(
                    user_message=self._get_last_user_msg(state),
                    tool_results=[],
                    agent_response=agent_response,
                )

                if judge_result.passed:
                    return result

                await self.error_memory.add_error(ErrorRecord(
                    timestamp=datetime.now(),
                    agent=agent_name,
                    skill=skill_name,
                    user_message=self._get_last_user_msg(state),
                    failed_response=agent_response,
                    issues=judge_result.issues,
                    suggestion=judge_result.suggestion,
                    retry_count=attempt,
                ))

                if attempt == self.config.max_retries:
                    return result

                state = self._inject_feedback(state, judge_result)

        return result

    def _get_policy(self, agent_name: str, skill_name: str | None) -> str:
        """Get reflection policy: skill-level overrides agent-level."""
        if not self.config.enabled:
            return "off"
        if skill_name and skill_name in self.config.skill_policies:
            return self.config.skill_policies[skill_name]
        return self.config.agent_policies.get(agent_name, "off")

    def _inject_feedback(self, state: dict, judge_result: JudgeResult) -> dict:
        """Inject judge feedback into state for retry."""
        feedback_msg = SystemMessage(content=(
            f"上一轮回复被质量检查拒绝。\n"
            f"问题：{'; '.join(judge_result.issues)}\n"
            f"修正建议：{judge_result.suggestion}\n"
            f"请据此重新生成回复。"
        ))
        messages = state.get("messages", [])
        state["messages"] = messages + [feedback_msg]
        return state

    async def _enrich_with_error_memory(self, state: dict, agent_name: str) -> dict:
        """Inject historical error context into state."""
        error_context = await self.error_memory.get_error_context(agent_name)
        forbidden = await self.error_memory.get_forbidden_patterns(agent_name)

        if error_context or forbidden:
            extra = error_context
            if forbidden:
                extra += "\n## 禁止事项（历史验证的错误模式）\n"
                for p in forbidden:
                    extra += f"- 禁止：{p}\n"
            messages = state.get("messages", [])
            state["messages"] = [SystemMessage(content=extra)] + messages

        return state

    def _extract_response(self, result: dict) -> str:
        """Extract the response text from agent result."""
        messages = result.get("messages", [])
        if messages:
            last = messages[-1]
            return getattr(last, "content", str(last))
        return ""

    def _get_last_user_msg(self, state: dict) -> str:
        """Get the last user message from state."""
        messages = state.get("messages", [])
        for msg in reversed(messages):
            role = getattr(msg, "type", None)
            if role == "human":
                return msg.content
        return ""

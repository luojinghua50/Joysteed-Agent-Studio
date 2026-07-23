import asyncio
import structlog

from src.config import StabilityConfig
from src.guardrails.loop_protection import LoopGuard
from src.guardrails.fallback import FallbackHandler
from src.guardrails.timeout import AgentTimeoutError

logger = structlog.get_logger()


class GuardrailEngine:
    """Unified stability engine wrapping agent execution."""

    def __init__(self, config: StabilityConfig | None = None):
        if config is None:
            config = StabilityConfig()
        self.config = config
        self.loop_guard = LoopGuard(
            max_tool_calls=config.max_tool_calls_per_turn,
            max_routing_loops=config.max_routing_loops,
            max_agent_hops=config.max_agent_hops,
            max_skill_steps=config.max_skill_steps,
        )
        self.fallback = FallbackHandler(config)

    async def execute_safe(
        self,
        agent_executor,
        state: dict,
        session_id: str,
        user_id: str,
        is_vip: bool = False,
    ) -> dict:
        """Safely execute an agent with all guardrails applied."""
        if not self.config.enabled:
            return await agent_executor.ainvoke(state)

        try:
            timeout = self.config.session_timeout if self.config.timeout_enabled else None

            if timeout:
                result = await asyncio.wait_for(
                    self._execute_with_guards(agent_executor, state, session_id),
                    timeout=timeout,
                )
            else:
                result = await self._execute_with_guards(agent_executor, state, session_id)

            return result

        except asyncio.TimeoutError:
            logger.warning("session_timeout", session_id=session_id)
            return await self.fallback.handle_failure(
                state, AgentTimeoutError("session_timeout"), "session_timeout"
            )
        except Exception as e:
            logger.error("unexpected_error", session_id=session_id, error=str(e))
            return await self.fallback.handle_failure(state, e, "unexpected_error")

    async def _execute_with_guards(self, agent_executor, state: dict, session_id: str) -> dict:
        """Internal execution with loop protection."""
        if self.config.loop_protection_enabled:
            self.loop_guard.reset()

        result = await agent_executor.ainvoke(state)

        if self.config.loop_protection_enabled:
            should_end, reason = self.loop_guard.should_force_end()
            if should_end:
                logger.warning("loop_protection_triggered", reason=reason)
                return await self.fallback.handle_failure(
                    state, RuntimeError(f"Loop limit: {reason}"), f"loop_{reason}"
                )

        return result

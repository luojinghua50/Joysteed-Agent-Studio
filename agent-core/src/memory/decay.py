import math
from datetime import datetime


class MemoryDecay:
    """Time-based memory decay calculations."""

    EPISODIC_DECAY_RATE = 0.01
    FACT_DECAY_RATE = 0.005

    def episodic_relevance(self, semantic_score: float, days_ago: int) -> float:
        """Compute episodic memory relevance with time decay."""
        decay = math.exp(-self.EPISODIC_DECAY_RATE * days_ago)
        return semantic_score * decay

    def fact_confidence(self, initial_confidence: float, days_since_update: int) -> float:
        """Compute fact confidence with time decay."""
        decayed = initial_confidence * math.exp(-self.FACT_DECAY_RATE * days_since_update)
        return max(decayed, 0.1)

    def is_stale(self, days_since_update: int, threshold: int = 90) -> bool:
        """Check if a profile field is stale."""
        return days_since_update > threshold


class ConfidenceManager:
    """Manages confidence scores for memory facts."""

    SOURCE_SCORES = {
        "user_explicit": 1.0,
        "tool_result": 0.95,
        "crm": 0.9,
        "agent_inferred": 0.6,
        "historical": 0.5,
    }

    def __init__(self, decay_rate: float = 0.005):
        self.decay_rate = decay_rate

    def initial_confidence(self, source: str) -> float:
        return self.SOURCE_SCORES.get(source, 0.5)

    def current_confidence(self, initial: float, days_since_update: int) -> float:
        decayed = initial * math.exp(-self.decay_rate * days_since_update)
        return max(decayed, 0.1)

    def should_reverify(self, confidence: float, threshold: float = 0.3) -> bool:
        return confidence < threshold

    def should_ask_user(self, confidence: float, threshold: float = 0.4) -> bool:
        return confidence < threshold


def format_fact_for_prompt(key: str, value: str, confidence: float) -> str:
    """按置信度分档标注事实，避免 Agent 把低置信度信息当确定事实用。

    ≥0.8 直出；≥0.4 标"可能已变更"；<0.4 标"待确认核实"。
    """
    if confidence >= 0.8:
        return f"- {key}: {value}"
    if confidence >= 0.4:
        return f"- {key}: {value}（历史记录，可能已变更）"
    return f"- {key}: {value}（待确认，请向用户核实）"

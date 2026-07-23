import pytest
from src.memory.decay import MemoryDecay, ConfidenceManager


class TestMemoryDecay:
    def test_episodic_relevance_no_decay(self):
        decay = MemoryDecay()
        result = decay.episodic_relevance(0.9, 0)
        assert result == pytest.approx(0.9, rel=1e-3)

    def test_episodic_relevance_with_decay(self):
        decay = MemoryDecay()
        result = decay.episodic_relevance(0.9, 70)
        assert result < 0.9
        assert result > 0

    def test_episodic_relevance_high_decay(self):
        decay = MemoryDecay()
        result_recent = decay.episodic_relevance(0.9, 1)
        result_old = decay.episodic_relevance(0.9, 365)
        assert result_recent > result_old

    def test_fact_confidence_no_decay(self):
        decay = MemoryDecay()
        result = decay.fact_confidence(1.0, 0)
        assert result == pytest.approx(1.0, rel=1e-3)

    def test_fact_confidence_with_decay(self):
        decay = MemoryDecay()
        result = decay.fact_confidence(1.0, 100)
        assert result < 1.0
        assert result >= 0.1

    def test_fact_confidence_minimum(self):
        decay = MemoryDecay()
        result = decay.fact_confidence(0.5, 10000)
        assert result >= 0.1

    def test_is_stale_true(self):
        decay = MemoryDecay()
        assert decay.is_stale(91) is True

    def test_is_stale_false(self):
        decay = MemoryDecay()
        assert decay.is_stale(89) is False

    def test_is_stale_custom_threshold(self):
        decay = MemoryDecay()
        assert decay.is_stale(31, threshold=30) is True
        assert decay.is_stale(29, threshold=30) is False


class TestConfidenceManager:
    def test_initial_confidence_user_explicit(self):
        cm = ConfidenceManager()
        assert cm.initial_confidence("user_explicit") == 1.0

    def test_initial_confidence_tool_result(self):
        cm = ConfidenceManager()
        assert cm.initial_confidence("tool_result") == 0.95

    def test_initial_confidence_unknown_source(self):
        cm = ConfidenceManager()
        assert cm.initial_confidence("unknown") == 0.5

    def test_current_confidence_decay(self):
        cm = ConfidenceManager()
        result = cm.current_confidence(1.0, 30)
        assert result < 1.0
        assert result > 0.5

    def test_should_reverify(self):
        cm = ConfidenceManager()
        assert cm.should_reverify(0.2) is True
        assert cm.should_reverify(0.5) is False

    def test_should_ask_user(self):
        cm = ConfidenceManager()
        assert cm.should_ask_user(0.3) is True
        assert cm.should_ask_user(0.5) is False

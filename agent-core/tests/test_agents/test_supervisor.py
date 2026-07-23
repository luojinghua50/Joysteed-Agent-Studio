import pytest
from src.agents.supervisor import _parse_intent, _intent_to_agent, route_by_intent


class TestParseIntent:
    def test_parse_faq(self):
        assert _parse_intent("faq") == "faq"
        assert _parse_intent("FAQ") == "faq"
        assert _parse_intent("  faq  ") == "faq"

    def test_parse_order(self):
        assert _parse_intent("order") == "order"
        assert _parse_intent("ORDER") == "order"

    def test_parse_complaint(self):
        assert _parse_intent("complaint") == "complaint"

    def test_parse_tech_support(self):
        assert _parse_intent("tech_support") == "tech_support"

    def test_parse_human(self):
        assert _parse_intent("human") == "human"

    def test_parse_unknown_defaults_to_human(self):
        assert _parse_intent("unknown") == "human"
        assert _parse_intent("") == "human"
        assert _parse_intent("random text") == "human"

    def test_parse_intent_in_sentence(self):
        assert _parse_intent("The intent is order") == "order"
        assert _parse_intent("I think this is a complaint") == "complaint"


class TestIntentToAgent:
    def test_mapping(self):
        assert _intent_to_agent("faq") == "faq"
        assert _intent_to_agent("order") == "order"
        assert _intent_to_agent("complaint") == "complaint"
        assert _intent_to_agent("tech_support") == "tech_support"
        assert _intent_to_agent("human") == "human_handoff"

    def test_unknown_defaults_to_human_handoff(self):
        assert _intent_to_agent("unknown") == "human_handoff"


class TestRouteByIntent:
    def test_route_faq(self):
        state = {"intent": "faq", "messages": [], "customer_id": "C001"}
        assert route_by_intent(state) == "faq"

    def test_route_order(self):
        state = {"intent": "order", "messages": [], "customer_id": "C001"}
        assert route_by_intent(state) == "order"

    def test_route_complaint(self):
        state = {"intent": "complaint", "messages": [], "customer_id": "C001"}
        assert route_by_intent(state) == "complaint"

    def test_route_tech_support(self):
        state = {"intent": "tech_support", "messages": [], "customer_id": "C001"}
        assert route_by_intent(state) == "tech_support"

    def test_route_human(self):
        state = {"intent": "human", "messages": [], "customer_id": "C001"}
        assert route_by_intent(state) == "human_handoff"

    def test_route_none_defaults_to_human(self):
        state = {"intent": None, "messages": [], "customer_id": "C001"}
        assert route_by_intent(state) == "human_handoff"

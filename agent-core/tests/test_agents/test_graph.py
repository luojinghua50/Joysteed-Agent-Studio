from src.agents.graph import _extra_body_for_model


def test_qwen3_disables_ollama_thinking():
    assert _extra_body_for_model("qwen3:14b") == {"think": False}


def test_non_qwen_models_do_not_get_ollama_options():
    assert _extra_body_for_model("claude-sonnet-4-6") == {}
    assert _extra_body_for_model("gpt-5.5") == {}

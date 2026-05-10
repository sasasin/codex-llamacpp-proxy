from codex_llamacpp_proxy.proxy import responses_usage_from_chat_usage


def test_responses_usage_defaults_when_chat_usage_is_missing() -> None:
    assert responses_usage_from_chat_usage(None) == {
        "input_tokens": 0,
        "input_tokens_details": {"cached_tokens": 0},
        "output_tokens": 0,
        "output_tokens_details": {"reasoning_tokens": 0},
        "total_tokens": 0,
    }

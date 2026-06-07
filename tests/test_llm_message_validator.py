"""Tests for llm-message-validator."""

from __future__ import annotations

import pytest

from llm_message_validator import ValidationError, ValidationResult, validate

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def ok_msgs():
    return [
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi"},
    ]


# ---------------------------------------------------------------------------
# ValidationResult
# ---------------------------------------------------------------------------


def test_result_valid_no_errors():
    result = ValidationResult()
    assert result.valid is True
    assert bool(result) is True


def test_result_invalid_with_errors():
    err = ValidationError("code", "message")
    result = ValidationResult(errors=[err])
    assert result.valid is False
    assert bool(result) is False


def test_result_errors_empty_by_default():
    result = ValidationResult()
    assert result.errors == []


# ---------------------------------------------------------------------------
# ValidationError
# ---------------------------------------------------------------------------


def test_error_str_includes_code_and_message():
    err = ValidationError("missing_role", "Role is missing.")
    s = str(err)
    assert "missing_role" in s
    assert "Role is missing." in s


def test_error_str_with_index():
    err = ValidationError("missing_role", "Role is missing.", index=2)
    s = str(err)
    assert "2" in s


def test_error_no_index():
    err = ValidationError("code", "msg")
    assert err.index is None


# ---------------------------------------------------------------------------
# Generic rules
# ---------------------------------------------------------------------------


def test_valid_simple_conversation():
    result = validate(ok_msgs())
    assert result.valid


def test_empty_list_error():
    result = validate([])
    assert not result.valid
    codes = [e.code for e in result.errors]
    assert "empty_list" in codes


def test_not_a_dict():
    result = validate(["not a dict"])
    assert not result.valid
    codes = [e.code for e in result.errors]
    assert "not_a_dict" in codes


def test_missing_role():
    result = validate([{"content": "hi"}])
    assert not result.valid
    assert any(e.code == "missing_role" for e in result.errors)


def test_missing_content():
    result = validate([{"role": "user"}])
    assert not result.valid
    assert any(e.code == "missing_content" for e in result.errors)


def test_empty_string_content():
    result = validate([{"role": "user", "content": ""}])
    assert not result.valid
    assert any(e.code == "empty_content" for e in result.errors)


def test_empty_list_content():
    result = validate([{"role": "user", "content": []}])
    assert not result.valid
    assert any(e.code == "empty_content_list" for e in result.errors)


def test_list_content_non_empty_valid():
    result = validate([{"role": "user", "content": [{"type": "text", "text": "hi"}]}])
    assert result.valid


def test_consecutive_user_messages():
    msgs = [
        {"role": "user", "content": "Hello"},
        {"role": "user", "content": "How are you?"},
    ]
    result = validate(msgs)
    assert not result.valid
    assert any(e.code == "consecutive_same_role" for e in result.errors)


def test_consecutive_assistant_messages():
    msgs = [
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi"},
        {"role": "assistant", "content": "What's up?"},
    ]
    result = validate(msgs)
    assert not result.valid
    assert any(e.code == "consecutive_same_role" for e in result.errors)


def test_system_then_user_ok():
    msgs = [
        {"role": "system", "content": "You are helpful."},
        {"role": "user", "content": "Hello"},
    ]
    result = validate(msgs)
    assert result.valid


def test_unknown_role_allowed_generically():
    # Generic rules allow unknown roles
    msgs = [{"role": "custom_role", "content": "hi"}]
    result = validate(msgs)
    # No error for the role itself in generic mode
    assert not any(e.code == "invalid_role" for e in result.errors)


def test_invalid_role_type():
    result = validate([{"role": 123, "content": "hi"}])
    assert not result.valid
    assert any(e.code == "invalid_role_type" for e in result.errors)


def test_error_has_correct_index():
    msgs = [
        {"role": "user", "content": "Hello"},
        {"content": "No role"},
    ]
    result = validate(msgs)
    errors_at_1 = [e for e in result.errors if e.index == 1]
    assert len(errors_at_1) > 0


def test_multiple_errors_collected():
    msgs = [
        {"content": "no role"},
        {"role": "assistant"},  # no content
    ]
    result = validate(msgs)
    assert len(result.errors) >= 2


def test_strict_raises_on_error():
    with pytest.raises(ValidationError):
        validate([], strict=True)


def test_strict_no_raise_on_valid():
    result = validate(ok_msgs(), strict=True)
    assert result.valid


def test_unknown_provider_raises():
    with pytest.raises(ValueError, match="Unknown provider"):
        validate(ok_msgs(), provider="unsupported")


# ---------------------------------------------------------------------------
# Anthropic provider
# ---------------------------------------------------------------------------


def test_anthropic_valid():
    result = validate(ok_msgs(), provider="anthropic")
    assert result.valid


def test_anthropic_system_in_messages_error():
    msgs = [
        {"role": "system", "content": "Be helpful."},
        {"role": "user", "content": "Hello"},
    ]
    result = validate(msgs, provider="anthropic")
    assert any(e.code == "system_in_messages" for e in result.errors)


def test_anthropic_invalid_role():
    msgs = [{"role": "user", "content": "Hi"}, {"role": "tool", "content": "result"}]
    result = validate(msgs, provider="anthropic")
    assert any(e.code == "invalid_role" for e in result.errors)


def test_anthropic_must_start_with_user():
    msgs = [{"role": "assistant", "content": "Hi"}]
    result = validate(msgs, provider="anthropic")
    assert any(e.code == "must_start_with_user" for e in result.errors)


def test_anthropic_tool_use_with_result():
    msgs = [
        {
            "role": "user",
            "content": "Search for cats",
        },
        {
            "role": "assistant",
            "content": [
                {
                    "type": "tool_use",
                    "id": "tool_123",
                    "name": "search",
                    "input": {"query": "cats"},
                }
            ],
        },
        {
            "role": "user",
            "content": [{"type": "tool_result", "tool_use_id": "tool_123", "content": "..."}],
        },
    ]
    result = validate(msgs, provider="anthropic")
    assert result.valid, result.errors


def test_anthropic_unmatched_tool_use():
    msgs = [
        {"role": "user", "content": "Search"},
        {
            "role": "assistant",
            "content": [{"type": "tool_use", "id": "tool_abc", "name": "search", "input": {}}],
        },
        # Missing tool_result
    ]
    result = validate(msgs, provider="anthropic")
    assert any(e.code == "unmatched_tool_use" for e in result.errors)


def test_anthropic_unmatched_tool_result():
    msgs = [
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi"},
        {
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": "nonexistent_id",
                    "content": "result",
                }
            ],
        },
    ]
    result = validate(msgs, provider="anthropic")
    assert any(e.code == "unmatched_tool_result" for e in result.errors)


def test_anthropic_consecutive_same_role_detected():
    msgs = [
        {"role": "user", "content": "Hi"},
        {"role": "user", "content": "Hello again"},
    ]
    result = validate(msgs, provider="anthropic")
    assert any(e.code == "consecutive_same_role" for e in result.errors)


def test_anthropic_unmatched_tool_result_in_assistant_message():
    # A tool_result referencing an unknown id must be flagged regardless of
    # the message role it appears in.
    msgs = [
        {"role": "user", "content": "Hi"},
        {
            "role": "assistant",
            "content": [{"type": "tool_result", "tool_use_id": "ghost", "content": "x"}],
        },
    ]
    result = validate(msgs, provider="anthropic")
    assert any(e.code == "unmatched_tool_result" for e in result.errors)


def test_anthropic_strict_raises_first_error():
    msgs = [{"role": "assistant", "content": "Hi"}]  # must start with user
    with pytest.raises(ValidationError):
        validate(msgs, provider="anthropic", strict=True)


# ---------------------------------------------------------------------------
# OpenAI provider
# ---------------------------------------------------------------------------


def test_openai_valid():
    result = validate(ok_msgs(), provider="openai")
    assert result.valid


def test_openai_system_role_ok():
    msgs = [
        {"role": "system", "content": "Be helpful."},
        {"role": "user", "content": "Hello"},
    ]
    result = validate(msgs, provider="openai")
    assert result.valid


def test_openai_tool_role_ok():
    msgs = [
        {"role": "user", "content": "Use a tool"},
        {
            "role": "assistant",
            "content": "Using tool...",
        },
        {"role": "tool", "tool_call_id": "call_123", "content": "result"},
    ]
    result = validate(msgs, provider="openai")
    assert result.valid


def test_openai_invalid_role():
    msgs = [{"role": "user", "content": "Hi"}, {"role": "unknown_role", "content": "x"}]
    result = validate(msgs, provider="openai")
    assert any(e.code == "invalid_role" for e in result.errors)


def test_openai_tool_missing_tool_call_id():
    msgs = [
        {"role": "user", "content": "Hi"},
        {"role": "tool", "content": "result"},  # no tool_call_id
    ]
    result = validate(msgs, provider="openai")
    assert any(e.code == "missing_tool_call_id" for e in result.errors)


def test_openai_function_role_ok():
    msgs = [
        {"role": "user", "content": "Hi"},
        {"role": "assistant", "content": "calling function"},
        {"role": "function", "name": "search", "content": "results"},
    ]
    result = validate(msgs, provider="openai")
    assert result.valid

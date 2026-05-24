"""Validate LLM conversation message lists before sending to a provider.

Catches structural bugs early:
  - Missing role / content fields
  - Invalid role values per provider
  - Consecutive messages with the same role
  - Empty content
  - Unmatched tool_use / tool_result pairs (Anthropic)
  - Misplaced system messages

Provider presets:
  - ``"anthropic"`` — Anthropic Claude messages API rules.
  - ``"openai"``    — OpenAI Chat Completions API rules.
  - ``None``         — Generic rules (role in user/assistant/system, no
                       consecutive same-role).

All rules operate on ``list[dict]``; any serialisable message structure
is accepted.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any  # noqa: F401 (used via dataclass)

# ---------------------------------------------------------------------------
# Public types
# ---------------------------------------------------------------------------


class ValidationError(ValueError):
    """One validation problem found in the message list.

    Attributes:
        code: Short machine-readable error code (e.g. ``"missing_role"``).
        message: Human-readable description.
        index: Index of the offending message, or ``None`` for list-level errors.
    """

    def __init__(self, code: str, message: str, index: int | None = None) -> None:
        self.code = code
        self.message = message
        self.index = index
        super().__init__(str(self))

    def __str__(self) -> str:
        loc = f" (message {self.index})" if self.index is not None else ""
        return f"[{self.code}]{loc} {self.message}"


@dataclass
class ValidationResult:
    """Outcome of a :func:`validate` call.

    Attributes:
        valid: ``True`` when there are no errors.
        errors: List of :class:`ValidationError` instances.
    """

    errors: list[ValidationError] = field(default_factory=list)

    @property
    def valid(self) -> bool:
        return len(self.errors) == 0

    def __bool__(self) -> bool:
        return self.valid


# ---------------------------------------------------------------------------
# Provider rule sets
# ---------------------------------------------------------------------------

_GENERIC_ROLES = {"user", "assistant", "system"}
_ANTHROPIC_ROLES = {"user", "assistant"}
_OPENAI_ROLES = {"user", "assistant", "system", "tool", "function"}


def _check_generic(
    messages: list[dict[str, Any]],
    errors: list[ValidationError],
) -> None:
    """Rules applied to all providers unless overridden."""
    if not messages:
        errors.append(ValidationError("empty_list", "Messages list must not be empty."))
        return

    prev_role: str | None = None

    for i, msg in enumerate(messages):
        if not isinstance(msg, dict):
            errors.append(ValidationError("not_a_dict", "Message must be a dict.", index=i))
            continue

        role = msg.get("role")
        content = msg.get("content")

        # role required
        if role is None:
            errors.append(ValidationError("missing_role", "Missing 'role' field.", index=i))
        elif not isinstance(role, str):
            errors.append(
                ValidationError(
                    "invalid_role_type",
                    f"'role' must be a string, got {type(role).__name__!r}.",
                    index=i,
                )
            )
        elif role not in _GENERIC_ROLES:
            pass  # unknown roles are allowed at generic level

        # content required (None and "" are invalid)
        if content is None:
            errors.append(ValidationError("missing_content", "Missing 'content' field.", index=i))
        elif isinstance(content, str) and content == "":
            errors.append(
                ValidationError("empty_content", "'content' must not be empty string.", index=i)
            )
        elif isinstance(content, list) and len(content) == 0:
            errors.append(
                ValidationError("empty_content_list", "'content' list must not be empty.", index=i)
            )

        # Consecutive same-role check (only for user/assistant)
        if isinstance(role, str) and role in ("user", "assistant"):
            if role == prev_role:
                errors.append(
                    ValidationError(
                        "consecutive_same_role",
                        f"Two consecutive {role!r} messages at index {i}.",
                        index=i,
                    )
                )
            prev_role = role


def _check_anthropic(
    messages: list[dict[str, Any]],
    errors: list[ValidationError],
) -> None:
    """Anthropic-specific rules on top of generic checks."""
    # system role not allowed in messages list (must be a top-level param)
    for i, msg in enumerate(messages):
        if not isinstance(msg, dict):
            continue
        role = msg.get("role")
        if role == "system":
            errors.append(
                ValidationError(
                    "system_in_messages",
                    "Anthropic API: system prompt must be passed as a top-level "
                    "'system' parameter, not as a message.",
                    index=i,
                )
            )
        elif isinstance(role, str) and role not in _ANTHROPIC_ROLES:
            errors.append(
                ValidationError(
                    "invalid_role",
                    f"Anthropic API: role must be 'user' or 'assistant', got {role!r}.",
                    index=i,
                )
            )

    # tool_use / tool_result pairing
    _check_anthropic_tool_pairs(messages, errors)

    # Must start with user
    if messages:
        first = messages[0]
        if isinstance(first, dict) and first.get("role") != "user":
            role = first.get("role")
            errors.append(
                ValidationError(
                    "must_start_with_user",
                    f"Anthropic API: conversation must start with a 'user' message, got {role!r}.",
                    index=0,
                )
            )


def _check_anthropic_tool_pairs(
    messages: list[dict[str, Any]],
    errors: list[ValidationError],
) -> None:
    """Verify every tool_use block has a matching tool_result in a later user message."""
    # Collect all tool_use ids
    open_tool_ids: dict[str, int] = {}  # id → message index

    for i, msg in enumerate(messages):
        if not isinstance(msg, dict):
            continue
        content = msg.get("content")
        role = msg.get("role")

        if not isinstance(content, list):
            continue

        for block in content:
            if not isinstance(block, dict):
                continue
            if block.get("type") == "tool_use":
                tid = block.get("id")
                if tid:
                    open_tool_ids[tid] = i
            elif block.get("type") == "tool_result":
                tid = block.get("tool_use_id")
                if tid in open_tool_ids:
                    del open_tool_ids[tid]
                elif role == "user":
                    errors.append(
                        ValidationError(
                            "unmatched_tool_result",
                            f"tool_result references unknown tool_use id {tid!r}.",
                            index=i,
                        )
                    )

    for tid, msg_idx in open_tool_ids.items():
        errors.append(
            ValidationError(
                "unmatched_tool_use",
                f"tool_use id {tid!r} has no matching tool_result.",
                index=msg_idx,
            )
        )


def _check_openai(
    messages: list[dict[str, Any]],
    errors: list[ValidationError],
) -> None:
    """OpenAI-specific rules."""
    for i, msg in enumerate(messages):
        if not isinstance(msg, dict):
            continue
        role = msg.get("role")
        if isinstance(role, str) and role not in _OPENAI_ROLES:
            errors.append(
                ValidationError(
                    "invalid_role",
                    f"OpenAI API: invalid role {role!r}. Must be one of: {sorted(_OPENAI_ROLES)}.",
                    index=i,
                )
            )

        # tool_call_id required for role=tool
        if role == "tool" and not msg.get("tool_call_id"):
            errors.append(
                ValidationError(
                    "missing_tool_call_id",
                    "OpenAI API: 'tool' role message must have 'tool_call_id'.",
                    index=i,
                )
            )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def validate(
    messages: list[dict[str, Any]],
    *,
    provider: str | None = None,
    strict: bool = False,
) -> ValidationResult:
    """Validate a list of LLM conversation messages.

    Args:
        messages: The messages list to validate.
        provider: Optional provider name: ``"anthropic"`` or ``"openai"``.
            ``None`` applies only generic rules.
        strict: If ``True``, raise :exc:`ValidationError` on the first
            error found instead of returning a :class:`ValidationResult`.

    Returns:
        :class:`ValidationResult` with ``.valid`` and ``.errors``.

    Raises:
        ValidationError: If *strict* is ``True`` and at least one error
            is found.
        ValueError: If *provider* is not recognised.

    Example::

        >>> from llm_message_validator import validate
        >>> result = validate([{"role": "user", "content": "Hi"}])
        >>> result.valid
        True
    """
    if provider is not None and provider not in ("anthropic", "openai"):
        raise ValueError(f"Unknown provider {provider!r}. Use 'anthropic', 'openai', or None.")

    errors: list[ValidationError] = []

    _check_generic(messages, errors)

    if provider == "anthropic":
        _check_anthropic(messages, errors)
    elif provider == "openai":
        _check_openai(messages, errors)

    if strict and errors:
        raise errors[0]

    return ValidationResult(errors=errors)

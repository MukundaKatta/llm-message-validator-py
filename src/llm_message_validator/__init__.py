"""llm-message-validator: validate LLM conversation messages before sending.

Catches common mistakes in message lists before they reach the API.

Quick start::

    from llm_message_validator import validate, ValidationError

    msgs = [
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi"},
    ]

    result = validate(msgs)
    result.valid    # True
    result.errors   # []

    # Provider-specific rules
    result = validate(msgs, provider="anthropic")
    result = validate(msgs, provider="openai")

    # Raise on first error
    validate(msgs, strict=True)
"""

from .core import ValidationError, ValidationResult, validate

__all__ = ["ValidationError", "ValidationResult", "validate"]
__version__ = "0.1.0"

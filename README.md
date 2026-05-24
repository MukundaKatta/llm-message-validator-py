# llm-message-validator

Validate LLM conversation message lists before sending to the API. Catches structural bugs early. Zero dependencies.

```python
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
validate(msgs, strict=True)  # raises ValidationError if invalid
```

## Install

```bash
pip install llm-message-validator
```

## What it catches

**Generic (all providers)**
- Empty messages list
- Missing `role` or `content`
- Empty string or empty list content
- Consecutive messages with the same `user`/`assistant` role

**Anthropic provider**
- `system` role in messages list (must be top-level param)
- Invalid roles (only `user`/`assistant` allowed)
- Must start with a `user` message
- `tool_use` blocks without matching `tool_result`
- `tool_result` blocks referencing unknown `tool_use` ids

**OpenAI provider**
- Invalid roles (must be one of `user`/`assistant`/`system`/`tool`/`function`)
- `tool` role messages missing `tool_call_id`

## API

### `validate(messages, *, provider=None, strict=False) → ValidationResult`

- `provider`: `"anthropic"`, `"openai"`, or `None` (generic only)
- `strict`: raise `ValidationError` on first error

### `ValidationResult`

- `.valid` — `bool`
- `.errors` — `list[ValidationError]`

### `ValidationError`

- `.code` — machine-readable string (e.g. `"missing_role"`)
- `.message` — human-readable description
- `.index` — message index, or `None` for list-level errors

## License

MIT

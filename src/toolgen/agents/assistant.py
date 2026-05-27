"""Assistant agent — the tool-calling agent.

Decides at each turn whether to:
1. Ask a clarifying question (when info is missing → disambiguation)
2. Make a tool call (with structured arguments)
3. Provide a final answer

Tool calls use structured JSON output with endpoint_id + arguments dict.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from toolgen.agents.llm_client import LLMClient
from toolgen.models import APIEndpoint, Message, ToolCall

logger = logging.getLogger(__name__)

_MAX_TOOL_PARAMS = 8
_MAX_HISTORY_MESSAGES = 8
_MAX_SESSION_CONTEXT_CHARS = 1200
_MAX_TOOL_RESULT_CHARS = 700
_MAX_VISIBLE_RESPONSE_CHARS = 320


def _build_tool_descriptions(available_tools: list[APIEndpoint]) -> str:
    """Format available tools compactly for the system prompt."""
    descriptions = []
    for ep in available_tools:
        params = [_format_parameter(p) for p in ep.parameters[:_MAX_TOOL_PARAMS]]
        if len(ep.parameters) > _MAX_TOOL_PARAMS:
            params.append(f"+{len(ep.parameters) - _MAX_TOOL_PARAMS} more")
        params_text = ", ".join(params) if params else "none"
        descriptions.append(
            f"- {ep.endpoint_id} [{ep.method.value}]: "
            f"{_shorten(ep.description or 'No description', 130)}; args: {params_text}"
        )

    return "\n".join(descriptions)


_ASSISTANT_SYSTEM_TEMPLATE = """\
You are a concise tool-using assistant.
Available tools:
{tools}

Choose one action each turn:
- Ask one short clarifying question if required info is missing.
- Call a tool when ready.
- Answer after results arrive.

Tool calls must be exactly this JSON and no prose:
{{"action": "tool_call", "endpoint": "tool_name/endpoint_name", "arguments": {{"param": "value"}}}}

Rules:
- Required args are marked required; ask instead of guessing.
- Use IDs/data from previous results; never invent them.
- Never mention tool names, endpoint names, API names, or internal identifiers in
  user-facing text.
- Keep user-facing prose under 2 sentences and 45 words.
"""


def generate_assistant_turn(
    client: LLMClient,
    conversation_history: list[Message],
    available_tools: list[APIEndpoint],
    session_context: dict[str, Any] | None = None,
    strict_live: bool = False,
) -> tuple[str | None, ToolCall | None]:
    """Generate the assistant's next turn.

    Returns:
        (text_response, tool_call) — exactly one will be non-None.
        text_response: natural language (clarification or final answer)
        tool_call: structured tool call to execute
    """
    tools_text = _build_tool_descriptions(available_tools)
    system_instruction = _ASSISTANT_SYSTEM_TEMPLATE.format(tools=tools_text)

    # Add session context if available (for grounding)
    if session_context:
        system_instruction += (
            "\n\nSession context:\n"
            f"{_compact_json(session_context, _MAX_SESSION_CONTEXT_CHARS)}"
        )

    # Build message history for the LLM
    messages = []
    for msg in _trim_history(conversation_history):
        if msg.role == "user":
            messages.append({"role": "user", "content": msg.content or ""})
        elif msg.role == "assistant":
            if msg.content:
                messages.append({"role": "model", "content": msg.content})
            elif msg.tool_calls:
                tc = msg.tool_calls[0]
                messages.append({
                    "role": "model",
                    "content": json.dumps({
                        "action": "tool_call",
                        "endpoint": tc.endpoint,
                        "arguments": tc.arguments,
                    }),
                })
        elif msg.role == "tool":
            content = msg.content
            messages.append({
                "role": "user",
                "content": f"[Tool result] {_compact_json(content, _MAX_TOOL_RESULT_CHARS)}",
            })

    if not messages:
        return "How can I help you today?", None

    response = client.generate_with_history(
        messages,
        system_instruction=system_instruction,
        temperature=0.5,
    )

    if response.error:
        if strict_live:
            raise RuntimeError(f"Assistant live LLM failed: {response.error}")
        logger.warning("Assistant generation failed: %s", response.error)
        return "I'm sorry, I encountered an issue. Could you repeat that?", None

    text = response.text.strip()

    # Try to parse as tool call
    tool_call = _try_parse_tool_call(text, available_tools)
    if tool_call:
        return None, tool_call

    # Otherwise it's a text response
    return _shorten_visible_response(text), None


def _format_parameter(parameter: Any) -> str:
    required = " required" if parameter.required else ""
    default = f"={parameter.default}" if parameter.default is not None else ""
    return f"{parameter.name}:{parameter.type.value}{required}{default}"


def _trim_history(conversation_history: list[Message]) -> list[Message]:
    selected = conversation_history[-_MAX_HISTORY_MESSAGES:]
    first_user = next((msg for msg in conversation_history if msg.role == "user"), None)
    if first_user and all(msg is not first_user for msg in selected):
        selected = [first_user, *selected[-(_MAX_HISTORY_MESSAGES - 1):]]
    return selected


def _compact_json(value: Any, max_chars: int) -> str:
    if isinstance(value, str):
        return _shorten(value, max_chars)
    try:
        text = json.dumps(value, default=str, separators=(",", ":"))
    except TypeError:
        text = str(value)
    return _shorten(text, max_chars)


def _shorten_visible_response(text: str) -> str:
    return _shorten(text.strip(), _MAX_VISIBLE_RESPONSE_CHARS)


def _shorten(text: str, max_chars: int) -> str:
    normalized = " ".join(str(text).split())
    if len(normalized) <= max_chars:
        return normalized
    boundary = max(
        normalized.rfind(". ", 0, max_chars),
        normalized.rfind("? ", 0, max_chars),
        normalized.rfind("! ", 0, max_chars),
    )
    if boundary >= max_chars // 2:
        return normalized[: boundary + 1]
    return normalized[: max_chars - 3].rstrip() + "..."


def _try_parse_tool_call(
    text: str,
    available_tools: list[APIEndpoint],
) -> ToolCall | None:
    """Try to parse the assistant's response as a structured tool call."""
    # Strip code fences if present
    clean = text.strip()
    if clean.startswith("```"):
        lines = clean.split("\n")
        if lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        clean = "\n".join(lines).strip()

    # Try to find JSON in the response
    start = clean.find("{")
    end = clean.rfind("}") + 1
    if start < 0 or end <= start:
        return None

    try:
        parsed = json.loads(clean[start:end])
    except json.JSONDecodeError:
        return None

    # Check if it's a tool call
    if parsed.get("action") != "tool_call":
        return None

    endpoint = parsed.get("endpoint", "")
    arguments = parsed.get("arguments", {})

    if not endpoint:
        return None
    if not isinstance(arguments, dict):
        logger.warning("Assistant produced non-object tool arguments for %s", endpoint)
        return None

    # Validate endpoint exists
    valid_ids = {ep.endpoint_id for ep in available_tools}
    if endpoint not in valid_ids:
        # Try fuzzy matching
        for valid_id in valid_ids:
            if endpoint.lower() == valid_id.lower():
                endpoint = valid_id
                break
        else:
            logger.warning("Assistant called unknown endpoint: %s", endpoint)
            # Still allow it — the orchestrator can handle it
            pass

    return ToolCall(endpoint=endpoint, arguments=arguments)

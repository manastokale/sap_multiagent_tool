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


def _build_tool_descriptions(available_tools: list[APIEndpoint]) -> str:
    """Format available tools for the system prompt."""
    descriptions = []
    for ep in available_tools:
        params = []
        for p in ep.parameters:
            req = " (REQUIRED)" if p.required else ""
            default = f", default={p.default}" if p.default is not None else ""
            params.append(f"    - {p.name}: {p.type.value}{req}{default} — {p.description}")

        params_text = "\n".join(params) if params else "    (no parameters)"
        descriptions.append(
            f"  {ep.endpoint_id} [{ep.method.value}]\n"
            f"    Description: {ep.description or 'No description'}\n"
            f"    Parameters:\n{params_text}"
        )

    return "\n\n".join(descriptions)


_ASSISTANT_SYSTEM_TEMPLATE = """\
You are a helpful AI assistant with access to tools. Your job is to help the \
user accomplish their task by calling the right tools at the right time.

Available tools:
{tools}

At each turn, you must decide one of three actions:
1. ASK a clarifying question if the user's request is ambiguous or missing required info
2. CALL a tool if you have enough info to proceed
3. RESPOND with a final answer after tools have returned results

For tool calls, respond with EXACTLY this JSON format (no other text):
{{"action": "tool_call", "endpoint": "tool_name/endpoint_name", "arguments": {{"param": "value"}}}}

For clarifying questions or final answers, respond with natural text only.

Rules:
- ALWAYS check if required parameters are available before calling a tool
- If required info is missing, ASK the user — don't guess or hallucinate values
- When you receive tool results, summarize them naturally for the user
- Use IDs and data from previous tool results — don't make up fake values
- Be concise and helpful
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
            f"\n\nSession context (results from previous tool calls):\n"
            f"{json.dumps(session_context, indent=2, default=str)}"
        )

    # Build message history for the LLM
    messages = []
    for msg in conversation_history:
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
            if isinstance(content, dict):
                content = json.dumps(content, default=str)
            messages.append({
                "role": "user",
                "content": f"[Tool Result]: {content}",
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
    return text, None


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

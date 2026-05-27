"""User simulator agent.

Generates realistic user messages based on the scenario plan and conversation
history. Simulates real user behavior: vague initial requests, responses to
clarifying questions, follow-up requests, and corrections.
"""

from __future__ import annotations

import logging

from toolgen.agents.llm_client import LLMClient
from toolgen.models import Message, ScenarioPlan

logger = logging.getLogger(__name__)

_MAX_HISTORY_MESSAGES = 6
_MAX_HISTORY_CHARS = 900
_MAX_USER_MESSAGE_CHARS = 220
_MAX_TOOL_SUMMARY_CHARS = 240

_USER_SIM_SYSTEM_INSTRUCTION = """\
Simulate a real user chatting with an AI assistant.

Rules:
- Stay in persona.
- Keep replies natural and brief, usually 1 sentence.
- Answer clarifying questions with requested details.
- Never mention tools, APIs, endpoints, or technical details.
"""


def generate_initial_message(
    client: LLMClient,
    plan: ScenarioPlan,
    strict_live: bool = False,
) -> str:
    """Generate the user's opening message based on the scenario."""
    missing = "; ".join(plan.disambiguation_points) if plan.disambiguation_points else "none"
    prompt = (
        f"Persona: {plan.user_persona}\n"
        f"Scenario: {plan.scenario}\n\n"
        f"Leave vague/missing: {missing}\n"
        "Write the opening user message in 35 words or fewer."
    )

    response = client.generate(
        prompt,
        system_instruction=_USER_SIM_SYSTEM_INSTRUCTION,
        temperature=0.8,
    )

    if response.error:
        if strict_live:
            raise RuntimeError(f"User simulator live LLM failed: {response.error}")
        logger.warning("User sim initial message failed: %s", response.error)
        return f"Hey, I need help with {plan.scenario}"

    return _shorten_user_message(response.text)


def generate_response(
    client: LLMClient,
    plan: ScenarioPlan,
    conversation_history: list[Message],
    tool_results_summary: str | None = None,
    strict_live: bool = False,
) -> str:
    """Generate a user response to the assistant's latest message."""
    # Build conversation context
    history_text = "\n".join(
        f"{msg.role.upper()}: {msg.content}"
        for msg in _trim_history(conversation_history)
        if msg.content and msg.role in ("user", "assistant")
    )
    history_text = _shorten(history_text, _MAX_HISTORY_CHARS)

    prompt = (
        f"Persona: {plan.user_persona}\n"
        f"Scenario: {plan.scenario}\n"
        f"Conversation so far:\n{history_text}\n\n"
    )

    if tool_results_summary:
        prompt += f"Latest assistant summary: {_shorten(tool_results_summary, _MAX_TOOL_SUMMARY_CHARS)}\n\n"

    prompt += (
        "Write the next user message in 35 words or fewer. "
        "Answer a question if asked; if complete, acknowledge briefly."
    )

    response = client.generate(
        prompt,
        system_instruction=_USER_SIM_SYSTEM_INSTRUCTION,
        temperature=0.8,
    )

    if response.error:
        if strict_live:
            raise RuntimeError(f"User simulator live LLM failed: {response.error}")
        logger.warning("User sim response failed: %s", response.error)
        return "Sure, that sounds good. Can you go ahead with that?"

    return _shorten_user_message(response.text)


def _shorten_user_message(text: str) -> str:
    return _shorten(text.strip(), _MAX_USER_MESSAGE_CHARS)


def _trim_history(conversation_history: list[Message]) -> list[Message]:
    selected = conversation_history[-_MAX_HISTORY_MESSAGES:]
    first_user = next((msg for msg in conversation_history if msg.role == "user"), None)
    if first_user and all(msg is not first_user for msg in selected):
        selected = [first_user, *selected[-(_MAX_HISTORY_MESSAGES - 1):]]
    return selected


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
    return normalized[: max_chars - 1].rstrip() + "..."


def should_end_conversation(
    plan: ScenarioPlan,
    conversation_history: list[Message],
    num_tool_calls_made: int,
) -> bool:
    """Heuristic: should the user end the conversation?

    Returns True if:
    - All expected tools have been called
    - The assistant has provided a final answer
    - The conversation has gone on long enough
    """
    expected_tools = len(plan.expected_tool_sequence)

    # All expected tools called
    if num_tool_calls_made >= expected_tools:
        # Check if assistant gave a final-looking message
        for msg in reversed(conversation_history):
            if msg.role == "assistant" and msg.content:
                # Simple heuristics for "task complete" indicators
                lower = msg.content.lower()
                if any(phrase in lower for phrase in [
                    "booked", "confirmed", "completed", "done",
                    "here are", "here's", "i've", "all set",
                    "anything else", "is there anything",
                ]):
                    return True
        return True  # All tools called, assume done

    return False

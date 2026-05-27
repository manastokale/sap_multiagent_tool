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

_USER_SIM_SYSTEM_INSTRUCTION = """\
You are simulating a real user interacting with an AI assistant.
You must stay in character based on the persona provided.

Rules:
- Keep messages concise and natural — like real chat messages
- When the assistant asks clarifying questions, provide the requested info
  (but sometimes be slightly vague to keep it realistic)
- If the assistant completes your task, acknowledge it naturally
- Don't be overly polite or formal unless the persona calls for it
- Don't mention tools, APIs, or technical details — you're just a user
- Vary message length: some short, some with more detail
- If the scenario calls for it, add follow-up requests after the main task
"""


def generate_initial_message(
    client: LLMClient,
    plan: ScenarioPlan,
    strict_live: bool = False,
) -> str:
    """Generate the user's opening message based on the scenario."""
    prompt = (
        f"You are playing this character: {plan.user_persona}\n\n"
        f"Scenario: {plan.scenario}\n\n"
        f"The following information should be intentionally left vague or missing "
        f"from your initial message (the assistant should ask about these):\n"
        f"- {chr(10).join(plan.disambiguation_points) if plan.disambiguation_points else 'none'}\n\n"
        f"Write your opening message to the assistant. Keep it natural and concise."
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

    return response.text.strip()


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
        for msg in conversation_history
        if msg.content and msg.role in ("user", "assistant")
    )

    prompt = (
        f"You are playing this character: {plan.user_persona}\n\n"
        f"Scenario: {plan.scenario}\n\n"
        f"Conversation so far:\n{history_text}\n\n"
    )

    if tool_results_summary:
        prompt += f"The assistant just got these tool results: {tool_results_summary}\n\n"

    prompt += (
        "Write your next message as the user. Keep it natural and concise.\n"
        "If the assistant asked a question, answer it.\n"
        "If the task seems complete, thank them or ask a follow-up."
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

    return response.text.strip()


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

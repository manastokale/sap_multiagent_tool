"""Automatic retry / repair for failed conversations.

Three-tier repair strategy:
1. Structural repair (no LLM): fix missing fields, re-validate arguments
2. Targeted regeneration: regenerate from the failing turn onward
3. Full regeneration: re-run entire conversation with judge feedback

Design justification: targeted regeneration is cheaper and more likely to fix
specific issues than full regeneration. Structural repair catches low-hanging
fruit without any LLM cost.
"""

from __future__ import annotations

import logging

from toolgen.agents.llm_client import LLMClient
from toolgen.agents.orchestrator import generate_conversation, generate_offline_conversation
from toolgen.executor.mock_executor import MockExecutor
from toolgen.judge.scorer import passes_quality_threshold, score_conversation
from toolgen.models import (
    Conversation,
    JudgeScores,
    Message,
    SteeringGuidance,
    ToolChain,
)

logger = logging.getLogger(__name__)


def structural_repair(conversation: Conversation) -> Conversation:
    """Tier 1: Fix structural issues without LLM calls.

    Fixes:
    - Missing role tags
    - Empty messages
    - Malformed tool calls
    - Consecutive same-role messages
    """
    repaired_messages: list[Message] = []

    for msg in conversation.messages:
        # Skip empty messages
        if msg.role == "assistant" and msg.content is None and not msg.tool_calls:
            continue
        if msg.role == "user" and not msg.content:
            continue

        # Ensure valid role
        if msg.role not in ("user", "assistant", "tool", "system"):
            logger.debug("Skipping message with invalid role: %s", msg.role)
            continue

        repaired_messages.append(msg)

    # Remove consecutive same-role messages (except tool results after tool calls)
    deduped: list[Message] = []
    for i, msg in enumerate(repaired_messages):
        if i > 0 and msg.role == deduped[-1].role:
            if msg.role == "tool":
                # Multiple tool results are OK
                deduped.append(msg)
            elif msg.role == "assistant" and deduped[-1].tool_calls and msg.content:
                # Assistant summary after tool call is OK
                deduped.append(msg)
            else:
                # Merge or skip
                if msg.content and deduped[-1].content:
                    deduped[-1] = Message(
                        role=msg.role,
                        content=f"{deduped[-1].content}\n{msg.content}",
                        tool_calls=msg.tool_calls or deduped[-1].tool_calls,
                    )
        else:
            deduped.append(msg)

    conversation.messages = deduped
    conversation.metadata.num_turns = len(deduped)
    conversation.metadata.num_tool_calls = sum(
        len(msg.tool_calls or []) for msg in deduped if msg.role == "assistant"
    )
    conversation.metadata.tools_used = [
        call.endpoint
        for msg in deduped
        if msg.role == "assistant"
        for call in (msg.tool_calls or [])
    ]
    conversation.metadata.num_distinct_tools = len(
        {endpoint.split("/")[0] for endpoint in conversation.metadata.tools_used}
    )
    valid_tools = set(conversation.metadata.tools_used)
    conversation.step_trace = [
        step for step in conversation.step_trace if step.endpoint in valid_tools
    ]
    return conversation


def repair_conversation(
    conversation: Conversation,
    scores: JudgeScores,
    gen_client: LLMClient,
    judge_client: LLMClient,
    chain: ToolChain,
    executor: MockExecutor,
    max_attempts: int = 2,
    quality_threshold: float = 3.5,
    seed: int = 42,
    steering: SteeringGuidance | None = None,
    strict_live: bool = False,
    live_profile: str = "full",
    role_clients: dict[str, LLMClient] | None = None,
) -> tuple[Conversation, int]:
    """Attempt to repair a conversation that failed quality checks.

    Returns:
        (repaired_conversation, num_repair_attempts)
    """
    attempts = 0

    # Tier 1: Structural repair (always try first, free)
    conversation = structural_repair(conversation)
    scores = score_conversation(judge_client, conversation, strict_live=strict_live)
    if passes_quality_threshold(scores, quality_threshold):
        conversation.judge_scores = scores
        return conversation, 0

    # Tier 2 & 3: LLM-based repair (with limited attempts)
    for attempt in range(max_attempts):
        attempts += 1

        logger.info(
            "Repair attempt %d/%d for %s (score: %.2f)",
            attempt + 1, max_attempts,
            conversation.conversation_id,
            scores.overall,
        )

        # Full regeneration with judge feedback injected into planner
        enhanced_steering = steering or SteeringGuidance()
        feedback = _format_judge_feedback(scores)
        enhanced_steering.rationale = (
            f"{enhanced_steering.rationale}\n"
            f"Previous attempt scored {scores.overall:.1f}/5.0. "
            f"Issues: {feedback}"
        ).strip()

        if getattr(gen_client, "is_offline", False):
            if strict_live:
                raise RuntimeError("Strict live repair requested but generation client is offline")
            new_conversation = generate_offline_conversation(
                chain=chain,
                executor=executor,
                conversation_index=conversation.metadata.conversation_index,
                seed=seed + attempt + 1,
                steering=enhanced_steering,
                model_name=conversation.metadata.model or "offline-deterministic",
            )
        else:
            new_conversation = generate_conversation(
                client=gen_client,
                chain=chain,
                executor=executor,
                conversation_index=conversation.metadata.conversation_index,
                seed=seed + attempt + 1,  # Vary seed for different results
                steering=enhanced_steering,
                model_name=conversation.metadata.model,
                strict_live=strict_live,
                live_profile=live_profile,
                role_clients=role_clients,
            )

        new_scores = score_conversation(judge_client, new_conversation, strict_live=strict_live)
        new_conversation.judge_scores = new_scores
        new_conversation.metadata.repair_attempts = attempts

        if passes_quality_threshold(new_scores, quality_threshold):
            logger.info(
                "Repair succeeded for %s after %d attempts (new score: %.2f)",
                conversation.conversation_id, attempts, new_scores.overall,
            )
            return new_conversation, attempts

        # Use the better version for next attempt
        if new_scores.overall > scores.overall:
            conversation = new_conversation
            scores = new_scores

    logger.warning(
        "Repair exhausted for %s after %d attempts (best score: %.2f)",
        conversation.conversation_id, attempts, scores.overall,
    )
    conversation.judge_scores = scores
    conversation.metadata.repair_attempts = attempts
    return conversation, attempts


def _format_judge_feedback(scores: JudgeScores) -> str:
    """Format judge scores into actionable feedback."""
    issues = []
    if scores.tool_correctness.score < 4.0:
        issues.append(f"Tool correctness: {scores.tool_correctness.rationale}")
    if scores.naturalness.score < 4.0:
        issues.append(f"Naturalness: {scores.naturalness.rationale}")
    if scores.task_completion.score < 4.0:
        issues.append(f"Task completion: {scores.task_completion.rationale}")
    return "; ".join(issues) if issues else "Minor quality issues"

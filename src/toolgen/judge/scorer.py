"""LLM-as-Judge scorer.

Scores each generated conversation on three dimensions:
  1. Tool Correctness: Are tool calls valid? Arguments correct? IDs chain properly?
  2. Naturalness: Does the conversation feel realistic? Is disambiguation natural?
  3. Task Completion: Was the user's goal achieved? Were necessary tools called?

Uses cheaper/lighter Flash-Lite models by default since judging
is a simpler task than generation.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from toolgen.agents.llm_client import LLMClient
from toolgen.models import (
    Conversation,
    JudgeScore,
    JudgeScores,
)

logger = logging.getLogger(__name__)

_JUDGE_SYSTEM_INSTRUCTION = """\
You are an expert evaluator of AI assistant conversations. You score conversations \
on three dimensions, each from 1.0 to 5.0.

Dimensions:
1. TOOL_CORRECTNESS (1-5): Are tool calls valid and properly formed?
   - 5: All tool calls use correct endpoints, proper argument types, and IDs chain perfectly
   - 3: Most calls correct, minor argument issues
   - 1: Wrong endpoints, hallucinated IDs, broken chains

2. NATURALNESS (1-5): Does the conversation feel like a real human-AI interaction?
   - 5: Completely natural flow, realistic disambiguation, appropriate tone
   - 3: Somewhat natural but with awkward transitions or unnatural language
   - 1: Robotic, repetitive, or clearly synthetic

3. TASK_COMPLETION (1-5): Was the user's goal fully accomplished?
   - 5: Goal fully achieved, all necessary tools called, clear confirmation
   - 3: Partially achieved, some steps missing or incomplete
   - 1: Goal not addressed, conversation went off-track

You MUST respond with ONLY valid JSON in this exact format:
{
  "tool_correctness": {"score": 4.5, "rationale": "explanation..."},
  "naturalness": {"score": 4.0, "rationale": "explanation..."},
  "task_completion": {"score": 5.0, "rationale": "explanation..."}
}
"""


def _format_conversation_for_judge(conversation: Conversation) -> str:
    """Format a conversation for the judge to review."""
    lines = []
    for msg in conversation.messages:
        if msg.role == "user":
            lines.append(f"USER: {msg.content}")
        elif msg.role == "assistant":
            if msg.content:
                lines.append(f"ASSISTANT: {msg.content}")
            if msg.tool_calls:
                for tc in msg.tool_calls:
                    lines.append(
                        f"ASSISTANT [TOOL CALL]: {tc.endpoint}({json.dumps(tc.arguments)})"
                    )
        elif msg.role == "tool":
            content = msg.content
            if isinstance(content, dict):
                content = json.dumps(content, default=str)
            # Truncate very long tool results
            if isinstance(content, str) and len(content) > 500:
                content = content[:500] + "..."
            lines.append(f"TOOL RESULT: {content}")

    return "\n".join(lines)


def score_conversation(
    judge_client: LLMClient,
    conversation: Conversation,
    strict_live: bool = False,
) -> JudgeScores:
    """Score a single conversation using the LLM-as-judge.

    Args:
        judge_client: LLM client configured with the judge model.
        conversation: The conversation to score.

    Returns:
        JudgeScores with scores and rationales for each dimension.
    """
    if getattr(judge_client, "is_offline", False):
        if strict_live:
            raise RuntimeError("Strict live judge requested but judge client is offline")
        return score_conversation_heuristic(conversation)

    formatted = _format_conversation_for_judge(conversation)

    metadata_summary = (
        f"Tools used: {', '.join(conversation.metadata.tools_used)}\n"
        f"Total turns: {conversation.metadata.num_turns}\n"
        f"Tool calls: {conversation.metadata.num_tool_calls}\n"
        f"Pattern: {conversation.metadata.pattern}"
    )

    prompt = (
        f"Score the following conversation:\n\n"
        f"--- CONVERSATION ---\n{formatted}\n--- END ---\n\n"
        f"Metadata:\n{metadata_summary}\n\n"
        f"Score this conversation on all three dimensions."
    )

    response = judge_client.generate_json(
        prompt,
        system_instruction=_JUDGE_SYSTEM_INSTRUCTION,
        temperature=0.2,  # Low temp for consistent scoring
    )

    if response.error or not response.parsed:
        if strict_live:
            raise RuntimeError(f"Judge live LLM failed for {conversation.conversation_id}: {response.error}")
        logger.warning(
            "Judge scoring failed for %s: %s",
            conversation.conversation_id,
            response.error,
        )
        return score_conversation_heuristic(conversation)

    try:
        return _parse_judge_response(response.parsed)
    except Exception as e:
        if strict_live:
            raise RuntimeError(f"Judge live LLM output failed schema validation: {e}") from e
        logger.warning("Failed to parse judge response: %s", e)
        return score_conversation_heuristic(conversation)


def score_conversation_heuristic(conversation: Conversation) -> JudgeScores:
    """Deterministic fallback judge for offline runs and tests.

    It is not a replacement for an LLM-as-judge, but it checks the same failure
    modes that matter for the assignment: valid structure, grounded IDs, natural
    disambiguation, and task completion.
    """
    tool_calls = []
    tool_results = []
    prior_ids: set[str] = set()
    grounded_id_issues = 0
    structural_issues = 0

    for msg in conversation.messages:
        if msg.role == "assistant" and msg.tool_calls:
            for call in msg.tool_calls:
                tool_calls.append(call)
                if not call.endpoint or not isinstance(call.arguments, dict):
                    structural_issues += 1
                for key, value in call.arguments.items():
                    lowered = key.lower()
                    if lowered == "id" or lowered.endswith("_id"):
                        if prior_ids and isinstance(value, str) and value not in prior_ids:
                            grounded_id_issues += 1
        elif msg.role == "tool":
            tool_results.append(msg)
            _collect_ids(msg.content, prior_ids)

    tool_score = 5.0
    if not tool_calls:
        tool_score -= 2.0
    if len(tool_results) < len(tool_calls):
        tool_score -= 1.0
    tool_score -= min(2.0, grounded_id_issues * 0.75)
    tool_score -= min(1.0, structural_issues * 0.5)
    tool_score = max(1.0, min(5.0, tool_score))

    has_clarification = any(
        msg.role == "assistant"
        and isinstance(msg.content, str)
        and "?" in msg.content
        for msg in conversation.messages
    )
    has_user_followup = sum(1 for msg in conversation.messages if msg.role == "user") >= 2
    naturalness = 4.0 + (0.5 if has_clarification and has_user_followup else 0.0)
    if len(conversation.messages) < 4:
        naturalness -= 1.0

    final_assistant = next(
        (
            msg
            for msg in reversed(conversation.messages)
            if msg.role == "assistant" and isinstance(msg.content, str) and msg.content.strip()
        ),
        None,
    )
    completion = 5.0 if final_assistant and tool_calls else 3.0
    if final_assistant and any(
        token in final_assistant.content.lower()
        for token in ("done", "completed", "confirmed", "reference")
    ):
        completion = min(5.0, completion + 0.0)
    elif final_assistant:
        completion -= 0.5

    return JudgeScores(
        tool_correctness=JudgeScore(
            score=round(tool_score, 2),
            rationale=(
                "Heuristic check over tool-call structure and prior-result ID grounding"
            ),
        ),
        naturalness=JudgeScore(
            score=round(max(1.0, min(5.0, naturalness)), 2),
            rationale="Heuristic check for multi-turn flow and clarification",
        ),
        task_completion=JudgeScore(
            score=round(max(1.0, min(5.0, completion)), 2),
            rationale="Heuristic check for final assistant completion after tool use",
        ),
    )


def _collect_ids(content: Any, ids: set[str]) -> None:
    if isinstance(content, dict):
        for key, value in content.items():
            lowered = key.lower()
            if isinstance(value, (dict, list)):
                _collect_ids(value, ids)
            elif lowered == "id" or lowered.endswith("_id") or lowered == "confirmation_id":
                ids.add(str(value))
    elif isinstance(content, list):
        for item in content:
            _collect_ids(item, ids)


def _parse_judge_response(parsed: dict[str, Any]) -> JudgeScores:
    """Parse the judge's JSON response into JudgeScores."""
    def _extract_score(data: Any) -> JudgeScore:
        if isinstance(data, dict):
            score = float(data.get("score", 3.0))
            rationale = str(data.get("rationale", ""))
        elif isinstance(data, (int, float)):
            score = float(data)
            rationale = ""
        else:
            score = 3.0
            rationale = ""
        # Clamp to valid range
        score = max(1.0, min(5.0, score))
        return JudgeScore(score=score, rationale=rationale)

    return JudgeScores(
        tool_correctness=_extract_score(parsed.get("tool_correctness", 3.0)),
        naturalness=_extract_score(parsed.get("naturalness", 3.0)),
        task_completion=_extract_score(parsed.get("task_completion", 3.0)),
    )


def _fallback_scores() -> JudgeScores:
    """Return neutral scores when the judge fails."""
    return JudgeScores(
        tool_correctness=JudgeScore(score=3.0, rationale="Judge unavailable"),
        naturalness=JudgeScore(score=3.0, rationale="Judge unavailable"),
        task_completion=JudgeScore(score=3.0, rationale="Judge unavailable"),
    )


def passes_quality_threshold(
    scores: JudgeScores,
    threshold: float = 3.5,
) -> bool:
    """Check if scores pass the quality threshold."""
    return scores.overall >= threshold

"""Tests for structural and retry repair."""

from toolgen.agents.llm_client import LLMClient
from toolgen.executor.mock_executor import MockExecutor
from toolgen.models import Conversation, ConversationMetadata, JudgeScore, JudgeScores, Message
from toolgen.repair.repairer import repair_conversation, structural_repair


def test_structural_repair_removes_empty_messages_and_updates_metadata():
    conv = Conversation(
        conversation_id="broken",
        messages=[
            Message(role="user", content="Find a hotel"),
            Message(role="assistant", content=None),
            Message(role="assistant", content="Sure."),
            Message(role="assistant", content="I can help."),
        ],
        metadata=ConversationMetadata(num_turns=4),
    )

    repaired = structural_repair(conv)

    assert repaired.metadata.num_turns == len(repaired.messages)
    assert all(msg.role in {"user", "assistant", "tool", "system"} for msg in repaired.messages)
    assert not any(
        msg.role == "assistant" and msg.content is None and not msg.tool_calls
        for msg in repaired.messages
    )


def test_offline_repair_regenerates_low_quality_conversation(sample_chain):
    low_scores = JudgeScores(
        tool_correctness=JudgeScore(score=1.5, rationale="No tool calls"),
        naturalness=JudgeScore(score=2.0, rationale="Incomplete"),
        task_completion=JudgeScore(score=1.0, rationale="Not done"),
    )
    broken = Conversation(
        conversation_id="conv_0001",
        messages=[Message(role="user", content="Can you help?")],
        judge_scores=low_scores,
        metadata=ConversationMetadata(conversation_index=1, model="offline-deterministic"),
    )

    repaired, attempts = repair_conversation(
        conversation=broken,
        scores=low_scores,
        gen_client=LLMClient(model="offline-deterministic"),
        judge_client=LLMClient(model="offline-heuristic"),
        chain=sample_chain,
        executor=MockExecutor(seed=123),
        max_attempts=1,
        quality_threshold=4.0,
        seed=123,
    )

    assert attempts == 1
    assert repaired.metadata.num_tool_calls >= 1
    assert repaired.judge_scores is not None
    assert repaired.judge_scores.overall >= 4.0

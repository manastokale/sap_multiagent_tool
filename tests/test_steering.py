"""Tests for cross-conversation diversity steering."""

from toolgen.memory.steering import DiversitySteerer
from toolgen.models import Conversation, ConversationMetadata, JudgeScore, JudgeScores


def _conversation(index: int, tools: list[str], domains: list[str], pattern: str) -> Conversation:
    return Conversation(
        conversation_id=f"conv_{index:04d}",
        judge_scores=JudgeScores(
            tool_correctness=JudgeScore(score=4.0),
            naturalness=JudgeScore(score=4.0),
            task_completion=JudgeScore(score=4.0),
        ),
        metadata=ConversationMetadata(
            conversation_index=index,
            tools_used=tools,
            category_domains=domains,
            pattern=pattern,
            num_tool_calls=len(tools),
            num_distinct_tools=len({tool.split("/")[0] for tool in tools}),
        ),
    )


def test_counter_based_steering_prefers_underused_domains():
    steerer = DiversitySteerer(enabled=True)
    for i in range(3):
        steerer.record_generation(
            _conversation(
                i,
                ["hotel_api/search_hotels", "flight_api/book_flight"],
                ["Travel"],
                "multi_step",
            )
        )

    guidance = steerer.get_steering_guidance(
        available_domains=["Travel", "Food", "Commerce"]
    )

    assert ["flight_api/book_flight", "hotel_api/search_hotels"] in sorted(
        guidance.avoid_tool_combinations
    )
    assert "Food" in guidance.prefer_domains or "Commerce" in guidance.prefer_domains
    assert guidance.rationale


def test_disabled_steering_returns_empty_guidance():
    steerer = DiversitySteerer(enabled=False)
    guidance = steerer.get_steering_guidance(available_domains=["Travel"])

    assert guidance.avoid_tool_combinations == []
    assert guidance.prefer_domains == []

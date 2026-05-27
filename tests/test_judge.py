"""Tests for the LLM-as-judge scorer."""


from toolgen.judge.scorer import (
    _parse_judge_response,
    passes_quality_threshold,
    _format_conversation_for_judge,
)
from toolgen.models import (
    Conversation,
    JudgeScore,
    JudgeScores,
    Message,
    ToolCall,
)


class TestParseJudgeResponse:
    def test_valid_response(self):
        parsed = {
            "tool_correctness": {"score": 4.5, "rationale": "Good tool usage"},
            "naturalness": {"score": 4.0, "rationale": "Natural flow"},
            "task_completion": {"score": 5.0, "rationale": "Complete"},
        }
        scores = _parse_judge_response(parsed)
        assert scores.tool_correctness.score == 4.5
        assert scores.naturalness.score == 4.0
        assert scores.task_completion.score == 5.0
        assert scores.overall == 4.5

    def test_numeric_scores(self):
        parsed = {
            "tool_correctness": 3.0,
            "naturalness": 4.0,
            "task_completion": 5.0,
        }
        scores = _parse_judge_response(parsed)
        assert scores.tool_correctness.score == 3.0

    def test_missing_fields(self):
        parsed = {"tool_correctness": {"score": 4.0}}
        scores = _parse_judge_response(parsed)
        assert scores.tool_correctness.score == 4.0
        assert scores.naturalness.score == 3.0  # Default

    def test_score_clamping(self):
        parsed = {
            "tool_correctness": {"score": 10.0},  # Too high
            "naturalness": {"score": -1.0},  # Too low
            "task_completion": {"score": 3.0},
        }
        scores = _parse_judge_response(parsed)
        assert scores.tool_correctness.score == 5.0
        assert scores.naturalness.score == 1.0


class TestQualityThreshold:
    def test_passes(self):
        scores = JudgeScores(
            tool_correctness=JudgeScore(score=4.0),
            naturalness=JudgeScore(score=4.0),
            task_completion=JudgeScore(score=4.0),
        )
        assert passes_quality_threshold(scores, 3.5)

    def test_fails(self):
        scores = JudgeScores(
            tool_correctness=JudgeScore(score=2.0),
            naturalness=JudgeScore(score=2.0),
            task_completion=JudgeScore(score=2.0),
        )
        assert not passes_quality_threshold(scores, 3.5)

    def test_borderline(self):
        scores = JudgeScores(
            tool_correctness=JudgeScore(score=3.5),
            naturalness=JudgeScore(score=3.5),
            task_completion=JudgeScore(score=3.5),
        )
        assert passes_quality_threshold(scores, 3.5)


class TestFormatConversation:
    def test_format_with_tool_calls(self):
        conv = Conversation(
            conversation_id="test",
            messages=[
                Message(role="user", content="Find hotels"),
                Message(
                    role="assistant",
                    content=None,
                    tool_calls=[ToolCall(endpoint="hotel/search", arguments={"city": "Paris"})],
                ),
                Message(role="tool", content={"results": [{"id": "h1"}]}),
                Message(role="assistant", content="Found a hotel!"),
            ],
        )
        formatted = _format_conversation_for_judge(conv)
        assert "USER: Find hotels" in formatted
        assert "TOOL CALL" in formatted
        assert "TOOL RESULT" in formatted
        assert "Found a hotel" in formatted

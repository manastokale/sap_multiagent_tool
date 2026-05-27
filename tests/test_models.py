"""Tests for Pydantic data models."""

import pytest

from toolgen.models import (
    APIEndpoint,
    ArgumentSource,
    Conversation,
    HTTPMethod,
    JudgeScore,
    JudgeScores,
    Message,
    Parameter,
    ParameterType,
    ToolChain,
    ToolStepTrace,
)


class TestParameterType:
    def test_standard_types(self):
        assert ParameterType.from_raw("string") == ParameterType.STRING
        assert ParameterType.from_raw("STRING") == ParameterType.STRING
        assert ParameterType.from_raw("number") == ParameterType.NUMBER
        assert ParameterType.from_raw("integer") == ParameterType.INTEGER
        assert ParameterType.from_raw("boolean") == ParameterType.BOOLEAN
        assert ParameterType.from_raw("array") == ParameterType.ARRAY
        assert ParameterType.from_raw("object") == ParameterType.OBJECT

    def test_aliases(self):
        assert ParameterType.from_raw("str") == ParameterType.STRING
        assert ParameterType.from_raw("int") == ParameterType.INTEGER
        assert ParameterType.from_raw("float") == ParameterType.NUMBER
        assert ParameterType.from_raw("bool") == ParameterType.BOOLEAN
        assert ParameterType.from_raw("list") == ParameterType.ARRAY
        assert ParameterType.from_raw("dict") == ParameterType.OBJECT
        assert ParameterType.from_raw("json") == ParameterType.OBJECT

    def test_none_defaults_to_string(self):
        assert ParameterType.from_raw(None) == ParameterType.STRING

    def test_unknown(self):
        assert ParameterType.from_raw("weird_type") == ParameterType.UNKNOWN

    def test_whitespace(self):
        assert ParameterType.from_raw("  string  ") == ParameterType.STRING


class TestHTTPMethod:
    def test_standard(self):
        assert HTTPMethod.from_raw("GET") == HTTPMethod.GET
        assert HTTPMethod.from_raw("post") == HTTPMethod.POST
        assert HTTPMethod.from_raw("PUT") == HTTPMethod.PUT

    def test_none_defaults_to_get(self):
        assert HTTPMethod.from_raw(None) == HTTPMethod.GET

    def test_invalid_defaults_to_get(self):
        assert HTTPMethod.from_raw("INVALID") == HTTPMethod.GET


class TestAPIEndpoint:
    def test_endpoint_id_auto_computed(self):
        ep = APIEndpoint(
            tool_name="my_tool",
            endpoint_name="my_endpoint",
        )
        assert ep.endpoint_id == "my_tool/my_endpoint"

    def test_required_parameters(self):
        ep = APIEndpoint(
            tool_name="test",
            endpoint_name="test",
            parameters=[
                Parameter(name="a", required=True),
                Parameter(name="b", required=False),
                Parameter(name="c", required=True),
            ],
        )
        assert len(ep.required_parameters) == 2
        assert len(ep.optional_parameters) == 1

    def test_param_names(self):
        ep = APIEndpoint(
            tool_name="test",
            endpoint_name="test",
            parameters=[
                Parameter(name="city"),
                Parameter(name="date"),
            ],
        )
        assert ep.param_names == {"city", "date"}


class TestToolChain:
    def test_chain_properties(self, sample_endpoints):
        chain = ToolChain(endpoints=sample_endpoints)
        assert chain.num_steps == 3
        assert chain.num_distinct_tools == 2
        assert "Travel" in chain.categories
        assert len(chain.endpoint_ids) == 3

    def test_single_step(self):
        ep = APIEndpoint(tool_name="t", endpoint_name="e")
        chain = ToolChain(endpoints=[ep])
        assert chain.num_steps == 1
        assert chain.num_distinct_tools == 1


class TestJudgeScores:
    def test_overall_score(self):
        scores = JudgeScores(
            tool_correctness=JudgeScore(score=4.0),
            naturalness=JudgeScore(score=3.0),
            task_completion=JudgeScore(score=5.0),
        )
        assert scores.overall == 4.0

    def test_score_bounds(self):
        with pytest.raises(Exception):
            JudgeScore(score=0.0)  # Below minimum
        with pytest.raises(Exception):
            JudgeScore(score=6.0)  # Above maximum


class TestConversation:
    def test_to_output_dict(self):
        conv = Conversation(
            conversation_id="conv_0001",
            messages=[
                Message(role="user", content="Hello"),
                Message(role="assistant", content="Hi there!"),
            ],
            step_trace=[
                ToolStepTrace(
                    step=1,
                    endpoint="hotel/search",
                    goal="Find hotels",
                    argument_sources={
                        "city": ArgumentSource(source="user_request", value="Paris")
                    },
                    output_refs={"$.results[0].id": "hotel_123"},
                )
            ],
            judge_scores=JudgeScores(
                tool_correctness=JudgeScore(score=4.5),
                naturalness=JudgeScore(score=4.0),
                task_completion=JudgeScore(score=5.0),
            ),
        )
        d = conv.to_output_dict()
        assert d["conversation_id"] == "conv_0001"
        assert len(d["messages"]) == 2
        assert d["step_trace"][0]["argument_sources"]["city"]["source"] == "user_request"
        assert "overall" in d["judge_scores"]
        assert d["judge_scores"]["overall"] == 4.5

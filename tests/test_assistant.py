"""Tests for assistant tool-call parsing."""

from toolgen.agents.assistant import _try_parse_tool_call
from toolgen.models import APIEndpoint


def test_parse_tool_call_rejects_non_object_arguments():
    endpoint = APIEndpoint(tool_name="hotel_api", endpoint_name="search_hotels")
    text = (
        '{"action": "tool_call", "endpoint": "hotel_api/search_hotels", '
        '"arguments": ["city", "Paris"]}'
    )

    assert _try_parse_tool_call(text, [endpoint]) is None

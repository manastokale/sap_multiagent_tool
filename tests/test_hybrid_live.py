"""Tests for quota-saving hybrid live orchestration."""

from __future__ import annotations

import json

from toolgen.agents.llm_client import LLMResponse
from toolgen.agents.orchestrator import generate_conversation
from toolgen.executor.mock_executor import MockExecutor


class FakeLiveClient:
    is_offline = False

    def __init__(self, endpoint_ids: list[str]):
        self.endpoint_ids = endpoint_ids
        self.json_calls = 0
        self.free_text_calls = 0
        self.history_calls = 0

    def generate_json(self, *_args, **_kwargs):
        self.json_calls += 1
        return LLMResponse(
            text="{}",
            parsed={
                "scenario": "The user needs a multi-step travel workflow.",
                "user_persona": "A concise business user.",
                "expected_tool_sequence": self.endpoint_ids,
                "disambiguation_points": ["city", "date"],
                "complexity": "multi_step",
            },
            model="fake-live",
        )

    def generate(self, *_args, **_kwargs):
        self.free_text_calls += 1
        return LLMResponse(text="This should not be used in hybrid mode.", model="fake-live")

    def generate_with_history(self, *_args, **_kwargs):
        endpoint = self.endpoint_ids[self.history_calls]
        self.history_calls += 1
        return LLMResponse(
            text=json.dumps(
                {
                    "action": "tool_call",
                    "endpoint": endpoint,
                    "arguments": {},
                }
            ),
            model="fake-live",
        )


def test_hybrid_live_skips_user_and_summary_llm_calls(sample_chain):
    client = FakeLiveClient(sample_chain.endpoint_ids)

    conversation = generate_conversation(
        client=client,
        chain=sample_chain,
        executor=MockExecutor(seed=5),
        conversation_index=0,
        seed=5,
        live_profile="hybrid",
        model_name="fake-live",
    )

    assert conversation.metadata.generation_profile == "hybrid"
    assert conversation.metadata.num_tool_calls == len(sample_chain.endpoint_ids)
    assert client.json_calls == 1
    assert client.history_calls == len(sample_chain.endpoint_ids)
    assert client.free_text_calls == 0
    assert any(
        msg.role == "assistant"
        and isinstance(msg.content, str)
        and "received the result" in msg.content
        for msg in conversation.messages
    )

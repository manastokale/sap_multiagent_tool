"""Tests for Gemini-only live LLM routing."""

import json

from toolgen.agents.llm_client import LLMClient, LLMResponse, SharedRateLimiter
from toolgen.config import ToolGenSettings, get_settings, parse_model_pool


def test_parse_model_pool_deduplicates_and_preserves_order():
    assert parse_model_pool(
        "gemini-3.1-flash-lite, test-model, gemini-3.1-flash-lite"
    ) == [
        "gemini-3.1-flash-lite",
        "test-model",
    ]


def test_settings_use_randomized_generation_and_judge_pools():
    settings = get_settings(
        llm_provider="gemini",
        GEMINI_API_KEY="test-key",
        randomize_models=True,
        generation_model_pool="gen-a, gen-b, gen-a",
        judge_model_pool="judge-a, judge-b",
    )

    assert settings.generation_models == ["gen-a", "gen-b"]
    assert settings.judge_models == ["judge-a", "judge-b"]


def test_settings_can_disable_randomized_models():
    settings = get_settings(
        llm_provider="gemini",
        GEMINI_API_KEY="test-key",
        randomize_models=False,
        generation_model="gemini-3.1-flash-lite",
        judge_model="gemini-3.1-flash-lite",
    )

    assert settings.generation_models == ["gemini-3.1-flash-lite"]
    assert settings.judge_models == ["gemini-3.1-flash-lite"]


def test_explicit_models_override_randomized_default_pools():
    settings = get_settings(
        llm_provider="auto",
        GEMINI_API_KEY="gemini-key",
        GROQ_API_KEY="unused-key",
        randomize_models=True,
        generation_model="gemini-3.1-flash-lite",
        judge_model="gemini-3.1-flash-lite",
        generation_model_pool="",
        judge_model_pool="",
    )

    assert settings.generation_models == ["gemini-3.1-flash-lite"]
    assert settings.judge_models == ["gemini-3.1-flash-lite"]


def test_explicit_model_pool_still_enables_randomization():
    settings = get_settings(
        llm_provider="gemini",
        GEMINI_API_KEY="gemini-key",
        randomize_models=True,
        generation_model="gemini-3.1-flash-lite",
        generation_model_pool="gen-a,gen-b",
        judge_model="gemini-3.1-flash-lite",
        judge_model_pool="judge-a,judge-b",
    )

    assert settings.generation_models == ["gen-a", "gen-b"]
    assert settings.judge_models == ["judge-a", "judge-b"]


def test_role_specific_models_fall_back_to_generation_models():
    settings = get_settings(
        llm_provider="gemini",
        GEMINI_API_KEY="test-key",
        randomize_models=False,
        generation_model="gemini-3.1-flash-lite",
        judge_model="gemini-3.1-flash-lite",
    )

    assert settings.planner_models == ["gemini-3.1-flash-lite"]
    assert settings.assistant_models == ["gemini-3.1-flash-lite"]
    assert settings.user_models == ["gemini-3.1-flash-lite"]
    assert settings.summary_models == ["gemini-3.1-flash-lite"]


def test_role_specific_model_pools_override_generation_pool():
    settings = get_settings(
        llm_provider="gemini",
        GEMINI_API_KEY="test-key",
        randomize_models=True,
        generation_model_pool="gen-a,gen-b",
        planner_model_pool="planner-a",
        assistant_model_pool="assistant-a,assistant-b",
        user_model="user-a",
        user_model_pool="",
        summary_model="summary-a",
        summary_model_pool="",
        judge_model="gemini-3.1-flash-lite",
    )

    assert settings.planner_models == ["planner-a"]
    assert settings.assistant_models == ["assistant-a", "assistant-b"]
    assert settings.user_models == ["user-a"]
    assert settings.summary_models == ["summary-a"]


def test_dotenv_values_override_stale_shell_exports(tmp_path, monkeypatch):
    env_file = tmp_path / ".env"
    env_file.write_text(
        "\n".join(
            [
                "GROQ_API_KEY=dotenv-unused-key",
                "TOOLGEN_LLM_PROVIDER=gemini",
                "TOOLGEN_GENERATION_MODEL=gemini-3.1-flash-lite",
                "TOOLGEN_JUDGE_MODEL=gemini-3.1-flash-lite",
                "TOOLGEN_RANDOMIZE_MODELS=false",
            ]
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("GEMINI_API_KEY", "shell-gemini-key")
    monkeypatch.setenv("TOOLGEN_LLM_PROVIDER", "offline")
    monkeypatch.setenv("TOOLGEN_GENERATION_MODEL", "stale-shell-model")
    monkeypatch.setenv("TOOLGEN_JUDGE_MODEL", "stale-shell-judge")
    monkeypatch.setenv("TOOLGEN_RANDOMIZE_MODELS", "true")

    settings = ToolGenSettings(_env_file=env_file)

    assert settings.llm_provider == "gemini"
    assert settings.groq_api_key == "dotenv-unused-key"
    assert settings.randomize_models is False
    assert settings.generation_models == ["gemini-3.1-flash-lite"]
    assert settings.judge_models == ["gemini-3.1-flash-lite"]


def test_auto_provider_ignores_unused_non_gemini_key():
    settings = get_settings(
        llm_provider="auto",
        GEMINI_API_KEY="",
        GROQ_API_KEY="unused-key",
        randomize_models=False,
    )

    assert settings.use_offline_llm is True
    assert settings.generation_models == ["offline-deterministic"]


def test_settings_normalizes_hybrid_live_profile():
    settings = get_settings(llm_provider="offline", live_profile="hybrid-live")

    assert settings.normalized_live_profile == "hybrid"


def test_llm_client_model_selection_is_seeded():
    client_a = LLMClient(api_key="test-key", model_pool=["m1", "m2", "m3"], seed=7)
    client_b = LLMClient(api_key="test-key", model_pool=["m1", "m2", "m3"], seed=7)

    sequence_a = [client_a._choose_model() for _ in range(10)]
    sequence_b = [client_b._choose_model() for _ in range(10)]

    assert sequence_a == sequence_b
    assert len(set(sequence_a)) > 1


def test_llm_client_string_model_pool_normalizes():
    client = LLMClient(api_key="test-key", model_pool=" m1, m2, m1 ", seed=1)

    assert client.model_pool == ("m1", "m2")


def test_shared_rate_limiter_counts_request_starts():
    counts: list[int] = []
    limiter = SharedRateLimiter(requests_per_minute=0, on_request_start=counts.append)

    limiter.acquire()
    limiter.acquire()

    assert limiter.request_count == 2
    assert counts == [1, 2]


def test_llm_client_json_requests_use_json_mime_type(monkeypatch):
    captured = {}

    def fake_generate_rest(
        model_name,
        contents,
        system_instruction,
        temperature,
        response_mime_type=None,
    ):
        captured["response_mime_type"] = response_mime_type
        return LLMResponse(text='{"ok": true}', model=model_name)

    client = LLMClient(api_key="test-key", model="gemini-3.1-flash-lite")
    monkeypatch.setattr(client, "_generate_gemini_rest", fake_generate_rest)

    response = client.generate_json("Return JSON")

    assert captured["response_mime_type"] == "application/json"
    assert response.parsed == {"ok": True}


def test_gemini_key_is_sent_as_header_not_query_param(monkeypatch):
    captured = {}

    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def read(self):
            return b'{"candidates":[{"content":{"parts":[{"text":"ok"}]}}]}'

    def fake_urlopen(request, timeout):
        captured["url"] = request.full_url
        captured["timeout"] = timeout
        captured["payload"] = json.loads(request.data.decode("utf-8"))
        captured["headers"] = {
            key.lower(): value for key, value in request.header_items()
        }
        return FakeResponse()

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)

    client = LLMClient(
        api_key="test-key",
        model="gemini-3.1-flash-lite",
        max_output_tokens=321,
    )
    response = client.generate("hello")

    assert response.text == "ok"
    assert "key=" not in captured["url"]
    assert captured["headers"]["x-goog-api-key"] == "test-key"
    assert captured["payload"]["generationConfig"]["maxOutputTokens"] == 321


def test_llm_client_falls_back_after_unavailable_model(monkeypatch):
    attempts = []
    rate_limits = []

    class FakeRateLimiter:
        def acquire(self):
            rate_limits.append("acquire")

    def fake_generate_rest(
        model_name,
        contents,
        system_instruction,
        temperature,
        response_mime_type=None,
    ):
        attempts.append(model_name)
        if model_name == "gemini-stale-preview":
            return LLMResponse(error="Gemini HTTP 404: no longer available", model=model_name)
        return LLMResponse(text='{"ok": true}', model=model_name)

    client = LLMClient(
        api_key="test-key",
        provider="gemini",
        model_pool=["gemini-stale-preview", "gemini-3.1-flash-lite"],
        rate_limiter=FakeRateLimiter(),
    )
    monkeypatch.setattr(client, "_generate_gemini_rest", fake_generate_rest)
    monkeypatch.setattr(
        client,
        "_candidate_model_specs",
        lambda: [
            ("gemini", "gemini-stale-preview"),
            ("gemini", "gemini-3.1-flash-lite"),
        ],
    )

    response = client.generate_json("Return JSON")

    assert attempts == ["gemini-stale-preview", "gemini-3.1-flash-lite"]
    assert len(rate_limits) == 2
    assert response.model == "gemini-3.1-flash-lite"
    assert response.parsed == {"ok": True}

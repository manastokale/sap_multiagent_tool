"""Tests for live LLM model routing."""

from toolgen.agents.llm_client import LLMClient, LLMResponse
from toolgen.config import ToolGenSettings, get_settings, parse_model_pool


def test_parse_model_pool_deduplicates_and_preserves_order():
    assert parse_model_pool("gemini-3.5-flash, gemini-2.5-flash, gemini-3.5-flash") == [
        "gemini-3.5-flash",
        "gemini-2.5-flash",
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
        generation_model="fixed-gen",
        judge_model="fixed-judge",
    )

    assert settings.generation_models == ["fixed-gen"]
    assert settings.judge_models == ["fixed-judge"]


def test_explicit_models_override_randomized_default_pools():
    settings = get_settings(
        llm_provider="auto",
        GEMINI_API_KEY="gemini-key",
        GROQ_API_KEY="groq-key",
        randomize_models=True,
        generation_model="llama-3.1-8b-instant",
        judge_model="llama-3.1-8b-instant",
    )

    assert settings.generation_models == ["llama-3.1-8b-instant"]
    assert settings.judge_models == ["llama-3.1-8b-instant"]


def test_explicit_model_pool_still_enables_randomization():
    settings = get_settings(
        llm_provider="auto",
        GEMINI_API_KEY="gemini-key",
        GROQ_API_KEY="groq-key",
        randomize_models=True,
        generation_model="llama-3.1-8b-instant",
        generation_model_pool="llama-3.1-8b-instant,qwen/qwen3-32b",
        judge_model="llama-3.1-8b-instant",
        judge_model_pool="llama-3.1-8b-instant,qwen/qwen3-32b",
    )

    assert settings.generation_models == ["llama-3.1-8b-instant", "qwen/qwen3-32b"]
    assert settings.judge_models == ["llama-3.1-8b-instant", "qwen/qwen3-32b"]


def test_dotenv_values_override_stale_shell_exports(tmp_path, monkeypatch):
    env_file = tmp_path / ".env"
    env_file.write_text(
        "\n".join(
            [
                "GROQ_API_KEY=dotenv-groq-key",
                "TOOLGEN_LLM_PROVIDER=groq",
                "TOOLGEN_GENERATION_MODEL=llama-3.1-8b-instant",
                "TOOLGEN_JUDGE_MODEL=llama-3.1-8b-instant",
                "TOOLGEN_RANDOMIZE_MODELS=false",
            ]
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("GEMINI_API_KEY", "shell-gemini-key")
    monkeypatch.setenv("TOOLGEN_LLM_PROVIDER", "gemini")
    monkeypatch.setenv("TOOLGEN_GENERATION_MODEL", "gemini-3.1-flash-lite")
    monkeypatch.setenv("TOOLGEN_JUDGE_MODEL", "gemini-3.1-flash-lite")
    monkeypatch.setenv("TOOLGEN_RANDOMIZE_MODELS", "true")

    settings = ToolGenSettings(_env_file=env_file)

    assert settings.llm_provider == "groq"
    assert settings.randomize_models is False
    assert settings.generation_models == ["llama-3.1-8b-instant"]
    assert settings.judge_models == ["llama-3.1-8b-instant"]


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
        return LLMResponse(text='{"ok": true}', model="gemini-2.5-flash")

    client = LLMClient(api_key="test-key", model="gemini-2.5-flash")
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
        captured["headers"] = {
            key.lower(): value for key, value in request.header_items()
        }
        return FakeResponse()

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)

    client = LLMClient(api_key="test-key", model="gemini-2.5-flash")
    response = client.generate("hello")

    assert response.text == "ok"
    assert "key=" not in captured["url"]
    assert captured["headers"]["x-goog-api-key"] == "test-key"


def test_groq_request_includes_api_client_headers(monkeypatch):
    captured = {}

    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def read(self):
            return b'{"choices":[{"message":{"content":"ok"}}]}'

    def fake_urlopen(request, timeout):
        captured["url"] = request.full_url
        captured["headers"] = {
            key.lower(): value for key, value in request.header_items()
        }
        captured["timeout"] = timeout
        return FakeResponse()

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)

    client = LLMClient(
        api_keys={"groq": "test-key"},
        provider="groq",
        model_pool=["llama-3.1-8b-instant"],
    )
    response = client.generate("hello")

    assert response.text == "ok"
    assert captured["url"] == "https://api.groq.com/openai/v1/chat/completions"
    assert captured["headers"]["authorization"] == "Bearer test-key"
    assert captured["headers"]["accept"] == "application/json"
    assert captured["headers"]["user-agent"] == "toolgen/0.1 python-urllib"


def test_auto_settings_include_groq_models_when_groq_key_exists():
    settings = get_settings(
        llm_provider="auto",
        GEMINI_API_KEY="",
        GROQ_API_KEY="test-key",
        randomize_models=True,
        generation_model="",
        judge_model="",
    )

    assert any(model.startswith("llama-") or model.startswith("qwen/") for model in settings.generation_models)
    assert not settings.use_offline_llm


def test_llm_client_can_route_to_groq_json(monkeypatch):
    captured = {}

    def fake_groq_rest(
        model_name,
        messages,
        system_instruction,
        temperature,
        json_mode=False,
    ):
        captured["model_name"] = model_name
        captured["json_mode"] = json_mode
        return LLMResponse(text='{"ok": true}', model=model_name)

    client = LLMClient(
        api_keys={"groq": "test-key"},
        provider="groq",
        model_pool=["llama-3.1-8b-instant"],
    )
    monkeypatch.setattr(client, "_generate_groq_rest", fake_groq_rest)

    response = client.generate_json("Return JSON")

    assert captured == {"model_name": "llama-3.1-8b-instant", "json_mode": True}
    assert response.parsed == {"ok": True}


def test_llm_client_falls_back_after_unavailable_model(monkeypatch):
    attempts = []

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
        model_pool=["gemini-stale-preview", "gemini-2.5-flash-lite"],
    )
    monkeypatch.setattr(client, "_generate_gemini_rest", fake_generate_rest)
    monkeypatch.setattr(
        client,
        "_candidate_model_specs",
        lambda: [
            ("gemini", "gemini-stale-preview"),
            ("gemini", "gemini-2.5-flash-lite"),
        ],
    )

    response = client.generate_json("Return JSON")

    assert attempts == ["gemini-stale-preview", "gemini-2.5-flash-lite"]
    assert response.model == "gemini-2.5-flash-lite"
    assert response.parsed == {"ok": True}

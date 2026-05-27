"""Central configuration for the ToolGen pipeline."""

from __future__ import annotations

import os
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, PydanticBaseSettingsSource, SettingsConfigDict


def _default_project_root() -> Path:
    """Resolve the project root for both source and installed CLI execution."""
    explicit_root = os.environ.get("TOOLGEN_PROJECT_ROOT", "").strip()
    if explicit_root:
        return Path(explicit_root).expanduser().resolve()

    for candidate in [Path.cwd(), *Path.cwd().parents]:
        if (
            (candidate / "data" / "toolenv" / "tools").exists()
            or (candidate / "pyproject.toml").exists()
        ):
            return candidate

    source_root = Path(__file__).resolve().parent.parent.parent
    if (source_root / "data" / "toolenv" / "tools").exists():
        return source_root

    return Path.cwd()


_PROJECT_ROOT = _default_project_root()
_ENV_FILE = _PROJECT_ROOT / ".env"

DEFAULT_GENERATION_MODEL = "gemini-2.5-flash"
DEFAULT_JUDGE_MODEL = "gemini-2.5-flash-lite"
DEFAULT_GEMINI_GENERATION_MODEL_POOL = (
    "gemini-3.1-flash-lite,"
    "gemini-2.5-flash,"
    "gemini-2.5-flash-lite"
)
DEFAULT_GEMINI_JUDGE_MODEL_POOL = "gemini-2.5-flash-lite,gemini-3.1-flash-lite"
DEFAULT_GROQ_GENERATION_MODEL_POOL = (
    "llama-3.1-8b-instant,"
    "llama-3.3-70b-versatile,"
    "qwen/qwen3-32b"
)
DEFAULT_GROQ_JUDGE_MODEL_POOL = "llama-3.1-8b-instant,qwen/qwen3-32b"


def parse_model_pool(value: str | list[str] | tuple[str, ...]) -> list[str]:
    """Parse a comma-delimited model pool while preserving order."""
    raw_models = value.replace("\n", ",").split(",") if isinstance(value, str) else value
    models: list[str] = []
    seen: set[str] = set()
    for raw_model in raw_models:
        model = str(raw_model).strip()
        if model and model not in seen:
            models.append(model)
            seen.add(model)
    return models


class ToolGenSettings(BaseSettings):
    """Settings loaded from environment / .env file."""

    model_config = SettingsConfigDict(
        env_prefix="TOOLGEN_",
        env_file=_ENV_FILE,
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- API keys ---
    gemini_api_key: str = Field(
        default="",
        alias="GEMINI_API_KEY",
        description="Gemini API key for LLM calls",
    )
    groq_api_key: str = Field(
        default="",
        alias="GROQ_API_KEY",
        description="Groq API key for LLM calls",
    )

    # --- Optional model hints commonly copied from other projects ---
    gemini_dialogue_model: str = Field(default="", alias="GEMINI_DIALOGUE_MODEL")
    gemini_dialogue_fallback_model: str = Field(
        default="", alias="GEMINI_DIALOGUE_FALLBACK_MODEL"
    )
    gemini_converge_model: str = Field(default="", alias="GEMINI_CONVERGE_MODEL")
    gemini_summary_model: str = Field(default="", alias="GEMINI_SUMMARY_MODEL")
    gemini_ask_model: str = Field(default="", alias="GEMINI_ASK_MODEL")
    gemini_branch_plan_model: str = Field(default="", alias="GEMINI_BRANCH_PLAN_MODEL")
    gemini_branch_scene_model: str = Field(default="", alias="GEMINI_BRANCH_SCENE_MODEL")

    # --- Model selection ---
    llm_provider: str = Field(
        default="auto",
        description="LLM provider: 'auto', 'offline', 'gemini', or 'groq'",
    )
    generation_model: str = Field(
        default=DEFAULT_GENERATION_MODEL,
        description="Model for conversation generation (planner, user sim, assistant)",
    )
    judge_model: str = Field(
        default=DEFAULT_JUDGE_MODEL,
        description="Model for LLM-as-judge scoring (cheaper, simpler task)",
    )
    randomize_models: bool = Field(
        default=True,
        description="Randomly pick from the configured provider-aware model pools per request",
    )
    generation_model_pool: str = Field(
        default="",
        description="Comma-separated generation models used when randomize_models is true",
    )
    judge_model_pool: str = Field(
        default="",
        description="Comma-separated judge models used when randomize_models is true",
    )
    require_live_llm: bool = Field(
        default=False,
        description="Fail instead of using offline or heuristic fallbacks for LLM generation/judging",
    )
    live_profile: str = Field(
        default="full",
        description="Live orchestration profile: 'full' or 'hybrid'",
    )

    # --- Paths ---
    data_dir: Path = Field(
        default=_PROJECT_ROOT / "data",
        description="Root directory for ToolBench data",
    )
    output_dir: Path = Field(
        default=_PROJECT_ROOT / "output",
        description="Directory for generated datasets",
    )
    artifacts_dir: Path = Field(
        default=_PROJECT_ROOT / "output" / "artifacts",
        description="Directory for derived registry and graph artifacts",
    )
    toolenv_dir: Path = Field(
        default=_PROJECT_ROOT / "data" / "toolenv" / "tools",
        description="Path to ToolBench toolenv/tools directory",
    )

    # --- Generation parameters ---
    default_seed: int = Field(default=42, description="Default random seed")
    default_num_conversations: int = Field(
        default=100, description="Default number of conversations to generate"
    )
    max_turns_per_conversation: int = Field(
        default=15, description="Maximum turns in a single conversation"
    )
    quality_threshold: float = Field(
        default=3.5, description="Minimum LLM-as-judge overall score to pass"
    )
    max_repair_attempts: int = Field(
        default=2, description="Maximum repair attempts per conversation"
    )

    # --- Rate limiting ---
    llm_requests_per_minute: int = Field(
        default=30, description="Rate limit for LLM API calls"
    )
    llm_request_timeout_seconds: int = Field(
        default=30, description="Timeout for a single LLM API request"
    )

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        """Prefer the project .env file over stale exported shell variables.

        The CLI is used as a local assessment runner, so the checked project .env should be
        the predictable source of truth. Explicit constructor overrides still win in tests
        and programmatic use.
        """
        return init_settings, dotenv_settings, env_settings, file_secret_settings

    @property
    def project_root(self) -> Path:
        return _PROJECT_ROOT

    @property
    def env_file(self) -> Path:
        return _ENV_FILE

    @property
    def use_offline_llm(self) -> bool:
        """Whether to use deterministic local generation and heuristic judging."""
        return not self.live_llm_enabled

    @property
    def live_llm_enabled(self) -> bool:
        """Whether at least one configured live provider can be used."""
        provider = self.llm_provider.strip().lower()
        if provider in {"offline", "local", "deterministic"}:
            return False
        if provider == "gemini":
            return bool(self.gemini_api_key)
        if provider == "groq":
            return bool(self.groq_api_key)
        return bool(self.gemini_api_key or self.groq_api_key)

    @property
    def api_keys(self) -> dict[str, str]:
        """Provider API keys, with empty values preserved for diagnostics."""
        return {
            "gemini": self.gemini_api_key,
            "groq": self.groq_api_key,
        }

    @property
    def generation_models(self) -> list[str]:
        """Generation models to hand to the LLM client."""
        if self.use_offline_llm:
            return ["offline-deterministic"]
        if not self.randomize_models:
            return [self.generation_model]
        return self._models_for_role("generation")

    @property
    def judge_models(self) -> list[str]:
        """Judge models to hand to the LLM client."""
        if self.use_offline_llm:
            return ["offline-heuristic"]
        if not self.randomize_models:
            return [self.judge_model]
        return self._models_for_role("judge")

    @property
    def normalized_live_profile(self) -> str:
        profile = self.live_profile.strip().lower().replace("-", "_")
        if profile in {"hybrid", "hybrid_live", "quota_saver", "fast"}:
            return "hybrid"
        return "full"

    def _models_for_role(self, role: str) -> list[str]:
        provider = self.llm_provider.strip().lower()
        explicit_pool = (
            parse_model_pool(self.generation_model_pool)
            if role == "generation"
            else parse_model_pool(self.judge_model_pool)
        )
        if explicit_pool:
            return explicit_pool

        explicit_model = self._explicit_model_for_role(role)
        if explicit_model:
            return [explicit_model]

        models: list[str] = []
        if provider in {"auto", "gemini"} and self.gemini_api_key:
            models.extend(self._gemini_hint_models(role))
            models.extend(
                parse_model_pool(
                    DEFAULT_GEMINI_GENERATION_MODEL_POOL
                    if role == "generation"
                    else DEFAULT_GEMINI_JUDGE_MODEL_POOL
                )
            )
        if provider in {"auto", "groq"} and self.groq_api_key:
            models.extend(
                parse_model_pool(
                    DEFAULT_GROQ_GENERATION_MODEL_POOL
                    if role == "generation"
                    else DEFAULT_GROQ_JUDGE_MODEL_POOL
                )
            )
        deduped = parse_model_pool(models)
        if deduped:
            return deduped
        return [self.generation_model if role == "generation" else self.judge_model]

    def _explicit_model_for_role(self, role: str) -> str:
        """Return a model explicitly set by env/overrides, ignoring class defaults."""
        field_name = "generation_model" if role == "generation" else "judge_model"
        if field_name not in self.model_fields_set:
            return ""
        value = getattr(self, field_name, "")
        return value.strip() if isinstance(value, str) else ""

    def _gemini_hint_models(self, role: str) -> list[str]:
        if role == "generation":
            return parse_model_pool(
                [
                    self.gemini_dialogue_model,
                    self.gemini_dialogue_fallback_model,
                    self.gemini_branch_plan_model,
                    self.gemini_branch_scene_model,
                    self.gemini_summary_model,
                ]
            )
        return parse_model_pool(
            [
                self.gemini_ask_model,
                self.gemini_converge_model,
                self.gemini_dialogue_fallback_model,
            ]
        )


def get_settings(**overrides) -> ToolGenSettings:
    """Create settings instance, optionally overriding values."""
    return ToolGenSettings(**overrides)

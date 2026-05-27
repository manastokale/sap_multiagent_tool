"""LLM client abstraction layer.

Wraps live Gemini and Groq calls with retry, rate limiting, provider fallback, and
structured output support. Designed to be swappable: the rest of the codebase only sees
this interface.
"""

from __future__ import annotations

import json
import logging
import random
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Sequence
from typing import Any

from pydantic import BaseModel

logger = logging.getLogger(__name__)


class LLMResponse(BaseModel):
    """Standardized LLM response."""
    text: str = ""
    parsed: dict[str, Any] | None = None
    model: str = ""
    usage: dict[str, Any] | None = None
    error: str | None = None


class LLMClient:
    """Live LLM client with retry, provider fallback, and structured output support.

    Usage:
        client = LLMClient(api_key="...", model="gemini-2.5-flash")
        response = client.generate("Tell me about Paris")
        structured = client.generate_json("Plan a scenario")
    """

    def __init__(
        self,
        api_key: str = "",
        model: str = "gemini-2.5-flash",
        model_pool: Sequence[str] | str | None = None,
        provider: str = "auto",
        api_keys: dict[str, str] | None = None,
        groq_api_key: str = "",
        requests_per_minute: int = 30,
        max_retries: int = 3,
        request_timeout_seconds: int = 30,
        seed: int | None = None,
    ):
        self._model_pool = _normalize_model_pool(model_pool if model_pool is not None else [model])
        self._model_name = self._model_pool[0]
        self._provider_mode = provider.strip().lower()
        self._api_keys = {
            "gemini": api_key,
            "groq": groq_api_key,
            **(api_keys or {}),
        }
        self._rng = random.Random(seed)
        self._rpm = requests_per_minute
        self._max_retries = max_retries
        self._request_timeout_seconds = request_timeout_seconds
        self._last_request_time = 0.0
        self._disabled_model_specs: set[tuple[str, str]] = set()

    @property
    def model_pool(self) -> tuple[str, ...]:
        """Models available to this client."""
        return tuple(self._model_pool)

    @property
    def is_offline(self) -> bool:
        """True when no remote LLM should be called."""
        return not self._available_model_specs()

    def _choose_model(self) -> str:
        """Choose a model for the next remote request."""
        specs = self._available_model_specs()
        if not specs:
            return self._model_name
        if len(specs) == 1:
            provider, model_name = specs[0]
        else:
            provider, model_name = self._rng.choice(specs)
        self._model_name = model_name
        logger.debug("Selected %s model: %s", provider, model_name)
        return self._model_name

    def _candidate_model_specs(self) -> list[tuple[str, str]]:
        specs = self._available_model_specs()
        if len(specs) <= 1:
            return specs
        primary = self._rng.choice(specs)
        remaining = [spec for spec in specs if spec != primary]
        self._rng.shuffle(remaining)
        return [primary, *remaining]

    def _available_model_specs(self) -> list[tuple[str, str]]:
        if self._provider_mode in {"offline", "local", "deterministic"}:
            return []

        specs: list[tuple[str, str]] = []
        for raw_model in self._model_pool:
            if raw_model.startswith("offline"):
                continue
            provider, model_name = _split_provider_model(raw_model, self._provider_mode)
            if (
                provider
                and self._api_keys.get(provider)
                and (provider, model_name) not in self._disabled_model_specs
            ):
                specs.append((provider, model_name))
        return specs

    def _rate_limit(self) -> None:
        """Simple rate limiter based on RPM."""
        if self._rpm <= 0:
            return
        min_interval = 60.0 / self._rpm
        elapsed = time.time() - self._last_request_time
        if elapsed < min_interval:
            time.sleep(min_interval - elapsed)
        self._last_request_time = time.time()

    def _generate_gemini_rest(
        self,
        model_name: str,
        contents: list[dict[str, Any]],
        system_instruction: str | None,
        temperature: float,
        response_mime_type: str | None = None,
    ) -> LLMResponse:
        """Call Gemini via REST with a real timeout.

        The older ``google-generativeai`` SDK can hang without respecting
        request options in some local Python versions. The REST API keeps the
        dependency surface smaller and makes timeout behavior explicit.
        """
        model_path = (
            model_name
            if model_name.startswith("models/")
            else f"models/{model_name}"
        )
        encoded_model = urllib.parse.quote(model_path, safe="/")
        url = (
            "https://generativelanguage.googleapis.com/v1beta/"
            f"{encoded_model}:generateContent"
        )

        generation_config: dict[str, Any] = {"temperature": temperature}
        if response_mime_type:
            generation_config["responseMimeType"] = response_mime_type

        payload: dict[str, Any] = {
            "contents": contents,
            "generationConfig": generation_config,
        }
        if system_instruction:
            payload["systemInstruction"] = {
                "parts": [{"text": system_instruction}],
            }

        request = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "x-goog-api-key": self._api_keys["gemini"],
            },
            method="POST",
        )

        try:
            with urllib.request.urlopen(
                request,
                timeout=self._request_timeout_seconds,
            ) as response:
                raw = response.read().decode("utf-8")
        except urllib.error.HTTPError as e:
            body = e.read().decode("utf-8", errors="replace")
            message = _extract_gemini_error(body) or e.reason
            return LLMResponse(
                error=f"Gemini HTTP {e.code}: {message}",
                model=model_name,
            )
        except urllib.error.URLError as e:
            return LLMResponse(error=f"Gemini request failed: {e.reason}", model=model_name)
        except TimeoutError:
            return LLMResponse(error="Gemini request timed out", model=model_name)

        try:
            data = json.loads(raw)
        except json.JSONDecodeError as e:
            return LLMResponse(error=f"Gemini response JSON parse error: {e}", model=model_name)

        text_parts: list[str] = []
        for candidate in data.get("candidates", []):
            content = candidate.get("content", {})
            for part in content.get("parts", []):
                if "text" in part:
                    text_parts.append(str(part["text"]))

        usage = data.get("usageMetadata")
        return LLMResponse(
            text="\n".join(text_parts).strip(),
            model=model_name,
            usage=usage if isinstance(usage, dict) else None,
        )

    def _generate_groq_rest(
        self,
        model_name: str,
        messages: list[dict[str, str]],
        system_instruction: str | None,
        temperature: float,
        json_mode: bool = False,
    ) -> LLMResponse:
        """Call Groq through its OpenAI-compatible chat completions API."""
        groq_messages: list[dict[str, str]] = []
        if system_instruction:
            groq_messages.append({"role": "system", "content": system_instruction})
        groq_messages.extend(messages or [{"role": "user", "content": ""}])

        payload: dict[str, Any] = {
            "model": model_name,
            "messages": groq_messages,
            "temperature": temperature,
        }
        if json_mode:
            payload["response_format"] = {"type": "json_object"}

        request = urllib.request.Request(
            "https://api.groq.com/openai/v1/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self._api_keys['groq']}",
                "Content-Type": "application/json",
                "Accept": "application/json",
                "User-Agent": "toolgen/0.1 python-urllib",
            },
            method="POST",
        )

        try:
            with urllib.request.urlopen(
                request,
                timeout=self._request_timeout_seconds,
            ) as response:
                raw = response.read().decode("utf-8")
        except urllib.error.HTTPError as e:
            body = e.read().decode("utf-8", errors="replace")
            message = _extract_openai_compatible_error(body) or e.reason
            return LLMResponse(error=f"Groq HTTP {e.code}: {message}", model=model_name)
        except urllib.error.URLError as e:
            return LLMResponse(error=f"Groq request failed: {e.reason}", model=model_name)
        except TimeoutError:
            return LLMResponse(error="Groq request timed out", model=model_name)

        try:
            data = json.loads(raw)
        except json.JSONDecodeError as e:
            return LLMResponse(error=f"Groq response JSON parse error: {e}", model=model_name)

        choices = data.get("choices") or []
        text = ""
        if choices and isinstance(choices[0], dict):
            message = choices[0].get("message") or {}
            text = str(message.get("content") or "")

        usage = data.get("usage")
        return LLMResponse(
            text=text.strip(),
            model=model_name,
            usage=usage if isinstance(usage, dict) else None,
        )

    def _generate_live(
        self,
        messages: list[dict[str, str]],
        system_instruction: str | None,
        temperature: float,
        json_mode: bool = False,
    ) -> LLMResponse:
        last_response: LLMResponse | None = None
        for attempt in range(self._max_retries):
            self._rate_limit()
            retry_after_seconds = 0.0
            for provider, model_name in self._candidate_model_specs():
                self._model_name = model_name
                if provider == "gemini":
                    contents = _messages_to_gemini_contents(messages)
                    response = self._generate_gemini_rest(
                        model_name=model_name,
                        contents=contents,
                        system_instruction=system_instruction,
                        temperature=temperature,
                        response_mime_type="application/json" if json_mode else None,
                    )
                elif provider == "groq":
                    response = self._generate_groq_rest(
                        model_name=model_name,
                        messages=messages,
                        system_instruction=system_instruction,
                        temperature=temperature,
                        json_mode=json_mode,
                    )
                else:
                    continue

                if not response.error:
                    return response

                last_response = response
                logger.warning("LLM request failed on %s/%s: %s", provider, model_name, response.error)
                if _is_permanent_model_error(response.error):
                    self._disabled_model_specs.add((provider, model_name))
                retry_after_seconds = max(
                    retry_after_seconds,
                    _retry_after_seconds(response.error),
                )
                if not _can_try_next_model(response.error):
                    break

            if attempt < self._max_retries - 1:
                time.sleep(max(float(2 ** attempt), retry_after_seconds))

        return last_response or LLMResponse(error="No live LLM providers available", model=self._model_name)

    def generate(
        self,
        prompt: str,
        system_instruction: str | None = None,
        temperature: float = 0.7,
    ) -> LLMResponse:
        """Generate a free-text response."""
        if self.is_offline:
            return LLMResponse(
                error="offline LLM mode does not generate free text",
                model=self._model_name,
            )

        return self._generate_live(
            messages=[{"role": "user", "content": prompt}],
            system_instruction=system_instruction,
            temperature=temperature,
            json_mode=False,
        )

    def generate_json(
        self,
        prompt: str,
        system_instruction: str | None = None,
        temperature: float = 0.3,
    ) -> LLMResponse:
        """Generate a response and parse it as JSON.

        Uses a lower temperature by default for more deterministic structured output.
        Wraps the prompt with JSON formatting instructions.
        """
        json_prompt = (
            f"{prompt}\n\n"
            "IMPORTANT: Respond ONLY with valid JSON. No markdown, no explanation, "
            "no code fences. Just the raw JSON object."
        )

        if self.is_offline:
            return LLMResponse(
                error="offline LLM mode does not generate structured responses",
                model=self._model_name,
            )

        response = self._generate_live(
            messages=[{"role": "user", "content": json_prompt}],
            system_instruction=system_instruction,
            temperature=temperature,
            json_mode=True,
        )

        if response.error:
            return response

        # Try to parse JSON from response
        text = response.text.strip()

        # Strip markdown code fences if present
        if text.startswith("```"):
            lines = text.split("\n")
            # Remove first and last lines if they're code fences
            if lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            text = "\n".join(lines)

        try:
            parsed = json.loads(text)
            response.parsed = parsed
        except json.JSONDecodeError as e:
            logger.warning("Failed to parse JSON from LLM response: %s", e)
            # Try to extract JSON from the response
            start = text.find("{")
            end = text.rfind("}") + 1
            if start >= 0 and end > start:
                try:
                    parsed = json.loads(text[start:end])
                    response.parsed = parsed
                except json.JSONDecodeError:
                    response.error = f"JSON parse error: {e}"
            else:
                response.error = f"JSON parse error: {e}"

        return response

    def generate_with_history(
        self,
        messages: list[dict[str, str]],
        system_instruction: str | None = None,
        temperature: float = 0.7,
    ) -> LLMResponse:
        """Generate a response given a conversation history.

        Messages should be in format: [{"role": "user"/"model", "content": "..."}]
        """
        if self.is_offline:
            return LLMResponse(
                error="offline LLM mode does not generate chat responses",
                model=self._model_name,
            )

        return self._generate_live(
            messages=_normalize_chat_messages(messages),
            system_instruction=system_instruction,
            temperature=temperature,
            json_mode=False,
        )


def _normalize_model_pool(model_pool: Sequence[str] | str) -> list[str]:
    raw_models = (
        model_pool.replace("\n", ",").split(",")
        if isinstance(model_pool, str)
        else model_pool
    )
    models: list[str] = []
    seen: set[str] = set()
    for raw_model in raw_models:
        model = str(raw_model).strip()
        if model and model not in seen:
            models.append(model)
            seen.add(model)
    return models or ["gemini-2.5-flash"]


def _extract_gemini_error(body: str) -> str:
    try:
        data = json.loads(body)
    except json.JSONDecodeError:
        return body[:500]

    error = data.get("error")
    if isinstance(error, dict):
        message = error.get("message")
        if message:
            return str(message)
    return body[:500]


def _extract_openai_compatible_error(body: str) -> str:
    try:
        data = json.loads(body)
    except json.JSONDecodeError:
        return body[:500]

    error = data.get("error")
    if isinstance(error, dict):
        message = error.get("message")
        if message:
            return str(message)
    if isinstance(error, str):
        return error
    return body[:500]


def _split_provider_model(raw_model: str, provider_mode: str) -> tuple[str | None, str]:
    model = raw_model.strip()
    if ":" in model:
        prefix, candidate = model.split(":", 1)
        if prefix in {"gemini", "groq"}:
            return prefix, candidate

    if provider_mode in {"gemini", "groq"}:
        return provider_mode, model

    if model.startswith("models/gemini") or model.startswith("gemini-"):
        return "gemini", model
    if model.startswith(
        (
            "llama-",
            "qwen/",
            "meta-llama/",
            "openai/",
            "deepseek/",
            "mixtral-",
            "compound-",
        )
    ):
        return "groq", model
    if "/" in model:
        return "groq", model
    return "gemini", model


def _normalize_chat_messages(messages: list[dict[str, str]]) -> list[dict[str, str]]:
    normalized: list[dict[str, str]] = []
    for msg in messages:
        role = msg.get("role", "user")
        content = str(msg.get("content", ""))
        if role == "model":
            role = "assistant"
        if role not in {"user", "assistant"}:
            role = "user"
        normalized.append({"role": role, "content": content})
    return normalized


def _messages_to_gemini_contents(messages: list[dict[str, str]]) -> list[dict[str, Any]]:
    contents: list[dict[str, Any]] = []
    for msg in messages:
        role = "model" if msg.get("role") in {"assistant", "model"} else "user"
        contents.append({"role": role, "parts": [{"text": str(msg.get("content", ""))}]})
    return contents or [{"role": "user", "parts": [{"text": ""}]}]


def _can_try_next_model(error: str) -> bool:
    retryable_fragments = (
        "HTTP 400:",
        "HTTP 401:",
        "HTTP 403:",
        "HTTP 404:",
        "HTTP 408:",
        "HTTP 409:",
        "HTTP 429:",
        "HTTP 500:",
        "HTTP 502:",
        "HTTP 503:",
        "request failed:",
        "timed out",
    )
    return any(fragment in error for fragment in retryable_fragments)


def _is_permanent_model_error(error: str) -> bool:
    permanent_fragments = (
        "HTTP 400:",
        "HTTP 403:",
        "HTTP 404:",
        "no longer available",
        "model_not_found",
    )
    return any(fragment in error for fragment in permanent_fragments)


def _retry_after_seconds(error: str) -> float:
    """Extract provider retry hints from quota errors."""
    match = re.search(r"retry in ([0-9]+(?:\.[0-9]+)?)s", error, flags=re.IGNORECASE)
    if not match:
        return 0.0
    return min(float(match.group(1)) + 1.0, 120.0)

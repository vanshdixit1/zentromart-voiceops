from __future__ import annotations

import json
import threading
import time
import urllib.error
import urllib.request


class LLMError(RuntimeError):
    pass


def _http_error_message(exc: urllib.error.HTTPError) -> str:
    """Return the provider's useful error detail without exposing request headers."""
    try:
        payload = json.loads(exc.read().decode("utf-8", errors="replace"))
        detail = payload.get("error", {})
        if isinstance(detail, dict):
            message = detail.get("message") or detail.get("code") or detail.get("type")
        else:
            message = str(detail)
        if message:
            return f"HTTP {exc.code}: {message}"
    except (json.JSONDecodeError, OSError, AttributeError):
        pass
    finally:
        exc.close()
    return f"HTTP {exc.code}: {exc.reason}"


class GeminiChatModel:
    """Google AI Studio chat client backed by the official Google GenAI SDK."""

    def __init__(self, api_key: str, model: str, timeout: float = 60,
                 min_request_interval: float = 0, fallback_api_key: str = ""):
        if not api_key:
            raise LLMError("Set GEMINI_API_KEY in .env before making live Gemini API calls.")
        if not model:
            raise LLMError("Set GEMINI_MODEL to a Gemini model ID before making API calls.")
        try:
            from google import genai
            from google.genai import types
        except ImportError as exc:
            raise LLMError("Install the Google GenAI SDK with: python -m pip install google-genai") from exc

        self.model = model
        self._types = types
        self.timeout = timeout
        self.fallback_api_key = fallback_api_key if fallback_api_key != api_key else ""
        self.min_request_interval = max(0.0, float(min_request_interval))
        self._last_request_at = 0.0
        self._request_lock = threading.Lock()
        try:
            self.client = genai.Client(
                api_key=api_key,
                vertexai=False,
                http_options=types.HttpOptions(
                    timeout=int(timeout * 1000),
                    retry_options=types.HttpRetryOptions(attempts=1),
                ),
            )
        except (TypeError, ValueError) as exc:
            raise LLMError(f"Could not initialize the Google AI Studio Gemini client: {exc}") from exc
        self.fallback_client = None
        if self.fallback_api_key:
            try:
                self.fallback_client = genai.Client(
                    api_key=self.fallback_api_key,
                    vertexai=False,
                    http_options=types.HttpOptions(
                        timeout=int(timeout * 1000),
                        retry_options=types.HttpRetryOptions(attempts=1),
                    ),
                )
            except (TypeError, ValueError) as exc:
                raise LLMError(f"Could not initialize the fallback Google AI Studio client: {exc}") from exc

    def complete(self, messages: list[dict[str, str]], *, json_mode: bool = False) -> str:
        system_instruction = "\n\n".join(
            str(message.get("content", "")) for message in messages if message.get("role") == "system"
        ).strip()
        contents = []
        for message in messages:
            role = message.get("role", "user")
            if role == "system":
                continue
            contents.append(self._types.Content(
                role="model" if role == "assistant" else "user",
                parts=[self._types.Part.from_text(text=str(message.get("content", "")))],
            ))
        if not contents:
            raise LLMError("Gemini request has no user or conversation content.")

        config_args = {}
        if system_instruction:
            config_args["system_instruction"] = system_instruction
        if json_mode:
            config_args["response_mime_type"] = "application/json"
        config = self._types.GenerateContentConfig(**config_args) if config_args else None
        try:
            response = self._generate(self.client, contents, config)
        except Exception as primary_error:
            if not self._is_quota_error(primary_error):
                raise self._request_error(primary_error) from primary_error
            if self.fallback_client is None:
                raise LLMError(
                    "Gemini quota/rate limit reached for GEMINI_API_KEY. Configure "
                    "GEMINI_FALLBACK_API_KEY from a different Google AI Studio project, or wait for reset."
                ) from primary_error
            try:
                response = self._generate(self.fallback_client, contents, config)
            except Exception as fallback_error:
                if self._is_quota_error(fallback_error):
                    raise LLMError(
                        "Both Gemini projects are at their quota/rate limit; requests were not retried further."
                    ) from fallback_error
                raise LLMError(
                    "Primary Gemini project hit its quota and the fallback request failed: "
                    f"{self._error_detail(fallback_error)}"
                ) from fallback_error

        try:
            text = response.text
        except (AttributeError, ValueError):
            text = None
        if not isinstance(text, str) or not text.strip():
            action_text = self._function_call_as_action(response)
            if action_text:
                return action_text
            finish = ""
            candidates = getattr(response, "candidates", None) or []
            if candidates:
                finish = str(getattr(candidates[0], "finish_reason", ""))
            detail = f" (finish reason: {finish})" if finish else ""
            raise LLMError(f"Gemini returned no text content{detail}.")
        return text

    def _generate(self, client, contents, config):
        with self._request_lock:
            if self.min_request_interval:
                elapsed = time.monotonic() - self._last_request_at
                delay = self.min_request_interval - elapsed
                if delay > 0:
                    time.sleep(delay)
            self._last_request_at = time.monotonic()
            return client.models.generate_content(
                model=self.model, contents=contents, config=config,
            )

    @staticmethod
    def _is_quota_error(exc: Exception) -> bool:
        code = getattr(exc, "code", None)
        status = str(getattr(exc, "status", "")).upper()
        if code == 429 or str(code) == "429" or status in {"RESOURCE_EXHAUSTED", "TOO_MANY_REQUESTS"}:
            return True
        message = str(exc).upper()
        return "429" in message and any(term in message for term in ("QUOTA", "RESOURCE_EXHAUSTED", "RATE LIMIT"))

    @staticmethod
    def _error_detail(exc: Exception) -> str:
        return str(exc).strip() or type(exc).__name__

    @classmethod
    def _request_error(cls, exc: Exception) -> LLMError:
        detail = cls._error_detail(exc)
        if ("504" in detail or "DEADLINE_EXCEEDED" in detail.upper() or
                "TIMED OUT" in detail.upper() or isinstance(exc, TimeoutError)):
            return LLMError(f"Gemini request timed out; no automatic retry was made: {detail}")
        return LLMError(f"Gemini request failed ({type(exc).__name__}): {detail}")

    @staticmethod
    def _function_call_as_action(response) -> str:
        """Translate Gemini function-call parts into the agent's existing JSON action protocol."""
        calls = list(getattr(response, "function_calls", None) or [])
        for candidate in getattr(response, "candidates", None) or []:
            content = getattr(candidate, "content", None)
            for part in getattr(content, "parts", None) or []:
                call = getattr(part, "function_call", None)
                if call is not None:
                    calls.append(call)
        for call in calls:
            name = str(getattr(call, "name", "") or "")
            if name not in {"lookup_order", "create_ticket"}:
                continue
            arguments = getattr(call, "args", None)
            if not isinstance(arguments, dict):
                arguments = {}
            return json.dumps({"action": name, **arguments}, ensure_ascii=False)
        return ""


class EmbeddingClient:
    """OpenAI-compatible embeddings client used to build the local in-memory vector index."""

    def __init__(self, api_key: str, base_url: str, model: str, timeout: float = 60):
        self.api_key, self.base_url, self.model, self.timeout = api_key, base_url.rstrip("/"), model, timeout

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not self.api_key or not self.model:
            raise LLMError("Set OPENAI_API_KEY and OPENAI_EMBEDDING_MODEL to enable embeddings.")
        body = {"model": self.model, "input": texts}
        request = urllib.request.Request(
            f"{self.base_url}/embeddings", data=json.dumps(body).encode(),
            headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                data = json.loads(response.read())
            ordered = sorted(data["data"], key=lambda row: row.get("index", 0))
            return [row["embedding"] for row in ordered]
        except urllib.error.HTTPError as exc:
            raise LLMError(f"Embedding request failed: {_http_error_message(exc)}") from exc
        except (urllib.error.URLError, TimeoutError, KeyError, IndexError, json.JSONDecodeError) as exc:
            raise LLMError(f"Embedding request failed: {exc}") from exc


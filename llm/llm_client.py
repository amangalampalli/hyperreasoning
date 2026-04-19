"""Minimal local LLM client for llama.cpp / llama-server HTTP endpoints."""

from __future__ import annotations

from dataclasses import dataclass
import os
import re
import time
from typing import Any, Callable

import httpx
import orjson


class LocalLLMError(RuntimeError):
    """Raised when the local LLM server cannot satisfy a request."""


@dataclass(slots=True)
class LocalLLMClient:
    """Small client for local completion/chat-compatible endpoints."""

    base_url: str = os.environ.get("LOCAL_LLM_BASE_URL", os.environ.get("LLM_BASE_URL", "http://localhost:8080"))
    mode: str = os.environ.get("LOCAL_LLM_MODE", os.environ.get("LLM_API_MODE", "auto"))
    model: str = os.environ.get("LOCAL_LLM_MODEL", os.environ.get("LLM_MODEL", "local-model"))
    reasoning_effort: str = os.environ.get("LOCAL_LLM_REASONING_EFFORT", "low")
    timeout: float = float(os.environ.get("LOCAL_LLM_TIMEOUT", "45"))
    max_retries: int = int(os.environ.get("LOCAL_LLM_MAX_RETRIES", "2"))
    retry_delay: float = float(os.environ.get("LOCAL_LLM_RETRY_DELAY", "0.75"))
    last_mode: str | None = None
    last_response_text: str | None = None
    total_prompt_tokens: int = 0
    total_completion_tokens: int = 0
    total_tokens: int = 0
    total_requests: int = 0
    progress_callback: Callable[[dict[str, Any]], None] | None = None

    def complete(
        self,
        prompt: str,
        temperature: float = 0.2,
        max_tokens: int | None = 1200,
        *,
        mode_override: str | None = None,
        extra_payload: dict[str, Any] | None = None,
        request_label: str | None = None,
    ) -> str:
        """Submit a prompt and return raw text."""

        errors_seen: list[str] = []
        modes = [mode_override] if mode_override is not None else self._resolve_modes()
        for mode in modes:
            for attempt in range(1, self.max_retries + 2):
                started_at = time.perf_counter()
                self._emit_progress(
                    {
                        "event": "llm_request_started",
                        "request_label": request_label,
                        "mode": mode,
                        "attempt": attempt,
                    }
                )
                try:
                    if mode == "chat":
                        text, response_payload = self._chat_complete(
                            prompt,
                            temperature=temperature,
                            max_tokens=max_tokens,
                            extra_payload=extra_payload,
                        )
                    else:
                        text, response_payload = self._raw_complete(
                            prompt,
                            temperature=temperature,
                            max_tokens=max_tokens,
                            extra_payload=extra_payload,
                        )
                    self.last_mode = mode
                    self.last_response_text = text
                    usage = _extract_token_usage(response_payload)
                    if usage["total_tokens"] == 0:
                        usage = {
                            "prompt_tokens": estimate_token_count(prompt),
                            "completion_tokens": estimate_token_count(text),
                            "total_tokens": estimate_token_count(prompt) + estimate_token_count(text),
                        }
                    self.total_prompt_tokens += usage["prompt_tokens"]
                    self.total_completion_tokens += usage["completion_tokens"]
                    self.total_tokens += usage["total_tokens"]
                    self.total_requests += 1
                    self._emit_progress(
                        {
                            "event": "llm_request_completed",
                            "request_label": request_label,
                            "mode": mode,
                            "attempt": attempt,
                            "elapsed_s": time.perf_counter() - started_at,
                            **usage,
                        }
                    )
                    return text
                except LocalLLMError as exc:
                    errors_seen.append(f"{mode} attempt {attempt}: {exc}")
                    self._emit_progress(
                        {
                            "event": "llm_request_failed",
                            "request_label": request_label,
                            "mode": mode,
                            "attempt": attempt,
                            "elapsed_s": time.perf_counter() - started_at,
                            "error": str(exc),
                        }
                    )
                    if attempt <= self.max_retries:
                        time.sleep(self.retry_delay)
        raise LocalLLMError("Local LLM request failed after retries. " + " | ".join(errors_seen[-4:]))

    def _emit_progress(self, event: dict[str, Any]) -> None:
        if self.progress_callback is not None:
            self.progress_callback(event)

    def _resolve_modes(self) -> list[str]:
        normalized = self.mode.strip().lower()
        if normalized == "chat":
            return ["chat"]
        if normalized == "completion":
            return ["completion"]
        return ["chat", "completion"]

    def _chat_complete(
        self,
        prompt: str,
        *,
        temperature: float,
        max_tokens: int | None,
        extra_payload: dict[str, Any] | None = None,
    ) -> tuple[str, dict[str, Any]]:
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature,
            "reasoning_effort": self.reasoning_effort,
            "cache_prompt": False,
        }
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens
        if extra_payload:
            payload.update(extra_payload)
        response = self._post_json("/v1/chat/completions", payload)
        choices = response.get("choices")
        if isinstance(choices, list) and choices:
            first = choices[0]
            if isinstance(first, dict):
                message = first.get("message", {})
                if isinstance(message, dict) and isinstance(message.get("content"), str):
                    return message["content"], response
        raise LocalLLMError(f"Unexpected chat response shape: {response!r}")

    def _raw_complete(
        self,
        prompt: str,
        *,
        temperature: float,
        max_tokens: int | None,
        extra_payload: dict[str, Any] | None = None,
    ) -> tuple[str, dict[str, Any]]:
        payload = {
            "prompt": prompt,
            "temperature": temperature,
            "stream": False,
            "reasoning_effort": self.reasoning_effort,
            "cache_prompt": False,
        }
        if max_tokens is not None:
            payload["n_predict"] = max_tokens
        if extra_payload:
            payload.update(extra_payload)
        response = self._post_json("/completion", payload)
        for key in ("content", "text"):
            value = response.get(key)
            if isinstance(value, str):
                return value, response
        choices = response.get("choices")
        if isinstance(choices, list) and choices:
            first = choices[0]
            if isinstance(first, dict):
                for key in ("text", "content"):
                    value = first.get(key)
                    if isinstance(value, str):
                        return value, response
        raise LocalLLMError(f"Unexpected completion response shape: {response!r}")

    def _post_json(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        url = self.base_url.rstrip("/") + path
        try:
            with httpx.Client(timeout=self.timeout) as client:
                response = client.post(
                    url,
                    content=orjson.dumps(payload),
                    headers={"Content-Type": "application/json"},
                )
                response.raise_for_status()
        except httpx.HTTPError as exc:
            raise LocalLLMError(f"Request failed for {url}: {exc}") from exc

        try:
            parsed = orjson.loads(response.content)
        except orjson.JSONDecodeError as exc:
            raise LocalLLMError(f"Server returned non-JSON response from {url}: {response.text[:400]!r}") from exc
        if not isinstance(parsed, dict):
            raise LocalLLMError(f"Expected JSON object from {url}, got {type(parsed)!r}")
        return parsed


def _extract_token_usage(response: dict[str, Any]) -> dict[str, int]:
    usage = response.get("usage")
    if isinstance(usage, dict):
        prompt_tokens = int(usage.get("prompt_tokens", 0) or 0)
        completion_tokens = int(usage.get("completion_tokens", 0) or 0)
        total_tokens = int(usage.get("total_tokens", prompt_tokens + completion_tokens) or 0)
        return {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
        }

    prompt_tokens = int(
        response.get("tokens_evaluated")
        or response.get("prompt_n")
        or response.get("num_prompt_tokens")
        or 0
    )
    completion_tokens = int(
        response.get("tokens_predicted")
        or response.get("completion_n")
        or response.get("num_completion_tokens")
        or 0
    )
    if prompt_tokens == 0 and completion_tokens == 0:
        return {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        }
    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": prompt_tokens + completion_tokens,
    }


def estimate_token_count(text: str) -> int:
    if not text:
        return 0
    pieces = re.findall(r"\w+|[^\w\s]", text, flags=re.UNICODE)
    return len(pieces)

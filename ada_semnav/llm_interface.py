"""Unified LLM interface with budget accounting."""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from typing import Any

from ada_semnav.llm_client import default_model_for_provider, infer_provider, resolve_client_kwargs


def _count_tokens_fallback(text: str) -> int:
    return max(1, len(text) // 4)


def _count_tokens(text: str, model: str) -> int:
    try:
        import tiktoken  # type: ignore
    except Exception:  # noqa: BLE001
        return _count_tokens_fallback(text)
    try:
        enc = tiktoken.encoding_for_model(model)
    except Exception:  # noqa: BLE001
        enc = tiktoken.get_encoding("cl100k_base")
    return len(enc.encode(text))


@dataclass
class LLMCallRecord:
    purpose: str
    backend: str
    model: str
    success: bool
    latency_ms: float
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    error: str
    request_preview: str
    response_preview: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class LLMInterface:
    """Single point for slow-process LLM calls and budget tracking."""

    def __init__(
        self,
        *,
        backend: str = "heuristic",
        model_name: str = "",
        call_budget: int | None = None,
        token_budget: int | None = None,
        latency_budget_ms: float | None = None,
        request_timeout_sec: float | None = None,
        api_mode: str = "auto",
    ) -> None:
        provider = infer_provider()
        self.backend = backend
        self.provider = provider
        self.model_name = model_name or default_model_for_provider(provider if provider != "none" else "openai")
        self.call_budget = int(call_budget) if call_budget is not None and call_budget >= 0 else None
        self.token_budget = int(token_budget) if token_budget is not None and token_budget >= 0 else None
        self.latency_budget_ms = float(latency_budget_ms) if latency_budget_ms is not None and latency_budget_ms >= 0 else None
        self.request_timeout_sec = (
            float(request_timeout_sec) if request_timeout_sec is not None and request_timeout_sec > 0 else None
        )
        self.api_mode = api_mode if api_mode in {"auto", "responses", "chat"} else "auto"
        self.records: list[LLMCallRecord] = []
        self._client: Any | None = None
        self._responses_supported: bool | None = None

    @property
    def call_count(self) -> int:
        return len(self.records)

    @property
    def used_tokens(self) -> int:
        return sum(r.total_tokens for r in self.records)

    @property
    def used_latency_ms(self) -> float:
        return sum(r.latency_ms for r in self.records)

    def can_call(self) -> bool:
        if self.backend != "openai":
            return False
        if self.call_budget is not None and self.call_count >= self.call_budget:
            return False
        if self.token_budget is not None and self.used_tokens >= self.token_budget:
            return False
        if self.latency_budget_ms is not None and self.used_latency_ms >= self.latency_budget_ms:
            return False
        return True

    @staticmethod
    def _preview(text: str, n: int = 160) -> str:
        t = " ".join(text.strip().split())
        if len(t) <= n:
            return t
        return t[: n - 3] + "..."

    @staticmethod
    def _parse_json_text(text: str) -> dict[str, Any] | None:
        text = (text or "").strip()
        if not text:
            return None
        try:
            payload = json.loads(text)
            return payload if isinstance(payload, dict) else None
        except Exception:  # noqa: BLE001
            pass
        l = text.find("{")
        r = text.rfind("}")
        if l >= 0 and r > l:
            try:
                payload = json.loads(text[l : r + 1])
                return payload if isinstance(payload, dict) else None
            except Exception:  # noqa: BLE001
                return None
        return None

    def _record(
        self,
        *,
        purpose: str,
        success: bool,
        latency_ms: float,
        prompt: str,
        response_text: str,
        error: str = "",
    ) -> LLMCallRecord:
        prompt_tokens = _count_tokens(prompt, self.model_name)
        completion_tokens = _count_tokens(response_text, self.model_name) if response_text else 0
        record = LLMCallRecord(
            purpose=purpose,
            backend=self.backend,
            model=self.model_name,
            success=success,
            latency_ms=float(latency_ms),
            prompt_tokens=int(prompt_tokens),
            completion_tokens=int(completion_tokens),
            total_tokens=int(prompt_tokens + completion_tokens),
            error=error,
            request_preview=self._preview(prompt),
            response_preview=self._preview(response_text),
        )
        self.records.append(record)
        return record

    def call_json(
        self,
        *,
        purpose: str,
        prompt: str,
        max_tokens: int = 256,
        temperature: float = 0.0,
    ) -> tuple[dict[str, Any] | None, LLMCallRecord]:
        if not self.can_call():
            record = self._record(
                purpose=purpose,
                success=False,
                latency_ms=0.0,
                prompt=prompt,
                response_text="",
                error="llm_disabled_or_budget_exceeded",
            )
            return None, record

        started = time.perf_counter()
        response_text = ""
        error = ""
        payload: dict[str, Any] | None = None
        mode = self.api_mode
        try_responses = mode in {"auto", "responses"}
        if mode == "auto" and self._responses_supported is False:
            try_responses = False
        try:
            client = self._get_client()
            if try_responses:
                try:
                    resp = client.responses.create(
                        model=self.model_name,
                        input=[{"role": "user", "content": prompt}],
                        text={"format": {"type": "json_object"}},
                        max_output_tokens=max_tokens,
                    )
                    response_text = str(getattr(resp, "output_text", "") or "")
                    payload = self._parse_json_text(response_text)
                    if payload is not None:
                        self._responses_supported = True
                    elif mode == "auto" and not response_text.strip():
                        self._responses_supported = False
                except Exception:  # noqa: BLE001
                    payload = None
                    if mode == "auto":
                        self._responses_supported = False

            if payload is None and mode != "responses":
                chat = client.chat.completions.create(
                    model=self.model_name,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
                if getattr(chat, "choices", None):
                    response_text = (chat.choices[0].message.content or "").strip()
                payload = self._parse_json_text(response_text)
            if payload is None:
                error = "invalid_json_response"
        except Exception as exc:  # noqa: BLE001
            msg = str(exc)
            error = "openai_package_missing" if "No module named" in msg and "openai" in msg else msg
            payload = None
        elapsed_ms = (time.perf_counter() - started) * 1000.0

        record = self._record(
            purpose=purpose,
            success=payload is not None,
            latency_ms=elapsed_ms,
            prompt=prompt,
            response_text=response_text,
            error=error,
        )
        return payload, record

    def _get_client(self) -> Any:
        if self._client is not None:
            return self._client
        from openai import OpenAI  # type: ignore

        kwargs = dict(resolve_client_kwargs(provider="auto"))
        if self.request_timeout_sec is not None:
            kwargs["timeout"] = self.request_timeout_sec
        self._client = OpenAI(**kwargs)
        return self._client

    def summary(self) -> dict[str, Any]:
        success = sum(1 for r in self.records if r.success)
        return {
            "backend": self.backend,
            "model": self.model_name,
            "calls": self.call_count,
            "successful_calls": success,
            "failed_calls": self.call_count - success,
            "total_tokens": self.used_tokens,
            "total_latency_ms": self.used_latency_ms,
            "call_budget": self.call_budget,
            "token_budget": self.token_budget,
            "latency_budget_ms": self.latency_budget_ms,
            "records": [r.to_dict() for r in self.records],
        }

"""Unified budget ledger for slow-process accounting."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class BudgetCaps:
    """Hard caps shared by compared methods within one protocol."""

    call_budget: int | None = None
    latency_budget_ms: float | None = None
    memory_token_budget: int | None = None
    context_token_budget: int | None = None


class BudgetManager:
    """Single source of truth for per-episode and run-level budget usage."""

    def __init__(self, caps: BudgetCaps) -> None:
        self.caps = BudgetCaps(
            call_budget=self._normalize_int_cap(caps.call_budget),
            latency_budget_ms=self._normalize_float_cap(caps.latency_budget_ms),
            memory_token_budget=self._normalize_int_cap(caps.memory_token_budget),
            context_token_budget=self._normalize_int_cap(caps.context_token_budget),
        )
        self.total_slow_calls = 0
        self.total_tokens = 0.0
        self.total_latency_ms = 0.0
        self.max_memory_tokens_total = 0.0
        self._episode_idx = -1
        self.reset_episode(episode_idx=0)

    @staticmethod
    def _normalize_int_cap(value: int | None) -> int | None:
        if value is None:
            return None
        v = int(value)
        return v if v >= 0 else None

    @staticmethod
    def _normalize_float_cap(value: float | None) -> float | None:
        if value is None:
            return None
        v = float(value)
        return v if v >= 0 else None

    def reset_episode(self, episode_idx: int) -> None:
        self._episode_idx = int(episode_idx)
        self.slow_calls_episode = 0
        self.tokens_episode = 0.0
        self.latency_episode_ms = 0.0
        self.tokens_this_call = 0.0
        self.latency_this_call_ms = 0.0
        self.memory_tokens_last = 0.0
        self.max_memory_tokens_episode = 0.0
        self.last_skip_reason = ""
        self._refresh_hard_cap_flag()

    def can_start_slow_call(self) -> tuple[bool, str]:
        if self.caps.call_budget is not None and self.slow_calls_episode >= self.caps.call_budget:
            self.last_skip_reason = "call_budget"
            self._refresh_hard_cap_flag()
            return False, self.last_skip_reason
        if self.caps.latency_budget_ms is not None and self.latency_episode_ms >= self.caps.latency_budget_ms:
            self.last_skip_reason = "latency_budget"
            self._refresh_hard_cap_flag()
            return False, self.last_skip_reason
        self.last_skip_reason = ""
        self._refresh_hard_cap_flag()
        return True, ""

    def reserve_slow_call(self) -> None:
        self.slow_calls_episode += 1
        self.total_slow_calls += 1
        self.tokens_this_call = 0.0
        self.latency_this_call_ms = 0.0
        self.last_skip_reason = ""
        self._refresh_hard_cap_flag()

    def finalize_slow_call(self, *, tokens_this_call: float, latency_this_call_ms: float) -> None:
        tokens = max(0.0, float(tokens_this_call))
        latency = max(0.0, float(latency_this_call_ms))
        self.tokens_this_call = tokens
        self.latency_this_call_ms = latency
        self.tokens_episode += tokens
        self.total_tokens += tokens
        self.latency_episode_ms += latency
        self.total_latency_ms += latency
        self._refresh_hard_cap_flag()

    def note_memory_tokens(self, token_estimate: float) -> None:
        token_value = max(0.0, float(token_estimate))
        self.memory_tokens_last = token_value
        self.max_memory_tokens_episode = max(self.max_memory_tokens_episode, token_value)
        self.max_memory_tokens_total = max(self.max_memory_tokens_total, token_value)
        self._refresh_hard_cap_flag()

    def remaining_budget_ratio(self) -> float:
        ratios: list[float] = []
        if self.caps.call_budget is not None and self.caps.call_budget > 0:
            remain = max(0.0, float(self.caps.call_budget - self.slow_calls_episode))
            ratios.append(remain / float(self.caps.call_budget))
        if self.caps.latency_budget_ms is not None and self.caps.latency_budget_ms > 0:
            remain = max(0.0, float(self.caps.latency_budget_ms - self.latency_episode_ms))
            ratios.append(remain / float(self.caps.latency_budget_ms))
        if self.caps.memory_token_budget is not None and self.caps.memory_token_budget > 0:
            remain = max(0.0, float(self.caps.memory_token_budget - self.memory_tokens_last))
            ratios.append(remain / float(self.caps.memory_token_budget))
        if not ratios:
            return 1.0
        return float(min(ratios))

    def hard_cap_reached(self) -> bool:
        return bool(self._hard_cap_reached)

    def _refresh_hard_cap_flag(self) -> None:
        call_hit = self.caps.call_budget is not None and self.slow_calls_episode >= self.caps.call_budget
        latency_hit = (
            self.caps.latency_budget_ms is not None and self.latency_episode_ms >= self.caps.latency_budget_ms
        )
        memory_hit = (
            self.caps.memory_token_budget is not None and self.memory_tokens_last >= self.caps.memory_token_budget
        )
        context_hit = (
            self.caps.context_token_budget is not None and self.tokens_this_call >= self.caps.context_token_budget
        )
        self._hard_cap_reached = bool(call_hit or latency_hit or memory_hit or context_hit)

    def snapshot(self) -> dict[str, Any]:
        return {
            "episode": int(self._episode_idx),
            "slow_calls_episode": int(self.slow_calls_episode),
            "tokens_episode": float(self.tokens_episode),
            "tokens_this_call": float(self.tokens_this_call),
            "latency_episode": float(self.latency_episode_ms),
            "latency_this_call": float(self.latency_this_call_ms),
            "remaining_budget_ratio": float(self.remaining_budget_ratio()),
            "hard_cap_reached": int(self.hard_cap_reached()),
            "last_skip_reason": str(self.last_skip_reason),
            "memory_tokens_last": float(self.memory_tokens_last),
            "max_memory_tokens_episode": float(self.max_memory_tokens_episode),
            "caps_call_budget": -1 if self.caps.call_budget is None else int(self.caps.call_budget),
            "caps_latency_budget_ms": (
                -1.0 if self.caps.latency_budget_ms is None else float(self.caps.latency_budget_ms)
            ),
            "caps_memory_token_budget": (
                -1 if self.caps.memory_token_budget is None else int(self.caps.memory_token_budget)
            ),
            "caps_context_token_budget": (
                -1 if self.caps.context_token_budget is None else int(self.caps.context_token_budget)
            ),
        }

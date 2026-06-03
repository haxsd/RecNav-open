"""Unified step/episode/event logging."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


class MetricsLogger:
    """Writes canonical experiment logs used by staged paper evaluation."""

    def __init__(self, out_dir: str | Path, append: bool = False) -> None:
        self.out_dir = Path(out_dir).resolve()
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.step_trace_path = self.out_dir / "step_trace.jsonl"
        self.episode_summary_path = self.out_dir / "episode_summary.jsonl"
        self.events_path = self.out_dir / "events.jsonl"
        self.memory_snapshot_path = self.out_dir / "memory_snapshot.jsonl"
        mode = "a" if append else "w"
        self._step_fp = self.step_trace_path.open(mode, encoding="utf-8")
        self._ep_fp = self.episode_summary_path.open(mode, encoding="utf-8")
        self._event_fp = self.events_path.open(mode, encoding="utf-8")
        self._memory_fp = self.memory_snapshot_path.open(mode, encoding="utf-8")

    def close(self) -> None:
        self._step_fp.close()
        self._ep_fp.close()
        self._event_fp.close()
        self._memory_fp.close()

    def __enter__(self) -> "MetricsLogger":
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self.close()

    @staticmethod
    def _write(fp: Any, payload: dict[str, Any]) -> None:
        fp.write(json.dumps(payload, ensure_ascii=False) + "\n")

    def log_step(self, payload: dict[str, Any]) -> None:
        self._write(self._step_fp, payload)

    def log_episode(self, payload: dict[str, Any]) -> None:
        self._write(self._ep_fp, payload)

    def log_event(self, event_type: str, payload: dict[str, Any]) -> None:
        row = {"event_type": event_type, **payload}
        self._write(self._event_fp, row)

    def log_memory(self, payload: dict[str, Any]) -> None:
        self._write(self._memory_fp, payload)

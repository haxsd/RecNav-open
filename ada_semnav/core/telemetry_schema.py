"""Canonical telemetry event helpers."""

from __future__ import annotations

from typing import Any

from ada_semnav.metrics_logger import MetricsLogger

SCHEMA_VERSION = "v1"
EVENT_SCHEMA = "recoverability_telemetry_v1"

CANONICAL_EVENT_TYPES = {
    "gate_evaluated",
    "gate_triggered",
    "budget_snapshot",
    "memory_updated",
    "memory_compressed",
    "recovery_started",
    "backtrack_finished",
    "candidate_edges_built",
    "edge_selected",
    "recovery_succeeded",
    "recovery_failed",
}


def _base_payload(
    *,
    episode: int,
    step: int,
    episode_id: str,
    scene_id: str,
    case_id: str = "",
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "event_schema": EVENT_SCHEMA,
        "episode": int(episode),
        "step": int(step),
        "episode_id": str(episode_id),
        "scene_id": str(scene_id),
        "case_id": str(case_id),
    }


def log_telemetry_event(
    logger: MetricsLogger,
    *,
    event_type: str,
    episode: int,
    step: int,
    episode_id: str,
    scene_id: str,
    case_id: str = "",
    payload: dict[str, Any] | None = None,
) -> None:
    if event_type not in CANONICAL_EVENT_TYPES:
        raise ValueError(f"Unsupported canonical telemetry event_type: {event_type}")
    row = _base_payload(
        episode=episode,
        step=step,
        episode_id=episode_id,
        scene_id=scene_id,
        case_id=case_id,
    )
    if payload:
        row.update(payload)
    logger.log_event(event_type, row)


def log_budget_snapshot_event(
    logger: MetricsLogger,
    *,
    budget_snapshot: dict[str, Any],
    episode: int,
    step: int,
    episode_id: str,
    scene_id: str,
    case_id: str = "",
    reason: str = "",
) -> None:
    payload = dict(budget_snapshot)
    if reason:
        payload["budget_reason"] = str(reason)
    log_telemetry_event(
        logger,
        event_type="budget_snapshot",
        episode=episode,
        step=step,
        episode_id=episode_id,
        scene_id=scene_id,
        case_id=case_id,
        payload=payload,
    )

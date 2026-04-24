"""Offline analyzer for RecNav staged logs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except Exception:  # noqa: BLE001
                continue
            if isinstance(payload, dict):
                rows.append(payload)
    return rows


def mean(values: list[float]) -> float:
    if not values:
        return 0.0
    return sum(values) / float(len(values))


def analyze(
    *,
    step_trace_path: Path,
    episode_summary_path: Path,
    events_path: Path | None = None,
) -> dict[str, Any]:
    step_rows = load_jsonl(step_trace_path)
    ep_rows = load_jsonl(episode_summary_path)
    event_rows = load_jsonl(events_path) if events_path is not None else []

    sr = mean([float(r.get("success", 0.0)) for r in ep_rows])
    spl = mean([float(r.get("spl", 0.0)) for r in ep_rows])

    llm_calls = 0
    llm_tokens = 0.0
    llm_latency_ms = 0.0
    for r in step_rows:
        llm_calls += int(r.get("llm_calls", 0))
        llm_tokens += float(r.get("llm_total_tokens", 0.0))
        llm_latency_ms += float(r.get("llm_latency_ms", 0.0))

    event_llm_calls = 0
    event_llm_tokens = 0.0
    event_llm_latency_ms = 0.0
    for ev in event_rows:
        if str(ev.get("event_type", "")) != "recovery_result":
            continue
        event_llm_calls += int(ev.get("llm_calls", 0))
        event_llm_tokens += float(ev.get("llm_total_tokens", 0.0))
        event_llm_latency_ms += float(ev.get("llm_latency_ms", 0.0))
    llm_calls += event_llm_calls
    llm_tokens += event_llm_tokens
    llm_latency_ms += event_llm_latency_ms

    recovery_triggers = 0
    recovery_success = 0
    recovery_re_fail = 0
    recovery_overheads: list[float] = []
    for ev in event_rows:
        et = str(ev.get("event_type", ""))
        if et == "recovery_trigger":
            recovery_triggers += 1
        elif et == "recovery_result":
            if bool(ev.get("success", False)):
                recovery_success += 1
            if bool(ev.get("failed_again_within_15", False)):
                recovery_re_fail += 1
            if "extra_steps" in ev:
                try:
                    recovery_overheads.append(float(ev["extra_steps"]))
                except Exception:  # noqa: BLE001
                    pass

    if recovery_triggers == 0:
        # Fallback from step trace when event logs are absent.
        recovery_triggers = sum(int(r.get("recovery_trigger", 0)) for r in step_rows)
    if recovery_success == 0:
        recovery_success = sum(int(r.get("recovery_success", 0)) for r in step_rows)

    total_steps = max(1, len(step_rows))
    rsr = float(recovery_success) / float(recovery_triggers) if recovery_triggers > 0 else 0.0
    rfr = float(recovery_re_fail) / float(recovery_success) if recovery_success > 0 else 0.0
    rpo = mean(recovery_overheads)
    rtr = float(recovery_triggers) / float(total_steps)

    return {
        "episodes": len(ep_rows),
        "steps": len(step_rows),
        "SR": sr,
        "SPL": spl,
        "Calls": llm_calls,
        "Tokens": llm_tokens,
        "Latency_ms": llm_latency_ms,
        "RSR": rsr,
        "RFR": rfr,
        "RPO": rpo,
        "RTR": rtr,
        "recovery_trigger_count": recovery_triggers,
        "recovery_success_count": recovery_success,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Analyze RecNav step/episode logs.")
    parser.add_argument("--log-dir", default="artifacts", help="Directory containing jsonl logs.")
    parser.add_argument("--step-trace", default="step_trace.jsonl", help="Step trace filename.")
    parser.add_argument("--episode-summary", default="episode_summary.jsonl", help="Episode summary filename.")
    parser.add_argument("--events", default="events.jsonl", help="Event filename.")
    parser.add_argument("--out-json", default="", help="Optional output summary json path.")
    args = parser.parse_args()

    log_dir = Path(args.log_dir).resolve()
    summary = analyze(
        step_trace_path=log_dir / args.step_trace,
        episode_summary_path=log_dir / args.episode_summary,
        events_path=log_dir / args.events,
    )
    if args.out_json:
        out = Path(args.out_json).resolve()
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"summary_json: {out}")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

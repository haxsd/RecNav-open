# Planner Trace Format (`planner_trace.jsonl`)

## Overview

The planner trace file records every VLM planning decision made during an episode.
It is the foundation of the **locked-trace causal protocol** used in all RecNav experiments.

## File Format

Each line is a JSON object representing one VLM planning call:

```jsonl
{"episode_id": "hm3d_val_00001", "step": 12, "direction": "turn_left", "goal_flag": false, "timestamp": 1714000000.0}
{"episode_id": "hm3d_val_00001", "step": 24, "direction": "move_forward", "goal_flag": true, "timestamp": 1714000001.2}
```

## Fields

| Field | Type | Description |
|-------|------|-------------|
| `episode_id` | string | HM3D episode identifier |
| `step` | int | Environment step number when this planning call was made |
| `direction` | string | VLM-selected action: `move_forward`, `turn_left`, `turn_right`, or `stop` |
| `goal_flag` | bool | Whether the VLM declared the goal as seen/reached |
| `timestamp` | float | Unix timestamp of the call (optional, for debugging) |

## Locked-Trace Replay Protocol

### Recording (F0 baseline run)

During a baseline run (`f0` variant), the runner writes one line per VLM call:
- `temperature=0.0` ensures deterministic VLM outputs
- `random_fallback=0` disables random action fallback on VLM errors

### Replaying (RecNav variant runs)

During a RecNav variant run (e.g., `adarec_locked`):

1. The trace file is loaded at the start of evaluation.
2. At each planning step, the runner reads the next line from the trace instead of calling the VLM.
3. A `_replay_active` flag starts as `True`.
4. When a recovery maneuver triggers, the flag is set to `False`.
5. After recovery completes, subsequent planning calls go to the **live VLM** (no longer replayed).

This guarantees:
- **Before recovery:** trajectories are identical between F0 and RecNav.
- **After recovery:** any divergence is caused solely by the recovery action.
- **Wins/losses** can be directly attributed to the recovery mechanism.

### Environment Variables

| Variable | Description |
|----------|-------------|
| `PLANNER_TRACE_SOURCE` | Path to the F0 trace file to replay |
| `EPISODE_SEED` | Must match between F0 and RecNav runs |
| `PROXIMITY_STOP` | Stopping distance threshold (consistent across runs) |

## Generating a Trace

```bash
# Step 1: Run F0 to record the trace
bash experiments/run_experiment.sh f0 42 100 200
# Trace saved to: artifacts/runs/experiments/f0_s200/seed_42/planner_trace.jsonl

# Step 2: Run RecNav replaying the trace
export PLANNER_TRACE_SOURCE=artifacts/runs/experiments/f0_s200/seed_42/planner_trace.jsonl
bash experiments/run_experiment.sh adarec_locked 42 100 200
```

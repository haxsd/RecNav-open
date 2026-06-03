# Planner Trace Format

`planner_trace.jsonl` stores one JSON object per planner call. The exact set of
fields may grow, but locked-trace replay expects at least:

```json
{
  "episode_idx": 0,
  "call_idx": 0,
  "action": 1,
  "goal_flag": false
}
```

Common fields:

- `episode_idx`: integer episode index inside the evaluated split/sample.
- `call_idx`: planner-call index within that episode.
- `action`: low-level action id selected by the planner.
- `goal_flag`: whether the planner believed the target was reached.
- `raw_response`: optional raw VLM response for audit/debugging.

For locked-trace experiments, all recovery variants should use the same
baseline `planner_trace.jsonl` and set `--replay_planner_trace`.

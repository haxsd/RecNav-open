# Reproducing RecNav Experiments

This guide describes the release workflow for the main RecNav experiments.

## Required Local Assets

The repository does not ship datasets or checkpoints. Before running evaluation,
prepare:

- HM3D scene assets under `data/scene_datasets/hm3d/`
- ObjectNav HM3D v2 episodes under `data/datasets/objectnav/hm3d/v2/`
- PixNav / GroundingDINO / SAM checkpoints under `pixnav/checkpoints/`
- an OpenAI-compatible API key if running VLM planning

Run:

```bash
./run.sh check-dataset
./run.sh env-check
```

## Main Runner

The low-level runner is:

```bash
python scripts/pixnav/run_pixnav_host_eval.py
```

The release experiments call it through:

```bash
bash experiments/run_experiment.sh <variant> <seed> [episodes] [max_steps]
```

Important output paths:

```text
artifacts/runs/experiments/<variant>_s<budget>/seed_<seed>/results.csv
artifacts/runs/experiments/<variant>_s<budget>/seed_<seed>/planner_trace.jsonl
artifacts/runs/experiments/<variant>_s<budget>/seed_<seed>/telemetry/
```

## Variants

```text
f0                 base PixNav host, no recovery
adarec_locked      full RecNav recovery using a baseline planner trace
random_walk_locked random macro-action recovery
frontier_locked    frontier-style recovery
llm_replan_locked  LLM replanning recovery
adarec_minus_i2    gate ablation
adarec_minus_i3    memory ablation
adarec_minus_i4    recovery ablation
```

## Locked-Trace Order

Run the baseline first to create the trace:

```bash
bash experiments/run_experiment.sh f0 42 100 200
```

Then replay that trace for RecNav:

```bash
export PLANNER_TRACE_SOURCE="$PWD/artifacts/runs/experiments/f0_s200/seed_42/planner_trace.jsonl"
bash experiments/run_experiment.sh adarec_locked 42 100 200
```

The baseline and recovery variant share the same planner trace until recovery
intervenes. This is the causal control used by the paper experiments.

## Batch Scripts

```bash
bash experiments/launch_e2_budget_curve.sh
bash experiments/launch_e1_table1.sh
bash experiments/launch_e3_ablation.sh
```

`launch_e2_budget_curve.sh` should be run first because it creates the baseline
traces used by E1 and E3.

## Analysis

```bash
./run.sh analyze-results --base artifacts/runs/experiments
./run.sh plot-budget --base artifacts/runs/experiments --out figures/budget_curve.pdf
./run.sh mcnemar \
  --baseline artifacts/runs/experiments/f0_s200/seed_42/results.csv \
  --treatment artifacts/runs/experiments/adarec_locked_s200/seed_42/results.csv
```

Do not compare methods that use different dataset splits, episode seeds,
checkpoints, stop thresholds, or planner traces.

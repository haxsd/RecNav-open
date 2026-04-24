# Reproducing Paper Results

This document describes how to reproduce the three key experiments in the paper.

## Prerequisites

1. Complete all installation steps in `README.md` (environment, checkpoints, data).
2. Copy `.env.example` to `.env` and fill in your LLM API key.
3. Activate the conda environment: `conda activate recnav`.

## Experiment Overview

| Experiment | Paper Element | Script | Description |
|------------|--------------|--------|-------------|
| E1 | Table 1 | `launch_e1_table1.sh` | Recovery strategy comparison |
| E2 | Figure 1 | `launch_e2_budget_curve.sh` | Budget-performance curve (6 budget levels) |
| E3 | Table 2 | `launch_e3_ablation.sh` | Component ablation (Gate, Memory, Recovery) |

## Step-by-Step

### Step 1: Run E2 first (generates F0 baselines + traces)

E2 runs both the F0 baseline and RecNav across all budget levels, and records
`planner_trace.jsonl` files that E1 and E3 depend on.

```bash
bash experiments/launch_e2_budget_curve.sh
```

**Runtime:** ~6 budget levels × 2 variants × 100 episodes. Expect 12-24 hours on a single GPU.

**Outputs:**
```
artifacts/runs/experiments/f0_s{100,150,200,250,300,500}/seed_42/results.csv
artifacts/runs/experiments/f0_s{100,150,200,250,300,500}/seed_42/planner_trace.jsonl
artifacts/runs/experiments/adarec_locked_s{100,150,200,250,300,500}/seed_42/results.csv
```

### Step 2: Run E1 (Table 1 — Recovery Strategy Comparison)

E1 replays the F0 s200 trace with different recovery strategies.

```bash
bash experiments/launch_e1_table1.sh
```

**Outputs:**
```
artifacts/runs/experiments/{random_walk,frontier,llm_replan}_locked_s200/seed_42/results.csv
```

### Step 3: Run E3 (Table 2 — Component Ablation)

E3 removes one component at a time from the full RecNav system.

```bash
bash experiments/launch_e3_ablation.sh
```

**Outputs:**
```
artifacts/runs/experiments/adarec_minus_{i2,i3,i4}_s200/seed_42/results.csv
```

### Step 4: Generate tables and figures

```bash
# Generate experiment tables (prints to stdout)
python scripts/pixnav/analyze_results.py --base artifacts/runs/experiments

# Generate budget-performance curve
mkdir -p figures
python scripts/pixnav/plot_budget_curve.py --out figures/budget_curve.pdf

# Statistical significance (McNemar test)
python scripts/tools/mcnemar_test.py \
  --baseline artifacts/runs/experiments/f0_s200/seed_42/results.csv \
  --treatment artifacts/runs/experiments/adarec_locked_s200/seed_42/results.csv
```

## Locked-Trace Protocol

All comparisons use the locked-trace protocol for causal attribution.
See `docs/planner_trace_format.md` for details.

Key settings (enforced by the experiment scripts):
- `EPISODE_SEED=42`
- `PROXIMITY_STOP=0.5`
- `temperature=0.0` (deterministic VLM)
- `random_fallback=0`

## Expected Results

Results should be close to (but not identical to) the paper values due to:
- Different GPU hardware → slightly different floating-point behavior in habitat-sim
- Different VLM API versions → minor variation in planning decisions

The locked-trace protocol eliminates VLM stochasticity within a single run,
but API model updates between our run and yours may cause small differences.

## Results CSV Format

Each `results.csv` has one row per episode:

| Column | Type | Description |
|--------|------|-------------|
| `episode_id` | str | HM3D episode identifier |
| `success` | bool | Whether the agent reached the goal |
| `spl` | float | Success weighted by Path Length |
| `dtg` | float | Final distance-to-goal (meters) |
| `steps` | int | Total steps taken |
| `recovery_count` | int | Number of recovery triggers |
| `recovery_improved` | int | Number of recoveries that improved DTG |

## Troubleshooting

- **`WARN: trace not found`** — Run E2 (F0 baselines) before E1/E3.
- **`No LLM credentials found`** — Set API key in `.env`.
- **`habitat-sim not found`** — Install habitat-sim and habitat-lab per `README.md`.
- **Results differ significantly** — Check VLM model version; use the same model as noted in the paper.

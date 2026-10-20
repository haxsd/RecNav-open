# RecNav: Budget-Aware Recovery Augmentation for Mapless Object Navigation

[中文](README.md) · [English](README-EN.md)

> A plug-in recovery layer for mapless ObjectNav: a pose-free gate, recovery memory and bounded macro-action recovery reduce getting lost and getting stuck.

This repository is the public code release of the paper method; RecNav runs on top of a matched PixNav-style base navigator. The Python package is still named `ada_semnav` for compatibility with the current project code; the method name used in the paper and documentation is RecNav.

## Features

- **Pose-free recovery gate**: decides when recovery starts without relying on ground-truth pose, so it fits mapless settings
- **Compact recovery memory**: stores a recoverable skeleton (nodes / edges) for bounded search during recovery
- **Bounded macro-action recovery**: ranks macro-action candidates and verifies progress in a closed loop, with call budgets and retry caps
- **Budget & telemetry**: records LLM calls, exploration steps and retries as budget events against a fixed schema
- **Locked-trace protocol**: the base navigator records `planner_trace.jsonl` first; recovery variants replay it until recovery is triggered, isolating the effect of recovery from VLM sampling and planner-call differences
- **Reproducible release**: launch scripts for the E1 / E2 / E3 experiments plus result summarization, budget curves and McNemar significance tests

## Quick start

The repository ships code only: no HM3D / ObjectNav data, model checkpoints or paper drafts. Install `habitat-sim` / `habitat-lab` first following the official Habitat instructions.

```bash
conda create -n recnav python=3.10
conda activate recnav
pip install -r requirements.txt
cp .env.example .env          # fill in Habitat paths, checkpoint paths and API keys
```

Install the external vision modules under `pixnav/thirdparty/` (GroundingDINO and Segment Anything, with `pip install -e`), put checkpoints under `pixnav/checkpoints/`, and place HM3D scenes and ObjectNav v2 episodes under `data/`.

Environment checks:

```bash
./run.sh check-dataset   # validate the HM3D / ObjectNav layout
./run.sh env-check       # check dependencies, paths and checkpoints
./run.sh llm-smoke       # test OpenAI-compatible model connectivity
./run.sh smoke-test --steps 10
```

## Reproduction

Run the baseline first to create the planner trace, then run RecNav with the locked trace:

```bash
bash experiments/run_experiment.sh f0 42 100 200
export PLANNER_TRACE_SOURCE="$PWD/artifacts/runs/experiments/f0_s200/seed_42/planner_trace.jsonl"
bash experiments/run_experiment.sh adarec_locked 42 100 200
```

Paper batches (run E2 first, since it creates the baseline traces used by E1 / E3):

```bash
bash experiments/launch_e2_budget_curve.sh
bash experiments/launch_e1_table1.sh
bash experiments/launch_e3_ablation.sh
```

Summarize and test results:

```bash
./run.sh analyze-results --base artifacts/runs/experiments
./run.sh plot-budget --base artifacts/runs/experiments --out figures/budget_curve.pdf
./run.sh mcnemar --baseline <baseline_csv> --treatment <treatment_csv>
```

## Documentation

- [docs/reproduce.md](docs/reproduce.md): full reproduction protocol and experiment variants
- [docs/planner_trace_format.md](docs/planner_trace_format.md): planner trace format
- [docs/artifact_policy.md](docs/artifact_policy.md): artifact and data policy
- [scripts/README.md](scripts/README.md): list of script entry points

## Acknowledgments & license

The base navigation policy builds on PixNav / Pixel Navigator, and open-vocabulary perception uses GroundingDINO and Segment Anything. License: MIT (see [LICENSE](LICENSE)).

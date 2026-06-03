# RecNav: Budget-Aware Recovery Augmentation for Mapless Object Navigation

This repository contains the public code release for RecNav, a recovery
augmentation layer for mapless ObjectNav. RecNav runs on top of a matched
PixNav-style base navigator and adds:

- a pose-free recovery gate,
- compact recovery-oriented memory,
- bounded macro-action recovery with closed-loop progress verification.

The Python package is still named `ada_semnav` for compatibility with the
current project code. The method name used in the paper and documentation is
RecNav.

## What Is Included

```text
ada_semnav/          RecNav gate, memory, recovery, telemetry, and host code
pixnav/              PixNav-side base navigator integration code
scripts/core/        Dataset and Habitat smoke checks
scripts/pixnav/      Main PixNav host evaluation and result utilities
scripts/tools/       Environment, LLM, and significance-test utilities
experiments/         Reproduction launch scripts
docs/                Reproduction and artifact notes
results/             Example CSV schema
run.sh               Small top-level launcher
```

The repository intentionally does not include HM3D/ObjectNav data, model
checkpoints, large runtime outputs, paper drafts, or third-party dependency
source trees.

## Setup

1. Create an environment. Habitat installation is platform-specific, so install
   `habitat-sim` and `habitat-lab` following the official Habitat instructions
   first.

```bash
conda create -n recnav python=3.10
conda activate recnav
pip install -r requirements.txt
```

2. Install external vision modules under `pixnav/thirdparty/`.

```bash
mkdir -p pixnav/thirdparty
git clone https://github.com/IDEA-Research/GroundingDINO.git pixnav/thirdparty/GroundingDINO
git clone https://github.com/facebookresearch/segment-anything.git pixnav/thirdparty/segment-anything
pip install -e pixnav/thirdparty/GroundingDINO
pip install -e pixnav/thirdparty/segment-anything
```

3. Download checkpoints into `pixnav/checkpoints/`.

```text
pixnav/checkpoints/
  GroundingDINO_SwinB_cfg.py
  groundingdino_swinb_cogcoor.pth
  sam_vit_h_4b8939.pth
  navigator.pth
```

4. Prepare Habitat data under `data/`, or set paths in `.env`.

```text
data/
  scene_datasets/hm3d/
  datasets/objectnav/hm3d/v2/
```

5. Configure local paths and API credentials.

```bash
cp .env.example .env
```

## Quick Checks

```bash
./run.sh check-dataset
./run.sh env-check
./run.sh llm-smoke
```

For a small Habitat reset/step check:

```bash
./run.sh smoke-test --steps 10
```

## Reproduction

Run a baseline and a locked-trace RecNav variant:

```bash
bash experiments/run_experiment.sh f0 42 100 200

export PLANNER_TRACE_SOURCE="$PWD/artifacts/runs/experiments/f0_s200/seed_42/planner_trace.jsonl"
bash experiments/run_experiment.sh adarec_locked 42 100 200
```

Run the paper-facing batches:

```bash
bash experiments/launch_e2_budget_curve.sh
bash experiments/launch_e1_table1.sh
bash experiments/launch_e3_ablation.sh
```

Summarize outputs:

```bash
./run.sh analyze-results --base artifacts/runs/experiments
./run.sh plot-budget --base artifacts/runs/experiments --out figures/budget_curve.pdf
./run.sh mcnemar \
  --baseline artifacts/runs/experiments/f0_s200/seed_42/results.csv \
  --treatment artifacts/runs/experiments/adarec_locked_s200/seed_42/results.csv
```

See `docs/reproduce.md` for the full protocol.

## Locked-Trace Protocol

The main comparison uses a locked-trace protocol. First, the base navigator
records `planner_trace.jsonl`. Recovery variants replay the same planner trace
until recovery is triggered. This isolates the effect of recovery from VLM
sampling and planner-call differences.

## Acknowledgments

This code builds on PixNav / Pixel Navigator as the base navigation policy and
uses GroundingDINO and Segment Anything for open-vocabulary perception.

## License

See `LICENSE`.

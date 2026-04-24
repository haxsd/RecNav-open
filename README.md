# RecNav: Budget-Aware Recovery Augmentation for Mapless Object Navigation

Anonymous code release for paper review.

RecNav is a plug-in recovery framework that detects navigation failures in real-time and executes budget-aware recovery maneuvers to escape deadlocks, improving success rate without retraining the base navigation policy.

## Key Components

| Module | Description | Code |
|--------|-------------|------|
| **Budget Manager (I1)** | Tracks step budget and allocates recovery resources | `ada_semnav/core/budget_manager.py` |
| **Recovery Gate (I2)** | Multi-signal failure detector (collision, oscillation, stagnation, etc.) | `ada_semnav/gate.py` |
| **Skeleton Memory (I3)** | Lightweight topological memory for visited-area tracking | `ada_semnav/skeleton_memory.py` |
| **Recovery Engine (I4)** | DTG-guided macro-action fan-out with closed-loop retry | `ada_semnav/recovery.py` |

## Repository Structure

```
RecNav-open/
├── ada_semnav/                 # Core RecNav library
│   ├── gate.py                 # Recovery gate (I2)
│   ├── recovery.py             # Recovery engine (I4)
│   ├── skeleton_memory.py      # Skeleton memory (I3)
│   ├── env_runner.py           # Habitat environment wrapper
│   ├── llm_interface.py        # LLM abstraction layer
│   ├── llm_client.py           # OpenAI-compatible LLM client
│   ├── core/
│   │   └── budget_manager.py   # Budget manager (I1)
│   └── hosts/
│       └── pixnav_host.py      # PixNav host interface
├── pixnav/                     # PixNav base navigator (adapted from PixelNav)
│   ├── gpt4v_planner.py        # VLM-based high-level planner
│   ├── policy_network.py       # PixelNav navigation skill
│   ├── cv_utils/               # GroundingDINO + SAM detection
│   ├── llm_utils/              # GPT request utilities
│   └── data_utils/             # Geometry tools
├── scripts/
│   ├── pixnav/
│   │   ├── run_pixnav_host_eval.py   # Main evaluation runner
│   │   ├── analyze_results.py        # Result table generator
│   │   └── plot_budget_curve.py      # Budget curve figure generator
│   └── tools/
│       ├── stage0_env_check.py       # Environment validation
│       └── mcnemar_test.py           # Statistical significance test
├── experiments/
│   ├── run_experiment.sh              # Unified experiment runner
│   ├── launch_e1_table1.sh            # E1: Recovery strategy comparison
│   ├── launch_e2_budget_curve.sh      # E2: Budget-performance curve
│   └── launch_e3_ablation.sh          # E3: Component ablation
├── docs/
│   ├── reproduce.md                   # Full reproduction guide
│   └── planner_trace_format.md        # Locked-trace file specification
├── results/
│   └── example_results.csv            # Example output format
├── run.sh                             # Top-level launcher
├── .env.example                       # Environment variable template
└── requirements.txt                   # Python dependencies
```

## Installation

### 1. Prerequisites

- Python 3.10+
- CUDA-capable GPU
- [habitat-sim](https://github.com/facebookresearch/habitat-sim) and [habitat-lab](https://github.com/facebookresearch/habitat-lab) (v0.3.x)
- [HM3D ObjectNav dataset](https://github.com/facebookresearch/habitat-lab/blob/main/DATASETS.md)

### 2. Install dependencies

```bash
conda create -n recnav python=3.10
conda activate recnav

# Install habitat-sim and habitat-lab first (follow their official guides)
# Then install remaining dependencies:
pip install -r requirements.txt
```

### 3. Install third-party modules

```bash
cd pixnav/thirdparty/GroundingDINO
pip install -e .
cd ../segment-anything
pip install -e .
```

> **Note:** The `pixnav/thirdparty/` directory uses git submodules. After cloning, run:
> ```bash
> git submodule update --init --recursive
> ```

### 4. Download checkpoints

Place the following checkpoints in `pixnav/checkpoints/`:

| Module | File | Source |
|--------|------|--------|
| GroundingDINO | `groundingdino_swinb_cogcoor.pth` | [Google Drive](https://drive.google.com/file/d/1kSH6AhUBrr-CxMrm4J3A9Pv__3WlCjDH/view) |
| SAM | `sam_vit_h_4b8939.pth` | [Google Drive](https://drive.google.com/file/d/1cc6fk71zAK_8HJQltAKyM65nlcoN1eh1/view) |
| PixelNav Skill | `navigator.pth` | [Checkpoint_A](https://drive.google.com/file/d/14iPb5buFOqEMuc_Luc_ShbVoo8xEIklu/view) |
| GroundingDINO Config | `GroundingDINO_SwinB_cfg.py` | Included in GroundingDINO repo |
| BERT tokenizer | `bert-base-uncased/` | `transformers` auto-downloads, or manually place |

### 5. Configure environment

```bash
cp .env.example .env
# Edit .env and fill in your API keys and paths
```

### 6. Prepare data

Place HM3D scene datasets under `data/`:
```
data/
├── datasets/objectnav/hm3d/v2/val/...
└── scene_datasets/hm3d/...
```

## Usage

### Run a single experiment variant

```bash
# Activate environment
conda activate recnav

# Run baseline (no recovery)
bash experiments/run_experiment.sh f0 42 100 200

# Run RecNav (full system)
bash experiments/run_experiment.sh adarec 42 100 200
```

**Arguments:** `<VARIANT> <SEED> [EPISODES] [MAX_STEPS]`

### Available experiment variants

| Variant | Description |
|---------|-------------|
| `f0` | PixNav baseline (no recovery) |
| `adarec` | RecNav full system |
| `llm_replan` | + LLM Replan recovery |
| `random_escape` | + Random displacement |
| `backtrack` | + Backtrack + constrained edge |
| `adarec_minus_i2` | Ablation: remove Gate |
| `adarec_minus_i3` | Ablation: remove Memory |
| `adarec_minus_i4` | Ablation: remove Recovery intelligence |

See `experiments/run_experiment.sh` for the full list of supported variants (E1-E7).

### Reproduce paper results (three key experiments)

```bash
# E1: Recovery strategy comparison (Table 1)
bash experiments/launch_e1_table1.sh

# E2: Budget-performance curve (Figure 1)
bash experiments/launch_e2_budget_curve.sh

# E3: Component ablation (Table 2)
bash experiments/launch_e3_ablation.sh

# Generate paper tables and figures from results
python scripts/pixnav/analyze_results.py --base artifacts/runs/experiments
python scripts/pixnav/plot_budget_curve.py --out figures/budget_curve.pdf

# Statistical significance test (McNemar)
python scripts/tools/mcnemar_test.py \
  --baseline artifacts/runs/experiments/f0_s200/seed_42/results.csv \
  --treatment artifacts/runs/experiments/adarec_locked_s200/seed_42/results.csv
```

**Expected outputs:**

```
artifacts/runs/experiments/<variant>_s<budget>/seed_<N>/results.csv
figures/budget_curve.pdf
```

### Locked-trace causal protocol

All RecNav experiments use a **locked-trace** protocol to ensure causal attribution:

1. **Record phase:** Run the baseline (`f0`) and record every VLM planning decision into `planner_trace.jsonl`.
2. **Replay phase:** All RecNav variants replay the *exact same* trace. Before a recovery triggers, the trajectory is identical to baseline.
3. **Divergence:** Only after a recovery maneuver does the agent diverge from the recorded trace and switch to live VLM calls.

This design ensures that any difference in success rate (wins/losses) is **100% attributable to the recovery mechanism**, not VLM stochasticity.

**Key controls:**
- `temperature=0.0` for VLM calls
- `random_fallback=0` (no random action on VLM failure)
- Shared `EPISODE_SEED` across all variants
- `PROXIMITY_STOP` threshold consistent across runs
- McNemar's exact test for statistical significance

See `docs/reproduce.md` for the full protocol and `docs/planner_trace_format.md` for the trace file specification.

### Analyze results

```bash
# Generate all experiment tables
python scripts/pixnav/analyze_results.py --base artifacts/runs/experiments

# Generate budget-performance curve
python scripts/pixnav/plot_budget_curve.py --out figures/budget_curve.pdf
```

## Acknowledgments

This project builds upon [PixelNav (ICRA 2024)](https://github.com/wzcai99/Pixel-Navigator) as the base navigation policy. The `pixnav/` directory contains adapted code from the original PixelNav repository with modifications for integration with the RecNav recovery framework (e.g., deterministic VLM calls, proximity-based stopping).

## Citation

Citation will be provided upon paper acceptance.

## License

This project is released under the MIT License. See [LICENSE](LICENSE) for details.

The `pixnav/` directory is adapted from [PixelNav](https://github.com/wzcai99/Pixel-Navigator) — please refer to the original repository for its licensing terms.

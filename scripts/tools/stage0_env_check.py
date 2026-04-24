#!/usr/bin/env python3
"""Stage 0: Environment verification probe for PixNav + Habitat."""

import os
import sys
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[2]
PIXNAV_DIR = ROOT_DIR / "pixnav"
sys.path.insert(0, str(PIXNAV_DIR))

print("=" * 60)
print("Stage 0: Environment Check")
print("=" * 60)

# 1. Core imports
import habitat

print(f"[OK] habitat {habitat.__version__}")

import torch

cuda_info = "N/A"
if torch.cuda.is_available():
    cuda_info = torch.cuda.get_device_name(0)
print(f"[OK] torch {torch.__version__} | CUDA: {torch.cuda.is_available()} | GPU: {cuda_info}")

import cv2

print(f"[OK] cv2 {cv2.__version__}")

import numpy as np

print(f"[OK] numpy {np.__version__}")

# 2. PixNav constants
from constants import (
    EPISODE_PREFIX,
    GROUNDING_DINO_CHECKPOINT_PATH,
    GROUNDING_DINO_CONFIG_PATH,
    HABITAT_ROOT_DIR,
    HM3D_CONFIG_PATH,
    POLICY_CHECKPOINT,
    SAM_CHECKPOINT_PATH,
    SCENE_PREFIX,
)

print(f"\nHABITAT_ROOT_DIR: {HABITAT_ROOT_DIR}")
print(f"HM3D_CONFIG_PATH: {HM3D_CONFIG_PATH}")
print(f"SCENE_PREFIX:      {SCENE_PREFIX}")
print(f"EPISODE_PREFIX:    {EPISODE_PREFIX}")

# 3. File existence
paths = {
    "policy_checkpoint": POLICY_CHECKPOINT,
    "grounding_dino_config": GROUNDING_DINO_CONFIG_PATH,
    "grounding_dino_ckpt": GROUNDING_DINO_CHECKPOINT_PATH,
    "sam_checkpoint": SAM_CHECKPOINT_PATH,
    "scene_dataset_config": os.path.join(
        SCENE_PREFIX, "hm3d_v0.2", "hm3d_annotated_basis.scene_dataset_config.json"
    ),
    "episode_data": EPISODE_PREFIX + "objectnav/hm3d/v2/val/val.json.gz",
}

print("\nFile existence check:")
all_ok = True
for name, path in paths.items():
    exists = os.path.exists(path)
    size_mb = os.path.getsize(path) / 1e6 if exists else 0
    status = f"[OK] {size_mb:.1f}MB" if exists else "[MISSING]"
    print(f"  {name}: {status}  ({path})")
    if not exists:
        all_ok = False

# 4. GroundingDINO import
print("\nGroundingDINO import:")
try:
    from cv_utils.detection_tools import initialize_dino_model

    _ = initialize_dino_model
    print("  [OK] detection_tools import")
except Exception as exc:
    print(f"  [WARN] {exc}")

# 5. SAM import
print("SAM import:")
try:
    from cv_utils.segmentation_tools import initialize_sam_model

    _ = initialize_sam_model
    print("  [OK] segmentation_tools import")
except Exception as exc:
    print(f"  [WARN] {exc}")

# 6. Policy network import
print("Policy network import:")
try:
    from policy_agent import Policy_Agent

    _ = Policy_Agent
    print("  [OK] Policy_Agent import")
except Exception as exc:
    print(f"  [WARN] {exc}")

# 7. Habitat config load test
print("\nHabitat config load test (hm3d val, 2 episodes):")
try:
    from config_utils import hm3d_config

    cfg = hm3d_config(stage="val", episodes=2)
    print("  [OK] hm3d_config loaded")
    env = habitat.Env(cfg)
    print(f"  [OK] Env created, num_episodes={env.number_of_episodes}")
    obs = env.reset()
    print(f"  [OK] First reset, rgb shape={obs['rgb'].shape}, goal={env.current_episode.object_category}")
    metrics = env.get_metrics()
    print(f"  [OK] Metrics keys: {list(metrics.keys())}")
    env.close()
    print("  [OK] Env closed")
except Exception as exc:
    print(f"  [FAIL] {exc}")
    import traceback

    traceback.print_exc()
    all_ok = False

print("\n" + "=" * 60)
if all_ok:
    print("Stage 0 environment check: ALL PASSED")
else:
    print("Stage 0 environment check: SOME ISSUES (see above)")
print("=" * 60)

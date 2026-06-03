import os


_PIXNAV_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_PIXNAV_DIR)

# Habitat / ObjectNav paths. Prefer explicit environment variables so the
# release does not depend on a local filesystem layout.
HABITAT_ROOT_DIR = os.environ.get("PIXNAV_HABITAT_ROOT_DIR", "")
HM3D_CONFIG_PATH = os.environ.get(
    "PIXNAV_HM3D_CONFIG_PATH",
    os.environ.get("HABITAT_CONFIG_PATH", "benchmark/nav/objectnav/objectnav_hm3d.yaml"),
)
MP3D_CONFIG_PATH = os.environ.get(
    "PIXNAV_MP3D_CONFIG_PATH",
    "benchmark/nav/objectnav/objectnav_mp3d.yaml",
)
SCENE_PREFIX = os.environ.get("PIXNAV_SCENE_PREFIX", os.path.join(_REPO_ROOT, "data", "scene_datasets") + os.sep)
EPISODE_PREFIX = os.environ.get("PIXNAV_EPISODE_PREFIX", os.path.join(_REPO_ROOT, "data", "datasets") + os.sep)

# Detection / segmentation configs and checkpoints.
GROUNDING_DINO_CONFIG_PATH = os.path.join(_PIXNAV_DIR, "checkpoints", "GroundingDINO_SwinB_cfg.py")
GROUNDING_DINO_CHECKPOINT_PATH = os.path.join(_PIXNAV_DIR, "checkpoints", "groundingdino_swinb_cogcoor.pth")
SAM_ENCODER_VERSION = "vit_h"
SAM_CHECKPOINT_PATH = os.path.join(_PIXNAV_DIR, "checkpoints", "sam_vit_h_4b8939.pth")

# PixNav policy checkpoint.
POLICY_CHECKPOINT = os.path.join(_PIXNAV_DIR, "checkpoints", "navigator.pth")

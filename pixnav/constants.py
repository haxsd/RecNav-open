import os


_PIXNAV_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_PIXNAV_DIR)

# data-collection related directory
HABITAT_ROOT_DIR = os.environ.get("PIXNAV_HABITAT_ROOT_DIR", os.path.join(os.path.expanduser("~"), "habitat-lab"))
HM3D_CONFIG_PATH = os.environ.get(
    "PIXNAV_HM3D_CONFIG_PATH",
    f"{HABITAT_ROOT_DIR}/habitat-lab/habitat/config/benchmark/nav/objectnav/objectnav_hm3d.yaml",
)
MP3D_CONFIG_PATH = os.environ.get(
    "PIXNAV_MP3D_CONFIG_PATH",
    f"{HABITAT_ROOT_DIR}/habitat-lab/habitat/config/benchmark/nav/objectnav/objectnav_mp3d.yaml",
)
SCENE_PREFIX = os.environ.get("PIXNAV_SCENE_PREFIX", f"{_REPO_ROOT}/data/scene_datasets/")
EPISODE_PREFIX = os.environ.get("PIXNAV_EPISODE_PREFIX", f"{_REPO_ROOT}/data/datasets/")

# detection & segmentation related configs and checkpoints
GROUNDING_DINO_CONFIG_PATH = os.path.join(_PIXNAV_DIR, "checkpoints", "GroundingDINO_SwinB_cfg.py")
GROUNDING_DINO_CHECKPOINT_PATH = os.path.join(_PIXNAV_DIR, "checkpoints", "groundingdino_swinb_cogcoor.pth")
SAM_ENCODER_VERSION = "vit_h"
SAM_CHECKPOINT_PATH = os.path.join(_PIXNAV_DIR, "checkpoints", "sam_vit_h_4b8939.pth")

# policy checkpoint
POLICY_CHECKPOINT = os.path.join(_PIXNAV_DIR, "checkpoints", "navigator.pth")

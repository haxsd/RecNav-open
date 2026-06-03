"""Stage 2 PixNav host adapter.

This adapter intentionally wraps only the local PixelNav navigation skill.
High-level GPT4V planning and goal localization stay outside the host boundary.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
import sys

import numpy as np


def _ensure_pixnav_import_path() -> Path:
    repo_root = Path(__file__).resolve().parents[2]
    pixnav_dir = repo_root / "pixnav"
    pixnav_path = str(pixnav_dir)
    if pixnav_path not in sys.path:
        sys.path.insert(0, pixnav_path)
    return pixnav_dir


_PIXNAV_DIR = _ensure_pixnav_import_path()

from constants import POLICY_CHECKPOINT  # type: ignore  # noqa: E402
from policy_agent import Policy_Agent  # type: ignore  # noqa: E402


@dataclass
class PixNavHostTelemetry:
    """Minimal telemetry exported by the Stage 2 host contract."""

    steps_since_reset: int = 0
    collision_steps: int = 0
    goal_bound: int = 0
    last_action: int = -1
    last_error: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "steps_since_reset": int(self.steps_since_reset),
            "collision_steps": int(self.collision_steps),
            "goal_bound": int(self.goal_bound),
            "last_action": int(self.last_action),
            "last_error": str(self.last_error),
        }


class PixNavHost:
    """Skill-only PixNav host used to freeze the Stage 2 host boundary.

    The host accepts an externally prepared pixel goal (`goal_image`, `goal_mask`)
    and exposes the PixelNav action prediction as the host action. It does not own
    high-level semantic search, target localization, or GPT4V planning.
    """

    def __init__(
        self,
        *,
        checkpoint_path: str | None = None,
        device: str = "cuda:0",
        enable_debug_images: bool = False,
    ) -> None:
        self.checkpoint_path = str(checkpoint_path or POLICY_CHECKPOINT)
        self.device = str(device)
        self.enable_debug_images = bool(enable_debug_images)
        self._policy = Policy_Agent(model_path=self.checkpoint_path, device=self.device)
        self._telemetry = PixNavHostTelemetry()
        self._goal_category = ""
        self._goal_bound = False
        self._last_debug_image: np.ndarray | None = None

    def reset(
        self,
        goal_category: str,
        *,
        goal_image: np.ndarray | None = None,
        goal_mask: np.ndarray | None = None,
    ) -> None:
        self._goal_category = str(goal_category)
        self._telemetry = PixNavHostTelemetry()
        self._goal_bound = False
        self._last_debug_image = None
        if goal_image is not None and goal_mask is not None:
            self._policy.reset(goal_image, goal_mask)
            self._goal_bound = True
            self._telemetry.goal_bound = 1

    def act(self, obs_rgb: np.ndarray, collided: bool = False) -> int:
        if not self._goal_bound:
            self._telemetry.last_error = (
                "PixNavHost requires externally supplied goal_image/goal_mask. "
                "High-level planning is outside the Stage 2 host boundary."
            )
            raise RuntimeError(self._telemetry.last_error)

        if self.enable_debug_images:
            action, debug_image = self._policy.step(
                obs_rgb,
                collide=collided,
                return_debug_image=True,
            )
            self._last_debug_image = debug_image
        else:
            action = self._policy.step(
                obs_rgb,
                collide=collided,
                return_debug_image=False,
            )
            self._last_debug_image = None

        self._telemetry.steps_since_reset += 1
        self._telemetry.last_action = int(action)
        if collided:
            self._telemetry.collision_steps += 1
        self._telemetry.last_error = ""
        return int(action)

    def get_host_state(self) -> dict[str, Any]:
        return {
            "host_name": "pixnav_skill_only",
            "host_version": "r14_stage2",
            "pixnav_dir": str(_PIXNAV_DIR),
            "checkpoint_path": str(self.checkpoint_path),
            "device": str(self.device),
            "goal_category": str(self._goal_category),
            "goal_bound": bool(self._goal_bound),
            "enable_debug_images": bool(self.enable_debug_images),
            "action_space_note": "Habitat discrete action ids from PixNav policy output.",
            "telemetry": self.export_telemetry(),
        }

    def export_telemetry(self) -> dict[str, Any]:
        payload = self._telemetry.to_dict()
        payload["goal_category"] = str(self._goal_category)
        payload["has_debug_image"] = int(self._last_debug_image is not None)
        return payload

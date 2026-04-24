"""Habitat episode runner with normalized step outputs."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import numpy as np


@dataclass(frozen=True)
class Position3D:
    x: float
    y: float
    z: float


@dataclass
class StepResult:
    obs: Any
    done: bool
    info: dict[str, Any]
    metrics: dict[str, Any]
    collision: bool
    position: Position3D | None
    heading_rad: float | None = None
    rotation_wxyz: tuple[float, float, float, float] | None = None


class HabitatEnvRunner:
    """Thin wrapper around Habitat to keep script logic focused on algorithms."""

    def __init__(
        self,
        *,
        config_path: str,
        overrides: list[str] | None = None,
        position_source: str = "auto",
    ) -> None:
        self.config_path = config_path
        self.overrides = list(overrides or [])
        source = str(position_source).strip().lower()
        self.position_source = source if source in {"auto", "sim", "gps"} else "auto"
        self._habitat = None
        self._cfg = None
        self._env = None

    @staticmethod
    def _load_habitat_config(habitat_mod: Any, config_path: str, overrides: list[str]) -> Any:
        if hasattr(habitat_mod, "get_config"):
            return habitat_mod.get_config(config_path=config_path, overrides=overrides)

        try:
            from habitat.config.default import get_config  # type: ignore

            try:
                return get_config(config_path=config_path, overrides=overrides)
            except TypeError:
                return get_config(config_paths=config_path, overrides=overrides)
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError("Failed to resolve Habitat get_config API.") from exc

    def open(self) -> None:
        if self._env is not None:
            return
        try:
            import habitat  # type: ignore
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError("Cannot import habitat. Run inside Habitat env.") from exc
        self._habitat = habitat
        self._cfg = self._load_habitat_config(habitat, self.config_path, self.overrides)
        self._env = habitat.Env(config=self._cfg)

    def close(self) -> None:
        if self._env is not None:
            try:
                self._env.close()
            finally:
                self._env = None
                self._cfg = None

    def __enter__(self) -> "HabitatEnvRunner":
        self.open()
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self.close()

    @property
    def env(self) -> Any:
        if self._env is None:
            raise RuntimeError("Runner is not opened.")
        return self._env

    @property
    def cfg(self) -> Any:
        if self._cfg is None:
            raise RuntimeError("Runner is not opened.")
        return self._cfg

    def episodes(self) -> list[Any]:
        return list(getattr(self.env, "episodes", []) or [])

    def set_episode(self, episode: Any) -> None:
        self.env.current_episode = episode

    def current_episode(self) -> Any:
        return getattr(self.env, "current_episode", None)

    def get_action_keys(self) -> list[str]:
        action_space = getattr(self.env, "action_space", None)
        if action_space is None:
            return []
        spaces = getattr(action_space, "spaces", None)
        if isinstance(spaces, dict):
            return list(spaces.keys())
        return []

    @staticmethod
    def resolve_collision(info: Any) -> bool:
        if not isinstance(info, dict):
            return False
        if isinstance(info.get("collision"), bool):
            return bool(info["collision"])
        collisions = info.get("collisions")
        if isinstance(collisions, dict) and isinstance(collisions.get("is_collision"), bool):
            return bool(collisions["is_collision"])
        return False

    def safe_metrics(self) -> dict[str, Any]:
        if not hasattr(self.env, "get_metrics"):
            return {}
        try:
            metrics = self.env.get_metrics()
            if isinstance(metrics, dict):
                return metrics
        except Exception:  # noqa: BLE001
            return {}
        return {}

    @staticmethod
    def parse_result(result: Any, env: Any) -> tuple[Any, bool, dict[str, Any]]:
        if isinstance(result, tuple) and len(result) >= 4:
            obs, _, done, info = result[0], result[1], bool(result[2]), result[3]
            return obs, done, info if isinstance(info, dict) else {}
        obs = result
        done = bool(getattr(env, "episode_over", False))
        info: dict[str, Any] = {}
        if hasattr(env, "get_metrics"):
            metrics = env.get_metrics()
            if isinstance(metrics, dict):
                info = metrics
        return obs, done, info

    def _extract_position_from_info(self, info: dict[str, Any]) -> Position3D | None:
        for key in ("agent_position", "position"):
            pos = info.get(key)
            if isinstance(pos, (list, tuple)) and len(pos) >= 3:
                return Position3D(float(pos[0]), float(pos[1]), float(pos[2]))
            if isinstance(pos, np.ndarray) and pos.shape[0] >= 3:
                return Position3D(float(pos[0]), float(pos[1]), float(pos[2]))
        return None

    def _extract_position_from_sim(self) -> Position3D | None:
        sim = getattr(self.env, "sim", None)
        if sim is not None:
            try:
                state = sim.get_agent_state()
                pos = state.position
                return Position3D(float(pos[0]), float(pos[1]), float(pos[2]))
            except Exception:  # noqa: BLE001
                return None
        return None

    @staticmethod
    def _extract_position_from_gps(obs: Any) -> Position3D | None:
        if isinstance(obs, dict):
            gps = obs.get("gps")
            if isinstance(gps, np.ndarray) and gps.size >= 2:
                return Position3D(float(gps[0]), 0.0, float(gps[1]))
            if isinstance(gps, (list, tuple)) and len(gps) >= 2:
                return Position3D(float(gps[0]), 0.0, float(gps[1]))
        return None

    def extract_position(self, info: dict[str, Any], obs: Any) -> Position3D | None:
        if self.position_source == "gps":
            pos = self._extract_position_from_gps(obs)
            if pos is not None:
                return pos
            pos = self._extract_position_from_info(info)
            if pos is not None:
                return pos
            return self._extract_position_from_sim()

        if self.position_source == "sim":
            pos = self._extract_position_from_info(info)
            if pos is not None:
                return pos
            pos = self._extract_position_from_sim()
            if pos is not None:
                return pos
            return self._extract_position_from_gps(obs)

        # auto
        pos = self._extract_position_from_info(info)
        if pos is not None:
            return pos
        pos = self._extract_position_from_sim()
        if pos is not None:
            return pos
        return self._extract_position_from_gps(obs)

    @staticmethod
    def rotation_to_wxyz(rotation: Any) -> tuple[float, float, float, float] | None:
        if rotation is None:
            return None
        if isinstance(rotation, np.ndarray):
            flat = rotation.reshape(-1)
            if flat.size >= 4:
                return float(flat[0]), float(flat[1]), float(flat[2]), float(flat[3])
        if isinstance(rotation, (list, tuple)) and len(rotation) >= 4:
            return float(rotation[0]), float(rotation[1]), float(rotation[2]), float(rotation[3])

        real = getattr(rotation, "real", None)
        imag = getattr(rotation, "imag", None)
        if real is not None and imag is not None:
            if isinstance(imag, np.ndarray):
                flat = imag.reshape(-1)
                if flat.size >= 3:
                    return float(real), float(flat[0]), float(flat[1]), float(flat[2])
            if isinstance(imag, (list, tuple)) and len(imag) >= 3:
                return float(real), float(imag[0]), float(imag[1]), float(imag[2])

        components = getattr(rotation, "components", None)
        if components is not None:
            flat = np.asarray(components).reshape(-1)
            if flat.size >= 4:
                return float(flat[0]), float(flat[1]), float(flat[2]), float(flat[3])

        return None

    @classmethod
    def heading_from_rotation(cls, rotation: Any) -> float | None:
        wxyz = cls.rotation_to_wxyz(rotation)
        if wxyz is None:
            return None
        w, x, y, z = wxyz
        forward_x = -2.0 * (x * z + y * w)
        forward_z = -(1.0 - 2.0 * (x * x + y * y))
        if not np.isfinite(forward_x) or not np.isfinite(forward_z):
            return None
        return float(math.atan2(forward_x, -forward_z))

    def extract_rotation(self, info: dict[str, Any], obs: Any) -> tuple[float, float, float, float] | None:
        for key in ("agent_rotation", "rotation"):
            rot = info.get(key)
            wxyz = self.rotation_to_wxyz(rot)
            if wxyz is not None:
                return wxyz

        sim = getattr(self.env, "sim", None)
        if sim is not None:
            try:
                state = sim.get_agent_state()
                return self.rotation_to_wxyz(state.rotation)
            except Exception:  # noqa: BLE001
                pass

        if isinstance(obs, dict):
            compass = obs.get("compass")
            if isinstance(compass, np.ndarray) and compass.size >= 1:
                heading = float(compass.reshape(-1)[0])
                half = heading / 2.0
                return float(math.cos(half)), 0.0, float(math.sin(half)), 0.0
            if isinstance(compass, (list, tuple)) and len(compass) >= 1:
                heading = float(compass[0])
                half = heading / 2.0
                return float(math.cos(half)), 0.0, float(math.sin(half)), 0.0
        return None

    def extract_heading(self, info: dict[str, Any], obs: Any) -> float | None:
        rotation = self.extract_rotation(info, obs)
        if rotation is not None:
            return self.heading_from_rotation(rotation)
        return None

    def success_distance(self) -> float:
        try:
            dist = float(self.cfg.habitat.task.measurements.success.success_distance)
            if dist > 0:
                return dist
        except Exception:  # noqa: BLE001
            pass
        return 0.1

    def reset(self) -> StepResult:
        obs = self.env.reset()
        metrics = self.safe_metrics()
        position = self.extract_position(metrics, obs)
        rotation_wxyz = self.extract_rotation(metrics, obs)
        heading_rad = self.heading_from_rotation(rotation_wxyz)
        return StepResult(
            obs=obs,
            done=bool(getattr(self.env, "episode_over", False)),
            info=metrics,
            metrics=metrics,
            collision=False,
            position=position,
            heading_rad=heading_rad,
            rotation_wxyz=rotation_wxyz,
        )

    def step(self, action: str) -> StepResult:
        result = self.env.step(action)
        obs, done, info = self.parse_result(result, self.env)
        collision = self.resolve_collision(info)
        metrics = self.safe_metrics()
        if info and metrics:
            merged = dict(metrics)
            merged.update(info)
        else:
            merged = dict(metrics or info)
        position = self.extract_position(merged, obs)
        rotation_wxyz = self.extract_rotation(merged, obs)
        heading_rad = self.heading_from_rotation(rotation_wxyz)
        return StepResult(
            obs=obs,
            done=bool(done or getattr(self.env, "episode_over", False)),
            info=merged,
            metrics=metrics,
            collision=collision,
            position=position,
            heading_rad=heading_rad,
            rotation_wxyz=rotation_wxyz,
        )

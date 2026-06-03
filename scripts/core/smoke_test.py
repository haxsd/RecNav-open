#!/usr/bin/env python
"""Run a minimal Habitat smoke test and write step logs."""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

DEFAULT_OBJECTNAV_V2_DATA_PATH = "data/datasets/objectnav/hm3d/v2/\\{split\\}/\\{split\\}.json.gz"
DEFAULT_DATASET_SPLIT = "val_mini"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a minimal Habitat reset+step smoke test.")
    parser.add_argument(
        "--config-path",
        default=os.environ.get("HABITAT_CONFIG_PATH", "benchmark/nav/objectnav/objectnav_hm3d.yaml"),
        help="Habitat config path.",
    )
    parser.add_argument("--steps", type=int, default=20, help="Number of steps to execute.")
    parser.add_argument("--seed", type=int, default=123, help="Random seed.")
    parser.add_argument(
        "--actions",
        default="move_forward,turn_left,turn_right",
        help="Comma-separated action sequence.",
    )
    parser.add_argument(
        "--log-file",
        default="logs/smoke_test.jsonl",
        help="Path to JSONL log output.",
    )
    parser.add_argument(
        "--override",
        action="append",
        default=[],
        help="Config override in KEY=VALUE form. Repeatable.",
    )
    return parser


def load_habitat_config(config_path: str, overrides: list[str]):
    import habitat

    if hasattr(habitat, "get_config"):
        return habitat.get_config(config_path=config_path, overrides=overrides)

    try:
        from habitat.config.default import get_config  # type: ignore

        try:
            return get_config(config_path=config_path, overrides=overrides)
        except TypeError:
            return get_config(config_paths=config_path, overrides=overrides)
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError("Failed to resolve Habitat get_config API.") from exc


def resolve_collision(info: Any) -> bool:
    if not isinstance(info, dict):
        return False
    if isinstance(info.get("collision"), bool):
        return bool(info["collision"])
    collisions = info.get("collisions")
    if isinstance(collisions, dict) and isinstance(collisions.get("is_collision"), bool):
        return bool(collisions["is_collision"])
    return False


def get_action_keys(env: Any) -> list[str]:
    action_space = getattr(env, "action_space", None)
    if action_space is None:
        return []
    spaces = getattr(action_space, "spaces", None)
    if isinstance(spaces, dict):
        return list(spaces.keys())
    return []


def write_jsonl(path: Path, record: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def main() -> int:
    args = build_parser().parse_args()

    try:
        import habitat
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError("Cannot import habitat. Run inside your WSL+conda Habitat env.") from exc

    actions = [a.strip() for a in args.actions.split(",") if a.strip()]
    if not actions:
        raise ValueError("No valid actions provided.")

    log_path = Path(args.log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    if log_path.exists():
        log_path.unlink()

    overrides = list(args.override)
    if not any(o.startswith("habitat.dataset.data_path=") for o in overrides):
        overrides.append(f"habitat.dataset.data_path={DEFAULT_OBJECTNAV_V2_DATA_PATH}")
        print(f"INFO: Auto override dataset path -> {DEFAULT_OBJECTNAV_V2_DATA_PATH}")
    if not any(o.startswith("habitat.dataset.split=") for o in overrides):
        overrides.append(f"habitat.dataset.split={DEFAULT_DATASET_SPLIT}")
        print(f"INFO: Auto override dataset split -> {DEFAULT_DATASET_SPLIT}")

    cfg = load_habitat_config(args.config_path, overrides)

    env = habitat.Env(config=cfg)
    try:
        if hasattr(env, "seed"):
            env.seed(args.seed)

        observations = env.reset()
        episode = getattr(env, "current_episode", None)
        episode_id = getattr(episode, "episode_id", "unknown")
        scene_id = getattr(episode, "scene_id", "unknown")

        available_actions = get_action_keys(env)
        missing_actions = [a for a in actions if available_actions and a not in available_actions]
        if missing_actions:
            raise RuntimeError(
                f"Action(s) not available: {missing_actions}. Available: {available_actions}"
            )

        write_jsonl(
            log_path,
            {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "event": "reset",
                "seed": args.seed,
                "episode_id": episode_id,
                "scene_id": scene_id,
                "observation_keys": sorted(list(observations.keys())) if isinstance(observations, dict) else [],
            },
        )

        collisions = 0
        steps_executed = 0
        for step_idx in range(args.steps):
            action = actions[step_idx % len(actions)]
            result = env.step(action)

            info: dict[str, Any] = {}
            done = bool(getattr(env, "episode_over", False))

            if isinstance(result, tuple) and len(result) >= 4:
                _, _, done, info = result[0], result[1], bool(result[2]), result[3]
            elif hasattr(env, "get_metrics"):
                metrics = env.get_metrics()
                if isinstance(metrics, dict):
                    info = metrics

            is_collision = resolve_collision(info)
            collisions += int(is_collision)
            steps_executed += 1

            write_jsonl(
                log_path,
                {
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "event": "step",
                    "step": step_idx,
                    "action": action,
                    "collision": is_collision,
                    "done": done,
                },
            )

            print(f"step={step_idx:03d} action={action:<14} collision={is_collision} done={done}")
            if done:
                break

        print("\nSmoke test summary")
        print(f"- episode_id: {episode_id}")
        print(f"- scene_id: {scene_id}")
        print(f"- steps_executed: {steps_executed}")
        print(f"- collisions: {collisions}")
        print(f"- log_file: {log_path.resolve()}")
        return 0
    finally:
        env.close()


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:  # noqa: BLE001
        print(f"ERROR: {exc}", file=sys.stderr)
        raise

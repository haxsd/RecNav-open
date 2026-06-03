#!/usr/bin/env python
"""Validate HM3Dmini dataset layout and key config files."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Check Habitat dataset paths before running experiments.")
    parser.add_argument(
        "--data-root",
        default=os.environ.get("HABITAT_DATA_PATH", "data"),
        help="Habitat data root directory (default: $HABITAT_DATA_PATH or ./data)",
    )
    parser.add_argument(
        "--scene-dir",
        default="scene_datasets/hm3d",
        help="Relative path from data root to HM3D scene directory.",
    )
    parser.add_argument(
        "--scene-config",
        default="scene_datasets/hm3d/hm3d_annotated_basis.scene_dataset_config.json",
        help="Relative path from data root to scene dataset config.",
    )
    parser.add_argument(
        "--glob",
        default="*.basis.glb",
        help="Scene file glob pattern used to verify at least one scene exists.",
    )
    parser.add_argument(
        "--semantic-glob",
        default="*.semantic.glb",
        help="Semantic file glob pattern used to verify semantic assets exist.",
    )
    parser.add_argument(
        "--objectnav-root",
        default="datasets/objectnav/hm3d/v2",
        help="Relative path from data root to ObjectNav HM3D v2 dataset root.",
    )
    parser.add_argument(
        "--objectnav-splits",
        default="train,val,val_mini",
        help="Comma-separated ObjectNav splits expected under objectnav-root.",
    )
    return parser


def check_readable_file(path: Path) -> tuple[bool, str]:
    if not path.exists():
        return False, f"MISSING: {path}"
    if not path.is_file():
        return False, f"NOT_FILE: {path}"
    try:
        with path.open("rb") as f:
            f.read(1)
    except OSError as exc:
        return False, f"UNREADABLE: {path} ({exc})"
    return True, f"OK: {path}"


def check_readable_dir(path: Path) -> tuple[bool, str]:
    if not path.exists():
        return False, f"MISSING: {path}"
    if not path.is_dir():
        return False, f"NOT_DIR: {path}"
    try:
        next(path.iterdir(), None)
    except OSError as exc:
        return False, f"UNREADABLE: {path} ({exc})"
    return True, f"OK: {path}"


def main() -> int:
    args = build_parser().parse_args()
    data_root = Path(args.data_root).expanduser().resolve()

    scene_dir = data_root / args.scene_dir
    scene_config = data_root / args.scene_config
    objectnav_root = data_root / args.objectnav_root

    checks: list[tuple[bool, str]] = [
        check_readable_dir(data_root),
        check_readable_dir(scene_dir),
        check_readable_file(scene_config),
        check_readable_dir(objectnav_root),
    ]

    for _, message in checks:
        print(message)

    scene_files = sorted(scene_dir.rglob(args.glob)) if scene_dir.exists() else []
    if scene_files:
        print(f"OK: found {len(scene_files)} scene file(s), example: {scene_files[0]}")
    else:
        print(f"MISSING: no scene files matching '{args.glob}' under {scene_dir}")
        checks.append((False, "scene files missing"))

    semantic_files = sorted(scene_dir.rglob(args.semantic_glob)) if scene_dir.exists() else []
    if semantic_files:
        print(f"OK: found {len(semantic_files)} semantic file(s), example: {semantic_files[0]}")
    else:
        print(f"MISSING: no semantic files matching '{args.semantic_glob}' under {scene_dir}")
        checks.append((False, "semantic files missing"))

    splits = [s.strip() for s in args.objectnav_splits.split(",") if s.strip()]
    for split in splits:
        split_dir = objectnav_root / split
        split_json = split_dir / f"{split}.json.gz"
        checks.append(check_readable_dir(split_dir))
        checks.append(check_readable_file(split_json))

    failed = [msg for ok, msg in checks if not ok]
    if failed:
        print("\nDataset check failed. Missing items:")
        for msg in failed:
            print(f"- {msg}")
        return 1

    print("\nDataset check passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

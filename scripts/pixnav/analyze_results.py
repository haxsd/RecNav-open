#!/usr/bin/env python3
"""Summarize RecNav result CSV files found under an experiment root."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path


DEFAULT_BASE = Path("artifacts/runs/experiments")


def as_float(row: dict[str, str], *names: str, default: float = 0.0) -> float:
    for name in names:
        value = row.get(name)
        if value not in (None, ""):
            return float(value)
    return default


def load_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def summarize(rows: list[dict[str, str]]) -> dict[str, float]:
    n = len(rows)
    if n == 0:
        return {}
    recovery_attempts = sum(as_float(r, "recovery_attempts", "recovery_count") for r in rows)
    recovery_successes = sum(as_float(r, "recovery_success_count", "recovery_improved") for r in rows)
    return {
        "episodes": float(n),
        "sr": sum(as_float(r, "success") for r in rows) / n,
        "spl": sum(as_float(r, "spl") for r in rows) / n,
        "dtg": sum(as_float(r, "distance_to_goal", "dtg") for r in rows) / n,
        "recovery_attempts_per_ep": recovery_attempts / n,
        "recovery_success_rate": recovery_successes / recovery_attempts if recovery_attempts else float("nan"),
    }


def parse_run_name(path: Path) -> tuple[str, str]:
    variant_dir = path.parent.parent.name
    seed_dir = path.parent.name
    seed = seed_dir.removeprefix("seed_")
    return variant_dir, seed


def fmt(value: float) -> str:
    if value != value:
        return "NA"
    return f"{value:.4f}"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, default=DEFAULT_BASE)
    args = parser.parse_args()

    csv_paths = sorted(args.base.glob("*/seed_*/results.csv"))
    if not csv_paths:
        print(f"No result CSV files found under {args.base}")
        return 0

    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    per_seed: list[tuple[str, str, dict[str, float]]] = []

    for path in csv_paths:
        variant, seed = parse_run_name(path)
        rows = load_rows(path)
        grouped[variant].extend(rows)
        per_seed.append((variant, seed, summarize(rows)))

    print("Per-seed summaries")
    print("variant,seed,episodes,sr,spl,dtg,recovery_attempts_per_ep,recovery_success_rate")
    for variant, seed, stats in per_seed:
        print(
            ",".join(
                [
                    variant,
                    seed,
                    str(int(stats["episodes"])),
                    fmt(stats["sr"]),
                    fmt(stats["spl"]),
                    fmt(stats["dtg"]),
                    fmt(stats["recovery_attempts_per_ep"]),
                    fmt(stats["recovery_success_rate"]),
                ]
            )
        )

    print()
    print("Pooled summaries")
    print("variant,episodes,sr,spl,dtg,recovery_attempts_per_ep,recovery_success_rate")
    for variant in sorted(grouped):
        stats = summarize(grouped[variant])
        print(
            ",".join(
                [
                    variant,
                    str(int(stats["episodes"])),
                    fmt(stats["sr"]),
                    fmt(stats["spl"]),
                    fmt(stats["dtg"]),
                    fmt(stats["recovery_attempts_per_ep"]),
                    fmt(stats["recovery_success_rate"]),
                ]
            )
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

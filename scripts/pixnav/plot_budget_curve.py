#!/usr/bin/env python3
"""Plot SR and SPL across step budgets from RecNav result CSV files."""

from __future__ import annotations

import argparse
import csv
import re
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


BUDGET_RE = re.compile(r"_s(\d+)$")


def load_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def mean_metric(rows: list[dict[str, str]], name: str) -> float:
    if not rows:
        return float("nan")
    return sum(float(r[name]) for r in rows) / len(rows)


def parse_variant_budget(run_name: str) -> tuple[str, int] | None:
    match = BUDGET_RE.search(run_name)
    if not match:
        return None
    budget = int(match.group(1))
    variant = run_name[: match.start()]
    return variant, budget


def collect(base: Path, variants: list[str]) -> dict[str, dict[int, list[dict[str, str]]]]:
    data: dict[str, dict[int, list[dict[str, str]]]] = defaultdict(lambda: defaultdict(list))
    for path in sorted(base.glob("*/seed_*/results.csv")):
        parsed = parse_variant_budget(path.parent.parent.name)
        if not parsed:
            continue
        variant, budget = parsed
        if variant in variants:
            data[variant][budget].extend(load_rows(path))
    return data


def plot_metric(ax, data, variants: list[str], metric: str) -> None:
    for variant in variants:
        budgets = sorted(data.get(variant, {}))
        values = [mean_metric(data[variant][budget], metric) for budget in budgets]
        if budgets:
            ax.plot(budgets, values, marker="o", label=variant)
    ax.set_xlabel("Step budget")
    ax.set_ylabel(metric.upper())
    ax.grid(True, alpha=0.25)
    ax.legend()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, default=Path("artifacts/runs/experiments"))
    parser.add_argument("--out", type=Path, default=Path("figures/budget_curve.pdf"))
    parser.add_argument("--variants", default="f0,adarec_locked")
    args = parser.parse_args()

    variants = [v.strip() for v in args.variants.split(",") if v.strip()]
    data = collect(args.base, variants)

    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    plot_metric(axes[0], data, variants, "success")
    plot_metric(axes[1], data, variants, "spl")
    fig.tight_layout()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=180, bbox_inches="tight")
    print(f"Saved {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

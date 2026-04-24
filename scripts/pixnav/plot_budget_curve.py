#!/usr/bin/env python3
"""
Generate Budget-Performance curve (Figure 1) from experiment results.
Plots SR and SPL vs step budget for F0 and AdaRec, with per-seed markers.

Usage: python plot_budget_curve.py [--base /path/to/experiments] [--out figure1_budget.pdf]
"""
import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

BASE = Path(__file__).resolve().parents[2] / "artifacts" / "runs" / "experiments"
SEEDS = [123, 231, 777]
BUDGETS = [100, 200, 300, 400, 500]


def load_csv(path: str) -> list[dict]:
    with open(path) as f:
        return list(csv.DictReader(f))


def get_metrics(variant: str, budget: int) -> dict:
    """Return {seed: {SR, SPL}} for a variant at a given budget."""
    results = {}
    for seed in SEEDS:
        if budget == 500:
            p = BASE / variant / f"seed_{seed}" / "results.csv"
        else:
            p = BASE / f"{variant}_s{budget}" / f"seed_{seed}" / "results.csv"
        if p.exists():
            rows = load_csv(str(p))
            n = len(rows)
            sr = sum(float(r["success"]) for r in rows) / n
            spl = sum(float(r["spl"]) for r in rows) / n
            results[seed] = {"SR": sr, "SPL": spl}
    return results


def plot_curve(metric: str, ax, f0_data, ar_data):
    """Plot one metric (SR or SPL) on the given axes."""
    f0_means, f0_stds, ar_means, ar_stds = [], [], [], []
    valid_budgets = []

    for b in BUDGETS:
        f0_vals = [f0_data[b][s][metric] for s in SEEDS if s in f0_data.get(b, {})]
        ar_vals = [ar_data[b][s][metric] for s in SEEDS if s in ar_data.get(b, {})]
        if f0_vals and ar_vals:
            valid_budgets.append(b)
            f0_means.append(np.mean(f0_vals))
            f0_stds.append(np.std(f0_vals))
            ar_means.append(np.mean(ar_vals))
            ar_stds.append(np.std(ar_vals))

    if not valid_budgets:
        ax.text(0.5, 0.5, "No data yet", ha="center", va="center", transform=ax.transAxes)
        return

    x = np.array(valid_budgets)
    f0_m = np.array(f0_means)
    f0_s = np.array(f0_stds)
    ar_m = np.array(ar_means)
    ar_s = np.array(ar_stds)

    ax.fill_between(x, f0_m - f0_s, f0_m + f0_s, alpha=0.15, color="tab:gray")
    ax.fill_between(x, ar_m - ar_s, ar_m + ar_s, alpha=0.15, color="tab:blue")
    ax.plot(x, f0_m, "o--", color="tab:gray", label="PixNav (F0)", linewidth=2, markersize=6)
    ax.plot(x, ar_m, "s-", color="tab:blue", label="RecNav", linewidth=2, markersize=6)

    # Per-seed scatter
    for b_idx, b in enumerate(valid_budgets):
        for s in SEEDS:
            if s in f0_data.get(b, {}):
                ax.scatter(b, f0_data[b][s][metric], color="tab:gray", alpha=0.3, s=20, zorder=5)
            if s in ar_data.get(b, {}):
                ax.scatter(b, ar_data[b][s][metric], color="tab:blue", alpha=0.3, s=20, zorder=5)

    ax.set_xlabel("Step Budget", fontsize=12)
    ax.set_ylabel(metric, fontsize=12)
    ax.set_xticks(BUDGETS)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(50, 550)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=str, default=str(BASE))
    parser.add_argument("--out", type=str, default="figure1_budget_curve.pdf")
    args = parser.parse_args()
    global BASE
    BASE = Path(args.base)

    # Collect data
    f0_data = {b: get_metrics("f0", b) for b in BUDGETS}
    ar_data = {b: get_metrics("adarec", b) for b in BUDGETS}

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.5))
    plot_curve("SR", ax1, f0_data, ar_data)
    plot_curve("SPL", ax2, f0_data, ar_data)
    ax1.set_title("Success Rate vs Step Budget", fontsize=13)
    ax2.set_title("SPL vs Step Budget", fontsize=13)
    fig.suptitle("Budget-Performance Curve (HM3D ObjectNav)", fontsize=14, y=1.02)
    fig.tight_layout()

    out_path = BASE / args.out
    fig.savefig(str(out_path), dpi=150, bbox_inches="tight")
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""McNemar exact test for paired success/failure CSVs."""

from __future__ import annotations

import argparse
import csv
from math import comb
from pathlib import Path


TRUE_VALUES = {"1", "true", "yes", "y"}
FALSE_VALUES = {"0", "false", "no", "n"}


def parse_bool(value: str) -> bool:
    normalized = str(value).strip().lower()
    if normalized in TRUE_VALUES:
        return True
    if normalized in FALSE_VALUES:
        return False
    try:
        return float(normalized) != 0.0
    except ValueError as exc:
        raise ValueError(f"Cannot parse boolean success value: {value!r}") from exc


def load_success(path: Path) -> dict[str, bool]:
    with path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows: dict[str, bool] = {}
        for idx, row in enumerate(reader):
            key = row.get("episode_idx") or row.get("episode_id") or str(idx)
            value = row.get("success", row.get("Success"))
            if value is None:
                raise ValueError(f"{path} has no success column")
            rows[str(key)] = parse_bool(value)
    return rows


def exact_two_sided_p(wins: int, losses: int) -> float:
    discordant = wins + losses
    if discordant == 0:
        return 1.0
    k = min(wins, losses)
    p_value = sum(comb(discordant, i) * (0.5 ** discordant) for i in range(k + 1))
    return min(1.0, 2.0 * p_value)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", required=True, type=Path)
    parser.add_argument("--treatment", required=True, type=Path)
    args = parser.parse_args()

    baseline = load_success(args.baseline)
    treatment = load_success(args.treatment)
    common = sorted(set(baseline) & set(treatment))
    if not common:
        raise ValueError("No aligned episodes found between the two CSV files.")

    wins = losses = ties_success = ties_fail = 0
    for key in common:
        b = baseline[key]
        t = treatment[key]
        if not b and t:
            wins += 1
        elif b and not t:
            losses += 1
        elif b and t:
            ties_success += 1
        else:
            ties_fail += 1

    p_value = exact_two_sided_p(wins, losses)
    baseline_sr = sum(1 for key in common if baseline[key]) / len(common)
    treatment_sr = sum(1 for key in common if treatment[key]) / len(common)

    print(f"episodes: {len(common)}")
    print(f"baseline_sr: {baseline_sr:.4f}")
    print(f"treatment_sr: {treatment_sr:.4f}")
    print(f"delta_sr: {treatment_sr - baseline_sr:+.4f}")
    print(f"wins: {wins}")
    print(f"losses: {losses}")
    print(f"ties_success: {ties_success}")
    print(f"ties_fail: {ties_fail}")
    print(f"mcnemar_exact_p: {p_value:.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

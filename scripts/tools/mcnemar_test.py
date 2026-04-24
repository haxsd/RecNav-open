#!/usr/bin/env python3
"""McNemar's exact test for paired binary outcomes (success/failure per episode).

Usage:
    python mcnemar_test.py --baseline results_f0.csv --treatment results_recnav.csv

Both CSVs must have a boolean column 'success' (1/0 or True/False) with rows
aligned by episode (same episode order, same seed).

Output: wins, losses, ties, McNemar p-value, and significance verdict.
"""
import argparse
import csv
import sys
from pathlib import Path


def load_success_column(csv_path: str) -> list[bool]:
    rows = []
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            val = row.get("success", row.get("Success", ""))
            if val.lower() in ("1", "true", "yes"):
                rows.append(True)
            elif val.lower() in ("0", "false", "no"):
                rows.append(False)
            else:
                raise ValueError(f"Unrecognized success value: {val!r} in {csv_path}")
    return rows


def mcnemar_exact(baseline: list[bool], treatment: list[bool]) -> dict:
    assert len(baseline) == len(treatment), (
        f"Episode count mismatch: baseline={len(baseline)}, treatment={len(treatment)}"
    )
    n = len(baseline)
    wins = 0
    losses = 0
    ties_both_success = 0
    ties_both_fail = 0

    for b, t in zip(baseline, treatment):
        if not b and t:
            wins += 1
        elif b and not t:
            losses += 1
        elif b and t:
            ties_both_success += 1
        else:
            ties_both_fail += 1

    discordant = wins + losses

    if discordant == 0:
        p_value = 1.0
    else:
        from math import comb
        k = min(wins, losses)
        p_value = 0.0
        for i in range(k + 1):
            p_value += comb(discordant, i) * (0.5 ** discordant)
        p_value *= 2.0
        p_value = min(p_value, 1.0)

    return {
        "n_episodes": n,
        "wins": wins,
        "losses": losses,
        "ties_both_success": ties_both_success,
        "ties_both_fail": ties_both_fail,
        "baseline_sr": sum(baseline) / n,
        "treatment_sr": sum(treatment) / n,
        "delta_sr": (sum(treatment) - sum(baseline)) / n,
        "p_value": p_value,
        "significant_005": p_value < 0.05,
        "significant_001": p_value < 0.01,
    }


def main():
    parser = argparse.ArgumentParser(description="McNemar's exact test for paired episode outcomes.")
    parser.add_argument("--baseline", required=True, help="CSV file for baseline (f0) results.")
    parser.add_argument("--treatment", required=True, help="CSV file for treatment (RecNav) results.")
    args = parser.parse_args()

    baseline = load_success_column(args.baseline)
    treatment = load_success_column(args.treatment)
    result = mcnemar_exact(baseline, treatment)

    print(f"Episodes:       {result['n_episodes']}")
    print(f"Baseline SR:    {result['baseline_sr']:.1%}")
    print(f"Treatment SR:   {result['treatment_sr']:.1%}")
    print(f"Delta SR:       {result['delta_sr']:+.1%}")
    print(f"Wins:           {result['wins']}  (baseline fail -> treatment success)")
    print(f"Losses:         {result['losses']}  (baseline success -> treatment fail)")
    print(f"Ties (both OK): {result['ties_both_success']}")
    print(f"Ties (both F):  {result['ties_both_fail']}")
    print(f"McNemar p:      {result['p_value']:.4f}", end="")
    if result["significant_001"]:
        print("  **")
    elif result["significant_005"]:
        print("  *")
    else:
        print()

    if not result["significant_005"]:
        print("Not significant at p<0.05.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""
Auto-generate all experiment tables (E1-E7) from CSV results.
Run anytime — it will report whatever data is available so far.

Usage: python analyze_results.py [--base /path/to/experiments]
"""
import argparse
import csv
import os
import sys
from collections import defaultdict
from pathlib import Path

BASE = Path(__file__).resolve().parents[2] / "artifacts" / "runs" / "experiments"

def _update_base(p: Path):
    global BASE
    BASE = p

SEEDS = [123, 231, 777]

# ── helpers ──────────────────────────────────────────────────────────────────

def load_csv(path: str) -> list[dict]:
    with open(path) as f:
        return list(csv.DictReader(f))


def summarize(rows: list[dict]) -> dict:
    n = len(rows)
    if n == 0:
        return {}
    sr = sum(float(r["success"]) for r in rows) / n
    spl = sum(float(r["spl"]) for r in rows) / n
    dtg = sum(float(r["distance_to_goal"]) for r in rows) / n
    rec_attempts = sum(int(r.get("recovery_attempts", 0)) for r in rows)
    rec_success = sum(int(r.get("recovery_success_count", 0)) for r in rows)
    calls_per_ep = rec_attempts / n if n else 0
    rec_sr = rec_success / rec_attempts if rec_attempts > 0 else float("nan")
    return {
        "n": n, "SR": sr, "SPL": spl, "DTG": dtg,
        "Calls/Ep": calls_per_ep, "RecSR": rec_sr,
        "triggers": sum(int(r.get("trigger_count", 0)) for r in rows) / n,
        "precision": _safe_mean(rows, "trigger_precision"),
        "FTR": _safe_mean(rows, "false_trigger_rate"),
    }


def _safe_mean(rows, col):
    vals = [float(r[col]) for r in rows if col in r and r[col] not in ("", "nan", "None")]
    return sum(vals) / len(vals) if vals else float("nan")


def try_load(variant: str, seed: int, budget: int = 0) -> list[dict] | None:
    if budget > 0:
        d = BASE / f"{variant}_s{budget}" / f"seed_{seed}" / "results.csv"
    else:
        d = BASE / variant / f"seed_{seed}" / "results.csv"
    if d.exists():
        return load_csv(str(d))
    return None


def multi_seed_summary(variant: str, seeds: list[int], budget: int = 0) -> dict | None:
    all_rows = []
    found_seeds = 0
    for s in seeds:
        rows = try_load(variant, s, budget)
        if rows:
            all_rows.extend(rows)
            found_seeds += 1
    if not all_rows:
        return None
    result = summarize(all_rows)
    result["seeds_found"] = found_seeds
    return result


def fmt(val, digits=3):
    if val is None or (isinstance(val, float) and (val != val)):  # NaN
        return "—"
    return f"{val:.{digits}f}"


# ── tables ───────────────────────────────────────────────────────────────────

def table_e1():
    """E1: Main Comparison Table (5 methods × 3 seeds)"""
    print("=" * 70)
    print("E1: Main Comparison Table (budget=500)")
    print("=" * 70)
    methods = [
        ("f0", "PixNav (F0)"),
        ("llm_replan", "+ LLM Replan"),
        ("random_escape", "+ Random Escape"),
        ("backtrack", "+ Backtrack"),
        ("adarec", "RecNav"),
    ]
    print(f"{'Method':<20s} {'SR':>6s} {'SPL':>6s} {'DTG':>6s} {'Calls/Ep':>9s} {'RecSR':>6s} {'Seeds':>5s}")
    print("-" * 60)
    for var, label in methods:
        s = multi_seed_summary(var, SEEDS)
        if s:
            print(f"{label:<20s} {fmt(s['SR']):>6s} {fmt(s['SPL']):>6s} {fmt(s['DTG'],2):>6s} "
                  f"{fmt(s['Calls/Ep'],2):>9s} {fmt(s['RecSR']):>6s} {s['seeds_found']:>5d}")
        else:
            print(f"{label:<20s} {'(no data)':>40s}")
    print()


def table_e2():
    """E2: Budget-Performance Curve"""
    print("=" * 70)
    print("E2: Budget-Performance Curve")
    print("=" * 70)
    budgets = [100, 200, 300, 400, 500]
    print(f"{'Budget':>6s}  {'F0 SR':>6s} {'F0 SPL':>7s}  {'AR SR':>6s} {'AR SPL':>7s}  {'ΔSR':>6s} {'ΔSPL':>7s}")
    print("-" * 60)
    for b in budgets:
        barg = b if b != 500 else 0
        f0 = multi_seed_summary("f0", SEEDS, barg)
        ar = multi_seed_summary("adarec", SEEDS, barg)
        if f0 and ar:
            dsr = ar["SR"] - f0["SR"]
            dspl = ar["SPL"] - f0["SPL"]
            print(f"{b:>6d}  {fmt(f0['SR']):>6s} {fmt(f0['SPL']):>7s}  "
                  f"{fmt(ar['SR']):>6s} {fmt(ar['SPL']):>7s}  "
                  f"{dsr:>+6.3f} {dspl:>+7.3f}")
        else:
            f0s = fmt(f0["SR"]) if f0 else "—"
            ars = fmt(ar["SR"]) if ar else "—"
            print(f"{b:>6d}  {f0s:>6s} {'—':>7s}  {ars:>6s} {'—':>7s}  {'—':>6s} {'—':>7s}")
    print()


def table_e3():
    """E3: Component Ablation"""
    print("=" * 70)
    print("E3: Component Ablation (seed=123)")
    print("=" * 70)
    variants = [
        ("adarec", "RecNav (full)"),
        ("adarec_no_memory", "− Memory"),
        ("adarec_no_gate", "− Gate"),
        ("f0", "− Recovery (F0)"),
    ]
    print(f"{'Variant':<22s} {'SR':>6s} {'SPL':>6s} {'RecSR':>6s}")
    print("-" * 45)
    for var, label in variants:
        s = multi_seed_summary(var, [123])
        if s:
            print(f"{label:<22s} {fmt(s['SR']):>6s} {fmt(s['SPL']):>6s} {fmt(s['RecSR']):>6s}")
        else:
            print(f"{label:<22s} {'(no data)':>30s}")
    print()


def table_e4():
    """E4: Recovery Design Ablation"""
    print("=" * 70)
    print("E4: Recovery Design Ablation (seed=123)")
    print("=" * 70)
    variants = [
        ("f0", "F0 (no recovery)"),
        ("backtrack", "R0 Backtrack"),
        ("r1_no_backtrack", "R1 No-backtrack"),
        ("adarec", "AdaRec Macro-action"),
    ]
    print(f"{'Variant':<22s} {'SR':>6s} {'SPL':>6s} {'Calls/Ep':>9s} {'RecSR':>6s}")
    print("-" * 55)
    for var, label in variants:
        s = multi_seed_summary(var, [123])
        if s:
            print(f"{label:<22s} {fmt(s['SR']):>6s} {fmt(s['SPL']):>6s} "
                  f"{fmt(s['Calls/Ep'],2):>9s} {fmt(s['RecSR']):>6s}")
        else:
            print(f"{label:<22s} {'(no data)':>35s}")
    print()


def table_e5():
    """E5: Gate Strategy Comparison"""
    print("=" * 70)
    print("E5: Gate Strategy (seed=123)")
    print("=" * 70)
    variants = [
        ("gate_always", "Always on STOP"),
        ("gate_fixed2", "Fixed k=2"),
        ("gate_fixed4", "Fixed k=4"),
        ("adarec", "Pose-Free (ours)"),
    ]
    print(f"{'Gate':<18s} {'SR':>6s} {'SPL':>6s} {'Trig/Ep':>8s} {'Prec':>6s} {'FTR':>6s}")
    print("-" * 50)
    for var, label in variants:
        s = multi_seed_summary(var, [123])
        if s:
            print(f"{label:<18s} {fmt(s['SR']):>6s} {fmt(s['SPL']):>6s} "
                  f"{fmt(s['triggers'],2):>8s} {fmt(s['precision']):>6s} {fmt(s['FTR']):>6s}")
        else:
            print(f"{label:<18s} {'(no data)':>35s}")
    print()


def table_e6():
    """E6: Memory Ablation"""
    print("=" * 70)
    print("E6: Memory Ablation (seed=123)")
    print("=" * 70)
    variants = [
        ("mem_disabled", "Disabled (dtg)"),
        ("mem_fifo", "FIFO (memory_dtg)"),
        ("adarec", "Skeleton (memory_dtg)"),
    ]
    print(f"{'Memory':<22s} {'SR':>6s} {'SPL':>6s} {'RecSR':>6s}")
    print("-" * 45)
    for var, label in variants:
        s = multi_seed_summary(var, [123])
        if s:
            print(f"{label:<22s} {fmt(s['SR']):>6s} {fmt(s['SPL']):>6s} {fmt(s['RecSR']):>6s}")
        else:
            print(f"{label:<22s} {'(no data)':>30s}")
    print()


def table_e7():
    """E7: Sensitivity Analysis"""
    print("=" * 70)
    print("E7: Sensitivity (seed=123)")
    print("=" * 70)
    variants = [
        ("sens_fan5", "Fan=5"),
        ("sens_fan9", "Fan=9"),
        ("adarec", "Fan=13 (default)"),
        ("sens_fwd1", "Fwd=1"),
        ("adarec", "Fwd=3 (default)"),
        ("sens_fwd5", "Fwd=5"),
        ("sens_no_abstain", "Abstain=Off"),
        ("adarec", "Abstain=On (default)"),
    ]
    print(f"{'Config':<20s} {'SR':>6s} {'SPL':>6s} {'RecSR':>6s}")
    print("-" * 40)
    for var, label in variants:
        s = multi_seed_summary(var, [123])
        if s:
            print(f"{label:<20s} {fmt(s['SR']):>6s} {fmt(s['SPL']):>6s} {fmt(s['RecSR']):>6s}")
        else:
            print(f"{label:<20s} {'(no data)':>25s}")
    print()


def progress_report():
    """Count completed runs."""
    total = 0
    for root, dirs, files in os.walk(BASE):
        for fn in files:
            if fn == "results.csv":
                total += 1
    print(f"[Progress] {total}/52 runs completed (CSV files found)")
    print()


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=str, default=str(BASE))
    args = parser.parse_args()
    _update_base(Path(args.base))

    progress_report()
    table_e1()
    table_e2()
    table_e3()
    table_e4()
    table_e5()
    table_e6()
    table_e7()


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Quick comparison: F0 vs AdaRec across all seeds and step budgets."""
import csv, os, glob

base = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "artifacts", "runs", "experiments")
variants = {}

for d in sorted(os.listdir(base)):
    full = os.path.join(base, d)
    if not os.path.isdir(full):
        continue
    for seed_dir in sorted(glob.glob(os.path.join(full, "seed_*"))):
        seed_name = os.path.basename(seed_dir)
        # skip backup dirs
        if "backup" in seed_name:
            continue
        csv_path = os.path.join(seed_dir, "results.csv")
        if not os.path.exists(csv_path):
            continue
        try:
            rows = list(csv.DictReader(open(csv_path)))
        except Exception:
            continue
        if not rows:
            continue
        n = len(rows)
        sr = sum(1 for r in rows if float(r.get("success", "0")) > 0.5) / n
        spl = sum(float(r.get("spl", "0")) for r in rows) / n
        soft_spl = sum(float(r.get("soft_spl", "0")) for r in rows) / n
        dtg = sum(float(r.get("distance_to_goal", "0")) for r in rows) / n
        rec_att = sum(int(r.get("recovery_attempts", "0")) for r in rows)
        rec_succ = sum(int(r.get("recovery_success_count", "0")) for r in rows)
        foc = sum(int(r.get("failure_opportunity_count", "0")) for r in rows)
        trig = sum(int(r.get("trigger_count", "0")) for r in rows)

        key = d
        if key not in variants:
            variants[key] = []
        variants[key].append(dict(
            seed=seed_name, n=n, sr=sr, spl=spl, soft_spl=soft_spl,
            dtg=dtg, rec_att=rec_att, rec_succ=rec_succ, foc=foc, trig=trig
        ))

hdr = f"{'Variant':<25} {'Seed':<15} {'N':>3} {'SR':>6} {'SPL':>6} {'SftSPL':>6} {'DTG':>6} {'Trig':>5} {'RecAtt':>6} {'RecSuc':>6} {'FOC':>4}"
print(hdr)
print("-" * len(hdr))

for var in sorted(variants.keys()):
    for s in variants[var]:
        print(f"{var:<25} {s['seed']:<15} {s['n']:>3} {s['sr']:>6.3f} {s['spl']:>6.3f} {s['soft_spl']:>6.3f} {s['dtg']:>6.2f} {s['trig']:>5} {s['rec_att']:>6} {s['rec_succ']:>6} {s['foc']:>4}")
    if len(variants[var]) > 1:
        total_n = sum(s['n'] for s in variants[var])
        avg_sr = sum(s['sr'] * s['n'] for s in variants[var]) / total_n
        avg_spl = sum(s['spl'] * s['n'] for s in variants[var]) / total_n
        avg_sspl = sum(s['soft_spl'] * s['n'] for s in variants[var]) / total_n
        avg_dtg = sum(s['dtg'] * s['n'] for s in variants[var]) / total_n
        print(f"  {'>> AVG':<23} {'':<15} {total_n:>3} {avg_sr:>6.3f} {avg_spl:>6.3f} {avg_sspl:>6.3f} {avg_dtg:>6.2f}")
    print()

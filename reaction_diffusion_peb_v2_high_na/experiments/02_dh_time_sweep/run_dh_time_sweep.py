"""Stage 2 — DH × time sweep on the Stage-1 clean geometry baseline.

Grid:
    DH_nm2_s ∈ {0.3, 0.5, 0.8, 1.0, 1.5}
    time_s   ∈ {15, 20, 30, 45, 60}
=> 25 runs at σ=0, kdep=0.5, kloss=0.005, Hmax=0.2, quencher=off.

Interior gate per run (same definition as Stage-1A):
    P_space_center_mean < 0.50
    P_line_center_mean  > 0.65
    contrast            > 0.15
    area_frac           < 0.90
    CD_final / pitch    < 0.85
    CD_final, LER_final finite

Selection (Stage 2 best operating point):
    among runs that pass the gate AND satisfy
        CD_shift_nm <= 3.0
        CD_final / pitch < 0.85
        area_frac < 0.90
    pick the one with the largest LER_reduction_pct.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from reaction_diffusion_peb_v2_high_na.experiments.run_sigma_sweep_helpers import (  # noqa: E402
    run_one_with_overrides,
    save_outputs,
)

V2_DIR = Path(__file__).resolve().parents[2]
DEFAULT_OUT = V2_DIR / "outputs"

CD_SHIFT_MAX = 3.0
CD_PITCH_FRAC_MAX_SEL = 0.85
AREA_FRAC_MAX_SEL = 0.90


def fail_reason(r: dict) -> str:
    """Return a short human-readable reason a run did not pass the interior gate."""
    if r["passed"]:
        return ""
    reasons = []
    if not r["cond_metrics"]:
        reasons.append("CD/LER not finite")
    if not r["cond_space"]:
        reasons.append(f"P_space_mean={r['P_space_center_mean']:.2f}>=0.50")
    if not r["cond_line"]:
        reasons.append(f"P_line_mean={r['P_line_center_mean']:.2f}<=0.65")
    if not r["cond_contrast"]:
        reasons.append(f"contrast={r['contrast']:.2f}<=0.15")
    if not r["cond_area"]:
        reasons.append(f"area_frac={r['area_frac']:.3f}>=0.90")
    if not r["cond_cd"]:
        reasons.append(f"CD/pitch={r['CD_pitch_frac']:.3f}>=0.85")
    return "; ".join(reasons)


def selection_reason(r: dict) -> str:
    """Reason a passing run is excluded from the best-operating-point selection."""
    if not r["passed"]:
        return "did not pass interior gate"
    issues = []
    if not (r["CD_shift_nm"] <= CD_SHIFT_MAX):
        issues.append(f"CD_shift={r['CD_shift_nm']:.2f}>{CD_SHIFT_MAX}")
    if not (r["CD_pitch_frac"] < CD_PITCH_FRAC_MAX_SEL):
        issues.append(f"CD/pitch={r['CD_pitch_frac']:.3f}>={CD_PITCH_FRAC_MAX_SEL}")
    if not (r["area_frac"] < AREA_FRAC_MAX_SEL):
        issues.append(f"area_frac={r['area_frac']:.3f}>={AREA_FRAC_MAX_SEL}")
    return "; ".join(issues)


def add_ler_pct(r: dict) -> dict:
    li = r["LER_initial_nm"]
    lf = r["LER_final_nm"]
    if np.isfinite(li) and np.isfinite(lf) and li > 0:
        r["LER_reduction_pct"] = float(100.0 * (li - lf) / li)
    else:
        r["LER_reduction_pct"] = float("nan")
    return r


def _print_grid(rows: list[dict], DHs: list[float], times: list[float]) -> None:
    """Print the LER reduction grid (DH rows × time cols) and the pass marker."""
    by_key = {(round(r["DH_nm2_s"], 4), round(r["time_s"], 4)): r for r in rows}
    print("\nLER reduction (%) — DH (rows) × time_s (cols):")
    header = "        " + " ".join(f"{t:>10.0f}" for t in times)
    print(header)
    for DH in DHs:
        cells = [f"{DH:>6.2f}: "]
        for t in times:
            r = by_key[(round(DH, 4), round(t, 4))]
            mark = "✓" if r["passed"] else "✗"
            cells.append(f"{r['LER_reduction_pct']:>8.2f}{mark}")
        print(" ".join(cells))


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, required=True)
    p.add_argument("--tag", type=str, default="02_dh_time_sweep")
    args = p.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())

    fig_dir = DEFAULT_OUT / "figures" / args.tag
    logs_dir = DEFAULT_OUT / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    DHs = [0.3, 0.5, 0.8, 1.0, 1.5]
    times = [15.0, 20.0, 30.0, 45.0, 60.0]

    print(f"=== Stage 2: DH × time sweep at σ=0, kdep=0.5, kloss=0.005, Hmax=0.2, quencher off ===")
    rows: list[dict] = []
    for DH in DHs:
        for t in times:
            r = run_one_with_overrides(cfg, sigma_nm=0.0, time_s=t, DH_nm2_s=DH)
            r = add_ler_pct(r)
            r["fail_reason"] = fail_reason(r)
            r["selection_reason"] = selection_reason(r)
            rows.append(r)

    save_outputs(rows, cfg, fig_dir, logs_dir / f"{args.tag}.csv")

    # Append fail/selection reasons + LER_reduction_pct to the CSV (save_outputs writes its own).
    extra_csv = logs_dir / f"{args.tag}_summary.csv"
    keys = [
        "DH_nm2_s", "time_s",
        "P_space_center_mean", "P_line_center_mean", "contrast",
        "area_frac", "CD_pitch_frac",
        "CD_initial_nm", "CD_final_nm", "CD_shift_nm",
        "LER_initial_nm", "LER_final_nm", "LER_reduction_pct",
        "passed", "fail_reason", "selection_reason",
    ]
    with extra_csv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow({k: r[k] for k in keys})

    # Pretty table.
    print("\n  DH    t       P_space   P_line  contrast  area    CD/p   CD_shift  LER_i   LER_f   LER%   pass  fail/select reason")
    for r in rows:
        mark = "✓" if r["passed"] else "✗"
        rsn = r["fail_reason"] if not r["passed"] else r["selection_reason"]
        print(f"  {r['DH_nm2_s']:>4.2f}  {r['time_s']:>4.0f}    "
              f"{r['P_space_center_mean']:>6.3f}  {r['P_line_center_mean']:>6.3f}   {r['contrast']:>6.3f}  "
              f"{r['area_frac']:>5.3f}  {r['CD_pitch_frac']:>5.3f}   {r['CD_shift_nm']:>+6.2f}   "
              f"{r['LER_initial_nm']:>5.2f}  {r['LER_final_nm']:>5.2f}  {r['LER_reduction_pct']:>+6.2f}  {mark}    {rsn}")

    _print_grid(rows, DHs, times)

    # ----- Best operating point selection -----
    candidates = [
        r for r in rows
        if r["passed"]
        and np.isfinite(r["LER_reduction_pct"])
        and r["CD_shift_nm"] <= CD_SHIFT_MAX
        and r["CD_pitch_frac"] < CD_PITCH_FRAC_MAX_SEL
        and r["area_frac"] < AREA_FRAC_MAX_SEL
    ]
    if not candidates:
        print("\nNo run satisfies the Stage-2 selection criteria.")
        return 1

    best = max(candidates, key=lambda r: r["LER_reduction_pct"])
    print("\n=== Stage 2 best operating point (max LER_reduction_pct subject to selection bounds) ===")
    summary = {
        "DH_nm2_s": best["DH_nm2_s"],
        "time_s": best["time_s"],
        "P_space_center_mean": best["P_space_center_mean"],
        "P_line_center_mean": best["P_line_center_mean"],
        "contrast": best["contrast"],
        "area_frac": best["area_frac"],
        "CD_initial_nm": best["CD_initial_nm"],
        "CD_final_nm": best["CD_final_nm"],
        "CD_shift_nm": best["CD_shift_nm"],
        "CD_pitch_frac": best["CD_pitch_frac"],
        "LER_initial_nm": best["LER_initial_nm"],
        "LER_final_nm": best["LER_final_nm"],
        "LER_reduction_pct": best["LER_reduction_pct"],
    }
    print(json.dumps(summary, indent=2))
    (logs_dir / f"{args.tag}_best.json").write_text(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

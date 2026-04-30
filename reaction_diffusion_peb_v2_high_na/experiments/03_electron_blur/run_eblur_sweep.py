"""Stage 3 — electron-blur σ sweep at two Stage-2 operating points.

OPs:
    robust OP             : DH=0.5 nm²/s, time=30 s
    algorithmic-best OP   : DH=0.8 nm²/s, time=20 s
σ sweep:
    sigma_nm ∈ {0, 1, 2, 3}

Gate (Stage-3 strengthened):
    interior gate (Stage 1A) AND
    P_line_margin = P_line_center_mean - 0.65 >= 0.03

Three-stage LER measurement:
    LER_design_initial    -> binary I       (sigma-independent)
    LER_after_eblur_H0    -> I_blurred
    LER_after_PEB_P       -> P (post-PEB)
    electron_blur_LER_reduction_pct = 100*(design - eblur)/design
    PEB_LER_reduction_pct           = 100*(eblur  - PEB) / eblur
    total_LER_reduction_pct         = 100*(design - PEB) / design
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

P_LINE_MARGIN_MIN = 0.03


def fail_reason(r: dict) -> str:
    if r["_stage3_passed"]:
        return ""
    reasons = []
    if not r["cond_metrics"]:
        reasons.append("CD/LER not finite")
    if not r["cond_space"]:
        reasons.append(f"P_space_mean={r['P_space_center_mean']:.2f}>=0.50")
    if not r["cond_line"]:
        reasons.append(f"P_line_mean={r['P_line_center_mean']:.2f}<=0.65")
    if r["cond_line"] and r["P_line_margin"] < P_LINE_MARGIN_MIN:
        reasons.append(f"P_line_margin={r['P_line_margin']:.3f}<0.03")
    if not r["cond_contrast"]:
        reasons.append(f"contrast={r['contrast']:.2f}<=0.15")
    if not r["cond_area"]:
        reasons.append(f"area_frac={r['area_frac']:.3f}>=0.90")
    if not r["cond_cd"]:
        reasons.append(f"CD/pitch={r['CD_pitch_frac']:.3f}>=0.85")
    return "; ".join(reasons)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, required=True)
    p.add_argument("--tag", type=str, default="03_electron_blur")
    args = p.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())

    fig_dir = DEFAULT_OUT / "figures" / args.tag
    logs_dir = DEFAULT_OUT / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    operating_points = [
        ("robust", 0.5, 30.0),
        ("algorithmic_best", 0.8, 20.0),
    ]
    sigmas = [0.0, 1.0, 2.0, 3.0]

    rows: list[dict] = []
    for op_name, DH, t in operating_points:
        for s in sigmas:
            r = run_one_with_overrides(cfg, sigma_nm=s, time_s=t, DH_nm2_s=DH)
            r["op_name"] = op_name
            # Stage-3 strengthened gate.
            r["_stage3_passed"] = bool(r["passed"] and r["P_line_margin"] >= P_LINE_MARGIN_MIN)
            r["fail_reason"] = fail_reason(r)
            rows.append(r)

    save_outputs(rows, cfg, fig_dir, logs_dir / f"{args.tag}.csv")

    # Compact summary CSV.
    src_keys = [
        "op_name", "sigma_nm", "DH_nm2_s", "time_s",
        "P_space_center_mean", "P_line_center_mean", "P_line_margin",
        "contrast", "area_frac", "CD_pitch_frac",
        "CD_initial_nm", "CD_final_nm", "CD_shift_nm",
        "LER_design_initial_nm", "LER_after_eblur_H0_nm", "LER_after_PEB_P_nm",
        "electron_blur_LER_reduction_pct", "PEB_LER_reduction_pct", "total_LER_reduction_pct",
    ]
    out_keys = src_keys + ["passed", "fail_reason"]
    summary_csv = logs_dir / f"{args.tag}_summary.csv"
    with summary_csv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=out_keys)
        w.writeheader()
        for r in rows:
            row = {k: r[k] for k in src_keys}
            row["passed"] = r["_stage3_passed"]
            row["fail_reason"] = r["fail_reason"]
            w.writerow(row)

    # Pretty table.
    print("=== Stage 3: electron-blur σ sweep at two Stage-2 OPs ===")
    print("OP        DH    t   σ    P_sp  P_ln   margin  cont   area   CD/p   CDshift   LER_des  LER_eb  LER_PEB   eblur%   PEB%    total%   pass  reason")
    for r in rows:
        mark = "✓" if r["_stage3_passed"] else "✗"
        rsn = r["fail_reason"]
        print(
            f"{r['op_name'][:9]:<9} "
            f"{r['DH_nm2_s']:>4.2f}  {r['time_s']:>4.0f}  {r['sigma_nm']:>3.0f}  "
            f"{r['P_space_center_mean']:>5.3f} {r['P_line_center_mean']:>5.3f}  {r['P_line_margin']:>+6.3f}  "
            f"{r['contrast']:>4.2f}  {r['area_frac']:>5.3f}  {r['CD_pitch_frac']:>5.3f}  "
            f"{r['CD_shift_nm']:>+6.2f}   "
            f"{r['LER_design_initial_nm']:>6.3f}  {r['LER_after_eblur_H0_nm']:>6.3f}  {r['LER_after_PEB_P_nm']:>6.3f}   "
            f"{r['electron_blur_LER_reduction_pct']:>+5.1f}  {r['PEB_LER_reduction_pct']:>+5.1f}   {r['total_LER_reduction_pct']:>+5.1f}    {mark}    {rsn}"
        )

    # Per-OP recommendation: largest σ that passes Stage-3 gate.
    print()
    for op_name, DH, t in operating_points:
        op_rows = [r for r in rows if r["op_name"] == op_name and r["_stage3_passed"]]
        if op_rows:
            best = max(op_rows, key=lambda r: r["sigma_nm"])
            print(f"[{op_name}]  DH={DH}, t={t}: largest passing σ = {best['sigma_nm']:.0f} nm  "
                  f"(total LER reduction = {best['total_LER_reduction_pct']:+.2f}%)")
        else:
            print(f"[{op_name}]  DH={DH}, t={t}: NO σ passes Stage-3 gate.")

    summary_json = logs_dir / f"{args.tag}_summary.json"
    summary_json.write_text(json.dumps(
        [{k: r[k] for k in src_keys} | {"passed": r["_stage3_passed"], "fail_reason": r["fail_reason"]}
         for r in rows],
        indent=2,
    ))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

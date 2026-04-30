"""Stage 4 — weak quencher sweep at the Stage-2 robust OP.

Operating point (fixed):
    DH = 0.5 nm^2/s, time = 30 s, kdep = 0.5 s^-1, Hmax = 0.2 mol/dm^3,
    kloss = 0.005 s^-1, pitch = 24 nm, line_cd = 12.5 nm.

Sweep:
    sigma_nm   ∈ {0, 1, 2, 3}
    Q0_mol_dm3 ∈ {0.0 (no-quencher baseline), 0.005, 0.01, 0.02, 0.03}
    kq_s_inv   ∈ {0.5, 1.0, 2.0}
    DQ_nm2_s   = 0.0
    Q0=0 rows: quencher disabled. Q0>0 rows: quencher enabled.

Total = 4 * (1 + 4*3) = 52 runs.

Per-row gate (Stage 3 strengthened):
    interior gate AND P_line_margin >= 0.03

Stage-4 robust-candidate criteria (vs same-sigma no-quencher baseline):
    passed (Stage 3 gate)
    P_line_margin >= 0.05
    dCD_shift_nm   < 0           # quencher reduces line widening
    darea_frac     < 0           # quencher reduces over-deprotect area
    dtotal_LER_pp >= -1.0        # total LER reduction does not drop > 1 pp
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from reaction_diffusion_peb_v2_high_na.experiments.run_sigma_sweep_helpers import (  # noqa: E402
    run_one_with_overrides,
)
from reaction_diffusion_peb_v2_high_na.src.visualization import plot_contour_overlay  # noqa: E402

V2_DIR = Path(__file__).resolve().parents[2]
DEFAULT_OUT = V2_DIR / "outputs"

P_LINE_MARGIN_GATE = 0.03
ROBUST_MARGIN_MIN = 0.05
DTOTAL_LER_MIN_PP = -1.0  # total LER reduction may drop at most 1 pp


def gate_pass(r: dict) -> bool:
    return bool(r["passed"] and r["P_line_margin"] >= P_LINE_MARGIN_GATE)


def fail_reason(r: dict) -> str:
    if gate_pass(r):
        return ""
    reasons = []
    if not r["cond_metrics"]:
        reasons.append("CD/LER not finite")
    if not r["cond_space"]:
        reasons.append(f"P_space_mean={r['P_space_center_mean']:.2f}>=0.50")
    if not r["cond_line"]:
        reasons.append(f"P_line_mean={r['P_line_center_mean']:.2f}<=0.65")
    if r["cond_line"] and r["P_line_margin"] < P_LINE_MARGIN_GATE:
        reasons.append(f"P_line_margin={r['P_line_margin']:.3f}<0.03")
    if not r["cond_contrast"]:
        reasons.append(f"contrast={r['contrast']:.2f}<=0.15")
    if not r["cond_area"]:
        reasons.append(f"area_frac={r['area_frac']:.3f}>=0.90")
    if not r["cond_cd"]:
        reasons.append(f"CD/pitch={r['CD_pitch_frac']:.3f}>=0.85")
    return "; ".join(reasons)


def robust_reason(r: dict) -> str:
    """Why a passing run is not a robust Stage-4 candidate."""
    if not gate_pass(r):
        return "gate fail"
    issues = []
    if r["P_line_margin"] < ROBUST_MARGIN_MIN:
        issues.append(f"P_line_margin={r['P_line_margin']:.3f}<0.05")
    # dCD_shift / darea_frac / dtotal_LER are computed post-baseline.
    if r["dCD_shift_nm"] is None:
        return "; ".join(issues)
    if not (r["dCD_shift_nm"] < 0):
        issues.append(f"dCD_shift={r['dCD_shift_nm']:+.2f} not<0")
    if not (r["darea_frac"] < 0):
        issues.append(f"darea_frac={r['darea_frac']:+.3f} not<0")
    if not (r["dtotal_LER_pp"] >= DTOTAL_LER_MIN_PP):
        issues.append(f"dtotal_LER={r['dtotal_LER_pp']:+.2f}pp<{DTOTAL_LER_MIN_PP}")
    return "; ".join(issues)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, required=True)
    p.add_argument("--tag", type=str, default="04_weak_quencher")
    args = p.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())

    fig_dir_sigma2 = DEFAULT_OUT / "figures" / f"{args.tag}_sigma2"
    fig_dir_summary = DEFAULT_OUT / "figures" / f"{args.tag}_summary"
    logs_dir = DEFAULT_OUT / "logs"
    fig_dir_sigma2.mkdir(parents=True, exist_ok=True)
    fig_dir_summary.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    sigmas = [0.0, 1.0, 2.0, 3.0]
    Q0s_pos = [0.005, 0.01, 0.02, 0.03]
    kqs = [0.5, 1.0, 2.0]

    rows: list[dict] = []
    print(f"=== Stage 4: weak quencher sweep at robust OP (DH=0.5, t=30) ===")

    for sigma in sigmas:
        # No-quencher baseline (per σ).
        baseline = run_one_with_overrides(
            cfg, sigma_nm=sigma, time_s=30.0, DH_nm2_s=0.5,
            quencher_enabled=False, Q0_mol_dm3=0.0, DQ_nm2_s=0.0, kq_s_inv=0.0,
        )
        baseline["sigma_nm_grp"] = sigma
        baseline["row_kind"] = "baseline"
        baseline["Q0_mol_dm3"] = 0.0
        baseline["kq_s_inv"] = 0.0
        rows.append(baseline)

        # Quencher rows.
        for Q0 in Q0s_pos:
            for kq in kqs:
                r = run_one_with_overrides(
                    cfg, sigma_nm=sigma, time_s=30.0, DH_nm2_s=0.5,
                    quencher_enabled=True, Q0_mol_dm3=Q0, DQ_nm2_s=0.0, kq_s_inv=kq,
                )
                r["sigma_nm_grp"] = sigma
                r["row_kind"] = "quencher"
                rows.append(r)

    # Compute deltas vs same-σ baseline.
    by_sigma_baseline = {
        r["sigma_nm_grp"]: r for r in rows if r["row_kind"] == "baseline"
    }
    for r in rows:
        b = by_sigma_baseline[r["sigma_nm_grp"]]
        if r["row_kind"] == "baseline":
            r["dCD_shift_nm"] = 0.0
            r["darea_frac"] = 0.0
            r["dtotal_LER_pp"] = 0.0
        else:
            r["dCD_shift_nm"] = float(r["CD_shift_nm"] - b["CD_shift_nm"]) if np.isfinite(r["CD_shift_nm"]) and np.isfinite(b["CD_shift_nm"]) else None
            r["darea_frac"] = float(r["area_frac"] - b["area_frac"])
            r["dtotal_LER_pp"] = float(r["total_LER_reduction_pct"] - b["total_LER_reduction_pct"]) if np.isfinite(r["total_LER_reduction_pct"]) and np.isfinite(b["total_LER_reduction_pct"]) else None

    for r in rows:
        r["gate_passed"] = gate_pass(r)
        r["fail_reason"] = fail_reason(r)
        r["robust_candidate"] = (
            gate_pass(r)
            and r["P_line_margin"] >= ROBUST_MARGIN_MIN
            and r["dCD_shift_nm"] is not None and r["dCD_shift_nm"] < 0.0
            and r["darea_frac"] < 0.0
            and r["dtotal_LER_pp"] is not None and r["dtotal_LER_pp"] >= DTOTAL_LER_MIN_PP
        )
        r["robust_reason"] = robust_reason(r)

    # Write summary CSV.
    keys = [
        "sigma_nm", "Q0_mol_dm3", "kq_s_inv", "row_kind",
        "P_space_center_mean", "P_line_center_mean", "P_line_margin",
        "contrast", "area_frac", "CD_pitch_frac",
        "CD_initial_nm", "CD_final_nm", "CD_shift_nm",
        "LER_design_initial_nm", "LER_after_eblur_H0_nm", "LER_after_PEB_P_nm",
        "electron_blur_LER_reduction_pct", "PEB_LER_reduction_pct", "total_LER_reduction_pct",
        "psd_design_high", "psd_eblur_high", "psd_PEB_high", "psd_high_band_reduction_pct",
        "dCD_shift_nm", "darea_frac", "dtotal_LER_pp",
        "gate_passed", "robust_candidate", "fail_reason", "robust_reason",
    ]
    with (logs_dir / f"{args.tag}_summary.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow({k: r[k] for k in keys})

    # Pretty table per σ.
    for sigma in sigmas:
        srows = [r for r in rows if r["sigma_nm_grp"] == sigma]
        print(f"\n  σ={sigma:.0f}")
        print("    Q0      kq    P_sp   P_ln    margin  area    CD/p   CDshift   dCD    darea    LERtot%   dLERpp   psdHigh%   gate  robust  reason")
        for r in srows:
            mark = "✓" if r["gate_passed"] else "✗"
            rmark = "★" if r["robust_candidate"] else (" " if r["row_kind"] == "baseline" else "·")
            rsn = r["fail_reason"] if not r["gate_passed"] else (r["robust_reason"] or "OK")
            dCD = r["dCD_shift_nm"] if r["dCD_shift_nm"] is not None else float("nan")
            dLER = r["dtotal_LER_pp"] if r["dtotal_LER_pp"] is not None else float("nan")
            print(
                f"    {r['Q0_mol_dm3']:>5.3f} {r['kq_s_inv']:>5.2f}  "
                f"{r['P_space_center_mean']:>5.3f} {r['P_line_center_mean']:>5.3f}  "
                f"{r['P_line_margin']:>+6.3f} {r['area_frac']:>5.3f}  {r['CD_pitch_frac']:>5.3f}  "
                f"{r['CD_shift_nm']:>+6.2f}  {dCD:>+5.2f}  {r['darea_frac']:>+6.3f}  "
                f"{r['total_LER_reduction_pct']:>+6.2f}   {dLER:>+5.2f}   "
                f"{r['psd_high_band_reduction_pct']:>+6.2f}     {mark}    {rmark}    {rsn}"
            )

    # Save σ=2 contour overlays.
    sigma2_rows = [r for r in rows if r["sigma_nm_grp"] == 2.0]
    for r in sigma2_rows:
        grid = r["_grid"]
        tag = f"sigma_2_Q0_{r['Q0_mol_dm3']}_kq_{r['kq_s_inv']}"
        plot_contour_overlay(
            P_field=r["_P_final"],
            x_nm=grid.x_nm,
            y_nm=grid.y_nm,
            threshold=cfg["development"]["P_threshold"],
            initial_edges=(grid.line_centers_nm,
                           r["_initial_edges"].left_edges_nm,
                           r["_initial_edges"].right_edges_nm),
            final_edges=(grid.line_centers_nm,
                         r["_final_edges"].left_edges_nm,
                         r["_final_edges"].right_edges_nm),
            title=f"σ=2  Q0={r['Q0_mol_dm3']}  kq={r['kq_s_inv']}  margin={r['P_line_margin']:+.3f}",
            out_path=str(fig_dir_sigma2 / f"contour_{tag}.png"),
        )

    # σ=2 Q0×kq heatmaps.
    def heatmap(metric_key: str, fname: str, cmap: str = "viridis"):
        Q0_vals = [0.005, 0.01, 0.02, 0.03]
        kq_vals = [0.5, 1.0, 2.0]
        grid = np.full((len(Q0_vals), len(kq_vals)), np.nan)
        for i, Q0 in enumerate(Q0_vals):
            for j, kq in enumerate(kq_vals):
                rs = [r for r in sigma2_rows if r["row_kind"] == "quencher"
                      and abs(r["Q0_mol_dm3"] - Q0) < 1e-9 and abs(r["kq_s_inv"] - kq) < 1e-9]
                if rs:
                    grid[i, j] = rs[0][metric_key]
        fig, ax = plt.subplots(figsize=(5.0, 4.0))
        im = ax.imshow(grid, origin="lower", aspect="auto", cmap=cmap,
                       extent=[kq_vals[0] - 0.25, kq_vals[-1] + 0.25,
                               Q0_vals[0] - 0.0025, Q0_vals[-1] + 0.0025])
        ax.set_xticks(kq_vals)
        ax.set_yticks(Q0_vals)
        ax.set_xlabel("kq [s⁻¹]")
        ax.set_ylabel("Q0 [mol/dm³]")
        ax.set_title(f"σ=2  {metric_key}")
        for i, Q0 in enumerate(Q0_vals):
            for j, kq in enumerate(kq_vals):
                v = grid[i, j]
                if np.isfinite(v):
                    ax.text(kq, Q0, f"{v:+.2f}", ha="center", va="center",
                            color="white", fontsize=8)
        fig.colorbar(im, ax=ax)
        fig.tight_layout()
        fig.savefig(str(fig_dir_summary / fname), dpi=150)
        plt.close(fig)

    heatmap("dCD_shift_nm", "sigma2_dCD_shift_heatmap.png", cmap="RdBu_r")
    heatmap("darea_frac", "sigma2_darea_frac_heatmap.png", cmap="RdBu_r")
    heatmap("dtotal_LER_pp", "sigma2_dtotal_LER_heatmap.png", cmap="RdBu_r")
    heatmap("P_line_margin", "sigma2_P_line_margin_heatmap.png", cmap="viridis")

    # JSON twin.
    (logs_dir / f"{args.tag}_summary.json").write_text(json.dumps(
        [{k: r[k] for k in keys} for r in rows],
        indent=2,
    ))

    # Recommendation.
    print("\n=== Stage 4 robust candidates (per σ) ===")
    for sigma in sigmas:
        srows = [r for r in rows
                 if r["sigma_nm_grp"] == sigma and r["robust_candidate"]]
        if srows:
            best = max(srows, key=lambda r: -r["dCD_shift_nm"])  # largest CD-shift reduction
            print(f"  σ={sigma:.0f}: {len(srows)} robust candidate(s). "
                  f"Pick (largest |dCD_shift|): Q0={best['Q0_mol_dm3']}, kq={best['kq_s_inv']}, "
                  f"dCD={best['dCD_shift_nm']:+.2f}, darea={best['darea_frac']:+.3f}, "
                  f"dLER={best['dtotal_LER_pp']:+.2f}pp, margin={best['P_line_margin']:.3f}")
        else:
            print(f"  σ={sigma:.0f}: no robust candidate.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

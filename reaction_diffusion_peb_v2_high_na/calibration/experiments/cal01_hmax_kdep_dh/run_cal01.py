"""Calibration Phase 1 — Hmax × kdep × DH sweep at the v2 recommended OP.

Targets (from calibration_targets.yaml):
    CD_locked  ≈ 15.0 nm   (tol 0.5)
    LER_locked ≈ 2.6  nm   (tol 0.3)
    distance_to_target = sqrt(((CD-15)/15)^2 + ((LER-2.6)/2.6)^2)
    pass_threshold = 0.10

Sweep:
    Hmax ∈ {0.15, 0.18, 0.20, 0.22}
    kdep ∈ {0.35, 0.50, 0.65}
    DH   ∈ {0.30, 0.50, 0.80}
=> 4 × 3 × 3 = 36 runs at pitch=24, dose=40, σ=2, t=30, Q0=0.02, kq=1.0.

Each row reports CD-locked metrics + status (Stage-5 classifier) +
distance_to_target. Per-Hmax heatmap of distance + the four core metrics.
Top-N best cells get a contour overlay figure.
"""
from __future__ import annotations

import argparse
import copy
import csv
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from reaction_diffusion_peb_v2_high_na.experiments.run_sigma_sweep_helpers import (  # noqa: E402
    run_one_with_overrides,
)
from reaction_diffusion_peb_v2_high_na.src.visualization import plot_contour_overlay  # noqa: E402

V2_DIR = Path(__file__).resolve().parents[3]
DEFAULT_OUT = V2_DIR / "outputs"

HMAXES = [0.15, 0.18, 0.20, 0.22]
KDEPS = [0.35, 0.50, 0.65]
DHS = [0.30, 0.50, 0.80]

# Stage-5 classification copied locally (the stage 5 module embeds it inline).
def classify(r: dict) -> str:
    if not (r["cond_metrics"] and np.isfinite(r["P_max"]) and np.isfinite(r["P_min"])
            and np.isfinite(r["LER_after_PEB_P_nm"])):
        return "unstable"
    if r["H_min"] < -1e-6 or r["P_min"] < -1e-6 or r["P_max"] > 1.0 + 1e-6:
        return "unstable"
    if (r["P_space_center_mean"] >= 0.50
            or r["area_frac"] >= 0.90
            or (np.isfinite(r["CD_pitch_frac"]) and r["CD_pitch_frac"] >= 0.85)):
        return "merged"
    if r["P_line_center_mean"] < 0.65:
        return "under_exposed"
    if r["contrast"] <= 0.15:
        return "low_contrast"
    if r["P_line_margin"] >= 0.05:
        return "robust_valid"
    return "valid"


def distance_to_target(cd_fixed, ler_locked, cd_target, ler_target):
    """L2 distance using the fixed-threshold CD (= the CD a developed pattern
    actually has, which is what literature/measured CD reports) and the
    CD-locked LER (= intrinsic roughness, free of contour-displacement bias)."""
    if not (np.isfinite(cd_fixed) and np.isfinite(ler_locked)):
        return float("nan")
    d_cd = (cd_fixed - cd_target) / cd_target
    d_ler = (ler_locked - ler_target) / ler_target
    return float(np.sqrt(d_cd * d_cd + d_ler * d_ler))


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, required=True)
    p.add_argument(
        "--targets", type=str,
        default=str(V2_DIR / "calibration" / "calibration_targets.yaml"),
        help="Path to calibration_targets.yaml",
    )
    p.add_argument("--tag", type=str, default="cal01_hmax_kdep_dh")
    args = p.parse_args()

    base_cfg = yaml.safe_load(Path(args.config).read_text())
    targets = yaml.safe_load(Path(args.targets).read_text())
    cd_target = float(targets["targets"]["cd_nm"]["value"])
    ler_target = float(targets["targets"]["ler_nm"]["value"])
    pass_thr = float(targets["scoring"]["distance_to_target"]["pass_threshold"])

    fig_dir = DEFAULT_OUT / "figures" / args.tag
    fig_dir.mkdir(parents=True, exist_ok=True)
    logs_dir = DEFAULT_OUT / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    print(f"=== Calibration Phase 1 [{args.tag}] ===")
    print(f"  targets  CD={cd_target} ± {targets['targets']['cd_nm']['tolerance_nm']} nm,"
          f"   LER={ler_target} ± {targets['targets']['ler_nm']['tolerance_nm']} nm")
    print(f"  pass_threshold = {pass_thr}")
    print()
    for Hmax in HMAXES:
        for kdep in KDEPS:
            for DH in DHS:
                cfg = copy.deepcopy(base_cfg)
                cfg["exposure"]["Hmax_mol_dm3"] = float(Hmax)
                cfg["peb"]["kdep_s_inv"] = float(kdep)
                cfg["peb"]["DH_nm2_s"] = float(DH)
                r = run_one_with_overrides(
                    cfg,
                    sigma_nm=cfg["exposure"]["electron_blur_sigma_nm"],
                    time_s=cfg["peb"]["time_s"],
                    DH_nm2_s=DH, kdep_s_inv=kdep, Hmax_mol_dm3=Hmax,
                    quencher_enabled=cfg["quencher"]["enabled"],
                    Q0_mol_dm3=cfg["quencher"]["Q0_mol_dm3"],
                    DQ_nm2_s=cfg["quencher"]["DQ_nm2_s"],
                    kq_s_inv=cfg["quencher"]["kq_s_inv"],
                )
                r["Hmax"] = float(Hmax)
                r["kdep"] = float(kdep)
                r["DH"] = float(DH)
                r["status"] = classify(r)
                # CD comparison uses fixed-threshold CD (= developed-pattern CD,
                # what measurement data reports). LER comparison uses CD-locked
                # LER (intrinsic roughness, free of displacement bias).
                r["distance_to_target"] = distance_to_target(
                    r["CD_final_nm"], r["LER_CD_locked_nm"], cd_target, ler_target,
                )
                # Selectable: only robust_valid / valid rows can be candidates.
                r["selectable"] = r["status"] in ("robust_valid", "valid")
                rows.append(r)

    print(
        "  Hmax  kdep   DH   status         CD_fix  CD_lock  LER_lock  P_th_lk  margin   area    psd_mid    score   selectable"
    )
    for r in rows:
        d = r["distance_to_target"]
        print(
            f"  {r['Hmax']:>4.2f}  {r['kdep']:>4.2f}  {r['DH']:>4.2f}  "
            f"{r['status']:<13}  "
            f"{r['CD_final_nm']:>5.2f}   {r['CD_locked_nm']:>5.2f}    {r['LER_CD_locked_nm']:>5.2f}    "
            f"{(r['P_threshold_locked'] if r['P_threshold_locked'] is not None else float('nan')):>5.3f}  "
            f"{r['P_line_margin']:>+6.3f}  {r['area_frac']:>5.3f}  "
            f"{r['psd_locked_mid']:>6.3f}     "
            f"{(d if np.isfinite(d) else float('nan')):>6.4f}    "
            f"{'yes' if r['selectable'] else 'no'}"
        )

    # Decision.
    selectable = [r for r in rows if r["selectable"] and np.isfinite(r["distance_to_target"])]
    selectable.sort(key=lambda r: r["distance_to_target"])
    if selectable and selectable[0]["distance_to_target"] < pass_thr:
        gate = "PASS"
    elif selectable and selectable[0]["distance_to_target"] < 0.20:
        gate = "PASS-marginal"
    else:
        gate = "FAIL"

    print(f"\n=== Phase 1 gate: {gate} ===")
    if selectable:
        top_n = min(5, len(selectable))
        print(f"\nTop {top_n} candidates by score (CD = fixed-threshold CD; LER = CD-locked LER):")
        print(f"  {'Hmax':>6} {'kdep':>6} {'DH':>6}  status         CD_fix  CD_lock  LER     score    margin   area   psd_mid")
        for r in selectable[:top_n]:
            print(
                f"  {r['Hmax']:>6.2f} {r['kdep']:>6.2f} {r['DH']:>6.2f}  "
                f"{r['status']:<13}  {r['CD_final_nm']:>5.2f}   {r['CD_locked_nm']:>5.2f}    "
                f"{r['LER_CD_locked_nm']:>5.2f}  "
                f"{r['distance_to_target']:>6.4f}  {r['P_line_margin']:>+6.3f}  "
                f"{r['area_frac']:>5.3f}  {r['psd_locked_mid']:>6.3f}"
            )
    else:
        print("  No selectable (robust_valid / valid) cell with finite metrics.")

    # CSV.
    keys = [
        "Hmax", "kdep", "DH",
        "status", "selectable",
        "CD_locked_nm", "LER_CD_locked_nm", "P_threshold_locked",
        "cd_lock_status",
        "P_line_center_mean", "P_space_center_mean", "P_line_margin",
        "contrast", "area_frac", "CD_pitch_frac",
        "psd_locked_mid", "psd_mid_band_reduction_locked_pct",
        "CD_initial_nm", "CD_final_nm", "CD_shift_nm",
        "LER_design_initial_nm", "LER_after_eblur_H0_nm", "LER_after_PEB_P_nm",
        "total_LER_reduction_pct", "total_LER_reduction_locked_pct",
        "H_peak", "H_min", "P_max", "P_min",
        "distance_to_target",
    ]
    with (logs_dir / f"{args.tag}_summary.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow({k: r[k] for k in keys})

    summary_json = {
        "targets": {"cd_nm": cd_target, "ler_nm": ler_target,
                     "pass_threshold": pass_thr},
        "gate": gate,
        "n_runs": len(rows),
        "n_selectable": len(selectable),
        "best": ({k: selectable[0][k] for k in [
            "Hmax", "kdep", "DH", "status", "CD_locked_nm", "LER_CD_locked_nm",
            "distance_to_target", "P_line_margin", "area_frac", "psd_locked_mid",
        ]} if selectable else None),
    }
    (logs_dir / f"{args.tag}_summary.json").write_text(json.dumps(summary_json, indent=2))

    # Heatmaps per Hmax: distance, CD_locked, LER_locked, margin, area, psd_mid.
    def _heatmap_per_hmax(metric_key: str, title_prefix: str, fname_prefix: str,
                           cmap: str = "viridis", vmin=None, vmax=None,
                           target=None, fmt: str = "{:+.2f}"):
        for Hmax in HMAXES:
            M = np.full((len(DHS), len(KDEPS)), np.nan)
            for r in rows:
                if abs(r["Hmax"] - Hmax) < 1e-9:
                    i = DHS.index(round(r["DH"], 4))
                    j = KDEPS.index(round(r["kdep"], 4))
                    v = r[metric_key]
                    if v is not None and (not isinstance(v, float) or np.isfinite(v)):
                        M[i, j] = float(v)
            fig, ax = plt.subplots(figsize=(5.5, 4.0))
            im = ax.imshow(M, origin="lower", aspect="auto", cmap=cmap,
                           vmin=vmin, vmax=vmax,
                           extent=[-0.5, len(KDEPS) - 0.5, -0.5, len(DHS) - 0.5])
            ax.set_xticks(range(len(KDEPS))); ax.set_xticklabels([str(k) for k in KDEPS])
            ax.set_yticks(range(len(DHS))); ax.set_yticklabels([str(d) for d in DHS])
            ax.set_xlabel("kdep [s⁻¹]")
            ax.set_ylabel("DH [nm²/s]")
            ax.set_title(f"{title_prefix}  Hmax={Hmax}")
            for i in range(len(DHS)):
                for j in range(len(KDEPS)):
                    if np.isfinite(M[i, j]):
                        ax.text(j, i, fmt.format(M[i, j]), ha="center", va="center",
                                color="white", fontsize=8)
            fig.colorbar(im, ax=ax)
            fig.tight_layout()
            fig.savefig(str(fig_dir / f"{fname_prefix}_Hmax_{Hmax:.2f}.png"), dpi=150)
            plt.close(fig)

    _heatmap_per_hmax("distance_to_target", "distance to target",
                       "heatmap_distance", cmap="viridis_r",
                       vmin=0.0, vmax=0.5, fmt="{:.3f}")
    _heatmap_per_hmax("CD_locked_nm", "CD_locked [nm]", "heatmap_CD_locked",
                       cmap="viridis", fmt="{:.2f}")
    _heatmap_per_hmax("LER_CD_locked_nm", "LER_locked [nm]", "heatmap_LER_locked",
                       cmap="viridis", fmt="{:.2f}")
    _heatmap_per_hmax("P_line_margin", "P_line_margin", "heatmap_margin",
                       cmap="viridis", fmt="{:+.3f}")
    _heatmap_per_hmax("area_frac", "area_frac", "heatmap_area",
                       cmap="viridis", vmin=0.0, vmax=1.0, fmt="{:.2f}")
    _heatmap_per_hmax("psd_locked_mid", "psd_mid_band (locked)", "heatmap_psd_mid",
                       cmap="viridis", fmt="{:.2f}")

    # Top-N contour overlays.
    for k, r in enumerate(selectable[:5]):
        grid = r["_grid"]
        plot_contour_overlay(
            P_field=r["_P_final"],
            x_nm=grid.x_nm,
            y_nm=grid.y_nm,
            threshold=base_cfg["development"]["P_threshold"],
            initial_edges=(grid.line_centers_nm,
                           r["_initial_edges"].left_edges_nm,
                           r["_initial_edges"].right_edges_nm),
            final_edges=(grid.line_centers_nm,
                         r["_final_edges"].left_edges_nm,
                         r["_final_edges"].right_edges_nm),
            title=(f"top {k+1}: Hmax={r['Hmax']:.2f} kdep={r['kdep']:.2f} DH={r['DH']:.2f}\n"
                    f"CD_lock={r['CD_locked_nm']:.2f} LER_lock={r['LER_CD_locked_nm']:.2f} "
                    f"score={r['distance_to_target']:.3f}"),
            out_path=str(fig_dir / f"top{k+1}_contour_Hmax_{r['Hmax']:.2f}_kdep_{r['kdep']:.2f}_DH_{r['DH']:.2f}.png"),
        )

    return 0 if gate != "FAIL" else 1


if __name__ == "__main__":
    raise SystemExit(main())

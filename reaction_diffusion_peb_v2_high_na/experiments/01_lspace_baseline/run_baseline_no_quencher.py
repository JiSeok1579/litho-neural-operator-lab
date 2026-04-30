"""Stage 1 — 24 nm pitch / 12.5 nm CD line-space baseline, no quencher.

Run from repo root:
    python -m reaction_diffusion_peb_v2_high_na.experiments.01_lspace_baseline.run_baseline_no_quencher \
        --config reaction_diffusion_peb_v2_high_na/configs/v2_baseline_lspace.yaml
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path

import numpy as np
import yaml

# Allow direct script invocation as well as -m.
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from reaction_diffusion_peb_v2_high_na.src.electron_blur import apply_gaussian_blur
from reaction_diffusion_peb_v2_high_na.src.exposure_high_na import dill_acid_generation, normalize_dose
from reaction_diffusion_peb_v2_high_na.src.fd_solver_2d import solve_peb_2d
from reaction_diffusion_peb_v2_high_na.src.geometry import line_space_intensity
from reaction_diffusion_peb_v2_high_na.src.metrics_edge import extract_edges
from reaction_diffusion_peb_v2_high_na.src.visualization import plot_contour_overlay, plot_field

V2_DIR = Path(__file__).resolve().parents[2]
DEFAULT_OUT = V2_DIR / "outputs"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, required=True)
    p.add_argument("--out_root", type=str, default=str(DEFAULT_OUT))
    p.add_argument("--tag", type=str, default="01_lspace_baseline")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())

    geom = cfg["geometry"]
    exp = cfg["exposure"]
    peb = cfg["peb"]
    qcf = cfg["quencher"]
    dev = cfg["development"]

    seed = cfg["run"].get("seed", 7)

    # 1. Build binary line-space intensity with edge roughness.
    I, grid = line_space_intensity(
        domain_x_nm=geom["domain_x_nm"],
        domain_y_nm=geom["domain_y_nm"],
        grid_spacing_nm=geom["grid_spacing_nm"],
        pitch_nm=geom["pitch_nm"],
        line_cd_nm=geom["line_cd_nm"],
        edge_roughness_amp_nm=(geom["edge_roughness_amp_nm"] if geom.get("edge_roughness_enabled", False) else 0.0),
        edge_roughness_corr_nm=geom.get("edge_roughness_corr_nm", 5.0),
        edge_roughness_seed=seed,
    )

    # 2. Apply electron blur.
    if exp.get("electron_blur_enabled", False):
        I_blurred = apply_gaussian_blur(I, dx_nm=grid.dx_nm, sigma_nm=exp["electron_blur_sigma_nm"])
    else:
        I_blurred = I.copy()

    # 3. Dill acid generation.
    dose_norm = normalize_dose(exp["dose_mJ_cm2"], exp["reference_dose_mJ_cm2"])
    H0 = dill_acid_generation(I_blurred, dose_norm=dose_norm, eta=exp["eta"], Hmax=exp["Hmax_mol_dm3"])

    # 4. Initial-edge metrics: use I_blurred as the "before-PEB" pseudo-P with threshold=0.5
    # of the blurred intensity, which gives a consistent starting edge to compare to PEB output.
    I_thresh = exp.get("initial_threshold", 0.5)
    initial_edges = extract_edges(
        field=I_blurred,
        x_nm=grid.x_nm,
        line_centers_nm=grid.line_centers_nm,
        pitch_nm=grid.pitch_nm,
        threshold=I_thresh,
    )

    # 5. Run PEB.
    res = solve_peb_2d(
        H0=H0,
        dx_nm=grid.dx_nm,
        DH_nm2_s=peb["DH_nm2_s"],
        kdep_s_inv=peb["kdep_s_inv"],
        kloss_s_inv=peb.get("kloss_s_inv", 0.0),
        time_s=peb["time_s"],
        dt_s=peb.get("dt_s", 0.5),
        quencher_enabled=qcf.get("enabled", False),
        Q0=qcf.get("Q0_mol_dm3", 0.0),
        DQ_nm2_s=qcf.get("DQ_nm2_s", 0.0),
        kq_s_inv=qcf.get("kq_s_inv", 0.0),
    )

    # 6. Final-edge metrics on P at threshold.
    P_threshold = dev["P_threshold"]
    final_edges = extract_edges(
        field=res.P,
        x_nm=grid.x_nm,
        line_centers_nm=grid.line_centers_nm,
        pitch_nm=grid.pitch_nm,
        threshold=P_threshold,
    )

    # 7. Bound checks.
    H_min = float(res.H.min())
    H_max = float(res.H.max())
    P_min = float(res.P.min())
    P_max = float(res.P.max())
    Q_min = float(res.Q.min()) if qcf.get("enabled", False) else 0.0
    nan_inf = bool(not np.isfinite(res.H).all() or not np.isfinite(res.P).all() or not np.isfinite(res.Q).all())
    contour_exists = bool(np.any(res.P >= P_threshold))
    area_above = float((res.P >= P_threshold).sum() * (grid.dx_nm ** 2))

    # 8. Persist outputs.
    out_root = Path(args.out_root)
    fig_dir = out_root / "figures" / args.tag
    fields_dir = out_root / "fields" / args.tag
    logs_dir = out_root / "logs"
    fig_dir.mkdir(parents=True, exist_ok=True)
    fields_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    if cfg["outputs"].get("save_fields", True):
        np.savez_compressed(
            fields_dir / "fields.npz",
            x_nm=grid.x_nm,
            y_nm=grid.y_nm,
            line_centers_nm=grid.line_centers_nm,
            I=I,
            I_blurred=I_blurred,
            H0=H0,
            H_final=res.H,
            P_final=res.P,
        )

    if cfg["outputs"].get("save_figures", True):
        plot_field(I_blurred, grid.x_nm, grid.y_nm, "Aerial intensity (after e-blur)",
                   str(fig_dir / "01_lspace_I_blurred.png"), cmap="gray", vmin=0, vmax=1, cbar_label="I")
        plot_field(H0, grid.x_nm, grid.y_nm, f"H0 (Hmax={exp['Hmax_mol_dm3']})",
                   str(fig_dir / "01_lspace_H0.png"), cmap="magma", cbar_label="H [mol/dm^3]")
        plot_field(res.H, grid.x_nm, grid.y_nm, f"H after PEB (t={peb['time_s']:.0f}s)",
                   str(fig_dir / "01_lspace_H_final.png"), cmap="magma", cbar_label="H [mol/dm^3]")
        plot_field(res.P, grid.x_nm, grid.y_nm, "P after PEB",
                   str(fig_dir / "01_lspace_P_final.png"), cmap="viridis", vmin=0, vmax=1, cbar_label="P")
        plot_contour_overlay(
            P_field=res.P,
            x_nm=grid.x_nm,
            y_nm=grid.y_nm,
            threshold=P_threshold,
            initial_edges=(grid.line_centers_nm, initial_edges.left_edges_nm, initial_edges.right_edges_nm),
            final_edges=(grid.line_centers_nm, final_edges.left_edges_nm, final_edges.right_edges_nm),
            title=f"P contour @ {P_threshold:.2f}  (white dashed=initial, black=final)",
            out_path=str(fig_dir / "01_lspace_contour_overlay.png"),
        )

    # 9. Metric row + JSON summary.
    summary = {
        "run_id": cfg["run"]["name"],
        "pitch_nm": grid.pitch_nm,
        "line_cd_nm": grid.line_cd_nm,
        "dose_mJ_cm2": exp["dose_mJ_cm2"],
        "dose_norm": dose_norm,
        "DH_nm2_s": peb["DH_nm2_s"],
        "time_s": peb["time_s"],
        "T_C": peb["temperature_C"],
        "Q0": qcf.get("Q0_mol_dm3", 0.0),
        "kq": qcf.get("kq_s_inv", 0.0),
        "kdep": peb["kdep_s_inv"],
        "kloss": peb.get("kloss_s_inv", 0.0),
        "P_threshold": P_threshold,
        "H_peak": H_max,
        "H_min": H_min,
        "P_max": P_max,
        "P_min": P_min,
        "Q_min": Q_min,
        "P_mean": float(res.P.mean()),
        "area_P_gt_threshold_nm2": area_above,
        "CD_initial_nm": initial_edges.cd_overall_mean_nm,
        "CD_final_nm": final_edges.cd_overall_mean_nm,
        "CD_shift_nm": float(final_edges.cd_overall_mean_nm - initial_edges.cd_overall_mean_nm),
        "LER_initial_nm": initial_edges.ler_mean_nm,
        "LER_final_nm": final_edges.ler_mean_nm,
        "LER_reduction_pct": (
            float(100.0 * (initial_edges.ler_mean_nm - final_edges.ler_mean_nm) / initial_edges.ler_mean_nm)
            if initial_edges.ler_mean_nm > 0
            else float("nan")
        ),
        "LWR_initial_nm": float(np.nanmean(initial_edges.lwr_nm)),
        "LWR_final_nm": float(np.nanmean(final_edges.lwr_nm)),
        "contour_exists": contour_exists,
        "nan_or_inf": nan_inf,
        "status": "ok" if (contour_exists and not nan_inf and H_min > -1e-6 and P_min >= -1e-6 and P_max <= 1.0 + 1e-6) else "fail",
    }

    csv_path = logs_dir / f"{args.tag}.csv"
    fieldnames = list(summary.keys())
    write_header = not csv_path.exists()
    with csv_path.open("a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            w.writeheader()
        w.writerow(summary)

    (logs_dir / f"{args.tag}.json").write_text(json.dumps(summary, indent=2))

    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

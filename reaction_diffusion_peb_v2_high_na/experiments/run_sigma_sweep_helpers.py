"""Shared sweep helpers for Stage-1 baseline tuning + σ=5 calibration."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from reaction_diffusion_peb_v2_high_na.src.electron_blur import apply_gaussian_blur
from reaction_diffusion_peb_v2_high_na.src.exposure_high_na import dill_acid_generation, normalize_dose
from reaction_diffusion_peb_v2_high_na.src.fd_solver_2d import solve_peb_2d
from reaction_diffusion_peb_v2_high_na.src.geometry import line_space_intensity
from reaction_diffusion_peb_v2_high_na.src.metrics_edge import (
    compute_edge_band_powers,
    extract_edges,
    stack_lr_edges,
)
from reaction_diffusion_peb_v2_high_na.src.visualization import plot_contour_overlay, plot_field

# Interior gate thresholds.
P_SPACE_MAX = 0.50
P_LINE_MIN = 0.65
CONTRAST_MIN = 0.15
AREA_FRAC_MAX = 0.90
CD_PITCH_FRAC_MAX = 0.85


def ensure_pitch_aligned_domain(geom: dict) -> dict:
    pitch = float(geom["pitch_nm"])
    domain = float(geom["domain_x_nm"])
    n_periods = int(round(domain / pitch))
    if n_periods < 1:
        n_periods = 1
    aligned = n_periods * pitch
    if abs(aligned - domain) > 1e-9:
        print(f"[domain] adjusting domain_x_nm: {domain} -> {aligned}  ({n_periods} periods of pitch {pitch})")
        geom = dict(geom)
        geom["domain_x_nm"] = aligned
    return geom


def run_one_with_overrides(
    cfg: dict,
    sigma_nm: float,
    time_s: float,
    DH_nm2_s: float | None = None,
    kdep_s_inv: float | None = None,
    Hmax_mol_dm3: float | None = None,
    quencher_enabled: bool | None = None,
    Q0_mol_dm3: float | None = None,
    DQ_nm2_s: float | None = None,
    kq_s_inv: float | None = None,
) -> dict:
    """Single run with per-call overrides; cfg supplies all unspecified params."""
    geom = ensure_pitch_aligned_domain(cfg["geometry"])
    exp = cfg["exposure"]
    peb = cfg["peb"]
    qcf = cfg["quencher"]
    dev = cfg["development"]
    seed = cfg["run"].get("seed", 7)

    Hmax_use = exp["Hmax_mol_dm3"] if Hmax_mol_dm3 is None else float(Hmax_mol_dm3)
    DH_use = peb["DH_nm2_s"] if DH_nm2_s is None else float(DH_nm2_s)
    kdep_use = peb["kdep_s_inv"] if kdep_s_inv is None else float(kdep_s_inv)

    qcf_enabled_use = qcf.get("enabled", False) if quencher_enabled is None else bool(quencher_enabled)
    Q0_use = qcf.get("Q0_mol_dm3", 0.0) if Q0_mol_dm3 is None else float(Q0_mol_dm3)
    DQ_use = qcf.get("DQ_nm2_s", 0.0) if DQ_nm2_s is None else float(DQ_nm2_s)
    kq_use = qcf.get("kq_s_inv", 0.0) if kq_s_inv is None else float(kq_s_inv)

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
    I_blurred = apply_gaussian_blur(I, dx_nm=grid.dx_nm, sigma_nm=sigma_nm) if sigma_nm > 0 else I.copy()
    dose_norm = normalize_dose(exp["dose_mJ_cm2"], exp["reference_dose_mJ_cm2"])
    H0 = dill_acid_generation(I_blurred, dose_norm=dose_norm, eta=exp["eta"], Hmax=Hmax_use)

    init_thr = exp.get("initial_threshold", 0.5)
    # Stage-3 measurement convention: three independent edge measurements.
    # `LER_design_initial`  is sigma-independent (binary I).
    # `LER_after_eblur_H0`  isolates the electron-blur-only smoothing.
    # `LER_after_PEB_P`     isolates the cumulative effect post-PEB.
    design_edges = extract_edges(I, grid.x_nm, grid.line_centers_nm, grid.pitch_nm, init_thr)
    initial_edges = extract_edges(I_blurred, grid.x_nm, grid.line_centers_nm, grid.pitch_nm, init_thr)

    res = solve_peb_2d(
        H0=H0,
        dx_nm=grid.dx_nm,
        DH_nm2_s=DH_use,
        kdep_s_inv=kdep_use,
        kloss_s_inv=peb.get("kloss_s_inv", 0.0),
        time_s=time_s,
        dt_s=peb.get("dt_s", 0.5),
        quencher_enabled=qcf_enabled_use,
        Q0=Q0_use,
        DQ_nm2_s=DQ_use,
        kq_s_inv=kq_use,
    )

    P_threshold = dev["P_threshold"]
    final_edges = extract_edges(res.P, grid.x_nm, grid.line_centers_nm, grid.pitch_nm, P_threshold)

    domain_area = float(grid.x_nm.size * grid.y_nm.size) * (grid.dx_nm ** 2)
    area_above = float((res.P >= P_threshold).sum() * grid.dx_nm ** 2)
    area_frac = area_above / domain_area

    centers = grid.line_centers_nm
    pitch = grid.pitch_nm

    if len(centers) >= 2:
        space_xs = 0.5 * (centers[:-1] + centers[1:])
        space_idxs = [int(np.argmin(np.abs(grid.x_nm - sx))) for sx in space_xs]
        P_space_strips = np.stack([res.P[:, i] for i in space_idxs], axis=0)
        P_space_center_mean = float(P_space_strips.mean())
        P_space_min = float(P_space_strips.min())
    else:
        P_space_center_mean = float(res.P.mean())
        P_space_min = float(res.P.min())

    line_idxs = [int(np.argmin(np.abs(grid.x_nm - c))) for c in centers]
    P_line_strips = np.stack([res.P[:, i] for i in line_idxs], axis=0)
    P_line_center_mean = float(P_line_strips.mean())
    P_line_max = float(P_line_strips.max())
    contrast = P_line_center_mean - P_space_center_mean

    cd_final = final_edges.cd_overall_mean_nm
    ler_final = final_edges.ler_mean_nm
    cd_init = initial_edges.cd_overall_mean_nm
    ler_init = initial_edges.ler_mean_nm
    ler_design = design_edges.ler_mean_nm
    metrics_finite = bool(np.isfinite(cd_final) and np.isfinite(ler_final))

    def _pct(a: float, b: float) -> float:
        if np.isfinite(a) and np.isfinite(b) and a > 0:
            return float(100.0 * (a - b) / a)
        return float("nan")

    eblur_red = _pct(ler_design, ler_init)
    peb_red = _pct(ler_init, ler_final)
    total_red = _pct(ler_design, ler_final)

    # PSD band powers per stage (averaged over left+right tracks of all lines).
    dy_nm = grid.dx_nm  # uniform spacing along y
    bp_design = compute_edge_band_powers(stack_lr_edges(design_edges), dy_nm=dy_nm)
    bp_eblur = compute_edge_band_powers(stack_lr_edges(initial_edges), dy_nm=dy_nm)
    bp_peb = compute_edge_band_powers(stack_lr_edges(final_edges), dy_nm=dy_nm)
    psd_high_red = _pct(bp_design[2], bp_peb[2])  # design -> PEB high-band reduction

    cond_space = P_space_center_mean < P_SPACE_MAX
    cond_line = P_line_center_mean > P_LINE_MIN
    cond_contrast = contrast > CONTRAST_MIN
    cond_area = area_frac < AREA_FRAC_MAX
    cond_cd = bool(np.isfinite(cd_final) and (cd_final < CD_PITCH_FRAC_MAX * pitch))
    passed = bool(cond_space and cond_line and cond_contrast and cond_area and cond_cd and metrics_finite)

    return {
        "sigma_nm": float(sigma_nm),
        "time_s": float(time_s),
        "DH_nm2_s": DH_use,
        "kdep_s_inv": kdep_use,
        "Hmax_mol_dm3": Hmax_use,
        "H_peak": float(res.H.max()),
        "H_min": float(res.H.min()),
        "P_max": float(res.P.max()),
        "P_min": float(res.P.min()),
        "P_space_min": P_space_min,
        "P_space_center_mean": P_space_center_mean,
        "P_line_max": P_line_max,
        "P_line_center_mean": P_line_center_mean,
        "P_line_margin": float(P_line_center_mean - P_LINE_MIN),
        "contrast": contrast,
        "P_mean": float(res.P.mean()),
        "area_P_gt_threshold_nm2": area_above,
        "area_frac": area_frac,
        "CD_initial_nm": cd_init,
        "CD_final_nm": cd_final,
        "CD_shift_nm": float(cd_final - cd_init) if metrics_finite else float("nan"),
        "CD_pitch_frac": float(cd_final / pitch) if metrics_finite else float("nan"),
        # Stage-1/2 backward-compat aliases.
        "LER_initial_nm": ler_init,
        "LER_final_nm": ler_final,
        # Stage-3 explicit names.
        "LER_design_initial_nm": ler_design,
        "LER_after_eblur_H0_nm": ler_init,
        "LER_after_PEB_P_nm": ler_final,
        "electron_blur_LER_reduction_pct": eblur_red,
        "PEB_LER_reduction_pct": peb_red,
        "total_LER_reduction_pct": total_red,
        # PSD band powers per stage (low / mid / high).
        "psd_design_low": float(bp_design[0]),
        "psd_design_mid": float(bp_design[1]),
        "psd_design_high": float(bp_design[2]),
        "psd_eblur_low": float(bp_eblur[0]),
        "psd_eblur_mid": float(bp_eblur[1]),
        "psd_eblur_high": float(bp_eblur[2]),
        "psd_PEB_low": float(bp_peb[0]),
        "psd_PEB_mid": float(bp_peb[1]),
        "psd_PEB_high": float(bp_peb[2]),
        "psd_high_band_reduction_pct": psd_high_red,
        # quencher state for downstream comparison.
        "quencher_enabled": qcf_enabled_use,
        "Q0_mol_dm3": Q0_use,
        "kq_s_inv": kq_use,
        "DQ_nm2_s": DQ_use,
        "cond_space": cond_space,
        "cond_line": cond_line,
        "cond_contrast": cond_contrast,
        "cond_area": cond_area,
        "cond_cd": cond_cd,
        "cond_metrics": metrics_finite,
        "passed": passed,
        "_grid": grid,
        "_design_edges": design_edges,
        "_initial_edges": initial_edges,
        "_final_edges": final_edges,
        "_P_final": res.P,
    }


def save_outputs(rows: list[dict], cfg: dict, fig_dir: Path, csv_path: Path) -> None:
    fig_dir.mkdir(parents=True, exist_ok=True)
    P_threshold = cfg["development"]["P_threshold"]
    for r in rows:
        grid = r["_grid"]
        tag = f"sigma_{r['sigma_nm']:.1f}_t_{r['time_s']:.0f}_DH_{r['DH_nm2_s']}_Hmax_{r['Hmax_mol_dm3']}"
        plot_field(r["_P_final"], grid.x_nm, grid.y_nm,
                   f"P  σ={r['sigma_nm']} nm  t={r['time_s']:.0f} s  DH={r['DH_nm2_s']}  Hmax={r['Hmax_mol_dm3']}",
                   str(fig_dir / f"P_{tag}.png"),
                   cmap="viridis", vmin=0, vmax=1, cbar_label="P")
        plot_contour_overlay(
            P_field=r["_P_final"],
            x_nm=grid.x_nm,
            y_nm=grid.y_nm,
            threshold=P_threshold,
            initial_edges=(grid.line_centers_nm,
                           r["_initial_edges"].left_edges_nm,
                           r["_initial_edges"].right_edges_nm),
            final_edges=(grid.line_centers_nm,
                         r["_final_edges"].left_edges_nm,
                         r["_final_edges"].right_edges_nm),
            title=f"P contour @ {P_threshold:.2f}  σ={r['sigma_nm']} t={r['time_s']:.0f} DH={r['DH_nm2_s']} Hmax={r['Hmax_mol_dm3']}",
            out_path=str(fig_dir / f"contour_{tag}.png"),
        )
    keys = [
        "sigma_nm", "time_s", "DH_nm2_s", "kdep_s_inv", "Hmax_mol_dm3",
        "H_peak", "H_min", "P_max", "P_min",
        "P_space_min", "P_space_center_mean", "P_line_max", "P_line_center_mean",
        "contrast", "P_mean", "area_P_gt_threshold_nm2", "area_frac",
        "CD_initial_nm", "CD_final_nm", "CD_shift_nm", "CD_pitch_frac",
        "LER_initial_nm", "LER_final_nm",
        "cond_space", "cond_line", "cond_contrast", "cond_area", "cond_cd", "cond_metrics",
        "passed",
    ]
    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow({k: r[k] for k in keys})


def print_table(rows: list[dict], extra_keys: tuple[str, ...] = ()) -> None:
    base = ["sigma_nm", "time_s"] + list(extra_keys) + [
        "P_space_center_mean", "P_line_center_mean", "contrast",
        "area_frac", "CD_pitch_frac", "LER_final_nm", "passed",
    ]
    header = " ".join(f"{k:>22}" for k in base)
    print(header)
    print("-" * len(header))
    for r in rows:
        cells = []
        for k in base:
            v = r[k]
            if isinstance(v, float):
                cells.append(f"{v:>22.4f}")
            else:
                cells.append(f"{str(v):>22}")
        print(" ".join(cells))

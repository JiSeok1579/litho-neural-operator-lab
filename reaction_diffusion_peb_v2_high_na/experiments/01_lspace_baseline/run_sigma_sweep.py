"""Stage-1 baseline tuning sweep with seam-artifact-resistant gate.

Acceptance gate (per run):
    P_space_center_mean < 0.50          # mid-line strip stays under threshold
    P_line_center_mean  > 0.65          # line-center strip clears threshold
    contrast = P_line_center_mean - P_space_center_mean > 0.15
    area_frac < 0.90                    # less than 90 % of domain over threshold
    CD_final  < 0.85 * pitch            # lines have not merged
    CD_final and LER_final finite

Phase 1: sigma sweep over {0,1,2,3,4,5} at time_s=60.
        → Pick the LARGEST sigma that passes.
Phase 2 (fallback if all fail): time sweep over {30,45,60} at the BEST sigma
        from Phase 1, where "best" = highest contrast.
        → Pick the SHORTEST time that passes.

Domain must be an integer multiple of pitch_nm to avoid a periodic-seam
artifact at the right edge that otherwise inflates P_min near the boundary.
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

from reaction_diffusion_peb_v2_high_na.src.electron_blur import apply_gaussian_blur
from reaction_diffusion_peb_v2_high_na.src.exposure_high_na import dill_acid_generation, normalize_dose
from reaction_diffusion_peb_v2_high_na.src.fd_solver_2d import solve_peb_2d
from reaction_diffusion_peb_v2_high_na.src.geometry import line_space_intensity
from reaction_diffusion_peb_v2_high_na.src.metrics_edge import extract_edges
from reaction_diffusion_peb_v2_high_na.src.visualization import plot_contour_overlay, plot_field

V2_DIR = Path(__file__).resolve().parents[2]
DEFAULT_OUT = V2_DIR / "outputs"

P_SPACE_MAX = 0.50
P_LINE_MIN = 0.65
CONTRAST_MIN = 0.15
AREA_FRAC_MAX = 0.90
CD_PITCH_FRAC_MAX = 0.85


def _ensure_pitch_aligned_domain(geom: dict) -> dict:
    """Snap domain_x_nm down to nearest integer multiple of pitch_nm; warn if changed."""
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


def run_one(cfg: dict, sigma_nm: float, time_s: float) -> dict:
    geom = _ensure_pitch_aligned_domain(cfg["geometry"])
    exp = cfg["exposure"]
    peb = cfg["peb"]
    qcf = cfg["quencher"]
    dev = cfg["development"]
    seed = cfg["run"].get("seed", 7)

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
    H0 = dill_acid_generation(I_blurred, dose_norm=dose_norm, eta=exp["eta"], Hmax=exp["Hmax_mol_dm3"])

    init_thr = exp.get("initial_threshold", 0.5)
    initial_edges = extract_edges(I_blurred, grid.x_nm, grid.line_centers_nm, grid.pitch_nm, init_thr)

    res = solve_peb_2d(
        H0=H0,
        dx_nm=grid.dx_nm,
        DH_nm2_s=peb["DH_nm2_s"],
        kdep_s_inv=peb["kdep_s_inv"],
        kloss_s_inv=peb.get("kloss_s_inv", 0.0),
        time_s=time_s,
        dt_s=peb.get("dt_s", 0.5),
        quencher_enabled=qcf.get("enabled", False),
        Q0=qcf.get("Q0_mol_dm3", 0.0),
        DQ_nm2_s=qcf.get("DQ_nm2_s", 0.0),
        kq_s_inv=qcf.get("kq_s_inv", 0.0),
    )

    P_threshold = dev["P_threshold"]
    final_edges = extract_edges(res.P, grid.x_nm, grid.line_centers_nm, grid.pitch_nm, P_threshold)

    domain_area = float(grid.x_nm.size * grid.y_nm.size) * (grid.dx_nm ** 2)
    area_above = float((res.P >= P_threshold).sum() * grid.dx_nm ** 2)
    area_frac = area_above / domain_area

    centers = grid.line_centers_nm
    pitch = grid.pitch_nm

    # Mid-line strips (between adjacent line centers). Use interior pairs only.
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
    metrics_finite = bool(np.isfinite(cd_final) and np.isfinite(ler_final))

    cond_space = P_space_center_mean < P_SPACE_MAX
    cond_line = P_line_center_mean > P_LINE_MIN
    cond_contrast = contrast > CONTRAST_MIN
    cond_area = area_frac < AREA_FRAC_MAX
    cond_cd = bool(np.isfinite(cd_final) and (cd_final < CD_PITCH_FRAC_MAX * pitch))
    cond_metrics = metrics_finite

    passed = bool(cond_space and cond_line and cond_contrast and cond_area and cond_cd and cond_metrics)

    return {
        "sigma_nm": float(sigma_nm),
        "time_s": float(time_s),
        "H_peak": float(res.H.max()),
        "H_min": float(res.H.min()),
        "P_max": float(res.P.max()),
        "P_min": float(res.P.min()),
        "P_space_min": P_space_min,
        "P_space_center_mean": P_space_center_mean,
        "P_line_max": P_line_max,
        "P_line_center_mean": P_line_center_mean,
        "contrast": contrast,
        "P_mean": float(res.P.mean()),
        "area_P_gt_threshold_nm2": area_above,
        "area_frac": area_frac,
        "CD_initial_nm": cd_init,
        "CD_final_nm": cd_final,
        "CD_shift_nm": float(cd_final - cd_init) if metrics_finite else float("nan"),
        "CD_pitch_frac": float(cd_final / pitch) if metrics_finite else float("nan"),
        "LER_initial_nm": ler_init,
        "LER_final_nm": ler_final,
        "cond_space": cond_space,
        "cond_line": cond_line,
        "cond_contrast": cond_contrast,
        "cond_area": cond_area,
        "cond_cd": cond_cd,
        "cond_metrics": cond_metrics,
        "passed": passed,
        "_grid": grid,
        "_initial_edges": initial_edges,
        "_final_edges": final_edges,
        "_P_final": res.P,
    }


def _save_outputs(rows: list[dict], cfg: dict, fig_dir: Path, csv_path: Path) -> None:
    fig_dir.mkdir(parents=True, exist_ok=True)
    P_threshold = cfg["development"]["P_threshold"]
    for r in rows:
        grid = r["_grid"]
        tag = f"sigma_{r['sigma_nm']:.1f}_t_{r['time_s']:.0f}"
        plot_field(r["_P_final"], grid.x_nm, grid.y_nm,
                   f"P  sigma={r['sigma_nm']} nm  t={r['time_s']:.0f} s",
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
            title=f"P contour @ {P_threshold:.2f}  sigma={r['sigma_nm']} nm  t={r['time_s']:.0f} s",
            out_path=str(fig_dir / f"contour_{tag}.png"),
        )
    keys = [
        "sigma_nm", "time_s",
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


def _print_table(rows: list[dict]) -> None:
    keys = ["sigma_nm", "time_s", "P_space_center_mean", "P_line_center_mean", "contrast",
            "area_frac", "CD_pitch_frac", "LER_final_nm", "passed"]
    header = " ".join(f"{k:>22}" for k in keys)
    print(header)
    print("-" * len(header))
    for r in rows:
        cells = []
        for k in keys:
            v = r[k]
            if isinstance(v, float):
                cells.append(f"{v:>22.4f}")
            else:
                cells.append(f"{str(v):>22}")
        print(" ".join(cells))


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, required=True)
    p.add_argument("--time", type=float, default=60.0,
                   help="Fixed time_s for Phase-1 sigma sweep")
    p.add_argument("--sigmas", type=str, default="0,1,2,3,4,5")
    p.add_argument("--fallback_times", type=str, default="30,45,60")
    p.add_argument("--tag", type=str, default="01_sigma_sweep")
    return _run(p.parse_args())


def _run(args) -> int:
    cfg = yaml.safe_load(Path(args.config).read_text())

    fig_dir_sigma = DEFAULT_OUT / "figures" / args.tag
    fig_dir_time = DEFAULT_OUT / "figures" / f"{args.tag}_time_fallback"
    logs_dir = DEFAULT_OUT / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    sigmas = [float(s) for s in args.sigmas.split(",")]
    fixed_time = float(args.time)
    fallback_times = [float(t) for t in args.fallback_times.split(",")]

    # ----- Phase 1: sigma sweep at fixed time -----
    print(f"=== Phase 1: sigma sweep (time={fixed_time:.0f}, DH/kdep/quencher per config, pitch-aligned domain) ===")
    rows_sigma = [run_one(cfg, sigma_nm=s, time_s=fixed_time) for s in sigmas]
    _print_table(rows_sigma)
    _save_outputs(rows_sigma, cfg, fig_dir_sigma, logs_dir / f"{args.tag}.csv")

    passed = [r for r in rows_sigma if r["passed"]]
    if passed:
        chosen = max(passed, key=lambda r: r["sigma_nm"])
        summary = {k: chosen[k] for k in [
            "sigma_nm", "time_s", "P_space_center_mean", "P_line_center_mean",
            "contrast", "area_frac",
            "CD_initial_nm", "CD_final_nm", "CD_shift_nm", "CD_pitch_frac",
            "LER_initial_nm", "LER_final_nm",
        ]}
        print(f"\nRECOMMENDED (largest passing sigma): sigma_nm={chosen['sigma_nm']:.1f}, time_s=60.0")
        print(json.dumps(summary, indent=2))
        return 0

    # ----- Phase 2: time sweep at best sigma (highest contrast in Phase 1) -----
    best_sigma = max(rows_sigma, key=lambda r: r["contrast"])["sigma_nm"]
    print(f"\nNo sigma passed acceptance gate. Falling back to time sweep at sigma={best_sigma:.1f} (highest contrast).")
    print(f"=== Phase 2: time sweep (sigma={best_sigma:.1f}, times={fallback_times}) ===")
    rows_time = [run_one(cfg, sigma_nm=best_sigma, time_s=t) for t in fallback_times]
    _print_table(rows_time)
    _save_outputs(rows_time, cfg, fig_dir_time, logs_dir / f"{args.tag}_time_fallback.csv")

    passed_t = [r for r in rows_time if r["passed"]]
    if passed_t:
        chosen = min(passed_t, key=lambda r: r["time_s"])  # shortest passing time
        summary = {k: chosen[k] for k in [
            "sigma_nm", "time_s", "P_space_center_mean", "P_line_center_mean",
            "contrast", "area_frac",
            "CD_initial_nm", "CD_final_nm", "CD_shift_nm", "CD_pitch_frac",
            "LER_initial_nm", "LER_final_nm",
        ]}
        print(f"\nRECOMMENDED (shortest passing time at sigma={chosen['sigma_nm']:.1f}): time_s={chosen['time_s']:.0f}")
        print(json.dumps(summary, indent=2))
        return 0

    print("\nNo (sigma, time) combination passed acceptance gate.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())

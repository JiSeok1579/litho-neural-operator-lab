"""Calibration Phase 2A — OAT sensitivity / controllability sweep.

Anchor (v2 OP):
    pitch=24, line_cd=12.5, dose=40, sigma=2, time=30,
    Hmax=0.20, kdep=0.50, DH=0.50, kloss=0.005,
    quencher Q0=0.02, kq=1.0, DQ=0,
    x-z anchor: thickness=20, amplitude=0.10, abs_len=30, period=6.75.

x-y OAT (single-variable sweeps around anchor):
    dose_mJ_cm2          ∈ {21, 28.4, 40, 44.2, 59, 60}
    electron_blur_sigma  ∈ {0, 1, 2, 3}
    DH_nm2_s             ∈ {0.3, 0.5, 0.8}

x-z OAT:
    dose_mJ_cm2          ∈ {21, 28.4, 40, 44.2, 59, 60}
    electron_blur_sigma  ∈ {0, 1, 2, 3}
    absorption_length_nm ∈ {15, 20, 30, 50, 100}
    DH_nm2_s             ∈ {0.3, 0.5, 0.8}

NOT a calibration to external reference. Reports controllability:
    - per (variable, metric) pair: relative span across the swept range.
    - per variable: the metric most affected.
    - status transitions during each x-y sweep.
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
from reaction_diffusion_peb_v2_high_na.src.exposure_high_na import (  # noqa: E402
    build_xz_intensity,
    dill_acid_generation,
    gaussian_blur_1d,
    line_space_intensity_1d,
    normalize_dose,
)
from reaction_diffusion_peb_v2_high_na.src.fd_solver_xz import solve_peb_xz  # noqa: E402
from reaction_diffusion_peb_v2_high_na.src.metrics_edge import (  # noqa: E402
    CD_LOCK_OK,
    compute_edge_band_powers,
    extract_edges,
    find_cd_lock_threshold,
    stack_lr_edges,
)

V2_DIR = Path(__file__).resolve().parents[3]
DEFAULT_OUT = V2_DIR / "outputs"


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


def run_xy(base_cfg: dict, **overrides) -> dict:
    cfg = copy.deepcopy(base_cfg)
    if "dose_mJ_cm2" in overrides:
        cfg["exposure"]["dose_mJ_cm2"] = float(overrides["dose_mJ_cm2"])
        cfg["exposure"]["dose_norm"] = (
            float(overrides["dose_mJ_cm2"]) / float(cfg["exposure"]["reference_dose_mJ_cm2"])
        )
    if "electron_blur_sigma_nm" in overrides:
        cfg["exposure"]["electron_blur_sigma_nm"] = float(overrides["electron_blur_sigma_nm"])
        cfg["exposure"]["electron_blur_enabled"] = float(overrides["electron_blur_sigma_nm"]) > 0.0
    if "DH_nm2_s" in overrides:
        cfg["peb"]["DH_nm2_s"] = float(overrides["DH_nm2_s"])

    r = run_one_with_overrides(
        cfg,
        sigma_nm=cfg["exposure"]["electron_blur_sigma_nm"],
        time_s=cfg["peb"]["time_s"],
        DH_nm2_s=cfg["peb"]["DH_nm2_s"],
        kdep_s_inv=cfg["peb"]["kdep_s_inv"],
        Hmax_mol_dm3=cfg["exposure"]["Hmax_mol_dm3"],
        quencher_enabled=cfg["quencher"]["enabled"],
        Q0_mol_dm3=cfg["quencher"]["Q0_mol_dm3"],
        DQ_nm2_s=cfg["quencher"]["DQ_nm2_s"],
        kq_s_inv=cfg["quencher"]["kq_s_inv"],
    )
    r["status"] = classify(r)
    return r


def run_xz(
    base_cfg: dict,
    *,
    dose_mJ_cm2: float,
    electron_blur_sigma_nm: float,
    DH_nm2_s: float,
    Hmax_mol_dm3: float,
    kdep_s_inv: float,
    kloss_s_inv: float,
    time_s: float,
    dt_s: float,
    quencher_enabled: bool,
    Q0_mol_dm3: float,
    DQ_nm2_s: float,
    kq_s_inv: float,
    pitch_nm: float,
    line_cd_nm: float,
    grid_spacing_nm: float,
    domain_x_nm: float,
    eta: float,
    reference_dose_mJ_cm2: float,
    film_thickness_nm: float,
    dz_nm: float,
    standing_wave_amplitude: float,
    standing_wave_period_nm: float,
    standing_wave_phase_rad: float,
    absorption_length_nm: float,
    P_threshold: float,
) -> dict:
    """Single x-z run + metrics."""
    I_x_binary, x_nm, line_centers_nm = line_space_intensity_1d(
        domain_x_nm=domain_x_nm, grid_spacing_nm=grid_spacing_nm,
        pitch_nm=pitch_nm, line_cd_nm=line_cd_nm,
    )
    I_x = gaussian_blur_1d(I_x_binary, dx_nm=grid_spacing_nm,
                            sigma_nm=electron_blur_sigma_nm)
    n_z = int(round(film_thickness_nm / dz_nm)) + 1
    z_nm = np.arange(n_z) * dz_nm
    I_xz = build_xz_intensity(
        I_x=I_x, z_nm=z_nm,
        standing_wave_period_nm=standing_wave_period_nm,
        standing_wave_amplitude=standing_wave_amplitude,
        standing_wave_phase_rad=standing_wave_phase_rad,
        absorption_length_nm=absorption_length_nm,
    )
    dose_norm = normalize_dose(dose_mJ_cm2, reference_dose_mJ_cm2)
    H0 = dill_acid_generation(I_xz, dose_norm=dose_norm, eta=eta, Hmax=Hmax_mol_dm3)
    res = solve_peb_xz(
        H0=H0, dx_nm=grid_spacing_nm, dz_nm=dz_nm,
        DH_nm2_s=DH_nm2_s, kdep_s_inv=kdep_s_inv, kloss_s_inv=kloss_s_inv,
        time_s=time_s, dt_s=dt_s,
        quencher_enabled=quencher_enabled, Q0=Q0_mol_dm3,
        DQ_nm2_s=DQ_nm2_s, kq_s_inv=kq_s_inv,
    )

    ix_line = int(np.argmin(np.abs(x_nm - line_centers_nm[len(line_centers_nm) // 2])))
    def _mod_pct(field):
        s = field[:, ix_line]
        m = float(s.mean())
        if abs(m) < 1e-12:
            return 0.0
        return float(100.0 * (s.max() - s.min()) / m)
    H0_zmod = _mod_pct(H0)
    Pf_zmod = _mod_pct(res.P)
    if H0_zmod > 1e-9:
        mod_red = float(100.0 * (H0_zmod - Pf_zmod) / H0_zmod)
    else:
        mod_red = float("nan")
    P_top = float(res.P[-1, ix_line])
    P_bot = float(res.P[0, ix_line])
    asym = float(abs(P_top - P_bot) / max(abs(P_top), abs(P_bot), 1e-12))

    edges_fixed = extract_edges(res.P, x_nm, line_centers_nm, pitch_nm, P_threshold)
    P_locked, cd_locked, cd_lock_status = find_cd_lock_threshold(
        res.P, x_nm=x_nm, line_centers_nm=line_centers_nm, pitch_nm=pitch_nm,
        cd_target_nm=line_cd_nm,
    )
    if cd_lock_status == CD_LOCK_OK and P_locked is not None:
        edges_locked = extract_edges(res.P, x_nm, line_centers_nm, pitch_nm, P_locked)
        ler_locked = float(edges_locked.ler_mean_nm)
        bp_locked = compute_edge_band_powers(stack_lr_edges(edges_locked), dy_nm=dz_nm)
        psd_mid_locked = float(bp_locked[1])
    else:
        ler_locked = float("nan")
        psd_mid_locked = float("nan")

    return {
        "dose_mJ_cm2": float(dose_mJ_cm2),
        "sigma_nm": float(electron_blur_sigma_nm),
        "DH_nm2_s": float(DH_nm2_s),
        "absorption_length_nm": float(absorption_length_nm),
        "film_thickness_nm": float(film_thickness_nm),
        "amplitude": float(standing_wave_amplitude),
        "H0_z_modulation_pct": H0_zmod,
        "P_final_z_modulation_pct": Pf_zmod,
        "modulation_reduction_pct": mod_red,
        "top_bottom_asymmetry": asym,
        "LER_fixed_threshold_nm": float(edges_fixed.ler_mean_nm),
        "LER_CD_locked_nm": ler_locked,
        "psd_mid_locked": psd_mid_locked,
        "P_threshold_locked": P_locked,
        "CD_locked_nm": cd_locked,
        "cd_lock_status": cd_lock_status,
        "H_min": float(res.H.min()),
        "P_min": float(res.P.min()),
        "P_max": float(res.P.max()),
    }


def _line_plot(xs, ys_dict, xlabel, title, out_path, ylabel="value"):
    fig, ax = plt.subplots(figsize=(6.0, 4.0))
    for name, ys in ys_dict.items():
        ys_safe = [v if v is not None and (not isinstance(v, float) or np.isfinite(v)) else np.nan for v in ys]
        ax.plot(xs, ys_safe, marker="o", label=name)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, loc="best")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _relative_span_pct(values, anchor) -> float:
    finite = [v for v in values if v is not None and (not isinstance(v, float) or np.isfinite(v))]
    if not finite or anchor is None or not np.isfinite(anchor) or abs(anchor) < 1e-12:
        return float("nan")
    return float(100.0 * (max(finite) - min(finite)) / abs(anchor))


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, required=True)
    p.add_argument("--tag", type=str, default="cal02a_sensitivity")
    args = p.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    sweeps = cfg["sweeps"]
    fig_dir = DEFAULT_OUT / "figures" / args.tag
    fig_dir.mkdir(parents=True, exist_ok=True)
    logs_dir = DEFAULT_OUT / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    # Common x-z parameters from anchor.
    xz_anchor = {
        "Hmax_mol_dm3": cfg["exposure"]["Hmax_mol_dm3"],
        "kdep_s_inv": cfg["peb"]["kdep_s_inv"],
        "kloss_s_inv": cfg["peb"]["kloss_s_inv"],
        "time_s": cfg["peb"]["time_s"],
        "dt_s": cfg["peb"]["dt_s"],
        "quencher_enabled": cfg["quencher"]["enabled"],
        "Q0_mol_dm3": cfg["quencher"]["Q0_mol_dm3"],
        "DQ_nm2_s": cfg["quencher"]["DQ_nm2_s"],
        "kq_s_inv": cfg["quencher"]["kq_s_inv"],
        "pitch_nm": cfg["geometry"]["pitch_nm"],
        "line_cd_nm": cfg["geometry"]["line_cd_nm"],
        "grid_spacing_nm": cfg["geometry"]["grid_spacing_nm"],
        "domain_x_nm": cfg["geometry"]["domain_x_nm"],
        "eta": cfg["exposure"]["eta"],
        "reference_dose_mJ_cm2": cfg["exposure"]["reference_dose_mJ_cm2"],
        "film_thickness_nm": cfg["film"]["thickness_anchor_nm"],
        "dz_nm": cfg["film"]["dz_nm"],
        "standing_wave_amplitude": cfg["standing_wave"]["amplitude_anchor"],
        "standing_wave_period_nm": cfg["standing_wave"]["period_nm"],
        "standing_wave_phase_rad": cfg["standing_wave"]["phase_rad"],
        "absorption_length_nm": cfg["standing_wave"]["absorption_length_anchor_nm"],
        "P_threshold": cfg["development"]["P_threshold"],
        # anchor values for variables we'll override per sweep
        "dose_mJ_cm2": cfg["exposure"]["dose_mJ_cm2"],
        "electron_blur_sigma_nm": cfg["exposure"]["electron_blur_sigma_nm"],
        "DH_nm2_s": cfg["peb"]["DH_nm2_s"],
    }

    # Anchor reference rows.
    anchor_xy = run_xy(cfg)
    anchor_xz = run_xz(cfg, **xz_anchor)

    print("\n=== Calibration Phase 2A — sensitivity / controllability ===\n")
    print(
        "anchor (xy): "
        f"CD_fix={anchor_xy['CD_final_nm']:.3f}  "
        f"LER_lock={anchor_xy['LER_CD_locked_nm']:.3f}  "
        f"margin={anchor_xy['P_line_margin']:+.3f}  area={anchor_xy['area_frac']:.3f}  "
        f"status={anchor_xy['status']}"
    )
    print(
        "anchor (xz): "
        f"H0_zmod={anchor_xz['H0_z_modulation_pct']:.2f}%  "
        f"P_zmod={anchor_xz['P_final_z_modulation_pct']:.2f}%  "
        f"mod_red={anchor_xz['modulation_reduction_pct']:.2f}%  "
        f"asym={anchor_xz['top_bottom_asymmetry']:.3f}  "
        f"LER_lock(z)={anchor_xz['LER_CD_locked_nm']:.3f}"
    )

    # ---- x-y OAT sweeps ----
    xy_metric_keys = ["CD_final_nm", "LER_CD_locked_nm",
                       "P_line_margin", "area_frac", "contrast"]
    xy_results = {var: [] for var in sweeps["xy"]}
    sensitivity_xy = []  # (variable, metric, span_pct)
    for var, values in sweeps["xy"].items():
        print(f"\nx-y sweep over {var} = {values}")
        for v in values:
            r = run_xy(cfg, **{var: v})
            xy_results[var].append({var: v, **{k: r[k] for k in xy_metric_keys},
                                       "status": r["status"]})
            print(f"  {var}={v}  CD_fix={r['CD_final_nm']:.3f}  "
                  f"LER_lock={r['LER_CD_locked_nm']:.3f}  "
                  f"margin={r['P_line_margin']:+.3f}  area={r['area_frac']:.3f}  "
                  f"status={r['status']}")
        for metric in xy_metric_keys:
            anchor_val = anchor_xy[metric]
            span = _relative_span_pct([row[metric] for row in xy_results[var]], anchor_val)
            sensitivity_xy.append((var, metric, span))

    # ---- x-z OAT sweeps ----
    xz_metric_keys = ["H0_z_modulation_pct", "P_final_z_modulation_pct",
                       "modulation_reduction_pct", "top_bottom_asymmetry",
                       "LER_CD_locked_nm", "psd_mid_locked"]
    xz_results = {var: [] for var in sweeps["xz"]}
    sensitivity_xz = []
    for var, values in sweeps["xz"].items():
        print(f"\nx-z sweep over {var} = {values}")
        for v in values:
            kwargs = dict(xz_anchor)
            # NOTE: variable name mapping (x-z runner uses electron_blur_sigma_nm)
            if var == "electron_blur_sigma_nm":
                kwargs["electron_blur_sigma_nm"] = float(v)
            else:
                kwargs[var] = float(v)
            r = run_xz(cfg, **kwargs)
            xz_results[var].append({var: v, **{k: r[k] for k in xz_metric_keys}})
            print(f"  {var}={v}  H0_zmod={r['H0_z_modulation_pct']:.2f}%  "
                  f"P_zmod={r['P_final_z_modulation_pct']:.2f}%  "
                  f"mod_red={r['modulation_reduction_pct']:.2f}%  "
                  f"asym={r['top_bottom_asymmetry']:.3f}  "
                  f"LER_lock(z)={r['LER_CD_locked_nm']:.3f}  "
                  f"PSD_mid={r['psd_mid_locked']:.3f}")
        for metric in xz_metric_keys:
            anchor_val = anchor_xz[metric]
            span = _relative_span_pct([row[metric] for row in xz_results[var]], anchor_val)
            sensitivity_xz.append((var, metric, span))

    # ---- per-variable line plots ----
    for var in sweeps["xy"]:
        rows = xy_results[var]
        xs = [row[var] for row in rows]
        _line_plot(
            xs,
            {m: [row[m] for row in rows] for m in xy_metric_keys},
            xlabel=var, ylabel="metric value",
            title=f"x-y sensitivity vs {var}",
            out_path=fig_dir / f"xy_sensitivity_{var}.png",
        )
    for var in sweeps["xz"]:
        rows = xz_results[var]
        xs = [row[var] for row in rows]
        _line_plot(
            xs,
            {m: [row[m] for row in rows] for m in xz_metric_keys},
            xlabel=var, ylabel="metric value",
            title=f"x-z sensitivity vs {var}",
            out_path=fig_dir / f"xz_sensitivity_{var}.png",
        )

    # Sensitivity summary table.
    print("\n=== Sensitivity (relative span pct vs anchor) ===")
    print("  domain  variable                  metric                       span_pct")
    for var, metric, span in sensitivity_xy:
        print(f"  xy      {var:<24}  {metric:<28}  {span:>8.2f}")
    for var, metric, span in sensitivity_xz:
        print(f"  xz      {var:<24}  {metric:<28}  {span:>8.2f}")

    # CSV.
    sens_path = logs_dir / f"{args.tag}_sensitivity.csv"
    with sens_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["domain", "variable", "metric", "anchor_value", "span_pct"])
        for var, metric, span in sensitivity_xy:
            w.writerow(["xy", var, metric, anchor_xy[metric], span])
        for var, metric, span in sensitivity_xz:
            w.writerow(["xz", var, metric, anchor_xz[metric], span])

    rows_path = logs_dir / f"{args.tag}_rows.csv"
    with rows_path.open("w", newline="") as f:
        # union of column keys.
        all_keys = ["domain", "variable", "value"] + list(set(
            list(xy_metric_keys) + list(xz_metric_keys) + ["status"]
        ))
        w = csv.DictWriter(f, fieldnames=all_keys)
        w.writeheader()
        for var in sweeps["xy"]:
            for row in xy_results[var]:
                out = {"domain": "xy", "variable": var, "value": row[var]}
                for k in xy_metric_keys + ["status"]:
                    out[k] = row.get(k)
                w.writerow(out)
        for var in sweeps["xz"]:
            for row in xz_results[var]:
                out = {"domain": "xz", "variable": var, "value": row[var]}
                for k in xz_metric_keys:
                    out[k] = row.get(k)
                w.writerow(out)

    summary = {
        "anchor_xy": {k: anchor_xy[k] for k in xy_metric_keys + ["status"]},
        "anchor_xz": {k: anchor_xz[k] for k in xz_metric_keys},
        "sensitivity_xy": [{"variable": v, "metric": m, "span_pct": s}
                           for v, m, s in sensitivity_xy],
        "sensitivity_xz": [{"variable": v, "metric": m, "span_pct": s}
                           for v, m, s in sensitivity_xz],
    }
    (logs_dir / f"{args.tag}_summary.json").write_text(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

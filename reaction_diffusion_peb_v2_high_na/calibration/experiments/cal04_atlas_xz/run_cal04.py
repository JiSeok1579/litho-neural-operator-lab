"""Phase 2B Part B — x-z sensitivity atlas (4D grid).

NOT a calibration. Sensitivity / controllability study at the frozen v2 OP.

Grid:
    film_thickness_nm   ∈ {15, 20, 30}
    standing_wave_amp   ∈ {0.0, 0.05, 0.10, 0.20}
    absorption_length   ∈ {15, 30, 60, 100}
    DH_nm2_s            ∈ {0.3, 0.5, 0.8}
=> 3 × 4 × 4 × 3 = 144 runs.

Per cell:
    H0_z_modulation_pct, H0_z_modulation_sw_only_pct (vs same-thickness A=0 baseline
                                                       at same DH and abs_len)
    P_final_z_modulation_pct
    modulation_reduction_pct
    top_bottom_asymmetry
    sidewall_x_displacement_std (= LER_CD_locked over z-tracks)
    psd_mid_band (locked)
    bounds + cd_lock_status
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

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

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


def run_xz_cell(cfg: dict, *, thickness: float, amplitude: float,
                 abs_len: float, DH: float) -> dict:
    geom = cfg["geometry"]
    exp = cfg["exposure"]
    peb = cfg["peb"]
    qcf = cfg["quencher"]
    sw = cfg["standing_wave"]
    film = cfg["film"]
    dev = cfg["development"]

    dx_nm = float(geom["grid_spacing_nm"])
    dz_nm = float(film["dz_nm"])
    pitch_nm = float(geom["pitch_nm"])
    line_cd_nm = float(geom["line_cd_nm"])
    domain_x_nm = float(geom["domain_x_nm"])

    I_binary, x_nm, line_centers_nm = line_space_intensity_1d(
        domain_x_nm=domain_x_nm, grid_spacing_nm=dx_nm,
        pitch_nm=pitch_nm, line_cd_nm=line_cd_nm,
    )
    sigma = float(exp["electron_blur_sigma_nm"]) if exp["electron_blur_enabled"] else 0.0
    I_x = gaussian_blur_1d(I_binary, dx_nm=dx_nm, sigma_nm=sigma)

    n_z = int(round(thickness / dz_nm)) + 1
    z_nm = np.arange(n_z) * dz_nm
    I_xz = build_xz_intensity(
        I_x=I_x, z_nm=z_nm,
        standing_wave_period_nm=float(sw["period_nm"]),
        standing_wave_amplitude=float(amplitude),
        standing_wave_phase_rad=float(sw["phase_rad"]),
        absorption_length_nm=float(abs_len),
    )
    dose_norm = normalize_dose(exp["dose_mJ_cm2"], exp["reference_dose_mJ_cm2"])
    H0 = dill_acid_generation(I_xz, dose_norm=dose_norm,
                                eta=float(exp["eta"]),
                                Hmax=float(exp["Hmax_mol_dm3"]))

    res = solve_peb_xz(
        H0=H0, dx_nm=dx_nm, dz_nm=dz_nm,
        DH_nm2_s=float(DH), kdep_s_inv=float(peb["kdep_s_inv"]),
        kloss_s_inv=float(peb.get("kloss_s_inv", 0.0)),
        time_s=float(peb["time_s"]),
        dt_s=float(peb.get("dt_s", 0.5)),
        quencher_enabled=bool(qcf["enabled"]),
        Q0=float(qcf["Q0_mol_dm3"]),
        DQ_nm2_s=float(qcf["DQ_nm2_s"]),
        kq_s_inv=float(qcf["kq_s_inv"]),
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
    P_top = float(res.P[-1, ix_line]); P_bot = float(res.P[0, ix_line])
    asym = float(abs(P_top - P_bot) / max(abs(P_top), abs(P_bot), 1e-12))

    P_threshold = float(dev["P_threshold"])
    edges_fixed = extract_edges(res.P, x_nm, line_centers_nm, pitch_nm, P_threshold)
    P_locked, cd_locked, cd_lock_status = find_cd_lock_threshold(
        res.P, x_nm=x_nm, line_centers_nm=line_centers_nm, pitch_nm=pitch_nm,
        cd_target_nm=line_cd_nm,
    )
    if cd_lock_status == CD_LOCK_OK and P_locked is not None:
        edges_locked = extract_edges(res.P, x_nm, line_centers_nm, pitch_nm, P_locked)
        sidewall_std = float(edges_locked.ler_mean_nm)
        bp_locked = compute_edge_band_powers(stack_lr_edges(edges_locked), dy_nm=dz_nm)
        psd_mid_locked = float(bp_locked[1])
    else:
        sidewall_std = float("nan")
        psd_mid_locked = float("nan")

    return {
        "thickness_nm": float(thickness),
        "amplitude": float(amplitude),
        "absorption_length_nm": float(abs_len),
        "DH_nm2_s": float(DH),
        "H0_z_modulation_pct": H0_zmod,
        "P_final_z_modulation_pct": Pf_zmod,
        "modulation_reduction_pct": mod_red,
        "top_bottom_asymmetry": asym,
        "sidewall_x_displacement_std_nm": sidewall_std,
        "psd_mid_band_locked": psd_mid_locked,
        "P_threshold_locked": P_locked,
        "CD_locked_nm": cd_locked,
        "cd_lock_status": cd_lock_status,
        "H_min": float(res.H.min()),
        "P_min": float(res.P.min()),
        "P_max": float(res.P.max()),
    }


def line_plot(xs, ys_dict, xlabel, title, out_path):
    fig, ax = plt.subplots(figsize=(6.5, 4.0))
    for name, ys in ys_dict.items():
        clean = [v if v is not None and (not isinstance(v, float) or np.isfinite(v)) else np.nan for v in ys]
        ax.plot(xs, clean, marker="o", label=name)
    ax.set_xlabel(xlabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, loc="best")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def heatmap_2d(M, x_vals, y_vals, xlabel, ylabel, title, out_path,
                cmap="viridis", vmin=None, vmax=None, fmt="{:.2f}"):
    fig, ax = plt.subplots(figsize=(5.5, 4.0))
    im = ax.imshow(M, origin="lower", aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax,
                   extent=[-0.5, len(x_vals) - 0.5, -0.5, len(y_vals) - 0.5])
    ax.set_xticks(range(len(x_vals))); ax.set_xticklabels([str(v) for v in x_vals])
    ax.set_yticks(range(len(y_vals))); ax.set_yticklabels([str(v) for v in y_vals])
    ax.set_xlabel(xlabel); ax.set_ylabel(ylabel); ax.set_title(title)
    for i in range(len(y_vals)):
        for j in range(len(x_vals)):
            v = M[i, j]
            if np.isfinite(v):
                ax.text(j, i, fmt.format(v), ha="center", va="center", color="white", fontsize=7)
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, required=True)
    p.add_argument("--tag", type=str, default="cal04_atlas_xz")
    args = p.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    fig_dir = DEFAULT_OUT / "figures" / args.tag
    fig_dir.mkdir(parents=True, exist_ok=True)
    logs_dir = DEFAULT_OUT / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    thicknesses = [float(t) for t in cfg["film"]["thickness_sweep_nm"]]
    amplitudes = [float(a) for a in cfg["standing_wave"]["amplitude_sweep"]]
    abs_lens = [float(L) for L in cfg["standing_wave"]["absorption_length_sweep"]]
    DHs = [float(d) for d in cfg["dh_sweep_nm2_s"]]

    print(f"=== Phase 2B Part B — x-z sensitivity atlas ===")
    print(f"  thickness ∈ {thicknesses}, A ∈ {amplitudes}, abs_len ∈ {abs_lens}, DH ∈ {DHs}")
    print(f"  total cells = {len(thicknesses)*len(amplitudes)*len(abs_lens)*len(DHs)}")

    rows: list[dict] = []
    for h in thicknesses:
        for A in amplitudes:
            for L in abs_lens:
                for D in DHs:
                    r = run_xz_cell(cfg, thickness=h, amplitude=A, abs_len=L, DH=D)
                    rows.append(r)

    # Standing-wave-only modulation: subtract same (thickness, abs_len, DH) A=0 baseline.
    base_lookup = {(r["thickness_nm"], r["absorption_length_nm"], r["DH_nm2_s"]):
                       r["H0_z_modulation_pct"]
                   for r in rows if r["amplitude"] == 0.0}
    for r in rows:
        key = (r["thickness_nm"], r["absorption_length_nm"], r["DH_nm2_s"])
        base = base_lookup.get(key, 0.0)
        r["H0_z_modulation_sw_only_pct"] = r["H0_z_modulation_pct"] - base

    # Bounds gate.
    n_pass = 0
    for r in rows:
        bounds_ok = (r["H_min"] >= -1e-6 and r["P_min"] >= -1e-6
                      and r["P_max"] <= 1.0 + 1e-6
                      and np.isfinite(r["P_max"]) and np.isfinite(r["P_min"]))
        r["bounds_ok"] = bool(bounds_ok)
        if bounds_ok:
            n_pass += 1
    print(f"\nbounds_ok: {n_pass}/{len(rows)}")

    # Marginal 1D plots: for each variable, plot mean ± span across other 3 vars.
    metrics = ["H0_z_modulation_sw_only_pct", "P_final_z_modulation_pct",
                "modulation_reduction_pct", "top_bottom_asymmetry",
                "sidewall_x_displacement_std_nm", "psd_mid_band_locked"]

    def _marginal_plot(var_key, var_values, var_label, out_path):
        ys_dict = {}
        for m in metrics:
            ys = []
            for v in var_values:
                vals = [r[m] for r in rows
                        if abs(r[var_key] - v) < 1e-9
                        and r[m] is not None
                        and (not isinstance(r[m], float) or np.isfinite(r[m]))]
                ys.append(float(np.mean(vals)) if vals else np.nan)
            ys_dict[m] = ys
        line_plot(var_values, ys_dict, var_label,
                   f"x-z atlas mean over {var_label}", out_path)

    _marginal_plot("thickness_nm", thicknesses, "thickness [nm]",
                    fig_dir / "marginal_thickness.png")
    _marginal_plot("amplitude", amplitudes, "amplitude",
                    fig_dir / "marginal_amplitude.png")
    _marginal_plot("absorption_length_nm", abs_lens, "absorption_length [nm]",
                    fig_dir / "marginal_absorption_length.png")
    _marginal_plot("DH_nm2_s", DHs, "DH [nm²/s]",
                    fig_dir / "marginal_DH.png")

    # 2D heatmaps thick × A at each (abs_len, DH) for headline metrics.
    for L in abs_lens:
        for D in DHs:
            for metric, fname_prefix, cmap, vmin, vmax, fmt in [
                ("modulation_reduction_pct", "modred", "RdBu_r", 0, 100, "{:.0f}"),
                ("sidewall_x_displacement_std_nm", "sidewall", "viridis", None, None, "{:.2f}"),
                ("top_bottom_asymmetry", "asym", "viridis", 0, 0.5, "{:.2f}"),
                ("psd_mid_band_locked", "psdmid", "viridis", None, None, "{:.2f}"),
            ]:
                M = np.full((len(thicknesses), len(amplitudes)), np.nan)
                for i, h in enumerate(thicknesses):
                    for j, A in enumerate(amplitudes):
                        cand = [r for r in rows
                                if abs(r["thickness_nm"] - h) < 1e-9
                                and abs(r["amplitude"] - A) < 1e-9
                                and abs(r["absorption_length_nm"] - L) < 1e-9
                                and abs(r["DH_nm2_s"] - D) < 1e-9]
                        if cand:
                            v = cand[0][metric]
                            if v is not None and (not isinstance(v, float) or np.isfinite(v)):
                                M[i, j] = float(v)
                heatmap_2d(M, amplitudes, thicknesses, "amplitude", "thickness [nm]",
                            f"{metric}  abs_len={L}  DH={D}",
                            fig_dir / f"thickA_{fname_prefix}_abs{int(L)}_DH{D}.png",
                            cmap=cmap, vmin=vmin, vmax=vmax, fmt=fmt)

    # Per-(variable, metric) sensitivity coefficient.
    sensitivity_rows = []
    anchor = next((r for r in rows
                    if abs(r["thickness_nm"] - 20.0) < 1e-9
                    and abs(r["amplitude"] - 0.10) < 1e-9
                    and abs(r["absorption_length_nm"] - 30.0) < 1e-9
                    and abs(r["DH_nm2_s"] - 0.5) < 1e-9), None)
    if anchor is not None:
        for var_key, var_values in [
            ("thickness_nm", thicknesses),
            ("amplitude", amplitudes),
            ("absorption_length_nm", abs_lens),
            ("DH_nm2_s", DHs),
        ]:
            for m in metrics:
                values = [r[m] for r in rows
                          if r[m] is not None and
                          (not isinstance(r[m], float) or np.isfinite(r[m]))]
                anchor_val = anchor[m]
                if not values or anchor_val is None or not np.isfinite(anchor_val) or abs(anchor_val) < 1e-12:
                    span_pct = float("nan")
                else:
                    span_pct = float(100.0 * (max(values) - min(values)) / abs(anchor_val))
                sensitivity_rows.append((var_key, m, span_pct))

    # CSV.
    keys = [
        "thickness_nm", "amplitude", "absorption_length_nm", "DH_nm2_s",
        "H0_z_modulation_pct", "H0_z_modulation_sw_only_pct",
        "P_final_z_modulation_pct", "modulation_reduction_pct",
        "top_bottom_asymmetry",
        "sidewall_x_displacement_std_nm", "psd_mid_band_locked",
        "P_threshold_locked", "CD_locked_nm", "cd_lock_status",
        "H_min", "P_min", "P_max", "bounds_ok",
    ]
    with (logs_dir / f"{args.tag}_summary.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow({k: r[k] for k in keys})

    summary_json = {"n_runs": len(rows),
                    "anchor_at_20_0p10_30_0p5": anchor and {k: anchor[k] for k in keys},
                    "sensitivity_full_grid": [
                        {"variable": v, "metric": m, "span_pct_full_grid": s}
                        for v, m, s in sensitivity_rows]}
    (logs_dir / f"{args.tag}_summary.json").write_text(json.dumps(summary_json, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

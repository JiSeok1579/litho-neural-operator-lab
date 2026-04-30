"""Stage 6 — x-z standing wave PEB sweep at the pitch=24 robust OP.

Operating point (fixed):
    pitch=24, line_cd=12.5, sigma=2 (e-blur), DH=0.5, time=30, kdep=0.5,
    Hmax=0.2, kloss=0.005, quencher Q0=0.02, kq=1.0, DQ=0.

Sweep:
    film_thickness_nm    ∈ {15, 20, 30}
    standing_wave_amp    ∈ {0.0, 0.05, 0.10, 0.20}
    period_nm            = 6.75   (z-modulation wavelength)
    absorption_length_nm = 30
    top / bottom BC      : no-flux (Neumann) — handled by even-mirror FFT in z

Initial exposure (per row):
    I(x, z) = I_x(x) * (1 + A*cos(2*pi*z/period + phase)) * exp(-z/abs_len)

Reported per row:
    H0_z_modulation_pct, H_final_z_modulation_pct, P_final_z_modulation_pct
    modulation_reduction_pct (H0 -> P_final)
    top_bottom_asymmetry
    CD_locked_LER (extract_edges on P(x, z) treating z as the edge-track axis;
        each "track" is the line-edge x-position at a given z, so this LER
        captures sidewall x-displacement along z)
    PSD_mid_band on those z-tracks
    bounds: H_min, P_min, P_max
    mass_budget_drift_pct (relative change of trapezoidal x-z integral of H
                           between H0 and H_final, used as solver QC)

Acceptance gate (per row):
    no NaN / no Inf
    H_min >= -1e-6, P_min >= -1e-6, P_max <= 1+1e-6
    A == 0  → H0_z_modulation_pct < 1 %
    A > 0   → H0_z_modulation_pct increases with A
    P_final_z_modulation_pct < H0_z_modulation_pct (PEB reduces modulation)
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

V2_DIR = Path(__file__).resolve().parents[2]
DEFAULT_OUT = V2_DIR / "outputs"


def _trapz_xz_sum(field: np.ndarray) -> float:
    return float(field[1:-1].sum() + 0.5 * (field[0].sum() + field[-1].sum()))


def relative_modulation_pct(field_zx: np.ndarray, ix_line: int) -> float:
    """Peak-to-peak / mean (%) of z-profile at a given line-center column."""
    z_strip = field_zx[:, ix_line]
    mean_v = float(z_strip.mean())
    if abs(mean_v) < 1e-12:
        return 0.0
    return float(100.0 * (z_strip.max() - z_strip.min()) / mean_v)


def edges_along_z(field_zx: np.ndarray, x_nm: np.ndarray, line_centers_nm: np.ndarray,
                  pitch_nm: float, threshold: float):
    """Re-use extract_edges with z as the iterated 'y-axis'. Returns EdgeResult."""
    return extract_edges(field_zx, x_nm=x_nm, line_centers_nm=line_centers_nm,
                          pitch_nm=pitch_nm, threshold=threshold)


def run_one(cfg: dict, film_thickness_nm: float, amplitude: float) -> dict:
    geom = cfg["geometry"]
    exp = cfg["exposure"]
    peb = cfg["peb"]
    qcf = cfg["quencher"]
    dev = cfg["development"]
    sw = cfg["standing_wave"]
    film = cfg["film"]

    grid_spacing_nm = float(geom["grid_spacing_nm"])
    dz_nm = float(film["dz_nm"])
    domain_x_nm = float(geom["domain_x_nm"])
    pitch_nm = float(geom["pitch_nm"])
    line_cd_nm = float(geom["line_cd_nm"])

    # 1D x-axis line-space + electron blur (no y-roughness in Stage 6).
    I_x_binary, x_nm, line_centers_nm = line_space_intensity_1d(
        domain_x_nm=domain_x_nm, grid_spacing_nm=grid_spacing_nm,
        pitch_nm=pitch_nm, line_cd_nm=line_cd_nm,
    )
    sigma_eblur = float(exp["electron_blur_sigma_nm"]) if exp["electron_blur_enabled"] else 0.0
    I_x = gaussian_blur_1d(I_x_binary, dx_nm=grid_spacing_nm, sigma_nm=sigma_eblur)

    # z grid: 0 .. film_thickness, inclusive.
    n_z = int(round(film_thickness_nm / dz_nm)) + 1
    z_nm = np.arange(n_z) * dz_nm

    I_xz = build_xz_intensity(
        I_x=I_x, z_nm=z_nm,
        standing_wave_period_nm=float(sw["period_nm"]),
        standing_wave_amplitude=float(amplitude),
        standing_wave_phase_rad=float(sw["phase_rad"]),
        absorption_length_nm=float(sw["absorption_length_nm"]) if sw["absorption_enabled"] else None,
    )

    dose_norm = normalize_dose(exp["dose_mJ_cm2"], exp["reference_dose_mJ_cm2"])
    H0 = dill_acid_generation(I_xz, dose_norm=dose_norm, eta=float(exp["eta"]),
                               Hmax=float(exp["Hmax_mol_dm3"]))

    res = solve_peb_xz(
        H0=H0, dx_nm=grid_spacing_nm, dz_nm=dz_nm,
        DH_nm2_s=float(peb["DH_nm2_s"]),
        kdep_s_inv=float(peb["kdep_s_inv"]),
        kloss_s_inv=float(peb.get("kloss_s_inv", 0.0)),
        time_s=float(peb["time_s"]),
        dt_s=float(peb.get("dt_s", 0.5)),
        quencher_enabled=bool(qcf["enabled"]),
        Q0=float(qcf["Q0_mol_dm3"]),
        DQ_nm2_s=float(qcf["DQ_nm2_s"]),
        kq_s_inv=float(qcf["kq_s_inv"]),
    )

    # z-modulation at line-center column (whichever interior line is closest to mid-x)
    ix_line = int(np.argmin(np.abs(x_nm - line_centers_nm[len(line_centers_nm) // 2])))
    H0_zmod = relative_modulation_pct(H0, ix_line)
    Hf_zmod = relative_modulation_pct(res.H, ix_line)
    Pf_zmod = relative_modulation_pct(res.P, ix_line)
    if H0_zmod > 1e-9:
        modulation_reduction_pct = float(100.0 * (H0_zmod - Pf_zmod) / H0_zmod)
    else:
        modulation_reduction_pct = float("nan")

    # Top/bottom asymmetry of P at line-center column.
    P_top = float(res.P[-1, ix_line])
    P_bot = float(res.P[0, ix_line])
    denom = max(abs(P_top), abs(P_bot), 1e-12)
    top_bottom_asymmetry = float(abs(P_top - P_bot) / denom)

    # CD-locked LER on (x,z) treating z as the edge-track axis.
    P_threshold = float(dev["P_threshold"])
    edges_fixed = edges_along_z(res.P, x_nm, line_centers_nm, pitch_nm, P_threshold)
    cd_target = float(line_cd_nm)
    P_locked, cd_locked, cd_lock_status = find_cd_lock_threshold(
        res.P, x_nm=x_nm, line_centers_nm=line_centers_nm, pitch_nm=pitch_nm,
        cd_target_nm=cd_target,
    )
    if cd_lock_status == CD_LOCK_OK and P_locked is not None:
        edges_locked = extract_edges(res.P, x_nm, line_centers_nm, pitch_nm, P_locked)
        ler_locked = float(edges_locked.ler_mean_nm)
        bp_locked = compute_edge_band_powers(stack_lr_edges(edges_locked), dy_nm=dz_nm)
        psd_mid_locked = float(bp_locked[1])
    else:
        ler_locked = float("nan")
        psd_mid_locked = float("nan")
        edges_locked = None

    # PSD mid-band of fixed-threshold edges.
    bp_fixed = compute_edge_band_powers(stack_lr_edges(edges_fixed), dy_nm=dz_nm)
    psd_mid_fixed = float(bp_fixed[1])

    # Bounds + mass budget QC.
    H_min = float(res.H.min())
    P_min = float(res.P.min())
    P_max = float(res.P.max())
    nan_inf = bool(not np.isfinite(res.H).all() or not np.isfinite(res.P).all())
    bounds_ok = (H_min >= -1e-6) and (P_min >= -1e-6) and (P_max <= 1.0 + 1e-6) and (not nan_inf)

    H0_trap = _trapz_xz_sum(H0)
    Hf_trap = _trapz_xz_sum(res.H)
    if abs(H0_trap) > 1e-12:
        mass_budget_drift_pct = float(100.0 * (Hf_trap - H0_trap) / H0_trap)
    else:
        mass_budget_drift_pct = float("nan")

    return {
        "film_thickness_nm": float(film_thickness_nm),
        "amplitude": float(amplitude),
        "n_z": int(n_z),
        "ix_line": int(ix_line),
        "H0_z_modulation_pct": H0_zmod,
        "H_final_z_modulation_pct": Hf_zmod,
        "P_final_z_modulation_pct": Pf_zmod,
        "modulation_reduction_pct": modulation_reduction_pct,
        "top_bottom_asymmetry": top_bottom_asymmetry,
        "CD_initial_nm": cd_target,
        "CD_locked_nm": cd_locked,
        "P_threshold_locked": P_locked,
        "cd_lock_status": cd_lock_status,
        "LER_fixed_threshold_nm": float(edges_fixed.ler_mean_nm),
        "LER_CD_locked_nm": ler_locked,
        "psd_mid_fixed": psd_mid_fixed,
        "psd_mid_locked": psd_mid_locked,
        "H_min": H_min,
        "P_min": P_min,
        "P_max": P_max,
        "nan_or_inf": nan_inf,
        "bounds_ok": bool(bounds_ok),
        "mass_budget_drift_pct": mass_budget_drift_pct,
        "_x_nm": x_nm,
        "_z_nm": z_nm,
        "_I_xz": I_xz,
        "_H0": H0,
        "_H_final": res.H,
        "_P_final": res.P,
        "_line_centers_nm": line_centers_nm,
    }


def plot_xz(field, x_nm, z_nm, title, out_path, cmap="viridis", vmin=None, vmax=None,
            cbar_label="", contour_threshold: float | None = None,
            second_threshold: float | None = None):
    fig, ax = plt.subplots(figsize=(6.0, 4.0))
    im = ax.imshow(
        field, origin="lower",
        extent=[x_nm[0], x_nm[-1], z_nm[0], z_nm[-1]],
        cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto",
    )
    if contour_threshold is not None:
        ax.contour(x_nm, z_nm, field, levels=[contour_threshold],
                   colors="red", linewidths=1.2)
    if second_threshold is not None:
        ax.contour(x_nm, z_nm, field, levels=[second_threshold],
                   colors="white", linewidths=1.0, linestyles="--")
    ax.set_xlabel("x [nm]")
    ax.set_ylabel("z [nm]")
    ax.set_title(title)
    cb = fig.colorbar(im, ax=ax)
    cb.set_label(cbar_label)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, required=True)
    p.add_argument("--tag", type=str, default="06_xz_standing_wave")
    args = p.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())

    fig_dir = DEFAULT_OUT / "figures" / args.tag
    fig_dir.mkdir(parents=True, exist_ok=True)
    logs_dir = DEFAULT_OUT / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    thicknesses = [float(t) for t in cfg["film"]["thickness_sweep_nm"]]
    amplitudes = [float(a) for a in cfg["standing_wave"]["amplitude_sweep"]]

    rows: list[dict] = []
    print(f"=== Stage 6: x-z standing wave sweep ===")
    print(f"  thickness ∈ {thicknesses} nm, amplitude ∈ {amplitudes}")
    print(f"  period={cfg['standing_wave']['period_nm']}, "
          f"absorption_length={cfg['standing_wave']['absorption_length_nm']}, "
          f"OP=DH={cfg['peb']['DH_nm2_s']} t={cfg['peb']['time_s']} "
          f"sigma={cfg['exposure']['electron_blur_sigma_nm']} "
          f"Q0={cfg['quencher']['Q0_mol_dm3']} kq={cfg['quencher']['kq_s_inv']}\n")

    for h in thicknesses:
        for A in amplitudes:
            r = run_one(cfg, film_thickness_nm=h, amplitude=A)
            rows.append(r)
            print(
                f"  thick={h:>4.0f}  A={A:>4.2f}  "
                f"zmod H0={r['H0_z_modulation_pct']:>6.2f}%  Hf={r['H_final_z_modulation_pct']:>6.2f}%  "
                f"Pf={r['P_final_z_modulation_pct']:>6.2f}%  red={r['modulation_reduction_pct']:>+7.2f}%  "
                f"asym={r['top_bottom_asymmetry']:>5.3f}  "
                f"CD_lock={r['CD_locked_nm']:>5.2f}  LER_lock={r['LER_CD_locked_nm']:>5.2f}  "
                f"PSDmid_lock={r['psd_mid_locked']:>5.2f}  "
                f"H_min={r['H_min']:>+.2e}  P_min={r['P_min']:>+.2e}  P_max={r['P_max']:>.4f}  "
                f"mass_drift={r['mass_budget_drift_pct']:>+5.2f}%  bounds={'ok' if r['bounds_ok'] else 'FAIL'}"
            )

    # Per-thickness "standing-wave-only" excess: subtract A=0 baseline so the
    # exponential-absorption envelope (which makes H0 already z-varying at A=0)
    # is removed from the standing-wave gate.
    baseline_by_thickness = {
        r["film_thickness_nm"]: r["H0_z_modulation_pct"]
        for r in rows if r["amplitude"] == 0.0
    }
    for r in rows:
        base = baseline_by_thickness.get(r["film_thickness_nm"], 0.0)
        r["H0_z_modulation_sw_only_pct"] = r["H0_z_modulation_pct"] - base

    # Per-row gates.
    def row_gate(r: dict) -> tuple[bool, str]:
        if not r["bounds_ok"]:
            return False, "bounds fail"
        # Standing-wave-only modulation must be near zero at A=0.
        if r["amplitude"] == 0.0 and abs(r["H0_z_modulation_sw_only_pct"]) >= 1.0:
            return False, f"A=0 but sw_only={r['H0_z_modulation_sw_only_pct']:.2f}%"
        return True, "ok"

    print("\nGate per row (in addition to A-monotonicity / PEB-reduction across rows):")
    for r in rows:
        ok, why = row_gate(r)
        r["row_gate_ok"] = ok
        r["row_gate_reason"] = why
        print(f"  thick={r['film_thickness_nm']:>4.0f} A={r['amplitude']:.2f}  "
              f"H0_zmod_total={r['H0_z_modulation_pct']:>6.2f}%  "
              f"H0_zmod_sw_only={r['H0_z_modulation_sw_only_pct']:>+6.2f}%  → "
              f"{'PASS' if ok else 'FAIL'}  ({why})")

    # Cross-row gates.
    print("\nCross-row gates:")
    cross_ok = True
    for h in thicknesses:
        ks = [r for r in rows if r["film_thickness_nm"] == h]
        ks.sort(key=lambda r: r["amplitude"])
        h_zmods = [r["H0_z_modulation_pct"] for r in ks]
        # Monotone increase with amplitude
        monotone = all(h_zmods[i] <= h_zmods[i + 1] + 1e-6 for i in range(len(h_zmods) - 1))
        # PEB reduces modulation in every nonzero-A row
        peb_reduces = all((r["amplitude"] == 0.0) or (r["P_final_z_modulation_pct"] < r["H0_z_modulation_pct"])
                          for r in ks)
        print(f"  thickness={h:>4.0f}  monotone-A: {'OK' if monotone else 'FAIL'}, "
              f"PEB-reduces-zmod: {'OK' if peb_reduces else 'FAIL'}")
        if not (monotone and peb_reduces):
            cross_ok = False

    print(f"\nOverall Stage 6 cross-row gate: {'OK' if cross_ok else 'FAIL'}")

    # Figures: I, H0, P_final for every (thickness, amplitude) cell.
    for r in rows:
        h = r["film_thickness_nm"]
        A = r["amplitude"]
        tag = f"thick_{int(h)}_A_{A:.2f}"
        plot_xz(r["_I_xz"], r["_x_nm"], r["_z_nm"],
                f"I(x,z)  thick={int(h)} A={A:.2f}",
                str(fig_dir / f"I_{tag}.png"), cmap="gray", vmin=0, cbar_label="I")
        plot_xz(r["_H0"], r["_x_nm"], r["_z_nm"],
                f"H0(x,z)  thick={int(h)} A={A:.2f}",
                str(fig_dir / f"H0_{tag}.png"), cmap="magma", cbar_label="H [mol/dm³]")
        plot_xz(r["_P_final"], r["_x_nm"], r["_z_nm"],
                f"P_final(x,z)  thick={int(h)} A={A:.2f}",
                str(fig_dir / f"P_{tag}.png"), cmap="viridis", vmin=0, vmax=1,
                cbar_label="P", contour_threshold=cfg["development"]["P_threshold"],
                second_threshold=r["P_threshold_locked"]
                                  if r["P_threshold_locked"] is not None and np.isfinite(r["P_threshold_locked"])
                                  else None)

    # Cross-row summary plots vs thickness.
    summary_dir = fig_dir / "summary"
    summary_dir.mkdir(parents=True, exist_ok=True)

    def _summary_plot(metric_key: str, ylabel: str, title: str, out_name: str,
                      ylog: bool = False) -> None:
        fig, ax = plt.subplots(figsize=(6.5, 4.0))
        for A in amplitudes:
            xs = []
            ys = []
            for h in thicknesses:
                cand = [r for r in rows
                        if abs(r["film_thickness_nm"] - h) < 1e-9
                        and abs(r["amplitude"] - A) < 1e-9]
                if cand:
                    val = cand[0][metric_key]
                    if val is not None and (not isinstance(val, float) or np.isfinite(val)):
                        xs.append(h)
                        ys.append(val)
            ax.plot(xs, ys, marker="o", label=f"A={A:.2f}")
        ax.set_xlabel("film thickness [nm]")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        if ylog:
            ax.set_yscale("log")
        ax.grid(True, alpha=0.3)
        ax.legend()
        fig.tight_layout()
        fig.savefig(str(summary_dir / out_name), dpi=150)
        plt.close(fig)

    _summary_plot("modulation_reduction_pct",
                   "PEB modulation reduction (%)",
                   "Stage 6 — modulation reduction vs thickness",
                   "modulation_reduction_vs_thickness.png")
    _summary_plot("LER_CD_locked_nm",
                   "side-wall LER (CD-locked) [nm]",
                   "Stage 6 — side-wall LER vs thickness",
                   "sidewall_ler_vs_thickness.png")
    _summary_plot("top_bottom_asymmetry",
                   "|P(top) - P(bottom)| / max",
                   "Stage 6 — top/bottom asymmetry vs thickness",
                   "top_bottom_asymmetry_vs_thickness.png")
    _summary_plot("H0_z_modulation_sw_only_pct",
                   "H0 standing-wave-only modulation (%)",
                   "Stage 6 — H0 standing-wave modulation vs thickness (excl. absorption)",
                   "H0_sw_only_vs_thickness.png")

    # CSV.
    keys = [
        "film_thickness_nm", "amplitude", "n_z", "ix_line",
        "H0_z_modulation_pct", "H0_z_modulation_sw_only_pct",
        "H_final_z_modulation_pct", "P_final_z_modulation_pct",
        "modulation_reduction_pct", "top_bottom_asymmetry",
        "CD_initial_nm", "CD_locked_nm", "P_threshold_locked", "cd_lock_status",
        "LER_fixed_threshold_nm", "LER_CD_locked_nm",
        "psd_mid_fixed", "psd_mid_locked",
        "H_min", "P_min", "P_max", "nan_or_inf", "bounds_ok",
        "mass_budget_drift_pct",
        "row_gate_ok", "row_gate_reason",
    ]
    with (logs_dir / f"{args.tag}_summary.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow({k: r[k] for k in keys})
    (logs_dir / f"{args.tag}_summary.json").write_text(json.dumps(
        [{k: r[k] for k in keys} for r in rows], indent=2,
    ))
    return 0 if cross_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())

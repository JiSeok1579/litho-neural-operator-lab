"""Render single-line x-z cross-sections for representative configurations.

For each entry in `configs/xz_companions.yaml`:
  1. build the 1D line-space + electron-blur intensity I_x(x),
  2. extend it into (x, z) with the standing-wave / absorption envelope,
  3. run the x-z PEB solver,
  4. crop the resulting fields to a single line — middle line ± pitch/2 — and
     save four side-view (x-z) figures: I, H0, H_final, P_final (with the
     fixed-threshold and CD-locked contours overlaid on P).

This is a presentation-only renderer. It does not change any sweep result;
it just gives a depth-profile view of one resist line under each chemistry.
"""
from __future__ import annotations

import argparse
import csv
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
    extract_edges,
    find_cd_lock_threshold,
)

V2_DIR = Path(__file__).resolve().parents[2]
DEFAULT_OUT = V2_DIR / "outputs"


def _crop_around_center(field: np.ndarray, x_nm: np.ndarray,
                        center_nm: float, half_window_nm: float):
    """Return (cropped field, cropped x). Only crops along x (axis=1)."""
    mask = (x_nm >= center_nm - half_window_nm) & (x_nm <= center_nm + half_window_nm)
    return field[:, mask], x_nm[mask]


def _xz_panel(field, x_nm, z_nm, *, title, out_path, cmap="viridis",
               vmin=None, vmax=None, cbar_label="",
               red_threshold=None, white_threshold=None,
               line_cd_nm=None, center_x_nm=None):
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    im = ax.imshow(
        field, origin="lower",
        extent=[x_nm[0], x_nm[-1], z_nm[0], z_nm[-1]],
        cmap=cmap, vmin=vmin, vmax=vmax, aspect="equal",
    )
    if red_threshold is not None:
        ax.contour(x_nm, z_nm, field, levels=[red_threshold],
                   colors="red", linewidths=1.2)
    if white_threshold is not None:
        ax.contour(x_nm, z_nm, field, levels=[white_threshold],
                   colors="white", linewidths=1.0, linestyles="--")
    if line_cd_nm is not None and center_x_nm is not None:
        ax.axvline(center_x_nm - line_cd_nm / 2.0, color="cyan", linewidth=0.6,
                   linestyle=":", alpha=0.8)
        ax.axvline(center_x_nm + line_cd_nm / 2.0, color="cyan", linewidth=0.6,
                   linestyle=":", alpha=0.8)
    ax.set_xlabel("x [nm]")
    ax.set_ylabel("z [nm]")
    ax.set_title(title)
    cb = fig.colorbar(im, ax=ax)
    cb.set_label(cbar_label)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def render_case(case: dict, defaults: dict, out_root: Path):
    tag = case["tag"]
    desc = case.get("description", tag)

    pitch = float(case["pitch_nm"])
    line_cd = float(case["line_cd_nm"])
    domain_x = float(case["domain_x_nm"])
    dx = float(case["grid_spacing_nm"])
    sigma = float(case["sigma_nm"])
    dose_mJ = float(case["dose_mJ_cm2"])
    ref_dose = float(case["reference_dose_mJ_cm2"])
    eta = float(case["eta"])
    Hmax = float(case["Hmax_mol_dm3"])
    DH = float(case["DH_nm2_s"])
    time_s = float(case["time_s"])
    kdep = float(case["kdep_s_inv"])
    kloss = float(case["kloss_s_inv"])
    qen = bool(case["quencher_enabled"])
    Q0 = float(case["Q0_mol_dm3"])
    DQ = float(case["DQ_nm2_s"])
    kq = float(case["kq_s_inv"])

    thickness = float(case.get("film_thickness_nm", defaults["film_thickness_nm"]))
    dz = float(case.get("dz_nm", defaults["dz_nm"]))
    period = float(case.get("standing_wave_period_nm", defaults["standing_wave_period_nm"]))
    amplitude = float(case.get("standing_wave_amplitude", defaults["standing_wave_amplitude"]))
    phase = float(case.get("standing_wave_phase_rad", defaults["standing_wave_phase_rad"]))
    abs_len = float(case.get("absorption_length_nm", defaults["absorption_length_nm"]))
    P_threshold = float(case.get("P_threshold", defaults["P_threshold"]))

    # 1D x intensity + blur
    I_binary, x_nm, line_centers_nm = line_space_intensity_1d(
        domain_x_nm=domain_x, grid_spacing_nm=dx,
        pitch_nm=pitch, line_cd_nm=line_cd,
    )
    I_x = gaussian_blur_1d(I_binary, dx_nm=dx, sigma_nm=sigma)

    # x-z exposure
    n_z = int(round(thickness / dz)) + 1
    z_nm = np.arange(n_z) * dz
    I_xz = build_xz_intensity(
        I_x=I_x, z_nm=z_nm,
        standing_wave_period_nm=period,
        standing_wave_amplitude=amplitude,
        standing_wave_phase_rad=phase,
        absorption_length_nm=abs_len,
    )

    dose_norm = normalize_dose(dose_mJ, ref_dose)
    H0 = dill_acid_generation(I_xz, dose_norm=dose_norm, eta=eta, Hmax=Hmax)

    res = solve_peb_xz(
        H0=H0, dx_nm=dx, dz_nm=dz,
        DH_nm2_s=DH, kdep_s_inv=kdep, kloss_s_inv=kloss,
        time_s=time_s, dt_s=0.5,
        quencher_enabled=qen, Q0=Q0, DQ_nm2_s=DQ, kq_s_inv=kq,
    )

    # Pick the middle line as the "single line" to crop around.
    if len(line_centers_nm) >= 1:
        center_x = float(line_centers_nm[len(line_centers_nm) // 2])
    else:
        center_x = domain_x / 2.0
    half_win = pitch / 2.0

    # CD-lock for white contour reference.
    P_locked, cd_locked, cd_lock_status = find_cd_lock_threshold(
        res.P, x_nm=x_nm, line_centers_nm=line_centers_nm, pitch_nm=pitch,
        cd_target_nm=line_cd,
    )
    locked_threshold = P_locked if cd_lock_status == CD_LOCK_OK else None

    out_dir = out_root / tag
    out_dir.mkdir(parents=True, exist_ok=True)

    # Crop fields around the middle line.
    I_c, x_c = _crop_around_center(I_xz, x_nm, center_x, half_win)
    H0_c, _ = _crop_around_center(H0, x_nm, center_x, half_win)
    Hf_c, _ = _crop_around_center(res.H, x_nm, center_x, half_win)
    Pf_c, _ = _crop_around_center(res.P, x_nm, center_x, half_win)

    common = dict(
        line_cd_nm=line_cd, center_x_nm=center_x,
    )

    _xz_panel(I_c, x_c, z_nm,
              title=f"{tag}\nI(x,z) cropped around line x={center_x:.0f} nm",
              out_path=out_dir / "I_xz.png",
              cmap="gray", vmin=0.0, cbar_label="I", **common)
    _xz_panel(H0_c, x_c, z_nm,
              title=f"{tag}\nH0(x,z)  Hmax={Hmax}",
              out_path=out_dir / "H0_xz.png",
              cmap="magma", cbar_label="H [mol/dm³]", **common)
    _xz_panel(Hf_c, x_c, z_nm,
              title=f"{tag}\nH(x,z) after PEB t={time_s:.0f}s",
              out_path=out_dir / "H_final_xz.png",
              cmap="magma", cbar_label="H [mol/dm³]", **common)
    _xz_panel(Pf_c, x_c, z_nm,
              title=(f"{tag}\nP(x,z)  red=fixed P=0.5  white=locked "
                     f"P={locked_threshold:.3f}" if locked_threshold is not None
                     else f"{tag}\nP(x,z)  red=fixed P=0.5  (locked unstable)"),
              out_path=out_dir / "P_final_xz.png",
              cmap="viridis", vmin=0.0, vmax=1.0,
              cbar_label="P",
              red_threshold=P_threshold,
              white_threshold=locked_threshold,
              **common)

    # CSV summary row (one per tag).
    summary = {
        "tag": tag,
        "description": desc,
        "pitch_nm": pitch,
        "line_cd_nm": line_cd,
        "dose_mJ_cm2": dose_mJ,
        "sigma_nm": sigma,
        "DH_nm2_s": DH,
        "time_s": time_s,
        "Hmax_mol_dm3": Hmax,
        "kdep_s_inv": kdep,
        "Q0_mol_dm3": Q0,
        "kq_s_inv": kq,
        "thickness_nm": thickness,
        "amplitude": amplitude,
        "absorption_length_nm": abs_len,
        "P_threshold_locked": P_locked,
        "CD_locked_nm": cd_locked,
        "cd_lock_status": cd_lock_status,
        "H_min": float(res.H.min()),
        "P_min": float(res.P.min()),
        "P_max": float(res.P.max()),
        "P_top_at_line_center": float(res.P[-1, np.argmin(np.abs(x_nm - center_x))]),
        "P_bottom_at_line_center": float(res.P[0, np.argmin(np.abs(x_nm - center_x))]),
    }
    return summary


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, required=True)
    p.add_argument("--tag", type=str, default="xz_companions")
    args = p.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    defaults = cfg["defaults"]
    out_root = DEFAULT_OUT / "figures" / args.tag
    out_root.mkdir(parents=True, exist_ok=True)
    logs_dir = DEFAULT_OUT / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    summaries = []
    print(f"=== xz companions — {len(cfg['cases'])} representative cases ===")
    for case in cfg["cases"]:
        print(f"  rendering: {case['tag']:<32}  ({case.get('description', '')})")
        s = render_case(case, defaults, out_root)
        summaries.append(s)

    # CSV summary.
    keys = list(summaries[0].keys())
    csv_path = logs_dir / f"{args.tag}_summary.csv"
    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for s in summaries:
            w.writerow(s)
    print(f"\nWrote {len(summaries)} entries → {csv_path}")
    print(f"figures        → {out_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

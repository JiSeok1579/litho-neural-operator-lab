"""PEB Phase-11 demo: molecular-blur post-processing of P_final.

Run:
    python reaction_diffusion_peb/experiments/11_advanced/run_molecular_blur.py

Sweeps ``sigma_nm in {0, 0.5, 1.0, 2.0}`` for the post-FD Gaussian
blur on the final ``P`` field. ``sigma_nm = 0`` returns the
deterministic Phase-8 baseline; positive ``sigma_nm`` models the
finite resist particle size / molecular-scale smoothing.

Reports per σ_blur:
  - max / mean P after blur
  - threshold area before / after blur
  - mean shift in threshold area
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import matplotlib.pyplot as plt

from reaction_diffusion_peb.src.deprotection import thresholded_area
from reaction_diffusion_peb.src.exposure import acid_generation
from reaction_diffusion_peb.src.full_reaction_diffusion import (
    evolve_full_reaction_diffusion_fd_at_T,
)
from reaction_diffusion_peb.src.stochastic_layers import molecular_blur_P
from reaction_diffusion_peb.src.synthetic_aerial import gaussian_spot
from reaction_diffusion_peb.src.visualization import save_figure

OUT_FIG = Path("reaction_diffusion_peb/outputs/figures")
OUT_LOG = Path("reaction_diffusion_peb/outputs/logs")

GRID_SIZE = 128
DX_NM = 1.0
SIGMA_PX = 12.0
HMAX = 0.2
ETA = 1.0
DOSE = 1.0
DH = 0.8
DQ = 0.1 * DH
KDEP_REF = 0.5
KLOSS_REF = 0.005
KQ_REF = 1.0
Q0 = 0.1
T_END_S = 60.0
P_THRESHOLD = 0.5
T_C = 100.0
T_REF_C = 100.0
EA = 100.0

SIGMA_BLUR_VALUES = [0.0, 0.5, 1.0, 2.0]


def main() -> None:
    I = gaussian_spot(GRID_SIZE, sigma_px=SIGMA_PX)
    H0 = acid_generation(I, dose=DOSE, eta=ETA, Hmax=HMAX)
    print(f"  H_0 peak={H0.max().item():.4f}  T={T_C} °C  t_end={T_END_S} s")

    _, _, P_raw = evolve_full_reaction_diffusion_fd_at_T(
        H0, Q0_mol_dm3=Q0, DH_nm2_s=DH, DQ_nm2_s=DQ,
        kq_ref_s_inv=KQ_REF, kloss_ref_s_inv=KLOSS_REF,
        kdep_ref_s_inv=KDEP_REF,
        temperature_c=T_C, temperature_ref_c=T_REF_C,
        activation_energy_kj_mol=EA,
        t_end_s=T_END_S, dx_nm=DX_NM,
    )
    raw_area = thresholded_area(P_raw, P_threshold=P_THRESHOLD)
    print(f"  raw P: max={P_raw.max().item():.4f}  area>0.5={raw_area} px")

    rows = [["sigma_blur_nm",
             "P_max", "P_mean",
             "area(P>0.5)_px", "area_shift_px"]]
    blurred_fields = [P_raw]
    rows.append([
        "0", f"{P_raw.max().item():.4f}", f"{P_raw.mean().item():.4f}",
        f"{raw_area}", "0",
    ])
    for sigma in SIGMA_BLUR_VALUES[1:]:
        P_b = molecular_blur_P(P_raw, sigma_nm=sigma, dx_nm=DX_NM)
        a = thresholded_area(P_b, P_threshold=P_THRESHOLD)
        rows.append([
            f"{sigma:g}",
            f"{P_b.max().item():.4f}",
            f"{P_b.mean().item():.4f}",
            f"{a}",
            f"{a - raw_area}",
        ])
        blurred_fields.append(P_b)

    fig, axes = plt.subplots(1, len(SIGMA_BLUR_VALUES),
                              figsize=(4 * len(SIGMA_BLUR_VALUES), 4))
    extent = (0.0, GRID_SIZE * DX_NM, 0.0, GRID_SIZE * DX_NM)
    for ax, sigma, P in zip(axes, SIGMA_BLUR_VALUES, blurred_fields):
        im = ax.imshow(P.numpy(), cmap="Greens", extent=extent,
                       origin="lower", vmin=0.0, vmax=1.0)
        a = thresholded_area(P, P_threshold=P_THRESHOLD)
        ax.set_title(
            f"sigma={sigma:g} nm   P_max={P.max().item():.3f}   area={a} px"
        )
        ax.set_xlabel("x [nm]"); ax.set_ylabel("y [nm]")
        ax.contour(P.numpy(), levels=[P_THRESHOLD],
                   extent=extent, origin="lower",
                   colors="black", linewidths=1.0)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle(f"PEB phase 11 — molecular blur on P_final  "
                 f"T={T_C} °C  t={T_END_S} s")
    fig.tight_layout()
    out_fig = save_figure(fig, OUT_FIG / "peb_phase11_molecular_blur.png")
    print(f"  wrote {out_fig}")

    OUT_LOG.mkdir(parents=True, exist_ok=True)
    log = OUT_LOG / "peb_phase11_molecular_blur_metrics.csv"
    with open(log, "w", newline="") as f:
        csv.writer(f).writerows(rows)
    print(f"  wrote {log}")

    print()
    print("molecular-blur metrics:")
    widths = [max(len(str(r[c])) for r in rows) for c in range(len(rows[0]))]
    for r in rows:
        print("  " + "  ".join(str(v).ljust(w) for v, w in zip(r, widths)))


if __name__ == "__main__":
    main()

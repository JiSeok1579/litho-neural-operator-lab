"""PEB Phase-4 demo: FD diffusion + acid loss with kloss sweep.

Run:
    python reaction_diffusion_peb/experiments/04_acid_loss/run_acid_loss_fd.py

Sweeps kloss in {0, 0.001, 0.005, 0.01, 0.05} 1/s. For each, runs the
FD evolution and compares the total mass to the analytic
``M_0 * exp(-k_loss * t)``. Saves figures + a metrics CSV.
"""

from __future__ import annotations

import csv
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import numpy as np
import torch

from reaction_diffusion_peb.src.exposure import acid_generation
from reaction_diffusion_peb.src.reaction_diffusion import (
    diffuse_acid_loss_fd,
    expected_mass_decay_factor,
    total_mass,
)
from reaction_diffusion_peb.src.synthetic_aerial import gaussian_spot
from reaction_diffusion_peb.src.visualization import (
    save_figure,
    show_diffusion_chain,
    show_diffusion_dh_sweep,
)

OUT_FIG = Path("reaction_diffusion_peb/outputs/figures")
OUT_LOG = Path("reaction_diffusion_peb/outputs/logs")

GRID_SIZE = 128
DX_NM = 1.0
SIGMA_PX = 12.0
HMAX = 0.2
ETA = 1.0
DOSE = 1.0
DH = 0.8
T_END = 60.0
KLOSS_VALUES = [0.0, 0.001, 0.005, 0.01, 0.05]


def main() -> None:
    I = gaussian_spot(GRID_SIZE, sigma_px=SIGMA_PX)
    H0 = acid_generation(I, dose=DOSE, eta=ETA, Hmax=HMAX)
    m0 = total_mass(H0, DX_NM)
    print(f"  H_0 peak={H0.max().item():.4f}, mass={m0:.3f} mol/dm^3 nm^2")

    rows = [["kloss_s_inv", "DH_nm2_s", "t_s", "H_peak",
             "mass_FD", "mass_expected_exp", "rel_mass_err"]]

    H_results = []
    for kloss in KLOSS_VALUES:
        H_t = diffuse_acid_loss_fd(
            H0, DH_nm2_s=DH, kloss_s_inv=kloss, t_end_s=T_END, dx_nm=DX_NM,
        )
        H_results.append(H_t)
        m_fd = total_mass(H_t, DX_NM)
        m_expected = m0 * expected_mass_decay_factor(kloss, T_END)
        rel = abs(m_fd - m_expected) / max(m_expected, 1e-12)
        rows.append([
            f"{kloss:.3f}",
            f"{DH:.2f}",
            f"{T_END:.0f}",
            f"{H_t.max().item():.4f}",
            f"{m_fd:.4f}",
            f"{m_expected:.4f}",
            f"{rel:.6f}",
        ])

    # Reuse the diffusion-sweep plot, captioned with kloss values
    fig = show_diffusion_dh_sweep(
        H0, H_results, [k for k in KLOSS_VALUES], dx_nm=DX_NM, t_end_s=T_END, Hmax=HMAX,
        suptitle=(f"PEB phase 4 (FD): kloss sweep at DH={DH} nm^2/s, "
                  f"t={T_END} s   (panels labelled by 'DH' = kloss here)"),
    )
    out = save_figure(fig, OUT_FIG / "peb_phase4_fd_kloss_sweep.png")
    print(f"  wrote {out}")

    # Reference run at kloss=0.005 (mid value)
    kloss_ref = 0.005
    H_ref = diffuse_acid_loss_fd(
        H0, DH_nm2_s=DH, kloss_s_inv=kloss_ref, t_end_s=T_END, dx_nm=DX_NM,
    )
    fig = show_diffusion_chain(
        H0, H_ref, dx_nm=DX_NM, t_end_s=T_END, Hmax=HMAX,
        suptitle=(f"PEB phase 4 (FD): DH={DH}, kloss={kloss_ref} 1/s, t={T_END} s   "
                  f"M decay = exp(-{kloss_ref}*{T_END}) = "
                  f"{expected_mass_decay_factor(kloss_ref, T_END):.3f}"),
    )
    out = save_figure(fig, OUT_FIG / "peb_phase4_fd_chain.png")
    print(f"  wrote {out}")

    OUT_LOG.mkdir(parents=True, exist_ok=True)
    log = OUT_LOG / "peb_phase4_fd_metrics.csv"
    with open(log, "w", newline="") as f:
        csv.writer(f).writerows(rows)
    print(f"  wrote {log}")

    print()
    print("metrics (mass_FD vs analytic exp(-kloss * t)):")
    widths = [max(len(str(r[c])) for r in rows) for c in range(len(rows[0]))]
    for r in rows:
        print("  " + "  ".join(str(v).ljust(w) for v, w in zip(r, widths)))


if __name__ == "__main__":
    main()

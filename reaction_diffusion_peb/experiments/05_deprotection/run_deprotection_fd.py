"""PEB Phase-5 demo: FD evolution of (H, P) with kdep sweep.

Run:
    python reaction_diffusion_peb/experiments/05_deprotection/run_deprotection_fd.py

Sweeps kdep in {0.0, 0.01, 0.05, 0.1, 0.5, 1.0} 1/s. For each, runs
the FD evolver and reports the six verification criteria from
study plan §5:

  1. kdep = 0       ->  P = 0 everywhere
  2. H_0 ~= 0 region  ->  P stays near 0
  3. H_max region   ->  P grows fastest
  4. P stays in [0, 1]
  5. larger kdep    ->  larger P_final
  6. threshold contour widens with kdep / time
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import torch

from reaction_diffusion_peb.src.deprotection import (
    evolve_acid_loss_deprotection_fd,
    thresholded_area,
)
from reaction_diffusion_peb.src.exposure import acid_generation
from reaction_diffusion_peb.src.synthetic_aerial import gaussian_spot
from reaction_diffusion_peb.src.visualization import (
    save_figure,
    show_deprotection_chain,
    show_deprotection_kdep_sweep,
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
KLOSS = 0.005
T_END = 60.0
P_THRESHOLD = 0.5
KDEP_VALUES = [0.0, 0.01, 0.05, 0.1, 0.5, 1.0]


def main() -> None:
    I = gaussian_spot(GRID_SIZE, sigma_px=SIGMA_PX)
    H0 = acid_generation(I, dose=DOSE, eta=ETA, Hmax=HMAX)
    n = GRID_SIZE
    center = (n // 2, n // 2)
    corner = (0, 0)
    H0_center = H0[center].item()
    H0_corner = H0[corner].item()
    print(f"  H_0 peak={H0.max().item():.4f}, center={H0_center:.4f}, "
          f"corner={H0_corner:.4f}")

    rows = [["kdep_s_inv", "H_peak_final", "P_max", "P_min",
             "P_mean", "P_center", "P_corner",
             "P_in_unit_interval",
             f"area(P>{P_THRESHOLD}) [px]"]]

    P_results = []
    for kdep in KDEP_VALUES:
        H_t, P_t = evolve_acid_loss_deprotection_fd(
            H0, DH_nm2_s=DH, kloss_s_inv=KLOSS, kdep_s_inv=kdep,
            t_end_s=T_END, dx_nm=DX_NM,
        )
        P_results.append(P_t)

        in_unit = bool((P_t >= 0).all().item() and (P_t <= 1).all().item())
        area_px = thresholded_area(P_t, P_threshold=P_THRESHOLD)
        rows.append([
            f"{kdep:.2f}",
            f"{H_t.max().item():.4f}",
            f"{P_t.max().item():.4f}",
            f"{P_t.min().item():.4f}",
            f"{P_t.mean().item():.4f}",
            f"{P_t[center].item():.4f}",
            f"{P_t[corner].item():.4f}",
            "yes" if in_unit else "NO",
            f"{area_px}",
        ])

    # kdep sweep figure
    fig = show_deprotection_kdep_sweep(
        H0, P_results, KDEP_VALUES, dx_nm=DX_NM, t_end_s=T_END,
        P_threshold=P_THRESHOLD,
        suptitle=(f"PEB phase 5 (FD): kdep sweep at DH={DH}, "
                  f"kloss={KLOSS}, t={T_END} s, threshold P>{P_THRESHOLD}"),
    )
    out = save_figure(fig, OUT_FIG / "peb_phase5_fd_kdep_sweep.png")
    print(f"  wrote {out}")

    # Reference chain at kdep=0.5 (mid-range)
    kdep_ref = 0.5
    H_ref, P_ref = evolve_acid_loss_deprotection_fd(
        H0, DH_nm2_s=DH, kloss_s_inv=KLOSS, kdep_s_inv=kdep_ref,
        t_end_s=T_END, dx_nm=DX_NM,
    )
    fig = show_deprotection_chain(
        H0, H_ref, P_ref, dx_nm=DX_NM, t_end_s=T_END,
        P_threshold=P_THRESHOLD, Hmax=HMAX,
        suptitle=(f"PEB phase 5 (FD): chain   DH={DH}   kloss={KLOSS}   "
                  f"kdep={kdep_ref}   t={T_END} s"),
    )
    out = save_figure(fig, OUT_FIG / "peb_phase5_fd_chain.png")
    print(f"  wrote {out}")

    # Time sweep (verification criterion 6)
    print()
    print("  time sweep at kdep=0.5 (criterion 6):")
    t_rows = [["t_end_s", f"area(P>{P_THRESHOLD}) [px]", "P_max", "P_mean"]]
    for t in [15.0, 30.0, 60.0, 120.0]:
        _, P_t = evolve_acid_loss_deprotection_fd(
            H0, DH_nm2_s=DH, kloss_s_inv=KLOSS, kdep_s_inv=0.5,
            t_end_s=t, dx_nm=DX_NM,
        )
        t_rows.append([
            f"{t:.0f}",
            f"{thresholded_area(P_t, P_threshold=P_THRESHOLD)}",
            f"{P_t.max().item():.4f}",
            f"{P_t.mean().item():.4f}",
        ])

    OUT_LOG.mkdir(parents=True, exist_ok=True)
    log_kdep = OUT_LOG / "peb_phase5_fd_kdep_metrics.csv"
    with open(log_kdep, "w", newline="") as f:
        csv.writer(f).writerows(rows)
    print(f"  wrote {log_kdep}")
    log_t = OUT_LOG / "peb_phase5_fd_time_metrics.csv"
    with open(log_t, "w", newline="") as f:
        csv.writer(f).writerows(t_rows)
    print(f"  wrote {log_t}")

    print()
    print("kdep sweep metrics:")
    widths = [max(len(str(r[c])) for r in rows) for c in range(len(rows[0]))]
    for r in rows:
        print("  " + "  ".join(str(v).ljust(w) for v, w in zip(r, widths)))
    print()
    print("time sweep metrics:")
    widths = [max(len(str(r[c])) for r in t_rows) for c in range(len(t_rows[0]))]
    for r in t_rows:
        print("  " + "  ".join(str(v).ljust(w) for v, w in zip(r, widths)))


if __name__ == "__main__":
    main()

"""PEB Phase-6 demo: PEB-time sweep at a fixed temperature.

Run:
    python reaction_diffusion_peb/experiments/06_temperature_peb/run_time_sweep.py

Holds T at the reference temperature (so the Arrhenius factor is 1)
and sweeps the PEB time t in {60, 75, 90} s. Verifies that the
threshold area grows monotonically with bake time.
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import torch

from reaction_diffusion_peb.src.arrhenius import (
    arrhenius_factor,
    evolve_acid_loss_deprotection_fd_at_T,
)
from reaction_diffusion_peb.src.deprotection import thresholded_area
from reaction_diffusion_peb.src.exposure import acid_generation
from reaction_diffusion_peb.src.synthetic_aerial import gaussian_spot
from reaction_diffusion_peb.src.visualization import (
    save_figure,
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

KDEP_REF = 0.5
KLOSS_REF = 0.005
EA_KJ_MOL = 100.0
T_C = 100.0
T_REF_C = 100.0     # same as T_C -> factor = 1 -> rates equal references
P_THRESHOLD = 0.5

T_VALUES_S = [60.0, 75.0, 90.0]


def main() -> None:
    I = gaussian_spot(GRID_SIZE, sigma_px=SIGMA_PX)
    H0 = acid_generation(I, dose=DOSE, eta=ETA, Hmax=HMAX)
    factor = arrhenius_factor(T_C, T_REF_C, EA_KJ_MOL)
    print(f"  H_0 peak={H0.max().item():.4f}, T={T_C}={T_REF_C}=T_ref °C, "
          f"factor={factor:.4f} (should be 1 by criterion 1)")

    rows = [["t_end_s", "T_c", "factor", "P_max", "P_mean",
             f"area(P>{P_THRESHOLD})_px", "P_in_unit_interval"]]

    P_results = []
    for t_end in T_VALUES_S:
        H_t, P_t = evolve_acid_loss_deprotection_fd_at_T(
            H0,
            DH_nm2_s=DH,
            kdep_ref_s_inv=KDEP_REF, kloss_ref_s_inv=KLOSS_REF,
            temperature_c=T_C, temperature_ref_c=T_REF_C,
            activation_energy_kj_mol=EA_KJ_MOL,
            t_end_s=t_end, dx_nm=DX_NM,
        )
        P_results.append(P_t)
        in_unit = bool((P_t >= 0).all().item() and (P_t <= 1).all().item())
        rows.append([
            f"{t_end:.0f}",
            f"{T_C:.0f}",
            f"{factor:.4f}",
            f"{P_t.max().item():.4f}",
            f"{P_t.mean().item():.4f}",
            f"{thresholded_area(P_t, P_threshold=P_THRESHOLD)}",
            "yes" if in_unit else "NO",
        ])

    fig = show_deprotection_kdep_sweep(
        H0, P_results, T_VALUES_S, dx_nm=DX_NM, t_end_s=max(T_VALUES_S),
        P_threshold=P_THRESHOLD,
        suptitle=(f"PEB phase 6 (FD): time sweep at T={T_C} °C   "
                  f"DH={DH}   kdep={KDEP_REF}   kloss={KLOSS_REF}   "
                  f"(panels labelled by t in seconds)"),
    )
    out = save_figure(fig, OUT_FIG / "peb_phase6_fd_time_sweep.png")
    print(f"  wrote {out}")

    OUT_LOG.mkdir(parents=True, exist_ok=True)
    log = OUT_LOG / "peb_phase6_fd_time_sweep_metrics.csv"
    with open(log, "w", newline="") as f:
        csv.writer(f).writerows(rows)
    print(f"  wrote {log}")

    print()
    print("time sweep metrics (T fixed at T_ref):")
    widths = [max(len(str(r[c])) for r in rows) for c in range(len(rows[0]))]
    for r in rows:
        print("  " + "  ".join(str(v).ljust(w) for v, w in zip(r, widths)))


if __name__ == "__main__":
    main()

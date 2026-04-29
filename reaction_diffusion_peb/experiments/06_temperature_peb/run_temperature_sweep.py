"""PEB Phase-6 demo: temperature sweep on the FD deprotection evolver.

Run:
    python reaction_diffusion_peb/experiments/06_temperature_peb/run_temperature_sweep.py

Sweeps T in {80, 90, 100, 110, 120, 125} °C at fixed PEB time, applies
the Arrhenius correction to ``k_dep`` and ``k_loss``, and runs the
Phase-5 FD evolver. Reports the five verification criteria:

  1. T = T_ref           ->  factor = 1
  2. T > T_ref, Ea > 0   ->  factor > 1
  3. Ea = 0              ->  factor = 1 always
  4. Higher T            ->  larger P_final and threshold area
  5. 125°C MOR preset    ->  separate, faster case

PINN is intentionally not retrained here — the existing Phase-5 PINN
was trained for one fixed (kdep, kloss) and would not generalize
across temperatures without adding T as an input feature. That is
the right scope for a future phase, not for this one.
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import torch

from reaction_diffusion_peb.src.arrhenius import (
    apply_arrhenius_to_rates,
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

# Phase-6 reference rates (Phase-5 numbers; Arrhenius scales them by
# the temperature factor).
KDEP_REF = 0.5
KLOSS_REF = 0.005
EA_KJ_MOL = 100.0
T_REF_C = 100.0
T_END_S = 60.0
P_THRESHOLD = 0.5

T_VALUES = [80.0, 90.0, 100.0, 110.0, 120.0, 125.0]


def main() -> None:
    I = gaussian_spot(GRID_SIZE, sigma_px=SIGMA_PX)
    H0 = acid_generation(I, dose=DOSE, eta=ETA, Hmax=HMAX)
    print(f"  H_0 peak={H0.max().item():.4f}, T_ref={T_REF_C} °C, "
          f"Ea={EA_KJ_MOL} kJ/mol")

    rows = [["T_c", "factor", "kdep_eff_s_inv", "kloss_eff_s_inv",
             "P_max", "P_mean", f"area(P>{P_THRESHOLD})_px",
             "P_in_unit_interval"]]

    P_results = []
    for T_c in T_VALUES:
        factor = arrhenius_factor(T_c, T_REF_C, EA_KJ_MOL)
        kdep_eff, kloss_eff = apply_arrhenius_to_rates(
            KDEP_REF, KLOSS_REF, T_c, T_REF_C, EA_KJ_MOL,
        )
        H_t, P_t = evolve_acid_loss_deprotection_fd_at_T(
            H0,
            DH_nm2_s=DH,
            kdep_ref_s_inv=KDEP_REF, kloss_ref_s_inv=KLOSS_REF,
            temperature_c=T_c, temperature_ref_c=T_REF_C,
            activation_energy_kj_mol=EA_KJ_MOL,
            t_end_s=T_END_S, dx_nm=DX_NM,
        )
        P_results.append(P_t)

        in_unit = bool((P_t >= 0).all().item() and (P_t <= 1).all().item())
        area_px = thresholded_area(P_t, P_threshold=P_THRESHOLD)
        rows.append([
            f"{T_c:.0f}",
            f"{factor:.4f}",
            f"{kdep_eff:.4f}",
            f"{kloss_eff:.6f}",
            f"{P_t.max().item():.4f}",
            f"{P_t.mean().item():.4f}",
            f"{area_px}",
            "yes" if in_unit else "NO",
        ])

    # Reuse the kdep-sweep plot, labelled by T this time.
    fig = show_deprotection_kdep_sweep(
        H0, P_results, T_VALUES, dx_nm=DX_NM, t_end_s=T_END_S,
        P_threshold=P_THRESHOLD,
        suptitle=(f"PEB phase 6 (FD): temperature sweep   "
                  f"DH={DH}   kdep_ref={KDEP_REF}   kloss_ref={KLOSS_REF}   "
                  f"Ea={EA_KJ_MOL} kJ/mol   T_ref={T_REF_C} °C   "
                  f"t={T_END_S} s   "
                  f"(panels labelled by T °C)"),
    )
    out = save_figure(fig, OUT_FIG / "peb_phase6_fd_temperature_sweep.png")
    print(f"  wrote {out}")

    # Ea=0 control case (criterion 3): should produce identical results
    # at every T, so just record P_max for one T to log it.
    rows_ea = [["case", "T_c", "factor", "P_max", "area(P>0.5)_px"]]
    for T_c in (80.0, 100.0, 125.0):
        factor_zero = arrhenius_factor(T_c, T_REF_C, 0.0)
        _, P_t = evolve_acid_loss_deprotection_fd_at_T(
            H0,
            DH_nm2_s=DH,
            kdep_ref_s_inv=KDEP_REF, kloss_ref_s_inv=KLOSS_REF,
            temperature_c=T_c, temperature_ref_c=T_REF_C,
            activation_energy_kj_mol=0.0,
            t_end_s=T_END_S, dx_nm=DX_NM,
        )
        rows_ea.append([
            "Ea=0",
            f"{T_c:.0f}",
            f"{factor_zero:.4f}",
            f"{P_t.max().item():.4f}",
            f"{thresholded_area(P_t, P_threshold=P_THRESHOLD)}",
        ])

    OUT_LOG.mkdir(parents=True, exist_ok=True)
    log_T = OUT_LOG / "peb_phase6_fd_temperature_sweep_metrics.csv"
    with open(log_T, "w", newline="") as f:
        csv.writer(f).writerows(rows)
    print(f"  wrote {log_T}")
    log_ea = OUT_LOG / "peb_phase6_fd_zero_Ea_check.csv"
    with open(log_ea, "w", newline="") as f:
        csv.writer(f).writerows(rows_ea)
    print(f"  wrote {log_ea}")

    print()
    print("temperature sweep metrics:")
    widths = [max(len(str(r[c])) for r in rows) for c in range(len(rows[0]))]
    for r in rows:
        print("  " + "  ".join(str(v).ljust(w) for v, w in zip(r, widths)))
    print()
    print("Ea=0 control (P_max should be identical across T):")
    widths = [max(len(str(r[c])) for r in rows_ea) for c in range(len(rows_ea[0]))]
    for r in rows_ea:
        print("  " + "  ".join(str(v).ljust(w) for v, w in zip(r, widths)))


if __name__ == "__main__":
    main()

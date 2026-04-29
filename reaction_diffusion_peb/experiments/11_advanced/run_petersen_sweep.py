"""PEB Phase-11 demo: Petersen nonlinear-diffusion sweep.

Run:
    python reaction_diffusion_peb/experiments/11_advanced/run_petersen_sweep.py

Sweeps ``alpha in {0, 0.5, 1, 2, 3}`` for the Petersen acid-mobility
form ``D_H(P) = D_H0 * exp(alpha * P)``. ``alpha = 0`` is the Phase-8
reference and is included so the equivalence is visible in the
output table.

Reports per-alpha:
  - peak / mean H, P
  - threshold area (P > 0.5)
  - acid-loss / quencher-neutralization integrals
  - H / Q mass-budget relative errors
  - dt_max + binding stability term (D_H_max grows as exp(alpha))
  - wall-clock
"""

from __future__ import annotations

import csv
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from reaction_diffusion_peb.src.deprotection import thresholded_area
from reaction_diffusion_peb.src.exposure import acid_generation
from reaction_diffusion_peb.src.petersen_diffusion import (
    evolve_petersen_full_fd_at_T_with_budget,
    stability_report_petersen,
)
from reaction_diffusion_peb.src.synthetic_aerial import gaussian_spot
from reaction_diffusion_peb.src.visualization import (
    save_figure,
    show_quencher_kq_sweep,
    show_quencher_mass_budget,
)

OUT_FIG = Path("reaction_diffusion_peb/outputs/figures")
OUT_LOG = Path("reaction_diffusion_peb/outputs/logs")

GRID_SIZE = 128
DX_NM = 1.0
SIGMA_PX = 12.0
HMAX = 0.2
ETA = 1.0
DOSE = 1.0
DH0 = 0.8
DQ = 0.1 * DH0
KDEP_REF = 0.5
KLOSS_REF = 0.005
KQ_REF = 1.0
Q0 = 0.1
T_END_S = 60.0
P_THRESHOLD = 0.5
T_C = 100.0
T_REF_C = 100.0
EA = 100.0

ALPHA_VALUES = [0.0, 0.5, 1.0, 2.0, 3.0]


def main() -> None:
    I = gaussian_spot(GRID_SIZE, sigma_px=SIGMA_PX)
    H0 = acid_generation(I, dose=DOSE, eta=ETA, Hmax=HMAX)
    print(f"  H_0 peak={H0.max().item():.4f}  D_H0={DH0}  Q_0={Q0}")
    print(f"  kq_ref={KQ_REF}  kdep_ref={KDEP_REF}  kloss_ref={KLOSS_REF}")
    print(f"  T={T_C}=T_ref  Ea={EA} kJ/mol  t_end={T_END_S} s")

    rows = [["alpha", "DH_max", "dt_max", "stiff_term",
             "H_peak", "Q_min", "P_max", "P_mean",
             f"area(P>{P_THRESHOLD})_px",
             "acid_loss_int", "neutralization_int",
             "H_budget_rel_err", "Q_budget_rel_err",
             "wall_clock_s"]]
    histories = []
    P_results, Q_results = [], []

    for alpha in ALPHA_VALUES:
        rep = stability_report_petersen(
            H0, Q0_mol_dm3=Q0,
            DH0_nm2_s=DH0, petersen_alpha=alpha, DQ_nm2_s=DQ,
            kq_ref_s_inv=KQ_REF, kloss_ref_s_inv=KLOSS_REF,
            kdep_ref_s_inv=KDEP_REF,
            temperature_c=T_C, temperature_ref_c=T_REF_C,
            activation_energy_kj_mol=EA,
            dx_nm=DX_NM,
        )
        t0 = time.perf_counter()
        H_t, Q_t, P_t, hist = evolve_petersen_full_fd_at_T_with_budget(
            H0, Q0_mol_dm3=Q0,
            DH0_nm2_s=DH0, petersen_alpha=alpha, DQ_nm2_s=DQ,
            kq_ref_s_inv=KQ_REF, kloss_ref_s_inv=KLOSS_REF,
            kdep_ref_s_inv=KDEP_REF,
            temperature_c=T_C, temperature_ref_c=T_REF_C,
            activation_energy_kj_mol=EA,
            t_end_s=T_END_S, dx_nm=DX_NM,
        )
        wall = time.perf_counter() - t0
        last = hist[-1]
        rows.append([
            f"{alpha:g}",
            f"{rep['DH_max']:.4g}",
            f"{rep['dt_max']:.4g}",
            rep["stiff_term"],
            f"{H_t.max().item():.4f}",
            f"{Q_t.min().item():.4f}",
            f"{P_t.max().item():.4f}",
            f"{P_t.mean().item():.4f}",
            f"{thresholded_area(P_t, P_threshold=P_THRESHOLD)}",
            f"{last.acid_loss_integral:.4e}",
            f"{last.quencher_neutralization_integral:.4e}",
            f"{last.H_budget_relative_error:.3e}",
            f"{last.Q_budget_relative_error:.3e}",
            f"{wall:.3f}",
        ])
        P_results.append(P_t)
        Q_results.append(Q_t)
        histories.append(hist)

    fig = show_quencher_kq_sweep(
        H0, P_results, Q_results, ALPHA_VALUES,
        Q0_mol_dm3=Q0, dx_nm=DX_NM, t_end_s=T_END_S,
        P_threshold=P_THRESHOLD,
        suptitle=(f"PEB phase 11 — Petersen alpha sweep   "
                  f"D_H0={DH0}  D_Q={DQ}  Q0={Q0}  "
                  f"kq_ref={KQ_REF}  kdep_ref={KDEP_REF}  "
                  f"kloss_ref={KLOSS_REF}  t={T_END_S} s   "
                  f"(panels labelled by alpha)"),
    )
    out_fig = save_figure(fig, OUT_FIG / "peb_phase11_petersen_sweep.png")
    print(f"  wrote {out_fig}")

    fig = show_quencher_mass_budget(
        histories, ALPHA_VALUES,
        suptitle="PEB phase 11 — Petersen mass-budget identity check",
    )
    out_fig = save_figure(fig, OUT_FIG / "peb_phase11_petersen_budget.png")
    print(f"  wrote {out_fig}")

    OUT_LOG.mkdir(parents=True, exist_ok=True)
    log = OUT_LOG / "peb_phase11_petersen_sweep_metrics.csv"
    with open(log, "w", newline="") as f:
        csv.writer(f).writerows(rows)
    print(f"  wrote {log}")

    print()
    print("Petersen alpha-sweep metrics:")
    widths = [max(len(str(r[c])) for r in rows) for c in range(len(rows[0]))]
    for r in rows:
        print("  " + "  ".join(str(v).ljust(w) for v, w in zip(r, widths)))


if __name__ == "__main__":
    main()

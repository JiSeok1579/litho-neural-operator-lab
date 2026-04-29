"""PEB Phase-8 demo: full reaction-diffusion (FD only).

Run:
    python reaction_diffusion_peb/experiments/08_full_reaction_diffusion/run_full_model.py

Sweeps T on the integrated (H, Q, P) system with Arrhenius scaling on
``kq``, ``kdep``, ``kloss``. Reports per-T:

  - peak / mean H, Q, P
  - threshold area (pixels with P > 0.5)
  - acid-loss and quencher-neutralization integrals
  - H and Q mass-budget relative errors at t_end
  - which stability term binds at that T

Phase 8 stays in the safe-kq regime (kq_ref = 1 1/s) so the diffusion
CFL stays binding at moderate T; at hot T the kq Arrhenius factor can
push the bimolecular term to be binding instead — that is logged here
so the regime change is visible.
"""

from __future__ import annotations

import csv
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from reaction_diffusion_peb.src.arrhenius import arrhenius_factor
from reaction_diffusion_peb.src.deprotection import thresholded_area
from reaction_diffusion_peb.src.exposure import acid_generation
from reaction_diffusion_peb.src.full_reaction_diffusion import (
    apply_arrhenius_to_full_rates,
    evolve_full_reaction_diffusion_fd_at_T_with_budget,
    stability_report_at_T,
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
DH = 0.8
DQ = 0.1 * DH
KDEP_REF = 0.5
KLOSS_REF = 0.005
KQ_REF = 1.0
Q0 = 0.1
T_END_S = 60.0
P_THRESHOLD = 0.5
T_REF_C = 100.0
EA_KJ_MOL = 100.0

T_VALUES = [80.0, 90.0, 100.0, 110.0, 120.0]


def main() -> None:
    I = gaussian_spot(GRID_SIZE, sigma_px=SIGMA_PX)
    H0 = acid_generation(I, dose=DOSE, eta=ETA, Hmax=HMAX)
    print(f"  H_0 peak={H0.max().item():.4f}  Q_0={Q0}  DH={DH}  DQ={DQ}")
    print(f"  kq_ref={KQ_REF}  kdep_ref={KDEP_REF}  kloss_ref={KLOSS_REF}")
    print(f"  Ea={EA_KJ_MOL} kJ/mol  T_ref={T_REF_C} °C  t={T_END_S} s")

    rows = [["T_c", "factor",
             "kq_eff", "kdep_eff", "kloss_eff",
             "dt_max", "stiff_term",
             "H_peak", "Q_min", "P_max", "P_mean",
             f"area(P>{P_THRESHOLD})_px",
             "acid_loss_int", "neutralization_int",
             "H_budget_rel_err", "Q_budget_rel_err",
             "wall_clock_s"]]
    histories = []
    P_results, Q_results = [], []

    for T_c in T_VALUES:
        f = arrhenius_factor(T_c, T_REF_C, EA_KJ_MOL)
        kdep_eff, kloss_eff, kq_eff = apply_arrhenius_to_full_rates(
            kdep_ref_s_inv=KDEP_REF, kloss_ref_s_inv=KLOSS_REF,
            kq_ref_s_inv=KQ_REF,
            temperature_c=T_c, temperature_ref_c=T_REF_C,
            activation_energy_kj_mol=EA_KJ_MOL,
        )
        bounds = stability_report_at_T(
            H0, Q0_mol_dm3=Q0,
            DH_nm2_s=DH, DQ_nm2_s=DQ,
            kq_ref_s_inv=KQ_REF, kloss_ref_s_inv=KLOSS_REF,
            kdep_ref_s_inv=KDEP_REF,
            temperature_c=T_c, temperature_ref_c=T_REF_C,
            activation_energy_kj_mol=EA_KJ_MOL,
            dx_nm=DX_NM,
        )

        t0 = time.perf_counter()
        H_t, Q_t, P_t, hist = evolve_full_reaction_diffusion_fd_at_T_with_budget(
            H0, Q0_mol_dm3=Q0,
            DH_nm2_s=DH, DQ_nm2_s=DQ,
            kq_ref_s_inv=KQ_REF, kloss_ref_s_inv=KLOSS_REF,
            kdep_ref_s_inv=KDEP_REF,
            temperature_c=T_c, temperature_ref_c=T_REF_C,
            activation_energy_kj_mol=EA_KJ_MOL,
            t_end_s=T_END_S, dx_nm=DX_NM,
        )
        wall = time.perf_counter() - t0

        last = hist[-1]
        rows.append([
            f"{T_c:.0f}",
            f"{f:.4f}",
            f"{kq_eff:.4f}",
            f"{kdep_eff:.4f}",
            f"{kloss_eff:.6f}",
            f"{bounds['dt_max']:.4g}",
            bounds["stiff_term"],
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

    # Reuse Phase-7 sweep visual; the per-column label is now the
    # temperature (we pass it through ``kq_values`` for compactness —
    # the helper just renders the label).
    fig = show_quencher_kq_sweep(
        H0, P_results, Q_results, T_VALUES,
        Q0_mol_dm3=Q0, dx_nm=DX_NM, t_end_s=T_END_S,
        P_threshold=P_THRESHOLD,
        suptitle=(f"PEB phase 8 — full reaction-diffusion T sweep:  "
                  f"DH={DH}  DQ={DQ}  Q0={Q0}  "
                  f"kq_ref={KQ_REF}  kdep_ref={KDEP_REF}  "
                  f"kloss_ref={KLOSS_REF}  Ea={EA_KJ_MOL} kJ/mol  "
                  f"t={T_END_S} s   (panels labelled by T °C)"),
    )
    out_fig = save_figure(fig, OUT_FIG / "peb_phase8_full_T_sweep.png")
    print(f"  wrote {out_fig}")

    fig = show_quencher_mass_budget(
        histories, T_VALUES,
        suptitle="PEB phase 8 — mass-budget identity check (per T °C)",
    )
    out_fig = save_figure(fig, OUT_FIG / "peb_phase8_full_T_budget.png")
    print(f"  wrote {out_fig}")

    OUT_LOG.mkdir(parents=True, exist_ok=True)
    log_main = OUT_LOG / "peb_phase8_full_T_sweep_metrics.csv"
    with open(log_main, "w", newline="") as f:
        csv.writer(f).writerows(rows)
    print(f"  wrote {log_main}")

    print()
    print("full-reaction-diffusion T sweep metrics:")
    widths = [max(len(str(r[c])) for r in rows) for c in range(len(rows[0]))]
    for r in rows:
        print("  " + "  ".join(str(v).ljust(w) for v, w in zip(r, widths)))


if __name__ == "__main__":
    main()

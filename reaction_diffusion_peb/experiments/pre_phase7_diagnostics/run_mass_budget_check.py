"""Pre-Phase-7 demo: mass-budget diagnostic on Phase-5 / Phase-6 setups.

Run:
    python reaction_diffusion_peb/experiments/pre_phase7_diagnostics/run_mass_budget_check.py

For each scenario, runs the FD evolver-with-budget and reports the
mass-conservation identity

    M_budget(t) = total_mass(H(t))
                  + integral_0^t k_loss * total_mass(H(tau)) dtau
                ≈ total_mass(H_0)

Saves:
    outputs/figures/peb_pre_phase7_mass_budget_curves.png
    outputs/logs/peb_pre_phase7_mass_budget.csv

The relative error should be at the level of float32 round-off (~1e-7
to 1e-6) for every scenario, including the high-temperature Arrhenius
case at 125 °C.
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import matplotlib.pyplot as plt
import numpy as np
import torch

from reaction_diffusion_peb.src.exposure import acid_generation
from reaction_diffusion_peb.src.mass_budget import (
    evolve_acid_loss_deprotection_fd_with_budget,
    evolve_acid_loss_deprotection_fd_with_budget_at_T,
)
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


def _scenarios() -> list[dict]:
    """Define the scenarios as a list of dicts. Some are plain Phase-5
    runs; some are Phase-6 Arrhenius runs."""
    return [
        # Phase-5 setups
        {"name": "phase5  kloss=0           kdep=0.0",
         "is_arrhenius": False,
         "DH": 0.8, "kloss": 0.0,    "kdep": 0.0, "t_end": 60.0},
        {"name": "phase5  kloss=0           kdep=0.5",
         "is_arrhenius": False,
         "DH": 0.8, "kloss": 0.0,    "kdep": 0.5, "t_end": 60.0},
        {"name": "phase5  kloss=0.005       kdep=0.5",
         "is_arrhenius": False,
         "DH": 0.8, "kloss": 0.005,  "kdep": 0.5, "t_end": 60.0},
        {"name": "phase5  kloss=0.05 (10x)  kdep=0.5",
         "is_arrhenius": False,
         "DH": 0.8, "kloss": 0.05,   "kdep": 0.5, "t_end": 60.0},
        # Phase-6 Arrhenius cases (T_ref=100, Ea=100 kJ/mol)
        {"name": "phase6  T= 80 °C  (factor 0.16)",
         "is_arrhenius": True,
         "DH": 0.8, "kloss_ref": 0.005, "kdep_ref": 0.5,
         "T_c": 80.0, "T_ref_c": 100.0, "Ea_kj": 100.0, "t_end": 60.0},
        {"name": "phase6  T=100 °C  (factor 1.00)",
         "is_arrhenius": True,
         "DH": 0.8, "kloss_ref": 0.005, "kdep_ref": 0.5,
         "T_c": 100.0, "T_ref_c": 100.0, "Ea_kj": 100.0, "t_end": 60.0},
        {"name": "phase6  T=120 °C  (factor 5.15)",
         "is_arrhenius": True,
         "DH": 0.8, "kloss_ref": 0.005, "kdep_ref": 0.5,
         "T_c": 120.0, "T_ref_c": 100.0, "Ea_kj": 100.0, "t_end": 60.0},
        {"name": "phase6  T=125 °C MOR (factor 7.57)",
         "is_arrhenius": True,
         "DH": 0.8, "kloss_ref": 0.005, "kdep_ref": 0.5,
         "T_c": 125.0, "T_ref_c": 100.0, "Ea_kj": 100.0, "t_end": 60.0},
    ]


def main() -> None:
    I = gaussian_spot(GRID_SIZE, sigma_px=SIGMA_PX)
    H0 = acid_generation(I, dose=DOSE, eta=ETA, Hmax=HMAX)
    print(f"  H_0 mass = {(H0.sum() * DX_NM ** 2).item():.4f} mol/dm^3 nm^2")

    rows = [["scenario", "mass_H_initial", "mass_H_final",
             "acid_loss_integral", "mass_budget_final",
             "mass_budget_relative_error"]]
    histories = []

    for sc in _scenarios():
        if sc["is_arrhenius"]:
            _, _, history = evolve_acid_loss_deprotection_fd_with_budget_at_T(
                H0,
                DH_nm2_s=sc["DH"],
                kdep_ref_s_inv=sc["kdep_ref"],
                kloss_ref_s_inv=sc["kloss_ref"],
                temperature_c=sc["T_c"],
                temperature_ref_c=sc["T_ref_c"],
                activation_energy_kj_mol=sc["Ea_kj"],
                t_end_s=sc["t_end"], dx_nm=DX_NM,
                n_log_points=21,
            )
        else:
            _, _, history = evolve_acid_loss_deprotection_fd_with_budget(
                H0,
                DH_nm2_s=sc["DH"],
                kloss_s_inv=sc["kloss"],
                kdep_s_inv=sc["kdep"],
                t_end_s=sc["t_end"], dx_nm=DX_NM,
                n_log_points=21,
            )
        histories.append((sc["name"], history))
        final = history[-1]
        rows.append([
            sc["name"],
            f"{history[0].mass_H:.4f}",
            f"{final.mass_H:.4f}",
            f"{final.acid_loss_integral:.4f}",
            f"{final.mass_budget:.4f}",
            f"{final.mass_budget_relative_error:.3e}",
        ])

    OUT_LOG.mkdir(parents=True, exist_ok=True)
    log = OUT_LOG / "peb_pre_phase7_mass_budget.csv"
    with open(log, "w", newline="") as f:
        csv.writer(f).writerows(rows)
    print(f"  wrote {log}")

    # Curves figure: mass_H, acid_loss_integral, mass_budget vs t for
    # each scenario.
    fig, axes = plt.subplots(2, 4, figsize=(20, 9))
    axes = axes.flatten()
    m0 = histories[0][1][0].mass_H
    for ax, (name, history) in zip(axes, histories):
        ts = [s.t_s for s in history]
        mass_H = [s.mass_H for s in history]
        loss = [s.acid_loss_integral for s in history]
        budget = [s.mass_budget for s in history]
        ax.plot(ts, mass_H, label="mass_H", color="C3")
        ax.plot(ts, loss, label="∫ k_loss * mass_H", color="C0")
        ax.plot(ts, budget, label="mass_H + ∫", color="black",
                linestyle="--")
        ax.axhline(m0, color="grey", linestyle=":", label="mass_H_initial")
        rel_err_ppm = history[-1].mass_budget_relative_error * 1e6
        ax.set_title(f"{name}\nrel err {rel_err_ppm:.2f} ppm")
        ax.set_xlabel("t [s]")
        ax.set_ylabel("mass [mol/dm^3 nm^2]")
        ax.legend(fontsize=7, loc="best")
        ax.grid(alpha=0.3)
    fig.suptitle("PEB pre-Phase-7: mass-budget diagnostic across "
                 "Phase 5 / Phase 6 FD scenarios")
    fig.tight_layout()
    out = save_figure(fig, OUT_FIG / "peb_pre_phase7_mass_budget_curves.png")
    print(f"  wrote {out}")

    print()
    print("mass budget summary:")
    widths = [max(len(str(r[c])) for r in rows) for c in range(len(rows[0]))]
    for r in rows:
        print("  " + "  ".join(str(v).ljust(w) for v, w in zip(r, widths)))


if __name__ == "__main__":
    main()

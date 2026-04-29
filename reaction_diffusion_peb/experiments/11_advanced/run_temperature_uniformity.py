"""PEB Phase-11 demo: temperature-uniformity ensemble.

Run:
    python reaction_diffusion_peb/experiments/11_advanced/run_temperature_uniformity.py

Sweeps the per-run temperature standard deviation
``temperature_uniformity_c in {0, 0.5, 1, 2}`` and reports the
ensemble mean / std of (H, Q, P) at ``t_end``. ``σ_T = 0`` is the
deterministic baseline; everything else shows how a wafer-level
non-uniformity propagates through the Phase-8 evolver.

Reports per σ_T:
  - mean / max P_std (acid spread)
  - threshold area for the ensemble-mean P
  - threshold area std across ensemble members
  - wall clock
"""

from __future__ import annotations

import csv
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import torch

from reaction_diffusion_peb.src.deprotection import thresholded_area
from reaction_diffusion_peb.src.exposure import acid_generation
from reaction_diffusion_peb.src.full_reaction_diffusion import (
    evolve_full_reaction_diffusion_fd_at_T,
)
from reaction_diffusion_peb.src.stochastic_layers import (
    temperature_uniformity_ensemble,
)
from reaction_diffusion_peb.src.synthetic_aerial import gaussian_spot
from reaction_diffusion_peb.src.visualization import save_figure

import matplotlib.pyplot as plt
import numpy as np

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

SIGMA_T_VALUES = [0.0, 0.5, 1.0, 2.0]
N_RUNS = 12
SEED = 20260430


def main() -> None:
    I = gaussian_spot(GRID_SIZE, sigma_px=SIGMA_PX)
    H0 = acid_generation(I, dose=DOSE, eta=ETA, Hmax=HMAX)
    print(f"  H_0 peak={H0.max().item():.4f}  T_nominal={T_C} °C  "
          f"n_runs={N_RUNS}")

    rows = [["sigma_T_c", "n_runs",
             "P_mean_max", "P_std_max", "P_std_mean",
             "area_mean_px", "area_std_px",
             "wall_clock_s"]]
    P_means, P_stds = [], []
    for sigma_T in SIGMA_T_VALUES:
        evolver_kwargs = dict(
            H0=H0, Q0_mol_dm3=Q0,
            DH_nm2_s=DH, DQ_nm2_s=DQ,
            kq_ref_s_inv=KQ_REF, kloss_ref_s_inv=KLOSS_REF,
            kdep_ref_s_inv=KDEP_REF,
            temperature_c=T_C, temperature_ref_c=T_REF_C,
            activation_energy_kj_mol=EA,
            t_end_s=T_END_S, dx_nm=DX_NM,
        )
        t0 = time.perf_counter()
        res = temperature_uniformity_ensemble(
            evolver=evolve_full_reaction_diffusion_fd_at_T,
            evolver_kwargs=evolver_kwargs,
            temperature_uniformity_c=sigma_T,
            n_runs=N_RUNS, seed=SEED,
        )
        wall = time.perf_counter() - t0

        # Per-member threshold area
        per_member_areas = []
        # We do not keep individual P fields in EnsembleResult for
        # memory reasons — re-run if we need them. For the area
        # statistic we re-evolve the same temperatures here.
        for T_run in res.temperatures_c:
            kw = dict(evolver_kwargs)
            kw["temperature_c"] = T_run
            _, _, P_run = evolve_full_reaction_diffusion_fd_at_T(**kw)
            per_member_areas.append(
                thresholded_area(P_run, P_threshold=P_THRESHOLD)
            )

        area_arr = torch.tensor(per_member_areas, dtype=torch.float32)
        mean_area = float(area_arr.mean().item())
        std_area = float(area_arr.std(unbiased=False).item()) if N_RUNS > 1 else 0.0

        rows.append([
            f"{sigma_T:g}", str(N_RUNS),
            f"{res.P_mean.max().item():.4f}",
            f"{res.P_std.max().item():.4f}",
            f"{res.P_std.mean().item():.4f}",
            f"{mean_area:.1f}",
            f"{std_area:.1f}",
            f"{wall:.3f}",
        ])
        P_means.append(res.P_mean)
        P_stds.append(res.P_std)

    # Plot: 2-row figure. Row 0 = ensemble-mean P; Row 1 = per-pixel
    # std. Columns labelled by σ_T.
    fig, axes = plt.subplots(2, len(SIGMA_T_VALUES),
                              figsize=(4 * len(SIGMA_T_VALUES), 8))
    extent = (0.0, GRID_SIZE * DX_NM, 0.0, GRID_SIZE * DX_NM)
    p_std_max = max(float(s.max().item()) for s in P_stds) or 1e-6

    for col, (sigma_T, P_mean, P_std) in enumerate(
        zip(SIGMA_T_VALUES, P_means, P_stds)
    ):
        ax = axes[0, col]
        im = ax.imshow(P_mean.numpy(), cmap="Greens", extent=extent,
                       origin="lower", vmin=0.0, vmax=1.0)
        ax.set_title(f"σ_T={sigma_T:g} °C  mean P")
        ax.set_xlabel("x [nm]"); ax.set_ylabel("y [nm]")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        ax = axes[1, col]
        im = ax.imshow(P_std.numpy(), cmap="inferno", extent=extent,
                       origin="lower", vmin=0.0, vmax=p_std_max)
        ax.set_title(f"σ_T={sigma_T:g} °C  P std")
        ax.set_xlabel("x [nm]"); ax.set_ylabel("y [nm]")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle(f"PEB phase 11 — temperature-uniformity ensemble  "
                 f"n_runs={N_RUNS}  T_nominal={T_C} °C  t={T_END_S} s")
    fig.tight_layout()
    out_fig = save_figure(
        fig, OUT_FIG / "peb_phase11_temperature_uniformity.png",
    )
    print(f"  wrote {out_fig}")

    OUT_LOG.mkdir(parents=True, exist_ok=True)
    log = OUT_LOG / "peb_phase11_temperature_uniformity_metrics.csv"
    with open(log, "w", newline="") as f:
        csv.writer(f).writerows(rows)
    print(f"  wrote {log}")

    print()
    print("temperature-uniformity ensemble metrics:")
    widths = [max(len(str(r[c])) for r in rows) for c in range(len(rows[0]))]
    for r in rows:
        print("  " + "  ".join(str(v).ljust(w) for v, w in zip(r, widths)))


if __name__ == "__main__":
    main()

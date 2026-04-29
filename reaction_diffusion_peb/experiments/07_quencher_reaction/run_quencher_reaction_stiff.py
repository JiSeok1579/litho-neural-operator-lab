"""PEB Phase-7 demo: acid-quencher reaction (stiff kq regime).

Run:
    python reaction_diffusion_peb/experiments/07_quencher_reaction/run_quencher_reaction_stiff.py

Sweeps ``k_q in {100, 300, 1000} 1/s`` — realistic CAR / MOR values.
At kq=1000 with H_max≈0.2 the stability-limiting term is

    dt <= 1 / (kq * H_max) ≈ 5 ms

so 60 s of evolution requires ~12k explicit steps. The point of this
demo is to show that the FD scheme stays bounded, mass-budgets close,
and the threshold area drops monotonically with kq. PINN training in
this regime is intentionally deferred (FUTURE_WORK item).
"""

from __future__ import annotations

import csv
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from reaction_diffusion_peb.src.deprotection import thresholded_area
from reaction_diffusion_peb.src.exposure import acid_generation
from reaction_diffusion_peb.src.quencher_reaction import (
    evolve_quencher_fd_with_budget,
    history_to_dicts,
    stability_report,
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
KDEP = 0.5
KLOSS = 0.005
Q0 = 0.1
T_END_S = 60.0
P_THRESHOLD = 0.5

KQ_VALUES = [100.0, 300.0, 1000.0]


def main() -> None:
    I = gaussian_spot(GRID_SIZE, sigma_px=SIGMA_PX)
    H0 = acid_generation(I, dose=DOSE, eta=ETA, Hmax=HMAX)
    print(f"  H_0 peak={H0.max().item():.4f}  Q_0={Q0}  DH={DH}  DQ={DQ}")
    print(f"  kdep={KDEP}  kloss={KLOSS}  t_end={T_END_S} s   STIFF kq sweep")

    rows = [["kq_s_inv", "dt_max_s", "stiff_term", "n_steps_est",
             "H_peak", "H_mean",
             "Q_min", "Q_mean",
             "P_max", "P_mean",
             f"area(P>{P_THRESHOLD})_px",
             "acid_loss_int", "neutralization_int",
             "H_budget_rel_err", "Q_budget_rel_err",
             "wall_clock_s"]]
    histories = []
    P_results, Q_results = [], []

    for kq in KQ_VALUES:
        bounds = stability_report(
            H0, Q0_mol_dm3=Q0,
            DH_nm2_s=DH, DQ_nm2_s=DQ,
            kq_s_inv=kq, kloss_s_inv=KLOSS, kdep_s_inv=KDEP,
            dx_nm=DX_NM,
        )
        n_steps_est = max(1, int(T_END_S / (0.5 * bounds["dt_max"])))
        t0 = time.perf_counter()
        H_t, Q_t, P_t, hist = evolve_quencher_fd_with_budget(
            H0, Q0_mol_dm3=Q0,
            DH_nm2_s=DH, DQ_nm2_s=DQ,
            kq_s_inv=kq, kloss_s_inv=KLOSS, kdep_s_inv=KDEP,
            t_end_s=T_END_S, dx_nm=DX_NM,
        )
        wall = time.perf_counter() - t0

        last = hist[-1]
        rows.append([
            f"{kq:g}",
            f"{bounds['dt_max']:.4g}",
            bounds["stiff_term"],
            f"{n_steps_est}",
            f"{H_t.max().item():.4f}",
            f"{H_t.mean().item():.4f}",
            f"{Q_t.min().item():.4f}",
            f"{Q_t.mean().item():.4f}",
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

    # Figures
    fig = show_quencher_kq_sweep(
        H0, P_results, Q_results, KQ_VALUES,
        Q0_mol_dm3=Q0, dx_nm=DX_NM, t_end_s=T_END_S,
        P_threshold=P_THRESHOLD,
        suptitle=(f"PEB phase 7 (stiff kq): DH={DH}  DQ={DQ}  Q0={Q0}  "
                  f"kdep={KDEP}  kloss={KLOSS}  t={T_END_S} s"),
    )
    out_fig = save_figure(fig, OUT_FIG / "peb_phase7_quencher_stiff_sweep.png")
    print(f"  wrote {out_fig}")

    fig = show_quencher_mass_budget(
        histories, KQ_VALUES,
        suptitle=f"PEB phase 7 (stiff kq) — mass-budget identity check",
    )
    out_fig = save_figure(fig, OUT_FIG / "peb_phase7_quencher_stiff_budget.png")
    print(f"  wrote {out_fig}")

    # CSV logs
    OUT_LOG.mkdir(parents=True, exist_ok=True)
    log_main = OUT_LOG / "peb_phase7_quencher_stiff_metrics.csv"
    with open(log_main, "w", newline="") as f:
        csv.writer(f).writerows(rows)
    print(f"  wrote {log_main}")

    log_hist = OUT_LOG / "peb_phase7_quencher_stiff_budget_history.csv"
    with open(log_hist, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["kq_s_inv", "t_s", "mass_H", "mass_Q",
                    "acid_loss_integral", "neutralization_integral",
                    "H_budget", "Q_budget",
                    "H_budget_relative_error", "Q_budget_relative_error"])
        for kq, hist in zip(KQ_VALUES, histories):
            for d in history_to_dicts(hist):
                w.writerow([
                    f"{kq:g}", f"{d['t_s']:.4f}",
                    f"{d['mass_H']:.6f}", f"{d['mass_Q']:.6f}",
                    f"{d['acid_loss_integral']:.6e}",
                    f"{d['quencher_neutralization_integral']:.6e}",
                    f"{d['H_budget']:.6f}", f"{d['Q_budget']:.6f}",
                    f"{d['H_budget_relative_error']:.3e}",
                    f"{d['Q_budget_relative_error']:.3e}",
                ])
    print(f"  wrote {log_hist}")

    print()
    print("stiff-kq sweep metrics:")
    widths = [max(len(str(r[c])) for r in rows) for c in range(len(rows[0]))]
    for r in rows:
        print("  " + "  ".join(str(v).ljust(w) for v, w in zip(r, widths)))


if __name__ == "__main__":
    main()

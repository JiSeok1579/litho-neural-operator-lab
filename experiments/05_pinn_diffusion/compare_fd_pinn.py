"""Phase 6 demo: PINN vs FD vs FFT for the 2D heat equation.

Run:
    python experiments/05_pinn_diffusion/compare_fd_pinn.py

Initial condition: 2D Gaussian ``A0(x,y) = exp(-(x^2+y^2)/(2 sigma^2))``
with sigma = 0.5 lambda. Diffusivity D = 0.1, t_end = 1, so the analytic
Gaussian widens from sigma_0 = 0.5 to sigma_t = sqrt(0.25 + 0.2) ~ 0.671
(diffusion length L = sqrt(2 D t) ~ 0.447).

Three solvers are compared against the closed-form Gaussian:

- FD     : explicit Euler 5-point Laplacian (Phase 5 module).
- FFT    : exact heat-kernel multiplication in Fourier space (Phase 5).
- PINN   : tanh-activated MLP with random Fourier features, trained for
           3000 iterations of Adam at lr=1e-3, weight_ic=100.

Saves under ``outputs/figures/``:
- phase6_pinn_vs_solvers.png — 2x4 (solutions / abs-errors)
- phase6_pinn_training.png  — PINN loss curve

Saves under ``outputs/logs/``:
- phase6_metrics.csv — solver, MSE, max-abs error, run-time

Physical takeaways verified by the run:
- FFT matches the analytic solution to round-off (~1e-7 max abs error).
- FD has truncation error ~5e-5 at this dx and t_end.
- PINN reaches ~1e-1 max abs error after 10000 iterations + a
  hard-IC architectural trick (A_pinn = A0(x,y) + t * MLP). This is
  worse than FD / FFT by roughly four orders of magnitude on a problem
  with a closed-form Gaussian solution, taking ~80 seconds of training
  to get there. This **is** the canonical teaching moment of the
  study plan §6.8: on simple diffusion the analytic / FFT / FD chain
  is much faster and much more accurate than a PINN. The PINN's edge
  shows up only when (a) the geometry / boundary conditions are
  irregular enough that grid solvers struggle, (b) inverse parameter
  estimation is needed, or (c) you need a continuous mesh-free
  representation queryable at arbitrary (x, y, t).
"""

from __future__ import annotations

import csv
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch

from src.common.grid import Grid2D
from src.common.visualization import (
    save_figure,
    show_pinn_training,
    show_pinn_vs_solvers,
)
from src.pinn.pinn_diffusion import (
    PINNDiffusion,
    gaussian_analytic_solution,
    gaussian_initial_condition,
    pinn_to_grid,
    train_pinn_diffusion,
)
from src.resist.diffusion_fd import diffuse_fd
from src.resist.diffusion_fft import diffuse_fft

OUT_FIG = Path("outputs/figures")
OUT_LOG = Path("outputs/logs")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

SIGMA = 0.5
D = 0.1
T_END = 1.0
N = 128
EXTENT = 8.0


def _mse(A: torch.Tensor, B: torch.Tensor) -> float:
    return float(((A - B) ** 2).mean().item())


def _max_abs_err(A: torch.Tensor, B: torch.Tensor) -> float:
    return float((A - B).abs().max().item())


def _time_call(fn, *args, n_warmup: int = 1, n_runs: int = 5, **kwargs):
    for _ in range(n_warmup):
        out = fn(*args, **kwargs)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(n_runs):
        out = fn(*args, **kwargs)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    elapsed = (time.time() - t0) / n_runs
    return out, elapsed


def main() -> None:
    grid = Grid2D(n=N, extent=EXTENT, device=DEVICE)
    X, Y = grid.meshgrid()

    # Ground truth (analytic Gaussian)
    A_truth_fn = gaussian_analytic_solution(sigma=SIGMA, D=D)
    A_truth = A_truth_fn(X, Y, T_END)
    A0_grid = A_truth_fn(X, Y, 0.0)
    A0_fn = gaussian_initial_condition(SIGMA)

    # FD
    A_fd, t_fd = _time_call(diffuse_fd, A0_grid, D=D, t_end=T_END, dx=grid.dx,
                             cfl_safety=0.5, n_runs=3)

    # FFT
    A_fft, t_fft = _time_call(diffuse_fft, A0_grid, grid, D, T_END, n_runs=20)

    # PINN
    pinn = PINNDiffusion(
        D=D, hard_ic=True, A0_callable=A0_fn,
        hidden=128, n_hidden_layers=4,
        n_fourier=32, fourier_scale=2.5, activation="tanh", seed=0,
        x_range=(-EXTENT / 2.0, EXTENT / 2.0),
        t_range=(0.0, T_END),
    ).to(DEVICE)

    print("training PINN...")
    # With hard_ic=True the IC is exact at t=0 by construction, so only
    # the PDE residual needs training pressure.
    history = train_pinn_diffusion(
        pinn, A0_callable=A0_fn, extent=EXTENT, t_end=T_END,
        n_iters=10000, lr=1e-3,
        n_collocation=4096, n_ic=128, n_ic_grid_side=8,
        weight_pde=1.0, weight_ic=0.0,  # IC is hard-constrained
        lr_decay_step=4000, lr_decay_gamma=0.5,
        log_every=100, seed=0,
    )
    print(f"  trained in {history.train_time_sec:.1f} s, "
          f"final total loss {history.loss_total[-1]:.4e}")

    A_pinn, t_pinn_inf = _time_call(pinn_to_grid, pinn, grid, T_END, n_runs=20)

    rows = [["solver", "mse_vs_analytic", "max_abs_err_vs_analytic",
             "wallclock_per_call_ms", "extra_info"]]
    rows.append(["analytic", f"{_mse(A_truth, A_truth):.4e}",
                 f"{_max_abs_err(A_truth, A_truth):.4e}",
                 "0.000", ""])
    rows.append(["FD", f"{_mse(A_fd, A_truth):.4e}",
                 f"{_max_abs_err(A_fd, A_truth):.4e}",
                 f"{t_fd * 1000.0:.3f}",
                 f"dx={grid.dx:.4f}, periodic BC"])
    rows.append(["FFT", f"{_mse(A_fft, A_truth):.4e}",
                 f"{_max_abs_err(A_fft, A_truth):.4e}",
                 f"{t_fft * 1000.0:.3f}",
                 "exact heat kernel"])
    rows.append(["PINN", f"{_mse(A_pinn, A_truth):.4e}",
                 f"{_max_abs_err(A_pinn, A_truth):.4e}",
                 f"{t_pinn_inf * 1000.0:.3f}",
                 f"train={history.train_time_sec:.1f}s"])

    fig = show_pinn_vs_solvers(
        A_truth, A_fd, A_fft, A_pinn, extent=grid.extent,
        suptitle=(f"phase 6   diffusion of Gaussian sigma={SIGMA}   "
                  f"D={D}   t={T_END}   grid n={N}, extent={EXTENT}"),
    )
    out = save_figure(fig, OUT_FIG / "phase6_pinn_vs_solvers.png")
    print(f"  wrote {out}")

    fig = show_pinn_training(history.to_dicts(),
                             suptitle="phase 6 PINN training")
    out = save_figure(fig, OUT_FIG / "phase6_pinn_training.png")
    print(f"  wrote {out}")

    OUT_LOG.mkdir(parents=True, exist_ok=True)
    log = OUT_LOG / "phase6_metrics.csv"
    with open(log, "w", newline="") as f:
        csv.writer(f).writerows(rows)
    print(f"  wrote {log}")

    print()
    print("metrics:")
    widths = [max(len(str(r[c])) for r in rows) for c in range(len(rows[0]))]
    for r in rows:
        print("  " + "  ".join(str(v).ljust(w) for v, w in zip(r, widths)))


if __name__ == "__main__":
    main()

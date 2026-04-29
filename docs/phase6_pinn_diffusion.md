# Phase 6 — PINN for the 2D heat equation

**Status:** ✅ done
**PR:** #8 · 14 tests added (105 / 105 total green)

## Goal

The lab's **first ML phase**. Train a Physics-Informed Neural Network
on the same Gaussian-IC heat equation that Phase 5's FD and FFT
solvers already validate, then compare all four (analytic / FD / FFT /
PINN) against a closed-form Gaussian.

## What landed

- `src/pinn/pinn_base.py` — `FourierFeatures` (frozen random sin / cos
  projection), `MLP`, `PINNBase` with input normalization to
  `[-1, 1]³` against configurable `x_range` / `t_range`.
- `src/pinn/pinn_diffusion.py`:
  - `PINNDiffusion(D, hard_ic=True, A0_callable=...)`. With
    `hard_ic=True`, the network output is built as
    `A_pinn = A0(x, y) + t * MLP(x, y, t)`, so the IC is exact at
    `t = 0` by construction.
  - `train_pinn_diffusion` runs Adam + step LR scheduler and combines
    a regular IC grid with a random splash.
  - Helpers: `pinn_to_grid`, `gaussian_initial_condition`,
    `gaussian_analytic_solution`.
- `src/pinn/pinn_reaction_diffusion.py` — coupled `(A, Q)` PINN with
  `pde_residuals`. Training loop deferred (no analytic baseline).
- Visualization: `show_pinn_vs_solvers` (2 × 4 solutions / errors),
  `show_pinn_training`.

## Demo

```bash
python experiments/05_pinn_diffusion/compare_fd_pinn.py
```

Saves `phase6_pinn_vs_solvers.png`, `phase6_pinn_training.png`,
`phase6_metrics.csv`.

## Verified results

`sigma=0.5, D=0.1, t_end=1, n=128, extent=8`:

| solver | MSE | max abs err | wall-clock per call |
|---|---|---|---|
| analytic | 0 | 0 | — |
| FD | 1.33e-10 | 4.86e-05 | 7.27 ms |
| FFT | 3.14e-16 | 1.79e-07 | 0.10 ms |
| PINN | 4.24e-04 | 1.71e-01 | 0.37 ms (after ~80 s training) |

PINN is **3 OOM worse than FD and 6 OOM worse than FFT**, costing an
80-second training tax for an inference cost that doesn't beat FD.

## Two design tricks needed to make the PINN even tractable

1. **Input normalization.** With raw physical coordinates (extent
   up to 8), Fourier features at any usable scale produce encodings
   oscillating dozens of cycles across the domain, and a small MLP
   cannot integrate that smoothly to fit a localized Gaussian.
   Normalizing inside `forward` gives the encoding a uniform
   interpretation.
2. **Hard initial condition.** A vanilla soft-IC PINN on a localized
   Gaussian falls into a trivial minimum where the network outputs
   ~0 for `t > 0` (random IC sampling is dominated by points where
   `A0 ≈ 0`, so the soft-IC penalty is satisfied while the PDE
   residual stalls at ~5e-2). The architectural fix
   `A_pinn = A0 + t * MLP` makes the IC exact at `t = 0` by
   construction, and all training pressure flows onto the dynamics.

## Key takeaway

The headline finding from study plan §6.8: on closed-form-tractable
diffusion the PINN loses to grid solvers on **both** accuracy and
speed. The PINN's value is on irregular geometries, inverse parameter
estimation, and mesh-free continuous representation — not on this
benchmark. Phase 8 moves the operator-learning torch to FNO instead.

## See also

- [PROGRESS.md §A.9](../PROGRESS.md)
- Source: [src/pinn/](../src/pinn/)

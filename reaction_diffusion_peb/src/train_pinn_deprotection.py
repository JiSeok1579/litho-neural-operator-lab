"""Training loop for the Phase-5 ``PINNDeprotection`` model.

Sums the H and P PDE residual MSEs and runs Adam (or AdamW) on the
combined loss. ``hard_ic=True`` makes the soft-IC term redundant by
construction; the trainer still keeps a ``weight_ic`` knob in case
someone wants a soft-IC sanity check.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable

import torch

from reaction_diffusion_peb.src.pinn_reaction_diffusion import PINNDeprotection


@dataclass
class DeprotectionTrainingHistory:
    iters: list[int]
    loss_total: list[float]
    loss_pde_H: list[float]
    loss_pde_P: list[float]
    loss_ic: list[float]
    train_time_sec: float


def train_pinn_deprotection(
    pinn: PINNDeprotection,
    H0_callable: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    x_range_nm: tuple[float, float],
    t_range_s: tuple[float, float],
    n_iters: int = 5000,
    lr: float = 1.0e-3,
    weight_decay: float = 0.0,
    n_collocation: int = 4096,
    n_ic: int = 512,
    n_ic_grid_side: int = 16,
    weight_pde_H: float = 1.0,
    weight_pde_P: float = 1.0,
    weight_ic: float = 0.0,
    lr_decay_step: int | None = None,
    lr_decay_gamma: float = 0.5,
    log_every: int = 50,
    seed: int = 0,
) -> DeprotectionTrainingHistory:
    if n_iters < 1:
        raise ValueError("n_iters must be >= 1")
    device = next(pinn.parameters()).device
    pinn.train()
    if weight_decay > 0:
        optim = torch.optim.AdamW(pinn.parameters(), lr=lr, weight_decay=weight_decay)
    else:
        optim = torch.optim.Adam(pinn.parameters(), lr=lr)
    if lr_decay_step is not None:
        scheduler = torch.optim.lr_scheduler.StepLR(
            optim, step_size=int(lr_decay_step), gamma=float(lr_decay_gamma)
        )
    else:
        scheduler = None
    g_cpu = torch.Generator(device="cpu").manual_seed(seed)

    x_low, x_high = x_range_nm
    t_low, t_high = t_range_s
    extent_x = x_high - x_low
    extent_t = t_high - t_low

    iters: list[int] = []
    losses_total: list[float] = []
    losses_pde_H: list[float] = []
    losses_pde_P: list[float] = []
    losses_ic: list[float] = []

    side = torch.linspace(x_low, x_high, n_ic_grid_side)
    XX, YY = torch.meshgrid(side, side, indexing="xy")
    x_ic_grid = XX.flatten().to(device)
    y_ic_grid = YY.flatten().to(device)
    t_ic_grid = torch.full_like(x_ic_grid, t_low)
    H0_grid_truth = H0_callable(x_ic_grid, y_ic_grid).to(device)

    t0 = time.time()
    for it in range(n_iters):
        x_col = (x_low + torch.rand(n_collocation, generator=g_cpu) * extent_x
                 ).to(device).requires_grad_(True)
        y_col = (x_low + torch.rand(n_collocation, generator=g_cpu) * extent_x
                 ).to(device).requires_grad_(True)
        t_col = (t_low + torch.rand(n_collocation, generator=g_cpu) * extent_t
                 ).to(device).requires_grad_(True)
        r_H, r_P = pinn.pde_residuals(x_col, y_col, t_col)
        loss_pde_H = (r_H ** 2).mean()
        loss_pde_P = (r_P ** 2).mean()

        # Sanity-only IC penalty (zero weight by default).
        x_ic_rand = (x_low + torch.rand(n_ic, generator=g_cpu) * extent_x).to(device)
        y_ic_rand = (x_low + torch.rand(n_ic, generator=g_cpu) * extent_x).to(device)
        t_ic_rand = torch.full_like(x_ic_rand, t_low)
        H0_rand = H0_callable(x_ic_rand, y_ic_rand).to(device)

        x_ic = torch.cat([x_ic_grid, x_ic_rand])
        y_ic = torch.cat([y_ic_grid, y_ic_rand])
        t_ic = torch.cat([t_ic_grid, t_ic_rand])
        H0_truth = torch.cat([H0_grid_truth, H0_rand])
        H_pred_ic, P_pred_ic = pinn(x_ic, y_ic, t_ic)
        loss_ic_H = ((H_pred_ic - H0_truth) ** 2).mean()
        loss_ic_P = (P_pred_ic ** 2).mean()
        loss_ic = loss_ic_H + loss_ic_P

        loss = (weight_pde_H * loss_pde_H
                + weight_pde_P * loss_pde_P
                + weight_ic * loss_ic)
        optim.zero_grad(set_to_none=True)
        loss.backward()
        optim.step()
        if scheduler is not None:
            scheduler.step()

        if (it % log_every == 0) or (it == n_iters - 1):
            iters.append(it)
            losses_total.append(float(loss.item()))
            losses_pde_H.append(float(loss_pde_H.item()))
            losses_pde_P.append(float(loss_pde_P.item()))
            losses_ic.append(float(loss_ic.item()))

    if device.type == "cuda":
        torch.cuda.synchronize()
    train_time = time.time() - t0

    return DeprotectionTrainingHistory(
        iters=iters,
        loss_total=losses_total,
        loss_pde_H=losses_pde_H,
        loss_pde_P=losses_pde_P,
        loss_ic=losses_ic,
        train_time_sec=train_time,
    )


def pinn_deprotection_to_grid(
    pinn: PINNDeprotection,
    grid_size: int,
    dx_nm: float,
    t_s: float,
    device: torch.device | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Evaluate ``pinn`` on a regular grid and return ``(H, P)``."""
    if device is None:
        device = next(pinn.parameters()).device
    pinn.eval()
    coord = (torch.arange(grid_size, dtype=torch.float32, device=device)
             - grid_size / 2.0) * dx_nm
    X, Y = torch.meshgrid(coord, coord, indexing="xy")
    Tt = torch.full_like(X, float(t_s))
    with torch.no_grad():
        H_flat, P_flat = pinn(X.flatten(), Y.flatten(), Tt.flatten())
    return H_flat.view(grid_size, grid_size), P_flat.view(grid_size, grid_size)

"""PINN for the 2D diffusion equation.

Solves

    dA/dt - D * (d2A/dx2 + d2A/dy2) = 0

over ``(x, y, t) in [-extent/2, extent/2]^2 x [0, t_end]`` by minimizing

    L = w_pde * mean(residual^2) + w_ic * mean((A_pred(t=0) - A0)^2)

The boundary condition is left implicit (the spatial domain is interior
to the training region). For the smooth Gaussian initial condition used
in the Phase-6 demo this is fine: the analytic solution decays well
before the box edge. For studies that demand explicit BCs, add a
boundary-residual term using the same ``pde_residual`` machinery.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable

import torch
from torch import nn

from src.common.grid import Grid2D
from src.pinn.pinn_base import PINNBase


@dataclass
class TrainingHistory:
    iters: list[int]
    loss_total: list[float]
    loss_pde: list[float]
    loss_ic: list[float]
    train_time_sec: float

    def to_dicts(self) -> list[dict]:
        return [
            {"iter": i, "loss_total": t, "loss_pde": p, "loss_ic": ic}
            for i, t, p, ic in zip(self.iters, self.loss_total, self.loss_pde, self.loss_ic)
        ]


class PINNDiffusion(PINNBase):
    """PINN that learns ``A(x, y, t)`` for a fixed diffusivity ``D``.

    With ``hard_ic=True`` (default), the initial condition is enforced
    architecturally:

        A_pinn(x, y, t) = A0(x, y) + t * MLP_output(x, y, t)

    so the network output exactly matches ``A0`` at ``t = 0`` regardless
    of weights. Training then only needs the PDE-residual loss term — the
    IC penalty becomes redundant (and is set to zero in the trainer
    by default when ``hard_ic`` is on).

    Without this trick, vanilla soft-IC PINNs on a localized 2D Gaussian
    fall into a trivial minimum where the network predicts ~0 for
    ``t > 0`` and only fits ``A0`` at ``t = 0``: most of the IC sample
    space is near zero anyway, so soft-IC is satisfied while the PDE
    residual stays moderate.
    """

    def __init__(self, D: float, hard_ic: bool = True,
                 A0_callable=None, **base_kwargs):
        super().__init__(**base_kwargs)
        if D < 0:
            raise ValueError("D must be non-negative")
        if hard_ic and A0_callable is None:
            raise ValueError("hard_ic=True requires an A0_callable")
        self.D = D
        self.hard_ic = hard_ic
        self.A0_callable = A0_callable

    def forward(self, x: torch.Tensor, y: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        if x.shape != y.shape or x.shape != t.shape:
            raise ValueError("x, y, t must share shape")
        x_n, y_n, t_n = self._normalize_xyt(x, y, t)
        xyt = torch.stack([x_n, y_n, t_n], dim=-1)
        h = self.fourier(xyt)
        raw = self.mlp(h).squeeze(-1)
        if self.hard_ic:
            base = self.A0_callable(x, y)
            return base + t * raw
        return raw

    def pde_residual(self, x: torch.Tensor, y: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Return ``dA/dt - D * (d2A/dx2 + d2A/dy2)`` at the collocation
        points. Inputs must be 1D tensors with ``requires_grad=True``."""
        if not (x.requires_grad and y.requires_grad and t.requires_grad):
            raise RuntimeError("collocation inputs must have requires_grad=True")
        A = self(x, y, t)
        ones = torch.ones_like(A)
        A_t = torch.autograd.grad(A, t, grad_outputs=ones,
                                  create_graph=True, retain_graph=True)[0]
        A_x = torch.autograd.grad(A, x, grad_outputs=ones,
                                  create_graph=True, retain_graph=True)[0]
        A_y = torch.autograd.grad(A, y, grad_outputs=ones,
                                  create_graph=True, retain_graph=True)[0]
        ones_x = torch.ones_like(A_x)
        A_xx = torch.autograd.grad(A_x, x, grad_outputs=ones_x,
                                   create_graph=True, retain_graph=True)[0]
        A_yy = torch.autograd.grad(A_y, y, grad_outputs=ones_x,
                                   create_graph=True)[0]
        return A_t - self.D * (A_xx + A_yy)


def train_pinn_diffusion(
    pinn: PINNDiffusion,
    A0_callable: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    extent: float,
    t_end: float,
    n_iters: int = 3000,
    lr: float = 1.0e-3,
    n_collocation: int = 4096,
    n_ic: int = 2048,
    n_ic_grid_side: int | None = None,
    weight_pde: float = 1.0,
    weight_ic: float = 100.0,
    lr_decay_step: int | None = None,
    lr_decay_gamma: float = 0.5,
    log_every: int = 100,
    seed: int = 0,
) -> TrainingHistory:
    """Train ``pinn`` in place; return a :class:`TrainingHistory`.

    The initial-condition points are sampled on a regular ``(n_ic_grid_side)^2``
    grid combined with ``n_ic`` random points. The grid component prevents
    a degenerate solution where the network fits the bulk of the IC space
    (which is near zero for a localized Gaussian) by predicting zero
    everywhere — a trivial minimum that random sampling alone struggles
    to escape.

    ``lr_decay_step`` enables a step LR schedule (Adam lr * gamma every
    ``step`` iters); pass ``None`` for a constant LR.
    """
    if n_iters < 1:
        raise ValueError("n_iters must be >= 1")
    device = next(pinn.parameters()).device
    pinn.train()
    optim = torch.optim.Adam(pinn.parameters(), lr=lr)
    if lr_decay_step is not None:
        scheduler = torch.optim.lr_scheduler.StepLR(
            optim, step_size=int(lr_decay_step), gamma=float(lr_decay_gamma)
        )
    else:
        scheduler = None
    g_cpu = torch.Generator(device="cpu").manual_seed(seed)

    iters: list[int] = []
    losses_total: list[float] = []
    losses_pde: list[float] = []
    losses_ic: list[float] = []

    # Pre-build a regular IC grid (deterministic, full coverage).
    if n_ic_grid_side is None:
        n_ic_grid_side = max(16, int(round(extent * 8)))  # ~8 points per length unit
    side = torch.linspace(-extent / 2.0, extent / 2.0, n_ic_grid_side)
    XX, YY = torch.meshgrid(side, side, indexing="xy")
    x_ic_grid = XX.flatten().to(device)
    y_ic_grid = YY.flatten().to(device)
    t_ic_grid = torch.zeros_like(x_ic_grid)
    A0_grid_truth = A0_callable(x_ic_grid, y_ic_grid).to(device)

    t0 = time.time()
    for it in range(n_iters):
        # PDE collocation points (random)
        x_col = ((torch.rand(n_collocation, generator=g_cpu) - 0.5) * extent).to(device).requires_grad_(True)
        y_col = ((torch.rand(n_collocation, generator=g_cpu) - 0.5) * extent).to(device).requires_grad_(True)
        t_col = (torch.rand(n_collocation, generator=g_cpu) * t_end).to(device).requires_grad_(True)
        residual = pinn.pde_residual(x_col, y_col, t_col)
        loss_pde = (residual ** 2).mean()

        # IC points: regular grid + a random splash
        x_ic_rand = ((torch.rand(n_ic, generator=g_cpu) - 0.5) * extent).to(device)
        y_ic_rand = ((torch.rand(n_ic, generator=g_cpu) - 0.5) * extent).to(device)
        t_ic_rand = torch.zeros(n_ic, device=device)
        A0_rand = A0_callable(x_ic_rand, y_ic_rand).to(device)

        x_ic = torch.cat([x_ic_grid, x_ic_rand])
        y_ic = torch.cat([y_ic_grid, y_ic_rand])
        t_ic = torch.cat([t_ic_grid, t_ic_rand])
        A0_truth = torch.cat([A0_grid_truth, A0_rand])
        A_pred = pinn(x_ic, y_ic, t_ic)
        loss_ic = ((A_pred - A0_truth) ** 2).mean()

        loss = weight_pde * loss_pde + weight_ic * loss_ic
        optim.zero_grad(set_to_none=True)
        loss.backward()
        optim.step()
        if scheduler is not None:
            scheduler.step()

        if (it % log_every == 0) or (it == n_iters - 1):
            iters.append(it)
            losses_total.append(float(loss.item()))
            losses_pde.append(float(loss_pde.item()))
            losses_ic.append(float(loss_ic.item()))

    if device.type == "cuda":
        torch.cuda.synchronize()
    train_time = time.time() - t0

    return TrainingHistory(
        iters=iters, loss_total=losses_total, loss_pde=losses_pde,
        loss_ic=losses_ic, train_time_sec=train_time,
    )


def pinn_to_grid(pinn: PINNBase, grid: Grid2D, t: float) -> torch.Tensor:
    """Evaluate ``pinn`` at every grid pixel at the given time."""
    pinn.eval()
    X, Y = grid.meshgrid()
    Tt = torch.full_like(X, float(t))
    with torch.no_grad():
        A_flat = pinn(X.flatten(), Y.flatten(), Tt.flatten())
    return A_flat.view(grid.n, grid.n)


def gaussian_initial_condition(sigma: float):
    """Return a callable ``A0(x, y) = exp(-(x^2+y^2) / (2 sigma^2))``."""
    if sigma <= 0:
        raise ValueError("sigma must be positive")

    def A0(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return torch.exp(-(x ** 2 + y ** 2) / (2.0 * sigma ** 2))

    return A0


def gaussian_analytic_solution(sigma: float, D: float):
    """Closed-form 2D heat-equation solution for a Gaussian initial state.

    A(x, y, t) = sigma^2 / (sigma^2 + 2 D t) * exp(-(x^2 + y^2) /
                 (2 (sigma^2 + 2 D t)))

    Useful as ground truth when checking PINN / FD / FFT solvers.
    """
    if sigma <= 0 or D < 0:
        raise ValueError("sigma > 0 and D >= 0 required")

    def A(x: torch.Tensor, y: torch.Tensor, t: float) -> torch.Tensor:
        s2 = sigma * sigma + 2.0 * D * t
        amp = (sigma * sigma) / s2
        return amp * torch.exp(-(x ** 2 + y ** 2) / (2.0 * s2))

    return A

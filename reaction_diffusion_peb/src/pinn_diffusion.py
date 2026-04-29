"""PINN for the PEB-submodule diffusion equation.

Solves

    dH/dt - D_H * laplacian(H) = 0

over ``(x, y, t) in x_range_nm x x_range_nm x t_range_s`` by minimizing

    L = w_pde * mean(residual^2) + w_ic * mean((H_pred(t=t_low) - H_0)^2)

With ``hard_ic=True`` (default) the initial condition is enforced
**architecturally**:

    H_pinn(x, y, t) = H_0(x, y) + ((t - t_low) / (t_high - t_low)) * MLP(...)

so the network output is exactly ``H_0(x, y)`` at ``t = t_low`` regardless
of weights and the soft-IC penalty becomes redundant. This was the trick
that escaped the trivial soft-IC local minimum in the main project's
Phase 6.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Callable

import torch

from reaction_diffusion_peb.src.pinn_base import PINNBase


@dataclass
class TrainingHistory:
    iters: list[int]
    loss_total: list[float]
    loss_pde: list[float]
    loss_ic: list[float]
    train_time_sec: float


class PINNDiffusion(PINNBase):
    """PINN that learns ``H(x, y, t)`` for a fixed diffusivity ``DH``.

    With ``hard_ic=True`` (default) the network output is

        H_pinn(x, y, t) = H_0(x, y) + t_norm * MLP(x, y, t)

    where ``t_norm = (t - t_low) / (t_high - t_low) in [0, 1]`` so the
    IC is exact at ``t = t_low``.
    """

    def __init__(self, DH_nm2_s: float, hard_ic: bool = True,
                 H0_callable=None, **base_kwargs):
        super().__init__(**base_kwargs)
        if DH_nm2_s < 0:
            raise ValueError("DH_nm2_s must be non-negative")
        if hard_ic and H0_callable is None:
            raise ValueError("hard_ic=True requires an H0_callable")
        self.DH_nm2_s = DH_nm2_s
        self.hard_ic = hard_ic
        self.H0_callable = H0_callable

    def forward(self, x: torch.Tensor, y: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        if x.shape != y.shape or x.shape != t.shape:
            raise ValueError("x, y, t must share shape")
        x_n, y_n, t_n = self._normalize_xyt(x, y, t)
        xyt = torch.stack([x_n, y_n, t_n], dim=-1)
        h = self.fourier(xyt)
        raw = self.mlp(h).squeeze(-1)
        if self.hard_ic:
            base = self.H0_callable(x, y)
            t_factor = (t - self._t_low) / (self._t_high - self._t_low)
            return base + t_factor * raw
        return raw

    def pde_residual(self, x: torch.Tensor, y: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Return ``dH/dt - DH * (d2H/dx2 + d2H/dy2)`` at the collocation
        points. Inputs must be 1D tensors with ``requires_grad=True``."""
        if not (x.requires_grad and y.requires_grad and t.requires_grad):
            raise RuntimeError("collocation inputs must have requires_grad=True")
        H = self(x, y, t)
        ones = torch.ones_like(H)
        H_t = torch.autograd.grad(H, t, grad_outputs=ones,
                                  create_graph=True, retain_graph=True)[0]
        H_x = torch.autograd.grad(H, x, grad_outputs=ones,
                                  create_graph=True, retain_graph=True)[0]
        H_y = torch.autograd.grad(H, y, grad_outputs=ones,
                                  create_graph=True, retain_graph=True)[0]
        ones_x = torch.ones_like(H_x)
        H_xx = torch.autograd.grad(H_x, x, grad_outputs=ones_x,
                                   create_graph=True, retain_graph=True)[0]
        H_yy = torch.autograd.grad(H_y, y, grad_outputs=ones_x,
                                   create_graph=True)[0]
        return H_t - self.DH_nm2_s * (H_xx + H_yy)


def gaussian_spot_acid_callable(
    sigma_nm: float,
    Hmax: float = 0.2,
    eta: float = 1.0,
    dose: float = 1.0,
    center_nm: tuple[float, float] = (0.0, 0.0),
):
    """Closed-form initial-acid field for a Gaussian-spot exposure.

    Reproduces:
        I(x, y) = exp(-((x - cx)^2 + (y - cy)^2) / (2 * sigma_nm^2))
        H_0(x, y) = Hmax * (1 - exp(-eta * dose * I(x, y)))

    Returns a callable ``H0(x_nm, y_nm) -> torch.Tensor`` suitable for
    use as the PINN's ``H0_callable``.
    """
    cx, cy = center_nm

    def H0(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        I = torch.exp(-((x - cx) ** 2 + (y - cy) ** 2) / (2.0 * sigma_nm ** 2))
        return Hmax * (1.0 - torch.exp(-eta * dose * I))

    return H0


def train_pinn_diffusion(
    pinn: PINNDiffusion,
    H0_callable: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    x_range_nm: tuple[float, float],
    t_range_s: tuple[float, float],
    n_iters: int = 5000,
    lr: float = 1.0e-3,
    weight_decay: float = 0.0,
    n_collocation: int = 4096,
    n_ic: int = 512,
    n_ic_grid_side: int = 16,
    weight_pde: float = 1.0,
    weight_ic: float = 0.0,
    lr_decay_step: int | None = None,
    lr_decay_gamma: float = 0.5,
    log_every: int = 50,
    seed: int = 0,
) -> TrainingHistory:
    """Train ``pinn`` in place; return a :class:`TrainingHistory`.

    With ``hard_ic=True`` on the model, ``weight_ic`` should stay 0 —
    the IC is already exact and the random IC samples below are just
    a cheap sanity guard.
    """
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
    losses_pde: list[float] = []
    losses_ic: list[float] = []

    # Pre-build a regular IC grid for cheap consistent coverage.
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
        residual = pinn.pde_residual(x_col, y_col, t_col)
        loss_pde = (residual ** 2).mean()

        # Optional soft-IC term (irrelevant when hard_ic=True; included
        # so the trainer can also drive a vanilla soft-IC model).
        x_ic_rand = (x_low + torch.rand(n_ic, generator=g_cpu) * extent_x).to(device)
        y_ic_rand = (x_low + torch.rand(n_ic, generator=g_cpu) * extent_x).to(device)
        t_ic_rand = torch.full_like(x_ic_rand, t_low)
        H0_rand = H0_callable(x_ic_rand, y_ic_rand).to(device)

        x_ic = torch.cat([x_ic_grid, x_ic_rand])
        y_ic = torch.cat([y_ic_grid, y_ic_rand])
        t_ic = torch.cat([t_ic_grid, t_ic_rand])
        H0_truth = torch.cat([H0_grid_truth, H0_rand])
        H_pred_ic = pinn(x_ic, y_ic, t_ic)
        loss_ic = ((H_pred_ic - H0_truth) ** 2).mean()

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


def pinn_to_grid(
    pinn: PINNBase,
    grid_size: int,
    dx_nm: float,
    t_s: float,
    device: torch.device | None = None,
) -> torch.Tensor:
    """Evaluate ``pinn`` at every grid pixel at the given physical time.

    Coordinates are in nm with extent ``grid_size * dx_nm``, centered on
    zero. Returns a ``(grid_size, grid_size)`` tensor.
    """
    if device is None:
        device = next(pinn.parameters()).device
    pinn.eval()
    coord = (torch.arange(grid_size, dtype=torch.float32, device=device)
             - grid_size / 2.0) * dx_nm
    X, Y = torch.meshgrid(coord, coord, indexing="xy")
    Tt = torch.full_like(X, float(t_s))
    with torch.no_grad():
        H_flat = pinn(X.flatten(), Y.flatten(), Tt.flatten())
    return H_flat.view(grid_size, grid_size)

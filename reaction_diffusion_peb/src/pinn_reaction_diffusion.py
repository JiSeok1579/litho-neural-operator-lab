"""PINNs for the PEB reaction-diffusion family.

Phase 4 adds the acid-loss term:

    dH/dt - D_H laplacian(H) + k_loss * H = 0

The PINN residual is

    r(x, y, t) = dH/dt - D_H * (d^2H/dx^2 + d^2H/dy^2) + k_loss * H

which is just :class:`PINNDiffusion`'s residual plus the linear loss.
We expose this as ``PINNDiffusionLoss`` — a thin subclass that keeps
the input normalization, hard-IC parameterization, and Fourier
features unchanged.

Phase 5 adds a second output channel ``P`` (deprotected fraction) and
a second PDE residual:

    dP/dt - k_dep * H * (1 - P) = 0

We expose this as :class:`PINNDeprotection`. The H equation is the
same as in Phase 4 (the H residual is unchanged), and ``P`` joins H
as a second output of the underlying MLP. Hard IC enforces
``H(0) = H_0`` and ``P(0) = 0`` exactly.
"""

from __future__ import annotations

import torch
from torch import nn

from reaction_diffusion_peb.src.pinn_base import PINNBase
from reaction_diffusion_peb.src.pinn_diffusion import PINNDiffusion


class PINNDiffusionLoss(PINNDiffusion):
    """PINN for ``dH/dt = D_H laplacian(H) - k_loss * H``.

    Inherits all the input-normalization, hard-IC, and forward / pde
    machinery from :class:`PINNDiffusion`; only ``pde_residual`` is
    overridden to add the linear loss term.
    """

    def __init__(
        self,
        DH_nm2_s: float,
        kloss_s_inv: float,
        hard_ic: bool = True,
        H0_callable=None,
        **base_kwargs,
    ):
        super().__init__(
            DH_nm2_s=DH_nm2_s, hard_ic=hard_ic, H0_callable=H0_callable,
            **base_kwargs,
        )
        if kloss_s_inv < 0:
            raise ValueError("kloss_s_inv must be non-negative")
        self.kloss_s_inv = kloss_s_inv

    def pde_residual(self, x: torch.Tensor, y: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
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
        return H_t - self.DH_nm2_s * (H_xx + H_yy) + self.kloss_s_inv * H


class PINNDeprotection(PINNBase):
    """PINN for the coupled (H, P) deprotection system.

    PDEs:
        dH/dt = D_H * laplacian(H) - k_loss * H
        dP/dt = k_dep * H * (1 - P)

    Initial conditions (enforced architecturally with ``hard_ic=True``):
        H(x, y, 0) = H_0(x, y)
        P(x, y, 0) = 0

    Parameterization with ``t_norm = (t - t_low) / (t_high - t_low)``:
        H_pinn = H_0(x, y) + t_norm * MLP_out[..., 0]
        P_pinn = 0          + t_norm * MLP_out[..., 1]
    """

    def __init__(
        self,
        DH_nm2_s: float,
        kloss_s_inv: float,
        kdep_s_inv: float,
        hard_ic: bool = True,
        H0_callable=None,
        **base_kwargs,
    ):
        super().__init__(**base_kwargs)
        for name, val in (("DH_nm2_s", DH_nm2_s),
                          ("kloss_s_inv", kloss_s_inv),
                          ("kdep_s_inv", kdep_s_inv)):
            if val < 0:
                raise ValueError(f"{name} must be non-negative")
        if hard_ic and H0_callable is None:
            raise ValueError("hard_ic=True requires an H0_callable")
        self.DH_nm2_s = DH_nm2_s
        self.kloss_s_inv = kloss_s_inv
        self.kdep_s_inv = kdep_s_inv
        self.hard_ic = hard_ic
        self.H0_callable = H0_callable

        # Replace the single-output head with a two-output head.
        body_out = self.mlp.net[-1].in_features
        self.mlp.net[-1] = nn.Linear(body_out, 2)

    def forward(self, x: torch.Tensor, y: torch.Tensor, t: torch.Tensor):
        """Return ``(H_pred, P_pred)``."""
        if x.shape != y.shape or x.shape != t.shape:
            raise ValueError("x, y, t must share shape")
        x_n, y_n, t_n = self._normalize_xyt(x, y, t)
        xyt = torch.stack([x_n, y_n, t_n], dim=-1)
        h = self.fourier(xyt)
        raw = self.mlp(h)  # (..., 2)
        if self.hard_ic:
            t_factor = (t - self._t_low) / (self._t_high - self._t_low)
            H_base = self.H0_callable(x, y)
            H_pred = H_base + t_factor * raw[..., 0]
            P_pred = t_factor * raw[..., 1]
        else:
            H_pred = raw[..., 0]
            P_pred = raw[..., 1]
        return H_pred, P_pred

    def pde_residuals(
        self, x: torch.Tensor, y: torch.Tensor, t: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return ``(r_H, r_P)`` at the collocation points."""
        if not (x.requires_grad and y.requires_grad and t.requires_grad):
            raise RuntimeError("collocation inputs must have requires_grad=True")
        H, P = self(x, y, t)
        ones = torch.ones_like(H)

        # H residual (Phase-4 form)
        H_t = torch.autograd.grad(H, t, grad_outputs=ones, create_graph=True, retain_graph=True)[0]
        H_x = torch.autograd.grad(H, x, grad_outputs=ones, create_graph=True, retain_graph=True)[0]
        H_y = torch.autograd.grad(H, y, grad_outputs=ones, create_graph=True, retain_graph=True)[0]
        ones_x = torch.ones_like(H_x)
        H_xx = torch.autograd.grad(H_x, x, grad_outputs=ones_x, create_graph=True, retain_graph=True)[0]
        H_yy = torch.autograd.grad(H_y, y, grad_outputs=ones_x, create_graph=True, retain_graph=True)[0]
        r_H = H_t - self.DH_nm2_s * (H_xx + H_yy) + self.kloss_s_inv * H

        # P residual
        P_t = torch.autograd.grad(P, t, grad_outputs=ones, create_graph=True)[0]
        r_P = P_t - self.kdep_s_inv * H * (1.0 - P)

        return r_H, r_P

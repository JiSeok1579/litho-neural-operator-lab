"""PINNs for the PEB reaction-diffusion family.

Phase 4 adds the acid-loss term:

    dH/dt - D_H laplacian(H) + k_loss * H = 0

The PINN residual is

    r(x, y, t) = dH/dt - D_H * (d^2H/dx^2 + d^2H/dy^2) + k_loss * H

which is just :class:`PINNDiffusion`'s residual plus the linear loss.
We expose this as ``PINNDiffusionLoss`` — a thin subclass that keeps
the input normalization, hard-IC parameterization, and Fourier
features unchanged.
"""

from __future__ import annotations

import torch

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

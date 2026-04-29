"""PINN for the coupled acid-quencher reaction-diffusion system.

System:

    dA/dt = D * (d2A/dx2 + d2A/dy2) - k * A * Q
    dQ/dt = -k * A * Q

This module exposes the network and the residuals; a full training loop
is left for a follow-up phase. The Phase 6 demo focuses on pure
diffusion because the analytic ground truth (Gaussian) gives a clean
benchmark against FD / FFT. Reaction-diffusion has no closed-form
solution in general, so any comparison would have to use FD as the
reference, which makes the PINN-vs-FD assessment circular.
"""

from __future__ import annotations

import torch
from torch import nn

from src.pinn.pinn_base import PINNBase


class PINNReactionDiffusion(PINNBase):
    """PINN that learns ``(A, Q)(x, y, t)`` jointly.

    The output dimension is 2 (A and Q stacked along the last axis).
    """

    def __init__(self, D: float, k: float, **base_kwargs):
        super().__init__(**base_kwargs)
        if D < 0 or k < 0:
            raise ValueError("D and k must be non-negative")
        # Replace the single-output head with a two-output head.
        # Keeping the rest of the base class (Fourier features + body) intact.
        body_out = self.mlp.net[-1].in_features
        self.mlp.net[-1] = nn.Linear(body_out, 2)
        self.D = D
        self.k = k

    def forward(self, x: torch.Tensor, y: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        if x.shape != y.shape or x.shape != t.shape:
            raise ValueError("x, y, t must share shape")
        x_n, y_n, t_n = self._normalize_xyt(x, y, t)
        xyt = torch.stack([x_n, y_n, t_n], dim=-1)
        h = self.fourier(xyt)
        return self.mlp(h)  # (N, 2): [..., 0]=A, [..., 1]=Q

    def split(self, AQ: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return AQ[..., 0], AQ[..., 1]

    def pde_residuals(
        self, x: torch.Tensor, y: torch.Tensor, t: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the residuals of the A and Q equations at the
        collocation points. Inputs must have ``requires_grad=True``."""
        if not (x.requires_grad and y.requires_grad and t.requires_grad):
            raise RuntimeError("collocation inputs must have requires_grad=True")
        AQ = self(x, y, t)
        A, Q = self.split(AQ)
        ones = torch.ones_like(A)

        # Time derivatives
        A_t = torch.autograd.grad(A, t, grad_outputs=ones, create_graph=True, retain_graph=True)[0]
        Q_t = torch.autograd.grad(Q, t, grad_outputs=ones, create_graph=True, retain_graph=True)[0]

        # Spatial first derivatives of A only (Q does not diffuse)
        A_x = torch.autograd.grad(A, x, grad_outputs=ones, create_graph=True, retain_graph=True)[0]
        A_y = torch.autograd.grad(A, y, grad_outputs=ones, create_graph=True, retain_graph=True)[0]
        ones_x = torch.ones_like(A_x)
        A_xx = torch.autograd.grad(A_x, x, grad_outputs=ones_x, create_graph=True, retain_graph=True)[0]
        A_yy = torch.autograd.grad(A_y, y, grad_outputs=ones_x, create_graph=True)[0]

        AQ_term = self.k * A * Q
        r_A = A_t - self.D * (A_xx + A_yy) + AQ_term
        r_Q = Q_t + AQ_term
        return r_A, r_Q

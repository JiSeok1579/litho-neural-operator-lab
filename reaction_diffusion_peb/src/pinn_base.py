"""PINN building blocks for the PEB submodule.

Self-contained — does not import from the main repo's ``src/pinn``.
The architecture follows the same recipe that worked in the main
project's Phase 6:

- Random Fourier feature encoding for ``(x, y, t)``.
- Tanh-MLP body (4 hidden layers x 64 wide by default).
- **Inputs are normalized to [-1, 1]^3 inside ``forward``** based on the
  configured ``x_range`` and ``t_range``. Without this the Fourier
  features at any usable scale oscillate too fast across a 128 nm
  domain for a small MLP to integrate, and a localized acid spot
  cannot be learned.

The autograd graph propagates the normalization correctly, so
``pde_residual`` computed against physical ``x, y, t`` (in nm and s)
automatically picks up the chain rule.
"""

from __future__ import annotations

import math

import torch
from torch import nn


class FourierFeatures(nn.Module):
    """Frozen random Fourier projection followed by sin / cos."""

    def __init__(self, in_dim: int, num_features: int = 16,
                 scale: float = 1.0, seed: int = 0):
        super().__init__()
        if in_dim < 1 or num_features < 1:
            raise ValueError("in_dim and num_features must be >= 1")
        if scale <= 0:
            raise ValueError("scale must be positive")
        g = torch.Generator().manual_seed(seed)
        B = scale * torch.randn(in_dim, num_features, generator=g)
        self.register_buffer("B", B)
        self.in_dim = in_dim
        self.num_features = num_features

    @property
    def out_dim(self) -> int:
        return 2 * self.num_features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        proj = 2.0 * math.pi * (x @ self.B)
        return torch.cat([torch.sin(proj), torch.cos(proj)], dim=-1)


class MLP(nn.Module):
    """Feed-forward MLP. Tanh by default for smooth PINN gradients."""

    def __init__(self, in_dim: int, out_dim: int = 1,
                 hidden: int = 64, n_hidden_layers: int = 4,
                 activation: str = "tanh"):
        super().__init__()
        act_map = {"tanh": nn.Tanh, "gelu": nn.GELU,
                   "silu": nn.SiLU, "relu": nn.ReLU}
        if activation not in act_map:
            raise ValueError(f"unknown activation {activation!r}")
        Act = act_map[activation]
        layers: list[nn.Module] = [nn.Linear(in_dim, hidden), Act()]
        for _ in range(n_hidden_layers - 1):
            layers.append(nn.Linear(hidden, hidden))
            layers.append(Act())
        layers.append(nn.Linear(hidden, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class PINNBase(nn.Module):
    """Maps ``(x, y, t) -> H``. PDE residuals are added by subclasses.

    Inputs are normalized to ``[-1, 1]^3`` inside ``forward`` based on
    the configured ``x_range`` (in nm) and ``t_range`` (in s). The
    autograd chain rule through the linear normalization ensures that
    PDE residuals computed against physical ``x, y, t`` come out with
    the right units.
    """

    def __init__(self, x_range_nm: tuple[float, float],
                 t_range_s: tuple[float, float],
                 hidden: int = 64, n_hidden_layers: int = 4,
                 n_fourier: int = 16, fourier_scale: float = 1.0,
                 activation: str = "tanh", seed: int = 0):
        super().__init__()
        if x_range_nm[0] >= x_range_nm[1]:
            raise ValueError("x_range_nm must be (low, high) with low < high")
        if t_range_s[0] >= t_range_s[1]:
            raise ValueError("t_range_s must be (low, high) with low < high")
        self.fourier = FourierFeatures(in_dim=3, num_features=n_fourier,
                                        scale=fourier_scale, seed=seed)
        self.mlp = MLP(in_dim=self.fourier.out_dim, out_dim=1,
                       hidden=hidden, n_hidden_layers=n_hidden_layers,
                       activation=activation)
        self.register_buffer("_x_low", torch.tensor(float(x_range_nm[0])))
        self.register_buffer("_x_high", torch.tensor(float(x_range_nm[1])))
        self.register_buffer("_t_low", torch.tensor(float(t_range_s[0])))
        self.register_buffer("_t_high", torch.tensor(float(t_range_s[1])))

    def _normalize_xyt(
        self, x: torch.Tensor, y: torch.Tensor, t: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x_n = 2.0 * (x - self._x_low) / (self._x_high - self._x_low) - 1.0
        y_n = 2.0 * (y - self._x_low) / (self._x_high - self._x_low) - 1.0
        t_n = 2.0 * (t - self._t_low) / (self._t_high - self._t_low) - 1.0
        return x_n, y_n, t_n

    def forward(self, x: torch.Tensor, y: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        if x.shape != y.shape or x.shape != t.shape:
            raise ValueError("x, y, t must share shape")
        x_n, y_n, t_n = self._normalize_xyt(x, y, t)
        xyt = torch.stack([x_n, y_n, t_n], dim=-1)
        h = self.fourier(xyt)
        return self.mlp(h).squeeze(-1)

"""Building blocks shared by every PINN in this lab.

Layout: ``(x, y, t) -> Fourier features -> MLP -> A``.

Random Fourier features are essential. A plain MLP on raw ``(x, y, t)``
suffers from spectral bias: high-frequency components of the target
field are learned much more slowly than low-frequency ones, so a 2D
diffusion problem with anything sharper than a wide Gaussian gets stuck
on the low-frequency contribution. Encoding inputs through random
Fourier projections lifts that bias and reaches usable accuracy in
~thousands of iterations rather than millions.
"""

from __future__ import annotations

import math

import torch
from torch import nn


class FourierFeatures(nn.Module):
    """Frozen random Fourier projection followed by sin/cos.

    For input ``x`` of shape ``(..., in_dim)`` the output has shape
    ``(..., 2 * num_features)`` formed by concatenating ``sin(2 pi x B)``
    and ``cos(2 pi x B)``. ``B`` is drawn once at construction with the
    given ``scale`` (a wider scale resolves higher frequencies but takes
    more training to fit).
    """

    def __init__(self, in_dim: int, num_features: int = 16, scale: float = 2.0,
                 seed: int = 0):
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
    """Simple feed-forward MLP. Activation defaults to tanh, which gives
    smooth gradients well-suited to PINN PDE residuals."""

    def __init__(self, in_dim: int, out_dim: int = 1, hidden: int = 64,
                 n_hidden_layers: int = 4, activation: str = "tanh"):
        super().__init__()
        act_map = {"tanh": nn.Tanh, "gelu": nn.GELU, "silu": nn.SiLU, "relu": nn.ReLU}
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
    """Maps ``(x, y, t)`` -> ``A``. PDE residuals are added by subclasses.

    Inputs are normalized to ``[-1, 1]^3`` inside :meth:`forward` based on
    the ``x_range`` and ``t_range`` provided at construction; the network
    therefore sees a fixed coordinate scale regardless of the physical
    domain size. The autograd graph picks up the (linear) chain rule for
    the normalization automatically, so PDE residuals computed against
    physical ``x, y, t`` come out with the right units.

    Without this normalization, a Fourier-feature encoding tuned for one
    domain size silently produces ill-scaled embeddings on a larger
    domain (the Fourier projections oscillate too fast for a small
    network to integrate), which is what makes naively-coded PINNs
    plateau on smooth targets.
    """

    def __init__(self, hidden: int = 64, n_hidden_layers: int = 4,
                 n_fourier: int = 16, fourier_scale: float = 1.0,
                 activation: str = "tanh", seed: int = 0,
                 x_range: tuple[float, float] = (-1.0, 1.0),
                 t_range: tuple[float, float] = (0.0, 1.0)):
        super().__init__()
        if x_range[0] >= x_range[1]:
            raise ValueError("x_range must be (low, high) with low < high")
        if t_range[0] >= t_range[1]:
            raise ValueError("t_range must be (low, high) with low < high")
        self.fourier = FourierFeatures(in_dim=3, num_features=n_fourier,
                                        scale=fourier_scale, seed=seed)
        self.mlp = MLP(in_dim=self.fourier.out_dim, out_dim=1,
                       hidden=hidden, n_hidden_layers=n_hidden_layers,
                       activation=activation)
        self.register_buffer("_x_low", torch.tensor(float(x_range[0])))
        self.register_buffer("_x_high", torch.tensor(float(x_range[1])))
        self.register_buffer("_t_low", torch.tensor(float(t_range[0])))
        self.register_buffer("_t_high", torch.tensor(float(t_range[1])))

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

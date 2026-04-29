"""Fourier Neural Operator (2D) implementation.

Reference: Li, Z. et al. "Fourier Neural Operator for Parametric Partial
Differential Equations" (ICLR 2021). Architecture:

    input  --(1x1 lift)-->  hidden channels
                           |
                           +-> N FNO blocks each:
                                   spectral_conv(x) + 1x1_conv(x)  -> GELU
                           |
                           +--(2-layer 1x1 head)--> output channels

The spectral convolution truncates the rfft2 spectrum to a low-mode
square (``modes_x x modes_y``), multiplies by learnable complex weights,
and inverts the FFT. This is what gives the FNO its mesh-free
generalization across resolutions: the operator is parameterized in
Fourier space rather than per-pixel.
"""

from __future__ import annotations

import torch
from torch import nn


class SpectralConv2d(nn.Module):
    """2D spectral convolution with low-mode truncation.

    For each (in_channel, out_channel) pair, learnable complex weights
    of shape ``(modes_x, modes_y)`` multiply the corresponding low-frequency
    block of the rfft2 spectrum. Two weight tensors handle the positive
    and negative ``x`` (height-axis) frequency blocks since rfft2 only
    halves the last axis.
    """

    def __init__(self, in_channels: int, out_channels: int,
                 modes_x: int, modes_y: int):
        super().__init__()
        if modes_x < 1 or modes_y < 1:
            raise ValueError("modes_x and modes_y must be >= 1")
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes_x = modes_x
        self.modes_y = modes_y

        scale = 1.0 / (in_channels * out_channels)
        # Two weight tensors: one for positive-x low modes, one for negative-x
        # low modes. (rfft2 keeps both x bands but folds y to 0..N/2.)
        self.weights1 = nn.Parameter(
            scale * torch.randn(in_channels, out_channels, modes_x, modes_y, dtype=torch.cfloat)
        )
        self.weights2 = nn.Parameter(
            scale * torch.randn(in_channels, out_channels, modes_x, modes_y, dtype=torch.cfloat)
        )

    @staticmethod
    def _einsum(x_ft: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        # x_ft: (B, C_in, X, Y) ; weights: (C_in, C_out, X, Y)
        return torch.einsum("bixy,ioxy->boxy", x_ft, weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        x_ft = torch.fft.rfft2(x, norm="ortho")  # (B, C, H, W//2+1)

        # Effective modes (clip if H/W are smaller than configured modes)
        mx = min(self.modes_x, H // 2)
        my = min(self.modes_y, x_ft.shape[-1])

        out_ft = torch.zeros(
            B, self.out_channels, H, x_ft.shape[-1],
            dtype=torch.cfloat, device=x.device,
        )
        # Positive x low modes
        out_ft[:, :, :mx, :my] = self._einsum(
            x_ft[:, :, :mx, :my],
            self.weights1[:, :, :mx, :my],
        )
        # Negative x low modes (the wrap-around band)
        out_ft[:, :, -mx:, :my] = self._einsum(
            x_ft[:, :, -mx:, :my],
            self.weights2[:, :, :mx, :my],
        )
        return torch.fft.irfft2(out_ft, s=(H, W), norm="ortho")


class FNOBlock2d(nn.Module):
    """One FNO block: spectral conv + 1x1 conv skip + GELU."""

    def __init__(self, channels: int, modes_x: int, modes_y: int):
        super().__init__()
        self.spectral = SpectralConv2d(channels, channels, modes_x, modes_y)
        self.skip = nn.Conv2d(channels, channels, kernel_size=1)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.spectral(x) + self.skip(x))


class FNO2d(nn.Module):
    """Stack of FNO blocks with 1x1 lift / projection heads."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        hidden: int = 32,
        modes_x: int = 16,
        modes_y: int = 16,
        n_layers: int = 4,
    ):
        super().__init__()
        if n_layers < 1:
            raise ValueError("n_layers must be >= 1")
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hidden = hidden
        self.modes_x = modes_x
        self.modes_y = modes_y

        self.lift = nn.Conv2d(in_channels, hidden, kernel_size=1)
        self.blocks = nn.ModuleList(
            [FNOBlock2d(hidden, modes_x, modes_y) for _ in range(n_layers)]
        )
        self.proj = nn.Sequential(
            nn.Conv2d(hidden, 4 * hidden, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(4 * hidden, out_channels, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.lift(x)
        for blk in self.blocks:
            x = blk(x)
        return self.proj(x)

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

"""Phase 10 — FNO operator surrogate for the PEB submodule.

Self-contained 2D Fourier Neural Operator (Li et al., 2021) plus a
small adapter that turns a Phase-9 dataset into the per-channel
input / output tensors the FNO expects.

Operator learned:

    (H0(x, y), parameters)  ->  (H_final(x, y), P_final(x, y))

with R(x, y) recovered by thresholding ``P_final``. The plan also
lists ``DeepONet`` as an option; this submodule does the FNO branch
because it is the natural fit for the regular grid ``128 x 128``
inputs and matches the architecture the main repo already uses for
its Phase-7→9 surrogate work. DeepONet is left as an explicit
follow-up.

Input-channel layout (in this exact order):

    0    H0(x, y)                          [mol/dm^3]
    1    DH                                broadcast scalar
    2    DQ_ratio                          broadcast scalar
    3    kq_ref                            broadcast scalar
    4    kdep_ref                          broadcast scalar
    5    kloss_ref                         broadcast scalar
    6    Q0                                broadcast scalar
    7    temperature_c (centered on T_ref) broadcast scalar
    8    activation_energy_kj_mol          broadcast scalar
    9    t_end_s                           broadcast scalar

Output-channel layout:

    0    H_final(x, y)                     [mol/dm^3]
    1    P_final(x, y)                     [0, 1]

Both inputs and outputs are normalized in pixel space by the per-
channel mean/std computed on the **train** split — the normalizer
is part of the saved checkpoint.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn


# --------------------------------------------------------------------------
# FNO building blocks
# --------------------------------------------------------------------------

class SpectralConv2d(nn.Module):
    """Standard FNO spectral convolution (low-frequency Fourier mixing).

    Applies a learned complex weight matrix to the lowest ``modes``
    Fourier coefficients in each spatial dimension; the rest pass
    through unchanged. Implementation follows the original FNO paper.
    """

    def __init__(self, in_channels: int, out_channels: int,
                 modes_x: int, modes_y: int):
        super().__init__()
        if modes_x < 1 or modes_y < 1:
            raise ValueError("modes must be >= 1")
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes_x = modes_x
        self.modes_y = modes_y
        scale = 1.0 / (in_channels * out_channels)
        # Two halves of the spectrum (top and bottom of the kx axis).
        self.weight_top = nn.Parameter(
            scale * torch.randn(in_channels, out_channels,
                                 modes_x, modes_y, dtype=torch.cfloat)
        )
        self.weight_bot = nn.Parameter(
            scale * torch.randn(in_channels, out_channels,
                                 modes_x, modes_y, dtype=torch.cfloat)
        )

    def _mat(self, x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        # x: (B, Cin, kx, ky)  w: (Cin, Cout, kx, ky)  -> (B, Cout, kx, ky)
        return torch.einsum("bixy,ioxy->boxy", x, w)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, _, H, W = x.shape
        x_ft = torch.fft.rfft2(x, norm="ortho")
        kx_max = min(self.modes_x, H // 2)
        ky_max = min(self.modes_y, W // 2 + 1)
        out_ft = torch.zeros(
            B, self.out_channels, H, W // 2 + 1,
            dtype=torch.cfloat, device=x.device,
        )
        out_ft[:, :, :kx_max, :ky_max] = self._mat(
            x_ft[:, :, :kx_max, :ky_max],
            self.weight_top[:, :, :kx_max, :ky_max],
        )
        out_ft[:, :, -kx_max:, :ky_max] = self._mat(
            x_ft[:, :, -kx_max:, :ky_max],
            self.weight_bot[:, :, :kx_max, :ky_max],
        )
        return torch.fft.irfft2(out_ft, s=(H, W), norm="ortho")


class FNOBlock(nn.Module):
    """One FNO layer: spectral conv + 1x1 conv + GELU."""

    def __init__(self, channels: int, modes_x: int, modes_y: int):
        super().__init__()
        self.spectral = SpectralConv2d(channels, channels, modes_x, modes_y)
        self.local = nn.Conv2d(channels, channels, kernel_size=1)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.spectral(x) + self.local(x))


class FNO2d(nn.Module):
    """Stacked-block 2D FNO: lift -> N blocks -> project."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        width: int = 32,
        n_blocks: int = 4,
        modes_x: int = 12,
        modes_y: int = 12,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.lift = nn.Conv2d(in_channels, width, kernel_size=1)
        self.blocks = nn.ModuleList(
            FNOBlock(width, modes_x, modes_y) for _ in range(n_blocks)
        )
        self.project = nn.Sequential(
            nn.Conv2d(width, width, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(width, out_channels, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.lift(x)
        for block in self.blocks:
            x = block(x)
        return self.project(x)

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())


# --------------------------------------------------------------------------
# Phase-9 dataset adapter
# --------------------------------------------------------------------------

INPUT_FIELD_NAMES: tuple[str, ...] = ("H0",)
INPUT_SCALAR_NAMES: tuple[str, ...] = (
    "DH_nm2_s", "DQ_ratio",
    "kq_ref_s_inv", "kdep_ref_s_inv", "kloss_ref_s_inv",
    "Q0_mol_dm3",
    "temperature_c",
    "activation_energy_kj_mol",
    "t_end_s",
)
OUTPUT_FIELD_NAMES: tuple[str, ...] = ("H_final", "P_final")


def make_input_tensor(arrays: dict[str, np.ndarray]) -> torch.Tensor:
    """Pack a Phase-9 archive into the FNO input tensor of shape
    ``(n_samples, n_channels, G, G)``.

    Channel order matches ``INPUT_FIELD_NAMES + INPUT_SCALAR_NAMES``.
    """
    H0 = torch.from_numpy(arrays["H0"]).float()
    n, G, _ = H0.shape
    field_channels = [H0]                                       # (n, G, G)
    scalar_channels = []
    for name in INPUT_SCALAR_NAMES:
        s = torch.from_numpy(arrays[name]).float()
        if name == "temperature_c":
            s = s - 100.0                                       # center on T_ref
        scalar_channels.append(s.view(n, 1, 1).expand(n, G, G))
    channels = torch.stack(field_channels + scalar_channels, dim=1)
    return channels                                             # (n, C, G, G)


def make_output_tensor(arrays: dict[str, np.ndarray]) -> torch.Tensor:
    H = torch.from_numpy(arrays["H_final"]).float()
    P = torch.from_numpy(arrays["P_final"]).float()
    return torch.stack([H, P], dim=1)                           # (n, 2, G, G)


@dataclass
class ChannelStats:
    """Per-channel (mean, std) used to whiten inputs / outputs."""

    mean: torch.Tensor
    std: torch.Tensor

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.mean.to(x.device)) / self.std.to(x.device)

    def denormalize(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.std.to(x.device) + self.mean.to(x.device)


def fit_channel_stats(x: torch.Tensor, eps: float = 1e-6) -> ChannelStats:
    """Per-channel mean/std over (n, *spatial) -> shape (1, C, 1, 1)."""
    if x.ndim != 4:
        raise ValueError("x must have shape (n, C, H, W)")
    mean = x.mean(dim=(0, 2, 3), keepdim=True)
    std = x.std(dim=(0, 2, 3), keepdim=True).clamp(min=eps)
    return ChannelStats(mean=mean, std=std)


# --------------------------------------------------------------------------
# loss / metrics
# --------------------------------------------------------------------------

def relative_l2(pred: torch.Tensor, target: torch.Tensor,
                eps: float = 1e-8) -> torch.Tensor:
    """Per-sample L2 relative error: ||pred - target||_2 / ||target||_2.

    Returns a 1D tensor of shape ``(n_samples,)``.
    """
    diff = (pred - target).reshape(pred.shape[0], -1)
    norm = target.reshape(target.shape[0], -1)
    return diff.norm(dim=1) / norm.norm(dim=1).clamp(min=eps)


def per_channel_relative_l2(
    pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-8,
) -> torch.Tensor:
    """Per-channel L2 relative error: shape ``(C,)``.

    Uses the global Frobenius norm across all samples and pixels for
    each channel — closer to the FNO paper's convention than a
    per-sample average.
    """
    diff = pred - target
    num = diff.flatten(1).flatten().reshape(-1, *diff.shape[1:])  # placeholder
    # Compute channel-wise norms directly.
    n_channels = pred.shape[1]
    out = torch.empty(n_channels, device=pred.device)
    for c in range(n_channels):
        d = (pred[:, c] - target[:, c]).reshape(-1)
        t = target[:, c].reshape(-1)
        out[c] = d.norm() / t.norm().clamp(min=eps)
    return out


def thresholded_iou(
    pred_P: torch.Tensor, target_P: torch.Tensor, threshold: float = 0.5,
) -> torch.Tensor:
    """Per-sample IoU of the (P > threshold) masks. Returns (n,)."""
    pm = pred_P > threshold
    tm = target_P > threshold
    inter = (pm & tm).flatten(1).sum(dim=1).float()
    union = (pm | tm).flatten(1).sum(dim=1).float().clamp(min=1.0)
    return inter / union


# --------------------------------------------------------------------------
# convenience
# --------------------------------------------------------------------------

def build_fno_for_dataset(
    width: int = 32,
    n_blocks: int = 4,
    modes_x: int = 12,
    modes_y: int = 12,
) -> FNO2d:
    """Construct an ``FNO2d`` with the input/output channel counts that
    match :func:`make_input_tensor` / :func:`make_output_tensor`."""
    in_channels = len(INPUT_FIELD_NAMES) + len(INPUT_SCALAR_NAMES)
    out_channels = len(OUTPUT_FIELD_NAMES)
    return FNO2d(
        in_channels=in_channels, out_channels=out_channels,
        width=width, n_blocks=n_blocks,
        modes_x=modes_x, modes_y=modes_y,
    )


def manual_seed_everything(seed: int) -> None:
    """Seed Python ``random``, NumPy, and PyTorch so a training run is
    reproducible up to non-deterministic CUDA reductions."""
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

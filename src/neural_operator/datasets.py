"""PyTorch Dataset wrappers for the Phase-7 NPZ archives.

The on-disk archive stores per-sample ``masks``, ``T_thin_real / imag``,
``T_3d_real / imag``, and ``theta`` (length-6 vector). The Dataset
assembles these into a 9-channel input tensor (mask + T_thin real / imag
+ 6 theta channels broadcast spatially) and a 2-channel target tensor
(``delta_T`` real / imag, optionally ``T_3d`` real / imag).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


class CorrectionDataset(Dataset):
    """Loads the Phase-7 paired NPZ into PyTorch tensors.

    Args:
        npz_path: path to the NPZ archive.
        target: ``"delta_T"`` (default) or ``"T_3d"`` — what the model is
                trained to predict.
        device: optional device to move the (small, mostly contiguous)
                arrays to at construction. For typical 800-sample
                datasets at n=128 this fits comfortably in memory.
    """

    INPUT_CHANNELS_PER_SAMPLE = 9
    OUTPUT_CHANNELS = 2

    def __init__(
        self,
        npz_path: str | Path,
        target: str = "delta_T",
        device: torch.device | None = None,
    ):
        if target not in ("delta_T", "T_3d"):
            raise ValueError("target must be 'delta_T' or 'T_3d'")
        z = np.load(npz_path, allow_pickle=True)
        self.masks = torch.from_numpy(z["masks"]).float()
        self.T_thin_real = torch.from_numpy(z["T_thin_real"]).float()
        self.T_thin_imag = torch.from_numpy(z["T_thin_imag"]).float()
        self.T_3d_real = torch.from_numpy(z["T_3d_real"]).float()
        self.T_3d_imag = torch.from_numpy(z["T_3d_imag"]).float()
        self.theta = torch.from_numpy(z["theta"]).float()
        self.theta_names = list(z["theta_names"])
        self.target = target
        self.grid_n = int(z["grid_n"])
        self.grid_extent = float(z["grid_extent"])

        if device is not None:
            self.masks = self.masks.to(device)
            self.T_thin_real = self.T_thin_real.to(device)
            self.T_thin_imag = self.T_thin_imag.to(device)
            self.T_3d_real = self.T_3d_real.to(device)
            self.T_3d_imag = self.T_3d_imag.to(device)
            self.theta = self.theta.to(device)

        self._n = self.masks.shape[0]
        self._H = self.masks.shape[1]
        self._W = self.masks.shape[2]

    def __len__(self) -> int:
        return self._n

    @property
    def n_theta(self) -> int:
        return self.theta.shape[1]

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        mask = self.masks[idx]               # (H, W)
        Tr = self.T_thin_real[idx]
        Ti = self.T_thin_imag[idx]
        theta = self.theta[idx]              # (n_theta,)
        theta_maps = theta.view(-1, 1, 1).expand(-1, self._H, self._W)
        x = torch.cat(
            [mask.unsqueeze(0), Tr.unsqueeze(0), Ti.unsqueeze(0), theta_maps],
            dim=0,
        )  # (1 + 1 + 1 + n_theta, H, W) = (9, H, W)
        if self.target == "delta_T":
            yr = self.T_3d_real[idx] - Tr
            yi = self.T_3d_imag[idx] - Ti
        else:
            yr = self.T_3d_real[idx]
            yi = self.T_3d_imag[idx]
        y = torch.stack([yr, yi], dim=0)
        return x, y

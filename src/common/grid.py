"""Real-space and frequency-space grids for 2D scalar diffraction.

The lab works in normalized units (wavelength lambda = 1). A grid covers
[-L/2, L/2)^2 with N x N samples, so the real-space pixel size is
dx = L / N. The matching centered frequency grid has spacing df = 1 / L
and Nyquist magnitude 1 / (2 * dx).
"""

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class Grid2D:
    """2D Cartesian grid with matching real and frequency coordinates.

    All tensors are produced lazily on the requested device / dtype, but the
    grid metadata (size, extent, spacings) is fixed at construction.
    """

    n: int
    extent: float = 1.0  # physical side length in normalized units
    device: torch.device = torch.device("cpu")
    dtype: torch.dtype = torch.float32

    @property
    def dx(self) -> float:
        return self.extent / self.n

    @property
    def df(self) -> float:
        return 1.0 / self.extent

    @property
    def f_nyquist(self) -> float:
        return 1.0 / (2.0 * self.dx)

    def axes(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Centered 1D coordinate axes (x, y) in real space."""
        coord = (torch.arange(self.n, device=self.device, dtype=self.dtype) - self.n / 2) * self.dx
        return coord, coord.clone()

    def freq_axes(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Centered 1D coordinate axes (fx, fy) in frequency space."""
        coord = (torch.arange(self.n, device=self.device, dtype=self.dtype) - self.n / 2) * self.df
        return coord, coord.clone()

    def meshgrid(self) -> tuple[torch.Tensor, torch.Tensor]:
        x, y = self.axes()
        # 'xy' indexing matches matplotlib imshow convention (X varies along columns)
        return torch.meshgrid(x, y, indexing="xy")

    def freq_meshgrid(self) -> tuple[torch.Tensor, torch.Tensor]:
        fx, fy = self.freq_axes()
        return torch.meshgrid(fx, fy, indexing="xy")

    def radial(self) -> torch.Tensor:
        X, Y = self.meshgrid()
        return torch.sqrt(X * X + Y * Y)

    def radial_freq(self) -> torch.Tensor:
        FX, FY = self.freq_meshgrid()
        return torch.sqrt(FX * FX + FY * FY)

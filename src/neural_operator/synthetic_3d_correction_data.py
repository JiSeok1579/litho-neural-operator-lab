"""Synthetic 3D-mask correction operator and paired-dataset generator.

Real 3D mask effects (absorber height, sidewall angle, oblique
incidence, polarization, multi-layer interference) require RCWA or FDTD
to compute. For this study lab we substitute a synthetic correction
operator that has the right qualitative behavior — it is multiplicative
in spatial-frequency space, has tunable amplitude / phase / asymmetry,
and is known in closed form so the FNO surrogate in Phase 8 can be
benchmarked against an exact reference. Concretely:

    T_3d(fx, fy) = T_thin(fx, fy) * C(fx, fy; theta)

with

    C(fx, fy; theta) = a(fx, fy) * exp(i * phi(fx, fy)) * (1 + s * tanh(c * fx))
    a(fx, fy)   = exp(-gamma * (fx**2 + fy**2))
    phi(fx, fy) = alpha * fx + beta * fy + delta * (fx**2 - fy**2)

The six theta parameters loosely correspond to:

    gamma  - bulk amplitude attenuation away from DC (mask-thickness drop)
    alpha  - linear phase along x (oblique illumination tilt-x)
    beta   - linear phase along y (oblique illumination tilt-y)
    delta  - astigmatic phase (anisotropic absorber profile)
    s      - asymmetric-shadow amplitude (depth-of-shadow asymmetry)
    c      - asymmetric-shadow sharpness (transition rate of the shadow)

This module provides:

- ``CorrectionParams``           : the theta dataclass.
- ``correction_operator``        : C as a 2D complex tensor on a Grid2D.
- ``apply_3d_correction``        : T_3d = T_thin * C.
- ``sample_correction_params``   : random theta within configured ranges.
- ``random_mask_sampler``        : a varied mask generator drawing from
                                   line-space, contact-hole arrays, and
                                   block-random patterns.
- ``generate_dataset``           : produces a paired NPZ archive for
                                   Phase 8 / 9 to consume.
"""

from __future__ import annotations

import random
from dataclasses import asdict, dataclass, fields
from pathlib import Path

import numpy as np
import torch

from src.common.fft_utils import fft2c
from src.common.grid import Grid2D
from src.mask import patterns
from src.mask.transmission import binary_transmission


THETA_NAMES: tuple[str, ...] = ("gamma", "alpha", "beta", "delta", "s", "c")


@dataclass(frozen=True)
class CorrectionParams:
    """Parameters for the synthetic 3D-mask correction operator."""

    gamma: float  # amplitude Gaussian taper coefficient (>= 0)
    alpha: float  # linear phase along fx
    beta: float   # linear phase along fy
    delta: float  # astigmatic phase coefficient
    s: float      # asymmetric-shadow amplitude (in [-1, 1])
    c: float      # asymmetric-shadow sharpness (>= 0)

    def to_array(self) -> np.ndarray:
        return np.array([self.gamma, self.alpha, self.beta,
                         self.delta, self.s, self.c], dtype=np.float32)

    @classmethod
    def from_array(cls, arr) -> "CorrectionParams":
        a = np.asarray(arr, dtype=np.float64).reshape(-1)
        if a.size != 6:
            raise ValueError("CorrectionParams.from_array expects length 6")
        return cls(*a.tolist())

    @classmethod
    def identity(cls) -> "CorrectionParams":
        """Theta that produces ``C = 1`` (no correction)."""
        return cls(gamma=0.0, alpha=0.0, beta=0.0, delta=0.0, s=0.0, c=0.0)


DEFAULT_RANGES: dict[str, tuple[float, float]] = {
    "gamma": (0.00, 0.30),
    "alpha": (-0.40, 0.40),
    "beta":  (-0.40, 0.40),
    "delta": (-0.30, 0.30),
    "s":     (-0.40, 0.40),
    "c":     (0.00, 4.00),
}


def correction_operator(
    grid: Grid2D,
    params: CorrectionParams,
) -> torch.Tensor:
    """Compute the complex 2D correction ``C(fx, fy)`` on ``grid``.

    The result has dtype ``complex64`` and shape ``(grid.n, grid.n)``.
    Identity parameters (all zeros except ``c`` is unused when ``s = 0``)
    yield ``C = 1`` exactly.
    """
    fx, fy = grid.freq_meshgrid()
    fr2 = fx * fx + fy * fy
    amp = torch.exp(-float(params.gamma) * fr2)
    phase = (
        float(params.alpha) * fx
        + float(params.beta) * fy
        + float(params.delta) * (fx * fx - fy * fy)
    )
    asym = 1.0 + float(params.s) * torch.tanh(float(params.c) * fx)
    real = (amp * asym) * torch.cos(phase)
    imag = (amp * asym) * torch.sin(phase)
    return torch.complex(real, imag).to(torch.complex64)


def apply_3d_correction(
    t_thin_spectrum: torch.Tensor,
    correction: torch.Tensor,
) -> torch.Tensor:
    """Multiply a thin-mask spectrum by the correction operator."""
    if not torch.is_complex(t_thin_spectrum):
        raise TypeError("t_thin_spectrum must be complex")
    if not torch.is_complex(correction):
        raise TypeError("correction must be complex")
    return t_thin_spectrum * correction


def sample_correction_params(
    rng: random.Random,
    ranges: dict[str, tuple[float, float]] | None = None,
) -> CorrectionParams:
    """Draw a random ``CorrectionParams`` from independent uniform ranges."""
    r = ranges if ranges is not None else DEFAULT_RANGES
    return CorrectionParams(
        gamma=rng.uniform(*r["gamma"]),
        alpha=rng.uniform(*r["alpha"]),
        beta=rng.uniform(*r["beta"]),
        delta=rng.uniform(*r["delta"]),
        s=rng.uniform(*r["s"]),
        c=rng.uniform(*r["c"]),
    )


def random_mask_sampler(grid: Grid2D, seed: int) -> torch.Tensor:
    """Generate a single mask drawn from one of three families.

    Mix:
    - 50 % block-random binary (block sizes 2-8 px, fill 0.2-0.7)
    - 25 % line-space (pitch 1-4 lambda, duty 0.3-0.7, vertical/horizontal)
    - 25 % contact-hole arrangements (2-6 randomly-placed disks)
    """
    rng = random.Random(seed)
    kind = rng.choices(
        population=["random_binary", "line_space", "contact_holes"],
        weights=[0.5, 0.25, 0.25],
        k=1,
    )[0]
    if kind == "random_binary":
        fill = rng.uniform(0.2, 0.7)
        block_size = rng.randint(2, 8)
        return patterns.random_binary(
            grid, fill_fraction=fill, block_size=block_size, seed=seed
        )
    if kind == "line_space":
        pitch = rng.uniform(1.0, 4.0)
        duty = rng.uniform(0.3, 0.7)
        orientation = rng.choice(["vertical", "horizontal"])
        return patterns.line_space(
            grid, pitch=pitch, duty=duty, orientation=orientation
        )
    # contact_holes
    n_holes = rng.randint(2, 6)
    mask = torch.zeros((grid.n, grid.n), device=grid.device, dtype=grid.dtype)
    for _ in range(n_holes):
        cx = rng.uniform(-grid.extent * 0.4, grid.extent * 0.4)
        cy = rng.uniform(-grid.extent * 0.4, grid.extent * 0.4)
        r = rng.uniform(0.3, 1.0)
        hole = patterns.contact_hole(grid, radius=r, center=(cx, cy))
        mask = torch.maximum(mask, hole)
    return mask


def generate_dataset(
    grid: Grid2D,
    n_samples: int,
    output_path: str | Path,
    seed: int = 0,
    mask_sampler=random_mask_sampler,
    param_ranges: dict[str, tuple[float, float]] | None = None,
    verbose: bool = True,
) -> dict:
    """Generate ``n_samples`` paired samples and save to a compressed NPZ.

    Returns a small dict of summary statistics (sample count, theta means
    and stds, output L1 / L2 norms) for the caller to log.
    """
    if n_samples < 1:
        raise ValueError("n_samples must be >= 1")
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    masks = np.zeros((n_samples, grid.n, grid.n), dtype=np.float32)
    T_thin_real = np.zeros_like(masks)
    T_thin_imag = np.zeros_like(masks)
    T_3d_real = np.zeros_like(masks)
    T_3d_imag = np.zeros_like(masks)
    theta_arr = np.zeros((n_samples, len(THETA_NAMES)), dtype=np.float32)

    rng = random.Random(seed)
    for i in range(n_samples):
        mask_seed = rng.randrange(0, 2**31 - 1)
        mask = mask_sampler(grid, seed=mask_seed)
        params = sample_correction_params(rng, ranges=param_ranges)

        t_complex = binary_transmission(mask)
        T_thin = fft2c(t_complex)
        C = correction_operator(grid, params)
        T_3d = apply_3d_correction(T_thin, C)

        masks[i] = mask.detach().cpu().to(torch.float32).numpy()
        T_thin_real[i] = T_thin.real.detach().cpu().numpy()
        T_thin_imag[i] = T_thin.imag.detach().cpu().numpy()
        T_3d_real[i] = T_3d.real.detach().cpu().numpy()
        T_3d_imag[i] = T_3d.imag.detach().cpu().numpy()
        theta_arr[i] = params.to_array()

        if verbose and ((i + 1) % max(1, n_samples // 10) == 0):
            print(f"  generated {i + 1}/{n_samples}")

    np.savez_compressed(
        output_path,
        masks=masks,
        T_thin_real=T_thin_real,
        T_thin_imag=T_thin_imag,
        T_3d_real=T_3d_real,
        T_3d_imag=T_3d_imag,
        theta=theta_arr,
        theta_names=np.array(list(THETA_NAMES), dtype=object),
        grid_n=np.array(grid.n, dtype=np.int32),
        grid_extent=np.array(grid.extent, dtype=np.float32),
    )

    summary = {
        "n_samples": int(n_samples),
        "grid_n": int(grid.n),
        "grid_extent": float(grid.extent),
        "theta_means": {n: float(theta_arr[:, k].mean())
                        for k, n in enumerate(THETA_NAMES)},
        "theta_stds": {n: float(theta_arr[:, k].std())
                       for k, n in enumerate(THETA_NAMES)},
        "T_thin_l2_mean": float(np.sqrt(T_thin_real ** 2 + T_thin_imag ** 2).mean()),
        "T_3d_l2_mean": float(np.sqrt(T_3d_real ** 2 + T_3d_imag ** 2).mean()),
        "delta_T_l2_mean": float(np.sqrt(
            (T_3d_real - T_thin_real) ** 2 + (T_3d_imag - T_thin_imag) ** 2
        ).mean()),
        "output_path": str(output_path),
    }
    return summary


def load_dataset(path: str | Path) -> dict:
    """Reload a saved NPZ archive into a dict of numpy arrays."""
    z = np.load(path, allow_pickle=True)
    return {k: z[k] for k in z.files}

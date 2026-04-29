"""Phase 11 stochastic layers — temperature uniformity ensemble +
molecular-blur post-processing.

Two small wrappers around the Phase-8 / Phase-11 evolvers:

  * ``temperature_uniformity_ensemble`` runs the full FD evolver
    ``n_runs`` times with the per-run temperature drawn from
    ``Normal(temperature_c, temperature_uniformity_c)`` and reports
    the (mean, std) of the resulting ``(H, Q, P)`` fields. This models
    the spatial / wafer-level temperature non-uniformity called for in
    the plan's ``stochastic.temperature_uniformity_c`` knob.

  * ``molecular_blur_2d`` applies a separable 1D Gaussian convolution
    to a final ``P`` (or any 2D field) at a length-scale set by
    ``particle_size_nm`` / ``molecular_blur_nm``. Periodic boundaries
    via ``torch.roll`` keep it consistent with the rest of the
    submodule's FFT / FD machinery.

z-axis / 3D film thickness is intentionally **not** in this module —
that requires rewriting the (H, Q, P) evolvers in 3D and is tracked
explicitly as a ``FUTURE_WORK`` item.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable

import numpy as np
import torch


# --------------------------------------------------------------------------
# temperature uniformity ensemble
# --------------------------------------------------------------------------

@dataclass
class EnsembleResult:
    """Aggregated output from a temperature-uniformity ensemble run.

    Each ``*_mean`` / ``*_std`` field is a 2D tensor of the same shape
    as the input ``H0``. ``temperatures_c`` records the perturbed
    temperatures used for each ensemble member (length ``n_runs``).
    """

    H_mean: torch.Tensor
    H_std: torch.Tensor
    Q_mean: torch.Tensor
    Q_std: torch.Tensor
    P_mean: torch.Tensor
    P_std: torch.Tensor
    temperatures_c: list[float]


def temperature_uniformity_ensemble(
    evolver: Callable,
    evolver_kwargs: dict,
    temperature_uniformity_c: float,
    n_runs: int,
    seed: int = 0,
) -> EnsembleResult:
    """Run ``evolver`` ``n_runs`` times with perturbed temperatures.

    ``evolver`` must accept ``temperature_c=...`` as a keyword argument
    and return ``(H_final, Q_final, P_final)``. The function copies
    ``evolver_kwargs``, replaces ``temperature_c`` with
    ``temperature_c + Normal(0, temperature_uniformity_c)`` per run,
    and aggregates the results.

    Setting ``temperature_uniformity_c = 0`` collapses every run to the
    same nominal temperature; the returned ``*_std`` arrays are
    identically zero — useful as a sanity check.
    """
    if n_runs < 1:
        raise ValueError("n_runs must be >= 1")
    if temperature_uniformity_c < 0:
        raise ValueError("temperature_uniformity_c must be non-negative")
    if "temperature_c" not in evolver_kwargs:
        raise ValueError(
            "evolver_kwargs must contain 'temperature_c' as the nominal "
            "temperature to perturb around"
        )
    rng = np.random.default_rng(seed)
    base_T = float(evolver_kwargs["temperature_c"])
    H_runs: list[torch.Tensor] = []
    Q_runs: list[torch.Tensor] = []
    P_runs: list[torch.Tensor] = []
    Ts: list[float] = []
    for _ in range(n_runs):
        delta = float(rng.normal(0.0, temperature_uniformity_c)) \
            if temperature_uniformity_c > 0 else 0.0
        T_run = base_T + delta
        kw = dict(evolver_kwargs)
        kw["temperature_c"] = T_run
        H, Q, P = evolver(**kw)
        H_runs.append(H)
        Q_runs.append(Q)
        P_runs.append(P)
        Ts.append(T_run)
    H_stack = torch.stack(H_runs)
    Q_stack = torch.stack(Q_runs)
    P_stack = torch.stack(P_runs)
    H_std = H_stack.std(dim=0, unbiased=False) if n_runs > 1 \
        else torch.zeros_like(H_runs[0])
    Q_std = Q_stack.std(dim=0, unbiased=False) if n_runs > 1 \
        else torch.zeros_like(Q_runs[0])
    P_std = P_stack.std(dim=0, unbiased=False) if n_runs > 1 \
        else torch.zeros_like(P_runs[0])
    return EnsembleResult(
        H_mean=H_stack.mean(dim=0), H_std=H_std,
        Q_mean=Q_stack.mean(dim=0), Q_std=Q_std,
        P_mean=P_stack.mean(dim=0), P_std=P_std,
        temperatures_c=Ts,
    )


# --------------------------------------------------------------------------
# molecular blur
# --------------------------------------------------------------------------

def _gaussian_kernel_1d(sigma_px: float, dtype: torch.dtype = torch.float32,
                        device: torch.device | str = "cpu") -> torch.Tensor:
    """1D normalized Gaussian kernel of radius ``ceil(3 * sigma_px)``.

    Returns shape ``(2 * radius + 1,)``. Sum is exactly 1 by
    construction (after the normalize step).
    """
    if sigma_px <= 0:
        raise ValueError("sigma_px must be positive")
    radius = max(1, int(math.ceil(3.0 * sigma_px)))
    x = torch.arange(-radius, radius + 1, dtype=dtype, device=device)
    k = torch.exp(-0.5 * (x / sigma_px) ** 2)
    return k / k.sum()


def molecular_blur_2d(
    field: torch.Tensor,
    sigma_nm: float,
    dx_nm: float,
) -> torch.Tensor:
    """Periodic Gaussian blur of a 2D field at length-scale ``sigma_nm``.

    Implemented as two separable 1D convolutions via ``torch.roll`` so
    it stays consistent with the rest of the submodule's periodic-BC
    machinery. ``sigma_nm = 0`` returns the input unchanged.
    """
    if sigma_nm < 0:
        raise ValueError("sigma_nm must be non-negative")
    if dx_nm <= 0:
        raise ValueError("dx_nm must be positive")
    if sigma_nm == 0.0:
        return field.clone()
    if field.ndim < 2:
        raise ValueError("field must have at least 2 dims (..., H, W)")
    sigma_px = sigma_nm / dx_nm
    kernel = _gaussian_kernel_1d(
        sigma_px, dtype=field.dtype, device=field.device,
    )
    radius = (kernel.shape[0] - 1) // 2

    # x-direction (last dim).
    out = torch.zeros_like(field)
    for i, w in enumerate(kernel):
        shift = i - radius
        out = out + w * torch.roll(field, shifts=shift, dims=-1)

    # y-direction.
    out2 = torch.zeros_like(out)
    for i, w in enumerate(kernel):
        shift = i - radius
        out2 = out2 + w * torch.roll(out, shifts=shift, dims=-2)
    return out2


def molecular_blur_P(P: torch.Tensor, sigma_nm: float,
                     dx_nm: float) -> torch.Tensor:
    """Convenience wrapper that clamps the blurred ``P`` back to
    ``[0, 1]`` so downstream thresholding stays well-defined."""
    blurred = molecular_blur_2d(P, sigma_nm=sigma_nm, dx_nm=dx_nm)
    return blurred.clamp(min=0.0, max=1.0)

"""Gradient-descent inverse mask optimization.

Pipeline:

    theta (raw, unconstrained)
       |  sigmoid(alpha * theta)
       v
    mask m in (0, 1)
       |  m -> complex64
       v
    transmission t = m + 0j
       |  coherent_aerial_image
       v
    aerial intensity I
       |  region-based losses + regularizers
       v
    scalar loss --(autograd)--> theta gradient

``alpha`` follows a step schedule that starts gentle (smooth gradient
landscape) and progressively sharpens (drives ``m`` to {0, 1}). This is the
binarization annealing pattern from study plan §3.8.

This module deliberately stays a *function* rather than a class; the
optimization state is small and self-contained, and a function is easier
to wrap in experiment scripts.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import torch

from src.common.grid import Grid2D
from src.inverse.losses import (
    background_loss,
    mean_intensity_in_region,
    target_loss,
)
from src.inverse.regularizers import binarization_penalty, total_variation
from src.optics.coherent_imaging import coherent_aerial_image
from src.optics.pupil import circular_pupil


@dataclass
class OptimizationResult:
    """Output of :func:`optimize_mask` — everything an experiment script needs."""

    theta: torch.Tensor          # raw parameter (final state, no grad)
    mask: torch.Tensor           # sigmoid(alpha_final * theta), in (0, 1)
    aerial_initial: torch.Tensor
    aerial_final: torch.Tensor
    target_region: torch.Tensor
    forbidden_region: torch.Tensor
    pupil: torch.Tensor
    history: list[dict]          # one dict per logged iteration


@dataclass
class LossWeights:
    target: float = 1.0
    background: float = 1.0
    tv: float = 1.0e-3
    binarization: float = 0.0


@dataclass
class OptimizationConfig:
    n_iters: int = 1000
    lr: float = 1.0e-2
    NA: float = 0.4
    wavelength: float = 1.0
    target_value: float = 1.0
    weights: LossWeights = field(default_factory=LossWeights)
    # (iteration_threshold, alpha_value); the latest threshold whose value
    # is <= the current iteration is active.
    alpha_schedule: Sequence[tuple[int, float]] = (
        (0, 1.0),
        (300, 4.0),
        (700, 12.0),
    )
    log_every: int = 10
    seed: int = 0


def _alpha_at(it: int, schedule: Sequence[tuple[int, float]]) -> float:
    a = schedule[0][1]
    for thresh, val in schedule:
        if it >= thresh:
            a = val
    return a


def optimize_mask(
    grid: Grid2D,
    target_region: torch.Tensor,
    forbidden_region: torch.Tensor,
    config: OptimizationConfig | None = None,
    init_theta: torch.Tensor | None = None,
) -> OptimizationResult:
    """Run gradient descent on a real-valued raw mask parameter.

    Returns an :class:`OptimizationResult` with the final mask, both aerial
    images (initial and final), and a history of loss / diagnostic scalars
    sampled every ``config.log_every`` iterations.
    """
    cfg = config if config is not None else OptimizationConfig()
    torch.manual_seed(cfg.seed)

    device = grid.device
    dtype = grid.dtype

    target_region = target_region.to(device=device, dtype=dtype)
    forbidden_region = forbidden_region.to(device=device, dtype=dtype)

    pupil = circular_pupil(grid, NA=cfg.NA, wavelength=cfg.wavelength)

    if init_theta is None:
        theta = torch.zeros((grid.n, grid.n), device=device, dtype=dtype, requires_grad=True)
    else:
        theta = init_theta.detach().to(device=device, dtype=dtype).clone().requires_grad_(True)

    optim = torch.optim.Adam([theta], lr=cfg.lr)

    aerial_initial: torch.Tensor | None = None
    history: list[dict] = []

    for it in range(cfg.n_iters):
        alpha = _alpha_at(it, cfg.alpha_schedule)
        m = torch.sigmoid(alpha * theta)
        t = m.to(torch.complex64)
        aerial = coherent_aerial_image(t, pupil, normalize=False)

        if aerial_initial is None:
            aerial_initial = aerial.detach().clone()

        l_target = target_loss(aerial, target_region, cfg.target_value)
        l_bg = background_loss(aerial, forbidden_region)
        l_tv = total_variation(m)
        l_bin = binarization_penalty(m)
        loss = (
            cfg.weights.target * l_target
            + cfg.weights.background * l_bg
            + cfg.weights.tv * l_tv
            + cfg.weights.binarization * l_bin
        )

        optim.zero_grad(set_to_none=True)
        loss.backward()
        optim.step()

        if (it % cfg.log_every == 0) or (it == cfg.n_iters - 1):
            with torch.no_grad():
                mean_t = mean_intensity_in_region(aerial, target_region).item()
                mean_b = mean_intensity_in_region(aerial, forbidden_region).item()
            history.append({
                "iter": it,
                "alpha": alpha,
                "loss_total": float(loss.item()),
                "loss_target": float(l_target.item()),
                "loss_background": float(l_bg.item()),
                "loss_tv": float(l_tv.item()),
                "loss_binarization": float(l_bin.item()),
                "mean_intensity_target": mean_t,
                "mean_intensity_background": mean_b,
            })

    with torch.no_grad():
        alpha_final = _alpha_at(cfg.n_iters, cfg.alpha_schedule)
        m_final = torch.sigmoid(alpha_final * theta)
        t_final = m_final.to(torch.complex64)
        aerial_final = coherent_aerial_image(t_final, pupil, normalize=False)

    return OptimizationResult(
        theta=theta.detach(),
        mask=m_final.detach(),
        aerial_initial=aerial_initial.detach() if aerial_initial is not None else aerial_final,
        aerial_final=aerial_final.detach(),
        target_region=target_region.detach(),
        forbidden_region=forbidden_region.detach(),
        pupil=pupil.detach(),
        history=history,
    )

"""Inverse mask optimization with a pluggable 3D-mask correction step.

Generalization of :func:`src.inverse.optimize_mask.optimize_mask`. The
imaging chain is the same Phase-3 pipeline plus one extra step that
turns the thin-mask spectrum into a 3D-corrected spectrum:

    theta -> sigmoid(alpha * theta) -> mask m
            -> binary_transmission        -> t
            -> fft2c                       -> T_thin
            -> CORRECTION (pluggable)      -> T_3d
            -> multiply by pupil           -> T_3d * P
            -> ifft2c                      -> wafer field E
            -> |E|^2                       -> aerial intensity I
            -> region losses + regularizers -> scalar loss
            -> backprop                    -> theta gradient

The "CORRECTION" step accepts a callable
``(mask, T_thin) -> T_3d`` so the same optimizer can be driven by:

- ``identity_correction_fn``    : no correction (study plan §9.5 case A)
- ``true_correction_fn``        : exact synthetic operator   (case B)
- ``fno_correction_fn``         : trained FNO surrogate      (case C)

This is the spine of the closed-loop experiment: every case shares the
optimizer, regularizers, region masks, and initialization; only the
correction call swaps. Phase 9's headline finding (surrogate-induced
optimizer dishonesty, study plan §9.6) comes from running case C and
then re-evaluating its optimized mask under case B.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Sequence

import torch

from src.common.fft_utils import fft2c, ifft2c
from src.common.grid import Grid2D
from src.inverse.losses import (
    background_loss,
    mean_intensity_in_region,
    target_loss,
)
from src.inverse.optimize_mask import LossWeights, _alpha_at
from src.inverse.regularizers import binarization_penalty, total_variation
from src.mask.transmission import binary_transmission
from src.neural_operator.synthetic_3d_correction_data import (
    CorrectionParams,
    correction_operator,
)
from src.optics.pupil import circular_pupil


CorrectionFn = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]


# ---- correction factories -------------------------------------------------

def identity_correction_fn() -> CorrectionFn:
    """T_3d = T_thin (no correction)."""
    def fn(_mask: torch.Tensor, T_thin: torch.Tensor) -> torch.Tensor:
        return T_thin
    return fn


def true_correction_fn(grid: Grid2D, params: CorrectionParams) -> CorrectionFn:
    """T_3d = T_thin * correction_operator(grid, params).

    The correction tensor is precomputed once at factory-call time;
    every optimizer step just does a complex multiply.
    """
    C = correction_operator(grid, params)

    def fn(_mask: torch.Tensor, T_thin: torch.Tensor) -> torch.Tensor:
        return T_thin * C

    return fn


def fno_correction_fn(
    fno_model,
    params: CorrectionParams,
    device: torch.device,
) -> CorrectionFn:
    """T_3d = T_thin + FNO(mask, T_thin, theta).

    FNO predicts ``delta_T`` (real / imag stacked as 2 channels). The
    network is set to eval mode but its weights stay attached to the
    autograd graph so gradients flow from the mask parameter through
    the FNO and out the other side; we just do not include FNO weights
    in the optimizer's parameter list, so they never get updated.
    """
    theta_vec = torch.tensor(
        params.to_array(), device=device, dtype=torch.float32
    )
    fno_model.eval()

    def fn(mask: torch.Tensor, T_thin: torch.Tensor) -> torch.Tensor:
        H, W = mask.shape
        theta_maps = theta_vec.view(-1, 1, 1).expand(-1, H, W)
        x = torch.cat(
            [
                mask.unsqueeze(0),
                T_thin.real.unsqueeze(0),
                T_thin.imag.unsqueeze(0),
                theta_maps,
            ],
            dim=0,
        ).unsqueeze(0)  # (1, 9, H, W)
        pred = fno_model(x)[0]  # (2, H, W)
        delta_T = torch.complex(pred[0], pred[1])
        return T_thin + delta_T

    return fn


# ---- result + config ------------------------------------------------------

@dataclass
class SurrogateOptimizationConfig:
    n_iters: int = 800
    lr: float = 5.0e-2
    NA: float = 0.4
    wavelength: float = 1.0
    target_value: float = 1.0
    weights: LossWeights = field(default_factory=lambda: LossWeights(
        target=1.0, background=1.0, tv=1.0e-3, binarization=0.0,
    ))
    alpha_schedule: Sequence[tuple[int, float]] = (
        (0, 1.0), (300, 4.0), (600, 12.0),
    )
    log_every: int = 10
    seed: int = 0


@dataclass
class SurrogateOptimizationResult:
    theta: torch.Tensor
    mask: torch.Tensor
    aerial_initial: torch.Tensor
    aerial_final: torch.Tensor
    target_region: torch.Tensor
    forbidden_region: torch.Tensor
    pupil: torch.Tensor
    history: list[dict]


# ---- main entry point -----------------------------------------------------

def optimize_mask_with_correction(
    grid: Grid2D,
    target_region: torch.Tensor,
    forbidden_region: torch.Tensor,
    correction_fn: CorrectionFn,
    config: SurrogateOptimizationConfig | None = None,
    init_theta: torch.Tensor | None = None,
) -> SurrogateOptimizationResult:
    """Phase-9 inverse design with a pluggable correction step."""
    cfg = config if config is not None else SurrogateOptimizationConfig()
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
        t = binary_transmission(m)
        T_thin = fft2c(t)
        T_3d = correction_fn(m, T_thin)
        E = ifft2c(T_3d * pupil)
        I = E.real ** 2 + E.imag ** 2

        if aerial_initial is None:
            aerial_initial = I.detach().clone()

        l_target = target_loss(I, target_region, cfg.target_value)
        l_bg = background_loss(I, forbidden_region)
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
                mean_t = mean_intensity_in_region(I, target_region).item()
                mean_b = mean_intensity_in_region(I, forbidden_region).item()
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
        t_final = binary_transmission(m_final)
        T_thin_final = fft2c(t_final)
        T_3d_final = correction_fn(m_final, T_thin_final)
        E_final = ifft2c(T_3d_final * pupil)
        aerial_final = (E_final.real ** 2 + E_final.imag ** 2).detach().clone()

    return SurrogateOptimizationResult(
        theta=theta.detach(),
        mask=m_final.detach(),
        aerial_initial=aerial_initial.detach() if aerial_initial is not None else aerial_final,
        aerial_final=aerial_final,
        target_region=target_region.detach(),
        forbidden_region=forbidden_region.detach(),
        pupil=pupil.detach(),
        history=history,
    )


# ---- validation helper ----------------------------------------------------

@torch.no_grad()
def evaluate_mask_under_correction(
    grid: Grid2D,
    mask: torch.Tensor,
    correction_fn: CorrectionFn,
    NA: float,
    wavelength: float = 1.0,
) -> torch.Tensor:
    """Compute the aerial intensity of a fixed ``mask`` through a given
    correction. This is the Phase-9 validation step: take an FNO-optimized
    mask and re-evaluate under the *true* correction to see whether the
    surrogate convinced the optimizer of an aerial that does not actually
    exist in the true physics.
    """
    pupil = circular_pupil(grid, NA=NA, wavelength=wavelength)
    t = binary_transmission(mask.to(grid.device))
    T_thin = fft2c(t)
    T_3d = correction_fn(mask, T_thin)
    E = ifft2c(T_3d * pupil)
    return (E.real ** 2 + E.imag ** 2).detach()

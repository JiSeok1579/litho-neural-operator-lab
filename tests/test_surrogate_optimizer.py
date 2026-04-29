"""Tests for :mod:`src.closed_loop.surrogate_optimizer`."""

from __future__ import annotations

import torch

from src.common.fft_utils import fft2c
from src.common.grid import Grid2D
from src.inverse.optimize_mask import LossWeights
from src.mask import patterns
from src.mask.transmission import binary_transmission
from src.closed_loop.surrogate_optimizer import (
    SurrogateOptimizationConfig,
    evaluate_mask_under_correction,
    fno_correction_fn,
    identity_correction_fn,
    optimize_mask_with_correction,
    true_correction_fn,
)
from src.neural_operator.fno2d import FNO2d
from src.neural_operator.synthetic_3d_correction_data import (
    CorrectionParams,
    correction_operator,
)


def _grid() -> Grid2D:
    return Grid2D(n=64, extent=8.0)


def test_identity_correction_returns_T_thin_unchanged():
    g = _grid()
    fn = identity_correction_fn()
    mask = patterns.contact_hole(g, radius=1.0)
    T_thin = fft2c(binary_transmission(mask))
    out = fn(mask, T_thin)
    assert torch.equal(out, T_thin)


def test_true_correction_matches_direct_call():
    g = _grid()
    params = CorrectionParams(gamma=0.1, alpha=0.2, beta=-0.1,
                              delta=0.0, s=0.3, c=1.5)
    C = correction_operator(g, params)
    fn = true_correction_fn(g, params)
    mask = patterns.contact_hole(g, radius=1.0)
    T_thin = fft2c(binary_transmission(mask))
    out = fn(mask, T_thin)
    assert torch.allclose(out, T_thin * C)


def test_fno_correction_shape_and_grad_through_mask():
    g = _grid()
    fno = FNO2d(in_channels=9, out_channels=2, hidden=8,
                modes_x=4, modes_y=4, n_layers=2)
    params = CorrectionParams(gamma=0.05, alpha=0.0, beta=0.0,
                              delta=0.0, s=0.0, c=0.0)
    fn = fno_correction_fn(fno, params, device=torch.device("cpu"))

    raw = torch.zeros((g.n, g.n), requires_grad=True)
    mask = torch.sigmoid(raw)
    T_thin = fft2c(binary_transmission(mask))
    T_3d = fn(mask, T_thin)
    assert T_3d.shape == (g.n, g.n)
    assert torch.is_complex(T_3d)
    # Gradient must flow back to raw
    loss = (T_3d.real ** 2 + T_3d.imag ** 2).sum()
    loss.backward()
    assert raw.grad is not None
    assert raw.grad.abs().max().item() > 0


def test_optimize_with_identity_correction_runs():
    g = _grid()
    target = patterns.contact_hole(g, radius=1.0)
    forbidden = 1.0 - target
    cfg = SurrogateOptimizationConfig(
        n_iters=20, lr=5e-2, NA=0.5,
        weights=LossWeights(target=1.0, background=1.0, tv=1e-4, binarization=0.0),
        alpha_schedule=((0, 1.0),),
        log_every=5,
    )
    res = optimize_mask_with_correction(
        g, target, forbidden, identity_correction_fn(), config=cfg,
    )
    assert res.mask.shape == (g.n, g.n)
    assert res.aerial_final.shape == (g.n, g.n)
    assert len(res.history) >= 1
    # Loss should decrease (or at least not blow up) on this small budget
    assert res.history[-1]["loss_target"] <= res.history[0]["loss_target"] + 1e-3


def test_evaluate_mask_under_correction_runs():
    g = _grid()
    mask = patterns.contact_hole(g, radius=1.0)
    params = CorrectionParams(gamma=0.1, alpha=0.0, beta=0.0,
                              delta=0.0, s=0.0, c=0.0)
    I = evaluate_mask_under_correction(
        g, mask, true_correction_fn(g, params), NA=0.5,
    )
    assert I.shape == (g.n, g.n)
    assert (I >= 0).all().item()


def test_identity_optimize_matches_phase3_in_spirit():
    """Optimization with identity correction should drive the target loss
    down on a simple disk target, just like Phase-3 optimize_mask."""
    g = _grid()
    target = patterns.contact_hole(g, radius=1.0)
    forbidden = 1.0 - target
    cfg = SurrogateOptimizationConfig(
        n_iters=80, lr=5e-2, NA=0.6,
        weights=LossWeights(target=1.0, background=0.5, tv=1e-4, binarization=0.0),
        alpha_schedule=((0, 1.0),),
        log_every=5,
    )
    res = optimize_mask_with_correction(
        g, target, forbidden, identity_correction_fn(), config=cfg,
    )
    h = res.history
    assert h[-1]["loss_target"] < h[0]["loss_target"]
    assert h[-1]["mean_intensity_target"] > h[0]["mean_intensity_target"]

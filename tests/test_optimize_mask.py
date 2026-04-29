"""Tests for :mod:`src.inverse.optimize_mask`.

These run a small optimization budget so the suite stays fast on CPU.
"""

from __future__ import annotations

import torch

from src.common.grid import Grid2D
from src.inverse.optimize_mask import (
    LossWeights,
    OptimizationConfig,
    optimize_mask,
)
from src.mask import patterns


def _setup(n: int = 64, extent: float = 8.0):
    grid = Grid2D(n=n, extent=extent)
    target = patterns.contact_hole(grid, radius=1.0)
    forbidden = 1.0 - target
    return grid, target, forbidden


def test_optimize_mask_returns_expected_shapes():
    grid, target, forbidden = _setup()
    cfg = OptimizationConfig(n_iters=20, lr=5e-2, log_every=5)
    res = optimize_mask(grid, target, forbidden, config=cfg)
    assert res.mask.shape == (grid.n, grid.n)
    assert res.aerial_final.shape == (grid.n, grid.n)
    assert res.aerial_initial.shape == (grid.n, grid.n)
    assert len(res.history) >= 1
    # Diagnostic keys are present
    h0 = res.history[0]
    for key in ("iter", "alpha", "loss_total", "loss_target",
                "loss_background", "loss_tv", "loss_binarization",
                "mean_intensity_target", "mean_intensity_background"):
        assert key in h0, f"missing history key: {key}"


def test_optimize_mask_decreases_target_loss():
    """A reasonable run with NA large enough to resolve the target should
    drive the target loss down from its initial value."""
    grid, target, forbidden = _setup(n=64, extent=8.0)
    cfg = OptimizationConfig(
        n_iters=80, lr=5e-2, NA=0.6,
        weights=LossWeights(target=1.0, background=0.5, tv=1e-4, binarization=0.0),
        alpha_schedule=((0, 1.0),),  # no annealing for the test
        log_every=5,
    )
    res = optimize_mask(grid, target, forbidden, config=cfg)
    h = res.history
    assert h[-1]["loss_target"] < h[0]["loss_target"], (
        f"target loss did not decrease: {h[0]['loss_target']} -> {h[-1]['loss_target']}"
    )


def test_optimize_mask_pushes_target_intensity_up():
    grid, target, forbidden = _setup(n=64, extent=8.0)
    cfg = OptimizationConfig(
        n_iters=80, lr=5e-2, NA=0.6,
        alpha_schedule=((0, 1.0),),
        log_every=5,
    )
    res = optimize_mask(grid, target, forbidden, config=cfg)
    h = res.history
    assert h[-1]["mean_intensity_target"] > h[0]["mean_intensity_target"]


def test_alpha_schedule_steps():
    """Verify the alpha schedule produces the expected piecewise-constant
    history of alpha values."""
    grid, target, forbidden = _setup()
    cfg = OptimizationConfig(
        n_iters=50, lr=5e-2,
        alpha_schedule=((0, 1.0), (10, 4.0), (30, 12.0)),
        log_every=1,
    )
    res = optimize_mask(grid, target, forbidden, config=cfg)
    alphas = [h["alpha"] for h in res.history]
    assert alphas[0] == 1.0
    assert alphas[10] == 4.0
    assert alphas[30] == 12.0


def test_optimize_mask_runs_on_cuda_if_available():
    if not torch.cuda.is_available():
        return
    grid = Grid2D(n=64, extent=8.0, device=torch.device("cuda"))
    target = patterns.contact_hole(grid, radius=1.0)
    forbidden = 1.0 - target
    cfg = OptimizationConfig(n_iters=20, lr=5e-2, log_every=10)
    res = optimize_mask(grid, target, forbidden, config=cfg)
    assert res.mask.device.type == "cuda"
    assert res.aerial_final.device.type == "cuda"

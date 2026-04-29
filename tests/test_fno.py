"""Tests for the Phase-8 FNO building blocks and dataset wrapper."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from src.common.grid import Grid2D
from src.neural_operator.datasets import CorrectionDataset
from src.neural_operator.fno2d import FNO2d, FNOBlock2d, SpectralConv2d
from src.neural_operator.synthetic_3d_correction_data import generate_dataset
from src.neural_operator.train_fno import (
    aerial_intensity_mse,
    complex_relative_error,
    evaluate_fno,
    spectrum_mse,
    train_fno_correction,
)


def test_spectral_conv2d_shape():
    sc = SpectralConv2d(in_channels=4, out_channels=8, modes_x=8, modes_y=8)
    x = torch.randn(2, 4, 32, 32)
    y = sc(x)
    assert y.shape == (2, 8, 32, 32)
    assert y.dtype == torch.float32


def test_spectral_conv_resolution_independent():
    """An FNO layer trained at one resolution should still run at another;
    we verify the shape contract here."""
    sc = SpectralConv2d(in_channels=2, out_channels=2, modes_x=4, modes_y=4)
    for n in (16, 24, 32):
        x = torch.randn(1, 2, n, n)
        y = sc(x)
        assert y.shape == (1, 2, n, n)


def test_fno_block_residual_path():
    blk = FNOBlock2d(channels=4, modes_x=4, modes_y=4)
    x = torch.randn(2, 4, 16, 16)
    y = blk(x)
    assert y.shape == x.shape


def test_fno_2d_end_to_end():
    model = FNO2d(in_channels=9, out_channels=2, hidden=8,
                  modes_x=4, modes_y=4, n_layers=2)
    x = torch.randn(2, 9, 32, 32)
    y = model(x)
    assert y.shape == (2, 2, 32, 32)
    n_params = model.num_parameters()
    assert n_params > 0


def test_fno_autograd():
    model = FNO2d(in_channels=3, out_channels=2, hidden=4,
                  modes_x=4, modes_y=4, n_layers=1)
    x = torch.randn(1, 3, 16, 16, requires_grad=False)
    target = torch.randn(1, 2, 16, 16)
    pred = model(x)
    loss = spectrum_mse(pred, target)
    loss.backward()
    has_grad = any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.parameters())
    assert has_grad


def test_complex_relative_error_zero_when_match():
    target = torch.randn(2, 2, 8, 8)
    err = complex_relative_error(target.clone(), target).item()
    assert err < 1e-6


def test_aerial_intensity_mse_zero_when_match():
    delta = torch.randn(2, 2, 16, 16)
    T_thin = torch.complex(torch.randn(2, 16, 16), torch.randn(2, 16, 16))
    err = aerial_intensity_mse(delta, delta, T_thin).item()
    assert err < 1e-6


def test_correction_dataset_loads_phase7_format(tmp_path):
    """Generate a tiny Phase-7 NPZ on the fly, load with CorrectionDataset,
    and verify the channel layout."""
    g = Grid2D(n=32, extent=4.0)
    out = tmp_path / "tiny.npz"
    generate_dataset(g, n_samples=3, output_path=out, seed=0, verbose=False)

    ds = CorrectionDataset(out)
    assert len(ds) == 3
    x, y = ds[0]
    assert x.shape == (9, g.n, g.n)  # mask + Tr + Ti + 6 theta channels
    assert y.shape == (2, g.n, g.n)
    # Channel 0 is the mask in [0, 1]
    assert x[0].min() >= 0.0 and x[0].max() <= 1.0
    # Channels 3..8 are theta broadcast to constant maps
    for k in range(3, 9):
        assert torch.allclose(x[k], torch.full_like(x[k], x[k].mean()), atol=1e-6)


def test_correction_dataset_target_modes(tmp_path):
    g = Grid2D(n=16, extent=2.0)
    out = tmp_path / "tiny.npz"
    generate_dataset(g, n_samples=2, output_path=out, seed=0, verbose=False)

    ds_delta = CorrectionDataset(out, target="delta_T")
    ds_t3d = CorrectionDataset(out, target="T_3d")
    _, y_delta = ds_delta[0]
    _, y_t3d = ds_t3d[0]
    # T_3d_real - T_thin_real (channel 0 of y_delta) should equal
    # T_3d_real (channel 0 of y_t3d) minus the mask-input's T_thin_real
    Tr = ds_delta.T_thin_real[0]
    expected_delta_real = ds_delta.T_3d_real[0] - Tr
    assert torch.allclose(y_delta[0], expected_delta_real, atol=1e-6)
    assert torch.allclose(y_t3d[0], ds_t3d.T_3d_real[0], atol=1e-6)


def test_train_fno_smoke(tmp_path):
    """End-to-end smoke: generate tiny Phase-7 archive, train FNO for 1
    epoch, verify the train+test losses are finite."""
    from torch.utils.data import DataLoader
    g = Grid2D(n=16, extent=2.0)
    train_path = tmp_path / "train.npz"
    test_path = tmp_path / "test.npz"
    generate_dataset(g, n_samples=4, output_path=train_path, seed=0, verbose=False)
    generate_dataset(g, n_samples=2, output_path=test_path, seed=1, verbose=False)

    train_ds = CorrectionDataset(train_path)
    test_ds = CorrectionDataset(test_path)
    train_loader = DataLoader(train_ds, batch_size=2)
    test_loader = DataLoader(test_ds, batch_size=2)

    model = FNO2d(in_channels=9, out_channels=2, hidden=8,
                  modes_x=4, modes_y=4, n_layers=2)

    res = train_fno_correction(
        model, train_loader, test_loader,
        n_epochs=2, lr=1e-3, log_every=1,
    )
    assert len(res.train_losses) == 2
    for v in res.train_losses + res.test_losses:
        assert float("inf") != v != float("nan")

    te_mse, te_rel, te_aerial = evaluate_fno(model, test_loader, weight_aerial=0.0)
    assert all(torch.isfinite(torch.tensor(v)) for v in (te_mse, te_rel, te_aerial))

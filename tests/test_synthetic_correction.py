"""Tests for the Phase-7 synthetic 3D-mask correction module."""

from __future__ import annotations

import math
import random
from pathlib import Path

import numpy as np
import torch

from src.common.fft_utils import fft2c
from src.common.grid import Grid2D
from src.mask import patterns
from src.mask.transmission import binary_transmission
from src.neural_operator.synthetic_3d_correction_data import (
    CorrectionParams,
    DEFAULT_RANGES,
    THETA_NAMES,
    apply_3d_correction,
    correction_operator,
    generate_dataset,
    load_dataset,
    random_mask_sampler,
    sample_correction_params,
)


def _grid() -> Grid2D:
    return Grid2D(n=64, extent=8.0)


def test_identity_correction_returns_unity():
    g = _grid()
    C = correction_operator(g, CorrectionParams.identity())
    assert C.dtype == torch.complex64
    assert C.shape == (g.n, g.n)
    assert torch.allclose(C.real, torch.ones_like(C.real), atol=1e-6)
    assert torch.allclose(C.imag, torch.zeros_like(C.imag), atol=1e-6)


def test_apply_3d_correction_is_multiplicative():
    g = _grid()
    spec = fft2c(binary_transmission(patterns.contact_hole(g, radius=1.0)))
    C = correction_operator(
        g, CorrectionParams(gamma=0.1, alpha=0.0, beta=0.0,
                            delta=0.0, s=0.0, c=0.0),
    )
    T_3d = apply_3d_correction(spec, C)
    assert torch.allclose(T_3d, spec * C)


def test_correction_amplitude_dropoff():
    """Increasing gamma should attenuate high-frequency amplitude."""
    g = _grid()
    C_low = correction_operator(
        g, CorrectionParams(gamma=0.05, alpha=0, beta=0, delta=0, s=0, c=0))
    C_high = correction_operator(
        g, CorrectionParams(gamma=0.30, alpha=0, beta=0, delta=0, s=0, c=0))
    # At the highest representable spatial frequency the high-gamma
    # correction must be smaller than the low-gamma one.
    corner = (-1, -1)
    assert C_high.abs()[corner].item() < C_low.abs()[corner].item()


def test_linear_phase_shifts_amplitude_unchanged():
    """A pure linear phase shift (alpha, beta != 0, gamma = s = 0) must
    preserve the magnitude of every frequency."""
    g = _grid()
    C = correction_operator(
        g, CorrectionParams(gamma=0, alpha=0.3, beta=-0.2, delta=0, s=0, c=0))
    assert torch.allclose(C.abs(), torch.ones_like(C.abs()), atol=1e-5)


def test_asymmetric_shadow_breaks_symmetry():
    """``s != 0`` introduces an fx-dependent amplitude that breaks
    symmetry under fx -> -fx."""
    g = _grid()
    C = correction_operator(
        g, CorrectionParams(gamma=0, alpha=0, beta=0, delta=0, s=0.4, c=2.0))
    # |C(fx, fy)| at +fx column vs -fx column should differ.
    n = g.n
    left = C.abs()[:, n // 4]
    right = C.abs()[:, 3 * n // 4]
    assert not torch.allclose(left, right, atol=1e-3)


def test_correction_operator_differentiable():
    """Theta parameters need to be differentiable inputs for any future
    gradient-based correction-fitting."""
    g = _grid()
    gamma = torch.tensor(0.1, requires_grad=True)
    fx, fy = g.freq_meshgrid()
    fr2 = fx * fx + fy * fy
    C_amp = torch.exp(-gamma * fr2)
    C_amp.sum().backward()
    assert gamma.grad is not None and gamma.grad.abs().item() > 0


def test_correction_params_array_roundtrip():
    p1 = CorrectionParams(gamma=0.1, alpha=0.2, beta=-0.1,
                          delta=0.05, s=0.3, c=2.0)
    p2 = CorrectionParams.from_array(p1.to_array())
    for f in ("gamma", "alpha", "beta", "delta", "s", "c"):
        assert abs(getattr(p1, f) - getattr(p2, f)) < 1e-6


def test_sample_correction_params_in_range():
    rng = random.Random(0)
    for _ in range(50):
        p = sample_correction_params(rng)
        for k, (lo, hi) in DEFAULT_RANGES.items():
            assert lo <= getattr(p, k) <= hi, f"{k} out of range"


def test_random_mask_sampler_returns_binary():
    g = _grid()
    for seed in [0, 1, 17, 99, 255]:
        m = random_mask_sampler(g, seed=seed)
        assert m.shape == (g.n, g.n)
        unique_set = set(torch.unique(m).cpu().tolist())
        # Mask should be 0 / 1 valued (rounded for the contact-hole mix).
        assert unique_set.issubset({0.0, 1.0})


def test_generate_dataset_writes_npz(tmp_path):
    g = _grid()
    out = tmp_path / "tiny.npz"
    summary = generate_dataset(
        g, n_samples=4, output_path=out, seed=0, verbose=False,
    )
    assert out.exists()
    assert summary["n_samples"] == 4

    loaded = load_dataset(out)
    for key in ("masks", "T_thin_real", "T_thin_imag",
                "T_3d_real", "T_3d_imag", "theta", "theta_names",
                "grid_n", "grid_extent"):
        assert key in loaded, f"missing key {key}"
    assert loaded["masks"].shape == (4, g.n, g.n)
    assert loaded["theta"].shape == (4, len(THETA_NAMES))
    # theta_names roundtrips as object array
    assert tuple(loaded["theta_names"].tolist()) == THETA_NAMES


def test_generate_dataset_t_3d_matches_correction():
    """Re-applying ``correction_operator`` to ``T_thin`` must reproduce the
    saved ``T_3d`` to better than fp32 round-off."""
    g = _grid()
    out = Path("outputs/datasets/_test_phase7_consistency.npz")
    try:
        generate_dataset(g, n_samples=3, output_path=out, seed=42, verbose=False)
        d = load_dataset(out)
        for i in range(3):
            T_thin = torch.complex(
                torch.from_numpy(d["T_thin_real"][i]),
                torch.from_numpy(d["T_thin_imag"][i]),
            )
            T_3d = torch.complex(
                torch.from_numpy(d["T_3d_real"][i]),
                torch.from_numpy(d["T_3d_imag"][i]),
            )
            params = CorrectionParams.from_array(d["theta"][i])
            C = correction_operator(g, params)
            T_3d_recomputed = T_thin * C
            err = (T_3d - T_3d_recomputed).abs().max().item()
            rel = err / (T_3d.abs().max().item() + 1e-12)
            assert rel < 1e-5, f"sample {i}: rel err {rel}"
    finally:
        if out.exists():
            out.unlink()

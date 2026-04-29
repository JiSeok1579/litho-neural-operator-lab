"""Round-trip and convention tests for :mod:`src.common.fft_utils`."""

from __future__ import annotations

import torch

from src.common.fft_utils import fft2c, ifft2c


def _rand_complex(n: int, seed: int = 0) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    return torch.complex(
        torch.randn(n, n, generator=g, dtype=torch.float32),
        torch.randn(n, n, generator=g, dtype=torch.float32),
    )


def test_fft_inverse_round_trip_cpu():
    x = _rand_complex(64)
    y = ifft2c(fft2c(x))
    err = (x - y).abs().max().item()
    assert err < 1e-5, f"round-trip error too large: {err}"


def test_fft_orthonormal_parseval():
    """With norm='ortho' the FFT preserves L2 norm (Parseval's identity)."""
    x = _rand_complex(64, seed=1)
    X = fft2c(x)
    energy_real = (x.abs() ** 2).sum().item()
    energy_freq = (X.abs() ** 2).sum().item()
    rel = abs(energy_real - energy_freq) / energy_real
    assert rel < 1e-5, f"parseval violated: rel error {rel}"


def test_fft_dc_at_center():
    """A constant field should yield all energy at the centered DC bin."""
    n = 32
    x = torch.ones(n, n, dtype=torch.complex64)
    X = fft2c(x)
    # All energy concentrated at the center pixel (n//2, n//2)
    center = (n // 2, n // 2)
    total_energy = X.abs().pow(2).sum().item()
    center_energy = X[center].abs().pow(2).item()
    assert center_energy / total_energy > 0.999, "DC not at center"


def test_fft_batch_dim():
    """Batched input should leave the batch dim untouched."""
    x = _rand_complex(32, seed=2).unsqueeze(0).expand(4, -1, -1).contiguous()
    X = fft2c(x)
    assert X.shape == x.shape
    # All entries in batch start identical, FFT is linear -> stay identical
    diffs = (X - X[0:1]).abs().max().item()
    assert diffs < 1e-4


def test_fft_runs_on_cuda_if_available():
    if not torch.cuda.is_available():
        return
    x = _rand_complex(64, seed=3).cuda()
    X = fft2c(x)
    assert X.device.type == "cuda"
    rt = ifft2c(X)
    err = (x - rt).abs().max().item()
    assert err < 1e-4, f"cuda round-trip error: {err}"

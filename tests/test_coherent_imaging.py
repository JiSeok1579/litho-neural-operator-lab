"""Sanity tests for :mod:`src.optics.coherent_imaging`."""

from __future__ import annotations

import torch

from src.common.grid import Grid2D
from src.mask import patterns, transmission
from src.optics.coherent_imaging import coherent_aerial_image, coherent_field
from src.optics.pupil import circular_pupil


def _grid() -> Grid2D:
    return Grid2D(n=128, extent=10.0)


def test_aerial_is_real_nonneg_in_unit_range():
    g = _grid()
    m = patterns.contact_hole(g, radius=1.0)
    t = transmission.binary_transmission(m)
    p = circular_pupil(g, NA=0.6)
    I = coherent_aerial_image(t, p, normalize=True)
    assert I.dtype == g.dtype
    assert (I >= 0).all().item()
    assert I.max().item() <= 1.0 + 1e-6


def test_full_pupil_recovers_mask_intensity():
    """With a pupil of all ones (no NA cutoff) the imaging chain is identity:
    aerial = |t|^2 = mask^2 (for a binary mask in {0, 1}, that equals mask)."""
    g = _grid()
    m = patterns.contact_hole(g, radius=1.0)
    t = transmission.binary_transmission(m)
    full_pupil = torch.ones((g.n, g.n), dtype=g.dtype)
    I = coherent_aerial_image(t, full_pupil, normalize=False)
    expected = m.to(g.dtype) ** 2
    err = (I - expected).abs().max().item()
    assert err < 1e-4, f"err {err}"


def test_dc_only_pupil_yields_uniform_intensity():
    """A pupil that passes only the DC bin produces a spatially uniform aerial
    equal to |mean(t)|^2."""
    g = _grid()
    m = patterns.contact_hole(g, radius=1.0)
    t = transmission.binary_transmission(m)
    pupil = torch.zeros((g.n, g.n), dtype=g.dtype)
    pupil[g.n // 2, g.n // 2] = 1.0
    I = coherent_aerial_image(t, pupil, normalize=False)
    rel_std = (I.std() / (I.mean() + 1e-12)).item()
    assert rel_std < 1e-3, f"DC-only aerial not uniform: rel std {rel_std}"


def test_higher_NA_recovers_mask_better():
    """Higher NA preserves more high-frequency content, so the normalized
    aerial sits closer to the (normalized) mask in MSE."""
    g = _grid()
    m = patterns.contact_hole(g, radius=1.5)
    t = transmission.binary_transmission(m)
    m_norm = m / (m.max() + 1e-12)

    def err_at_NA(NA: float) -> float:
        I = coherent_aerial_image(t, circular_pupil(g, NA=NA), normalize=True)
        return ((I - m_norm) ** 2).mean().item()

    e_low = err_at_NA(0.2)
    e_high = err_at_NA(1.0)
    assert e_high < e_low, f"high NA error {e_high} should be < low NA error {e_low}"


def test_low_NA_blocks_high_frequencies():
    """For a line-space whose period is an integer number of pixels, the
    DFT lines sit exactly on integer bins. With NA below the fundamental
    frequency, the only spectrum that survives the pupil is the DC bin,
    so the aerial collapses to a perfectly uniform field.
    """
    # extent / N must give dx that divides pitch evenly. n=256, extent=16
    # -> dx = 0.0625; pitch = 1.0 spans exactly 16 pixels per period.
    g = Grid2D(n=256, extent=16.0)
    m = patterns.line_space(g, pitch=1.0, duty=0.5, orientation="vertical")
    t = transmission.binary_transmission(m)
    # Fundamental at fx = 1 / pitch = 1.0 cycles/lambda; NA=0.5 cuts at 0.5
    I = coherent_aerial_image(t, circular_pupil(g, NA=0.5), normalize=False)
    rel_std = (I.std() / (I.mean() + 1e-12)).item()
    assert rel_std < 1e-4, f"low-NA aerial not uniform: rel std {rel_std}"


def test_unaligned_pitch_low_NA_strongly_suppresses_variation():
    """Even when the pitch does not fit an integer number of pixels (the
    typical case in inverse design), an NA below the fundamental must
    suppress aerial variation by a large factor relative to a high NA
    that lets the fundamental through."""
    g = Grid2D(n=256, extent=20.0)
    m = patterns.line_space(g, pitch=1.0, duty=0.5, orientation="vertical")
    t = transmission.binary_transmission(m)

    def rel_std(NA: float) -> float:
        I = coherent_aerial_image(t, circular_pupil(g, NA=NA), normalize=False)
        return (I.std() / (I.mean() + 1e-12)).item()

    s_low = rel_std(0.4)   # below fundamental at 1.0
    s_high = rel_std(2.0)  # well above fundamental
    assert s_low < 0.5, f"low-NA std {s_low} too large"
    assert s_high > 5 * s_low, f"high-NA std {s_high} should dominate low {s_low}"


def test_autograd_through_aerial():
    """Gradients must flow from an intensity-based loss back to a real-valued
    mask parameter — Phase 3 inverse design depends on this."""
    g = _grid()
    raw = torch.zeros((g.n, g.n), dtype=g.dtype, requires_grad=True)
    mask = torch.sigmoid(raw + 1.0)
    t = mask.to(torch.complex64)
    p = circular_pupil(g, NA=0.5)
    I = coherent_aerial_image(t, p, normalize=False)
    loss = I.sum()
    loss.backward()
    assert raw.grad is not None
    assert raw.grad.abs().max().item() > 0


def test_field_complex_dtype():
    g = _grid()
    m = patterns.contact_hole(g, radius=1.0)
    t = transmission.binary_transmission(m)
    p = circular_pupil(g, NA=0.5)
    E = coherent_field(t, p)
    assert torch.is_complex(E)
    assert E.shape == (g.n, g.n)


def test_rejects_real_transmission():
    g = _grid()
    real_input = torch.zeros((g.n, g.n), dtype=g.dtype)
    p = circular_pupil(g, NA=0.5)
    try:
        coherent_field(real_input, p)
    except TypeError:
        pass
    else:
        assert False, "real transmission should raise"

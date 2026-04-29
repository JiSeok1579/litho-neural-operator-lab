"""Sanity tests for :mod:`src.mask.patterns` and :mod:`src.mask.transmission`."""

from __future__ import annotations

import math

import torch

from src.common.grid import Grid2D
from src.mask import patterns, transmission


def _grid(n: int = 64, extent: float = 1.0) -> Grid2D:
    return Grid2D(n=n, extent=extent)


def test_line_space_shape_and_values():
    g = _grid()
    m = patterns.line_space(g, pitch=0.2, duty=0.5, orientation="vertical")
    assert m.shape == (g.n, g.n)
    assert torch.all((m == 0) | (m == 1))
    # Each row should contain the same 1D profile (vertical stripes)
    assert torch.allclose(m[0], m[-1])


def test_line_space_orientation():
    g = _grid()
    v = patterns.line_space(g, pitch=0.2, orientation="vertical")
    h = patterns.line_space(g, pitch=0.2, orientation="horizontal")
    assert not torch.equal(v, h)
    assert torch.equal(v.T, h)


def test_contact_hole_area_ratio():
    g = _grid(n=128, extent=1.0)
    radius = 0.1
    m = patterns.contact_hole(g, radius=radius)
    measured = m.sum().item() * (g.dx ** 2)
    expected = math.pi * radius ** 2
    assert abs(measured - expected) / expected < 0.05


def test_isolated_line_width():
    g = _grid(n=128, extent=1.0)
    width = 0.1
    m = patterns.isolated_line(g, width=width, orientation="vertical")
    # Sum across one row equals number of open pixels in width
    open_pixels = m[0].sum().item()
    measured_width = open_pixels * g.dx
    assert abs(measured_width - width) < 2 * g.dx


def test_two_bar_disjoint():
    g = _grid(n=128, extent=1.0)
    m = patterns.two_bar(g, width=0.05, gap=0.1)
    # Two disconnected components in any single row that crosses both bars
    row = m[g.n // 2]
    transitions = (row[1:] - row[:-1]).abs().sum().item()
    assert transitions == 4  # two bars => 4 edges


def test_random_binary_density():
    g = _grid(n=128)
    m = patterns.random_binary(g, fill_fraction=0.3, seed=0)
    measured = m.mean().item()
    assert abs(measured - 0.3) < 0.05


def test_binary_transmission_dtype():
    g = _grid()
    m = patterns.contact_hole(g, radius=0.1)
    t = transmission.binary_transmission(m)
    assert t.dtype == torch.complex64
    assert torch.allclose(t.real, m)
    assert torch.all(t.imag == 0)


def test_attenuated_phase_shift_amplitude():
    g = _grid()
    m = patterns.contact_hole(g, radius=0.1)
    t = transmission.attenuated_phase_shift(m, attenuation=0.06, phase_rad=math.pi)
    # Open: |t| = 1
    open_mag = t[m > 0.5].abs()
    assert torch.allclose(open_mag, torch.ones_like(open_mag), atol=1e-6)
    # Blocked: |t| = sqrt(0.06)
    blocked_mag = t[m < 0.5].abs()
    assert torch.allclose(blocked_mag, torch.full_like(blocked_mag, math.sqrt(0.06)), atol=1e-6)
    # Blocked phase = pi (cos pi = -1)
    blocked_real = t[m < 0.5].real
    assert torch.all(blocked_real < 0)

"""Phase 11 tests — temperature-uniformity ensemble + molecular blur."""

from __future__ import annotations

import math

import pytest
import torch

from reaction_diffusion_peb.src.exposure import acid_generation
from reaction_diffusion_peb.src.full_reaction_diffusion import (
    evolve_full_reaction_diffusion_fd_at_T,
)
from reaction_diffusion_peb.src.stochastic_layers import (
    EnsembleResult,
    molecular_blur_2d,
    molecular_blur_P,
    temperature_uniformity_ensemble,
)
from reaction_diffusion_peb.src.synthetic_aerial import gaussian_spot


GRID = 32
DX = 1.0
T_END = 4.0


@pytest.fixture(scope="module")
def H0():
    I = gaussian_spot(GRID, sigma_px=4.0)
    return acid_generation(I, dose=1.0, eta=1.0, Hmax=0.2)


@pytest.fixture(scope="module")
def base_evolver_kwargs(H0):
    return dict(
        H0=H0, Q0_mol_dm3=0.1,
        DH_nm2_s=0.8, DQ_nm2_s=0.08,
        kq_ref_s_inv=1.0, kloss_ref_s_inv=0.005, kdep_ref_s_inv=0.5,
        temperature_c=100.0, temperature_ref_c=100.0,
        activation_energy_kj_mol=100.0,
        t_end_s=T_END, dx_nm=DX,
    )


# ---- temperature_uniformity_ensemble -----------------------------------

def test_ensemble_returns_correct_shapes(base_evolver_kwargs, H0):
    res = temperature_uniformity_ensemble(
        evolver=evolve_full_reaction_diffusion_fd_at_T,
        evolver_kwargs=dict(base_evolver_kwargs),
        temperature_uniformity_c=0.1,
        n_runs=3, seed=0,
    )
    assert isinstance(res, EnsembleResult)
    for field in (res.H_mean, res.H_std, res.Q_mean, res.Q_std,
                   res.P_mean, res.P_std):
        assert field.shape == H0.shape
    assert len(res.temperatures_c) == 3


def test_ensemble_zero_uniformity_collapses_to_deterministic(
    base_evolver_kwargs,
):
    """At σ_T = 0 every run uses the nominal temperature; std must be
    identically zero and mean must equal the deterministic Phase-8
    output."""
    res = temperature_uniformity_ensemble(
        evolver=evolve_full_reaction_diffusion_fd_at_T,
        evolver_kwargs=dict(base_evolver_kwargs),
        temperature_uniformity_c=0.0,
        n_runs=4, seed=1,
    )
    H_det, Q_det, P_det = evolve_full_reaction_diffusion_fd_at_T(
        **base_evolver_kwargs,
    )
    assert torch.allclose(res.H_std, torch.zeros_like(res.H_std), atol=1e-8)
    assert torch.allclose(res.Q_std, torch.zeros_like(res.Q_std), atol=1e-8)
    assert torch.allclose(res.P_std, torch.zeros_like(res.P_std), atol=1e-8)
    assert torch.allclose(res.H_mean, H_det)
    assert torch.allclose(res.Q_mean, Q_det)
    assert torch.allclose(res.P_mean, P_det)
    # Every member should report the nominal T verbatim.
    for T in res.temperatures_c:
        assert math.isclose(T, base_evolver_kwargs["temperature_c"])


def test_ensemble_seeded_runs_are_reproducible(base_evolver_kwargs):
    res_a = temperature_uniformity_ensemble(
        evolver=evolve_full_reaction_diffusion_fd_at_T,
        evolver_kwargs=dict(base_evolver_kwargs),
        temperature_uniformity_c=0.5,
        n_runs=4, seed=42,
    )
    res_b = temperature_uniformity_ensemble(
        evolver=evolve_full_reaction_diffusion_fd_at_T,
        evolver_kwargs=dict(base_evolver_kwargs),
        temperature_uniformity_c=0.5,
        n_runs=4, seed=42,
    )
    assert res_a.temperatures_c == res_b.temperatures_c
    assert torch.allclose(res_a.P_mean, res_b.P_mean)


def test_ensemble_positive_uniformity_produces_nonzero_std(
    base_evolver_kwargs,
):
    res = temperature_uniformity_ensemble(
        evolver=evolve_full_reaction_diffusion_fd_at_T,
        evolver_kwargs=dict(base_evolver_kwargs),
        temperature_uniformity_c=2.0,
        n_runs=6, seed=7,
    )
    assert float(res.P_std.max().item()) > 0.0


def test_ensemble_rejects_zero_runs(base_evolver_kwargs):
    with pytest.raises(ValueError):
        temperature_uniformity_ensemble(
            evolver=evolve_full_reaction_diffusion_fd_at_T,
            evolver_kwargs=dict(base_evolver_kwargs),
            temperature_uniformity_c=0.5,
            n_runs=0,
        )


def test_ensemble_rejects_missing_temperature_kwarg():
    kw = {"foo": 1.0}
    with pytest.raises(ValueError):
        temperature_uniformity_ensemble(
            evolver=lambda **k: None,
            evolver_kwargs=kw,
            temperature_uniformity_c=0.1,
            n_runs=2,
        )


# ---- molecular_blur_2d -------------------------------------------------

def test_molecular_blur_zero_sigma_returns_input():
    f = torch.randn(GRID, GRID)
    out = molecular_blur_2d(f, sigma_nm=0.0, dx_nm=DX)
    assert torch.allclose(out, f)


def test_molecular_blur_constant_field_unchanged():
    f = torch.full((GRID, GRID), 0.7)
    out = molecular_blur_2d(f, sigma_nm=2.0, dx_nm=DX)
    assert torch.allclose(out, f, atol=1e-6)


def test_molecular_blur_preserves_total_mass():
    """Periodic BC + normalized kernel ⇒ blur conserves the integral."""
    f = torch.randn(GRID, GRID).abs()
    out = molecular_blur_2d(f, sigma_nm=2.0, dx_nm=DX)
    assert math.isclose(float(out.sum().item()),
                         float(f.sum().item()), rel_tol=1e-5)


def test_molecular_blur_smooths_a_spike():
    """A delta becomes a Gaussian — the central peak must drop after blur."""
    f = torch.zeros(GRID, GRID)
    f[GRID // 2, GRID // 2] = 1.0
    out = molecular_blur_2d(f, sigma_nm=2.0, dx_nm=DX)
    assert float(out[GRID // 2, GRID // 2].item()) < 1.0
    assert float(out[GRID // 2, GRID // 2].item()) > 0.0
    assert float(out.sum().item()) == pytest.approx(1.0, abs=1e-5)


def test_molecular_blur_rejects_negative_sigma():
    f = torch.zeros(GRID, GRID)
    with pytest.raises(ValueError):
        molecular_blur_2d(f, sigma_nm=-0.1, dx_nm=DX)


def test_molecular_blur_rejects_nonpositive_dx():
    f = torch.zeros(GRID, GRID)
    with pytest.raises(ValueError):
        molecular_blur_2d(f, sigma_nm=1.0, dx_nm=0.0)


# ---- molecular_blur_P --------------------------------------------------

def test_molecular_blur_P_clamps_to_unit_interval():
    P = torch.rand(GRID, GRID) * 1.4 - 0.2     # span [-0.2, 1.2]
    out = molecular_blur_P(P, sigma_nm=2.0, dx_nm=DX)
    assert float(out.min().item()) >= 0.0
    assert float(out.max().item()) <= 1.0


def test_molecular_blur_P_zero_sigma_clamps_only():
    P = torch.tensor([[-0.1, 0.5], [0.7, 1.2]])
    out = molecular_blur_P(P, sigma_nm=0.0, dx_nm=1.0)
    expected = P.clamp(min=0.0, max=1.0)
    assert torch.allclose(out, expected)

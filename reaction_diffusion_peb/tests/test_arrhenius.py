"""Tests for the PEB Phase-6 Arrhenius temperature correction.

The five user-listed verification criteria are mapped 1-to-1:

  1. T = T_ref               ->  k(T) = k_ref      (criterion 1)
  2. T > T_ref, Ea > 0       ->  k(T) > k_ref      (criterion 2)
  3. Ea = 0                  ->  k(T) = k_ref      (criterion 3)
  4. Higher T -> higher P_final, larger threshold area (criterion 4)
  5. 125°C MOR preset gives a different result than 100°C (criterion 5)
"""

from __future__ import annotations

import math

import torch

from reaction_diffusion_peb.src.arrhenius import (
    apply_arrhenius_to_rates,
    arrhenius_factor,
    celsius_to_kelvin,
    evolve_acid_loss_deprotection_fd_at_T,
)
from reaction_diffusion_peb.src.deprotection import thresholded_area
from reaction_diffusion_peb.src.exposure import acid_generation
from reaction_diffusion_peb.src.synthetic_aerial import gaussian_spot


def _gaussian_H0(grid_size: int = 64, sigma_px: float = 6.0,
                Hmax: float = 0.2) -> torch.Tensor:
    I = gaussian_spot(grid_size, sigma_px=sigma_px)
    return acid_generation(I, dose=1.0, eta=1.0, Hmax=Hmax)


# ---- Arrhenius factor contracts ------------------------------------------

def test_factor_at_reference_temperature_is_one():
    assert arrhenius_factor(100.0, 100.0, 100.0) == 1.0
    assert arrhenius_factor(20.0, 20.0, 50.0) == 1.0


def test_factor_with_zero_Ea_is_one_for_any_T():
    # criterion 3: Ea=0 removes temperature dependence
    for T_c in (-50.0, 0.0, 50.0, 125.0, 300.0):
        assert arrhenius_factor(T_c, temperature_ref_c=100.0,
                                activation_energy_kj_mol=0.0) == 1.0


def test_factor_increases_with_T_for_positive_Ea():
    # criterion 2: T > T_ref with Ea > 0 -> factor > 1
    Ea = 100.0
    f80 = arrhenius_factor(80.0, 100.0, Ea)
    f100 = arrhenius_factor(100.0, 100.0, Ea)
    f125 = arrhenius_factor(125.0, 100.0, Ea)
    assert f80 < f100 == 1.0 < f125
    # And specifically for the 125°C preset:
    assert f125 > 1.0


def test_factor_monotonic_in_temperature():
    Ea = 100.0
    Ts = [80, 90, 100, 110, 120, 125]
    factors = [arrhenius_factor(T, 100.0, Ea) for T in Ts]
    for k in range(len(factors) - 1):
        assert factors[k] < factors[k + 1] + 1e-12


def test_factor_against_hand_calculation():
    """Spot-check the formula against an explicit hand calculation."""
    Ea_kj = 100.0
    T_c, T_ref_c = 110.0, 100.0
    R = 8.314
    Ea_J = Ea_kj * 1000
    T_K = T_c + 273.15
    T_ref_K = T_ref_c + 273.15
    expected = math.exp(-Ea_J / R * (1.0 / T_K - 1.0 / T_ref_K))
    measured = arrhenius_factor(T_c, T_ref_c, Ea_kj)
    assert abs(measured - expected) / expected < 1e-12


def test_celsius_to_kelvin_conversion():
    assert abs(celsius_to_kelvin(0.0) - 273.15) < 1e-9
    assert abs(celsius_to_kelvin(100.0) - 373.15) < 1e-9
    assert abs(celsius_to_kelvin(-273.15) - 0.0) < 1e-9


def test_apply_arrhenius_scales_both_rates():
    kdep_ref, kloss_ref = 0.5, 0.005
    kdep_eff, kloss_eff = apply_arrhenius_to_rates(
        kdep_ref, kloss_ref,
        temperature_c=110.0, temperature_ref_c=100.0,
        activation_energy_kj_mol=100.0,
    )
    factor = arrhenius_factor(110.0, 100.0, 100.0)
    assert abs(kdep_eff - kdep_ref * factor) < 1e-12
    assert abs(kloss_eff - kloss_ref * factor) < 1e-12


def test_invalid_args_raise():
    try:
        arrhenius_factor(100.0, 100.0, -1.0)
    except ValueError:
        pass
    else:
        raise AssertionError("negative Ea should raise")
    try:
        apply_arrhenius_to_rates(-0.1, 0.005, 100.0, 100.0, 100.0)
    except ValueError:
        pass
    else:
        raise AssertionError("negative kdep_ref should raise")


# ---- FD evolver wrapper -------------------------------------------------

def test_evolve_at_T_ref_matches_phase5():
    """At T = T_ref the wrapper should reproduce the Phase-5 FD evolver
    exactly (the Arrhenius factor is 1)."""
    from reaction_diffusion_peb.src.deprotection import (
        evolve_acid_loss_deprotection_fd,
    )
    H0 = _gaussian_H0()
    H_a, P_a = evolve_acid_loss_deprotection_fd_at_T(
        H0,
        DH_nm2_s=0.8, kdep_ref_s_inv=0.5, kloss_ref_s_inv=0.005,
        temperature_c=100.0, temperature_ref_c=100.0,
        activation_energy_kj_mol=100.0,
        t_end_s=60.0, dx_nm=1.0,
    )
    H_b, P_b = evolve_acid_loss_deprotection_fd(
        H0, DH_nm2_s=0.8, kloss_s_inv=0.005, kdep_s_inv=0.5,
        t_end_s=60.0, dx_nm=1.0,
    )
    assert torch.allclose(H_a, H_b)
    assert torch.allclose(P_a, P_b)


def test_higher_T_gives_higher_P_final():
    """criterion 4: temperature sweep increases P_max and threshold area."""
    H0 = _gaussian_H0()
    P_maxes = []
    areas = []
    for T_c in (80.0, 100.0, 120.0):
        _, P = evolve_acid_loss_deprotection_fd_at_T(
            H0,
            DH_nm2_s=0.8, kdep_ref_s_inv=0.5, kloss_ref_s_inv=0.005,
            temperature_c=T_c, temperature_ref_c=100.0,
            activation_energy_kj_mol=100.0,
            t_end_s=60.0, dx_nm=1.0,
        )
        P_maxes.append(P.max().item())
        areas.append(thresholded_area(P, P_threshold=0.5))
    for k in range(len(P_maxes) - 1):
        assert P_maxes[k] < P_maxes[k + 1] + 1e-9
        assert areas[k] <= areas[k + 1]
    # Sweep should produce a meaningful change
    assert P_maxes[-1] > P_maxes[0] + 1e-3


def test_125c_mor_preset_differs_from_100c():
    """criterion 5: 125°C MOR preset behaves differently from T_ref."""
    H0 = _gaussian_H0()
    _, P_100 = evolve_acid_loss_deprotection_fd_at_T(
        H0, DH_nm2_s=0.8, kdep_ref_s_inv=0.5, kloss_ref_s_inv=0.005,
        temperature_c=100.0, temperature_ref_c=100.0,
        activation_energy_kj_mol=100.0,
        t_end_s=60.0, dx_nm=1.0,
    )
    _, P_125 = evolve_acid_loss_deprotection_fd_at_T(
        H0, DH_nm2_s=0.8, kdep_ref_s_inv=0.5, kloss_ref_s_inv=0.005,
        temperature_c=125.0, temperature_ref_c=100.0,
        activation_energy_kj_mol=100.0,
        t_end_s=60.0, dx_nm=1.0,
    )
    assert P_125.max().item() > P_100.max().item() + 1e-3


def test_zero_Ea_makes_temperature_irrelevant():
    """criterion 3 (FD-level): with Ea = 0, evolution at any T matches T_ref."""
    H0 = _gaussian_H0()
    _, P_ref = evolve_acid_loss_deprotection_fd_at_T(
        H0, DH_nm2_s=0.8, kdep_ref_s_inv=0.5, kloss_ref_s_inv=0.005,
        temperature_c=100.0, temperature_ref_c=100.0,
        activation_energy_kj_mol=0.0,
        t_end_s=60.0, dx_nm=1.0,
    )
    _, P_125 = evolve_acid_loss_deprotection_fd_at_T(
        H0, DH_nm2_s=0.8, kdep_ref_s_inv=0.5, kloss_ref_s_inv=0.005,
        temperature_c=125.0, temperature_ref_c=100.0,
        activation_energy_kj_mol=0.0,
        t_end_s=60.0, dx_nm=1.0,
    )
    assert torch.allclose(P_ref, P_125)

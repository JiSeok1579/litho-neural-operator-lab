"""Arrhenius temperature correction for PEB reaction rates.

Phase 6 introduces only this single piece of physics on top of Phase 5:
the deprotection and acid-loss rates become temperature-dependent
through the Arrhenius form

    k(T) = k_ref * exp(-Ea/R * (1/T_K - 1/T_ref_K))

with

    T_K     = temperature_c + 273.15           (target temperature, K)
    T_ref_K = temperature_ref_c + 273.15       (reference temperature, K)
    R       = 8.314 J / (mol K)
    Ea      = activation energy (Joules / mol; the input here is in
              kJ / mol for human readability)

Conventions:
- Positive ``Ea`` and ``T > T_ref`` give a multiplicative factor ``> 1``
  (rates speed up).
- ``T == T_ref`` gives a factor of exactly 1.
- ``Ea == 0`` removes temperature dependence entirely.

This file does NOT introduce any new physics beyond the rate
correction. The H and P PDEs are exactly Phase 4 + Phase 5 with
``k_loss`` and ``k_dep`` replaced by their Arrhenius-corrected values.
"""

from __future__ import annotations

import math


# --- physical constants ----------------------------------------------------

R_GAS_J_PER_MOL_K: float = 8.314           # ideal gas constant
ABSOLUTE_ZERO_C: float = -273.15           # for T_C -> T_K conversion


def celsius_to_kelvin(T_c: float) -> float:
    return float(T_c) - ABSOLUTE_ZERO_C


def arrhenius_factor(
    temperature_c: float,
    temperature_ref_c: float,
    activation_energy_kj_mol: float,
) -> float:
    """Multiplicative Arrhenius rate factor ``k(T) / k(T_ref)``.

    ``activation_energy_kj_mol`` is given in **kJ / mol** for human
    readability and converted to J / mol internally.

    Cases:
    - ``T == T_ref``      ->  factor = 1 exactly.
    - ``Ea == 0``         ->  factor = 1 always.
    - ``Ea > 0, T > Tref``->  factor > 1 (rates speed up).
    """
    if activation_energy_kj_mol < 0:
        raise ValueError("activation_energy_kj_mol must be non-negative")
    Ea_J_per_mol = float(activation_energy_kj_mol) * 1000.0
    if Ea_J_per_mol == 0.0:
        return 1.0
    T_K = celsius_to_kelvin(temperature_c)
    T_ref_K = celsius_to_kelvin(temperature_ref_c)
    if T_K <= 0 or T_ref_K <= 0:
        raise ValueError("absolute temperature must be positive")
    if T_K == T_ref_K:
        return 1.0
    return math.exp(-Ea_J_per_mol / R_GAS_J_PER_MOL_K * (1.0 / T_K - 1.0 / T_ref_K))


def apply_arrhenius_to_rates(
    kdep_ref_s_inv: float,
    kloss_ref_s_inv: float,
    temperature_c: float,
    temperature_ref_c: float,
    activation_energy_kj_mol: float,
) -> tuple[float, float]:
    """Return ``(k_dep(T), k_loss(T))`` after Arrhenius correction.

    Both rates are scaled by the same factor (the assumption is that
    the activation energy is the same for the two reactions; this
    matches the Phase-6 narrow scope and can be extended later).
    """
    if kdep_ref_s_inv < 0:
        raise ValueError("kdep_ref_s_inv must be non-negative")
    if kloss_ref_s_inv < 0:
        raise ValueError("kloss_ref_s_inv must be non-negative")
    factor = arrhenius_factor(
        temperature_c=temperature_c,
        temperature_ref_c=temperature_ref_c,
        activation_energy_kj_mol=activation_energy_kj_mol,
    )
    return kdep_ref_s_inv * factor, kloss_ref_s_inv * factor


def evolve_acid_loss_deprotection_fd_at_T(
    H0,
    DH_nm2_s: float,
    kdep_ref_s_inv: float,
    kloss_ref_s_inv: float,
    temperature_c: float,
    temperature_ref_c: float,
    activation_energy_kj_mol: float,
    t_end_s: float,
    dx_nm: float,
    n_steps: int | None = None,
    cfl_safety: float = 0.5,
):
    """Convenience wrapper: apply Arrhenius to the rates, then call the
    Phase-5 FD evolver. Returns ``(H_final, P_final)``."""
    # Local import to avoid a top-level cycle while keeping the helper
    # available at the module's public surface.
    from reaction_diffusion_peb.src.deprotection import (
        evolve_acid_loss_deprotection_fd,
    )

    kdep_eff, kloss_eff = apply_arrhenius_to_rates(
        kdep_ref_s_inv=kdep_ref_s_inv,
        kloss_ref_s_inv=kloss_ref_s_inv,
        temperature_c=temperature_c,
        temperature_ref_c=temperature_ref_c,
        activation_energy_kj_mol=activation_energy_kj_mol,
    )
    return evolve_acid_loss_deprotection_fd(
        H0,
        DH_nm2_s=DH_nm2_s,
        kloss_s_inv=kloss_eff,
        kdep_s_inv=kdep_eff,
        t_end_s=t_end_s,
        dx_nm=dx_nm,
        n_steps=n_steps,
        cfl_safety=cfl_safety,
    )

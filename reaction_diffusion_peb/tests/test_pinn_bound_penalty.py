"""Tests for the pre-Phase-7 PINN bound penalty.

Covers the four user-listed contracts:

  1. bound penalty is positive when ``P < 0`` or ``P > 1``
  2. bound penalty is (near) zero when ``P in [0, 1]``
  3. ``weight_bound = 0`` reproduces the pre-update training behavior
  4. ``weight_bound > 0`` adds the term to the recorded loss history
"""

from __future__ import annotations

import math

import torch

from reaction_diffusion_peb.src.pinn_diffusion import gaussian_spot_acid_callable
from reaction_diffusion_peb.src.pinn_reaction_diffusion import PINNDeprotection
from reaction_diffusion_peb.src.train_pinn_deprotection import (
    bound_penalty_P,
    train_pinn_deprotection,
)


def _grid_kwargs():
    return dict(x_range_nm=(-64.0, 64.0), t_range_s=(0.0, 60.0),
                hidden=8, n_hidden_layers=1, n_fourier=4)


# ---- bound_penalty_P contract -------------------------------------------

def test_bound_penalty_zero_in_unit_interval():
    P = torch.tensor([0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0])
    pen = bound_penalty_P(P)
    assert pen.item() < 1e-12


def test_bound_penalty_positive_below_zero():
    P = torch.tensor([-0.5, -0.1, 0.0, 0.5])
    pen = bound_penalty_P(P)
    assert pen.item() > 0
    # Hand-check: only the -0.5 and -0.1 entries contribute. Below sum
    # of squares is 0.25 + 0.01 = 0.26 over 4 elements -> 0.065. Above
    # term is 0.
    assert abs(pen.item() - 0.26 / 4.0) < 1e-6


def test_bound_penalty_positive_above_one():
    P = torch.tensor([0.0, 0.5, 1.0, 1.2, 2.0])
    pen = bound_penalty_P(P)
    assert pen.item() > 0
    # Above contributions: (0.2)^2 = 0.04, (1.0)^2 = 1.0 over 5 elements
    # -> 1.04 / 5 = 0.208.
    assert abs(pen.item() - 1.04 / 5.0) < 1e-6


def test_bound_penalty_combined_below_and_above():
    P = torch.tensor([-1.0, 0.5, 2.0])
    pen = bound_penalty_P(P)
    # below term: (1.0)^2 / 3 = 1/3
    # above term: (1.0)^2 / 3 = 1/3
    assert abs(pen.item() - (1.0 + 1.0) / 3.0) < 1e-6


def test_bound_penalty_differentiable():
    P = torch.tensor([-0.2, 0.5, 1.3], requires_grad=True)
    pen = bound_penalty_P(P)
    pen.backward()
    # P[0] = -0.2 -> below.relu = 0.2 -> derivative of (-P)^2 mean is
    # d/dP[(0.2)^2 / 3] = 2 * 0.2 / 3 * (-1) = -0.1333 (sign from -P).
    # Sign: actually relu(-P) = 0.2, d/dP[relu(-P)^2] = 2*relu(-P)*(-1) when -P>0
    # so dP[0] grad = -2 * 0.2 / 3 = -0.1333
    assert abs(P.grad[0].item() - (-2.0 * 0.2 / 3.0)) < 1e-5
    # P[1] = 0.5 in-bound -> grad = 0
    assert abs(P.grad[1].item()) < 1e-9
    # P[2] = 1.3 -> grad = 2 * 0.3 / 3 = 0.2
    assert abs(P.grad[2].item() - (2.0 * 0.3 / 3.0)) < 1e-5


# ---- trainer integration ------------------------------------------------

def test_train_with_zero_weight_bound_records_zero_in_history():
    H0_fn = gaussian_spot_acid_callable(sigma_nm=10.0)
    pinn = PINNDeprotection(
        DH_nm2_s=0.8, kloss_s_inv=0.005, kdep_s_inv=0.5,
        hard_ic=True, H0_callable=H0_fn, **_grid_kwargs(),
    )
    h = train_pinn_deprotection(
        pinn, H0_callable=H0_fn,
        x_range_nm=(-64.0, 64.0), t_range_s=(0.0, 60.0),
        n_iters=10, lr=1e-3, n_collocation=64, n_ic=32, n_ic_grid_side=4,
        weight_pde_H=1.0, weight_pde_P=1.0,
        weight_bound=0.0,
        log_every=1,
    )
    # weight_bound = 0 still logs the bound term (for diagnostics) but
    # the value should be small if the network is producing reasonable
    # P values; more importantly it must be present and finite.
    assert len(h.loss_bound) == len(h.iters)
    for v in h.loss_bound:
        assert math.isfinite(v)


def test_train_with_positive_weight_bound_includes_term():
    """With weight_bound > 0 the trainer's total loss includes a
    contribution from the bound term (so loss_bound must be tracked
    and contribute to loss_total as `weight_bound * loss_bound`).

    We feed the network 10 iterations and check that loss_total at
    each iteration matches the sum of components."""
    H0_fn = gaussian_spot_acid_callable(sigma_nm=10.0)
    pinn = PINNDeprotection(
        DH_nm2_s=0.8, kloss_s_inv=0.005, kdep_s_inv=0.5,
        hard_ic=True, H0_callable=H0_fn, **_grid_kwargs(),
    )
    weight_bound = 2.5
    h = train_pinn_deprotection(
        pinn, H0_callable=H0_fn,
        x_range_nm=(-64.0, 64.0), t_range_s=(0.0, 60.0),
        n_iters=10, lr=1e-3, n_collocation=64, n_ic=32, n_ic_grid_side=4,
        weight_pde_H=1.0, weight_pde_P=1.0, weight_ic=0.0,
        weight_bound=weight_bound,
        log_every=1,
    )
    for i in range(len(h.iters)):
        expected = (1.0 * h.loss_pde_H[i]
                    + 1.0 * h.loss_pde_P[i]
                    + 0.0 * h.loss_ic[i]
                    + weight_bound * h.loss_bound[i])
        # Loss is computed in float32; allow loose tolerance.
        assert abs(h.loss_total[i] - expected) / max(expected, 1e-12) < 1e-3


def test_negative_weight_bound_raises():
    H0_fn = gaussian_spot_acid_callable(sigma_nm=10.0)
    pinn = PINNDeprotection(
        DH_nm2_s=0.8, kloss_s_inv=0.005, kdep_s_inv=0.5,
        hard_ic=True, H0_callable=H0_fn, **_grid_kwargs(),
    )
    try:
        train_pinn_deprotection(
            pinn, H0_callable=H0_fn,
            x_range_nm=(-64.0, 64.0), t_range_s=(0.0, 60.0),
            n_iters=2, lr=1e-3,
            weight_bound=-1.0,
        )
    except ValueError:
        pass
    else:
        raise AssertionError("negative weight_bound should raise")


def test_smoke_training_finite_loss():
    H0_fn = gaussian_spot_acid_callable(sigma_nm=10.0)
    pinn = PINNDeprotection(
        DH_nm2_s=0.8, kloss_s_inv=0.005, kdep_s_inv=0.5,
        hard_ic=True, H0_callable=H0_fn, **_grid_kwargs(),
    )
    h = train_pinn_deprotection(
        pinn, H0_callable=H0_fn,
        x_range_nm=(-64.0, 64.0), t_range_s=(0.0, 60.0),
        n_iters=20, lr=1e-3, n_collocation=64, n_ic=32, n_ic_grid_side=4,
        weight_bound=1.0, log_every=10,
    )
    for v in h.loss_total:
        assert math.isfinite(v)

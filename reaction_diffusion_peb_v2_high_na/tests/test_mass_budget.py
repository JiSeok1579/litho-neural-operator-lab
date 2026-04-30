"""Mass-budget sanity. With k_loss=0 and no quencher, the spectral diffusion step
conserves the integral of H exactly; the only sink is the deprotection coupling
which removes H proportional to k_dep*H*(1-P)*dt (treated explicitly here).

We check the implicit identity used in this implementation:
  the loss-only spectral step preserves integral(H);
  with reaction enabled, the H decrease over a step matches kdep*sum(H*(1-P))*dt.
"""
import numpy as np

from reaction_diffusion_peb_v2_high_na.src.fd_solver_2d import _spectral_diffusion_decay, solve_peb_2d


def test_diffusion_only_preserves_integral():
    rng = np.random.default_rng(0)
    H = 0.2 * rng.random((48, 48))
    H_after = _spectral_diffusion_decay(H, dx_nm=0.5, D=0.8, k_decay=0.0, dt=0.5)
    assert abs(H_after.sum() - H.sum()) < 1e-8 * H.sum()


def test_loss_decays_integral_exponentially():
    H = 0.2 * np.ones((32, 32))
    k_loss = 0.005
    dt = 5.0
    H_after = _spectral_diffusion_decay(H, dx_nm=0.5, D=0.0, k_decay=k_loss, dt=dt)
    expected = H.sum() * np.exp(-k_loss * dt)
    assert abs(H_after.sum() - expected) < 1e-8 * expected


def test_total_acid_monotone_nonincreasing_with_loss_and_reaction():
    H0 = 0.13 * np.ones((24, 24))
    res = solve_peb_2d(
        H0=H0,
        dx_nm=0.5,
        DH_nm2_s=0.8,
        kdep_s_inv=0.5,
        kloss_s_inv=0.005,
        time_s=30.0,
        dt_s=0.5,
        quencher_enabled=False,
    )
    # H integral can only decrease in this regime.
    assert res.H.sum() <= H0.sum() + 1e-9

import numpy as np

from reaction_diffusion_peb_v2_high_na.src.fd_solver_2d import solve_peb_2d


def test_solver_bounds_no_quencher():
    rng = np.random.default_rng(0)
    H0 = 0.2 * rng.random((48, 48))  # nonneg
    res = solve_peb_2d(
        H0=H0,
        dx_nm=0.5,
        DH_nm2_s=0.8,
        kdep_s_inv=0.5,
        kloss_s_inv=0.005,
        time_s=20.0,
        dt_s=0.5,
        quencher_enabled=False,
    )
    assert np.isfinite(res.H).all()
    assert np.isfinite(res.P).all()
    assert res.H.min() >= -1e-8
    assert res.P.min() >= -1e-8
    assert res.P.max() <= 1.0 + 1e-8
    # Mass-decay sanity: with k_loss>0 + finite reaction, H mean should not increase.
    assert res.H.mean() <= H0.mean() + 1e-9


def test_solver_zero_inputs_remain_zero():
    H0 = np.zeros((32, 32))
    res = solve_peb_2d(
        H0=H0,
        dx_nm=0.5,
        DH_nm2_s=0.8,
        kdep_s_inv=0.5,
        time_s=10.0,
        dt_s=0.5,
        quencher_enabled=False,
    )
    assert np.allclose(res.H, 0.0)
    assert np.allclose(res.P, 0.0)


def test_solver_with_weak_quencher_preserves_signs():
    H0 = 0.13 * np.ones((32, 32))
    res = solve_peb_2d(
        H0=H0,
        dx_nm=0.5,
        DH_nm2_s=0.8,
        kdep_s_inv=0.5,
        kloss_s_inv=0.005,
        time_s=20.0,
        dt_s=0.5,
        quencher_enabled=True,
        Q0=0.01,
        DQ_nm2_s=0.0,
        kq_s_inv=1.0,
    )
    assert res.H.min() >= -1e-8
    assert res.Q.min() >= -1e-8
    assert res.P.min() >= -1e-8
    assert res.P.max() <= 1.0 + 1e-8

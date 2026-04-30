import numpy as np

from reaction_diffusion_peb_v2_high_na.src.exposure_high_na import (
    build_xz_intensity,
    dill_acid_generation,
    line_space_intensity_1d,
)
from reaction_diffusion_peb_v2_high_na.src.fd_solver_xz import (
    _even_mirror_extend_z,
    _spectral_diffusion_decay_xz,
    solve_peb_xz,
)


def test_even_mirror_length_for_typical_grid():
    field = np.arange(4 * 6).reshape(4, 6).astype(np.float64)
    ext = _even_mirror_extend_z(field)
    assert ext.shape == (6, 6)
    np.testing.assert_array_equal(ext[0], field[0])
    np.testing.assert_array_equal(ext[3], field[3])
    np.testing.assert_array_equal(ext[4], field[2])
    np.testing.assert_array_equal(ext[5], field[1])


def _trapz_xz_integral(field: np.ndarray) -> float:
    """Trapezoidal integral over z (with weight 0.5 at z=0 and z=Lz) summed over x.
    This is the discrete integral preserved by the even-mirror FFT trick."""
    return float(field[1:-1].sum() + 0.5 * (field[0].sum() + field[-1].sum()))


def test_diffusion_only_preserves_trapezoidal_z_integral():
    """Even-mirror FFT diffusion preserves the trapezoidal z-integral times the
    plain x-sum, not the naive double sum. Verify the right invariant."""
    rng = np.random.default_rng(0)
    H = 0.2 * rng.random((30, 60))
    H_after = _spectral_diffusion_decay_xz(H, dx_nm=0.5, dz_nm=0.5, D=0.5,
                                             k_decay=0.0, dt=0.5)
    inv_before = _trapz_xz_integral(H)
    inv_after = _trapz_xz_integral(H_after)
    assert abs(inv_after - inv_before) < 1e-7 * abs(inv_before)


def test_diffusion_only_naive_sum_changes_within_boundary_correction():
    """The naive total sum is not exactly preserved (it differs by half-weights
    on the two z-boundaries). The discrepancy must be small relative to the
    boundary-row sums."""
    rng = np.random.default_rng(0)
    H = 0.2 * rng.random((30, 60))
    H_after = _spectral_diffusion_decay_xz(H, dx_nm=0.5, dz_nm=0.5, D=0.5,
                                             k_decay=0.0, dt=0.5)
    # the difference is at most one boundary-row's worth (≤ 1 * row_sum)
    boundary_scale = max(H[0].sum() + H[-1].sum(), 1e-12)
    assert abs(H_after.sum() - H.sum()) < boundary_scale


def test_loss_only_decays_z_integral_exponentially():
    H = 0.2 * np.ones((30, 40))
    k_loss = 0.005
    dt = 4.0
    H_after = _spectral_diffusion_decay_xz(H, dx_nm=0.5, dz_nm=0.5, D=0.0,
                                             k_decay=k_loss, dt=dt)
    expected = H.sum() * np.exp(-k_loss * dt)
    assert abs(H_after.sum() - expected) < 1e-7 * expected


def test_solver_bounds_xz_no_quencher():
    rng = np.random.default_rng(1)
    H0 = 0.2 * rng.random((30, 40))
    res = solve_peb_xz(H0=H0, dx_nm=0.5, dz_nm=0.5, DH_nm2_s=0.5,
                        kdep_s_inv=0.5, kloss_s_inv=0.005,
                        time_s=10.0, dt_s=0.5)
    assert np.isfinite(res.H).all()
    assert np.isfinite(res.P).all()
    assert res.H.min() >= -1e-8
    assert res.P.min() >= -1e-8
    assert res.P.max() <= 1.0 + 1e-8
    assert res.H.sum() <= H0.sum() + 1e-8


def test_xz_exposure_zero_amplitude_is_separable():
    I_x, x, _ = line_space_intensity_1d(domain_x_nm=120.0, grid_spacing_nm=0.5,
                                          pitch_nm=24.0, line_cd_nm=12.5)
    z = np.arange(40) * 0.5
    I_xz = build_xz_intensity(I_x, z_nm=z, standing_wave_period_nm=6.75,
                                standing_wave_amplitude=0.0,
                                absorption_length_nm=30.0)
    # Without modulation, every z-row should be I_x scaled by the same envelope.
    env = np.exp(-z / 30.0)
    expected = env[:, None] * I_x[None, :]
    np.testing.assert_allclose(I_xz, expected)


def test_xz_exposure_with_amplitude_has_z_modulation():
    I_x, x, _ = line_space_intensity_1d(domain_x_nm=120.0, grid_spacing_nm=0.5,
                                          pitch_nm=24.0, line_cd_nm=12.5)
    z = np.arange(40) * 0.5
    I_xz = build_xz_intensity(I_x, z_nm=z, standing_wave_period_nm=6.75,
                                standing_wave_amplitude=0.20,
                                absorption_length_nm=None)
    line_col = int(np.argmax(I_x))
    line_strip = I_xz[:, line_col]
    relative_modulation = (line_strip.max() - line_strip.min()) / line_strip.mean()
    assert relative_modulation > 0.3

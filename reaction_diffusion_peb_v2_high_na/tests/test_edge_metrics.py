import numpy as np

from reaction_diffusion_peb_v2_high_na.src.metrics_edge import (
    CD_LOCK_HIGH,
    CD_LOCK_LOW,
    CD_LOCK_OK,
    compute_edge_band_powers,
    extract_edges,
    find_cd_lock_threshold,
    stack_lr_edges,
)


def _make_smooth_lines(nx=128, ny=64, pitch=24.0, cd=12.0, dx=0.5, sharpness=20.0):
    x = (np.arange(nx) + 0.5) * dx
    y = (np.arange(ny) + 0.5) * dx
    centers = (np.arange(int(np.floor(nx * dx / pitch))) + 0.5) * pitch
    field = np.zeros((ny, nx))
    for c in centers:
        # Smooth box: sigmoid * sigmoid with sharpness.
        d_left = (x - (c - cd / 2.0)) * sharpness
        d_right = ((c + cd / 2.0) - x) * sharpness
        line = 1.0 / (1.0 + np.exp(-d_left)) * 1.0 / (1.0 + np.exp(-d_right))
        field += line[None, :] * np.ones((ny, 1))
    return field, x, y, centers, dx, pitch


def test_extract_edges_smooth_box_recovers_cd():
    field, x, y, centers, dx, pitch = _make_smooth_lines()
    res = extract_edges(field, x_nm=x, line_centers_nm=centers, pitch_nm=pitch, threshold=0.5)
    assert res.cd_overall_mean_nm > 0.0
    assert abs(res.cd_overall_mean_nm - 12.0) < 0.5
    # No roughness in y → LER and LWR ~ 0.
    assert res.ler_mean_nm < 0.05
    assert res.lwr_mean_nm < 0.05


def test_extract_edges_handles_no_crossing():
    nx, ny, dx = 64, 16, 0.5
    x = (np.arange(nx) + 0.5) * dx
    field = np.zeros((ny, nx))  # all sub-threshold
    centers = np.array([16.0])
    res = extract_edges(field, x_nm=x, line_centers_nm=centers, pitch_nm=24.0, threshold=0.5)
    assert np.isnan(res.left_edges_nm).all()
    assert np.isnan(res.right_edges_nm).all()


def test_psd_bands_concentrate_input_frequency():
    """A pure sinusoidal edge at a known wavelength should put all power in the matching band."""
    ny = 240
    dy = 0.5
    y = np.arange(ny) * dy
    # 8 nm wavelength -> freq 0.125 nm^-1 -> mid band [0.05, 0.20)
    edge_track = 1.0 * np.sin(2 * np.pi * y / 8.0)
    powers = compute_edge_band_powers(edge_track[None, :], dy_nm=dy)
    assert powers[1] > powers[0]
    assert powers[1] > powers[2]
    assert powers[0] / powers[1] < 0.05
    assert powers[2] / powers[1] < 0.05


def test_psd_bands_zero_signal_zero_power():
    powers = compute_edge_band_powers(np.zeros((4, 32)), dy_nm=0.5)
    assert np.allclose(powers, 0.0)


def test_stack_lr_edges_concatenates():
    field, x, y, centers, _, pitch = _make_smooth_lines()
    res = extract_edges(field, x_nm=x, line_centers_nm=centers, pitch_nm=pitch, threshold=0.5)
    stacked = stack_lr_edges(res)
    assert stacked.shape[0] == 2 * res.left_edges_nm.shape[0]
    assert stacked.shape[1] == res.left_edges_nm.shape[1]


def test_cd_lock_finds_threshold_for_smooth_box():
    """Smooth box with edges at ±cd/2: locking to a target CD inside the
    achievable range should converge."""
    # gentle sharpness so that CD(threshold) varies measurably across [0.2, 0.8]
    field, x, y, centers, _, pitch = _make_smooth_lines(cd=12.0, sharpness=2.0)
    # CD(P=0.8) ≈ 10.6, CD(P=0.5) = 12.0, CD(P=0.2) ≈ 13.4 → target 11.0 is reachable
    P_locked, cd_locked, status = find_cd_lock_threshold(
        field, x_nm=x, line_centers_nm=centers, pitch_nm=pitch,
        cd_target_nm=11.0, cd_tol_nm=0.1,
    )
    assert status == CD_LOCK_OK, f"unexpected status {status}"
    assert P_locked is not None and 0.2 < P_locked < 0.8
    assert abs(cd_locked - 11.0) < 0.1


def test_cd_lock_target_below_min_threshold_returns_high_bound():
    """A very small target CD should not be reachable in [0.2, 0.8]: the
    contour at P=0.8 is still wider than 1 nm in a smooth box."""
    field, x, y, centers, _, pitch = _make_smooth_lines(cd=12.0, sharpness=5.0)
    P_locked, cd_locked, status = find_cd_lock_threshold(
        field, x_nm=x, line_centers_nm=centers, pitch_nm=pitch,
        cd_target_nm=0.5, cd_tol_nm=0.1,
    )
    assert status == CD_LOCK_HIGH
    assert P_locked == 0.8
    assert cd_locked > 0.5


def test_cd_lock_target_above_max_threshold_returns_low_bound():
    """A very large target CD should not be reachable: contour at P=0.2 is
    still narrower than ~half-pitch."""
    field, x, y, centers, _, pitch = _make_smooth_lines(cd=8.0, sharpness=5.0)
    P_locked, cd_locked, status = find_cd_lock_threshold(
        field, x_nm=x, line_centers_nm=centers, pitch_nm=pitch,
        cd_target_nm=20.0, cd_tol_nm=0.1,
    )
    assert status == CD_LOCK_LOW
    assert P_locked == 0.2
    assert cd_locked < 20.0

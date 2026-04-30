import numpy as np

from reaction_diffusion_peb_v2_high_na.src.geometry import line_centers, line_space_intensity


def test_line_centers_count_and_spacing():
    centers = line_centers(domain_x_nm=128.0, pitch_nm=24.0)
    # floor(128/24) = 5
    assert centers.size == 5
    diffs = np.diff(centers)
    assert np.allclose(diffs, 24.0)
    assert np.isclose(centers[0], 12.0)


def test_line_space_intensity_no_roughness_geometry():
    I, grid = line_space_intensity(
        domain_x_nm=48.0,
        domain_y_nm=8.0,
        grid_spacing_nm=0.5,
        pitch_nm=24.0,
        line_cd_nm=12.0,
        edge_roughness_amp_nm=0.0,
    )
    assert I.shape == (16, 96)
    assert set(np.unique(I).tolist()).issubset({0.0, 1.0})
    # Each line covers half the pitch ⇒ ~50% area filled.
    assert 0.45 < I.mean() < 0.55
    assert grid.line_centers_nm.size == 2


def test_line_space_with_roughness_keeps_binary():
    I, _ = line_space_intensity(
        domain_x_nm=48.0,
        domain_y_nm=64.0,
        grid_spacing_nm=0.5,
        pitch_nm=24.0,
        line_cd_nm=12.0,
        edge_roughness_amp_nm=1.0,
        edge_roughness_corr_nm=5.0,
        edge_roughness_seed=7,
    )
    assert set(np.unique(I).tolist()).issubset({0.0, 1.0})
    # Line area shouldn't drift far from 50%.
    assert 0.40 < I.mean() < 0.60

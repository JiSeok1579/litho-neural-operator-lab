import numpy as np

from reaction_diffusion_peb_v2_high_na.src.metrics_edge import extract_edges


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

"""Phase 10 tests — FNO surrogate building blocks and dataset adapter."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from reaction_diffusion_peb.src.dataset_builder import (
    generate_sample,
    random_safe_spec,
)
from reaction_diffusion_peb.src.fno_surrogate import (
    ChannelStats,
    FNO2d,
    INPUT_FIELD_NAMES,
    INPUT_SCALAR_NAMES,
    OUTPUT_FIELD_NAMES,
    OUTPUT_FIELD_NAMES_WITH_R,
    SpectralConv2d,
    area_error,
    build_fno_for_dataset,
    fit_channel_stats,
    make_input_tensor,
    make_output_tensor,
    make_output_tensor_with_R,
    manual_seed_everything,
    mask_iou_from_logits,
    p_max_error,
    per_channel_relative_l2,
    relative_l2,
    thresholded_iou,
)


GRID = 32


def _build_arrays(n: int = 4, grid: int = GRID) -> dict:
    """Synthesize a tiny Phase-9-shaped dictionary of arrays."""
    rng = np.random.default_rng(0)
    samples = []
    for _ in range(n):
        spec = random_safe_spec(rng, grid_size=grid)
        samples.append(generate_sample(spec, grid_size=grid, dx_nm=1.0))
    arrays = {
        "H0": np.stack([s.H0 for s in samples]),
        "H_final": np.stack([s.H_final for s in samples]),
        "P_final": np.stack([s.P_final for s in samples]),
        "R": np.stack([s.R for s in samples]),
    }
    for name in INPUT_SCALAR_NAMES:
        arrays[name] = np.array(
            [getattr(s.spec, name if name != "Hmax" else "Hmax")
             for s in samples],
            dtype=np.float32,
        )
    return arrays


# ---- SpectralConv2d ----------------------------------------------------

def test_spectral_conv_preserves_shape():
    layer = SpectralConv2d(in_channels=3, out_channels=5, modes_x=4, modes_y=4)
    x = torch.randn(2, 3, 16, 16)
    y = layer(x)
    assert y.shape == (2, 5, 16, 16)
    assert y.dtype == torch.float32


def test_spectral_conv_modes_clamp_to_grid():
    """modes_x/modes_y bigger than the grid must not error — they
    should be silently clamped to the available band."""
    layer = SpectralConv2d(2, 2, modes_x=64, modes_y=64)
    x = torch.randn(1, 2, 8, 8)
    y = layer(x)
    assert y.shape == (1, 2, 8, 8)


def test_spectral_conv_rejects_zero_modes():
    with pytest.raises(ValueError):
        SpectralConv2d(1, 1, modes_x=0, modes_y=4)


# ---- FNO2d -------------------------------------------------------------

def test_fno2d_forward_shape():
    model = FNO2d(in_channels=4, out_channels=2,
                  width=8, n_blocks=2, modes_x=4, modes_y=4)
    x = torch.randn(3, 4, GRID, GRID)
    y = model(x)
    assert y.shape == (3, 2, GRID, GRID)


def test_fno2d_deterministic_under_seed():
    manual_seed_everything(42)
    m1 = FNO2d(in_channels=2, out_channels=1, width=4, n_blocks=2,
               modes_x=4, modes_y=4)
    manual_seed_everything(42)
    m2 = FNO2d(in_channels=2, out_channels=1, width=4, n_blocks=2,
               modes_x=4, modes_y=4)
    x = torch.randn(2, 2, GRID, GRID)
    assert torch.allclose(m1(x), m2(x))


def test_fno2d_num_parameters_positive():
    model = FNO2d(in_channels=2, out_channels=2, width=4, n_blocks=2,
                  modes_x=4, modes_y=4)
    assert model.num_parameters() > 0


# ---- dataset adapter ---------------------------------------------------

def test_make_input_tensor_channel_layout():
    arrays = _build_arrays(n=4)
    x = make_input_tensor(arrays)
    n_chan = len(INPUT_FIELD_NAMES) + len(INPUT_SCALAR_NAMES)
    assert x.shape == (4, n_chan, GRID, GRID)
    # Channel 0 must equal H0 exactly.
    H0 = torch.from_numpy(arrays["H0"]).float()
    assert torch.allclose(x[:, 0], H0)
    # Channel 1 (DH_nm2_s) is constant per sample — identical across pixels.
    for i in range(4):
        v = x[i, 1]
        assert torch.allclose(v, v.flatten()[0].expand_as(v))


def test_make_input_tensor_temperature_centered_on_Tref():
    arrays = _build_arrays(n=2)
    x = make_input_tensor(arrays)
    T_idx = 1 + INPUT_SCALAR_NAMES.index("temperature_c")
    raw_T = arrays["temperature_c"]
    for i in range(2):
        actual = float(x[i, T_idx, 0, 0].item())
        expected = float(raw_T[i] - 100.0)
        assert abs(actual - expected) < 1e-5


def test_make_output_tensor_channel_layout():
    arrays = _build_arrays(n=3)
    y = make_output_tensor(arrays)
    assert y.shape == (3, len(OUTPUT_FIELD_NAMES), GRID, GRID)
    H_final = torch.from_numpy(arrays["H_final"]).float()
    P_final = torch.from_numpy(arrays["P_final"]).float()
    assert torch.allclose(y[:, 0], H_final)
    assert torch.allclose(y[:, 1], P_final)


# ---- ChannelStats ------------------------------------------------------

def test_channel_stats_round_trip():
    x = torch.randn(8, 3, 16, 16) * 5 + 7
    stats = fit_channel_stats(x)
    z = stats.normalize(x)
    # After whitening, mean ≈ 0 and std ≈ 1 per channel.
    assert torch.allclose(z.mean(dim=(0, 2, 3)), torch.zeros(3), atol=1e-5)
    assert torch.allclose(z.std(dim=(0, 2, 3)), torch.ones(3), atol=1e-3)
    # Round-trip recovery.
    x_back = stats.denormalize(z)
    assert torch.allclose(x_back, x, atol=1e-5)


def test_channel_stats_handles_constant_channel():
    """A constant input channel has std=0; the eps clamp must not
    produce NaN/Inf."""
    x = torch.zeros(4, 2, 8, 8)
    x[:, 0] = 3.7
    stats = fit_channel_stats(x)
    z = stats.normalize(x)
    assert torch.isfinite(z).all()


# ---- metrics -----------------------------------------------------------

def test_relative_l2_zero_when_pred_eq_target():
    target = torch.randn(3, 2, 8, 8)
    pred = target.clone()
    err = relative_l2(pred, target)
    assert torch.allclose(err, torch.zeros(3), atol=1e-7)


def test_per_channel_relative_l2_zero_when_pred_eq_target():
    target = torch.randn(3, 2, 8, 8)
    err = per_channel_relative_l2(target, target)
    assert torch.allclose(err, torch.zeros(2), atol=1e-7)


def test_thresholded_iou_perfect_match():
    pred = torch.zeros(2, 1, 8, 8)
    target = torch.zeros(2, 1, 8, 8)
    pred[0, 0, :4, :4] = 1.0
    target[0, 0, :4, :4] = 1.0
    iou = thresholded_iou(pred, target)
    assert torch.allclose(iou[0:1], torch.ones(1))
    # Both empty -> union is clamped to 1, IoU is 0/1 = 0.
    assert torch.allclose(iou[1:2], torch.zeros(1))


def test_thresholded_iou_no_overlap():
    pred = torch.zeros(1, 1, 8, 8)
    target = torch.zeros(1, 1, 8, 8)
    pred[0, 0, :4, :4] = 1.0
    target[0, 0, 4:, 4:] = 1.0
    iou = thresholded_iou(pred, target)
    assert float(iou.item()) == 0.0


# ---- build_fno_for_dataset ---------------------------------------------

def test_build_fno_for_dataset_matches_adapter():
    model = build_fno_for_dataset(width=4, n_blocks=2, modes_x=4, modes_y=4)
    arrays = _build_arrays(n=2)
    x = make_input_tensor(arrays)
    y = model(x)
    assert y.shape[1] == len(OUTPUT_FIELD_NAMES)
    assert y.shape[2] == x.shape[2]
    assert y.shape[3] == x.shape[3]


def test_build_fno_for_dataset_with_R_head_has_three_outputs():
    model = build_fno_for_dataset(width=4, n_blocks=2, modes_x=4, modes_y=4,
                                   with_R_head=True)
    arrays = _build_arrays(n=2)
    x = make_input_tensor(arrays)
    y = model(x)
    assert y.shape[1] == len(OUTPUT_FIELD_NAMES_WITH_R) == 3


def test_make_output_tensor_with_R_channel_layout():
    arrays = _build_arrays(n=3)
    y = make_output_tensor_with_R(arrays)
    assert y.shape == (3, 3, GRID, GRID)
    np.testing.assert_allclose(y[:, 0].numpy(), arrays["H_final"])
    np.testing.assert_allclose(y[:, 1].numpy(), arrays["P_final"])
    np.testing.assert_allclose(y[:, 2].numpy(), arrays["R"])
    # R is binary
    uniq = torch.unique(y[:, 2])
    assert set(uniq.tolist()).issubset({0.0, 1.0})


# ---- area / p_max metrics ----------------------------------------------

def test_area_error_zero_when_masks_match():
    P = torch.zeros(3, 1, 8, 8)
    P[0, 0, :3, :3] = 1.0
    P[1, 0, 4:, 4:] = 1.0
    err = area_error(P, P, threshold=0.5)
    assert torch.allclose(err, torch.zeros(3))


def test_area_error_counts_pixel_difference():
    pred = torch.zeros(1, 1, 8, 8)
    pred[0, 0, :2, :2] = 1.0           # 4 pixels
    target = torch.zeros(1, 1, 8, 8)
    target[0, 0, :3, :3] = 1.0         # 9 pixels
    err = area_error(pred, target, threshold=0.5)
    assert float(err.item()) == 5.0


def test_p_max_error_zero_when_pred_eq_target():
    P = torch.rand(3, 1, 8, 8)
    err = p_max_error(P, P)
    assert torch.allclose(err, torch.zeros(3), atol=1e-7)


def test_p_max_error_picks_per_sample_max_diff():
    pred = torch.zeros(2, 1, 4, 4)
    target = torch.zeros(2, 1, 4, 4)
    pred[0, 0, 0, 0] = 0.7
    target[0, 0, 0, 0] = 0.9
    pred[1, 0, 1, 1] = 0.2
    target[1, 0, 1, 1] = 0.8
    err = p_max_error(pred, target)
    assert torch.allclose(err, torch.tensor([0.2, 0.6]), atol=1e-6)


# ---- mask IoU from logits ----------------------------------------------

def test_mask_iou_from_logits_perfect_match():
    target = torch.zeros(1, 1, 8, 8)
    target[0, 0, :3, :3] = 1.0
    # Logit > 0 means sigmoid > 0.5; -> matches target
    logit = torch.full((1, 1, 8, 8), -10.0)
    logit[0, 0, :3, :3] = +10.0
    iou = mask_iou_from_logits(logit, target)
    assert float(iou.item()) == pytest.approx(1.0)


def test_mask_iou_from_logits_threshold_at_zero():
    """A logit exactly at 0 is treated as the negative class
    (matches the documented ``> 0`` rule)."""
    target = torch.zeros(1, 1, 4, 4)
    target[0, 0, :2, :2] = 1.0
    logit = torch.full((1, 1, 4, 4), -1.0)
    logit[0, 0, :2, :2] = 0.0          # NOT > 0
    iou = mask_iou_from_logits(logit, target)
    assert float(iou.item()) == 0.0    # no overlap, target has 4 pixels

"""Phase 9 tests — dataset_builder utilities."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from reaction_diffusion_peb.src.dataset_builder import (
    AERIAL_KIND_CODES,
    DATASET_FIELD_NAMES,
    DATASET_SCALAR_NAMES,
    SampleArrays,
    SampleSpec,
    aerial_from_spec,
    generate_sample,
    load_dataset,
    make_split_indices,
    parameter_ranges,
    random_safe_spec,
    random_stiff_spec,
    save_dataset,
)


GRID = 32          # small grid for fast tests
DX = 1.0


@pytest.fixture
def tmp_dataset_dir(tmp_path):
    return tmp_path


# ---- aerial dispatch ----------------------------------------------------

def test_aerial_from_spec_supports_every_kind():
    for kind in AERIAL_KIND_CODES:
        if kind == "gaussian_spot":
            spec = SampleSpec(aerial_kind=kind, aerial_param_a=4.0)
        elif kind == "line_space":
            spec = SampleSpec(aerial_kind=kind, aerial_param_a=12.0,
                              aerial_param_b=0.5)
        elif kind == "contact_array":
            spec = SampleSpec(aerial_kind=kind, aerial_param_a=12.0,
                              aerial_param_b=3.0)
        elif kind == "two_spot":
            spec = SampleSpec(aerial_kind=kind, aerial_param_a=4.0,
                              aerial_param_b=12.0)
        I = aerial_from_spec(spec, grid_size=GRID)
        assert I.shape == (GRID, GRID)
        assert torch.is_tensor(I)
        assert float(I.min().item()) >= 0.0
        assert float(I.max().item()) <= 1.0 + 1e-6


def test_aerial_from_spec_rejects_unknown_kind():
    spec = SampleSpec(aerial_kind="nope", aerial_param_a=4.0)
    with pytest.raises(ValueError):
        aerial_from_spec(spec, grid_size=GRID)


# ---- random spec sampling ----------------------------------------------

def test_random_safe_spec_is_deterministic_with_seed():
    rng_a = np.random.default_rng(0)
    rng_b = np.random.default_rng(0)
    s1 = random_safe_spec(rng_a)
    s2 = random_safe_spec(rng_b)
    assert s1 == s2


def test_random_safe_spec_kq_in_safe_range():
    rng = np.random.default_rng(1)
    for _ in range(20):
        spec = random_safe_spec(rng)
        assert 0.5 <= spec.kq_ref_s_inv <= 5.0
        # Other rate ranges also bounded
        assert 0.0 < spec.kdep_ref_s_inv < 5.0
        assert 0.0 < spec.kloss_ref_s_inv < 1.0


def test_random_stiff_spec_kq_in_stiff_range():
    rng = np.random.default_rng(2)
    for _ in range(10):
        spec = random_stiff_spec(rng)
        assert 100.0 <= spec.kq_ref_s_inv <= 1000.0


# ---- generate_sample ----------------------------------------------------

def test_generate_sample_shapes_and_ranges():
    rng = np.random.default_rng(3)
    spec = random_safe_spec(rng)
    out = generate_sample(spec, grid_size=GRID, dx_nm=DX)
    assert isinstance(out, SampleArrays)
    for name in DATASET_FIELD_NAMES:
        arr = getattr(out, name)
        assert arr.shape == (GRID, GRID)
        assert arr.dtype == np.float32
    # physical contracts
    assert float(out.H0.min()) >= 0.0
    assert float(out.H_final.min()) >= 0.0
    assert float(out.Q_final.min()) >= 0.0
    assert 0.0 <= float(out.P_final.min())
    assert float(out.P_final.max()) <= 1.0
    # mask is binary
    uniq = np.unique(out.R)
    assert set(uniq.tolist()).issubset({0.0, 1.0})


def test_generate_sample_zero_time_returns_initial_conditions():
    rng = np.random.default_rng(4)
    spec = random_safe_spec(rng)
    spec.t_end_s = 0.0
    out = generate_sample(spec, grid_size=GRID, dx_nm=DX)
    # H_final == H0 because no evolution happens.
    np.testing.assert_allclose(out.H_final, out.H0, atol=1e-6)
    assert np.all(out.P_final == 0.0)


# ---- splits -------------------------------------------------------------

def test_make_split_indices_partitions_correctly():
    splits = make_split_indices(50, fractions=(0.8, 0.1, 0.1), seed=0)
    train, val, test = splits["train"], splits["val"], splits["test"]
    all_idx = sorted(train + val + test)
    assert all_idx == list(range(50))
    assert len(set(train) & set(val)) == 0
    assert len(set(train) & set(test)) == 0
    assert len(set(val) & set(test)) == 0
    assert len(val) >= 1 and len(test) >= 1


def test_make_split_indices_is_deterministic():
    a = make_split_indices(40, seed=7)
    b = make_split_indices(40, seed=7)
    assert a == b


def test_make_split_indices_rejects_bad_fractions():
    with pytest.raises(ValueError):
        make_split_indices(10, fractions=(0.5, 0.4, 0.0))


def test_make_split_indices_rejects_too_few_samples():
    with pytest.raises(ValueError):
        make_split_indices(2)


# ---- save / load round-trip --------------------------------------------

def test_save_load_round_trip(tmp_dataset_dir):
    rng = np.random.default_rng(5)
    samples = [generate_sample(random_safe_spec(rng), grid_size=GRID, dx_nm=DX)
               for _ in range(4)]
    splits = make_split_indices(len(samples), seed=0)
    meta = {
        "grid_size": GRID,
        "grid_spacing_nm": DX,
        "P_threshold": 0.5,
        "solver": "fd",
        "regime": "safe",
        "seed": 5,
        "splits": splits,
    }
    out_path = tmp_dataset_dir / "ds.npz"
    save_dataset(out_path, samples, meta)
    arrays, loaded_meta = load_dataset(out_path)

    # field arrays
    for name in DATASET_FIELD_NAMES:
        assert arrays[name].shape == (4, GRID, GRID)
        assert arrays[name].dtype == np.float32
        np.testing.assert_allclose(
            arrays[name][0], getattr(samples[0], name)
        )
    # scalars
    for name in DATASET_SCALAR_NAMES:
        assert arrays[name].shape == (4,)
    # aerial code
    assert arrays["aerial_kind_code"].shape == (4,)
    assert arrays["aerial_kind_code"].dtype == np.int8
    # metadata round-trip
    assert loaded_meta["grid_size"] == GRID
    assert loaded_meta["regime"] == "safe"
    assert loaded_meta["splits"]["train"] == splits["train"]
    assert loaded_meta["aerial_kind_codes"] == AERIAL_KIND_CODES
    assert loaded_meta["n_samples"] == 4


def test_save_dataset_rejects_empty(tmp_dataset_dir):
    with pytest.raises(ValueError):
        save_dataset(tmp_dataset_dir / "empty.npz", [], {})


# ---- parameter_ranges --------------------------------------------------

def test_parameter_ranges_summarizes_min_max_mean():
    rng = np.random.default_rng(6)
    samples = [generate_sample(random_safe_spec(rng), grid_size=GRID, dx_nm=DX)
               for _ in range(5)]
    ranges = parameter_ranges(samples)
    assert "scalars" in ranges
    assert "aerial_kind_counts" in ranges
    for name in DATASET_SCALAR_NAMES:
        entry = ranges["scalars"][name]
        assert entry["min"] <= entry["mean"] <= entry["max"]
    total = sum(ranges["aerial_kind_counts"].values())
    assert total == 5

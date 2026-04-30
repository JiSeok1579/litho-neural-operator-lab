"""Smoke tests for v3 sampler / prefilter / labeler / metrics_io."""
from __future__ import annotations

import numpy as np
import pytest

from reaction_diffusion_peb_v3_screening.src.budget_prefilter import (
    diffusion_length_nm,
    fundamental_blur_attenuation,
    score_candidate,
    score_all,
    select_top_n,
)
from reaction_diffusion_peb_v3_screening.src.candidate_sampler import (
    CandidateSpace,
    sample_candidates,
)
from reaction_diffusion_peb_v3_screening.src.labeler import (
    LABEL_ORDER,
    LabelThresholds,
    label_one,
)
from reaction_diffusion_peb_v3_screening.src.metrics_io import (
    FEATURE_KEYS,
    REGRESSION_TARGETS,
    build_feature_matrix,
)


CANDIDATE_SPACE_YAML = "reaction_diffusion_peb_v3_screening/configs/candidate_space.yaml"


# ---------------- candidate_sampler ----------------

def test_sobol_sampler_returns_n_candidates_with_required_keys():
    space = CandidateSpace.from_yaml(CANDIDATE_SPACE_YAML)
    cs = sample_candidates(space, n=64, method="sobol", seed=7)
    assert len(cs) == 64
    required = {"pitch_nm", "line_cd_ratio", "dose_mJ_cm2",
                "sigma_nm", "DH_nm2_s", "time_s",
                "Hmax_mol_dm3", "kdep_s_inv", "Q0_mol_dm3", "kq_s_inv",
                "abs_len_nm", "line_cd_nm", "domain_x_nm", "dose_norm",
                "reference_dose_mJ_cm2", "kloss_s_inv", "P_threshold"}
    for c in cs:
        assert required.issubset(c.keys())
        assert c["pitch_nm"] in {18, 20, 24, 28, 32}
        assert c["line_cd_ratio"] in {0.45, 0.52, 0.60}
        assert 21.0 <= c["dose_mJ_cm2"] <= 60.0
        assert 0.0 <= c["sigma_nm"] <= 3.0
        assert c["domain_x_nm"] == c["pitch_nm"] * 5
        assert abs(c["line_cd_nm"] - c["pitch_nm"] * c["line_cd_ratio"]) < 1e-9


def test_lhs_sampler_runs():
    space = CandidateSpace.from_yaml(CANDIDATE_SPACE_YAML)
    cs = sample_candidates(space, n=32, method="latin_hypercube", seed=1)
    assert len(cs) == 32


# ---------------- budget_prefilter ----------------

def test_diffusion_length_known_value():
    assert abs(diffusion_length_nm(0.5, 30.0) - np.sqrt(2.0 * 0.5 * 30.0)) < 1e-12


def test_fundamental_blur_attenuation_endpoints():
    assert fundamental_blur_attenuation(24.0, 0.0) == 1.0
    # Damping increases with σ.
    a3 = fundamental_blur_attenuation(24.0, 3.0)
    a8 = fundamental_blur_attenuation(24.0, 8.0)
    assert 0.0 < a8 < a3 < 1.0
    assert a8 < 0.2


def test_score_candidate_is_in_unit_interval():
    space = CandidateSpace.from_yaml(CANDIDATE_SPACE_YAML)
    cs = sample_candidates(space, n=128, method="sobol", seed=7)
    scored = score_all(cs)
    for s in scored:
        assert 0.0 <= s["prefilter_score"] <= 1.0


def test_select_top_n_orders_by_score():
    space = CandidateSpace.from_yaml(CANDIDATE_SPACE_YAML)
    cs = sample_candidates(space, n=200, method="sobol", seed=7)
    scored = score_all(cs)
    top = select_top_n(scored, 25)
    assert len(top) == 25
    for i in range(len(top) - 1):
        assert top[i]["prefilter_score"] >= top[i + 1]["prefilter_score"]


# ---------------- labeler ----------------

def _ok_row(**overrides) -> dict:
    base = {
        "P_max": 0.85, "P_min": 0.10, "H_min": 0.02,
        "P_space_center_mean": 0.30, "P_line_center_mean": 0.78,
        "contrast": 0.48, "area_frac": 0.62,
        "CD_pitch_frac": 0.62, "CD_locked_nm": 12.4,
        "P_line_margin": 0.13,
        "LER_after_PEB_P_nm": 2.50,
        "LER_design_initial_nm": 2.77,
        "LER_CD_locked_nm": 2.40,
    }
    base.update(overrides)
    return base


def test_labeler_robust_valid():
    assert label_one(_ok_row()) == "robust_valid"


def test_labeler_margin_risk():
    r = _ok_row(P_line_margin=0.02)
    assert label_one(r) == "margin_risk"


def test_labeler_under_exposed():
    r = _ok_row(P_line_center_mean=0.55, P_line_margin=-0.10)
    assert label_one(r) == "under_exposed"


def test_labeler_merged():
    r = _ok_row(P_space_center_mean=0.60, area_frac=0.95, CD_pitch_frac=0.95)
    assert label_one(r) == "merged"


def test_labeler_roughness_degraded():
    r = _ok_row(LER_CD_locked_nm=4.5, LER_design_initial_nm=2.0)
    assert label_one(r) == "roughness_degraded"


def test_labeler_numerical_invalid_nan():
    r = _ok_row(P_max=float("nan"))
    assert label_one(r) == "numerical_invalid"


def test_labeler_numerical_invalid_bounds():
    r = _ok_row(P_max=1.5)
    assert label_one(r) == "numerical_invalid"


def test_labeler_threshold_overrides():
    t = LabelThresholds(P_line_margin_robust=0.20)
    r = _ok_row(P_line_margin=0.10)
    # Now 0.10 < 0.20, so margin_risk.
    assert label_one(r, t=t) == "margin_risk"


def test_label_order_is_consistent():
    assert "robust_valid" in LABEL_ORDER
    assert len(LABEL_ORDER) == 6


# ---------------- metrics_io feature builder ----------------

def test_build_feature_matrix_shape():
    space = CandidateSpace.from_yaml(CANDIDATE_SPACE_YAML)
    cs = sample_candidates(space, n=10, method="sobol", seed=7)
    X = build_feature_matrix(cs)
    assert X.shape == (10, len(FEATURE_KEYS))
    assert np.isfinite(X).all()


def test_regression_targets_constant():
    assert REGRESSION_TARGETS == ["CD_locked_nm", "LER_CD_locked_nm",
                                    "area_frac", "P_line_margin"]

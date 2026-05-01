"""Stage 06A — process_variation + yield_optimizer tests."""
from __future__ import annotations

import numpy as np
import pytest
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

from reaction_diffusion_peb_v3_screening.src.candidate_sampler import (
    CandidateSpace,
    sample_candidates,
)
from reaction_diffusion_peb_v3_screening.src.metrics_io import (
    FEATURE_KEYS,
    REGRESSION_TARGETS,
    build_feature_matrix,
)
from reaction_diffusion_peb_v3_screening.src.process_variation import (
    KnobSpec,
    VariationSpec,
    feature_matrix_from_recipes,
    sample_variations,
)
from reaction_diffusion_peb_v3_screening.src.yield_optimizer import (
    SUMMARY_COLUMNS,
    YieldScoreConfig,
    evaluate_recipes,
    evaluate_single_recipe,
)


CANDIDATE_SPACE_YAML = "reaction_diffusion_peb_v3_screening/configs/candidate_space.yaml"


# ---------------- process_variation ----------------

def _v2_op_recipe(space: CandidateSpace) -> dict:
    """A self-consistent v2-OP-like recipe usable by both modules."""
    base = {p["name"]: (p["values"][0] if p["type"] == "choice" else p["low"])
            for p in space.parameters}
    base.update({
        "pitch_nm":      24,
        "line_cd_ratio": 0.52,
        "dose_mJ_cm2":   40.0,
        "sigma_nm":      2.0,
        "DH_nm2_s":      0.5,
        "time_s":        30.0,
        "Hmax_mol_dm3":  0.2,
        "kdep_s_inv":    0.5,
        "Q0_mol_dm3":    0.02,
        "kq_s_inv":      1.0,
        "abs_len_nm":    50.0,
        "line_cd_nm":    24 * 0.52,
    })
    base.update(space.fixed)
    base["_id"] = "v2_op"
    return base


def _default_var_spec() -> VariationSpec:
    return VariationSpec(
        knobs=[
            KnobSpec("dose_mJ_cm2",  relative=True,  absolute=False, width=0.05),
            KnobSpec("time_s",       relative=False, absolute=True,  width=2.0),
            KnobSpec("sigma_nm",     relative=False, absolute=True,  width=0.2),
            KnobSpec("DH_nm2_s",     relative=True,  absolute=False, width=0.10),
            KnobSpec("Q0_mol_dm3",   relative=True,  absolute=False, width=0.10),
            KnobSpec("Hmax_mol_dm3", relative=True,  absolute=False, width=0.05),
        ],
        line_cd_abs_nm=0.5,
    )


def test_sample_variations_returns_n_rows_with_full_feature_set():
    space = CandidateSpace.from_yaml(CANDIDATE_SPACE_YAML)
    base = _v2_op_recipe(space)
    spec = _default_var_spec()
    rng = np.random.default_rng(0)
    rows = sample_variations(base, spec, n=64, space=space, rng=rng)
    assert len(rows) == 64
    for r in rows:
        for k in FEATURE_KEYS:
            assert k in r


def test_variations_are_clipped_to_candidate_space_bounds():
    space = CandidateSpace.from_yaml(CANDIDATE_SPACE_YAML)
    # Force the base recipe to sit at the upper edge so positive
    # variations of dose / DH / Hmax / Q0 must be clipped.
    base = _v2_op_recipe(space)
    base["dose_mJ_cm2"] = 60.0    # candidate-space upper bound
    base["DH_nm2_s"]    = 0.8     # upper bound
    base["Hmax_mol_dm3"] = 0.22   # upper bound
    base["Q0_mol_dm3"]  = 0.03    # upper bound
    base["sigma_nm"]    = 3.0     # upper bound
    base["time_s"]      = 45.0    # upper bound
    spec = _default_var_spec()
    rng = np.random.default_rng(1)
    rows = sample_variations(base, spec, n=400, space=space, rng=rng)
    bounds = {p["name"]: p for p in space.parameters
              if p["type"] == "uniform"}
    for r in rows:
        for name, p in bounds.items():
            assert p["low"] - 1e-12 <= r[name] <= p["high"] + 1e-12, (
                f"{name} = {r[name]} fell outside [{p['low']}, {p['high']}]"
            )


def test_variations_zero_width_returns_base_unchanged():
    space = CandidateSpace.from_yaml(CANDIDATE_SPACE_YAML)
    base = _v2_op_recipe(space)
    # Zero-width variation spec — every knob must come back exactly.
    spec = VariationSpec(knobs=[], line_cd_abs_nm=0.0)
    rng = np.random.default_rng(2)
    rows = sample_variations(base, spec, n=32, space=space, rng=rng)
    for r in rows:
        for k in FEATURE_KEYS:
            assert r[k] == base[k]


def test_line_cd_variation_propagates_through_ratio_to_line_cd_nm():
    space = CandidateSpace.from_yaml(CANDIDATE_SPACE_YAML)
    base = _v2_op_recipe(space)
    # Only line_cd channel active.
    spec = VariationSpec(knobs=[], line_cd_abs_nm=0.5)
    rng = np.random.default_rng(3)
    rows = sample_variations(base, spec, n=200, space=space, rng=rng)
    pitch = float(base["pitch_nm"])
    half = 0.5 / pitch
    ratio_lo, ratio_hi = 0.45, 0.60     # candidate-space line_cd_ratio choice min/max
    for r in rows:
        # ratio should sit within [base - half, base + half] but clipped to choice min/max
        assert ratio_lo - 1e-12 <= r["line_cd_ratio"] <= ratio_hi + 1e-12
        target_lo = max(ratio_lo, float(base["line_cd_ratio"]) - half)
        target_hi = min(ratio_hi, float(base["line_cd_ratio"]) + half)
        assert target_lo - 1e-12 <= r["line_cd_ratio"] <= target_hi + 1e-12
        # derived line_cd_nm stays consistent with ratio × pitch
        assert abs(r["line_cd_nm"] - r["line_cd_ratio"] * pitch) < 1e-9


def test_feature_matrix_from_recipes_matches_metrics_io_layout():
    space = CandidateSpace.from_yaml(CANDIDATE_SPACE_YAML)
    cs = sample_candidates(space, n=8, method="sobol", seed=11)
    a = feature_matrix_from_recipes(cs)
    b = build_feature_matrix(cs)
    np.testing.assert_allclose(a, b)


# ---------------- yield_optimizer ----------------

def _toy_classifier(seed: int = 0):
    """Synthetic classifier trained on the v3 feature set with the v3
    label vocabulary so `predict_proba` and `classes_` line up the way
    the optimizer expects."""
    rng = np.random.default_rng(seed)
    space = CandidateSpace.from_yaml(CANDIDATE_SPACE_YAML)
    cs = sample_candidates(space, n=200, method="sobol", seed=seed)
    X = build_feature_matrix(cs)
    classes = ["robust_valid", "margin_risk", "merged",
               "under_exposed", "roughness_degraded", "numerical_invalid"]
    # Hand-crafted assignment so each class is represented at least once.
    y = []
    for i, c in enumerate(cs):
        # margin-leaning: low DH, low time → under_exposed
        if c["DH_nm2_s"] < 0.4 and c["time_s"] < 25:
            y.append("under_exposed")
        elif c["DH_nm2_s"] > 0.7 and c["time_s"] > 40:
            y.append("merged")
        elif c["sigma_nm"] > 2.5:
            y.append("roughness_degraded")
        elif c["dose_mJ_cm2"] > 55:
            y.append("numerical_invalid")
        elif c["dose_mJ_cm2"] < 25:
            y.append("margin_risk")
        else:
            y.append("robust_valid")
    # Make sure all 6 classes are present.
    y[0] = "numerical_invalid"; y[1] = "margin_risk"; y[2] = "merged"
    y[3] = "roughness_degraded"; y[4] = "under_exposed"; y[5] = "robust_valid"
    clf = RandomForestClassifier(n_estimators=30, n_jobs=1, random_state=seed)
    clf.fit(X, y)
    return clf


def _toy_regressor4(seed: int = 0):
    rng = np.random.default_rng(seed)
    space = CandidateSpace.from_yaml(CANDIDATE_SPACE_YAML)
    cs = sample_candidates(space, n=200, method="sobol", seed=seed)
    X = build_feature_matrix(cs)
    Y = np.column_stack([
        15.0 + 0.2 * (X[:, 2] - 40.0) + rng.normal(0, 0.3, size=X.shape[0]),  # CD_locked
        2.5  + 0.3 * X[:, 3]          + rng.normal(0, 0.1, size=X.shape[0]),  # LER_CD_locked
        0.55 + 0.0 * X[:, 0]          + rng.normal(0, 0.02, size=X.shape[0]),  # area_frac
        0.10 - 0.005 * X[:, 3]        + rng.normal(0, 0.02, size=X.shape[0]),  # P_line_margin
    ])
    reg = RandomForestRegressor(n_estimators=30, n_jobs=1, random_state=seed)
    reg.fit(X, Y)
    return reg


def _toy_cd_fixed(seed: int = 0):
    rng = np.random.default_rng(seed)
    space = CandidateSpace.from_yaml(CANDIDATE_SPACE_YAML)
    cs = sample_candidates(space, n=200, method="sobol", seed=seed)
    X = build_feature_matrix(cs)
    y = 14.5 + 0.05 * (X[:, 2] - 40.0) + rng.normal(0, 0.2, size=X.shape[0])
    reg = RandomForestRegressor(n_estimators=30, n_jobs=1, random_state=seed)
    reg.fit(X, y)
    return reg


def _default_score_cfg() -> YieldScoreConfig:
    return YieldScoreConfig(
        class_weights={
            "robust_valid":         1.0,
            "margin_risk":         -0.5,
            "merged":              -2.0,
            "under_exposed":       -2.0,
            "roughness_degraded":  -1.5,
            "numerical_invalid":   -3.0,
        },
        cd_target_nm=15.0, cd_tolerance_nm=1.0, cd_penalty_weight=1.0,
        ler_max_nm=3.0,    ler_penalty_weight=1.0,
    )


def test_evaluate_recipes_returns_summary_columns_and_score_in_range():
    space = CandidateSpace.from_yaml(CANDIDATE_SPACE_YAML)
    clf = _toy_classifier(0); reg = _toy_regressor4(0); aux = _toy_cd_fixed(0)
    spec = _default_var_spec()
    cs = sample_candidates(space, n=8, method="sobol", seed=42)
    rows = evaluate_recipes(cs, clf, reg, aux, spec, n_var=20,
                            space=space, score_cfg=_default_score_cfg(),
                            seed=99)
    assert len(rows) == 8
    for r in rows:
        for col in SUMMARY_COLUMNS:
            assert col in r
        # Score upper bound = 1.0 (P=1 robust_valid, no penalty); lower
        # bound = -3.0 from numerical_invalid term plus arbitrarily large
        # penalties. Sanity: not NaN, finite.
        assert np.isfinite(r["yield_score"])
        # Class probabilities sum ≈ 1.
        psum = sum(r[k] for k in [
            "p_robust_valid", "p_margin_risk", "p_merged",
            "p_under_exposed", "p_roughness_degraded", "p_numerical_invalid",
        ])
        assert abs(psum - 1.0) < 1e-9


def test_evaluate_recipes_seed_reproducible():
    space = CandidateSpace.from_yaml(CANDIDATE_SPACE_YAML)
    clf = _toy_classifier(0); reg = _toy_regressor4(0); aux = _toy_cd_fixed(0)
    spec = _default_var_spec()
    cs = sample_candidates(space, n=4, method="sobol", seed=42)
    a = evaluate_recipes(cs, clf, reg, aux, spec, n_var=16,
                         space=space, score_cfg=_default_score_cfg(), seed=7)
    b = evaluate_recipes(cs, clf, reg, aux, spec, n_var=16,
                         space=space, score_cfg=_default_score_cfg(), seed=7)
    for ra, rb in zip(a, b):
        assert ra["yield_score"] == rb["yield_score"]
        assert ra["p_robust_valid"] == rb["p_robust_valid"]


def test_evaluate_single_recipe_baseline_pipeline():
    space = CandidateSpace.from_yaml(CANDIDATE_SPACE_YAML)
    clf = _toy_classifier(0); reg = _toy_regressor4(0); aux = _toy_cd_fixed(0)
    spec = _default_var_spec()
    base = _v2_op_recipe(space)
    out = evaluate_single_recipe(
        base, clf, reg, aux, spec, n_var=32, space=space,
        score_cfg=_default_score_cfg(), seed=13,
    )
    assert "yield_score" in out
    assert np.isfinite(out["yield_score"])
    # Baseline OP must be inside the candidate space — its mean recipe
    # values must equal the input.
    for k in [
        "pitch_nm", "line_cd_ratio", "dose_mJ_cm2", "sigma_nm",
        "DH_nm2_s", "time_s", "Hmax_mol_dm3", "kdep_s_inv",
        "Q0_mol_dm3", "kq_s_inv", "abs_len_nm",
    ]:
        assert out[k] == base[k]


def test_yield_score_penalty_kicks_in_outside_cd_window():
    """Hand-crafted aux regressor that returns CD = 18.0 (far outside the
    ±1 nm window). yield_score must drop by exactly the penalty term
    relative to a CD = 15.0 reference."""
    space = CandidateSpace.from_yaml(CANDIDATE_SPACE_YAML)
    clf = _toy_classifier(0); reg = _toy_regressor4(0)

    # Aux regressor that always returns 18.0 — we override .predict.
    class ConstantAux:
        def __init__(self, v):
            self.v = float(v)
        def predict(self, X):
            return np.full(X.shape[0], self.v)

    base = _v2_op_recipe(space)
    cfg = _default_score_cfg()
    spec = VariationSpec(knobs=[], line_cd_abs_nm=0.0)  # zero variation → identical recipe rows

    out_15 = evaluate_single_recipe(base, clf, reg, ConstantAux(15.0),
                                    spec, n_var=8, space=space,
                                    score_cfg=cfg, seed=0)
    out_18 = evaluate_single_recipe(base, clf, reg, ConstantAux(18.0),
                                    spec, n_var=8, space=space,
                                    score_cfg=cfg, seed=0)
    expected_pen_15 = max(0.0, abs(15.0 - cfg.cd_target_nm) - cfg.cd_tolerance_nm)
    expected_pen_18 = max(0.0, abs(18.0 - cfg.cd_target_nm) - cfg.cd_tolerance_nm)
    assert out_15["cd_error_penalty"] == pytest.approx(expected_pen_15)
    assert out_18["cd_error_penalty"] == pytest.approx(expected_pen_18)
    # Score difference is exactly the penalty difference (everything else
    # is identical because spec is zero-variation).
    assert out_15["yield_score"] - out_18["yield_score"] == pytest.approx(
        expected_pen_18 - expected_pen_15
    )

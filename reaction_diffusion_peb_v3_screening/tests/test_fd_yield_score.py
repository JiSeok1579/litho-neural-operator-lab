"""Stage 06B — fd_yield_score tests."""
from __future__ import annotations

import math

import numpy as np
import pytest

from reaction_diffusion_peb_v3_screening.src.fd_yield_score import (
    ALL_CLASSES,
    empirical_class_probs,
    fd_yield_score_from_rows,
    fd_yield_score_per_recipe,
    nominal_yield_score,
    spearman,
    topk_overlap,
)
from reaction_diffusion_peb_v3_screening.src.yield_optimizer import (
    YieldScoreConfig,
)


def _cfg() -> YieldScoreConfig:
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


def _row(label: str, cd_final=15.0, ler=2.5, cd_locked=12.5,
         margin=0.10, area=0.6) -> dict:
    return {
        "label": label,
        "CD_final_nm":      cd_final,
        "LER_CD_locked_nm": ler,
        "CD_locked_nm":     cd_locked,
        "P_line_margin":    margin,
        "area_frac":        area,
    }


def test_empirical_class_probs_sum_to_one_when_all_labels_known():
    rows = [_row("robust_valid")] * 80 + [_row("merged")] * 20
    p = empirical_class_probs(rows)
    assert math.isclose(sum(p.values()), 1.0, abs_tol=1e-12)
    assert math.isclose(p["robust_valid"], 0.8)
    assert math.isclose(p["merged"], 0.2)
    for c in ALL_CLASSES:
        assert c in p


def test_empirical_probs_handle_unknown_label_silently():
    rows = [_row("robust_valid")] * 50 + [_row("???")] * 50
    p = empirical_class_probs(rows)
    # Unknown labels are dropped from the canonical 6-class breakdown,
    # so the canonical sum < 1.
    assert math.isclose(p["robust_valid"], 0.5)
    assert sum(p.values()) < 1.0 + 1e-12


def test_fd_score_one_hot_robust_valid_no_penalty_returns_one():
    cfg = _cfg()
    out = nominal_yield_score(_row("robust_valid", cd_final=15.0, ler=2.5), cfg)
    assert out["P_FD_robust_valid"] == 1.0
    assert out["FD_cd_error_penalty"] == 0.0
    assert out["FD_ler_penalty"] == 0.0
    assert out["FD_yield_score"] == pytest.approx(1.0)


def test_fd_score_cd_outside_tolerance_subtracts_penalty():
    cfg = _cfg()
    # CD = 17 → penalty = max(0, |17-15| - 1) / 1 = 1.0
    out = nominal_yield_score(_row("robust_valid", cd_final=17.0), cfg)
    assert out["FD_cd_error_penalty"] == pytest.approx(1.0)
    assert out["FD_yield_score"] == pytest.approx(0.0)


def test_fd_score_ler_above_max_subtracts_penalty():
    cfg = _cfg()
    # LER = 4.0 → penalty = max(0, 4 - 3) = 1.0
    out = nominal_yield_score(_row("robust_valid", ler=4.0), cfg)
    assert out["FD_ler_penalty"] == pytest.approx(1.0)
    assert out["FD_yield_score"] == pytest.approx(0.0)


def test_fd_score_mixed_labels_match_class_weight_sum():
    cfg = _cfg()
    rows = (
        [_row("robust_valid")] * 80
        + [_row("merged",        cd_final=15.0, ler=2.5)] * 10
        + [_row("under_exposed", cd_final=15.0, ler=2.5)] * 10
    )
    out = fd_yield_score_from_rows(rows, cfg)
    expected_class_sum = 1.0 * 0.8 + (-2.0) * 0.1 + (-2.0) * 0.1
    # CD_final_nm and LER are at target → no penalty.
    assert out["FD_yield_score"] == pytest.approx(expected_class_sum)
    assert out["P_FD_robust_valid"]  == pytest.approx(0.8)
    assert out["P_FD_merged"]        == pytest.approx(0.1)
    assert out["P_FD_under_exposed"] == pytest.approx(0.1)


def test_fd_score_per_recipe_buckets_correctly():
    cfg = _cfg()
    rows = []
    for r in [_row("robust_valid")] * 5:
        d = dict(r); d["source_recipe_id"] = "A_001"
        rows.append(d)
    for r in [_row("merged")] * 5:
        d = dict(r); d["source_recipe_id"] = "A_002"
        rows.append(d)
    out = fd_yield_score_per_recipe(rows, cfg)
    assert "A_001" in out and "A_002" in out
    assert out["A_001"]["P_FD_robust_valid"] == 1.0
    assert out["A_002"]["P_FD_merged"]       == 1.0


def test_fd_score_handles_nan_cd_and_ler_without_kicking_in_penalty():
    """A numerical_invalid row may have NaN CD/LER; the penalty must
    fall back to zero rather than NaN-poisoning the score."""
    cfg = _cfg()
    row = _row("numerical_invalid", cd_final=float("nan"), ler=float("nan"))
    out = nominal_yield_score(row, cfg)
    assert out["P_FD_numerical_invalid"] == 1.0
    assert out["FD_cd_error_penalty"] == 0.0
    assert out["FD_ler_penalty"] == 0.0
    # numerical_invalid weight is -3.0
    assert out["FD_yield_score"] == pytest.approx(-3.0)


def test_spearman_perfect_rank_returns_one():
    a = np.array([1.0, 2.0, 3.0, 4.0])
    b = np.array([10.0, 20.0, 30.0, 40.0])
    assert spearman(a, b) == pytest.approx(1.0)


def test_spearman_reversed_rank_returns_minus_one():
    a = np.array([1.0, 2.0, 3.0, 4.0])
    b = np.array([40.0, 30.0, 20.0, 10.0])
    assert spearman(a, b) == pytest.approx(-1.0)


def test_topk_overlap_basic():
    a = ["A", "B", "C", "D", "E"]
    b = ["B", "A", "F", "G", "C"]
    assert topk_overlap(a, b, k=3) == 2     # {A,B} ∩ {B,A,F} = {A,B}
    assert topk_overlap(a, b, k=5) == 3     # {A,B,C} overlap


def test_fd_score_reproducible_under_same_input():
    cfg = _cfg()
    rows = (
        [_row("robust_valid")] * 30
        + [_row("merged", cd_final=18.0)] * 70
    )
    a = fd_yield_score_from_rows(rows, cfg)
    b = fd_yield_score_from_rows(rows, cfg)
    assert a == b

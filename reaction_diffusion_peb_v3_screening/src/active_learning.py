"""Uncertainty-driven candidate selection."""
from __future__ import annotations

import numpy as np

from .surrogate_classifier import classifier_uncertainty
from .surrogate_regressor import regressor_per_tree_std

# Per-target index inside the regressor's output vector.
TARGET_INDEX = {
    "CD_locked_nm":      0,
    "LER_CD_locked_nm":  1,
    "area_frac":         2,
    "P_line_margin":     3,
}


def acquisition_indices(
    clf, reg,
    X_pool: np.ndarray,
    *,
    classifier_unc_threshold: float = 0.6,
    regressor_std_min_nm: float = 0.4,
    combine: str = "union",
    top_k: int = 500,
) -> tuple[np.ndarray, dict]:
    """Pick top_k candidate indices from X_pool that are uncertain.

    Two uncertainty signals (existing behaviour):
        a) classifier max-class probability below classifier_unc_threshold
        b) regressor per-tree std above regressor_std_min_nm on any output
    """
    cls_unc = classifier_uncertainty(clf, X_pool)
    cls_uncertain = cls_unc > (1.0 - classifier_unc_threshold)

    reg_std = regressor_per_tree_std(reg, X_pool)            # (N, T)
    reg_max_std = reg_std.max(axis=1)                        # (N,)
    reg_uncertain = reg_max_std > regressor_std_min_nm

    if combine == "intersection":
        mask = cls_uncertain & reg_uncertain
    else:
        mask = cls_uncertain | reg_uncertain

    score = (cls_unc / max(cls_unc.max(), 1e-9)) + (reg_max_std / max(reg_max_std.max(), 1e-9))
    score_masked = np.where(mask, score, -np.inf)
    order = np.argsort(-score_masked)
    selected = order[: max(0, top_k)]
    selected = selected[np.isfinite(score_masked[selected])]

    return selected, {
        "n_classifier_uncertain": int(cls_uncertain.sum()),
        "n_regressor_uncertain":  int(reg_uncertain.sum()),
        "n_combined_mask":        int(mask.sum()),
        "n_selected":             int(len(selected)),
        "classifier_uncertainty_summary": {
            "mean": float(cls_unc.mean()),
            "max":  float(cls_unc.max()),
            "p90":  float(np.quantile(cls_unc, 0.90)),
        },
        "regressor_std_summary": {
            "mean": float(reg_max_std.mean()),
            "max":  float(reg_max_std.max()),
            "p90":  float(np.quantile(reg_max_std, 0.90)),
        },
    }


def acquisition_indices_v2(
    clf, reg,
    X_pool: np.ndarray,
    *,
    classifier_unc_threshold: float = 0.6,
    regressor_std_min_nm: float = 0.4,
    margin_band: tuple[float, float] = (-0.02, 0.07),
    P_space_band: tuple[float, float] = (0.40, 0.55),
    P_line_band: tuple[float, float] = (0.60, 0.70),
    top_k: int = 500,
) -> tuple[np.ndarray, dict]:
    """Stage 04B acquisition. Combines four uncertainty signals (union):

    a) classifier_uncertainty: 1 − max class probability above
       (1 − classifier_unc_threshold).
    b) regressor per-tree std on any output above regressor_std_min_nm.
    c) predicted P_line_margin inside `margin_band` (near-threshold).
    d) predicted area_frac / P_space proxies near merged or under_exposed
       boundary — implemented via predicted P_line_margin sign change
       region using the regressor's outputs.

    Note: the regressor is trained on (CD_locked, LER_locked, area_frac,
    P_line_margin) so we infer near-merged via low predicted P_line_margin
    + high predicted area_frac, and near-under_exposed via predicted
    P_line_margin near the lower-band bound.
    """
    # Existing two signals.
    cls_unc = classifier_uncertainty(clf, X_pool)
    cls_uncertain = cls_unc > (1.0 - classifier_unc_threshold)

    reg_std = regressor_per_tree_std(reg, X_pool)
    reg_max_std = reg_std.max(axis=1)
    reg_uncertain = reg_max_std > regressor_std_min_nm

    # Regressor mean prediction on the pool (mean across trees == reg.predict).
    Y_mean = reg.predict(X_pool)
    pred_margin = Y_mean[:, TARGET_INDEX["P_line_margin"]]
    pred_area   = Y_mean[:, TARGET_INDEX["area_frac"]]

    margin_near_threshold = (pred_margin >= margin_band[0]) & (pred_margin <= margin_band[1])
    near_merged = (pred_area >= 0.85) & (pred_margin >= 0.0)        # high area + still passing → merged candidate
    near_under  = pred_margin <= margin_band[0]                      # below threshold → under_exposed candidate

    boundary_uncertain = margin_near_threshold | near_merged | near_under

    mask = cls_uncertain | reg_uncertain | boundary_uncertain

    # Combined score (sum of normalised uncertainties + small bonus for
    # near-threshold predictions).
    cls_norm = cls_unc / max(cls_unc.max(), 1e-9)
    reg_norm = reg_max_std / max(reg_max_std.max(), 1e-9)
    margin_dist = np.minimum(
        np.abs(pred_margin - margin_band[0]),
        np.abs(pred_margin - margin_band[1]),
    )
    margin_dist = margin_dist / max(margin_dist.max(), 1e-9)
    margin_bonus = (1.0 - margin_dist)  # 1 at the band edges, 0 far away
    score = cls_norm + reg_norm + 0.5 * margin_bonus
    score_masked = np.where(mask, score, -np.inf)

    order = np.argsort(-score_masked)
    selected = order[: max(0, top_k)]
    selected = selected[np.isfinite(score_masked[selected])]

    return selected, {
        "n_pool":                  int(X_pool.shape[0]),
        "n_classifier_uncertain":  int(cls_uncertain.sum()),
        "n_regressor_uncertain":   int(reg_uncertain.sum()),
        "n_margin_near_threshold": int(margin_near_threshold.sum()),
        "n_near_merged":           int(near_merged.sum()),
        "n_near_under_exposed":    int(near_under.sum()),
        "n_combined_mask":         int(mask.sum()),
        "n_selected":              int(len(selected)),
        "classifier_uncertainty_summary": {
            "mean": float(cls_unc.mean()),
            "max":  float(cls_unc.max()),
            "p90":  float(np.quantile(cls_unc, 0.90)),
        },
        "regressor_std_summary": {
            "mean": float(reg_max_std.mean()),
            "max":  float(reg_max_std.max()),
            "p90":  float(np.quantile(reg_max_std, 0.90)),
        },
        "predicted_margin_summary": {
            "mean": float(pred_margin.mean()),
            "fraction_below_zero": float((pred_margin < 0.0).mean()),
            "fraction_in_band":    float(margin_near_threshold.mean()),
        },
    }

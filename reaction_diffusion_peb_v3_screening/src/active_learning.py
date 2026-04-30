"""Uncertainty-driven candidate selection."""
from __future__ import annotations

import numpy as np

from .surrogate_classifier import classifier_uncertainty
from .surrogate_regressor import regressor_per_tree_std


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

    classifier_unc_threshold: max class probability below this is uncertain.
    regressor_std_min_nm:     per-tree std above this on any output is uncertain.
    combine: "union" or "intersection".

    Returns (indices, diagnostics).
    """
    cls_unc = classifier_uncertainty(clf, X_pool)
    cls_uncertain = (cls_unc > (1.0 - classifier_unc_threshold))

    reg_std = regressor_per_tree_std(reg, X_pool)            # (N, T)
    reg_max_std = reg_std.max(axis=1)                        # (N,)
    reg_uncertain = reg_max_std > regressor_std_min_nm

    if combine == "intersection":
        mask = cls_uncertain & reg_uncertain
    else:
        mask = cls_uncertain | reg_uncertain

    # Combined score: sum of normalised uncertainties (0-1 ish).
    score = (cls_unc / max(cls_unc.max(), 1e-9)) + (reg_max_std / max(reg_max_std.max(), 1e-9))
    score_masked = np.where(mask, score, -np.inf)

    order = np.argsort(-score_masked)
    selected = order[: max(0, top_k)]
    selected = selected[np.isfinite(score_masked[selected])]

    diag = {
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
    return selected, diag

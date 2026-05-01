"""Stage 06A — surrogate-driven nominal-yield-proxy scoring.

For each base recipe we draw N process variations, run the closed Stage
04C classifier + 4-target regressor + auxiliary CD_fixed regressor on
all variations, average the per-class probabilities, and combine into
yield_score per the user-specified weights.

Vectorised: all (N_recipes × N_variations) variations are stacked into a
single feature matrix and submitted to the surrogate in one call. This
is the difference between ~30 s and ~hours for 5,000 × 200 = 1 M.

NOTE: the 04C classifier exposes `classes_` in fitted order. We map by
name so the score formula is robust to whatever order sklearn happened
to settle on.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .candidate_sampler import CandidateSpace
from .metrics_io import FEATURE_KEYS, REGRESSION_TARGETS
from .process_variation import (
    VariationSpec,
    feature_matrix_from_recipes,
    sample_variations,
)


# REGRESSION_TARGETS = [CD_locked_nm, LER_CD_locked_nm, area_frac, P_line_margin]
LER_LOCKED_IDX = REGRESSION_TARGETS.index("LER_CD_locked_nm")
CD_LOCKED_IDX  = REGRESSION_TARGETS.index("CD_locked_nm")


@dataclass
class YieldScoreConfig:
    class_weights: dict[str, float]
    cd_target_nm: float
    cd_tolerance_nm: float
    cd_penalty_weight: float
    ler_max_nm: float
    ler_penalty_weight: float

    @classmethod
    def from_yaml_dict(cls, d: dict) -> "YieldScoreConfig":
        return cls(
            class_weights=dict(d["class_weights"]),
            cd_target_nm=float(d["cd_target_nm"]),
            cd_tolerance_nm=float(d["cd_tolerance_nm"]),
            cd_penalty_weight=float(d.get("cd_penalty_weight", 1.0)),
            ler_max_nm=float(d["ler_max_nm"]),
            ler_penalty_weight=float(d.get("ler_penalty_weight", 1.0)),
        )


# Columns we always emit per recipe so the downstream summary CSV /
# pareto plot / sensitivity script can read a stable layout.
SUMMARY_COLUMNS = [
    "recipe_id",
    "yield_score",
    "p_robust_valid",
    "p_margin_risk",
    "p_roughness_degraded",
    "p_under_exposed",
    "p_merged",
    "p_numerical_invalid",
    "cd_error_penalty",
    "ler_penalty",
    "mean_cd_fixed",   "std_cd_fixed",
    "mean_cd_locked",  "std_cd_locked",
    "mean_ler_locked", "std_ler_locked",
    "mean_area_frac",  "std_area_frac",
    "mean_p_line_margin", "std_p_line_margin",
] + FEATURE_KEYS


def _expand_variations(
    recipes: list[dict],
    spec: VariationSpec,
    n_var: int,
    space: CandidateSpace,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (X_stacked, recipe_index) where X_stacked has shape
    (N_recipes * n_var, n_features) and recipe_index maps each row back
    to its base-recipe index."""
    base_rng = np.random.default_rng(seed)
    rows: list[dict] = []
    rec_idx: list[int] = []
    for i, base in enumerate(recipes):
        # Per-recipe RNG so variation streams are independent and
        # reproducible regardless of the recipe's position in the list.
        sub_seed = int(base_rng.integers(0, 2**31 - 1))
        sub_rng = np.random.default_rng(sub_seed)
        var_rows = sample_variations(base, spec, n_var, space, rng=sub_rng)
        rows.extend(var_rows)
        rec_idx.extend([i] * n_var)
    X = feature_matrix_from_recipes(rows, FEATURE_KEYS)
    return X, np.asarray(rec_idx, dtype=np.int64)


def _per_recipe_mean_std(values: np.ndarray, idx: np.ndarray, n_recipes: int) -> tuple[np.ndarray, np.ndarray]:
    """Group `values` by `idx` and return (mean, std) of length n_recipes."""
    means = np.zeros(n_recipes, dtype=np.float64)
    stds  = np.zeros(n_recipes, dtype=np.float64)
    for r in range(n_recipes):
        mask = idx == r
        v = values[mask]
        if v.size == 0:
            means[r] = np.nan
            stds[r]  = np.nan
        else:
            means[r] = float(np.mean(v))
            stds[r]  = float(np.std(v))
    return means, stds


def _per_recipe_mean_proba(probs: np.ndarray, idx: np.ndarray, n_recipes: int) -> np.ndarray:
    """Average per-class probabilities by recipe index. probs shape
    (N_total, n_classes). Returns shape (n_recipes, n_classes)."""
    out = np.zeros((n_recipes, probs.shape[1]), dtype=np.float64)
    for r in range(n_recipes):
        mask = idx == r
        if mask.any():
            out[r] = probs[mask].mean(axis=0)
    return out


def evaluate_recipes(
    recipes: list[dict],
    classifier,
    regressor4,
    cd_fixed_aux,
    spec: VariationSpec,
    n_var: int,
    space: CandidateSpace,
    score_cfg: YieldScoreConfig,
    seed: int = 0,
) -> list[dict]:
    """Score each recipe end-to-end. Returns a list of dicts, one per
    recipe, in the same order as `recipes`. Output columns line up with
    SUMMARY_COLUMNS."""
    n = len(recipes)
    if n == 0:
        return []

    X_all, rec_idx = _expand_variations(recipes, spec, n_var, space, seed=seed)

    proba = classifier.predict_proba(X_all)            # (N_total, n_classes)
    classes = list(classifier.classes_)
    y4 = regressor4.predict(X_all)                      # (N_total, 4)
    cd_fixed = cd_fixed_aux.predict(X_all)              # (N_total,)

    p_recipe = _per_recipe_mean_proba(proba, rec_idx, n)

    # Per-recipe regression mean/std.
    cd_locked_mean, cd_locked_std   = _per_recipe_mean_std(y4[:, CD_LOCKED_IDX],  rec_idx, n)
    ler_locked_mean, ler_locked_std = _per_recipe_mean_std(y4[:, LER_LOCKED_IDX], rec_idx, n)
    area_mean, area_std             = _per_recipe_mean_std(y4[:, REGRESSION_TARGETS.index("area_frac")],   rec_idx, n)
    margin_mean, margin_std         = _per_recipe_mean_std(y4[:, REGRESSION_TARGETS.index("P_line_margin")],rec_idx, n)
    cd_fixed_mean, cd_fixed_std     = _per_recipe_mean_std(cd_fixed,             rec_idx, n)

    # Penalties.
    cd_pen = np.maximum(
        0.0,
        np.abs(cd_fixed_mean - score_cfg.cd_target_nm) - score_cfg.cd_tolerance_nm,
    ) / max(score_cfg.cd_tolerance_nm, 1e-12)
    ler_pen = np.maximum(0.0, ler_locked_mean - score_cfg.ler_max_nm) / 1.0

    # Class-prob × weight contribution.
    score_class = np.zeros(n, dtype=np.float64)
    class_to_col = {c: j for j, c in enumerate(classes)}
    for cname, w in score_cfg.class_weights.items():
        if cname not in class_to_col:
            continue
        score_class += float(w) * p_recipe[:, class_to_col[cname]]

    score = (
        score_class
        - score_cfg.cd_penalty_weight  * cd_pen
        - score_cfg.ler_penalty_weight * ler_pen
    )

    # Helper to read a class probability column safely.
    def pcol(name: str) -> np.ndarray:
        if name not in class_to_col:
            return np.zeros(n, dtype=np.float64)
        return p_recipe[:, class_to_col[name]]

    out = []
    for i, base in enumerate(recipes):
        row = {
            "recipe_id": base.get("_id", i),
            "yield_score": float(score[i]),
            "p_robust_valid":       float(pcol("robust_valid")[i]),
            "p_margin_risk":        float(pcol("margin_risk")[i]),
            "p_roughness_degraded": float(pcol("roughness_degraded")[i]),
            "p_under_exposed":      float(pcol("under_exposed")[i]),
            "p_merged":             float(pcol("merged")[i]),
            "p_numerical_invalid":  float(pcol("numerical_invalid")[i]),
            "cd_error_penalty":     float(cd_pen[i]),
            "ler_penalty":          float(ler_pen[i]),
            "mean_cd_fixed":        float(cd_fixed_mean[i]),
            "std_cd_fixed":         float(cd_fixed_std[i]),
            "mean_cd_locked":       float(cd_locked_mean[i]),
            "std_cd_locked":        float(cd_locked_std[i]),
            "mean_ler_locked":      float(ler_locked_mean[i]),
            "std_ler_locked":       float(ler_locked_std[i]),
            "mean_area_frac":       float(area_mean[i]),
            "std_area_frac":        float(area_std[i]),
            "mean_p_line_margin":   float(margin_mean[i]),
            "std_p_line_margin":    float(margin_std[i]),
        }
        for k in FEATURE_KEYS:
            row[k] = float(base[k])
        out.append(row)
    return out


def evaluate_single_recipe(
    recipe: dict,
    classifier,
    regressor4,
    cd_fixed_aux,
    spec: VariationSpec,
    n_var: int,
    space: CandidateSpace,
    score_cfg: YieldScoreConfig,
    seed: int = 0,
) -> dict:
    """Convenience wrapper for the v2-frozen-OP baseline. Identical pipeline."""
    rows = evaluate_recipes(
        [recipe], classifier, regressor4, cd_fixed_aux,
        spec, n_var, space, score_cfg, seed=seed,
    )
    return rows[0]

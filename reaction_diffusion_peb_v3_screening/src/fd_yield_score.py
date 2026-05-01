"""Stage 06B — empirical FD yield-score helper.

Stage 06A scored a recipe as
    yield_score = sum_c w_c · P_surrogate(c | x_v)            (averaged over MC variations v)
                  − cd_penalty(mean(CD_fixed))
                  − ler_penalty(mean(LER_CD_locked))

For Stage 06B we replace the surrogate by FD outputs. Every FD run
returns a *hard* label, so the per-class probability is just the
empirical frequency over the MC ensemble:
    P_FD(c) = (#FD runs with label c) / N

Apart from that substitution the formula is identical, so an FD MC
yield_score is directly comparable to a surrogate yield_score under the
same scoring config.

For a single nominal FD run (no MC) the per-class frequency degenerates
to a one-hot, and the FD yield_score becomes
    1.0 + w_label_c − cd_pen − ler_pen     (label one-hot probability = 1)

This gives a usable per-recipe scalar even with N=1 — it is documented
in the docstring of `nominal_yield_score`.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from typing import Iterable

import numpy as np

from .yield_optimizer import YieldScoreConfig


# Same canonical class set as the labeller.
ALL_CLASSES = (
    "robust_valid",
    "margin_risk",
    "merged",
    "under_exposed",
    "roughness_degraded",
    "numerical_invalid",
)

# FD CSV uses CD_final_nm (= v2 helper output), which is the same
# physical quantity that Stage 06A's auxiliary regressor learned to
# predict and which the score formula reads as "CD_fixed".
CD_FIXED_FD_FIELD = "CD_final_nm"
LER_FD_FIELD      = "LER_CD_locked_nm"
CD_LOCKED_FD_FIELD = "CD_locked_nm"
P_LINE_MARGIN_FD_FIELD = "P_line_margin"
AREA_FRAC_FD_FIELD = "area_frac"


def _safe_float(v) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return float("nan")


def empirical_class_probs(rows: Iterable[dict]) -> dict[str, float]:
    """Return P_FD(c) for c in ALL_CLASSES, computed as (count_c / N).
    Unknown labels are silently dropped so the denominator stays valid
    even if the labeller introduces a new class — the score formula
    only weights labels listed in ALL_CLASSES."""
    rows = list(rows)
    n = len(rows)
    if n == 0:
        return {c: 0.0 for c in ALL_CLASSES}
    counts = Counter(r.get("label", "") for r in rows)
    return {c: float(counts.get(c, 0)) / n for c in ALL_CLASSES}


def _agg_mean_std(rows: list[dict], field: str) -> tuple[float, float]:
    vals = np.array([_safe_float(r.get(field)) for r in rows], dtype=np.float64)
    finite = vals[np.isfinite(vals)]
    if finite.size == 0:
        return float("nan"), float("nan")
    return float(np.mean(finite)), float(np.std(finite))


def fd_yield_score_from_rows(
    rows: list[dict],
    cfg: YieldScoreConfig,
) -> dict:
    """Return a flat dict carrying P_FD per class, mean/std FD metrics,
    and the empirical FD yield_score for one base recipe.

    With N=1 (Part 1 nominal verification) the per-class probabilities
    are one-hot on the label of that single FD run, so the score is the
    weight-of-that-label minus the CD/LER penalties.
    """
    p = empirical_class_probs(rows)
    cd_fixed_mean, cd_fixed_std   = _agg_mean_std(rows, CD_FIXED_FD_FIELD)
    cd_locked_mean, cd_locked_std = _agg_mean_std(rows, CD_LOCKED_FD_FIELD)
    ler_locked_mean, ler_locked_std = _agg_mean_std(rows, LER_FD_FIELD)
    margin_mean, margin_std       = _agg_mean_std(rows, P_LINE_MARGIN_FD_FIELD)
    area_mean,   area_std         = _agg_mean_std(rows, AREA_FRAC_FD_FIELD)

    if not np.isfinite(cd_fixed_mean):
        cd_pen = 0.0
    else:
        cd_pen = max(
            0.0,
            abs(cd_fixed_mean - cfg.cd_target_nm) - cfg.cd_tolerance_nm,
        ) / max(cfg.cd_tolerance_nm, 1e-12)

    if not np.isfinite(ler_locked_mean):
        ler_pen = 0.0
    else:
        ler_pen = max(0.0, ler_locked_mean - cfg.ler_max_nm)

    score_class = 0.0
    for cname, w in cfg.class_weights.items():
        score_class += float(w) * p.get(cname, 0.0)
    score = (score_class
             - cfg.cd_penalty_weight  * cd_pen
             - cfg.ler_penalty_weight * ler_pen)

    return {
        "n_fd_rows":        len(rows),
        "FD_yield_score":   float(score),
        "P_FD_robust_valid":       float(p["robust_valid"]),
        "P_FD_margin_risk":        float(p["margin_risk"]),
        "P_FD_merged":             float(p["merged"]),
        "P_FD_under_exposed":      float(p["under_exposed"]),
        "P_FD_roughness_degraded": float(p["roughness_degraded"]),
        "P_FD_numerical_invalid":  float(p["numerical_invalid"]),
        "FD_cd_error_penalty":     float(cd_pen),
        "FD_ler_penalty":          float(ler_pen),
        "mean_cd_fixed_fd":        float(cd_fixed_mean),
        "std_cd_fixed_fd":         float(cd_fixed_std),
        "mean_cd_locked_fd":       float(cd_locked_mean),
        "std_cd_locked_fd":        float(cd_locked_std),
        "mean_ler_locked_fd":      float(ler_locked_mean),
        "std_ler_locked_fd":       float(ler_locked_std),
        "mean_p_line_margin_fd":   float(margin_mean),
        "std_p_line_margin_fd":    float(margin_std),
        "mean_area_frac_fd":       float(area_mean),
        "std_area_frac_fd":        float(area_std),
    }


def group_rows_by_recipe(
    rows: Iterable[dict],
    key: str = "source_recipe_id",
) -> dict[str, list[dict]]:
    """Bucket FD rows by their source recipe id so we can score per
    recipe independently."""
    out: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        out[str(r.get(key, ""))].append(r)
    return dict(out)


def fd_yield_score_per_recipe(
    rows: Iterable[dict],
    cfg: YieldScoreConfig,
    key: str = "source_recipe_id",
) -> dict[str, dict]:
    """Return {recipe_id: scoring_dict} for a flat list of FD rows."""
    buckets = group_rows_by_recipe(rows, key=key)
    return {rid: fd_yield_score_from_rows(rs, cfg) for rid, rs in buckets.items()}


def nominal_yield_score(row: dict, cfg: YieldScoreConfig) -> dict:
    """Convenience wrapper for Part 1: a single FD row produces a
    one-hot per-class probability vector. The returned dict matches the
    shape of `fd_yield_score_from_rows`."""
    return fd_yield_score_from_rows([row], cfg)


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    """Spearman ρ with tie correction (uses scipy's `spearmanr`, which
    averages ranks of tied values). Returns nan if fewer than 3 finite
    pairs or if either side is constant (after ties)."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    finite = np.isfinite(a) & np.isfinite(b)
    if finite.sum() < 3:
        return float("nan")
    a = a[finite]; b = b[finite]
    if np.unique(a).size < 2 or np.unique(b).size < 2:
        return float("nan")
    from scipy.stats import spearmanr
    rho, _ = spearmanr(a, b)
    return float(rho) if np.isfinite(rho) else float("nan")


def topk_overlap(rank_a: list[str], rank_b: list[str], k: int) -> int:
    """Return |set(rank_a[:k]) ∩ set(rank_b[:k])|."""
    return len(set(rank_a[:k]) & set(rank_b[:k]))

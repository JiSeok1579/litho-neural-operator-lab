"""Stage 06P -- evaluate the refreshed surrogate stack.

Reads:
    outputs/yield_optimization/stage06G_top_recipes.csv
    outputs/yield_optimization/stage06J_mode_b_top_recipes.csv
    outputs/yield_optimization/stage06J_mode_b_recipe_summary.csv
    outputs/yield_optimization/stage06J_mode_b_vs_mode_a_comparison.csv
    outputs/yield_optimization/stage06I_mode_a_final_recipes.yaml
    outputs/yield_optimization/stage06G_strict_score_config.yaml
    outputs/yield_optimization/stage06P_recipe_manifest.yaml
    outputs/yield_optimization/stage06M_B_J1453_time_deep_mc.csv
    outputs/yield_optimization/stage06M_time_deep_mc.csv
    outputs/labels/stage06J_mode_b_fd_sanity.csv
    outputs/labels/stage06J_mode_b_fd_mc_optional.csv
    outputs/labels/stage06J_B_fd_top10_mc.csv
    outputs/labels/stage06J_B_fd_representative_mc.csv
    outputs/labels/stage06J_B_fd_top100_nominal.csv
    outputs/logs/stage06H_fd_verification_summary.json
    outputs/logs/stage06L_false_pass_reduction.json
    outputs/logs/stage06P_model_metrics.json
    outputs/models/stage06L_*.joblib
    outputs/models/stage06P_*.joblib

Writes:
    outputs/yield_optimization/stage06P_mode_b_rescored_candidates.csv
    outputs/yield_optimization/stage06P_J1453_vs_G4867_prediction_check.csv
    outputs/yield_optimization/stage06P_false_pass_demotions.csv
    outputs/yield_optimization/stage06P_manifest_update.yaml
    outputs/logs/stage06P_mode_b_ranking_comparison.json
    outputs/logs/stage06P_false_pass_reduction.json
    outputs/logs/stage06P_time_window_learning.json
    outputs/figures/06_yield_optimization/
        stage06P_strict_score_pred_vs_fd.png
        stage06P_mode_b_ranking_before_after.png
        stage06P_J1453_vs_G4867_predicted_time_window.png
        stage06P_false_pass_demotions.png
        stage06P_feature_importance_strict_score.png
        stage06P_mode_a_vs_mode_b_score_distribution.png
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from reaction_diffusion_peb_v3_screening.src.candidate_sampler import (
    CandidateSpace,
)
from reaction_diffusion_peb_v3_screening.src.fd_yield_score import spearman, topk_overlap
from reaction_diffusion_peb_v3_screening.src.metrics_io import (
    FEATURE_KEYS, load_model, read_labels_csv,
)
from reaction_diffusion_peb_v3_screening.src.process_variation import (
    VariationSpec, sample_variations,
)
from reaction_diffusion_peb_v3_screening.src.yield_optimizer import (
    YieldScoreConfig, evaluate_recipes,
)

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_stage06g_strict_optimization import (  # noqa: E402
    StrictScoreConfig, compute_strict_score,
)
from build_stage06l_dataset import per_row_strict_score  # noqa: E402


V3_DIR = ROOT / "reaction_diffusion_peb_v3_screening"
CD_TARGET_NM = 15.0
HARD_FAIL_LABELS = {"under_exposed", "merged",
                     "roughness_degraded", "numerical_invalid"}
J1453_DET_TIME_OFFSETS = [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5]


def _safe_float(v) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return float("nan")


def _coerce(rows, keys):
    for r in rows:
        for k in keys:
            if k in r:
                r[k] = _safe_float(r.get(k))


def _build_X(rows: list[dict]) -> np.ndarray:
    X = np.zeros((len(rows), len(FEATURE_KEYS)), dtype=np.float64)
    for i, r in enumerate(rows):
        for j, k in enumerate(FEATURE_KEYS):
            X[i, j] = _safe_float(r.get(k))
    return X


# --------------------------------------------------------------------------
# Re-score candidates with one surrogate stack: classifier + 4-target +
# aux + strict_score head. Mirrors analyze_stage06l rescore_candidates.
# --------------------------------------------------------------------------
def rescore_candidates(rows: list[dict], clf, reg, aux, strict_reg,
                          var_spec: VariationSpec, n_var: int,
                          space: CandidateSpace, score_cfg: YieldScoreConfig,
                          strict_cfg: StrictScoreConfig, seed: int,
                          tag: str) -> list[dict]:
    cand_pool = []
    for r in rows:
        cand = {k: float(r[k]) for k in FEATURE_KEYS}
        cand["pitch_nm"]    = float(cand["pitch_nm"])
        cand["line_cd_nm"]  = cand["pitch_nm"] * cand["line_cd_ratio"]
        cand["domain_x_nm"] = cand["pitch_nm"] * 5.0
        cand["dose_norm"]   = cand["dose_mJ_cm2"] / float(space.fixed["reference_dose_mJ_cm2"])
        for fk, fv in space.fixed.items():
            cand.setdefault(fk, fv)
        cand["_id"] = f"{tag}_rescore_{r.get('recipe_id', r.get('_id', '?'))}"
        cand_pool.append((r, cand))

    eval_rows = evaluate_recipes(
        [c for _, c in cand_pool], clf, reg, aux,
        var_spec, n_var, space, score_cfg, seed=seed,
    )
    for r in eval_rows:
        r.update(compute_strict_score(r, strict_cfg))

    # Per-row strict_score head needs raw variation features.
    base_rng = np.random.default_rng(seed + 1)
    out: list[dict] = []
    for (orig, cand), eval_row in zip(cand_pool, eval_rows):
        sub_rng = np.random.default_rng(int(base_rng.integers(0, 2**31 - 1)))
        variations = sample_variations(cand, var_spec, n_var, space, rng=sub_rng)
        Xv = np.array([[_safe_float(v.get(k)) for k in FEATURE_KEYS]
                         for v in variations])
        per_row_pred = strict_reg.predict(Xv)
        out.append({
            **{k: _safe_float(orig.get(k)) for k in FEATURE_KEYS},
            "recipe_id":         orig.get("recipe_id", orig.get("_id", "?")),
            f"strict_score_{tag}_eval":         float(eval_row["strict_score"]),
            f"strict_score_{tag}_direct_mean":  float(np.mean(per_row_pred)),
            f"strict_score_{tag}_direct_std":   float(np.std(per_row_pred)),
            f"yield_score_{tag}":               float(eval_row["yield_score"]),
            f"p_robust_valid_{tag}":            float(eval_row["p_robust_valid"]),
            f"mean_cd_fixed_{tag}":             float(eval_row["mean_cd_fixed"]),
            f"std_cd_fixed_{tag}":              float(eval_row["std_cd_fixed"]),
            f"mean_ler_locked_{tag}":           float(eval_row["mean_ler_locked"]),
            f"std_ler_locked_{tag}":            float(eval_row["std_ler_locked"]),
            f"mean_p_line_margin_{tag}":        float(eval_row["mean_p_line_margin"]),
        })
    return out


# --------------------------------------------------------------------------
# FD MC truth lookup table (Mode A 06H + Mode B 06J + 06J-B).
# --------------------------------------------------------------------------
def _mc_aggregate_strict_pass(rows: list[dict], cd_tol: float, ler_cap: float,
                                 strict_yaml: dict) -> dict:
    n = len(rows)
    if n == 0:
        return {"n": 0, "strict_pass_prob": float("nan"),
                "robust_prob": float("nan"),
                "mean_strict_score_per_row": float("nan")}
    n_robust = sum(1 for r in rows if str(r.get("label", "")) == "robust_valid")
    n_sp = sum(
        1 for r in rows
        if str(r.get("label", "")) == "robust_valid"
            and abs(_safe_float(r.get("CD_final_nm")) - CD_TARGET_NM) <= cd_tol
            and _safe_float(r.get("LER_CD_locked_nm")) <= ler_cap
    )
    strict_per_row = np.array([per_row_strict_score(r, strict_yaml) for r in rows])
    return {
        "n": int(n),
        "strict_pass_prob": float(n_sp / n),
        "robust_prob":      float(n_robust / n),
        "mean_strict_score_per_row": float(np.mean(strict_per_row)),
    }


def build_fd_truth(*, summary_06h: dict,
                      fd_06j_mc: list[dict],
                      fd_06j_b_mc: list[dict],
                      cd_tol: float, ler_cap: float,
                      strict_yaml: dict) -> dict:
    fd_truth: dict[str, dict] = {}
    # Mode A 06H FD MC.
    for r in summary_06h.get("fd_top10_mc_aggr", []):
        fd_truth[r["recipe_id"]] = {
            "mode": "mode_a",
            "fd_mc_strict_pass_prob": float(r.get("p_strict_pass", float("nan"))),
            "fd_mc_n":                int(r.get("n_mc", 0)),
            "source":                  "06H_top10_mc_100var",
        }
    for r in summary_06h.get("fd_rep_mc_aggr", []):
        fd_truth[r["recipe_id"]] = {
            "mode": "mode_a",
            "fd_mc_strict_pass_prob": float(r.get("p_strict_pass", float("nan"))),
            "fd_mc_n":                int(r.get("n_mc", 0)),
            "source":                  "06H_rep_mc_300var",
            "rep_kind":                str(r.get("rep_kind", "")),
        }
    # Mode B 06J FD MC.
    by_recipe_06j: dict[str, list[dict]] = {}
    for r in fd_06j_mc:
        rid = str(r.get("source_recipe_id", ""))
        by_recipe_06j.setdefault(rid, []).append(r)
    for rid, rs in by_recipe_06j.items():
        agg = _mc_aggregate_strict_pass(rs, cd_tol, ler_cap, strict_yaml)
        fd_truth[rid] = {
            "mode": "mode_b",
            "fd_mc_strict_pass_prob": agg["strict_pass_prob"],
            "fd_mc_n":                agg["n"],
            "source":                  "06J_mc_100var",
        }
    # Mode B 06J-B FD MC (top-10 + representative); overrides 06J on
    # overlap because 06J-B is denser and at 06L-ranking scale.
    by_recipe_06jb: dict[str, list[dict]] = {}
    for r in fd_06j_b_mc:
        rid = str(r.get("source_recipe_id", ""))
        by_recipe_06jb.setdefault(rid, []).append(r)
    for rid, rs in by_recipe_06jb.items():
        agg = _mc_aggregate_strict_pass(rs, cd_tol, ler_cap, strict_yaml)
        fd_truth[rid] = {
            "mode": "mode_b",
            "fd_mc_strict_pass_prob": agg["strict_pass_prob"],
            "fd_mc_n":                agg["n"],
            "source":                  "06J_B_mc",
        }
    return fd_truth


# --------------------------------------------------------------------------
# Time-window learning: compare 06P-predicted per-offset trends to FD
# truth from 06M (G_4867) and 06M-B (J_1453).
# --------------------------------------------------------------------------
def _per_offset_pred(rows_at_offset: list[dict], clf, reg, aux, strict_reg,
                       cd_tol: float, ler_cap: float) -> dict:
    """Predict surrogate-side strict_pass_prob / robust_valid_prob /
    defect_prob / mean_strict_score / mean_cd_error / mean_ler / mean_margin
    for the FD rows at one time offset."""
    if not rows_at_offset:
        return {"n": 0}
    X = _build_X(rows_at_offset)
    proba = clf.predict_proba(X)
    classes = list(clf.classes_)
    Y4 = reg.predict(X)
    cd_pred = aux.predict(X)
    strict_pred = strict_reg.predict(X)

    def _pcol(name: str) -> np.ndarray:
        if name not in classes:
            return np.zeros(len(rows_at_offset))
        return proba[:, classes.index(name)]

    p_robust = _pcol("robust_valid")
    p_margin = _pcol("margin_risk")
    defects = (_pcol("under_exposed") + _pcol("merged")
                + _pcol("roughness_degraded") + _pcol("numerical_invalid"))

    # Surrogate-side strict_pass proxy: predicted CD within tol AND
    # predicted LER under cap AND predicted-most-likely-class==robust_valid.
    pred_cls = np.array([classes[int(np.argmax(p))] for p in proba])
    cd_within = np.abs(cd_pred - CD_TARGET_NM) <= cd_tol
    ler_under = Y4[:, 1] <= ler_cap   # LER_CD_locked_nm is index 1
    strict_pass_pred = (cd_within & ler_under & (pred_cls == "robust_valid")).astype(float)

    return {
        "n": len(rows_at_offset),
        "pred_strict_pass_prob":  float(np.mean(strict_pass_pred)),
        "pred_robust_valid_prob": float(np.mean(p_robust)),
        "pred_margin_risk_prob":  float(np.mean(p_margin)),
        "pred_defect_prob":       float(np.mean(defects)),
        "pred_mean_strict_score": float(np.mean(strict_pred)),
        "pred_mean_cd_pred":      float(np.mean(cd_pred)),
        "pred_mean_cd_error":     float(np.mean(np.abs(cd_pred - CD_TARGET_NM))),
        "pred_mean_ler":          float(np.mean(Y4[:, 1])),
        "pred_mean_margin":       float(np.mean(Y4[:, 3])),
    }


def _fd_per_offset(rows_at_offset: list[dict], cd_tol: float,
                     ler_cap: float, strict_yaml: dict) -> dict:
    if not rows_at_offset:
        return {"n": 0}
    n = len(rows_at_offset)
    n_robust = sum(1 for r in rows_at_offset
                    if str(r.get("label", "")) == "robust_valid")
    n_margin = sum(1 for r in rows_at_offset
                    if str(r.get("label", "")) == "margin_risk")
    n_defect = sum(1 for r in rows_at_offset
                    if str(r.get("label", "")) in HARD_FAIL_LABELS)
    n_sp = sum(
        1 for r in rows_at_offset
        if str(r.get("label", "")) == "robust_valid"
            and abs(_safe_float(r.get("CD_final_nm")) - CD_TARGET_NM) <= cd_tol
            and _safe_float(r.get("LER_CD_locked_nm")) <= ler_cap
    )
    strict_per_row = np.array([per_row_strict_score(r, strict_yaml)
                                  for r in rows_at_offset])
    cd_err = np.array([abs(_safe_float(r.get("CD_final_nm")) - CD_TARGET_NM)
                          for r in rows_at_offset])
    ler = np.array([_safe_float(r.get("LER_CD_locked_nm"))
                       for r in rows_at_offset])
    mar = np.array([_safe_float(r.get("P_line_margin"))
                       for r in rows_at_offset])
    finite_cd  = np.isfinite(cd_err)
    finite_ler = np.isfinite(ler)
    finite_mar = np.isfinite(mar)
    return {
        "n": int(n),
        "fd_strict_pass_prob":  float(n_sp / n),
        "fd_robust_valid_prob": float(n_robust / n),
        "fd_margin_risk_prob":  float(n_margin / n),
        "fd_defect_prob":       float(n_defect / n),
        "fd_mean_strict_score": float(np.mean(strict_per_row)),
        "fd_mean_cd_error":     float(np.mean(cd_err[finite_cd])) if finite_cd.any() else float("nan"),
        "fd_mean_ler":          float(np.mean(ler[finite_ler])) if finite_ler.any() else float("nan"),
        "fd_mean_margin":       float(np.mean(mar[finite_mar])) if finite_mar.any() else float("nan"),
    }


def time_window_learning_check(rows_06m: list[dict],
                                  rows_06m_b: list[dict],
                                  *, clf, reg, aux, strict_reg,
                                  cd_tol: float, ler_cap: float,
                                  strict_yaml: dict) -> dict:
    """For each integer offset in [-5,5], aggregate 06P prediction
    metrics on 06M (G_4867) and 06M-B (J_1453) deterministic-offset MC
    rows. Compare to FD truth at the same offset and recipe."""
    def _det_int(r: dict) -> int | None:
        if str(r.get("scenario", "")) != "det_offset":
            return None
        t = _safe_float(r.get("time_offset_s"))
        if not np.isfinite(t):
            return None
        if abs(t - round(t)) > 1e-6:
            return None
        ti = int(round(t))
        return ti if ti in J1453_DET_TIME_OFFSETS else None

    g_by_off: dict[int, list[dict]] = {}
    for r in rows_06m:
        ti = _det_int(r)
        if ti is not None:
            g_by_off.setdefault(ti, []).append(r)
    j_by_off: dict[int, list[dict]] = {}
    for r in rows_06m_b:
        ti = _det_int(r)
        if ti is not None:
            j_by_off.setdefault(ti, []).append(r)

    table_rows: list[dict] = []
    for off in J1453_DET_TIME_OFFSETS:
        g_rows = g_by_off.get(off, [])
        j_rows = j_by_off.get(off, [])
        g_pred = _per_offset_pred(g_rows, clf, reg, aux, strict_reg,
                                       cd_tol, ler_cap)
        j_pred = _per_offset_pred(j_rows, clf, reg, aux, strict_reg,
                                       cd_tol, ler_cap)
        g_fd   = _fd_per_offset(g_rows, cd_tol, ler_cap, strict_yaml)
        j_fd   = _fd_per_offset(j_rows, cd_tol, ler_cap, strict_yaml)
        table_rows.append({
            "time_offset_s":            float(off),
            "G4867_pred_strict_pass":   g_pred.get("pred_strict_pass_prob", float("nan")),
            "J1453_pred_strict_pass":   j_pred.get("pred_strict_pass_prob", float("nan")),
            "G4867_fd_strict_pass":     g_fd.get("fd_strict_pass_prob", float("nan")),
            "J1453_fd_strict_pass":     j_fd.get("fd_strict_pass_prob", float("nan")),
            "G4867_pred_robust_valid":  g_pred.get("pred_robust_valid_prob", float("nan")),
            "J1453_pred_robust_valid":  j_pred.get("pred_robust_valid_prob", float("nan")),
            "G4867_fd_robust_valid":    g_fd.get("fd_robust_valid_prob", float("nan")),
            "J1453_fd_robust_valid":    j_fd.get("fd_robust_valid_prob", float("nan")),
            "G4867_pred_defect":        g_pred.get("pred_defect_prob", float("nan")),
            "J1453_pred_defect":        j_pred.get("pred_defect_prob", float("nan")),
            "G4867_fd_defect":          g_fd.get("fd_defect_prob", float("nan")),
            "J1453_fd_defect":          j_fd.get("fd_defect_prob", float("nan")),
            "G4867_pred_mean_strict_score": g_pred.get("pred_mean_strict_score", float("nan")),
            "J1453_pred_mean_strict_score": j_pred.get("pred_mean_strict_score", float("nan")),
            "G4867_fd_mean_strict_score":   g_fd.get("fd_mean_strict_score", float("nan")),
            "J1453_fd_mean_strict_score":   j_fd.get("fd_mean_strict_score", float("nan")),
        })

    # Spearman of pred vs FD per recipe (over offsets).
    def _rho(pred_key: str, fd_key: str) -> float:
        p = np.array([row[pred_key] for row in table_rows])
        f = np.array([row[fd_key]   for row in table_rows])
        m = np.isfinite(p) & np.isfinite(f)
        if m.sum() < 3:
            return float("nan")
        return spearman(p[m], f[m])

    summary = {
        "G_4867": {
            "spearman_strict_pass":   _rho("G4867_pred_strict_pass",
                                              "G4867_fd_strict_pass"),
            "spearman_robust_valid":  _rho("G4867_pred_robust_valid",
                                              "G4867_fd_robust_valid"),
            "spearman_defect":        _rho("G4867_pred_defect",
                                              "G4867_fd_defect"),
            "spearman_mean_strict_score": _rho(
                "G4867_pred_mean_strict_score",
                "G4867_fd_mean_strict_score",
            ),
        },
        "J_1453": {
            "spearman_strict_pass":   _rho("J1453_pred_strict_pass",
                                              "J1453_fd_strict_pass"),
            "spearman_robust_valid":  _rho("J1453_pred_robust_valid",
                                              "J1453_fd_robust_valid"),
            "spearman_defect":        _rho("J1453_pred_defect",
                                              "J1453_fd_defect"),
            "spearman_mean_strict_score": _rho(
                "J1453_pred_mean_strict_score",
                "J1453_fd_mean_strict_score",
            ),
        },
    }

    # Predicted relative advantage: J_1453 - G_4867 strict_pass per offset.
    rel_pred = np.array([row["J1453_pred_strict_pass"] - row["G4867_pred_strict_pass"]
                            for row in table_rows])
    rel_fd   = np.array([row["J1453_fd_strict_pass"] - row["G4867_fd_strict_pass"]
                            for row in table_rows])
    fin = np.isfinite(rel_pred) & np.isfinite(rel_fd)
    summary["J_minus_G_strict_pass"] = {
        "spearman_pred_vs_fd": (spearman(rel_pred[fin], rel_fd[fin])
                                  if fin.sum() >= 3 else float("nan")),
        "n_pred_advantage":   int(np.sum(rel_pred[fin] > 0)),
        "n_fd_advantage":     int(np.sum(rel_fd[fin] > 0)),
        "mean_pred_advantage": float(np.mean(rel_pred[fin]))
                                  if fin.any() else float("nan"),
        "mean_fd_advantage":   float(np.mean(rel_fd[fin]))
                                  if fin.any() else float("nan"),
    }

    return {"per_offset": table_rows, "summary": summary}


# --------------------------------------------------------------------------
# Plot helpers
# --------------------------------------------------------------------------
def plot_strict_score_pred_vs_fd(pairs: list[dict], out_path: Path) -> None:
    if not pairs:
        return
    rho = spearman(np.array([r["pred_strict_06p"] for r in pairs]),
                     np.array([r["fd_mc_strict_pass_prob"] for r in pairs]))
    fig, ax = plt.subplots(figsize=(9.0, 6.5))
    colors = {"mode_a": "#1f77b4", "mode_b": "#d62728"}
    for mode, color in colors.items():
        sub = [p for p in pairs if p["mode"] == mode]
        if not sub:
            continue
        ax.scatter([p["pred_strict_06p"] for p in sub],
                    [p["fd_mc_strict_pass_prob"] for p in sub],
                    c=color, s=80, alpha=0.85, edgecolor="white", lw=0.6,
                    label=f"{mode} (n={len(sub)})")
        for p in sub:
            ax.annotate(p["recipe_id"],
                          (p["pred_strict_06p"], p["fd_mc_strict_pass_prob"]),
                          fontsize=8, xytext=(4, 4),
                          textcoords="offset points")
    ax.set_xlabel("06P direct strict_score (mean over 200 MC variations)")
    ax.set_ylabel("FD MC strict_pass_prob")
    ax.set_title(f"Stage 06P -- direct strict_score head vs FD MC truth  "
                  f"(Spearman rho = {rho:.3f})")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best", fontsize=10)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_rank_before_after(rs_06l: list[dict], rs_06p: list[dict],
                              highlight_ids: dict, out_path: Path,
                              title: str) -> None:
    fig, ax = plt.subplots(figsize=(11.0, 6.0))
    rank_06l = {r["recipe_id"]: i + 1
                  for i, r in enumerate(sorted(
                      rs_06l, key=lambda x: -float(x["strict_score_06l_direct_mean"])))}
    rank_06p = {r["recipe_id"]: i + 1
                  for i, r in enumerate(sorted(
                      rs_06p, key=lambda x: -float(x["strict_score_06p_direct_mean"])))}
    for r in rs_06l:
        rid = r["recipe_id"]
        if rid not in rank_06p:
            continue
        old = rank_06l[rid]; new = rank_06p[rid]
        is_hi = rid in highlight_ids
        ax.plot([0, 1], [old, new], "-",
                 color=highlight_ids.get(rid, "#9aaecf") if is_hi else "#9aaecf",
                 lw=1.7 if is_hi else 0.4,
                 alpha=0.95 if is_hi else 0.30,
                 zorder=5 if is_hi else 1)
        if is_hi:
            ax.scatter([0, 1], [old, new], s=44,
                        color=highlight_ids[rid], zorder=6)
            ax.annotate(rid, (1.02, new), fontsize=8,
                          color=highlight_ids[rid], verticalalignment="center")
    ax.invert_yaxis()
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["06L direct strict_score rank",
                          "06P direct strict_score rank"])
    ax.set_ylabel("rank (lower = better)")
    ax.set_title(title)
    ax.set_xlim(-0.05, 1.20)
    ax.grid(True, alpha=0.25, axis="y")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_J1453_vs_G4867_time_window(per_off: list[dict], out_path: Path) -> None:
    if not per_off:
        return
    offs = np.array([r["time_offset_s"] for r in per_off])
    fig, axes = plt.subplots(2, 2, figsize=(12.5, 8.5))
    pairs = [
        ("strict_pass", "strict_pass_prob",
         "G4867_pred_strict_pass", "J1453_pred_strict_pass",
         "G4867_fd_strict_pass",   "J1453_fd_strict_pass"),
        ("robust_valid", "robust_valid_prob",
         "G4867_pred_robust_valid", "J1453_pred_robust_valid",
         "G4867_fd_robust_valid",   "J1453_fd_robust_valid"),
        ("defect", "defect_prob",
         "G4867_pred_defect", "J1453_pred_defect",
         "G4867_fd_defect",   "J1453_fd_defect"),
        ("mean_strict_score", "mean strict_score",
         "G4867_pred_mean_strict_score", "J1453_pred_mean_strict_score",
         "G4867_fd_mean_strict_score",   "J1453_fd_mean_strict_score"),
    ]
    for ax, (label, ylabel, gp, jp, gf, jf) in zip(axes.flat, pairs):
        ax.plot(offs, [r[gp] for r in per_off], "-o",
                 color="#1f77b4", label="G_4867 06P pred", lw=1.5)
        ax.plot(offs, [r[gf] for r in per_off], "--o",
                 color="#1f77b4", label="G_4867 FD truth", alpha=0.55)
        ax.plot(offs, [r[jp] for r in per_off], "-s",
                 color="#d62728", label="J_1453 06P pred", lw=1.5)
        ax.plot(offs, [r[jf] for r in per_off], "--s",
                 color="#d62728", label="J_1453 FD truth", alpha=0.55)
        ax.set_xlabel("time offset (s)")
        ax.set_ylabel(ylabel)
        ax.set_title(label)
        ax.grid(True, alpha=0.25)
        ax.legend(loc="best", fontsize=8)
    fig.suptitle("Stage 06P -- predicted vs FD time-window trend "
                  "(deterministic offsets, 100 MC each)",
                  fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_false_pass_demotions(demos: list[dict], out_path: Path) -> None:
    if not demos:
        fig, ax = plt.subplots(figsize=(8.5, 4.0))
        ax.text(0.5, 0.5, "No 06J false-PASS candidates available for "
                            "06P demotion analysis (or none had baseline ranks).",
                ha="center", va="center", fontsize=11)
        ax.axis("off")
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        return
    rids = [d["recipe_id"] for d in demos]
    rj   = np.array([float(d["rank_06j"]) for d in demos])
    rl   = np.array([float(d["rank_06l"]) for d in demos])
    rp   = np.array([float(d["rank_06p"]) for d in demos])
    fig, ax = plt.subplots(figsize=(11.5, 5.5))
    x = np.arange(len(rids))
    w = 0.27
    ax.bar(x - w, rj, width=w, color="#9aaecf",  label="06J surrogate rank")
    ax.bar(x,       rl, width=w, color="#d6a728",  label="06L direct strict_score rank")
    ax.bar(x + w, rp, width=w, color="#d62728",  label="06P direct strict_score rank")
    for i, d in enumerate(demos):
        ax.text(i, max(rj[i], rl[i], rp[i]) + 1,
                f"{d['fd_label']}", ha="center", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(rids, rotation=20, ha="right", fontsize=9)
    ax.set_xlabel("06J false-PASS recipe (FD label != robust_valid)")
    ax.set_ylabel("rank under each ranker (lower = higher up)")
    ax.set_title("Stage 06P -- ranking change for 06J false-PASS recipes "
                  "(06J vs 06L vs 06P)")
    ax.grid(True, alpha=0.25, axis="y")
    ax.legend(loc="best", fontsize=10)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_feature_importance(metrics: dict, out_path: Path) -> None:
    imp = metrics.get("feature_importance_strict_score", [])
    if not imp:
        return
    keys = FEATURE_KEYS[: len(imp)]
    order = np.argsort(imp)[::-1]
    keys = [keys[i] for i in order]
    vals = [imp[i] for i in order]
    fig, ax = plt.subplots(figsize=(11.0, 5.5))
    ax.bar(np.arange(len(keys)), vals, color="#d62728", alpha=0.85,
            edgecolor="#1f1f1f")
    ax.set_xticks(np.arange(len(keys)))
    ax.set_xticklabels(keys, rotation=25, ha="right", fontsize=9)
    ax.set_ylabel("strict_score head feature importance (06P RF)")
    ax.set_title("Stage 06P strict_score head feature importance")
    ax.grid(True, alpha=0.25, axis="y")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_mode_a_vs_mode_b_distribution(rescored_06g, rescored_06j, out_path):
    sa = np.array([_safe_float(r.get("strict_score_06p_direct_mean"))
                     for r in rescored_06g])
    sb = np.array([_safe_float(r.get("strict_score_06p_direct_mean"))
                     for r in rescored_06j])
    fig, ax = plt.subplots(figsize=(10.5, 5.5))
    bins = np.linspace(min(sa.min(), sb.min()), max(sa.max(), sb.max()), 40)
    ax.hist(sa, bins=bins, alpha=0.55, color="#1f77b4",
            label=f"Mode A 06G top-100 (n={len(sa)})")
    ax.hist(sb, bins=bins, alpha=0.55, color="#d62728",
            label=f"Mode B 06J top-100 (n={len(sb)})")
    ax.set_xlabel("06P direct strict_score (mean over 200 MC)")
    ax.set_ylabel("count")
    ax.set_title("Stage 06P -- Mode A vs Mode B strict_score distribution")
    ax.legend(loc="best", fontsize=10)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# --------------------------------------------------------------------------
# Driver
# --------------------------------------------------------------------------
def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str,
                   default=str(V3_DIR / "configs" / "yield_optimization.yaml"))
    p.add_argument("--strict_cfg_yaml", type=str,
                   default=str(V3_DIR / "outputs" / "yield_optimization"
                               / "stage06G_strict_score_config.yaml"))
    p.add_argument("--top_06g_csv", type=str,
                   default=str(V3_DIR / "outputs" / "yield_optimization"
                               / "stage06G_top_recipes.csv"))
    p.add_argument("--top_06j_csv", type=str,
                   default=str(V3_DIR / "outputs" / "yield_optimization"
                               / "stage06J_mode_b_top_recipes.csv"))
    p.add_argument("--summary_06j_csv", type=str,
                   default=str(V3_DIR / "outputs" / "yield_optimization"
                               / "stage06J_mode_b_recipe_summary.csv"))
    p.add_argument("--fd_06j_nominal_csv", type=str,
                   default=str(V3_DIR / "outputs" / "labels"
                               / "stage06J_mode_b_fd_sanity.csv"))
    p.add_argument("--fd_06j_mc_csv", type=str,
                   default=str(V3_DIR / "outputs" / "labels"
                               / "stage06J_mode_b_fd_mc_optional.csv"))
    p.add_argument("--fd_06j_b_top10_csv", type=str,
                   default=str(V3_DIR / "outputs" / "labels"
                               / "stage06J_B_fd_top10_mc.csv"))
    p.add_argument("--fd_06j_b_repmc_csv", type=str,
                   default=str(V3_DIR / "outputs" / "labels"
                               / "stage06J_B_fd_representative_mc.csv"))
    p.add_argument("--fd_06j_b_nom_csv", type=str,
                   default=str(V3_DIR / "outputs" / "labels"
                               / "stage06J_B_fd_top100_nominal.csv"))
    p.add_argument("--summary_06h_json", type=str,
                   default=str(V3_DIR / "outputs" / "logs"
                               / "stage06H_fd_verification_summary.json"))
    p.add_argument("--fp_06l_json", type=str,
                   default=str(V3_DIR / "outputs" / "logs"
                               / "stage06L_false_pass_reduction.json"))
    p.add_argument("--metrics_06p_json", type=str,
                   default=str(V3_DIR / "outputs" / "logs"
                               / "stage06P_model_metrics.json"))
    p.add_argument("--clf_06l", type=str,
                   default=str(V3_DIR / "outputs" / "models" / "stage06L_classifier.joblib"))
    p.add_argument("--reg_06l", type=str,
                   default=str(V3_DIR / "outputs" / "models" / "stage06L_regressor.joblib"))
    p.add_argument("--aux_06l", type=str,
                   default=str(V3_DIR / "outputs" / "models"
                               / "stage06L_aux_cd_fixed_regressor.joblib"))
    p.add_argument("--strict_06l", type=str,
                   default=str(V3_DIR / "outputs" / "models"
                               / "stage06L_strict_score_regressor.joblib"))
    p.add_argument("--clf_06p", type=str,
                   default=str(V3_DIR / "outputs" / "models" / "stage06P_classifier.joblib"))
    p.add_argument("--reg_06p", type=str,
                   default=str(V3_DIR / "outputs" / "models" / "stage06P_regressor.joblib"))
    p.add_argument("--aux_06p", type=str,
                   default=str(V3_DIR / "outputs" / "models"
                               / "stage06P_aux_cd_fixed_regressor.joblib"))
    p.add_argument("--strict_06p", type=str,
                   default=str(V3_DIR / "outputs" / "models"
                               / "stage06P_strict_score_regressor.joblib"))
    p.add_argument("--m_06m_csv", type=str,
                   default=str(V3_DIR / "outputs" / "yield_optimization"
                               / "stage06M_time_deep_mc.csv"))
    p.add_argument("--m_06m_b_csv", type=str,
                   default=str(V3_DIR / "outputs" / "yield_optimization"
                               / "stage06M_B_J1453_time_deep_mc.csv"))
    p.add_argument("--n_var_eval", type=int, default=200)
    p.add_argument("--seed_eval",  type=int, default=8585)
    args = p.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    space = CandidateSpace.from_yaml(cfg["candidate_space_yaml"])
    var_spec = VariationSpec.from_yaml_dict(cfg["process_variation"])
    score_cfg = YieldScoreConfig.from_yaml_dict(cfg["yield_score"])

    strict_yaml = yaml.safe_load(Path(args.strict_cfg_yaml).read_text())
    cd_tol  = float(strict_yaml["thresholds"]["cd_tol_nm"])
    ler_cap = float(strict_yaml["thresholds"]["ler_cap_nm"])
    strict_cfg = StrictScoreConfig(cd_tol_nm=cd_tol, ler_cap_nm=ler_cap)

    # Load 06L + 06P joblibs.
    clf_l, _ = load_model(args.clf_06l)
    reg_l, _ = load_model(args.reg_06l)
    aux_l, _ = load_model(args.aux_06l)
    sr_l,  _ = load_model(args.strict_06l)
    clf_p, _ = load_model(args.clf_06p)
    reg_p, _ = load_model(args.reg_06p)
    aux_p, _ = load_model(args.aux_06p)
    sr_p,  _ = load_model(args.strict_06p)

    # Read inputs.
    rows_06g = read_labels_csv(args.top_06g_csv)
    _coerce(rows_06g, ["rank_strict", "strict_score", "yield_score"] + FEATURE_KEYS)
    rows_06j = read_labels_csv(args.top_06j_csv)
    _coerce(rows_06j, ["strict_score", "yield_score"] + FEATURE_KEYS)

    rows_06j_summary = read_labels_csv(args.summary_06j_csv)
    _coerce(rows_06j_summary, ["strict_score", "yield_score"] + FEATURE_KEYS)

    fd_06j_nominal = read_labels_csv(args.fd_06j_nominal_csv) \
        if Path(args.fd_06j_nominal_csv).exists() else []
    _coerce(fd_06j_nominal, ["CD_final_nm", "CD_locked_nm",
                                "LER_CD_locked_nm", "area_frac",
                                "P_line_margin", "strict_score_surrogate"])
    fd_06j_mc = read_labels_csv(args.fd_06j_mc_csv) \
        if Path(args.fd_06j_mc_csv).exists() else []
    _coerce(fd_06j_mc, ["CD_final_nm", "CD_locked_nm", "LER_CD_locked_nm",
                          "area_frac", "P_line_margin", "variation_idx"])
    fd_06j_b_top10 = read_labels_csv(args.fd_06j_b_top10_csv) \
        if Path(args.fd_06j_b_top10_csv).exists() else []
    _coerce(fd_06j_b_top10, ["CD_final_nm", "CD_locked_nm",
                                  "LER_CD_locked_nm", "area_frac",
                                  "P_line_margin"])
    fd_06j_b_repmc = read_labels_csv(args.fd_06j_b_repmc_csv) \
        if Path(args.fd_06j_b_repmc_csv).exists() else []
    _coerce(fd_06j_b_repmc, ["CD_final_nm", "CD_locked_nm",
                                  "LER_CD_locked_nm", "area_frac",
                                  "P_line_margin"])
    fd_06j_b_nom = read_labels_csv(args.fd_06j_b_nom_csv) \
        if Path(args.fd_06j_b_nom_csv).exists() else []
    _coerce(fd_06j_b_nom, ["CD_final_nm", "CD_locked_nm",
                                "LER_CD_locked_nm", "area_frac",
                                "P_line_margin"])
    summary_06h = json.loads(Path(args.summary_06h_json).read_text())
    fp_06l_payload = json.loads(Path(args.fp_06l_json).read_text())

    # ----- Re-score 06G top-100 (Mode A) and 06J top-100 (Mode B) -----
    print(f"  re-scoring 06G top-{len(rows_06g)} with 06L stack ...")
    rescored_06g_l = rescore_candidates(
        rows_06g, clf_l, reg_l, aux_l, sr_l,
        var_spec, args.n_var_eval, space, score_cfg, strict_cfg,
        seed=args.seed_eval, tag="06l")
    print(f"  re-scoring 06G top-{len(rows_06g)} with 06P stack ...")
    rescored_06g_p = rescore_candidates(
        rows_06g, clf_p, reg_p, aux_p, sr_p,
        var_spec, args.n_var_eval, space, score_cfg, strict_cfg,
        seed=args.seed_eval, tag="06p")
    for r in rescored_06g_l + rescored_06g_p:
        r["mode"] = "mode_a"

    print(f"  re-scoring 06J top-{len(rows_06j)} with 06L stack ...")
    rescored_06j_l = rescore_candidates(
        rows_06j, clf_l, reg_l, aux_l, sr_l,
        var_spec, args.n_var_eval, space, score_cfg, strict_cfg,
        seed=args.seed_eval + 1, tag="06l")
    print(f"  re-scoring 06J top-{len(rows_06j)} with 06P stack ...")
    rescored_06j_p = rescore_candidates(
        rows_06j, clf_p, reg_p, aux_p, sr_p,
        var_spec, args.n_var_eval, space, score_cfg, strict_cfg,
        seed=args.seed_eval + 1, tag="06p")
    for r in rescored_06j_l + rescored_06j_p:
        r["mode"] = "mode_b"

    # ----- FD MC truth lookup -----
    fd_06j_b_mc_pool = fd_06j_b_top10 + fd_06j_b_repmc
    fd_truth = build_fd_truth(
        summary_06h=summary_06h,
        fd_06j_mc=fd_06j_mc,
        fd_06j_b_mc=fd_06j_b_mc_pool,
        cd_tol=cd_tol, ler_cap=ler_cap, strict_yaml=strict_yaml,
    )

    by_id_06g_p = {r["recipe_id"]: r for r in rescored_06g_p}
    by_id_06j_p = {r["recipe_id"]: r for r in rescored_06j_p}
    by_id_06g_l = {r["recipe_id"]: r for r in rescored_06g_l}
    by_id_06j_l = {r["recipe_id"]: r for r in rescored_06j_l}

    pairs_06p: list[dict] = []
    pairs_06l: list[dict] = []
    for rid, truth in fd_truth.items():
        rs_p = by_id_06g_p.get(rid) or by_id_06j_p.get(rid)
        rs_l = by_id_06g_l.get(rid) or by_id_06j_l.get(rid)
        if rs_p is not None:
            pairs_06p.append({
                "recipe_id":             rid,
                "mode":                   truth["mode"],
                "pred_strict_06p":        float(rs_p["strict_score_06p_direct_mean"]),
                "fd_mc_strict_pass_prob": truth["fd_mc_strict_pass_prob"],
                "fd_mc_n":                truth["fd_mc_n"],
            })
        if rs_l is not None:
            pairs_06l.append({
                "recipe_id":             rid,
                "mode":                   truth["mode"],
                "pred_strict_06l":        float(rs_l["strict_score_06l_direct_mean"]),
                "fd_mc_strict_pass_prob": truth["fd_mc_strict_pass_prob"],
            })

    def _spearman_pairs(pairs: list[dict], pred_key: str, mode: str | None) -> tuple[float, int]:
        pp = [p for p in pairs if (mode is None or p["mode"] == mode)]
        if len(pp) < 3:
            return float("nan"), len(pp)
        a = np.array([p[pred_key] for p in pp])
        b = np.array([p["fd_mc_strict_pass_prob"] for p in pp])
        return spearman(a, b), len(pp)

    rho_06p_overall, n_overall_p = _spearman_pairs(pairs_06p, "pred_strict_06p", None)
    rho_06p_mode_a,  n_a_p       = _spearman_pairs(pairs_06p, "pred_strict_06p", "mode_a")
    rho_06p_mode_b,  n_b_p       = _spearman_pairs(pairs_06p, "pred_strict_06p", "mode_b")
    rho_06l_overall, n_overall_l = _spearman_pairs(pairs_06l, "pred_strict_06l", None)
    rho_06l_mode_a,  n_a_l       = _spearman_pairs(pairs_06l, "pred_strict_06l", "mode_a")
    rho_06l_mode_b,  n_b_l       = _spearman_pairs(pairs_06l, "pred_strict_06l", "mode_b")

    # ----- Top-k overlap (06P top-10 vs FD top-10) -----
    fd_pairs_b_sorted = sorted(
        [p for p in pairs_06p if p["mode"] == "mode_b"],
        key=lambda p: -p["fd_mc_strict_pass_prob"],
    )
    pred_b_sorted = sorted(
        [p for p in pairs_06p if p["mode"] == "mode_b"],
        key=lambda p: -p["pred_strict_06p"],
    )
    top10_overlap_b = topk_overlap(
        [p["recipe_id"] for p in fd_pairs_b_sorted],
        [p["recipe_id"] for p in pred_b_sorted],
        k=10,
    )

    # ----- 06L vs 06P Mode B ranking comparison (top-100) -----
    rank_06j = {r["recipe_id"]: i + 1
                  for i, r in enumerate(sorted(
                      rows_06j, key=lambda x: -float(x.get("strict_score", 0.0))))}
    rank_06l_b = {r["recipe_id"]: i + 1
                    for i, r in enumerate(sorted(
                        rescored_06j_l, key=lambda x: -float(x["strict_score_06l_direct_mean"])))}
    rank_06p_b = {r["recipe_id"]: i + 1
                    for i, r in enumerate(sorted(
                        rescored_06j_p, key=lambda x: -float(x["strict_score_06p_direct_mean"])))}
    common_ids = sorted(set(rank_06j) & set(rank_06l_b) & set(rank_06p_b))
    rj = np.array([rank_06j[i]   for i in common_ids])
    rl = np.array([rank_06l_b[i] for i in common_ids])
    rp = np.array([rank_06p_b[i] for i in common_ids])
    rho_l_vs_p = spearman(rl.astype(float), rp.astype(float)) \
                   if len(common_ids) >= 3 else float("nan")
    rho_j_vs_p = spearman(rj.astype(float), rp.astype(float)) \
                   if len(common_ids) >= 3 else float("nan")
    top10_jvp = topk_overlap(
        [r["recipe_id"] for r in rows_06j[:10]],
        [r for r, _ in sorted(rank_06p_b.items(), key=lambda kv: kv[1])][:10],
        k=10,
    )
    top10_lvp = topk_overlap(
        [r for r, _ in sorted(rank_06l_b.items(), key=lambda kv: kv[1])][:10],
        [r for r, _ in sorted(rank_06p_b.items(), key=lambda kv: kv[1])][:10],
        k=10,
    )

    # J_1453 promotion under 06P top-10 in Mode B?
    j1453_in_06p_top10_b = bool(any(
        r["recipe_id"] == "J_1453"
        for r in sorted(rescored_06j_p,
                          key=lambda r: -float(r["strict_score_06p_direct_mean"]))[:10]
    ))
    g4867_in_06p_top10_a = bool(any(
        r["recipe_id"] == "G_4867"
        for r in sorted(rescored_06g_p,
                          key=lambda r: -float(r["strict_score_06p_direct_mean"]))[:10]
    ))

    # ----- Time-window learning check -----
    rows_06m = read_labels_csv(args.m_06m_csv) if Path(args.m_06m_csv).exists() else []
    rows_06m_b = read_labels_csv(args.m_06m_b_csv) if Path(args.m_06m_b_csv).exists() else []
    twin = time_window_learning_check(
        rows_06m, rows_06m_b,
        clf=clf_p, reg=reg_p, aux=aux_p, strict_reg=sr_p,
        cd_tol=cd_tol, ler_cap=ler_cap, strict_yaml=strict_yaml,
    )

    # ----- False-PASS demotion analysis -----
    by_id_06j_summary = {r["recipe_id"]: r for r in rows_06j_summary}
    candidates_for_fp: list[dict] = []
    for r in fd_06j_nominal:
        rid = str(r.get("source_recipe_id", ""))
        sur = by_id_06j_summary.get(rid)
        if sur is not None:
            candidates_for_fp.append(sur)
    if candidates_for_fp:
        rescored_fp_l = rescore_candidates(
            candidates_for_fp, clf_l, reg_l, aux_l, sr_l,
            var_spec, args.n_var_eval, space, score_cfg, strict_cfg,
            seed=args.seed_eval + 7, tag="06l")
        rescored_fp_p = rescore_candidates(
            candidates_for_fp, clf_p, reg_p, aux_p, sr_p,
            var_spec, args.n_var_eval, space, score_cfg, strict_cfg,
            seed=args.seed_eval + 7, tag="06p")
        pool_06j = sorted(candidates_for_fp,
                            key=lambda r: -_safe_float(r.get("strict_score")))
        rank_06j_fp = {r["recipe_id"]: i + 1 for i, r in enumerate(pool_06j)}
        pool_06l = sorted(rescored_fp_l,
                            key=lambda r: -float(r["strict_score_06l_direct_mean"]))
        rank_06l_fp = {r["recipe_id"]: i + 1 for i, r in enumerate(pool_06l)}
        pool_06p = sorted(rescored_fp_p,
                            key=lambda r: -float(r["strict_score_06p_direct_mean"]))
        rank_06p_fp = {r["recipe_id"]: i + 1 for i, r in enumerate(pool_06p)}

        demos: list[dict] = []
        for r in fd_06j_nominal:
            rid = str(r.get("source_recipe_id", ""))
            label = str(r.get("label", ""))
            if label not in HARD_FAIL_LABELS:
                continue
            if rid not in rank_06j_fp or rid not in rank_06l_fp or rid not in rank_06p_fp:
                continue
            sl = next(x for x in rescored_fp_l if x["recipe_id"] == rid)
            sp = next(x for x in rescored_fp_p if x["recipe_id"] == rid)
            demos.append({
                "recipe_id":          rid,
                "fd_label":           label,
                "rank_06j":           int(rank_06j_fp[rid]),
                "rank_06l":           int(rank_06l_fp[rid]),
                "rank_06p":           int(rank_06p_fp[rid]),
                "delta_rank_06p_vs_06l": int(rank_06p_fp[rid] - rank_06l_fp[rid]),
                "delta_rank_06p_vs_06j": int(rank_06p_fp[rid] - rank_06j_fp[rid]),
                "strict_score_06j_surrogate":
                    float(by_id_06j_summary[rid].get("strict_score", float("nan"))),
                "strict_score_06l_direct":
                    float(sl["strict_score_06l_direct_mean"]),
                "strict_score_06p_direct":
                    float(sp["strict_score_06p_direct_mean"]),
            })
    else:
        demos = []

    # ----- Outputs -----
    yopt_dir = V3_DIR / "outputs" / "yield_optimization"
    logs_dir = V3_DIR / "outputs" / "logs"
    fig_dir  = V3_DIR / "outputs" / "figures" / "06_yield_optimization"
    fig_dir.mkdir(parents=True, exist_ok=True)

    # Mode B re-scored candidates table (06L + 06P side-by-side).
    side_cols = ["recipe_id", "mode",
                  "strict_score_06l_eval", "strict_score_06l_direct_mean",
                  "strict_score_06p_eval", "strict_score_06p_direct_mean",
                  "yield_score_06l", "yield_score_06p",
                  "p_robust_valid_06l", "p_robust_valid_06p",
                  "mean_cd_fixed_06l", "mean_cd_fixed_06p",
                  "mean_ler_locked_06l", "mean_ler_locked_06p",
                  "mean_p_line_margin_06l", "mean_p_line_margin_06p",
                  "rank_06j", "rank_06l", "rank_06p"] + FEATURE_KEYS
    side_rows = []
    for rid in common_ids:
        rL = by_id_06j_l.get(rid)
        rP = by_id_06j_p.get(rid)
        side_rows.append({
            "recipe_id": rid, "mode": "mode_b",
            **{k: rL.get(k) for k in
                ["strict_score_06l_eval", "strict_score_06l_direct_mean",
                 "yield_score_06l", "p_robust_valid_06l",
                 "mean_cd_fixed_06l", "mean_ler_locked_06l",
                 "mean_p_line_margin_06l"]},
            **{k: rP.get(k) for k in
                ["strict_score_06p_eval", "strict_score_06p_direct_mean",
                 "yield_score_06p", "p_robust_valid_06p",
                 "mean_cd_fixed_06p", "mean_ler_locked_06p",
                 "mean_p_line_margin_06p"]},
            **{k: rP.get(k) for k in FEATURE_KEYS},
            "rank_06j": rank_06j[rid],
            "rank_06l": rank_06l_b[rid],
            "rank_06p": rank_06p_b[rid],
        })
    with (yopt_dir / "stage06P_mode_b_rescored_candidates.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=side_cols, extrasaction="ignore")
        w.writeheader()
        for r in side_rows:
            w.writerow(r)

    # J_1453 vs G_4867 prediction-vs-FD time-window check.
    twin_cols = list(twin["per_offset"][0].keys()) if twin["per_offset"] else []
    with (yopt_dir / "stage06P_J1453_vs_G4867_prediction_check.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=twin_cols, extrasaction="ignore")
        w.writeheader()
        for r in twin["per_offset"]:
            w.writerow(r)

    # False-PASS demotions table.
    if demos:
        cols_d = ["recipe_id", "fd_label", "rank_06j", "rank_06l", "rank_06p",
                  "delta_rank_06p_vs_06l", "delta_rank_06p_vs_06j",
                  "strict_score_06j_surrogate",
                  "strict_score_06l_direct", "strict_score_06p_direct"]
        with (yopt_dir / "stage06P_false_pass_demotions.csv").open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=cols_d, extrasaction="ignore")
            w.writeheader()
            for r in demos:
                w.writerow(r)
    else:
        with (yopt_dir / "stage06P_false_pass_demotions.csv").open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["recipe_id", "fd_label",
                                                  "rank_06j", "rank_06l",
                                                  "rank_06p"])
            w.writeheader()

    # Manifest update.
    manifest_update = {
        "stage": "06P",
        "policy": {
            "v2_OP_frozen": True,
            "published_data_loaded": False,
            "external_calibration": "none",
        },
        "preferred_surrogate": "stage06P",
        "preferred_recipes": {
            "mode_a": {
                "primary": "G_4867",
                "selection_basis": "06I FD MC + 06H stability; 06P preserves",
            },
            "mode_b": {
                "primary": "J_1453",
                "selection_basis": (
                    "06J Mode B strict_score #1; 06J-B FD MC strict_pass = 0.747; "
                    "06M-B time-window wider than G_4867; 06P direct strict_score "
                    f"head ranks J_1453 in Mode B top-10 = {j1453_in_06p_top10_b}."
                ),
                "j1453_in_06p_top10_b": j1453_in_06p_top10_b,
            },
        },
        "ranking_notes": {
            "ranking_strict_score_spearman_overall_06p_vs_fd": rho_06p_overall,
            "ranking_strict_score_spearman_mode_b_06p_vs_fd": rho_06p_mode_b,
            "ranking_strict_score_spearman_mode_a_06p_vs_fd": rho_06p_mode_a,
            "ranking_strict_score_spearman_mode_b_06l_vs_fd": rho_06l_mode_b,
            "improved_over_06l_mode_b": (
                bool(np.isfinite(rho_06p_mode_b) and np.isfinite(rho_06l_mode_b)
                       and rho_06p_mode_b >= rho_06l_mode_b - 0.01)
            ),
        },
        "false_pass": {
            "n_input": int(fp_06l_payload.get("false_pass_06j_input_count", 0)),
            "n_demoted_06p_vs_06l":
                int(sum(1 for d in demos
                          if d["delta_rank_06p_vs_06l"] > 0)),
            "n_demoted_06p_vs_06j":
                int(sum(1 for d in demos
                          if d["delta_rank_06p_vs_06j"] > 0)),
        },
        "time_window_learning": twin["summary"],
        "models": {
            "classifier":      str(args.clf_06p),
            "regressor4":      str(args.reg_06p),
            "aux_cd_fixed":    str(args.aux_06p),
            "strict_score":    str(args.strict_06p),
        },
        "datasets": {
            "training_csv":    str(V3_DIR / "outputs" / "labels"
                                     / "stage06P_training_dataset.csv"),
            "recipe_manifest": str(V3_DIR / "outputs" / "yield_optimization"
                                     / "stage06P_recipe_manifest.yaml"),
        },
    }
    (yopt_dir / "stage06P_manifest_update.yaml").write_text(
        yaml.safe_dump(manifest_update, sort_keys=False))

    # Logs.
    (logs_dir / "stage06P_mode_b_ranking_comparison.json").write_text(
        json.dumps({
            "stage": "06P",
            "policy": cfg["policy"],
            "n_pairs_06p_overall": n_overall_p,
            "n_pairs_06p_mode_a":  n_a_p,
            "n_pairs_06p_mode_b":  n_b_p,
            "n_pairs_06l_overall": n_overall_l,
            "n_pairs_06l_mode_a":  n_a_l,
            "n_pairs_06l_mode_b":  n_b_l,
            "spearman_06p_overall": rho_06p_overall,
            "spearman_06p_mode_a":  rho_06p_mode_a,
            "spearman_06p_mode_b":  rho_06p_mode_b,
            "spearman_06l_overall": rho_06l_overall,
            "spearman_06l_mode_a":  rho_06l_mode_a,
            "spearman_06l_mode_b":  rho_06l_mode_b,
            "delta_spearman_overall_06p_minus_06l":
                ((rho_06p_overall - rho_06l_overall)
                 if np.isfinite(rho_06p_overall) and np.isfinite(rho_06l_overall)
                 else None),
            "delta_spearman_mode_b_06p_minus_06l":
                ((rho_06p_mode_b - rho_06l_mode_b)
                 if np.isfinite(rho_06p_mode_b) and np.isfinite(rho_06l_mode_b)
                 else None),
            "rank_spearman_06l_vs_06p_mode_b": rho_l_vs_p,
            "rank_spearman_06j_vs_06p_mode_b": rho_j_vs_p,
            "top10_overlap_mode_b_06l_vs_06p": top10_lvp,
            "top10_overlap_mode_b_06j_vs_06p": top10_jvp,
            "top10_overlap_mode_b_pred_vs_fd": top10_overlap_b,
            "j1453_in_06p_top10_mode_b":      j1453_in_06p_top10_b,
            "g4867_in_06p_top10_mode_a":      g4867_in_06p_top10_a,
        }, indent=2, default=float))

    (logs_dir / "stage06P_false_pass_reduction.json").write_text(
        json.dumps({
            "stage": "06P",
            "policy": cfg["policy"],
            "false_pass_06j_input_count":
                int(fp_06l_payload.get("false_pass_06j_input_count", 0)),
            "false_pass_06l_demoted_count":
                int(fp_06l_payload.get("false_pass_06j_demoted_count", 0)),
            "false_pass_06p_demoted_vs_06l_count":
                int(sum(1 for d in demos if d["delta_rank_06p_vs_06l"] > 0)),
            "false_pass_06p_demoted_vs_06j_count":
                int(sum(1 for d in demos if d["delta_rank_06p_vs_06j"] > 0)),
            "false_pass_demotions": demos,
        }, indent=2, default=float))

    (logs_dir / "stage06P_time_window_learning.json").write_text(
        json.dumps({
            "stage": "06P",
            "policy": cfg["policy"],
            "per_offset": twin["per_offset"],
            "summary":    twin["summary"],
            "j1453_wider_window_predicted": bool(
                twin["summary"]["J_minus_G_strict_pass"]["mean_pred_advantage"]
                > 0
            ),
            "j1453_wider_window_fd": bool(
                twin["summary"]["J_minus_G_strict_pass"]["mean_fd_advantage"]
                > 0
            ),
        }, indent=2, default=float))

    # ----- Figures -----
    plot_strict_score_pred_vs_fd(
        pairs_06p, fig_dir / "stage06P_strict_score_pred_vs_fd.png",
    )
    plot_rank_before_after(
        rescored_06j_l, rescored_06j_p,
        highlight_ids={"J_1453": "#d62728", "J_2261": "#d62728",
                          "J_4793": "#d62728", "J_3413": "#d62728",
                          "J_4164": "#d62728"},
        out_path=fig_dir / "stage06P_mode_b_ranking_before_after.png",
        title="Stage 06P -- Mode B (06J top-100) ranking 06L -> 06P",
    )
    plot_J1453_vs_G4867_time_window(
        twin["per_offset"],
        fig_dir / "stage06P_J1453_vs_G4867_predicted_time_window.png",
    )
    plot_false_pass_demotions(
        demos, fig_dir / "stage06P_false_pass_demotions.png",
    )
    metrics_06p = json.loads(Path(args.metrics_06p_json).read_text())
    plot_feature_importance(
        metrics_06p, fig_dir / "stage06P_feature_importance_strict_score.png",
    )
    plot_mode_a_vs_mode_b_distribution(
        rescored_06g_p, rescored_06j_p,
        fig_dir / "stage06P_mode_a_vs_mode_b_score_distribution.png",
    )

    # ----- Console summary -----
    print(f"\nStage 06P -- analysis summary")
    print(f"  Spearman strict_score_06p vs FD MC strict_pass_prob:")
    print(f"    overall (n={n_overall_p}):     {rho_06p_overall:+.3f}  "
          f"(06L: {rho_06l_overall:+.3f})")
    print(f"    Mode A  (n={n_a_p}):     {rho_06p_mode_a:+.3f}  "
          f"(06L: {rho_06l_mode_a:+.3f})")
    print(f"    Mode B  (n={n_b_p}):     {rho_06p_mode_b:+.3f}  "
          f"(06L: {rho_06l_mode_b:+.3f})")
    print(f"  Mode B rank: 06L vs 06P Spearman = {rho_l_vs_p:+.3f}, "
          f"06J vs 06P = {rho_j_vs_p:+.3f}")
    print(f"  Top-10 overlap Mode B: 06L<->06P {top10_lvp}, "
          f"06J<->06P {top10_jvp}, pred<->FD {top10_overlap_b}")
    print(f"  J_1453 in 06P Mode B top-10:    {j1453_in_06p_top10_b}")
    print(f"  G_4867 in 06P Mode A top-10:    {g4867_in_06p_top10_a}")
    print(f"  06J nominal false-PASS rows:     "
          f"{fp_06l_payload.get('false_pass_06j_input_count', 0)}")
    print(f"    of those, demoted 06P vs 06L:  "
          f"{sum(1 for d in demos if d['delta_rank_06p_vs_06l'] > 0)}")
    print(f"    of those, demoted 06P vs 06J:  "
          f"{sum(1 for d in demos if d['delta_rank_06p_vs_06j'] > 0)}")
    if demos:
        for d in demos:
            print(f"      {d['recipe_id']}  fd_label={d['fd_label']:<22}  "
                  f"rank: 06J #{d['rank_06j']} -> 06L #{d['rank_06l']} -> "
                  f"06P #{d['rank_06p']}")
    print(f"  Time-window learning (Spearman pred vs FD):")
    s_g = twin["summary"]["G_4867"]
    s_j = twin["summary"]["J_1453"]
    print(f"    G_4867: strict_pass={s_g['spearman_strict_pass']:+.3f}  "
          f"robust_valid={s_g['spearman_robust_valid']:+.3f}  "
          f"defect={s_g['spearman_defect']:+.3f}  "
          f"strict_score={s_g['spearman_mean_strict_score']:+.3f}")
    print(f"    J_1453: strict_pass={s_j['spearman_strict_pass']:+.3f}  "
          f"robust_valid={s_j['spearman_robust_valid']:+.3f}  "
          f"defect={s_j['spearman_defect']:+.3f}  "
          f"strict_score={s_j['spearman_mean_strict_score']:+.3f}")
    rel = twin["summary"]["J_minus_G_strict_pass"]
    print(f"    J - G strict_pass advantage: pred={rel['mean_pred_advantage']:+.3f}  "
          f"FD={rel['mean_fd_advantage']:+.3f}  "
          f"Spearman={rel['spearman_pred_vs_fd']:+.3f}")
    print(f"  outputs ->")
    print(f"    {yopt_dir / 'stage06P_mode_b_rescored_candidates.csv'}")
    print(f"    {yopt_dir / 'stage06P_J1453_vs_G4867_prediction_check.csv'}")
    print(f"    {yopt_dir / 'stage06P_false_pass_demotions.csv'}")
    print(f"    {yopt_dir / 'stage06P_manifest_update.yaml'}")
    print(f"    {logs_dir / 'stage06P_mode_b_ranking_comparison.json'}")
    print(f"    {logs_dir / 'stage06P_false_pass_reduction.json'}")
    print(f"    {logs_dir / 'stage06P_time_window_learning.json'}")
    print(f"    {fig_dir} / stage06P_*.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

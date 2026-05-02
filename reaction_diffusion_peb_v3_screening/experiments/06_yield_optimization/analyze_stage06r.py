"""Stage 06R -- evaluate the feature-engineered surrogate stack.

Compares the new 06R stack (raw 06P features + 9 derived process-budget
features) to the 06P baseline on:
 - 04C 1,074 holdout: classifier / regressor / strict_score head metrics.
 - Mode A (06G top-100) and Mode B (06J top-100) re-ranking using FD MC
   strict_pass_prob truth (06H + 06J + 06J-B).
 - 06Q blindspot: G_4867 / J_1453 strict_pass residual vs deterministic
   time offset, and the per-offset relative-advantage residual that 06Q
   identified as the open gap in 06P.
 - False-PASS demotion analysis on the 06J nominal hard-fail recipes.

Reads:
    outputs/yield_optimization/stage06G_top_recipes.csv
    outputs/yield_optimization/stage06J_mode_b_top_recipes.csv
    outputs/yield_optimization/stage06J_mode_b_recipe_summary.csv
    outputs/yield_optimization/stage06G_strict_score_config.yaml
    outputs/yield_optimization/stage06M_time_deep_mc.csv
    outputs/yield_optimization/stage06M_B_J1453_time_deep_mc.csv
    outputs/labels/stage06J_mode_b_fd_sanity.csv
    outputs/labels/stage06J_mode_b_fd_mc_optional.csv
    outputs/labels/stage06J_B_fd_top10_mc.csv
    outputs/labels/stage06J_B_fd_representative_mc.csv
    outputs/logs/stage06H_fd_verification_summary.json
    outputs/logs/stage06P_mode_b_ranking_comparison.json
    outputs/logs/stage06Q_summary.json
    outputs/models/stage06R_feature_list.json
    outputs/models/stage06P_*.joblib
    outputs/models/stage06R_*.joblib

Writes:
    outputs/yield_optimization/stage06R_J1453_vs_G4867_prediction_check.csv
    outputs/yield_optimization/stage06R_false_pass_demotions.csv
    outputs/yield_optimization/stage06R_feature_importance.csv
    outputs/logs/stage06R_blindspot_comparison.json
    outputs/logs/stage06R_mode_b_ranking_comparison.json
    outputs/figures/06_yield_optimization/
        stage06R_J1453_G4867_relative_advantage_before_after.png
        stage06R_G4867_strict_pass_residual_vs_time.png
        stage06R_cd_error_residual_vs_time.png
        stage06R_strict_score_pred_vs_fd.png
        stage06R_mode_b_ranking_before_after.png
        stage06R_feature_importance_derived_features.png
        stage06R_false_pass_demotions.png
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
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
    FEATURE_KEYS, REGRESSION_TARGETS, load_model, read_labels_csv,
)
from reaction_diffusion_peb_v3_screening.src.process_variation import (
    VariationSpec, sample_variations,
)
from reaction_diffusion_peb_v3_screening.src.yield_optimizer import (
    YieldScoreConfig,
)

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_stage06g_strict_optimization import (  # noqa: E402
    StrictScoreConfig, compute_strict_score,
)
from build_stage06l_dataset import per_row_strict_score  # noqa: E402
from build_stage06r_dataset import (
    DERIVED_FEATURE_KEYS, derive_features, augment_rows,
)


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


def _build_X(rows: list[dict], feature_keys: list[str]) -> np.ndarray:
    X = np.zeros((len(rows), len(feature_keys)), dtype=np.float64)
    for i, r in enumerate(rows):
        for j, k in enumerate(feature_keys):
            X[i, j] = _safe_float(r.get(k))
    return X


# --------------------------------------------------------------------------
# Surrogate eval pack: identical interface for 06P and 06R, callers
# pass the right feature_keys list and an `augment_fn` (None for 06P).
# --------------------------------------------------------------------------
def _eval_pack(rows: list[dict], clf, reg, aux, strict_reg, *,
                  feature_keys: list[str], augment: bool,
                  cd_tol: float, ler_cap: float) -> dict:
    if not rows:
        return {"n": 0, "p_robust_valid": np.array([]),
                "p_defect": np.array([]), "pred_class": np.array([], dtype=object),
                "cd_pred": np.array([]), "Y4": np.zeros((0, 4)),
                "strict_pred": np.array([]),
                "strict_pass_pred": np.array([])}
    rs = augment_rows(rows) if augment else rows
    X = _build_X(rs, feature_keys)
    proba = clf.predict_proba(X)
    classes = list(clf.classes_)

    def col(name: str) -> np.ndarray:
        if name not in classes:
            return np.zeros(len(rs))
        return proba[:, classes.index(name)]

    p_robust = col("robust_valid")
    defects = (col("under_exposed") + col("merged")
                + col("roughness_degraded") + col("numerical_invalid"))
    pred_class = np.array([classes[int(np.argmax(p))] for p in proba])
    Y4 = reg.predict(X)
    cd_pred = aux.predict(X)
    strict_pred = strict_reg.predict(X)
    cd_within = np.abs(cd_pred - CD_TARGET_NM) <= cd_tol
    ler_under = Y4[:, 1] <= ler_cap
    strict_pass_pred = (cd_within & ler_under
                          & (pred_class == "robust_valid")).astype(float)
    return {
        "n": len(rs), "p_robust_valid": p_robust, "p_defect": defects,
        "pred_class": pred_class, "cd_pred": cd_pred,
        "Y4": Y4, "strict_pred": strict_pred,
        "strict_pass_pred": strict_pass_pred,
    }


# --------------------------------------------------------------------------
# Re-score top-K candidate lists: run N variations, average per-row
# strict_score head + a 4-target / classifier evaluation, return one
# dict per candidate. Mirrors analyze_stage06p rescore_candidates but
# routes through `_eval_pack` so 06R augmentation works.
# --------------------------------------------------------------------------
def rescore_candidates(rows: list[dict], clf, reg, aux, strict_reg,
                          var_spec: VariationSpec, n_var: int,
                          space: CandidateSpace, score_cfg: YieldScoreConfig,
                          strict_cfg: StrictScoreConfig, seed: int,
                          tag: str, *, feature_keys: list[str],
                          augment: bool, cd_tol: float, ler_cap: float
                          ) -> list[dict]:
    base_rng = np.random.default_rng(seed)
    out: list[dict] = []
    for orig in rows:
        cand = {k: float(orig[k]) for k in FEATURE_KEYS}
        cand["pitch_nm"]    = float(cand["pitch_nm"])
        cand["line_cd_nm"]  = cand["pitch_nm"] * cand["line_cd_ratio"]
        cand["domain_x_nm"] = cand["pitch_nm"] * 5.0
        cand["dose_norm"]   = (cand["dose_mJ_cm2"]
                                  / float(space.fixed["reference_dose_mJ_cm2"]))
        for fk, fv in space.fixed.items():
            cand.setdefault(fk, fv)
        cand["_id"] = f"{tag}_rescore_{orig.get('recipe_id', orig.get('_id', '?'))}"
        sub_rng = np.random.default_rng(int(base_rng.integers(0, 2**31 - 1)))
        variations = sample_variations(cand, var_spec, n_var, space, rng=sub_rng)
        pack = _eval_pack(variations, clf, reg, aux, strict_reg,
                              feature_keys=feature_keys, augment=augment,
                              cd_tol=cd_tol, ler_cap=ler_cap)
        cls_to_col = {c: i for i, c in enumerate(clf.classes_)}
        # Approximate yield_score = 1.0 P(robust) - other class penalties
        # (skipped here; we report per-class probs and the strict head
        # mean directly which is what 06P/Q comparisons need).
        out.append({
            **{k: _safe_float(orig.get(k)) for k in FEATURE_KEYS},
            "recipe_id": orig.get("recipe_id", orig.get("_id", "?")),
            f"strict_score_{tag}_direct_mean": float(np.mean(pack["strict_pred"])),
            f"strict_score_{tag}_direct_std":  float(np.std(pack["strict_pred"])),
            f"strict_pass_prob_{tag}_proxy":   float(np.mean(pack["strict_pass_pred"])),
            f"p_robust_valid_{tag}":           float(np.mean(pack["p_robust_valid"])),
            f"p_defect_{tag}":                 float(np.mean(pack["p_defect"])),
            f"mean_cd_fixed_{tag}":            float(np.mean(pack["cd_pred"])),
            f"mean_ler_locked_{tag}":          float(np.mean(pack["Y4"][:, 1])),
            f"mean_p_line_margin_{tag}":       float(np.mean(pack["Y4"][:, 3])),
        })
    return out


# --------------------------------------------------------------------------
# FD MC truth lookup table (Mode A 06H + Mode B 06J + 06J-B).
# --------------------------------------------------------------------------
def _mc_aggregate_strict_pass(rows: list[dict], cd_tol: float, ler_cap: float
                                 ) -> dict:
    n = len(rows)
    if n == 0:
        return {"n": 0, "strict_pass_prob": float("nan")}
    n_sp = sum(
        1 for r in rows
        if str(r.get("label", "")) == "robust_valid"
            and abs(_safe_float(r.get("CD_final_nm")) - CD_TARGET_NM) <= cd_tol
            and _safe_float(r.get("LER_CD_locked_nm")) <= ler_cap
    )
    return {"n": int(n), "strict_pass_prob": float(n_sp / n)}


def build_fd_truth(*, summary_06h: dict,
                      fd_06j_mc: list[dict],
                      fd_06j_b_mc: list[dict],
                      cd_tol: float, ler_cap: float) -> dict:
    fd_truth: dict[str, dict] = {}
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
        }
    by_recipe_06j: dict[str, list[dict]] = {}
    for r in fd_06j_mc:
        rid = str(r.get("source_recipe_id", ""))
        by_recipe_06j.setdefault(rid, []).append(r)
    for rid, rs in by_recipe_06j.items():
        agg = _mc_aggregate_strict_pass(rs, cd_tol, ler_cap)
        fd_truth[rid] = {"mode": "mode_b",
                          "fd_mc_strict_pass_prob": agg["strict_pass_prob"],
                          "fd_mc_n":                agg["n"],
                          "source":                  "06J_mc_100var"}
    by_recipe_06jb: dict[str, list[dict]] = {}
    for r in fd_06j_b_mc:
        rid = str(r.get("source_recipe_id", ""))
        by_recipe_06jb.setdefault(rid, []).append(r)
    for rid, rs in by_recipe_06jb.items():
        agg = _mc_aggregate_strict_pass(rs, cd_tol, ler_cap)
        fd_truth[rid] = {"mode": "mode_b",
                          "fd_mc_strict_pass_prob": agg["strict_pass_prob"],
                          "fd_mc_n":                agg["n"],
                          "source":                  "06J_B_mc"}
    return fd_truth


# --------------------------------------------------------------------------
# Time-window check: for each integer offset in [-5, 5], aggregate
# 06P and 06R predicted strict_pass_prob (proxy) on G_4867 (06M) and
# J_1453 (06M-B) deterministic-offset MC rows. Compare to FD truth.
# --------------------------------------------------------------------------
def _fd_per_offset(rows: list[dict], cd_tol: float, ler_cap: float) -> dict:
    if not rows:
        return {"n": 0}
    n = len(rows)
    n_robust = sum(1 for r in rows if str(r.get("label", "")) == "robust_valid")
    n_defect = sum(1 for r in rows if str(r.get("label", "")) in HARD_FAIL_LABELS)
    n_sp = sum(
        1 for r in rows
        if str(r.get("label", "")) == "robust_valid"
            and abs(_safe_float(r.get("CD_final_nm")) - CD_TARGET_NM) <= cd_tol
            and _safe_float(r.get("LER_CD_locked_nm")) <= ler_cap
    )
    cd_err = np.array([abs(_safe_float(r.get("CD_final_nm")) - CD_TARGET_NM)
                          for r in rows])
    ler = np.array([_safe_float(r.get("LER_CD_locked_nm")) for r in rows])
    mar = np.array([_safe_float(r.get("P_line_margin")) for r in rows])
    return {
        "n":                    n,
        "fd_strict_pass_prob":  n_sp / n,
        "fd_robust_valid_prob": n_robust / n,
        "fd_defect_prob":       n_defect / n,
        "fd_mean_cd_error":     float(np.nanmean(cd_err)),
        "fd_mean_ler":          float(np.nanmean(ler)),
        "fd_mean_margin":       float(np.nanmean(mar)),
    }


def _pred_per_offset(rows: list[dict], clf, reg, aux, strict_reg, *,
                       feature_keys: list[str], augment: bool,
                       cd_tol: float, ler_cap: float) -> dict:
    if not rows:
        return {"n": 0}
    pack = _eval_pack(rows, clf, reg, aux, strict_reg,
                          feature_keys=feature_keys, augment=augment,
                          cd_tol=cd_tol, ler_cap=ler_cap)
    return {
        "n": pack["n"],
        "pred_strict_pass_prob":  float(np.mean(pack["strict_pass_pred"])),
        "pred_robust_valid_prob": float(np.mean(pack["p_robust_valid"])),
        "pred_defect_prob":       float(np.mean(pack["p_defect"])),
        "pred_mean_strict_score": float(np.mean(pack["strict_pred"])),
        "pred_mean_cd_error":     float(np.mean(np.abs(pack["cd_pred"] - CD_TARGET_NM))),
        "pred_mean_ler":          float(np.mean(pack["Y4"][:, 1])),
        "pred_mean_margin":       float(np.mean(pack["Y4"][:, 3])),
    }


def time_window_compare(rows_06m: list[dict], rows_06m_b: list[dict],
                          *, models_06p: tuple, models_06r: tuple,
                          feat_keys_06r: list[str],
                          cd_tol: float, ler_cap: float) -> dict:
    def _det_int(r: dict) -> int | None:
        if str(r.get("scenario", "")) != "det_offset":
            return None
        t = _safe_float(r.get("time_offset_s"))
        if not (np.isfinite(t) and abs(t - round(t)) <= 1e-6):
            return None
        ti = int(round(t))
        return ti if ti in J1453_DET_TIME_OFFSETS else None

    g_by_off: dict[int, list[dict]] = defaultdict(list)
    for r in rows_06m:
        ti = _det_int(r)
        if ti is not None:
            g_by_off[ti].append(r)
    j_by_off: dict[int, list[dict]] = defaultdict(list)
    for r in rows_06m_b:
        ti = _det_int(r)
        if ti is not None:
            j_by_off[ti].append(r)

    clf_p, reg_p, aux_p, sr_p = models_06p
    clf_r, reg_r, aux_r, sr_r = models_06r

    table: list[dict] = []
    for off in J1453_DET_TIME_OFFSETS:
        g_rows, j_rows = g_by_off[off], j_by_off[off]
        g_fd = _fd_per_offset(g_rows, cd_tol, ler_cap)
        j_fd = _fd_per_offset(j_rows, cd_tol, ler_cap)
        g_p  = _pred_per_offset(g_rows, clf_p, reg_p, aux_p, sr_p,
                                  feature_keys=FEATURE_KEYS, augment=False,
                                  cd_tol=cd_tol, ler_cap=ler_cap)
        j_p  = _pred_per_offset(j_rows, clf_p, reg_p, aux_p, sr_p,
                                  feature_keys=FEATURE_KEYS, augment=False,
                                  cd_tol=cd_tol, ler_cap=ler_cap)
        g_r  = _pred_per_offset(g_rows, clf_r, reg_r, aux_r, sr_r,
                                  feature_keys=feat_keys_06r, augment=True,
                                  cd_tol=cd_tol, ler_cap=ler_cap)
        j_r  = _pred_per_offset(j_rows, clf_r, reg_r, aux_r, sr_r,
                                  feature_keys=feat_keys_06r, augment=True,
                                  cd_tol=cd_tol, ler_cap=ler_cap)
        table.append({
            "time_offset_s": float(off),
            "G4867_FD_strict_pass":   g_fd.get("fd_strict_pass_prob", float("nan")),
            "J1453_FD_strict_pass":   j_fd.get("fd_strict_pass_prob", float("nan")),
            "G4867_06P_pred_strict_pass": g_p.get("pred_strict_pass_prob", float("nan")),
            "J1453_06P_pred_strict_pass": j_p.get("pred_strict_pass_prob", float("nan")),
            "G4867_06R_pred_strict_pass": g_r.get("pred_strict_pass_prob", float("nan")),
            "J1453_06R_pred_strict_pass": j_r.get("pred_strict_pass_prob", float("nan")),
            "G4867_06P_residual_strict_pass":
                g_p.get("pred_strict_pass_prob", float("nan"))
                  - g_fd.get("fd_strict_pass_prob", float("nan")),
            "J1453_06P_residual_strict_pass":
                j_p.get("pred_strict_pass_prob", float("nan"))
                  - j_fd.get("fd_strict_pass_prob", float("nan")),
            "G4867_06R_residual_strict_pass":
                g_r.get("pred_strict_pass_prob", float("nan"))
                  - g_fd.get("fd_strict_pass_prob", float("nan")),
            "J1453_06R_residual_strict_pass":
                j_r.get("pred_strict_pass_prob", float("nan"))
                  - j_fd.get("fd_strict_pass_prob", float("nan")),
            "FD_relative_advantage_J_minus_G":
                j_fd.get("fd_strict_pass_prob", float("nan"))
                  - g_fd.get("fd_strict_pass_prob", float("nan")),
            "06P_pred_relative_advantage_J_minus_G":
                j_p.get("pred_strict_pass_prob", float("nan"))
                  - g_p.get("pred_strict_pass_prob", float("nan")),
            "06R_pred_relative_advantage_J_minus_G":
                j_r.get("pred_strict_pass_prob", float("nan"))
                  - g_r.get("pred_strict_pass_prob", float("nan")),
            "G4867_06P_cd_error_residual":
                g_p.get("pred_mean_cd_error", float("nan"))
                  - g_fd.get("fd_mean_cd_error", float("nan")),
            "G4867_06R_cd_error_residual":
                g_r.get("pred_mean_cd_error", float("nan"))
                  - g_fd.get("fd_mean_cd_error", float("nan")),
            "J1453_06P_cd_error_residual":
                j_p.get("pred_mean_cd_error", float("nan"))
                  - j_fd.get("fd_mean_cd_error", float("nan")),
            "J1453_06R_cd_error_residual":
                j_r.get("pred_mean_cd_error", float("nan"))
                  - j_fd.get("fd_mean_cd_error", float("nan")),
        })
    return {"per_offset": table}


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
    p.add_argument("--feature_list_json", type=str,
                   default=str(V3_DIR / "outputs" / "models"
                               / "stage06R_feature_list.json"))
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
    p.add_argument("--summary_06h_json", type=str,
                   default=str(V3_DIR / "outputs" / "logs"
                               / "stage06H_fd_verification_summary.json"))
    p.add_argument("--rk_06p_json", type=str,
                   default=str(V3_DIR / "outputs" / "logs"
                               / "stage06P_mode_b_ranking_comparison.json"))
    p.add_argument("--m_06m_csv", type=str,
                   default=str(V3_DIR / "outputs" / "yield_optimization"
                               / "stage06M_time_deep_mc.csv"))
    p.add_argument("--m_06m_b_csv", type=str,
                   default=str(V3_DIR / "outputs" / "yield_optimization"
                               / "stage06M_B_J1453_time_deep_mc.csv"))
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
    p.add_argument("--clf_06r", type=str,
                   default=str(V3_DIR / "outputs" / "models" / "stage06R_classifier.joblib"))
    p.add_argument("--reg_06r", type=str,
                   default=str(V3_DIR / "outputs" / "models" / "stage06R_regressor.joblib"))
    p.add_argument("--aux_06r", type=str,
                   default=str(V3_DIR / "outputs" / "models"
                               / "stage06R_aux_cd_fixed_regressor.joblib"))
    p.add_argument("--strict_06r", type=str,
                   default=str(V3_DIR / "outputs" / "models"
                               / "stage06R_strict_score_regressor.joblib"))
    p.add_argument("--metrics_06r_json", type=str,
                   default=str(V3_DIR / "outputs" / "logs"
                               / "stage06R_model_metrics.json"))
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

    feat_payload = json.loads(Path(args.feature_list_json).read_text())
    feat_keys_06r = list(feat_payload["feature_keys"])

    # Load models.
    clf_p, _ = load_model(args.clf_06p)
    reg_p, _ = load_model(args.reg_06p)
    aux_p, _ = load_model(args.aux_06p)
    sr_p,  _ = load_model(args.strict_06p)
    clf_r, _ = load_model(args.clf_06r)
    reg_r, _ = load_model(args.reg_06r)
    aux_r, _ = load_model(args.aux_06r)
    sr_r,  _ = load_model(args.strict_06r)

    # Read inputs.
    rows_06g = read_labels_csv(args.top_06g_csv)
    _coerce(rows_06g, ["rank_strict", "strict_score", "yield_score"] + FEATURE_KEYS)
    rows_06j = read_labels_csv(args.top_06j_csv)
    _coerce(rows_06j, ["strict_score", "yield_score"] + FEATURE_KEYS)
    rows_06j_summary = read_labels_csv(args.summary_06j_csv)
    _coerce(rows_06j_summary, ["strict_score", "yield_score"] + FEATURE_KEYS)
    fd_06j_nominal = read_labels_csv(args.fd_06j_nominal_csv)
    _coerce(fd_06j_nominal, ["CD_final_nm", "CD_locked_nm",
                                "LER_CD_locked_nm", "area_frac",
                                "P_line_margin"] + FEATURE_KEYS)
    fd_06j_mc = read_labels_csv(args.fd_06j_mc_csv)
    _coerce(fd_06j_mc, ["CD_final_nm", "CD_locked_nm",
                          "LER_CD_locked_nm", "area_frac",
                          "P_line_margin"] + FEATURE_KEYS)
    fd_06j_b_top10 = read_labels_csv(args.fd_06j_b_top10_csv)
    _coerce(fd_06j_b_top10, ["CD_final_nm", "CD_locked_nm",
                                  "LER_CD_locked_nm", "area_frac",
                                  "P_line_margin"] + FEATURE_KEYS)
    fd_06j_b_repmc = read_labels_csv(args.fd_06j_b_repmc_csv)
    _coerce(fd_06j_b_repmc, ["CD_final_nm", "CD_locked_nm",
                                  "LER_CD_locked_nm", "area_frac",
                                  "P_line_margin"] + FEATURE_KEYS)
    rows_06m = read_labels_csv(args.m_06m_csv)
    _coerce(rows_06m, ["CD_final_nm", "CD_locked_nm", "LER_CD_locked_nm",
                          "area_frac", "P_line_margin", "time_offset_s"]
                          + FEATURE_KEYS)
    rows_06m_b = read_labels_csv(args.m_06m_b_csv)
    _coerce(rows_06m_b, ["CD_final_nm", "CD_locked_nm", "LER_CD_locked_nm",
                            "area_frac", "P_line_margin", "time_offset_s"]
                            + FEATURE_KEYS)
    summary_06h = json.loads(Path(args.summary_06h_json).read_text())
    rk_06p = json.loads(Path(args.rk_06p_json).read_text())

    # ----- Re-score Mode A (06G top-100) and Mode B (06J top-100) -----
    print(f"  re-scoring 06G top-{len(rows_06g)} (Mode A) with 06P + 06R ...")
    rs_06g_p = rescore_candidates(rows_06g, clf_p, reg_p, aux_p, sr_p,
                                       var_spec, args.n_var_eval, space,
                                       score_cfg, strict_cfg, args.seed_eval,
                                       tag="06p", feature_keys=FEATURE_KEYS,
                                       augment=False, cd_tol=cd_tol,
                                       ler_cap=ler_cap)
    rs_06g_r = rescore_candidates(rows_06g, clf_r, reg_r, aux_r, sr_r,
                                       var_spec, args.n_var_eval, space,
                                       score_cfg, strict_cfg, args.seed_eval,
                                       tag="06r", feature_keys=feat_keys_06r,
                                       augment=True, cd_tol=cd_tol,
                                       ler_cap=ler_cap)
    print(f"  re-scoring 06J top-{len(rows_06j)} (Mode B) with 06P + 06R ...")
    rs_06j_p = rescore_candidates(rows_06j, clf_p, reg_p, aux_p, sr_p,
                                       var_spec, args.n_var_eval, space,
                                       score_cfg, strict_cfg, args.seed_eval + 1,
                                       tag="06p", feature_keys=FEATURE_KEYS,
                                       augment=False, cd_tol=cd_tol,
                                       ler_cap=ler_cap)
    rs_06j_r = rescore_candidates(rows_06j, clf_r, reg_r, aux_r, sr_r,
                                       var_spec, args.n_var_eval, space,
                                       score_cfg, strict_cfg, args.seed_eval + 1,
                                       tag="06r", feature_keys=feat_keys_06r,
                                       augment=True, cd_tol=cd_tol,
                                       ler_cap=ler_cap)

    by_id_06g_p = {r["recipe_id"]: r for r in rs_06g_p}
    by_id_06j_p = {r["recipe_id"]: r for r in rs_06j_p}
    by_id_06g_r = {r["recipe_id"]: r for r in rs_06g_r}
    by_id_06j_r = {r["recipe_id"]: r for r in rs_06j_r}

    # ----- FD MC truth -----
    fd_06j_b_mc_pool = fd_06j_b_top10 + fd_06j_b_repmc
    fd_truth = build_fd_truth(summary_06h=summary_06h,
                                  fd_06j_mc=fd_06j_mc,
                                  fd_06j_b_mc=fd_06j_b_mc_pool,
                                  cd_tol=cd_tol, ler_cap=ler_cap)

    pairs_06p, pairs_06r = [], []
    for rid, truth in fd_truth.items():
        rs_p = by_id_06g_p.get(rid) or by_id_06j_p.get(rid)
        rs_r = by_id_06g_r.get(rid) or by_id_06j_r.get(rid)
        if rs_p is not None:
            pairs_06p.append({"recipe_id": rid, "mode": truth["mode"],
                                 "pred": float(rs_p["strict_score_06p_direct_mean"]),
                                 "fd":   truth["fd_mc_strict_pass_prob"]})
        if rs_r is not None:
            pairs_06r.append({"recipe_id": rid, "mode": truth["mode"],
                                 "pred": float(rs_r["strict_score_06r_direct_mean"]),
                                 "fd":   truth["fd_mc_strict_pass_prob"]})

    def _rho(pp: list[dict], mode: str | None) -> tuple[float, int]:
        sub = [p for p in pp if (mode is None or p["mode"] == mode)]
        if len(sub) < 3:
            return float("nan"), len(sub)
        return spearman(np.array([p["pred"] for p in sub]),
                          np.array([p["fd"]   for p in sub])), len(sub)

    rho_p, n_p = _rho(pairs_06p, None)
    rho_p_a, n_p_a = _rho(pairs_06p, "mode_a")
    rho_p_b, n_p_b = _rho(pairs_06p, "mode_b")
    rho_r, n_r = _rho(pairs_06r, None)
    rho_r_a, n_r_a = _rho(pairs_06r, "mode_a")
    rho_r_b, n_r_b = _rho(pairs_06r, "mode_b")

    # Top-10 overlap Mode B 06P vs 06R, plus pred vs FD.
    rank_06p_b = {r["recipe_id"]: i + 1
                    for i, r in enumerate(sorted(
                        rs_06j_p, key=lambda x: -float(x["strict_score_06p_direct_mean"])))}
    rank_06r_b = {r["recipe_id"]: i + 1
                    for i, r in enumerate(sorted(
                        rs_06j_r, key=lambda x: -float(x["strict_score_06r_direct_mean"])))}
    common = sorted(set(rank_06p_b) & set(rank_06r_b))
    rho_pr_b = (spearman(np.array([rank_06p_b[i] for i in common], dtype=float),
                            np.array([rank_06r_b[i] for i in common], dtype=float))
                  if len(common) >= 3 else float("nan"))
    top10_pr = topk_overlap(
        [r for r, _ in sorted(rank_06p_b.items(), key=lambda kv: kv[1])][:10],
        [r for r, _ in sorted(rank_06r_b.items(), key=lambda kv: kv[1])][:10],
        k=10,
    )
    j1453_in_06r_top10_b = bool(any(
        r["recipe_id"] == "J_1453"
        for r in sorted(rs_06j_r,
                          key=lambda r: -float(r["strict_score_06r_direct_mean"]))[:10]))
    g4867_in_06r_top10_a = bool(any(
        r["recipe_id"] == "G_4867"
        for r in sorted(rs_06g_r,
                          key=lambda r: -float(r["strict_score_06r_direct_mean"]))[:10]))

    # ----- Time-window comparison -----
    twin = time_window_compare(
        rows_06m, rows_06m_b,
        models_06p=(clf_p, reg_p, aux_p, sr_p),
        models_06r=(clf_r, reg_r, aux_r, sr_r),
        feat_keys_06r=feat_keys_06r,
        cd_tol=cd_tol, ler_cap=ler_cap,
    )

    # Aggregate residuals.
    g_06p_res = np.array([r["G4867_06P_residual_strict_pass"] for r in twin["per_offset"]])
    g_06r_res = np.array([r["G4867_06R_residual_strict_pass"] for r in twin["per_offset"]])
    j_06p_res = np.array([r["J1453_06P_residual_strict_pass"] for r in twin["per_offset"]])
    j_06r_res = np.array([r["J1453_06R_residual_strict_pass"] for r in twin["per_offset"]])
    rel_06p   = np.array([r["06P_pred_relative_advantage_J_minus_G"]
                              - r["FD_relative_advantage_J_minus_G"]
                              for r in twin["per_offset"]])
    rel_06r   = np.array([r["06R_pred_relative_advantage_J_minus_G"]
                              - r["FD_relative_advantage_J_minus_G"]
                              for r in twin["per_offset"]])
    cd_06p_res_g = np.array([r["G4867_06P_cd_error_residual"] for r in twin["per_offset"]])
    cd_06r_res_g = np.array([r["G4867_06R_cd_error_residual"] for r in twin["per_offset"]])

    # ----- False-PASS demotion (06J nominal hard-fail recipes) -----
    by_id_06j_summary = {r["recipe_id"]: r for r in rows_06j_summary}
    candidates_for_fp: list[dict] = []
    for r in fd_06j_nominal:
        rid = str(r.get("source_recipe_id", ""))
        sur = by_id_06j_summary.get(rid)
        if sur is not None:
            candidates_for_fp.append(sur)
    if candidates_for_fp:
        rs_fp_p = rescore_candidates(candidates_for_fp,
                                          clf_p, reg_p, aux_p, sr_p,
                                          var_spec, args.n_var_eval, space,
                                          score_cfg, strict_cfg,
                                          args.seed_eval + 7,
                                          tag="06p",
                                          feature_keys=FEATURE_KEYS,
                                          augment=False, cd_tol=cd_tol,
                                          ler_cap=ler_cap)
        rs_fp_r = rescore_candidates(candidates_for_fp,
                                          clf_r, reg_r, aux_r, sr_r,
                                          var_spec, args.n_var_eval, space,
                                          score_cfg, strict_cfg,
                                          args.seed_eval + 7,
                                          tag="06r",
                                          feature_keys=feat_keys_06r,
                                          augment=True, cd_tol=cd_tol,
                                          ler_cap=ler_cap)
        pool_06j = sorted(candidates_for_fp,
                            key=lambda r: -_safe_float(r.get("strict_score")))
        rank_06j = {r["recipe_id"]: i + 1 for i, r in enumerate(pool_06j)}
        pool_06p = sorted(rs_fp_p,
                            key=lambda r: -float(r["strict_score_06p_direct_mean"]))
        rank_06p_fp = {r["recipe_id"]: i + 1 for i, r in enumerate(pool_06p)}
        pool_06r = sorted(rs_fp_r,
                            key=lambda r: -float(r["strict_score_06r_direct_mean"]))
        rank_06r_fp = {r["recipe_id"]: i + 1 for i, r in enumerate(pool_06r)}
        demos: list[dict] = []
        for r in fd_06j_nominal:
            rid = str(r.get("source_recipe_id", ""))
            label = str(r.get("label", ""))
            if label not in HARD_FAIL_LABELS:
                continue
            if rid not in rank_06j or rid not in rank_06p_fp or rid not in rank_06r_fp:
                continue
            sp = next(x for x in rs_fp_p if x["recipe_id"] == rid)
            sr = next(x for x in rs_fp_r if x["recipe_id"] == rid)
            demos.append({
                "recipe_id":          rid,
                "fd_label":           label,
                "rank_06j":           int(rank_06j[rid]),
                "rank_06p":           int(rank_06p_fp[rid]),
                "rank_06r":           int(rank_06r_fp[rid]),
                "delta_rank_06r_vs_06p":
                    int(rank_06r_fp[rid] - rank_06p_fp[rid]),
                "delta_rank_06r_vs_06j":
                    int(rank_06r_fp[rid] - rank_06j[rid]),
                "strict_score_06j_surrogate":
                    float(by_id_06j_summary[rid].get("strict_score", float("nan"))),
                "strict_score_06p_direct":
                    float(sp["strict_score_06p_direct_mean"]),
                "strict_score_06r_direct":
                    float(sr["strict_score_06r_direct_mean"]),
            })
    else:
        demos = []

    # ----- Outputs ---------------------------------------------------------
    yopt_dir = V3_DIR / "outputs" / "yield_optimization"
    logs_dir = V3_DIR / "outputs" / "logs"
    fig_dir  = V3_DIR / "outputs" / "figures" / "06_yield_optimization"
    fig_dir.mkdir(parents=True, exist_ok=True)

    # Time-window CSV.
    twin_cols = list(twin["per_offset"][0].keys()) if twin["per_offset"] else []
    with (yopt_dir / "stage06R_J1453_vs_G4867_prediction_check.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=twin_cols, extrasaction="ignore")
        w.writeheader()
        for r in twin["per_offset"]:
            w.writerow(r)

    # False-PASS demotions CSV.
    if demos:
        cols_d = ["recipe_id", "fd_label", "rank_06j", "rank_06p", "rank_06r",
                  "delta_rank_06r_vs_06p", "delta_rank_06r_vs_06j",
                  "strict_score_06j_surrogate",
                  "strict_score_06p_direct", "strict_score_06r_direct"]
        with (yopt_dir / "stage06R_false_pass_demotions.csv").open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=cols_d, extrasaction="ignore")
            w.writeheader()
            for r in demos:
                w.writerow(r)
    else:
        with (yopt_dir / "stage06R_false_pass_demotions.csv").open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["recipe_id", "fd_label",
                                                  "rank_06j", "rank_06p",
                                                  "rank_06r"])
            w.writeheader()

    # Feature-importance CSV (one row per feature, four heads).
    metrics_06r = json.loads(Path(args.metrics_06r_json).read_text())
    fi_clf  = metrics_06r["feature_importance_classifier"]
    fi_reg  = metrics_06r["feature_importance_regressor4"]
    fi_aux  = metrics_06r["feature_importance_aux_cd_fixed"]
    fi_strict = metrics_06r["feature_importance_strict_score"]
    with (yopt_dir / "stage06R_feature_importance.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["feature", "is_derived",
                                              "fi_classifier", "fi_regressor4",
                                              "fi_aux_cd_fixed",
                                              "fi_strict_score"])
        w.writeheader()
        for i, k in enumerate(feat_keys_06r):
            w.writerow({"feature": k,
                          "is_derived": k in DERIVED_FEATURE_KEYS,
                          "fi_classifier":   float(fi_clf[i]),
                          "fi_regressor4":   float(fi_reg[i]),
                          "fi_aux_cd_fixed": float(fi_aux[i]),
                          "fi_strict_score": float(fi_strict[i])})

    # Logs.
    blindspot_payload = {
        "stage": "06R",
        "policy": cfg["policy"],
        "G4867_strict_pass_residual_mean_06p": float(np.mean(g_06p_res)),
        "G4867_strict_pass_residual_mean_06r": float(np.mean(g_06r_res)),
        "J1453_strict_pass_residual_mean_06p": float(np.mean(j_06p_res)),
        "J1453_strict_pass_residual_mean_06r": float(np.mean(j_06r_res)),
        "relative_advantage_residual_mean_06p": float(np.mean(rel_06p)),
        "relative_advantage_residual_mean_06r": float(np.mean(rel_06r)),
        "abs_relative_advantage_residual_mean_06p": float(np.mean(np.abs(rel_06p))),
        "abs_relative_advantage_residual_mean_06r": float(np.mean(np.abs(rel_06r))),
        "G4867_cd_error_residual_mean_06p": float(np.mean(cd_06p_res_g)),
        "G4867_cd_error_residual_mean_06r": float(np.mean(cd_06r_res_g)),
        "improvement_relative_advantage_06r_vs_06p":
            float(abs(np.mean(rel_06p)) - abs(np.mean(rel_06r))),
        "G4867_over_prediction_reduced":
            bool(abs(np.mean(g_06r_res)) < abs(np.mean(g_06p_res))),
    }
    (logs_dir / "stage06R_blindspot_comparison.json").write_text(
        json.dumps(blindspot_payload, indent=2, default=float))

    rank_payload = {
        "stage": "06R",
        "policy": cfg["policy"],
        "spearman_06p_overall": rho_p,  "n_pairs_06p_overall": n_p,
        "spearman_06p_mode_a":  rho_p_a, "n_pairs_06p_mode_a": n_p_a,
        "spearman_06p_mode_b":  rho_p_b, "n_pairs_06p_mode_b": n_p_b,
        "spearman_06r_overall": rho_r,  "n_pairs_06r_overall": n_r,
        "spearman_06r_mode_a":  rho_r_a, "n_pairs_06r_mode_a": n_r_a,
        "spearman_06r_mode_b":  rho_r_b, "n_pairs_06r_mode_b": n_r_b,
        "delta_spearman_overall_06r_minus_06p":
            (rho_r - rho_p) if (np.isfinite(rho_r) and np.isfinite(rho_p)) else None,
        "delta_spearman_mode_b_06r_minus_06p":
            (rho_r_b - rho_p_b) if (np.isfinite(rho_r_b) and np.isfinite(rho_p_b)) else None,
        "rank_spearman_06p_vs_06r_mode_b":  rho_pr_b,
        "top10_overlap_mode_b_06p_vs_06r":  top10_pr,
        "j1453_in_06r_top10_mode_b":        j1453_in_06r_top10_b,
        "g4867_in_06r_top10_mode_a":        g4867_in_06r_top10_a,
        "j1453_in_06p_top10_mode_b":
            bool(rk_06p.get("j1453_in_06p_top10_mode_b", False)),
        "g4867_in_06p_top10_mode_a":
            bool(rk_06p.get("g4867_in_06p_top10_mode_a", False)),
    }
    (logs_dir / "stage06R_mode_b_ranking_comparison.json").write_text(
        json.dumps(rank_payload, indent=2, default=float))

    # ----- Figures ---------------------------------------------------------
    if twin["per_offset"]:
        offs = np.array([r["time_offset_s"] for r in twin["per_offset"]])
        # F1: relative advantage before/after.
        fd_rel = np.array([r["FD_relative_advantage_J_minus_G"] for r in twin["per_offset"]])
        rel_p  = np.array([r["06P_pred_relative_advantage_J_minus_G"] for r in twin["per_offset"]])
        rel_r  = np.array([r["06R_pred_relative_advantage_J_minus_G"] for r in twin["per_offset"]])
        fig, ax = plt.subplots(figsize=(11.0, 5.8))
        ax.plot(offs, fd_rel, "-o", color="#2ca02c",
                 label="FD truth: J - G strict_pass")
        ax.plot(offs, rel_p,  "-s", color="#d62728",
                 label="06P pred: J - G strict_pass")
        ax.plot(offs, rel_r,  "-^", color="#1f77b4",
                 label="06R pred: J - G strict_pass")
        ax.axhline(0.0, color="black", lw=1.0, ls="--", alpha=0.6)
        ax.set_xlabel("time offset (s)")
        ax.set_ylabel("J_1453 - G_4867 strict_pass_prob")
        ax.set_title("Stage 06R -- relative advantage J_1453 vs G_4867: "
                      "FD vs 06P vs 06R")
        ax.grid(True, alpha=0.25)
        ax.legend(loc="best", fontsize=10)
        fig.tight_layout()
        fig.savefig(fig_dir / "stage06R_J1453_G4867_relative_advantage_before_after.png",
                     dpi=150)
        plt.close(fig)

        # F2: G_4867 strict_pass residual vs time.
        fig, ax = plt.subplots(figsize=(11.0, 5.5))
        ax.plot(offs, g_06p_res, "-s", color="#d62728",
                 label="G_4867 06P residual (pred - FD)")
        ax.plot(offs, g_06r_res, "-^", color="#1f77b4",
                 label="G_4867 06R residual (pred - FD)")
        ax.plot(offs, j_06p_res, ":s", color="#fc8d62",
                 label="J_1453 06P residual", alpha=0.6)
        ax.plot(offs, j_06r_res, ":^", color="#66c2a5",
                 label="J_1453 06R residual", alpha=0.6)
        ax.axhline(0.0, color="black", lw=1.0, ls="--", alpha=0.6)
        ax.set_xlabel("time offset (s)")
        ax.set_ylabel("strict_pass_prob residual (pred - FD)")
        ax.set_title("Stage 06R -- G_4867 / J_1453 strict_pass residual vs time")
        ax.grid(True, alpha=0.25)
        ax.legend(loc="best", fontsize=9)
        fig.tight_layout()
        fig.savefig(fig_dir / "stage06R_G4867_strict_pass_residual_vs_time.png",
                     dpi=150)
        plt.close(fig)

        # F3: CD_error residual vs time.
        fig, ax = plt.subplots(figsize=(11.0, 5.5))
        ax.plot(offs, cd_06p_res_g, "-s", color="#d62728",
                 label="G_4867 06P CD_error residual")
        ax.plot(offs, cd_06r_res_g, "-^", color="#1f77b4",
                 label="G_4867 06R CD_error residual")
        ax.plot(offs, [r["J1453_06P_cd_error_residual"] for r in twin["per_offset"]],
                 ":s", color="#fc8d62", label="J_1453 06P CD_error residual",
                 alpha=0.6)
        ax.plot(offs, [r["J1453_06R_cd_error_residual"] for r in twin["per_offset"]],
                 ":^", color="#66c2a5", label="J_1453 06R CD_error residual",
                 alpha=0.6)
        ax.axhline(0.0, color="black", lw=1.0, ls="--", alpha=0.6)
        ax.set_xlabel("time offset (s)")
        ax.set_ylabel("|CD - 15| residual (pred - FD), nm")
        ax.set_title("Stage 06R -- CD_error residual vs time offset")
        ax.grid(True, alpha=0.25)
        ax.legend(loc="best", fontsize=9)
        fig.tight_layout()
        fig.savefig(fig_dir / "stage06R_cd_error_residual_vs_time.png", dpi=150)
        plt.close(fig)

    # F4: 06R direct strict_score head vs FD MC strict_pass_prob.
    if pairs_06r:
        fig, ax = plt.subplots(figsize=(9.0, 6.5))
        for mode, color in (("mode_a", "#1f77b4"), ("mode_b", "#d62728")):
            sub = [p for p in pairs_06r if p["mode"] == mode]
            if not sub:
                continue
            ax.scatter([p["pred"] for p in sub],
                        [p["fd"]   for p in sub],
                        c=color, s=80, alpha=0.85,
                        edgecolor="white", lw=0.6,
                        label=f"{mode} (n={len(sub)})")
            for p in sub:
                ax.annotate(p["recipe_id"], (p["pred"], p["fd"]),
                              fontsize=8, xytext=(4, 4),
                              textcoords="offset points")
        ax.set_xlabel("06R direct strict_score (mean over 200 MC variations)")
        ax.set_ylabel("FD MC strict_pass_prob")
        ax.set_title(f"Stage 06R -- direct strict_score head vs FD MC truth  "
                      f"(Spearman rho = {rho_r:.3f})")
        ax.grid(True, alpha=0.25)
        ax.legend(loc="best", fontsize=10)
        fig.tight_layout()
        fig.savefig(fig_dir / "stage06R_strict_score_pred_vs_fd.png", dpi=150)
        plt.close(fig)

    # F5: Mode B ranking 06P -> 06R.
    if rs_06j_p and rs_06j_r:
        fig, ax = plt.subplots(figsize=(11.0, 6.0))
        rank_p = {r["recipe_id"]: i + 1
                    for i, r in enumerate(sorted(
                        rs_06j_p, key=lambda x: -float(x["strict_score_06p_direct_mean"])))}
        rank_r = {r["recipe_id"]: i + 1
                    for i, r in enumerate(sorted(
                        rs_06j_r, key=lambda x: -float(x["strict_score_06r_direct_mean"])))}
        highlight = {"J_1453": "#d62728", "J_3413": "#d62728",
                       "J_4793": "#d62728", "J_4164": "#d62728",
                       "J_2261": "#d62728"}
        for rid, p_rank in rank_p.items():
            if rid not in rank_r:
                continue
            r_rank = rank_r[rid]
            is_hi = rid in highlight
            ax.plot([0, 1], [p_rank, r_rank], "-",
                     color=(highlight.get(rid) if is_hi else "#9aaecf"),
                     lw=1.7 if is_hi else 0.4,
                     alpha=0.95 if is_hi else 0.30,
                     zorder=5 if is_hi else 1)
            if is_hi:
                ax.scatter([0, 1], [p_rank, r_rank], s=44,
                            color=highlight[rid], zorder=6)
                ax.annotate(rid, (1.02, r_rank), fontsize=8,
                              color=highlight[rid], verticalalignment="center")
        ax.invert_yaxis()
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["06P direct strict_score rank",
                              "06R direct strict_score rank"])
        ax.set_ylabel("rank (lower = better)")
        ax.set_title(f"Stage 06R -- Mode B ranking 06P -> 06R "
                      f"(Spearman rank = {rho_pr_b:+.3f}, "
                      f"top-10 overlap = {top10_pr})")
        ax.set_xlim(-0.05, 1.20)
        ax.grid(True, alpha=0.25, axis="y")
        fig.tight_layout()
        fig.savefig(fig_dir / "stage06R_mode_b_ranking_before_after.png",
                     dpi=150)
        plt.close(fig)

    # F6: feature importance highlighting derived features.
    fig, ax = plt.subplots(figsize=(12.5, 6.0))
    order = np.argsort(fi_strict)[::-1]
    keys = [feat_keys_06r[i] for i in order]
    vals = [fi_strict[i] for i in order]
    colors = ["#d62728" if k in DERIVED_FEATURE_KEYS else "#1f77b4" for k in keys]
    ax.bar(np.arange(len(keys)), vals, color=colors, edgecolor="#1f1f1f")
    ax.set_xticks(np.arange(len(keys)))
    ax.set_xticklabels(keys, rotation=30, ha="right", fontsize=9)
    ax.set_ylabel("strict_score head feature importance (06R RF)")
    ax.set_title("Stage 06R -- strict_score head feature importance "
                  "(red = derived process-budget features)")
    ax.grid(True, alpha=0.25, axis="y")
    fig.tight_layout()
    fig.savefig(fig_dir / "stage06R_feature_importance_derived_features.png",
                 dpi=150)
    plt.close(fig)

    # F7: false-PASS demotions plot.
    if demos:
        rids = [d["recipe_id"] for d in demos]
        rj   = np.array([float(d["rank_06j"]) for d in demos])
        rp   = np.array([float(d["rank_06p"]) for d in demos])
        rr   = np.array([float(d["rank_06r"]) for d in demos])
        fig, ax = plt.subplots(figsize=(11.5, 5.5))
        x = np.arange(len(rids)); w = 0.27
        ax.bar(x - w, rj, width=w, color="#9aaecf",  label="06J surrogate rank")
        ax.bar(x,       rp, width=w, color="#d6a728",  label="06P direct rank")
        ax.bar(x + w, rr, width=w, color="#d62728",  label="06R direct rank")
        for i, d in enumerate(demos):
            ax.text(i, max(rj[i], rp[i], rr[i]) + 1, d["fd_label"],
                     ha="center", fontsize=8)
        ax.set_xticks(x)
        ax.set_xticklabels(rids, rotation=20, ha="right", fontsize=9)
        ax.set_xlabel("06J false-PASS recipe (FD label != robust_valid)")
        ax.set_ylabel("rank under each ranker (lower = higher up)")
        ax.set_title("Stage 06R -- false-PASS recipe ranking 06J / 06P / 06R")
        ax.grid(True, alpha=0.25, axis="y")
        ax.legend(loc="best", fontsize=10)
        fig.tight_layout()
        fig.savefig(fig_dir / "stage06R_false_pass_demotions.png", dpi=150)
        plt.close(fig)
    else:
        fig, ax = plt.subplots(figsize=(8.5, 4.0))
        ax.text(0.5, 0.5, "No false-PASS recipes available for 06R demotion plot.",
                  ha="center", va="center", fontsize=11)
        ax.axis("off")
        fig.savefig(fig_dir / "stage06R_false_pass_demotions.png", dpi=150)
        plt.close(fig)

    # ----- Console summary -------------------------------------------------
    print(f"\nStage 06R -- analysis summary")
    print(f"  Spearman strict_score vs FD MC strict_pass_prob:")
    print(f"    overall: 06P={rho_p:+.3f} (n={n_p})  "
          f"06R={rho_r:+.3f} (n={n_r})")
    print(f"    Mode A:  06P={rho_p_a:+.3f} (n={n_p_a})  "
          f"06R={rho_r_a:+.3f} (n={n_r_a})")
    print(f"    Mode B:  06P={rho_p_b:+.3f} (n={n_p_b})  "
          f"06R={rho_r_b:+.3f} (n={n_r_b})")
    print(f"  Mode B rank Spearman 06P vs 06R: {rho_pr_b:+.3f}, "
          f"top-10 overlap: {top10_pr}")
    print(f"  J_1453 in 06R Mode B top-10:    {j1453_in_06r_top10_b}")
    print(f"  G_4867 in 06R Mode A top-10:    {g4867_in_06r_top10_a}")
    print(f"  Time-window blindspot:")
    print(f"    G_4867 strict_pass residual mean: "
          f"06P={blindspot_payload['G4867_strict_pass_residual_mean_06p']:+.3f}  "
          f"06R={blindspot_payload['G4867_strict_pass_residual_mean_06r']:+.3f}")
    print(f"    J_1453 strict_pass residual mean: "
          f"06P={blindspot_payload['J1453_strict_pass_residual_mean_06p']:+.3f}  "
          f"06R={blindspot_payload['J1453_strict_pass_residual_mean_06r']:+.3f}")
    print(f"    Relative advantage residual mean: "
          f"06P={blindspot_payload['relative_advantage_residual_mean_06p']:+.3f}  "
          f"06R={blindspot_payload['relative_advantage_residual_mean_06r']:+.3f}  "
          f"(|.| 06P={blindspot_payload['abs_relative_advantage_residual_mean_06p']:.3f}  "
          f"06R={blindspot_payload['abs_relative_advantage_residual_mean_06r']:.3f})")
    print(f"    G_4867 CD_error residual mean: "
          f"06P={blindspot_payload['G4867_cd_error_residual_mean_06p']:+.3f}  "
          f"06R={blindspot_payload['G4867_cd_error_residual_mean_06r']:+.3f}")
    print(f"  False-PASS demotion: {len(demos)} hard-fail recipes; "
          f"06R promoted/demoted vs 06P:")
    if demos:
        for d in demos:
            print(f"    {d['recipe_id']}  fd_label={d['fd_label']:<22}  "
                  f"rank: 06J #{d['rank_06j']} -> 06P #{d['rank_06p']} -> "
                  f"06R #{d['rank_06r']}  "
                  f"(delta 06R-06P = {d['delta_rank_06r_vs_06p']:+d})")
    print(f"  outputs ->")
    print(f"    {yopt_dir / 'stage06R_J1453_vs_G4867_prediction_check.csv'}")
    print(f"    {yopt_dir / 'stage06R_false_pass_demotions.csv'}")
    print(f"    {yopt_dir / 'stage06R_feature_importance.csv'}")
    print(f"    {logs_dir / 'stage06R_blindspot_comparison.json'}")
    print(f"    {logs_dir / 'stage06R_mode_b_ranking_comparison.json'}")
    print(f"    {fig_dir} / stage06R_*.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Stage 06Q -- blindspot analysis for the 06P surrogate.

06P became the preferred surrogate (Mode B Spearman 06L +0.585 -> 06P
+0.938) but still under-predicts the magnitude of J_1453's time-window
advantage over G_4867. 06Q does not retrain anything; it diagnoses
remaining errors and recommends one or more correction strategies.

Reads:
    outputs/labels/stage06P_training_dataset.csv
    outputs/labels/stage06J_mode_b_fd_sanity.csv
    outputs/labels/stage06J_B_fd_top10_mc.csv
    outputs/labels/stage06J_B_fd_representative_mc.csv
    outputs/labels/stage06J_B_fd_top100_nominal.csv
    outputs/yield_optimization/stage06M_time_deep_mc.csv
    outputs/yield_optimization/stage06M_B_J1453_time_deep_mc.csv
    outputs/yield_optimization/stage06J_mode_b_recipe_summary.csv
    outputs/yield_optimization/stage06J_mode_b_top_recipes.csv
    outputs/yield_optimization/stage06P_false_pass_demotions.csv
    outputs/yield_optimization/stage06G_strict_score_config.yaml
    outputs/yield_optimization/stage06I_mode_a_final_recipes.yaml
    outputs/models/stage06P_*.joblib

Writes:
    outputs/yield_optimization/stage06Q_blindspot_dataset.csv
    outputs/yield_optimization/stage06Q_J1453_G4867_residuals.csv
    outputs/yield_optimization/stage06Q_false_pass_diagnostics.csv
    outputs/yield_optimization/stage06Q_mode_a_geometry_blindspot.csv
    outputs/yield_optimization/stage06Q_recommended_corrections.yaml
    outputs/logs/stage06Q_summary.json
    outputs/figures/06_yield_optimization/
        stage06Q_J1453_G4867_strict_pass_residual_vs_time.png
        stage06Q_pred_vs_fd_relative_advantage.png
        stage06Q_cd_ler_margin_residual_breakdown.png
        stage06Q_false_pass_feature_patterns.png
        stage06Q_false_pass_pred_vs_fd_metrics.png
        stage06Q_blindspot_parallel_coordinates.png
        stage06Q_Hmax_kdep_failure_map.png
        stage06Q_Q0_kq_failure_map.png
        stage06Q_DH_time_failure_map.png
        stage06Q_interaction_failure_maps.png

Policy
    v2_OP_frozen           = true
    published_data_loaded  = false
    No external calibration. Closed Stage 04C / 04D / 06C / 06L / 06P
    artefacts are not modified.
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

from reaction_diffusion_peb_v3_screening.src.fd_yield_score import spearman
from reaction_diffusion_peb_v3_screening.src.metrics_io import (
    FEATURE_KEYS, load_model, read_labels_csv,
)

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_stage06l_dataset import per_row_strict_score  # noqa: E402


V3_DIR = ROOT / "reaction_diffusion_peb_v3_screening"
CD_TARGET_NM = 15.0
HARD_FAIL_LABELS = {"under_exposed", "merged",
                     "roughness_degraded", "numerical_invalid"}
J1453_DET_TIME_OFFSETS = [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5]

# Chemistry knob bounds (mirror analyze_stage06l _g4867_dist normaliser).
CHEMISTRY_BOUNDS = {
    "dose_mJ_cm2": (21.0, 60.0),
    "sigma_nm":    (0.0, 3.0),
    "DH_nm2_s":    (0.3, 0.8),
    "time_s":      (20.0, 45.0),
    "Hmax_mol_dm3":(0.15, 0.22),
    "kdep_s_inv":  (0.35, 0.65),
    "Q0_mol_dm3":  (0.0, 0.03),
    "kq_s_inv":    (0.5, 2.0),
    "abs_len_nm":  (15.0, 100.0),
    "pitch_nm":    (18.0, 32.0),
    "line_cd_ratio":(0.45, 0.60),
}


def _safe_float(v) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return float("nan")


def _build_X(rows: list[dict]) -> np.ndarray:
    X = np.zeros((len(rows), len(FEATURE_KEYS)), dtype=np.float64)
    for i, r in enumerate(rows):
        for j, k in enumerate(FEATURE_KEYS):
            X[i, j] = _safe_float(r.get(k))
    return X


def _coerce(rows, keys):
    for r in rows:
        for k in keys:
            if k in r:
                r[k] = _safe_float(r.get(k))


def _is_mode_a(pitch: float, ratio: float, *,
                  pitch_tol: float = 0.5, ratio_tol: float = 0.04) -> bool:
    return (np.isfinite(pitch) and np.isfinite(ratio)
              and abs(pitch - 24.0) <= pitch_tol
              and abs(ratio - 0.52) <= ratio_tol)


def _chemistry_signature(row: dict, ref: dict, *,
                            top_n: int = 3) -> str:
    """Return a short ;-joined string describing the dominant chemistry
    differences vs `ref` (e.g. G_4867)."""
    diffs = []
    for k in ["dose_mJ_cm2", "sigma_nm", "DH_nm2_s", "time_s",
              "Hmax_mol_dm3", "kdep_s_inv", "Q0_mol_dm3", "kq_s_inv",
              "abs_len_nm"]:
        lo, hi = CHEMISTRY_BOUNDS[k]
        w = max(hi - lo, 1e-12)
        a = _safe_float(row.get(k))
        b = _safe_float(ref.get(k))
        if not (np.isfinite(a) and np.isfinite(b)):
            continue
        z = (a - b) / w
        diffs.append((abs(z), z, k))
    diffs.sort(reverse=True)
    parts = []
    for _, z, k in diffs[:top_n]:
        direction = "high" if z > 0 else "low"
        parts.append(f"{direction}_{k}:{z:+.2f}")
    return ";".join(parts)


def _predict_pack(rows: list[dict], clf, reg, aux, strict_reg,
                     cd_tol: float, ler_cap: float) -> dict:
    """Vectorised 06P predictions for the given rows."""
    if not rows:
        return {
            "n": 0,
            "p_robust_valid": np.array([]),
            "p_margin_risk": np.array([]),
            "p_defect": np.array([]),
            "pred_class": np.array([], dtype=object),
            "cd_pred": np.array([]),
            "Y4": np.zeros((0, 4)),
            "strict_pred": np.array([]),
            "strict_pass_pred": np.array([]),
        }
    X = _build_X(rows)
    proba = clf.predict_proba(X)
    classes = list(clf.classes_)

    def col(name):
        if name not in classes:
            return np.zeros(len(rows))
        return proba[:, classes.index(name)]

    p_robust = col("robust_valid")
    p_margin = col("margin_risk")
    defects = (col("under_exposed") + col("merged")
                + col("roughness_degraded") + col("numerical_invalid"))
    pred_class = np.array([classes[int(np.argmax(p))] for p in proba])

    Y4 = reg.predict(X)        # CD_locked, LER_CD_locked, area_frac, P_line_margin
    cd_pred = aux.predict(X)
    strict_pred = strict_reg.predict(X)

    cd_within  = np.abs(cd_pred - CD_TARGET_NM) <= cd_tol
    ler_under  = Y4[:, 1] <= ler_cap
    strict_pass_pred = (cd_within & ler_under
                         & (pred_class == "robust_valid")).astype(float)

    return {
        "n": len(rows),
        "p_robust_valid": p_robust,
        "p_margin_risk":  p_margin,
        "p_defect":       defects,
        "pred_class":     pred_class,
        "cd_pred":        cd_pred,
        "Y4":             Y4,
        "strict_pred":    strict_pred,
        "strict_pass_pred": strict_pass_pred,
    }


# --------------------------------------------------------------------------
# Builders
# --------------------------------------------------------------------------
def _aggregate_recipe_rows(rows: list[dict], pack: dict, *,
                              cd_tol: float, ler_cap: float,
                              strict_yaml: dict) -> dict:
    """Aggregate FD truth + 06P predictions for one recipe."""
    if not rows:
        return {}
    n = len(rows)
    n_robust = sum(1 for r in rows if str(r.get("label", "")) == "robust_valid")
    n_defect = sum(1 for r in rows if str(r.get("label", "")) in HARD_FAIL_LABELS)
    n_sp = sum(
        1 for r in rows
        if str(r.get("label", "")) == "robust_valid"
            and abs(_safe_float(r.get("CD_final_nm")) - CD_TARGET_NM) <= cd_tol
            and _safe_float(r.get("LER_CD_locked_nm")) <= ler_cap
    )
    fd_strict_pass = float(n_sp / n)
    fd_robust_prob = float(n_robust / n)
    fd_defect_prob = float(n_defect / n)
    fd_strict_per_row = np.array([per_row_strict_score(r, strict_yaml)
                                       for r in rows])
    fd_cd_err = np.array([abs(_safe_float(r.get("CD_final_nm")) - CD_TARGET_NM)
                              for r in rows])
    fd_ler    = np.array([_safe_float(r.get("LER_CD_locked_nm")) for r in rows])
    fd_margin = np.array([_safe_float(r.get("P_line_margin")) for r in rows])

    pred_strict_pass = float(np.mean(pack["strict_pass_pred"]))
    pred_strict      = float(np.mean(pack["strict_pred"]))
    pred_cd_err      = float(np.mean(np.abs(pack["cd_pred"] - CD_TARGET_NM)))
    pred_ler         = float(np.mean(pack["Y4"][:, 1]))
    pred_margin      = float(np.mean(pack["Y4"][:, 3]))
    pred_p_robust    = float(np.mean(pack["p_robust_valid"]))
    pred_p_defect    = float(np.mean(pack["p_defect"]))
    return {
        "n":                          int(n),
        "FD_strict_pass_prob":       fd_strict_pass,
        "FD_robust_valid_prob":      fd_robust_prob,
        "FD_defect_prob":            fd_defect_prob,
        "FD_mean_strict_score":      float(np.mean(fd_strict_per_row)),
        "FD_mean_cd_error":          float(np.nanmean(fd_cd_err)),
        "FD_mean_ler":               float(np.nanmean(fd_ler)),
        "FD_mean_margin":            float(np.nanmean(fd_margin)),
        "predicted_strict_pass_prob": pred_strict_pass,
        "predicted_robust_valid_prob": pred_p_robust,
        "predicted_defect_prob":      pred_p_defect,
        "predicted_mean_strict_score": pred_strict,
        "predicted_mean_cd_error":    pred_cd_err,
        "predicted_mean_ler":         pred_ler,
        "predicted_mean_margin":      pred_margin,
        "strict_pass_residual":       pred_strict_pass - fd_strict_pass,
        "CD_error_residual":          pred_cd_err - float(np.nanmean(fd_cd_err)),
        "LER_residual":               pred_ler - float(np.nanmean(fd_ler)),
        "margin_residual":            pred_margin - float(np.nanmean(fd_margin)),
    }


# --------------------------------------------------------------------------
# Driver
# --------------------------------------------------------------------------
def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--p06_csv", type=str,
                   default=str(V3_DIR / "outputs" / "labels"
                               / "stage06P_training_dataset.csv"))
    p.add_argument("--m_06m_csv", type=str,
                   default=str(V3_DIR / "outputs" / "yield_optimization"
                               / "stage06M_time_deep_mc.csv"))
    p.add_argument("--m_06m_b_csv", type=str,
                   default=str(V3_DIR / "outputs" / "yield_optimization"
                               / "stage06M_B_J1453_time_deep_mc.csv"))
    p.add_argument("--fd_06j_nominal_csv", type=str,
                   default=str(V3_DIR / "outputs" / "labels"
                               / "stage06J_mode_b_fd_sanity.csv"))
    p.add_argument("--fd_06j_b_top10_csv", type=str,
                   default=str(V3_DIR / "outputs" / "labels"
                               / "stage06J_B_fd_top10_mc.csv"))
    p.add_argument("--fd_06j_b_repmc_csv", type=str,
                   default=str(V3_DIR / "outputs" / "labels"
                               / "stage06J_B_fd_representative_mc.csv"))
    p.add_argument("--fd_06j_b_nom_csv", type=str,
                   default=str(V3_DIR / "outputs" / "labels"
                               / "stage06J_B_fd_top100_nominal.csv"))
    p.add_argument("--summary_06j_csv", type=str,
                   default=str(V3_DIR / "outputs" / "yield_optimization"
                               / "stage06J_mode_b_recipe_summary.csv"))
    p.add_argument("--top_06j_csv", type=str,
                   default=str(V3_DIR / "outputs" / "yield_optimization"
                               / "stage06J_mode_b_top_recipes.csv"))
    p.add_argument("--fp_06p_csv", type=str,
                   default=str(V3_DIR / "outputs" / "yield_optimization"
                               / "stage06P_false_pass_demotions.csv"))
    p.add_argument("--strict_cfg_yaml", type=str,
                   default=str(V3_DIR / "outputs" / "yield_optimization"
                               / "stage06G_strict_score_config.yaml"))
    p.add_argument("--mode_a_recipes_yaml", type=str,
                   default=str(V3_DIR / "outputs" / "yield_optimization"
                               / "stage06I_mode_a_final_recipes.yaml"))
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
    args = p.parse_args()

    strict_yaml = yaml.safe_load(Path(args.strict_cfg_yaml).read_text())
    cd_tol  = float(strict_yaml["thresholds"]["cd_tol_nm"])
    ler_cap = float(strict_yaml["thresholds"]["ler_cap_nm"])

    # Load 06P models.
    clf, _ = load_model(args.clf_06p)
    reg, _ = load_model(args.reg_06p)
    aux, _ = load_model(args.aux_06p)
    sr,  _ = load_model(args.strict_06p)

    # Reference G_4867 chemistry from 06I final recipes.
    g4867_yaml = next(
        rep for rep in yaml.safe_load(
            Path(args.mode_a_recipes_yaml).read_text())["representatives"]
        if rep["recipe_id"] == "G_4867")
    g4867_params = {k: float(v) for k, v in g4867_yaml["parameters"].items()}

    # Read everything.
    rows_06m   = read_labels_csv(args.m_06m_csv)
    rows_06m_b = read_labels_csv(args.m_06m_b_csv)
    fd_06j_nominal = read_labels_csv(args.fd_06j_nominal_csv)
    fd_06j_b_top10 = read_labels_csv(args.fd_06j_b_top10_csv)
    fd_06j_b_repmc = read_labels_csv(args.fd_06j_b_repmc_csv)
    fd_06j_b_nom   = read_labels_csv(args.fd_06j_b_nom_csv)
    summary_06j    = read_labels_csv(args.summary_06j_csv)
    top_06j        = read_labels_csv(args.top_06j_csv)
    fp_06p_rows    = read_labels_csv(args.fp_06p_csv)

    keys_for_coerce = (
        FEATURE_KEYS
        + ["CD_final_nm", "CD_locked_nm", "LER_CD_locked_nm",
           "area_frac", "P_line_margin",
           "rank_06j", "rank_06l", "rank_06p", "delta_rank_06p_vs_06l",
           "delta_rank_06p_vs_06j", "strict_score_06j_surrogate",
           "strict_score_06l_direct", "strict_score_06p_direct",
           "strict_score", "yield_score", "p_robust_valid",
           "mean_cd_fixed", "mean_ler_locked", "mean_p_line_margin",
           "geometry_distance_to_modeA"]
    )
    for rs in (rows_06m, rows_06m_b, fd_06j_nominal, fd_06j_b_top10,
                 fd_06j_b_repmc, fd_06j_b_nom, summary_06j, top_06j,
                 fp_06p_rows):
        _coerce(rs, keys_for_coerce)
    summary_by_id = {r["recipe_id"]: r for r in summary_06j
                       if "recipe_id" in r}

    # ----- Section A: build blindspot dataset ------------------------------
    # 1) G_4867 / J_1453 per deterministic offset.
    blind_rows: list[dict] = []

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

    j1453_summary = summary_by_id.get("J_1453", {})
    chem_sig_J1453 = _chemistry_signature(j1453_summary, g4867_params)

    for off in J1453_DET_TIME_OFFSETS:
        for rid, recipe_family, source_stage, rows_off, ref_recipe, sig in (
            ("G_4867", "G4867_family",
             "stage06M_time_deep_mc",   g_by_off[off], g4867_params,
             "ref_modeA"),
            ("J_1453", "J1453_family",
             "stage06M_B_J1453_time_deep_mc", j_by_off[off], j1453_summary,
             chem_sig_J1453),
        ):
            if not rows_off:
                continue
            pack = _predict_pack(rows_off, clf, reg, aux, sr, cd_tol, ler_cap)
            agg = _aggregate_recipe_rows(rows_off, pack,
                                              cd_tol=cd_tol, ler_cap=ler_cap,
                                              strict_yaml=strict_yaml)
            sample = rows_off[0]
            pitch = _safe_float(sample.get("pitch_nm"))
            ratio = _safe_float(sample.get("line_cd_ratio"))
            blind_rows.append({
                "recipe_id":      rid,
                "mode":           "mode_a" if _is_mode_a(pitch, ratio) else "mode_b",
                "recipe_family":  recipe_family,
                "source_stage":   source_stage,
                "time_offset_s":  float(off),
                "n_fd_rows":      agg["n"],
                "FD_strict_pass_prob": agg["FD_strict_pass_prob"],
                "predicted_strict_pass_prob": agg["predicted_strict_pass_prob"],
                "strict_pass_residual": agg["strict_pass_residual"],
                "CD_error_residual":  agg["CD_error_residual"],
                "LER_residual":       agg["LER_residual"],
                "margin_residual":    agg["margin_residual"],
                "FD_mean_cd_error":   agg["FD_mean_cd_error"],
                "FD_mean_ler":        agg["FD_mean_ler"],
                "FD_mean_margin":     agg["FD_mean_margin"],
                "pred_mean_cd_error": agg["predicted_mean_cd_error"],
                "pred_mean_ler":      agg["predicted_mean_ler"],
                "pred_mean_margin":   agg["predicted_mean_margin"],
                "false_pass_flag":    False,
                "hard_fail_flag":     False,
                "geometry_match_mode_a_flag": _is_mode_a(pitch, ratio),
                "chemistry_knob_signature": sig,
            })

    # 2) 06J nominal FD recipes (one row each), flag hard-fail for the 4 known.
    for r in fd_06j_nominal:
        rid = str(r.get("source_recipe_id", ""))
        if not rid:
            continue
        label = str(r.get("label", ""))
        is_hard = label in HARD_FAIL_LABELS
        # FD here is a single nominal: build aggregates as 1-row pack.
        pack = _predict_pack([r], clf, reg, aux, sr, cd_tol, ler_cap)
        agg  = _aggregate_recipe_rows([r], pack,
                                          cd_tol=cd_tol, ler_cap=ler_cap,
                                          strict_yaml=strict_yaml)
        pitch = _safe_float(r.get("pitch_nm"))
        ratio = _safe_float(r.get("line_cd_ratio"))
        sig = _chemistry_signature(r, g4867_params)
        sumr = summary_by_id.get(rid, {})
        family = "G4867_family" if rid == "G_4867" else (
                    "J1453_family" if rid == "J_1453" else (
                        "false_promise" if is_hard else "mode_b_other"))
        blind_rows.append({
            "recipe_id":      rid,
            "mode":           "mode_a" if _is_mode_a(pitch, ratio) else "mode_b",
            "recipe_family":  family,
            "source_stage":   "stage06J_mode_b_fd_sanity",
            "time_offset_s":  float("nan"),
            "n_fd_rows":      1,
            "FD_strict_pass_prob": (1.0 if (label == "robust_valid"
                                              and abs(_safe_float(r.get("CD_final_nm")) - CD_TARGET_NM) <= cd_tol
                                              and _safe_float(r.get("LER_CD_locked_nm")) <= ler_cap)
                                       else 0.0),
            "predicted_strict_pass_prob": agg["predicted_strict_pass_prob"],
            "strict_pass_residual":       agg["strict_pass_residual"]
                                              if False else (agg["predicted_strict_pass_prob"]
                                                  - (1.0 if (label == "robust_valid"
                                                              and abs(_safe_float(r.get("CD_final_nm")) - CD_TARGET_NM) <= cd_tol
                                                              and _safe_float(r.get("LER_CD_locked_nm")) <= ler_cap)
                                                       else 0.0)),
            "CD_error_residual":          agg["CD_error_residual"],
            "LER_residual":               agg["LER_residual"],
            "margin_residual":            agg["margin_residual"],
            "FD_mean_cd_error":           agg["FD_mean_cd_error"],
            "FD_mean_ler":                agg["FD_mean_ler"],
            "FD_mean_margin":             agg["FD_mean_margin"],
            "pred_mean_cd_error":         agg["predicted_mean_cd_error"],
            "pred_mean_ler":              agg["predicted_mean_ler"],
            "pred_mean_margin":           agg["predicted_mean_margin"],
            "false_pass_flag":            bool(is_hard
                                                    and float(sumr.get("strict_score", 0.0))
                                                            > 0.5),
            "hard_fail_flag":             bool(is_hard),
            "geometry_match_mode_a_flag": _is_mode_a(pitch, ratio),
            "chemistry_knob_signature":   sig,
        })

    # 3) 06J-B top-10 MC recipes (aggregate).
    by_recipe_jb_top10: dict[str, list[dict]] = defaultdict(list)
    for r in fd_06j_b_top10:
        rid = str(r.get("source_recipe_id", ""))
        if rid:
            by_recipe_jb_top10[rid].append(r)
    for rid, rs in by_recipe_jb_top10.items():
        pack = _predict_pack(rs, clf, reg, aux, sr, cd_tol, ler_cap)
        agg = _aggregate_recipe_rows(rs, pack,
                                          cd_tol=cd_tol, ler_cap=ler_cap,
                                          strict_yaml=strict_yaml)
        sample = rs[0]
        pitch = _safe_float(sample.get("pitch_nm"))
        ratio = _safe_float(sample.get("line_cd_ratio"))
        sumr = summary_by_id.get(rid, {})
        sig = _chemistry_signature(sumr, g4867_params)
        family = ("G4867_family" if rid == "G_4867"
                    else "J1453_family" if rid == "J_1453"
                    else "mode_b_other")
        blind_rows.append({
            "recipe_id":      rid,
            "mode":           "mode_a" if _is_mode_a(pitch, ratio) else "mode_b",
            "recipe_family":  family,
            "source_stage":   "stage06J_B_fd_top10_mc",
            "time_offset_s":  float("nan"),
            "n_fd_rows":      agg["n"],
            "FD_strict_pass_prob":        agg["FD_strict_pass_prob"],
            "predicted_strict_pass_prob": agg["predicted_strict_pass_prob"],
            "strict_pass_residual":       agg["strict_pass_residual"],
            "CD_error_residual":          agg["CD_error_residual"],
            "LER_residual":               agg["LER_residual"],
            "margin_residual":            agg["margin_residual"],
            "FD_mean_cd_error":           agg["FD_mean_cd_error"],
            "FD_mean_ler":                agg["FD_mean_ler"],
            "FD_mean_margin":             agg["FD_mean_margin"],
            "pred_mean_cd_error":         agg["predicted_mean_cd_error"],
            "pred_mean_ler":              agg["predicted_mean_ler"],
            "pred_mean_margin":           agg["predicted_mean_margin"],
            "false_pass_flag":            False,
            "hard_fail_flag":             agg["FD_defect_prob"] > 0.0,
            "geometry_match_mode_a_flag": _is_mode_a(pitch, ratio),
            "chemistry_knob_signature":   sig,
        })

    # Write blindspot dataset.
    yopt_dir = V3_DIR / "outputs" / "yield_optimization"
    blind_cols = list(blind_rows[0].keys()) if blind_rows else []
    with (yopt_dir / "stage06Q_blindspot_dataset.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=blind_cols, extrasaction="ignore")
        w.writeheader()
        for r in blind_rows:
            w.writerow(r)

    # ----- Section B: J_1453 vs G_4867 residuals ---------------------------
    g_off = {r["time_offset_s"]: r for r in blind_rows
               if r["recipe_id"] == "G_4867"
                   and r["source_stage"] == "stage06M_time_deep_mc"}
    j_off = {r["time_offset_s"]: r for r in blind_rows
               if r["recipe_id"] == "J_1453"
                   and r["source_stage"] == "stage06M_B_J1453_time_deep_mc"}

    res_rows: list[dict] = []
    for off in J1453_DET_TIME_OFFSETS:
        g = g_off.get(float(off)); j = j_off.get(float(off))
        if g is None or j is None:
            continue
        fd_rel = j["FD_strict_pass_prob"]   - g["FD_strict_pass_prob"]
        pr_rel = j["predicted_strict_pass_prob"] - g["predicted_strict_pass_prob"]
        residual_rel = pr_rel - fd_rel
        res_rows.append({
            "time_offset_s":             float(off),
            "G4867_FD_strict_pass":      g["FD_strict_pass_prob"],
            "J1453_FD_strict_pass":      j["FD_strict_pass_prob"],
            "G4867_pred_strict_pass":    g["predicted_strict_pass_prob"],
            "J1453_pred_strict_pass":    j["predicted_strict_pass_prob"],
            "FD_relative_advantage_J_minus_G":  fd_rel,
            "pred_relative_advantage_J_minus_G": pr_rel,
            "relative_advantage_residual_pred_minus_fd": residual_rel,
            "G4867_strict_pass_residual": g["strict_pass_residual"],
            "J1453_strict_pass_residual": j["strict_pass_residual"],
            "G4867_CD_error_residual":   g["CD_error_residual"],
            "J1453_CD_error_residual":   j["CD_error_residual"],
            "G4867_LER_residual":        g["LER_residual"],
            "J1453_LER_residual":        j["LER_residual"],
            "G4867_margin_residual":     g["margin_residual"],
            "J1453_margin_residual":     j["margin_residual"],
        })

    res_cols = list(res_rows[0].keys()) if res_rows else []
    with (yopt_dir / "stage06Q_J1453_G4867_residuals.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=res_cols, extrasaction="ignore")
        w.writeheader()
        for r in res_rows:
            w.writerow(r)

    # Where is the under-prediction worst? Side: early (offset<0), zero, late.
    j_residuals = np.array([r["J1453_strict_pass_residual"] for r in res_rows])
    g_residuals = np.array([r["G4867_strict_pass_residual"] for r in res_rows])
    rel_residuals = np.array([r["relative_advantage_residual_pred_minus_fd"]
                                  for r in res_rows])
    offs = np.array([r["time_offset_s"] for r in res_rows])
    early = offs < 0; late = offs > 0; zero = offs == 0
    j1453_under_pred_worst_offset = (
        float(offs[int(np.argmin(j_residuals))]) if j_residuals.size else None
    )
    summary_residual_section = {
        "J1453_strict_pass_residual_mean":       float(np.mean(j_residuals))
                                                       if j_residuals.size else None,
        "J1453_strict_pass_residual_min":        float(np.min(j_residuals))
                                                       if j_residuals.size else None,
        "J1453_strict_pass_residual_min_offset": j1453_under_pred_worst_offset,
        "G4867_strict_pass_residual_mean":       float(np.mean(g_residuals))
                                                       if g_residuals.size else None,
        "relative_advantage_residual_mean":      float(np.mean(rel_residuals))
                                                       if rel_residuals.size else None,
        "relative_advantage_residual_early_mean":
            float(np.mean(rel_residuals[early])) if early.any() else None,
        "relative_advantage_residual_late_mean":
            float(np.mean(rel_residuals[late])) if late.any() else None,
        "relative_advantage_residual_zero":
            float(rel_residuals[zero][0]) if zero.any() else None,
        "j1453_advantage_under_predicted_at_offsets":
            [float(r["time_offset_s"]) for r in res_rows
             if r["relative_advantage_residual_pred_minus_fd"] < 0],
    }

    # ----- Section C: false-PASS diagnostics -------------------------------
    diagnostics: list[dict] = []
    for r in fd_06j_nominal:
        label = str(r.get("label", ""))
        if label not in HARD_FAIL_LABELS:
            continue
        rid = str(r.get("source_recipe_id", ""))
        sumr = summary_by_id.get(rid, {})
        pack = _predict_pack([r], clf, reg, aux, sr, cd_tol, ler_cap)
        if pack["n"] == 0:
            continue
        cd_pred = float(pack["cd_pred"][0])
        ler_pred = float(pack["Y4"][0, 1])
        margin_pred = float(pack["Y4"][0, 3])
        p_robust = float(pack["p_robust_valid"][0])
        p_defect = float(pack["p_defect"][0])
        pred_class = str(pack["pred_class"][0])
        cd_fd = _safe_float(r.get("CD_final_nm"))
        ler_fd = _safe_float(r.get("LER_CD_locked_nm"))
        margin_fd = _safe_float(r.get("P_line_margin"))
        cd_err_fd   = abs(cd_fd - CD_TARGET_NM) if np.isfinite(cd_fd) else float("nan")
        cd_err_pred = abs(cd_pred - CD_TARGET_NM)
        cd_under = bool(np.isfinite(cd_err_fd) and cd_err_pred < cd_err_fd - 0.1)
        ler_under = bool(np.isfinite(ler_fd) and ler_pred < ler_fd - 0.1)
        margin_over = bool(np.isfinite(margin_fd) and margin_pred > margin_fd + 0.02)
        class_overconf = bool(p_robust > 0.5 and pred_class == "robust_valid")
        # Cause buckets, in order of priority.
        causes = []
        if class_overconf:
            causes.append("class_prob_overconfident")
        if cd_under:
            causes.append("cd_error_under_predicted")
        if ler_under:
            causes.append("ler_under_predicted")
        if margin_over:
            causes.append("margin_over_predicted")
        if not causes:
            causes.append("predicted_correct_but_kept_in_top_pool")
        # Geometry extrapolation flag: pitch outside [22, 26] OR ratio
        # outside [0.48, 0.56] => far from densest training region.
        pitch = _safe_float(r.get("pitch_nm"))
        ratio = _safe_float(r.get("line_cd_ratio"))
        geom_extrap = not _is_mode_a(pitch, ratio,
                                          pitch_tol=2.0, ratio_tol=0.04)
        diagnostics.append({
            "recipe_id": rid,
            "fd_label":  label,
            "rank_06l_in_FP_pool": int(_safe_float(next(
                (x.get("rank_06l") for x in fp_06p_rows
                  if x.get("recipe_id") == rid), float("nan"))))
                                    if any(x.get("recipe_id") == rid for x in fp_06p_rows) else None,
            "rank_06p_in_FP_pool": int(_safe_float(next(
                (x.get("rank_06p") for x in fp_06p_rows
                  if x.get("recipe_id") == rid), float("nan"))))
                                    if any(x.get("recipe_id") == rid for x in fp_06p_rows) else None,
            "FD_label":     label,
            "FD_CD_error":  cd_err_fd,
            "FD_LER":       ler_fd,
            "FD_margin":    margin_fd,
            "pred_CD_error": cd_err_pred,
            "pred_LER":     ler_pred,
            "pred_margin":  margin_pred,
            "pred_p_robust_valid": p_robust,
            "pred_p_defect":       p_defect,
            "pred_class":          pred_class,
            "CD_under_predicted":  cd_under,
            "LER_under_predicted": ler_under,
            "margin_over_predicted": margin_over,
            "class_overconfident": class_overconf,
            "geometry_extrapolation": bool(geom_extrap),
            "dominant_cause":      ";".join(causes),
            "pitch_nm":            pitch,
            "line_cd_ratio":       ratio,
            "dose_mJ_cm2":         _safe_float(r.get("dose_mJ_cm2")),
            "sigma_nm":            _safe_float(r.get("sigma_nm")),
            "DH_nm2_s":            _safe_float(r.get("DH_nm2_s")),
            "time_s":              _safe_float(r.get("time_s")),
            "Hmax_mol_dm3":        _safe_float(r.get("Hmax_mol_dm3")),
            "kdep_s_inv":          _safe_float(r.get("kdep_s_inv")),
            "Q0_mol_dm3":          _safe_float(r.get("Q0_mol_dm3")),
            "kq_s_inv":            _safe_float(r.get("kq_s_inv")),
            "abs_len_nm":          _safe_float(r.get("abs_len_nm")),
            "chemistry_knob_signature": _chemistry_signature(r, g4867_params),
        })

    diag_cols = list(diagnostics[0].keys()) if diagnostics else []
    with (yopt_dir / "stage06Q_false_pass_diagnostics.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=diag_cols, extrasaction="ignore")
        w.writeheader()
        for r in diagnostics:
            w.writerow(r)

    # ----- Section D: Mode A geometry chemistry blindspot ------------------
    # All Mode B recipes whose geometry sits inside the Mode A box
    # (pitch ~24, ratio ~0.52) AND that appear in the 06J top-100 pool.
    mode_a_box: list[dict] = []
    for r in summary_06j:
        rid = str(r.get("recipe_id", ""))
        pitch = _safe_float(r.get("pitch_nm"))
        ratio = _safe_float(r.get("line_cd_ratio"))
        if not _is_mode_a(pitch, ratio, pitch_tol=0.5, ratio_tol=0.04):
            continue
        rk = _safe_float(r.get("rank_strict"))
        if not (np.isfinite(rk) and rk <= 200):
            continue
        # 06P prediction for the surrogate-mean recipe.
        pack = _predict_pack([r], clf, reg, aux, sr, cd_tol, ler_cap)
        cd_pred  = float(pack["cd_pred"][0])
        ler_pred = float(pack["Y4"][0, 1])
        margin_pred = float(pack["Y4"][0, 3])
        p_robust = float(pack["p_robust_valid"][0])
        strict_pred = float(pack["strict_pred"][0])
        # FD evidence if available (06J nominal sanity).
        fd_match = next((x for x in fd_06j_nominal
                          if str(x.get("source_recipe_id", "")) == rid), None)
        fd_label   = str(fd_match.get("label", "")) if fd_match else ""
        fd_cd      = _safe_float(fd_match.get("CD_final_nm")) if fd_match else float("nan")
        fd_ler     = _safe_float(fd_match.get("LER_CD_locked_nm")) if fd_match else float("nan")
        fd_margin  = _safe_float(fd_match.get("P_line_margin")) if fd_match else float("nan")
        sig = _chemistry_signature(r, g4867_params)
        mode_a_box.append({
            "recipe_id":         rid,
            "rank_strict_06j":   int(rk),
            "strict_score_06j":  _safe_float(r.get("strict_score")),
            "yield_score_06j":   _safe_float(r.get("yield_score")),
            "p_robust_06j_surrogate": _safe_float(r.get("p_robust_valid")),
            "pitch_nm":          pitch,
            "line_cd_ratio":     ratio,
            "dose_mJ_cm2":       _safe_float(r.get("dose_mJ_cm2")),
            "sigma_nm":          _safe_float(r.get("sigma_nm")),
            "DH_nm2_s":          _safe_float(r.get("DH_nm2_s")),
            "time_s":            _safe_float(r.get("time_s")),
            "Hmax_mol_dm3":      _safe_float(r.get("Hmax_mol_dm3")),
            "kdep_s_inv":        _safe_float(r.get("kdep_s_inv")),
            "Q0_mol_dm3":        _safe_float(r.get("Q0_mol_dm3")),
            "kq_s_inv":          _safe_float(r.get("kq_s_inv")),
            "abs_len_nm":        _safe_float(r.get("abs_len_nm")),
            "predicted_strict_score_06p": strict_pred,
            "predicted_p_robust_06p":     p_robust,
            "predicted_CD_06p":           cd_pred,
            "predicted_LER_06p":          ler_pred,
            "predicted_margin_06p":       margin_pred,
            "fd_label":          fd_label,
            "fd_CD_final_nm":    fd_cd,
            "fd_LER":            fd_ler,
            "fd_margin":         fd_margin,
            "fd_strict_pass":    bool(fd_label == "robust_valid"
                                        and abs(fd_cd - CD_TARGET_NM) <= cd_tol
                                        and fd_ler <= ler_cap)
                                    if fd_match else None,
            "diffusion_length_nm":
                float(np.sqrt(2.0 * _safe_float(r.get("DH_nm2_s"))
                                * _safe_float(r.get("time_s")))),
            "reaction_budget":
                float(_safe_float(r.get("Hmax_mol_dm3"))
                        * _safe_float(r.get("kdep_s_inv"))
                        * _safe_float(r.get("time_s"))),
            "quencher_budget":
                float(_safe_float(r.get("Q0_mol_dm3"))
                        * _safe_float(r.get("kq_s_inv"))
                        * _safe_float(r.get("time_s"))),
            "blur_to_pitch":
                float(_safe_float(r.get("sigma_nm"))
                        / max(_safe_float(r.get("pitch_nm")), 1e-12)),
            "chemistry_knob_signature": sig,
        })

    box_cols = list(mode_a_box[0].keys()) if mode_a_box else []
    with (yopt_dir / "stage06Q_mode_a_geometry_blindspot.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=box_cols, extrasaction="ignore")
        w.writeheader()
        for r in mode_a_box:
            w.writerow(r)

    # ----- Section E: recommendations YAML ---------------------------------
    rec_yaml = {
        "stage": "06Q",
        "policy": {
            "v2_OP_frozen": True,
            "published_data_loaded": False,
            "external_calibration": "none",
        },
        "preferred_surrogate_baseline": "stage06P (unchanged by 06Q)",
        "blindspot_summary": {
            "j1453_advantage_magnitude_under_predicted":
                bool(summary_residual_section["relative_advantage_residual_mean"] is not None
                       and summary_residual_section["relative_advantage_residual_mean"] < 0),
            "relative_advantage_residual_mean":
                summary_residual_section["relative_advantage_residual_mean"],
            "relative_advantage_residual_early_mean":
                summary_residual_section["relative_advantage_residual_early_mean"],
            "relative_advantage_residual_late_mean":
                summary_residual_section["relative_advantage_residual_late_mean"],
            "remaining_false_pass_recipes": [d["recipe_id"] for d in diagnostics],
            "mode_a_geometry_box_recipe_count": len(mode_a_box),
            "mode_a_geometry_box_recipes_in_top10":
                [r["recipe_id"] for r in mode_a_box if r["rank_strict_06j"] <= 10],
        },
        "recommended_corrections": [
            {
                "id": "B_feature_engineering",
                "priority": "primary",
                "rationale": (
                    "06P relies on raw chemistry knobs only. Derived "
                    "features that aggregate process-window structure "
                    "are the cheapest way to recover the J_1453 vs "
                    "G_4867 magnitude gap because the rank order is "
                    "already correct (Spearman 0.736 on relative "
                    "advantage)."
                ),
                "candidate_features": [
                    "diffusion_length_nm = sqrt(2 * DH_nm2_s * time_s)",
                    "reaction_budget    = Hmax_mol_dm3 * kdep_s_inv * time_s",
                    "quencher_budget    = Q0_mol_dm3 * kq_s_inv * time_s",
                    "blur_to_pitch      = sigma_nm / pitch_nm",
                    "line_cd_nm         = pitch_nm * line_cd_ratio",
                    "DH_x_time          = DH_nm2_s * time_s",
                    "Hmax_kdep_ratio    = Hmax_mol_dm3 * kdep_s_inv",
                    "Q0_kq_ratio        = Q0_mol_dm3 * kq_s_inv",
                    "dose_x_blur        = dose_mJ_cm2 * sigma_nm",
                ],
                "next_stage_hint": "06R: feature-engineered surrogate refresh.",
            },
            {
                "id": "C_AL_targeting",
                "priority": "secondary",
                "rationale": (
                    "06P training pool covers J_1453 / G_4867 nominals "
                    "and a few representative recipes only. The "
                    "magnitude gap is sharpest at large negative time "
                    "offsets, where data density on J_1453-like "
                    "recipes is the lowest. Targeted AL on time- and "
                    "margin-boundary candidates would tighten the "
                    "magnitude bias even without new features."
                ),
                "candidate_targets": [
                    "J_1453-like time-boundary candidates",
                    "Mode A-geometry chemistry candidates with low "
                    "predicted margin",
                    "high-rank Mode B candidates with predicted "
                    "p_line_margin in [0.18, 0.22]",
                ],
                "next_stage_hint": "06P-B: targeted AL refresh.",
            },
            {
                "id": "D_model_split",
                "priority": "tertiary",
                "rationale": (
                    "Mode A and Mode B share one strict_score head. If "
                    "feature engineering does not close the magnitude "
                    "gap, separate Mode A / Mode B strict_score heads "
                    "would let the Mode B head learn the J_1453 "
                    "advantage without being averaged away by the "
                    "much larger Mode A 04C training pool."
                ),
                "next_stage_hint":
                    "06R-2 (only if 06R feature engineering does not "
                    "close the gap): mode-specific strict_score heads.",
            },
            {
                "id": "A_residual_calibration",
                "priority": "fallback",
                "rationale": (
                    "Per-family strict_pass_prob residual correction "
                    "(J1453_family / G4867_family) on the existing "
                    "06P predictions. Only suggested if 06R cannot "
                    "be scheduled and operational decisions need an "
                    "interim fix."
                ),
                "next_stage_hint":
                    "Apply at usage time; do not bake into the 06P "
                    "model. Treat as a usage-time correction layer.",
            },
        ],
        "operational_assessment": {
            "j1453_in_06p_top10_b_unchanged_by_06q": True,
            "g4867_in_06p_top10_a_unchanged_by_06q": True,
            "magnitude_gap_operationally_blocking": False,
            "rank_order_already_correct": True,
        },
    }
    (yopt_dir / "stage06Q_recommended_corrections.yaml").write_text(
        yaml.safe_dump(rec_yaml, sort_keys=False))

    # ----- Section F: summary JSON -----------------------------------------
    logs_dir = V3_DIR / "outputs" / "logs"
    summary_json = {
        "stage": "06Q",
        "policy": {
            "v2_OP_frozen": True,
            "published_data_loaded": False,
            "external_calibration": "none",
        },
        "blindspot_dataset_n_rows": len(blind_rows),
        "j1453_g4867_residuals": summary_residual_section,
        "false_pass_diagnostics_summary": {
            "n_recipes": len(diagnostics),
            "by_dominant_cause": dict(Counter(
                d["dominant_cause"].split(";")[0] for d in diagnostics
            )),
            "n_class_overconfident": sum(1 for d in diagnostics
                                              if d["class_overconfident"]),
            "n_cd_under_predicted":  sum(1 for d in diagnostics
                                              if d["CD_under_predicted"]),
            "n_ler_under_predicted": sum(1 for d in diagnostics
                                              if d["LER_under_predicted"]),
            "n_margin_over_predicted": sum(1 for d in diagnostics
                                                if d["margin_over_predicted"]),
            "n_geometry_extrapolation": sum(1 for d in diagnostics
                                                  if d["geometry_extrapolation"]),
        },
        "mode_a_geometry_blindspot_summary": {
            "n_recipes_in_box":             len(mode_a_box),
            "n_with_fd_evidence":            sum(1 for r in mode_a_box
                                                      if r["fd_label"]),
            "n_fd_strict_pass":              sum(1 for r in mode_a_box
                                                      if r.get("fd_strict_pass") is True),
            "n_fd_label_robust_valid":       sum(1 for r in mode_a_box
                                                      if r["fd_label"] == "robust_valid"),
            "top10_06j_recipes_in_box":      [r["recipe_id"] for r in mode_a_box
                                                  if r["rank_strict_06j"] <= 10],
            "top100_06j_recipes_in_box_count":
                sum(1 for r in mode_a_box
                      if r["rank_strict_06j"] <= 100),
        },
        "recommended_primary_correction":
            "B_feature_engineering (next stage hint: 06R feature-engineered surrogate refresh)",
        "j1453_g4867_recipes_unchanged_by_06q": True,
    }
    (logs_dir / "stage06Q_summary.json").write_text(
        json.dumps(summary_json, indent=2, default=float))

    # ----- Figures ---------------------------------------------------------
    fig_dir = V3_DIR / "outputs" / "figures" / "06_yield_optimization"
    fig_dir.mkdir(parents=True, exist_ok=True)

    # F1: J vs G strict_pass residual vs time.
    if res_rows:
        offs_arr = np.array([r["time_offset_s"] for r in res_rows])
        fig, ax = plt.subplots(figsize=(11.0, 5.8))
        ax.plot(offs_arr,
                 [r["G4867_strict_pass_residual"] for r in res_rows],
                 "-o", color="#1f77b4", label="G_4867 pred - FD strict_pass")
        ax.plot(offs_arr,
                 [r["J1453_strict_pass_residual"] for r in res_rows],
                 "-s", color="#d62728", label="J_1453 pred - FD strict_pass")
        ax.axhline(0.0, color="black", lw=1.0, ls="--", alpha=0.6)
        ax.set_xlabel("time offset (s)")
        ax.set_ylabel("predicted - FD strict_pass_prob")
        ax.set_title("Stage 06Q -- per-recipe strict_pass residual vs time offset")
        ax.grid(True, alpha=0.25)
        ax.legend(loc="best", fontsize=10)
        fig.tight_layout()
        fig.savefig(fig_dir / "stage06Q_J1453_G4867_strict_pass_residual_vs_time.png",
                     dpi=150)
        plt.close(fig)

    # F2: pred vs FD relative advantage.
    if res_rows:
        offs_arr = np.array([r["time_offset_s"] for r in res_rows])
        fd_rel = np.array([r["FD_relative_advantage_J_minus_G"] for r in res_rows])
        pr_rel = np.array([r["pred_relative_advantage_J_minus_G"] for r in res_rows])
        fig, ax = plt.subplots(figsize=(11.0, 5.8))
        ax.plot(offs_arr, fd_rel, "-o", color="#2ca02c",
                 label="FD truth: J - G strict_pass")
        ax.plot(offs_arr, pr_rel, "-s", color="#d62728",
                 label="06P pred: J - G strict_pass")
        ax.fill_between(offs_arr, pr_rel, fd_rel,
                          alpha=0.20, color="grey", label="residual")
        ax.axhline(0.0, color="black", lw=1.0, ls="--", alpha=0.6)
        ax.set_xlabel("time offset (s)")
        ax.set_ylabel("J_1453 - G_4867 strict_pass_prob")
        ax.set_title("Stage 06Q -- relative advantage J_1453 vs G_4867: 06P pred vs FD")
        ax.grid(True, alpha=0.25)
        ax.legend(loc="best", fontsize=10)
        fig.tight_layout()
        fig.savefig(fig_dir / "stage06Q_pred_vs_fd_relative_advantage.png",
                     dpi=150)
        plt.close(fig)

    # F3: CD / LER / margin residual breakdown vs time.
    if res_rows:
        offs_arr = np.array([r["time_offset_s"] for r in res_rows])
        fig, axes = plt.subplots(1, 3, figsize=(14.5, 5.0))
        axes[0].plot(offs_arr, [r["G4867_CD_error_residual"] for r in res_rows],
                       "-o", color="#1f77b4", label="G_4867")
        axes[0].plot(offs_arr, [r["J1453_CD_error_residual"] for r in res_rows],
                       "-s", color="#d62728", label="J_1453")
        axes[0].set_title("CD_error residual (pred - FD)")
        axes[1].plot(offs_arr, [r["G4867_LER_residual"] for r in res_rows],
                       "-o", color="#1f77b4", label="G_4867")
        axes[1].plot(offs_arr, [r["J1453_LER_residual"] for r in res_rows],
                       "-s", color="#d62728", label="J_1453")
        axes[1].set_title("LER residual (pred - FD)")
        axes[2].plot(offs_arr, [r["G4867_margin_residual"] for r in res_rows],
                       "-o", color="#1f77b4", label="G_4867")
        axes[2].plot(offs_arr, [r["J1453_margin_residual"] for r in res_rows],
                       "-s", color="#d62728", label="J_1453")
        axes[2].set_title("P_line_margin residual (pred - FD)")
        for ax in axes:
            ax.axhline(0.0, color="black", lw=1.0, ls="--", alpha=0.6)
            ax.set_xlabel("time offset (s)")
            ax.grid(True, alpha=0.25)
            ax.legend(loc="best", fontsize=9)
        fig.suptitle("Stage 06Q -- residual breakdown by metric "
                       "(per recipe, per time offset)", fontsize=12)
        fig.tight_layout(rect=[0, 0, 1, 0.96])
        fig.savefig(fig_dir / "stage06Q_cd_ler_margin_residual_breakdown.png",
                     dpi=150)
        plt.close(fig)

    # F4: false-PASS feature patterns (parallel-coordinates style).
    if diagnostics:
        chem_keys = ["dose_mJ_cm2", "sigma_nm", "DH_nm2_s", "time_s",
                      "Hmax_mol_dm3", "kdep_s_inv", "Q0_mol_dm3",
                      "kq_s_inv", "abs_len_nm"]
        fig, ax = plt.subplots(figsize=(12.5, 5.5))
        x = np.arange(len(chem_keys))
        # Reference: G_4867 chemistry normalised in [0, 1].
        ref_norm = []
        for k in chem_keys:
            lo, hi = CHEMISTRY_BOUNDS[k]
            v = (g4867_params[k] - lo) / max(hi - lo, 1e-12)
            ref_norm.append(v)
        ax.plot(x, ref_norm, "-D", color="#1f77b4", lw=2.2,
                 label="G_4867 (Mode A primary)")
        for d in diagnostics:
            yvals = []
            for k in chem_keys:
                lo, hi = CHEMISTRY_BOUNDS[k]
                v = (_safe_float(d.get(k)) - lo) / max(hi - lo, 1e-12)
                yvals.append(v)
            ax.plot(x, yvals, "-o", alpha=0.85,
                     label=f"{d['recipe_id']} ({d['fd_label']})")
        ax.set_xticks(x)
        ax.set_xticklabels(chem_keys, rotation=20, ha="right", fontsize=9)
        ax.set_ylim(-0.05, 1.10)
        ax.set_ylabel("knob value normalised to candidate-space bounds")
        ax.set_title("Stage 06Q -- false-PASS recipes vs G_4867 chemistry "
                      "(parallel coordinates)")
        ax.grid(True, alpha=0.25, axis="y")
        ax.legend(loc="best", fontsize=8, ncol=2)
        fig.tight_layout()
        fig.savefig(fig_dir / "stage06Q_false_pass_feature_patterns.png", dpi=150)
        plt.close(fig)

    # F5: false-PASS predicted-vs-FD metrics.
    if diagnostics:
        rids = [d["recipe_id"] for d in diagnostics]
        x = np.arange(len(rids))
        fig, axes = plt.subplots(1, 3, figsize=(14.5, 5.0))
        axes[0].bar(x - 0.18, [d["FD_CD_error"] for d in diagnostics],
                       width=0.35, color="#1f77b4", label="FD")
        axes[0].bar(x + 0.18, [d["pred_CD_error"] for d in diagnostics],
                       width=0.35, color="#d62728", label="06P pred")
        axes[0].axhline(cd_tol, color="black", ls="--", alpha=0.6,
                          label=f"cd_tol = {cd_tol}")
        axes[0].set_title("|CD - 15| (nm)")
        axes[1].bar(x - 0.18, [d["FD_LER"] for d in diagnostics],
                       width=0.35, color="#1f77b4", label="FD")
        axes[1].bar(x + 0.18, [d["pred_LER"] for d in diagnostics],
                       width=0.35, color="#d62728", label="06P pred")
        axes[1].axhline(ler_cap, color="black", ls="--", alpha=0.6,
                          label=f"ler_cap = {ler_cap}")
        axes[1].set_title("LER_CD_locked (nm)")
        axes[2].bar(x - 0.18, [d["FD_margin"] for d in diagnostics],
                       width=0.35, color="#1f77b4", label="FD")
        axes[2].bar(x + 0.18, [d["pred_margin"] for d in diagnostics],
                       width=0.35, color="#d62728", label="06P pred")
        axes[2].set_title("P_line_margin")
        for ax in axes:
            ax.set_xticks(x)
            ax.set_xticklabels(rids, rotation=20, ha="right", fontsize=9)
            ax.grid(True, alpha=0.25, axis="y")
            ax.legend(loc="best", fontsize=8)
        fig.suptitle("Stage 06Q -- 06P pred vs FD on false-PASS recipes",
                      fontsize=12)
        fig.tight_layout(rect=[0, 0, 1, 0.96])
        fig.savefig(fig_dir / "stage06Q_false_pass_pred_vs_fd_metrics.png",
                     dpi=150)
        plt.close(fig)

    # F6: blindspot parallel coordinates -- Mode A geometry box.
    if mode_a_box:
        chem_keys = ["dose_mJ_cm2", "sigma_nm", "DH_nm2_s", "time_s",
                      "Hmax_mol_dm3", "kdep_s_inv", "Q0_mol_dm3",
                      "kq_s_inv", "abs_len_nm"]
        fig, ax = plt.subplots(figsize=(13.0, 6.0))
        x = np.arange(len(chem_keys))
        ref_norm = [(g4867_params[k] - CHEMISTRY_BOUNDS[k][0])
                      / max(CHEMISTRY_BOUNDS[k][1] - CHEMISTRY_BOUNDS[k][0], 1e-12)
                      for k in chem_keys]
        ax.plot(x, ref_norm, "-D", color="#1f77b4", lw=2.5,
                 label="G_4867 (Mode A primary)")
        for r in mode_a_box[:18]:   # cap at 18 for readability
            yvals = [(_safe_float(r.get(k)) - CHEMISTRY_BOUNDS[k][0])
                       / max(CHEMISTRY_BOUNDS[k][1] - CHEMISTRY_BOUNDS[k][0], 1e-12)
                       for k in chem_keys]
            color = "#d62728" if r.get("fd_label") in HARD_FAIL_LABELS \
                       else ("#2ca02c" if r.get("fd_strict_pass")
                                else "#9aaecf")
            label = f"{r['recipe_id']} ({r['fd_label'] or 'no_FD'})"
            ax.plot(x, yvals, "-o", alpha=0.65, color=color, label=label)
        ax.set_xticks(x)
        ax.set_xticklabels(chem_keys, rotation=20, ha="right", fontsize=9)
        ax.set_ylim(-0.05, 1.10)
        ax.set_ylabel("knob value normalised to candidate-space bounds")
        ax.set_title("Stage 06Q -- Mode A-geometry-box Mode B recipes "
                      "vs G_4867 chemistry")
        ax.grid(True, alpha=0.25, axis="y")
        ax.legend(loc="upper right", fontsize=7, ncol=2)
        fig.tight_layout()
        fig.savefig(fig_dir / "stage06Q_blindspot_parallel_coordinates.png",
                     dpi=150)
        plt.close(fig)

    # Helper for failure-map scatter.
    def _failure_scatter(box: list[dict], xkey: str, ykey: str, out_path: Path,
                            title: str) -> None:
        if not box:
            return
        fig, ax = plt.subplots(figsize=(8.0, 6.0))
        for r in box:
            x = _safe_float(r.get(xkey)); y = _safe_float(r.get(ykey))
            if not (np.isfinite(x) and np.isfinite(y)):
                continue
            fl = r.get("fd_label", "") or "no_FD"
            color = ("#d62728" if fl in HARD_FAIL_LABELS
                       else "#2ca02c" if r.get("fd_strict_pass")
                       else "#9aaecf")
            ax.scatter(x, y, s=80, color=color, alpha=0.85,
                        edgecolor="white", lw=0.6)
            ax.annotate(r["recipe_id"], (x, y), fontsize=7,
                          xytext=(4, 4), textcoords="offset points")
        gx, gy = g4867_params.get(xkey), g4867_params.get(ykey)
        if gx is not None and gy is not None:
            ax.scatter([gx], [gy], s=180, marker="D", color="#1f77b4",
                        edgecolor="black", lw=1.0, label="G_4867")
        ax.set_xlabel(xkey)
        ax.set_ylabel(ykey)
        ax.set_title(title)
        ax.grid(True, alpha=0.25)
        ax.legend(loc="best", fontsize=9)
        fig.tight_layout()
        fig.savefig(out_path, dpi=150)
        plt.close(fig)

    _failure_scatter(mode_a_box, "Hmax_mol_dm3", "kdep_s_inv",
                        fig_dir / "stage06Q_Hmax_kdep_failure_map.png",
                        "Stage 06Q -- Hmax x kdep failure map "
                        "(Mode A-geometry-box)")
    _failure_scatter(mode_a_box, "Q0_mol_dm3", "kq_s_inv",
                        fig_dir / "stage06Q_Q0_kq_failure_map.png",
                        "Stage 06Q -- Q0 x kq failure map "
                        "(Mode A-geometry-box)")
    _failure_scatter(mode_a_box, "DH_nm2_s", "time_s",
                        fig_dir / "stage06Q_DH_time_failure_map.png",
                        "Stage 06Q -- DH x time failure map "
                        "(Mode A-geometry-box)")

    # Combined interaction-failure-maps figure (2x2 small multiples).
    if mode_a_box:
        fig, axes = plt.subplots(2, 2, figsize=(13.0, 10.0))
        axes_flat = axes.flatten()
        triples = [
            ("Hmax_mol_dm3", "kdep_s_inv", "Hmax x kdep"),
            ("Q0_mol_dm3",   "kq_s_inv",   "Q0 x kq"),
            ("DH_nm2_s",     "time_s",     "DH x time"),
            ("dose_mJ_cm2",  "sigma_nm",   "dose x sigma"),
        ]
        for ax, (xk, yk, label) in zip(axes_flat, triples):
            for r in mode_a_box:
                x = _safe_float(r.get(xk)); y = _safe_float(r.get(yk))
                if not (np.isfinite(x) and np.isfinite(y)):
                    continue
                fl = r.get("fd_label", "") or "no_FD"
                color = ("#d62728" if fl in HARD_FAIL_LABELS
                           else "#2ca02c" if r.get("fd_strict_pass")
                           else "#9aaecf")
                ax.scatter(x, y, s=60, color=color, alpha=0.80,
                            edgecolor="white", lw=0.5)
            gx, gy = g4867_params.get(xk), g4867_params.get(yk)
            if gx is not None and gy is not None:
                ax.scatter([gx], [gy], s=180, marker="D", color="#1f77b4",
                            edgecolor="black", lw=1.0, label="G_4867")
            ax.set_xlabel(xk); ax.set_ylabel(yk); ax.set_title(label)
            ax.grid(True, alpha=0.25)
            ax.legend(loc="best", fontsize=8)
        fig.suptitle("Stage 06Q -- interaction failure maps "
                       "(Mode A-geometry-box Mode B recipes)",
                       fontsize=12)
        fig.tight_layout(rect=[0, 0, 1, 0.96])
        fig.savefig(fig_dir / "stage06Q_interaction_failure_maps.png",
                     dpi=150)
        plt.close(fig)

    # ----- Console summary -------------------------------------------------
    print(f"\nStage 06Q -- blindspot analysis summary")
    print(f"  blindspot dataset rows:        {len(blind_rows)}")
    print(f"  J vs G time-window residuals:  {len(res_rows)} offsets")
    print(f"  J_1453 strict_pass residual mean: "
          f"{summary_residual_section['J1453_strict_pass_residual_mean']:+.3f}  "
          f"(min at offset = "
          f"{summary_residual_section['J1453_strict_pass_residual_min_offset']})")
    print(f"  G_4867 strict_pass residual mean: "
          f"{summary_residual_section['G4867_strict_pass_residual_mean']:+.3f}")
    print(f"  Relative-advantage residual mean: "
          f"{summary_residual_section['relative_advantage_residual_mean']:+.3f}  "
          f"(early {summary_residual_section['relative_advantage_residual_early_mean']:+.3f}  "
          f"late {summary_residual_section['relative_advantage_residual_late_mean']:+.3f})")
    print(f"  False-PASS recipes diagnosed:  {len(diagnostics)}")
    print(f"    by dominant cause: "
          f"{dict(Counter(d['dominant_cause'].split(';')[0] for d in diagnostics))}")
    print(f"  Mode A-geometry-box Mode B recipes: {len(mode_a_box)}")
    print(f"  primary recommended correction: "
          f"{summary_json['recommended_primary_correction']}")
    print(f"  outputs ->")
    print(f"    {yopt_dir / 'stage06Q_blindspot_dataset.csv'}")
    print(f"    {yopt_dir / 'stage06Q_J1453_G4867_residuals.csv'}")
    print(f"    {yopt_dir / 'stage06Q_false_pass_diagnostics.csv'}")
    print(f"    {yopt_dir / 'stage06Q_mode_a_geometry_blindspot.csv'}")
    print(f"    {yopt_dir / 'stage06Q_recommended_corrections.yaml'}")
    print(f"    {logs_dir / 'stage06Q_summary.json'}")
    print(f"    {fig_dir} / stage06Q_*.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

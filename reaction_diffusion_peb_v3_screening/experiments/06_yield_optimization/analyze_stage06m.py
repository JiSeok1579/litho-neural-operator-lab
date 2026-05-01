"""Stage 06M -- aggregate the time_s deep-MC FD rows, compute time
budget thresholds, run a cheap surrogate-side Gaussian sweep, and
plot 7 figures.

Reads:
    outputs/yield_optimization/stage06M_time_deep_mc.csv
    outputs/yield_optimization/stage06I_mode_a_final_recipes.yaml
    outputs/yield_optimization/stage06G_strict_score_config.yaml
    outputs/labels/stage06E_fd_baseline_v2_op_mc.csv  (optional overlay)

Writes:
    outputs/yield_optimization/stage06M_time_budget_summary.csv
    outputs/logs/stage06M_summary.json
    outputs/figures/06_yield_optimization/
        stage06M_strict_pass_prob_vs_time_offset.png
        stage06M_robust_valid_prob_vs_time_offset.png
        stage06M_cd_error_vs_time_offset.png
        stage06M_ler_vs_time_offset.png
        stage06M_margin_vs_time_offset.png
        stage06M_failure_breakdown_vs_time_offset.png
        stage06M_time_budget_window.png
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
from reaction_diffusion_peb_v3_screening.src.metrics_io import (
    FEATURE_KEYS,
    load_model,
    read_labels_csv,
)
from reaction_diffusion_peb_v3_screening.src.process_variation import (
    VariationSpec,
    sample_variations,
)
from reaction_diffusion_peb_v3_screening.src.yield_optimizer import (
    YieldScoreConfig,
    evaluate_recipes,
)

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_stage06g_strict_optimization import (  # noqa: E402
    StrictScoreConfig,
    compute_strict_score,
)


V3_DIR = ROOT / "reaction_diffusion_peb_v3_screening"
CD_TARGET_NM = 15.0

HARD_FAIL_LABELS = {"under_exposed", "merged",
                     "roughness_degraded", "numerical_invalid"}


def _safe_float(v) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return float("nan")


def _coerce(rows: list[dict], keys: list[str]) -> None:
    for r in rows:
        for k in keys:
            if k in r:
                r[k] = _safe_float(r.get(k))


def per_row_strict_score_from_fd(row: dict, strict_yaml: dict) -> float:
    label = str(row.get("label", ""))
    cw = strict_yaml["class_weights"]
    sp = strict_yaml["strict_penalties"]
    th = strict_yaml["thresholds"]
    cd_tol  = float(th["cd_tol_nm"])
    ler_cap = float(th["ler_cap_nm"])
    cd = _safe_float(row.get("CD_final_nm"))
    ler = _safe_float(row.get("LER_CD_locked_nm"))
    margin = _safe_float(row.get("P_line_margin", 0.0))
    if np.isfinite(cd):
        cd_pen = max(0.0, abs(cd - CD_TARGET_NM) - cd_tol) / max(cd_tol, 1e-12)
    else:
        cd_pen = 0.0
    if np.isfinite(ler):
        ler_pen = max(0.0, ler - ler_cap) / max(float(sp["ler_std_norm_nm"]), 1e-12)
    else:
        ler_pen = 0.0
    return float(
        cw.get(label, 0.0)
        - float(sp["cd_strict_weight"])  * cd_pen
        - float(sp["ler_strict_weight"]) * ler_pen
        + float(sp["margin_bonus"])      * (margin if np.isfinite(margin) else 0.0)
    )


def aggregate_cell(rows: list[dict], strict_yaml: dict) -> dict:
    n = len(rows)
    if n == 0:
        return {}
    labels = [str(r.get("label", "")) for r in rows]
    cd = np.array([_safe_float(r.get("CD_final_nm")) for r in rows])
    ler = np.array([_safe_float(r.get("LER_CD_locked_nm")) for r in rows])
    margin = np.array([_safe_float(r.get("P_line_margin")) for r in rows])
    cd_err = np.abs(cd - CD_TARGET_NM)
    cd_tol  = float(strict_yaml["thresholds"]["cd_tol_nm"])
    ler_cap = float(strict_yaml["thresholds"]["ler_cap_nm"])
    strict_per_row = np.array([per_row_strict_score_from_fd(r, strict_yaml)
                                  for r in rows])

    n_robust = sum(1 for l in labels if l == "robust_valid")
    n_margin = sum(1 for l in labels if l == "margin_risk")
    n_hard = sum(1 for l in labels if l in HARD_FAIL_LABELS)
    n_strict_pass = sum(
        1 for r in rows
        if str(r.get("label", "")) == "robust_valid"
            and abs(_safe_float(r.get("CD_final_nm")) - CD_TARGET_NM) <= cd_tol
            and _safe_float(r.get("LER_CD_locked_nm")) <= ler_cap
    )
    return {
        "n_mc":               n,
        "robust_valid_prob":  float(n_robust / n),
        "margin_risk_prob":   float(n_margin / n),
        "defect_prob":        float(n_hard / n),
        "p_under_exposed":    float(sum(1 for l in labels if l == "under_exposed") / n),
        "p_merged":           float(sum(1 for l in labels if l == "merged") / n),
        "p_roughness_degraded": float(sum(1 for l in labels if l == "roughness_degraded") / n),
        "p_numerical_invalid":  float(sum(1 for l in labels if l == "numerical_invalid") / n),
        "strict_pass_prob":   float(n_strict_pass / n),
        "mean_cd_final":      float(np.nanmean(cd)),
        "std_cd_final":       float(np.nanstd(cd)),
        "mean_cd_error":      float(np.nanmean(cd_err)),
        "std_cd_error":       float(np.nanstd(cd_err)),
        "mean_ler_locked":    float(np.nanmean(ler)),
        "std_ler_locked":     float(np.nanstd(ler)),
        "mean_p_line_margin": float(np.nanmean(margin)),
        "std_p_line_margin":  float(np.nanstd(margin)),
        "mean_strict_score":  float(np.nanmean(strict_per_row)),
        "std_strict_score":   float(np.nanstd(strict_per_row)),
    }


# --------------------------------------------------------------------------
# Budget thresholds from per-offset aggregates.
# --------------------------------------------------------------------------
def estimate_budget(offsets: np.ndarray, vals: np.ndarray,
                       threshold: float, *, sense: str = "ge") -> tuple[float, float]:
    """Return the contiguous run of offsets where vals satisfies the
    threshold (`>= threshold` if sense='ge', `<= threshold` if 'le'),
    constrained to include offset = 0. Returns (min_offset, max_offset)
    or (nan, nan) if even offset 0 fails.
    """
    if sense == "ge":
        ok = vals >= threshold - 1e-9
    else:
        ok = vals <= threshold + 1e-9
    zero_idx = int(np.argmin(np.abs(offsets)))
    if not ok[zero_idx]:
        return float("nan"), float("nan")
    lo = zero_idx
    while lo > 0 and ok[lo - 1]:
        lo -= 1
    hi = zero_idx
    while hi < len(ok) - 1 and ok[hi + 1]:
        hi += 1
    return float(offsets[lo]), float(offsets[hi])


# --------------------------------------------------------------------------
# Plots
# --------------------------------------------------------------------------
def _line_plot(offsets, vals, *, ylabel, title, baseline_value,
                threshold_lines, out_path, color="#1f77b4"):
    fig, ax = plt.subplots(figsize=(9.5, 5.5))
    ax.plot(offsets, vals, "-o", color=color, lw=1.6, markersize=6)
    ax.axvline(0.0, color="#1f1f1f", lw=0.6, alpha=0.6)
    if baseline_value is not None and np.isfinite(baseline_value):
        ax.scatter([0.0], [baseline_value], s=90, marker="*",
                    color="#7a0a0a", edgecolor="white", lw=1.0,
                    label=f"offset = 0 baseline = {baseline_value:.3f}",
                    zorder=5)
    for thr_label, thr_val, color2 in threshold_lines:
        ax.axhline(thr_val, color=color2, ls="--", lw=1.0, alpha=0.85,
                    label=f"{thr_label} = {thr_val}")
    ax.set_xlabel("time_s offset (s, around G_4867 nominal)")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_failure_breakdown(offsets, aggrs, out_path):
    classes = [
        ("robust_valid_prob", "robust_valid",      "#2ca02c"),
        ("margin_risk_prob",   "margin_risk",       "#ffbf00"),
        ("p_under_exposed",    "under_exposed",     "#1f77b4"),
        ("p_merged",           "merged",            "#d62728"),
        ("p_roughness_degraded","roughness",         "#9467bd"),
        ("p_numerical_invalid","numerical",         "#8c564b"),
    ]
    M = np.array([[a.get(k, 0.0) for k, _, _ in classes] for a in aggrs])
    fig, ax = plt.subplots(figsize=(11.0, 5.5))
    bottoms = np.zeros(len(offsets))
    for j, (col, label, color) in enumerate(classes):
        ax.bar(offsets, M[:, j], bottom=bottoms,
                color=color, alpha=0.85, label=label,
                edgecolor="white", lw=0.5, width=0.7)
        bottoms = bottoms + M[:, j]
    ax.axvline(0.0, color="#1f1f1f", lw=0.6, alpha=0.6)
    ax.set_xlabel("time_s offset (s)"); ax.set_ylabel("MC class probability")
    ax.set_title("Stage 06M -- failure-mode breakdown vs time offset (G_4867)")
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.20),
              ncol=6, fontsize=9, framealpha=0.95)
    ax.set_ylim(0, 1.02)
    ax.grid(True, alpha=0.25, axis="y")
    fig.tight_layout(rect=[0, 0.05, 1, 1])
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_time_budget_window(offsets, strict_pass, robust_prob, defect_prob,
                              window: tuple[float, float], out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(11.0, 6.0))
    ax.plot(offsets, strict_pass, "-o", color="#1f77b4", lw=1.7,
            label="strict_pass_prob (>= 0.5)")
    ax.plot(offsets, robust_prob, "-s", color="#2ca02c", lw=1.4,
            label="robust_valid_prob (>= 0.8)")
    ax.plot(offsets, defect_prob, "-^", color="#d62728", lw=1.4,
            label="defect_prob (<= 0.05)")
    ax.axhline(0.5,  color="#1f77b4", ls=":", lw=0.7, alpha=0.6)
    ax.axhline(0.8,  color="#2ca02c", ls=":", lw=0.7, alpha=0.6)
    ax.axhline(0.05, color="#d62728", ls=":", lw=0.7, alpha=0.6)
    if all(np.isfinite(window)):
        ax.axvspan(window[0], window[1], alpha=0.15, color="#2ca02c",
                    label=f"all-criteria window: [{window[0]:.0f}, {window[1]:.0f}] s")
    ax.axvline(0.0, color="#1f1f1f", lw=0.6, alpha=0.6)
    ax.set_xlabel("time_s offset (s)"); ax.set_ylabel("probability")
    ax.set_title("Stage 06M -- time budget window for G_4867")
    ax.set_ylim(-0.02, 1.02)
    ax.grid(True, alpha=0.25)
    ax.legend(loc="center right", fontsize=10, framealpha=0.95)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# --------------------------------------------------------------------------
# Cheap surrogate-side Gaussian sweep (sigma_t in {1, 2, 3}).
# --------------------------------------------------------------------------
def surrogate_gaussian_sweep(base_params: dict, sigmas: list[float],
                                n_var: int, space: CandidateSpace,
                                var_spec: VariationSpec,
                                score_cfg: YieldScoreConfig,
                                strict_cfg: StrictScoreConfig,
                                clf, reg, aux,
                                seed: int) -> list[dict]:
    out = []
    rng = np.random.default_rng(seed)
    base_cand = {k: float(v) for k, v in base_params.items()}
    base_cand["pitch_nm"]    = float(base_cand["pitch_nm"])
    base_cand["line_cd_nm"]  = base_cand["pitch_nm"] * base_cand["line_cd_ratio"]
    base_cand["domain_x_nm"] = base_cand["pitch_nm"] * 5.0
    base_cand["dose_norm"]   = base_cand["dose_mJ_cm2"] / float(space.fixed["reference_dose_mJ_cm2"])
    for fk, fv in space.fixed.items():
        base_cand.setdefault(fk, fv)
    base_cand["_id"] = "G_4867_gauss_eval"
    base_time = float(base_cand["time_s"])
    bounds_time = next((float(p["high"]), float(p["low"]))
                          for p in space.parameters
                          if p["type"] == "uniform" and p["name"] == "time_s")
    hi_t, lo_t = bounds_time

    for sigma in sigmas:
        sub_rng = np.random.default_rng(int(rng.integers(0, 2**31 - 1)))
        # Build a synthetic candidate pool of n_var "centres" each with its
        # own time draw, then run evaluate_recipes once with n_var = 1 each.
        # Or simpler: use the existing variation pipeline but rewrite time
        # afterwards. We loop via evaluate_recipes which needs a pool.
        candidates = []
        for j in range(n_var):
            new_time = float(np.clip(base_time + sub_rng.normal(0.0, float(sigma)),
                                         lo_t, hi_t))
            c = dict(base_cand)
            c["time_s"] = new_time
            c["domain_x_nm"] = float(c["pitch_nm"]) * 5.0
            c["dose_norm"]   = float(c["dose_mJ_cm2"]) / float(space.fixed["reference_dose_mJ_cm2"])
            c["_id"] = f"G_4867_gauss_s{int(sigma)}_{j:04d}"
            candidates.append(c)
        # Score each with full process variation (all knobs incl. time YAML
        # ±2s). We use 50 internal MC variations per candidate to keep it
        # cheap; the OUTER Gaussian on time is the dominant signal here.
        rows = evaluate_recipes(
            candidates, clf, reg, aux,
            var_spec, 50, space, score_cfg,
            seed=int(sub_rng.integers(0, 2**31 - 1)),
        )
        for r in rows:
            r.update(compute_strict_score(r, strict_cfg))
        # Aggregate over outer Gaussian draws.
        cd_err = np.array([abs(float(r["mean_cd_fixed"]) - CD_TARGET_NM) for r in rows])
        ler = np.array([float(r["mean_ler_locked"]) for r in rows])
        margin = np.array([float(r["mean_p_line_margin"]) for r in rows])
        p_robust = np.array([float(r["p_robust_valid"]) for r in rows])
        strict = np.array([float(r["strict_score"]) for r in rows])
        # surrogate strict_pass_prob proxy: P_robust * P(|CD-15|<=cd_tol)
        # * P(LER<=ler_cap), per-recipe Gaussian using the mean/std the
        # surrogate already returned. We just use the cell mean.
        cd_tol  = strict_cfg.cd_tol_nm; ler_cap = strict_cfg.ler_cap_nm
        from math import erf, sqrt
        def _g(mu, sd, lo, hi):
            if sd <= 1e-12: return 1.0 if lo <= mu <= hi else 0.0
            z_hi = (hi - mu) / (sd * sqrt(2)); z_lo = (lo - mu) / (sd * sqrt(2))
            return 0.5 * (erf(z_hi) - erf(z_lo))
        proxies = []
        for r in rows:
            proxies.append(
                float(r["p_robust_valid"])
                * _g(float(r["mean_cd_fixed"]), float(r["std_cd_fixed"]),
                      CD_TARGET_NM - cd_tol, CD_TARGET_NM + cd_tol)
                * _g(float(r["mean_ler_locked"]), float(r["std_ler_locked"]),
                      -np.inf, ler_cap)
            )
        out.append({
            "sigma_time_s":     float(sigma),
            "n_outer_draws":    int(n_var),
            "mean_cd_error":    float(np.mean(cd_err)),
            "std_cd_error":     float(np.std(cd_err)),
            "mean_ler_locked":  float(np.mean(ler)),
            "std_ler_locked":   float(np.std(ler)),
            "mean_p_robust_valid": float(np.mean(p_robust)),
            "mean_strict_score":   float(np.mean(strict)),
            "mean_strict_pass_proxy": float(np.mean(proxies)),
            "std_strict_pass_proxy":  float(np.std(proxies)),
        })
    return out


# --------------------------------------------------------------------------
# Driver
# --------------------------------------------------------------------------
def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str,
                   default=str(V3_DIR / "configs" / "yield_optimization.yaml"))
    p.add_argument("--manifest_yaml", type=str,
                   default=str(V3_DIR / "outputs" / "yield_optimization"
                               / "stage06I_mode_a_final_recipes.yaml"))
    p.add_argument("--mc_csv", type=str,
                   default=str(V3_DIR / "outputs" / "yield_optimization"
                               / "stage06M_time_deep_mc.csv"))
    p.add_argument("--strict_cfg_yaml", type=str,
                   default=str(V3_DIR / "outputs" / "yield_optimization"
                               / "stage06G_strict_score_config.yaml"))
    p.add_argument("--clf", type=str,
                   default=str(V3_DIR / "outputs" / "models"
                               / "stage06H_classifier.joblib"))
    p.add_argument("--reg", type=str,
                   default=str(V3_DIR / "outputs" / "models"
                               / "stage06H_regressor.joblib"))
    p.add_argument("--aux", type=str,
                   default=str(V3_DIR / "outputs" / "models"
                               / "stage06H_aux_cd_fixed_regressor.joblib"))
    p.add_argument("--gauss_sigmas", type=str, default="1,2,3")
    p.add_argument("--gauss_n_outer", type=int, default=400)
    p.add_argument("--seed_gauss", type=int, default=9494)
    args = p.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    space = CandidateSpace.from_yaml(cfg["candidate_space_yaml"])
    var_spec = VariationSpec.from_yaml_dict(cfg["process_variation"])
    score_cfg = YieldScoreConfig.from_yaml_dict(cfg["yield_score"])

    strict_yaml = yaml.safe_load(Path(args.strict_cfg_yaml).read_text())
    cd_tol  = float(strict_yaml["thresholds"]["cd_tol_nm"])
    ler_cap = float(strict_yaml["thresholds"]["ler_cap_nm"])
    strict_cfg = StrictScoreConfig(cd_tol_nm=cd_tol, ler_cap_nm=ler_cap)

    manifest = yaml.safe_load(Path(args.manifest_yaml).read_text())
    base = next(r for r in manifest["representatives"]
                  if r["recipe_id"] == manifest["primary_recommended_recipe"])
    base_params = {k: float(v) for k, v in base["parameters"].items()}
    base_time = float(base_params["time_s"])

    fd_rows = read_labels_csv(args.mc_csv)
    _coerce(fd_rows, ["CD_final_nm", "CD_locked_nm", "LER_CD_locked_nm",
                       "area_frac", "P_line_margin",
                       "time_offset_s", "sigma_time_s", "variation_idx"])

    # ----- Aggregate per deterministic offset -----
    det_rows = [r for r in fd_rows if str(r.get("scenario", "")) == "det_offset"]
    by_offset: dict[float, list[dict]] = defaultdict(list)
    for r in det_rows:
        by_offset[round(_safe_float(r["time_offset_s"]), 3)].append(r)
    offsets_sorted = sorted(by_offset.keys())

    aggr_rows = []
    for off in offsets_sorted:
        a = aggregate_cell(by_offset[off], strict_yaml)
        a["time_offset_s"] = float(off)
        a["target_time_s"] = float(base_time + off)
        aggr_rows.append(a)

    # Arrays for plotting / budget estimation.
    offsets_arr   = np.array([a["time_offset_s"]      for a in aggr_rows])
    strict_pass   = np.array([a["strict_pass_prob"]    for a in aggr_rows])
    robust_prob   = np.array([a["robust_valid_prob"]   for a in aggr_rows])
    defect_prob   = np.array([a["defect_prob"]         for a in aggr_rows])
    margin_risk   = np.array([a["margin_risk_prob"]    for a in aggr_rows])
    cd_err_mean   = np.array([a["mean_cd_error"]       for a in aggr_rows])
    ler_mean      = np.array([a["mean_ler_locked"]     for a in aggr_rows])
    margin_mean   = np.array([a["mean_p_line_margin"]  for a in aggr_rows])
    strict_mean   = np.array([a["mean_strict_score"]   for a in aggr_rows])

    # Baselines @ offset 0.
    zero_idx = int(np.argmin(np.abs(offsets_arr)))

    # ----- Budget thresholds -----
    win_strict = estimate_budget(offsets_arr, strict_pass, 0.5, sense="ge")
    win_robust = estimate_budget(offsets_arr, robust_prob, 0.8, sense="ge")
    win_defect = estimate_budget(offsets_arr, defect_prob, 0.05, sense="le")

    if all(np.isfinite([win_strict[0], win_robust[0], win_defect[0]])):
        all_lo = max(win_strict[0], win_robust[0], win_defect[0])
        all_hi = min(win_strict[1], win_robust[1], win_defect[1])
        if all_lo > all_hi:
            all_lo, all_hi = float("nan"), float("nan")
    else:
        all_lo, all_hi = float("nan"), float("nan")

    budget_summary_rows = [
        {"criterion": "strict_pass_prob >= 0.5",
         "min_offset_s": win_strict[0], "max_offset_s": win_strict[1],
         "width_s": (win_strict[1] - win_strict[0])
                       if all(np.isfinite(win_strict)) else float("nan")},
        {"criterion": "robust_valid_prob >= 0.8",
         "min_offset_s": win_robust[0], "max_offset_s": win_robust[1],
         "width_s": (win_robust[1] - win_robust[0])
                       if all(np.isfinite(win_robust)) else float("nan")},
        {"criterion": "defect_prob <= 0.05",
         "min_offset_s": win_defect[0], "max_offset_s": win_defect[1],
         "width_s": (win_defect[1] - win_defect[0])
                       if all(np.isfinite(win_defect)) else float("nan")},
        {"criterion": "all of the above",
         "min_offset_s": all_lo, "max_offset_s": all_hi,
         "width_s": (all_hi - all_lo)
                       if all(np.isfinite([all_lo, all_hi])) else float("nan")},
    ]

    # ----- Failure-mode dominance under early/late offsets -----
    # If every FD label is robust_valid in the perturbation box, the
    # "failure" wrt strict_pass is CD_error overshoot, not a label flip.
    # Report that case explicitly rather than picking an arbitrary
    # zero-probability label class as the "dominant" mode.
    neg_mask = offsets_arr < 0; pos_mask = offsets_arr > 0
    NO_LABEL_FLIP_THRESHOLD = 0.005   # max non-robust prob below this -> no flip

    def _dominant_fail(mask):
        if not np.any(mask):
            return None, 0.0
        avg = {
            "under_exposed":      float(np.mean([a["p_under_exposed"]      for a, m in zip(aggr_rows, mask) if m])),
            "merged":             float(np.mean([a["p_merged"]             for a, m in zip(aggr_rows, mask) if m])),
            "roughness_degraded": float(np.mean([a["p_roughness_degraded"]  for a, m in zip(aggr_rows, mask) if m])),
            "numerical_invalid":  float(np.mean([a["p_numerical_invalid"]   for a, m in zip(aggr_rows, mask) if m])),
            "margin_risk":        float(np.mean([a["margin_risk_prob"]      for a, m in zip(aggr_rows, mask) if m])),
        }
        kind, val = max(avg.items(), key=lambda kv: kv[1])
        if val < NO_LABEL_FLIP_THRESHOLD:
            return "cd_error_overshoot_no_label_flip", float(val)
        return kind, float(val)

    neg_fail_kind, neg_fail_p = _dominant_fail(neg_mask)
    pos_fail_kind, pos_fail_p = _dominant_fail(pos_mask)

    # ----- Optional Gaussian FD scenario aggregate -----
    gauss_rows = [r for r in fd_rows if str(r.get("scenario", "")) == "gaussian_time"]
    gauss_aggr = aggregate_cell(gauss_rows, strict_yaml) if gauss_rows else {}
    gauss_sigma = float(gauss_rows[0]["sigma_time_s"]) if gauss_rows else float("nan")

    # ----- Surrogate-side Gaussian sweep -----
    sigmas = [float(s) for s in args.gauss_sigmas.split(",") if s.strip()]
    print(f"  surrogate gaussian sweep on time_s: sigmas = {sigmas}, "
          f"outer draws = {args.gauss_n_outer}")
    clf, _ = load_model(args.clf); reg, _ = load_model(args.reg)
    aux, _ = load_model(args.aux)
    sur_gauss = surrogate_gaussian_sweep(
        base_params, sigmas, args.gauss_n_outer, space, var_spec,
        score_cfg, strict_cfg, clf, reg, aux, seed=args.seed_gauss,
    )

    # ----- Outputs -----
    yopt_dir = V3_DIR / "outputs" / "yield_optimization"
    logs_dir = V3_DIR / "outputs" / "logs"
    fig_dir  = V3_DIR / "outputs" / "figures" / "06_yield_optimization"
    fig_dir.mkdir(parents=True, exist_ok=True); yopt_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    # Time budget summary CSV.
    with (yopt_dir / "stage06M_time_budget_summary.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["criterion", "min_offset_s", "max_offset_s", "width_s"])
        w.writeheader()
        for row in budget_summary_rows:
            w.writerow(row)

    # Figures.
    base_strict_pass = float(strict_pass[zero_idx])
    base_robust = float(robust_prob[zero_idx])
    base_defect = float(defect_prob[zero_idx])
    base_cd_err = float(cd_err_mean[zero_idx])
    base_ler = float(ler_mean[zero_idx])
    base_margin = float(margin_mean[zero_idx])

    _line_plot(offsets_arr, strict_pass,
                ylabel="strict_pass_prob (FD MC)",
                title="Stage 06M -- strict_pass_prob vs time_s offset (G_4867)",
                baseline_value=base_strict_pass,
                threshold_lines=[("strict_pass >= 0.5", 0.5, "#1f77b4")],
                out_path=fig_dir / "stage06M_strict_pass_prob_vs_time_offset.png",
                color="#1f77b4")
    _line_plot(offsets_arr, robust_prob,
                ylabel="robust_valid_prob (FD MC)",
                title="Stage 06M -- robust_valid_prob vs time_s offset",
                baseline_value=base_robust,
                threshold_lines=[("robust_valid >= 0.8", 0.8, "#2ca02c")],
                out_path=fig_dir / "stage06M_robust_valid_prob_vs_time_offset.png",
                color="#2ca02c")
    _line_plot(offsets_arr, cd_err_mean,
                ylabel="mean CD_error (nm)",
                title="Stage 06M -- mean CD_error vs time_s offset",
                baseline_value=base_cd_err,
                threshold_lines=[("strict CD_tol = 0.5", 0.5, "#1f1f1f")],
                out_path=fig_dir / "stage06M_cd_error_vs_time_offset.png",
                color="#9467bd")
    _line_plot(offsets_arr, ler_mean,
                ylabel="mean LER_CD_locked (nm)",
                title="Stage 06M -- mean LER vs time_s offset",
                baseline_value=base_ler,
                threshold_lines=[("strict LER_cap = 3.0", 3.0, "#1f1f1f")],
                out_path=fig_dir / "stage06M_ler_vs_time_offset.png",
                color="#d62728")
    _line_plot(offsets_arr, margin_mean,
                ylabel="mean P_line_margin",
                title="Stage 06M -- mean P_line_margin vs time_s offset",
                baseline_value=base_margin,
                threshold_lines=[],
                out_path=fig_dir / "stage06M_margin_vs_time_offset.png",
                color="#ff7f0e")
    plot_failure_breakdown(offsets_arr, aggr_rows,
                              fig_dir / "stage06M_failure_breakdown_vs_time_offset.png")
    plot_time_budget_window(offsets_arr, strict_pass, robust_prob, defect_prob,
                              (all_lo, all_hi),
                              fig_dir / "stage06M_time_budget_window.png")

    # ----- Acceptance & summary JSON -----
    acceptance = {
        "n_fd_runs_total":               int(len(fd_rows)),
        "n_fd_runs_deterministic":       int(len(det_rows)),
        "n_fd_runs_gaussian":            int(len(gauss_rows)),
        "min_required_fd_runs":          1000,
        "fd_budget_met":                 bool(len(fd_rows) >= 1000),
        "strict_pass_curve_produced":    True,
        "time_budget_estimated":         True,
        "primary_failure_mode_negative": neg_fail_kind,
        "primary_failure_mode_positive": pos_fail_kind,
        "policy_v2_OP_frozen":           bool(cfg["policy"].get("v2_OP_frozen", True)),
        "policy_published_data_loaded":  bool(cfg["policy"].get("published_data_loaded", False)),
        "policy_external_calibration":   "none",
        "tolerates_+/-3s_at_strict>=0.5": bool(
            np.isfinite(win_strict[0]) and np.isfinite(win_strict[1])
            and (win_strict[0] <= -3.0 + 1e-9) and (win_strict[1] >= 3.0 - 1e-9)
        ),
    }

    payload = {
        "stage": "06M",
        "policy": cfg["policy"],
        "primary_recipe_id": "G_4867",
        "base_time_s":       base_time,
        "strict_thresholds": {"cd_tol_nm": cd_tol, "ler_cap_nm": ler_cap},
        "deterministic_offsets_aggregates": aggr_rows,
        "time_budget": {
            "strict_pass_prob_ge_0.5":  list(win_strict),
            "robust_valid_prob_ge_0.8": list(win_robust),
            "defect_prob_le_0.05":      list(win_defect),
            "all_three":                 [all_lo, all_hi],
        },
        "primary_failure_mode_under_negative_offset": {
            "kind": neg_fail_kind, "avg_probability": neg_fail_p,
        },
        "primary_failure_mode_under_positive_offset": {
            "kind": pos_fail_kind, "avg_probability": pos_fail_p,
        },
        "gaussian_fd_scenario": {
            "sigma_time_s":   gauss_sigma,
            "aggregate":      gauss_aggr,
        },
        "surrogate_gaussian_sweep": sur_gauss,
        "interpretation": {
            "tolerates_+/-3s_at_strict>=0.5": acceptance["tolerates_+/-3s_at_strict>=0.5"],
            "negative_side_dominant_failure": neg_fail_kind,
            "positive_side_dominant_failure": pos_fail_kind,
            "asymmetry_note": (
                "If the positive-offset failure rises faster than the "
                "negative-offset failure, late-bake is the harder side; "
                "vice versa for early-bake. Symmetric collapse means the "
                "recipe is timing-fragile in both directions."
            ),
        },
        "acceptance": acceptance,
    }
    (logs_dir / "stage06M_summary.json").write_text(
        json.dumps(payload, indent=2, default=float))

    # ----- Console summary -----
    print(f"\nStage 06M -- analysis summary")
    print(f"  base_time_s = {base_time:.3f}")
    print(f"  per-offset FD MC aggregates (deterministic):")
    print(f"  {'offset(s)':>10} {'strict':>8} {'robust':>8} {'margin_r':>9} {'defect':>7} "
           f"{'CD_err':>7} {'LER':>7}")
    for a in aggr_rows:
        print(f"  {a['time_offset_s']:>10.1f} {a['strict_pass_prob']:>8.3f} "
              f"{a['robust_valid_prob']:>8.3f} {a['margin_risk_prob']:>9.3f} "
              f"{a['defect_prob']:>7.3f} {a['mean_cd_error']:>7.3f} "
              f"{a['mean_ler_locked']:>7.3f}")
    print(f"  time budget (deterministic):")
    for r in budget_summary_rows:
        if all(np.isfinite([r['min_offset_s'], r['max_offset_s']])):
            print(f"    {r['criterion']:<28} -> [{r['min_offset_s']:+.1f}, "
                  f"{r['max_offset_s']:+.1f}] s  (width {r['width_s']:.1f} s)")
        else:
            print(f"    {r['criterion']:<28} -> not satisfied at offset 0")
    print(f"  primary failure under early bake (offset < 0): {neg_fail_kind} "
          f"(avg prob {neg_fail_p:.3f})")
    print(f"  primary failure under late bake (offset > 0):  {pos_fail_kind} "
          f"(avg prob {pos_fail_p:.3f})")
    print(f"  surrogate gaussian sweep (sigma_time -> mean strict_pass_proxy):")
    for g in sur_gauss:
        print(f"    sigma_t = {g['sigma_time_s']:.1f}s  proxy = "
              f"{g['mean_strict_pass_proxy']:.3f} +/- {g['std_strict_pass_proxy']:.3f}")
    print(f"  acceptance: tolerates +/- 3s at strict >= 0.5: "
          f"{acceptance['tolerates_+/-3s_at_strict>=0.5']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

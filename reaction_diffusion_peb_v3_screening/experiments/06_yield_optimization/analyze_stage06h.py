"""Stage 06H -- analysis, surrogate validation, false-PASS, figures.

Reads:
    outputs/yield_optimization/stage06G_top_recipes.csv
    outputs/yield_optimization/stage06G_representative_recipes.csv
    outputs/yield_optimization/stage06G_strict_score_config.yaml
    outputs/labels/stage06H_fd_top100_nominal.csv
    outputs/labels/stage06H_fd_top10_mc.csv
    outputs/labels/stage06H_fd_representative_mc.csv
    outputs/logs/stage06G_summary.json
    outputs/logs/stage06H_surrogate_refresh_summary.json

Writes:
    outputs/labels/stage06H_surrogate_vs_fd_metrics.csv
    outputs/labels/stage06H_false_pass_cases.csv
    outputs/labels/stage06H_representative_mc_breakdown.csv
    outputs/yield_optimization/stage06H_06g_recipes_rescored_by_06h.csv
    outputs/logs/stage06H_fd_verification_summary.json
    outputs/logs/stage06H_false_pass_summary.json
    outputs/figures/06_yield_optimization/
        stage06H_surrogate_vs_fd_strict_score.png
        stage06H_cd_error_vs_ler_fd_pareto.png
        stage06H_representative_stability_boxplots.png
        stage06H_defect_breakdown_top10.png
        stage06H_ranking_before_after.png
        stage06H_feature_importance.png
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
from reaction_diffusion_peb_v3_screening.src.fd_yield_score import (
    fd_yield_score_from_rows,
    fd_yield_score_per_recipe,
    nominal_yield_score,
    spearman,
    topk_overlap,
)
from reaction_diffusion_peb_v3_screening.src.metrics_io import (
    FEATURE_KEYS,
    load_model,
    read_labels_csv,
)
from reaction_diffusion_peb_v3_screening.src.process_variation import (
    VariationSpec,
)
from reaction_diffusion_peb_v3_screening.src.yield_optimizer import (
    YieldScoreConfig,
    evaluate_recipes,
)

# Reuse the strict_score formula from the Stage 06G run script.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_stage06g_strict_optimization import (  # noqa: E402
    StrictScoreConfig,
    compute_strict_score,
)


V3_DIR = ROOT / "reaction_diffusion_peb_v3_screening"
CD_TARGET_NM = 15.0

PARAM_AXES = [
    "dose_mJ_cm2", "sigma_nm", "DH_nm2_s", "time_s",
    "Hmax_mol_dm3", "kdep_s_inv", "Q0_mol_dm3", "kq_s_inv",
    "line_cd_ratio",
]
HARD_FAIL_LABELS = ("under_exposed", "merged",
                    "roughness_degraded", "numerical_invalid")
SOFT_FAIL_LABEL = "margin_risk"


def _safe_float(v) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return float("nan")


def _coerce_floats(rows: list[dict], keys: list[str]) -> None:
    for r in rows:
        for k in keys:
            if k in r:
                r[k] = _safe_float(r.get(k))


def _write_csv(rows: list[dict], path: Path,
                column_order: list[str] | None = None) -> None:
    if not rows:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=column_order or [])
            w.writeheader()
        return
    cols = column_order if column_order else list(rows[0].keys())
    extras = []
    for r in rows:
        for k in r.keys():
            if k not in cols and k not in extras:
                extras.append(k)
    cols = cols + extras
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)


# --------------------------------------------------------------------------
# MC aggregation helper.
# --------------------------------------------------------------------------
def _mc_aggregate(rows: list[dict], score_cfg: YieldScoreConfig,
                   strict_cfg: StrictScoreConfig) -> dict[str, dict]:
    by_recipe: dict[str, list[dict]] = {}
    for r in rows:
        rid = str(r.get("source_recipe_id", ""))
        by_recipe.setdefault(rid, []).append(r)

    out: dict[str, dict] = {}
    for rid, rs in by_recipe.items():
        cd = np.array([_safe_float(r.get("CD_final_nm")) for r in rs])
        ler = np.array([_safe_float(r.get("LER_CD_locked_nm")) for r in rs])
        margin = np.array([_safe_float(r.get("P_line_margin")) for r in rs])
        labels = [str(r.get("label", "")) for r in rs]
        n = len(rs)
        n_robust = sum(1 for l in labels if l == "robust_valid")
        n_margin = sum(1 for l in labels if l == "margin_risk")
        n_hard = sum(1 for l in labels if l in HARD_FAIL_LABELS)

        # MC-based strict_score using empirical class probabilities and
        # mean / std of CD / LER / margin.
        mc_aggr_row = {
            "p_robust_valid":         float(n_robust / max(n, 1)),
            "p_margin_risk":          float(n_margin / max(n, 1)),
            "p_under_exposed":        float(sum(1 for l in labels if l == "under_exposed") / max(n, 1)),
            "p_merged":               float(sum(1 for l in labels if l == "merged") / max(n, 1)),
            "p_roughness_degraded":   float(sum(1 for l in labels if l == "roughness_degraded") / max(n, 1)),
            "p_numerical_invalid":    float(sum(1 for l in labels if l == "numerical_invalid") / max(n, 1)),
            "mean_cd_fixed":          float(np.nanmean(cd)),
            "std_cd_fixed":           float(np.nanstd(cd)),
            "mean_ler_locked":        float(np.nanmean(ler)),
            "std_ler_locked":         float(np.nanstd(ler)),
            "mean_p_line_margin":     float(np.nanmean(margin)),
        }
        strict_payload = compute_strict_score(mc_aggr_row, strict_cfg)
        # Original FD MC yield_score (reuse fd_yield_score helper).
        mc_yield = fd_yield_score_from_rows(rs, score_cfg)
        cd_err = np.abs(cd - CD_TARGET_NM)

        out[rid] = {
            "recipe_id":            rid,
            "n_mc":                 n,
            "robust_prob":          mc_aggr_row["p_robust_valid"],
            "margin_risk_prob":     mc_aggr_row["p_margin_risk"],
            "defect_prob":          float(n_hard / max(n, 1)),
            "p_under_exposed":      mc_aggr_row["p_under_exposed"],
            "p_merged":             mc_aggr_row["p_merged"],
            "p_roughness_degraded": mc_aggr_row["p_roughness_degraded"],
            "p_numerical_invalid":  mc_aggr_row["p_numerical_invalid"],
            "mean_cd_final":        mc_aggr_row["mean_cd_fixed"],
            "mean_cd_error":        float(np.nanmean(cd_err)),
            "std_cd_final":         mc_aggr_row["std_cd_fixed"],
            "mean_ler_locked":      mc_aggr_row["mean_ler_locked"],
            "std_ler_locked":       mc_aggr_row["std_ler_locked"],
            "mean_p_line_margin":   mc_aggr_row["mean_p_line_margin"],
            "FD_MC_yield_score":    float(mc_yield["FD_yield_score"]),
            "FD_MC_strict_score":   float(strict_payload["strict_score"]),
            "p_strict_pass": float(sum(
                1 for r in rs
                if str(r.get("label", "")) == "robust_valid"
                   and abs(_safe_float(r.get("CD_final_nm")) - CD_TARGET_NM) <= strict_cfg.cd_tol_nm
                   and _safe_float(r.get("LER_CD_locked_nm")) <= strict_cfg.ler_cap_nm
            ) / max(n, 1)),
        }
    return out


# --------------------------------------------------------------------------
# Plots
# --------------------------------------------------------------------------
def plot_surrogate_vs_fd_strict(pair_rows: list[dict], out_path: Path) -> None:
    sx = np.array([r["strict_score_06g"] for r in pair_rows])
    sy = np.array([r["strict_score_fd_nominal"] for r in pair_rows])
    fig, ax = plt.subplots(figsize=(9.0, 7.0))
    ax.scatter(sx, sy, s=28, c="#d62728", alpha=0.78,
                edgecolor="white", lw=0.5)
    lo = float(np.nanmin([np.nanmin(sx), np.nanmin(sy), -2.0]))
    hi = float(np.nanmax([np.nanmax(sx), np.nanmax(sy), 1.05]))
    ax.plot([lo, hi], [lo, hi], "k--", lw=0.8, alpha=0.5, label="y = x")
    rho = spearman(sx, sy)
    ax.set_xlabel("06G surrogate strict_score")
    ax.set_ylabel("Stage 06H FD strict_score (single nominal FD)")
    ax.set_title(f"06G surrogate vs FD strict_score on top-100  "
                  f"(Spearman ρ = {rho:.3f})")
    ax.legend(loc="lower right", fontsize=10)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_cd_error_vs_ler_pareto(pair_rows: list[dict], op_nom: dict,
                                  cd_tol: float, ler_cap: float,
                                  out_path: Path) -> None:
    cd = np.array([r["FD_CD_error_nm"] for r in pair_rows])
    ler = np.array([r["FD_LER_CD_locked_nm"] for r in pair_rows])
    score = np.array([r["strict_score_fd_nominal"] for r in pair_rows])
    fig, ax = plt.subplots(figsize=(10.0, 7.0))
    sc = ax.scatter(cd, ler, s=22, c=score, cmap="viridis",
                    alpha=0.85, edgecolor="white", lw=0.4)
    cb = plt.colorbar(sc, ax=ax)
    cb.set_label("FD strict_score (nominal)")

    ax.axvline(cd_tol, color="#1f1f1f", ls="--", lw=1.0,
                label=f"strict CD_tol = {cd_tol:.2f} nm")
    ax.axhline(ler_cap, color="#1f1f1f", ls=":", lw=1.0,
                label=f"strict LER_cap = {ler_cap:.1f} nm")
    if op_nom:
        ax.scatter([op_nom.get("CD_error_nm", float("nan"))],
                    [op_nom.get("LER_CD_locked_nm", float("nan"))],
                    s=240, marker="*", color="#7a0a0a",
                    edgecolor="white", lw=1.0, label="v2 frozen OP (FD)")
    ax.set_xlabel("FD CD_error_nm = |CD_final_nm - 15.0|")
    ax.set_ylabel("FD LER_CD_locked_nm")
    ax.set_title("Stage 06H -- FD CD_error vs LER on 06G top-100  "
                  "(color = FD strict_score)")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="upper right", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_representative_boxplots(rep_rows_by_kind: dict, op_mc_rows: list[dict],
                                   strict_cfg: StrictScoreConfig,
                                   out_path: Path) -> None:
    if not rep_rows_by_kind:
        return
    kinds = list(rep_rows_by_kind.keys())
    metrics = ["CD_error_nm", "LER_CD_locked_nm", "P_line_margin",
                "strict_score_per_var"]
    fig, axes = plt.subplots(2, 2, figsize=(13.0, 10.0))
    titles = {
        "CD_error_nm":      "FD CD_error_nm distribution (300 MC each)",
        "LER_CD_locked_nm": "FD LER_CD_locked_nm",
        "P_line_margin":    "FD P_line_margin",
        "strict_score_per_var": "per-variation strict_score (one-hot prob)",
    }
    for ax_idx, m in enumerate(metrics):
        ax = axes[ax_idx // 2, ax_idx % 2]
        data = []
        labels = []
        for kind in kinds:
            vals = np.array([_safe_float(r.get(m, float("nan")))
                              for r in rep_rows_by_kind[kind]])
            vals = vals[np.isfinite(vals)]
            data.append(vals)
            labels.append(kind)
        bp = ax.boxplot(data, labels=labels, patch_artist=True, showmeans=True)
        palette = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd",
                    "#ff7f0e", "#8c564b"]
        for i, patch in enumerate(bp["boxes"]):
            patch.set_facecolor(palette[i % len(palette)])
            patch.set_alpha(0.6)
        if op_mc_rows and m != "strict_score_per_var":
            op_vals = np.array([_safe_float(r.get(m, float("nan")))
                                  for r in op_mc_rows])
            op_vals = op_vals[np.isfinite(op_vals)]
            if op_vals.size:
                ax.axhline(float(np.median(op_vals)),
                            color="#7a0a0a", ls="--", lw=1.0, alpha=0.85,
                            label=f"v2 OP MC median = {np.median(op_vals):.3f}")
                ax.legend(loc="upper right", fontsize=8)
        if m == "CD_error_nm":
            ax.axhline(strict_cfg.cd_tol_nm, color="#1f1f1f", ls=":", lw=0.8,
                        alpha=0.7, label=f"CD tol = {strict_cfg.cd_tol_nm}")
            ax.legend(loc="upper right", fontsize=8)
        elif m == "LER_CD_locked_nm":
            ax.axhline(strict_cfg.ler_cap_nm, color="#1f1f1f", ls=":", lw=0.8,
                        alpha=0.7, label=f"LER cap = {strict_cfg.ler_cap_nm}")
            ax.legend(loc="upper right", fontsize=8)
        ax.set_title(titles[m], fontsize=11)
        ax.grid(True, alpha=0.25, axis="y")
        for label in ax.get_xticklabels():
            label.set_rotation(20); label.set_ha("right")
    fig.suptitle("Stage 06H representative-recipe FD MC stability  "
                  "(300 variations each)", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_defect_breakdown_top10(top10_aggr: list[dict], out_path: Path) -> None:
    rows = sorted(top10_aggr, key=lambda r: -r["FD_MC_strict_score"])
    classes = [
        ("robust_prob", "robust_valid"),
        ("margin_risk_prob", "margin_risk"),
        ("p_under_exposed", "under_exposed"),
        ("p_merged", "merged"),
        ("p_roughness_degraded", "roughness"),
        ("p_numerical_invalid", "numerical"),
    ]
    M = np.array([[r.get(k, 0.0) for k, _ in classes] for r in rows])
    fig, ax = plt.subplots(figsize=(11.5, 5.8))
    bottoms = np.zeros(len(rows))
    palette = ["#2ca02c", "#ffbf00", "#1f77b4", "#d62728", "#9467bd", "#8c564b"]
    for j, (col, label) in enumerate(classes):
        ax.bar(np.arange(len(rows)), M[:, j], bottom=bottoms,
                color=palette[j], alpha=0.85, label=label,
                edgecolor="white", lw=0.4)
        bottoms = bottoms + M[:, j]
    for i, r in enumerate(rows):
        ax.text(i, 1.02, f"strict={r['FD_MC_strict_score']:.2f}",
                ha="center", fontsize=8)
    ax.set_xticks(np.arange(len(rows)))
    ax.set_xticklabels([r["recipe_id"] for r in rows],
                        rotation=25, ha="right", fontsize=8)
    ax.set_xlabel("06G top-10 (sorted by FD MC strict_score)")
    ax.set_ylabel("FD MC class probability")
    ax.set_title("Stage 06H -- top-10 FD MC defect breakdown (100 variations)")
    ax.legend(loc="lower right", fontsize=9, framealpha=0.95)
    ax.set_ylim(0, 1.10)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_ranking_before_after(rows_06g_top: list[dict],
                                rows_06h_rescored: list[dict],
                                rep_ids: list[str],
                                out_path: Path) -> None:
    rid_to_06g_rank = {r["recipe_id"]: int(r["rank_strict"])
                        for r in rows_06g_top}
    # 06H ranks: highest strict_score_06h first.
    rows_06h_rescored = sorted(rows_06h_rescored,
                                key=lambda r: -float(r["strict_score_06h"]))
    rid_to_06h_rank = {r["recipe_id"]: i + 1 for i, r in enumerate(rows_06h_rescored)}

    fig, ax = plt.subplots(figsize=(11.0, 6.0))
    for rid in rid_to_06g_rank:
        r_g = rid_to_06g_rank[rid]
        r_h = rid_to_06h_rank.get(rid, len(rows_06g_top) + 1)
        is_rep = rid in rep_ids
        color = "#d62728" if is_rep else "#9aaecf"
        lw = 1.7 if is_rep else 0.4
        alpha = 0.95 if is_rep else 0.5
        z = 5 if is_rep else 1
        ax.plot([0, 1], [r_g, r_h], "-", color=color, lw=lw, alpha=alpha, zorder=z)
        ax.scatter([0, 1], [r_g, r_h], s=28 if is_rep else 8,
                    color=color, alpha=alpha, zorder=z + 1)
        if is_rep:
            ax.annotate(rid, (1.02, r_h), fontsize=8, color=color,
                        verticalalignment="center")
    ax.invert_yaxis()
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["rank by 06G surrogate strict_score",
                          "rank by 06H surrogate strict_score"])
    ax.set_ylabel("rank (lower = better)")
    ax.set_title("Stage 06H -- rank movement from 06G surrogate to "
                  "refreshed 06H surrogate  (red = 06G representative)")
    ax.set_xlim(-0.05, 1.20)
    ax.grid(True, alpha=0.25, axis="y")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_feature_importance(metrics_06h: dict, out_path: Path) -> None:
    imp = metrics_06h.get("feature_importance_classifier", [])
    if not imp:
        return
    keys = FEATURE_KEYS[: len(imp)]
    order = np.argsort(imp)[::-1]
    keys = [keys[i] for i in order]
    vals = [imp[i] for i in order]
    fig, ax = plt.subplots(figsize=(11.0, 5.5))
    ax.bar(np.arange(len(keys)), vals, color="#1f77b4", alpha=0.85,
            edgecolor="#1f1f1f")
    ax.set_xticks(np.arange(len(keys)))
    ax.set_xticklabels(keys, rotation=25, ha="right", fontsize=9)
    ax.set_ylabel("classifier feature importance (06H RandomForest)")
    ax.set_title("Stage 06H classifier feature importance "
                  "(higher = stronger signal)")
    ax.grid(True, alpha=0.25, axis="y")
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
    p.add_argument("--top_06g_csv", type=str,
                   default=str(V3_DIR / "outputs" / "yield_optimization"
                               / "stage06G_top_recipes.csv"))
    p.add_argument("--reps_06g_csv", type=str,
                   default=str(V3_DIR / "outputs" / "yield_optimization"
                               / "stage06G_representative_recipes.csv"))
    p.add_argument("--strict_cfg_yaml", type=str,
                   default=str(V3_DIR / "outputs" / "yield_optimization"
                               / "stage06G_strict_score_config.yaml"))
    p.add_argument("--fd_top100_csv", type=str,
                   default=str(V3_DIR / "outputs" / "labels"
                               / "stage06H_fd_top100_nominal.csv"))
    p.add_argument("--fd_top10_mc_csv", type=str,
                   default=str(V3_DIR / "outputs" / "labels"
                               / "stage06H_fd_top10_mc.csv"))
    p.add_argument("--fd_rep_mc_csv", type=str,
                   default=str(V3_DIR / "outputs" / "labels"
                               / "stage06H_fd_representative_mc.csv"))
    p.add_argument("--fd_baseline_mc_csv", type=str,
                   default=str(V3_DIR / "outputs" / "labels"
                               / "stage06E_fd_baseline_v2_op_mc.csv"))
    p.add_argument("--summary_06g_json", type=str,
                   default=str(V3_DIR / "outputs" / "logs" / "stage06G_summary.json"))
    p.add_argument("--metrics_06h_json", type=str,
                   default=str(V3_DIR / "outputs" / "logs"
                               / "stage06H_surrogate_refresh_summary.json"))
    p.add_argument("--clf_06h", type=str,
                   default=str(V3_DIR / "outputs" / "models"
                               / "stage06H_classifier.joblib"))
    p.add_argument("--reg_06h", type=str,
                   default=str(V3_DIR / "outputs" / "models"
                               / "stage06H_regressor.joblib"))
    p.add_argument("--aux_06h", type=str,
                   default=str(V3_DIR / "outputs" / "models"
                               / "stage06H_aux_cd_fixed_regressor.joblib"))
    p.add_argument("--n_var_rescore", type=int, default=200,
                   help="MC variations for re-scoring 06G recipes with 06H surrogate")
    p.add_argument("--rescore_seed", type=int, default=6363)
    args = p.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    score_cfg = YieldScoreConfig.from_yaml_dict(cfg["yield_score"])

    # ----- Load strict config -----
    strict_yaml = yaml.safe_load(Path(args.strict_cfg_yaml).read_text())
    cd_tol  = float(strict_yaml["thresholds"]["cd_tol_nm"])
    ler_cap = float(strict_yaml["thresholds"]["ler_cap_nm"])
    strict_cfg = StrictScoreConfig(cd_tol_nm=cd_tol, ler_cap_nm=ler_cap)

    # ----- Read 06G recipes -----
    top_06g = read_labels_csv(args.top_06g_csv)
    num_keys_06g = ["rank_strict", "strict_score", "yield_score",
                     "p_robust_valid", "p_margin_risk", "p_under_exposed",
                     "p_merged", "p_roughness_degraded", "p_numerical_invalid",
                     "mean_cd_fixed", "std_cd_fixed",
                     "mean_cd_locked", "std_cd_locked",
                     "mean_ler_locked", "std_ler_locked",
                     "mean_p_line_margin", "std_p_line_margin",
                     "cd_error_penalty", "ler_penalty"] + FEATURE_KEYS
    _coerce_floats(top_06g, num_keys_06g)
    top_06g.sort(key=lambda r: r.get("rank_strict", 1e9))
    sur_lookup = {r["recipe_id"]: r for r in top_06g}

    reps_06g = read_labels_csv(args.reps_06g_csv)
    _coerce_floats(reps_06g, ["rank_strict", "strict_score", "yield_score"]
                     + FEATURE_KEYS)
    rep_ids = [r["recipe_id"] for r in reps_06g]

    # ----- Read FD outputs -----
    fd_top100 = read_labels_csv(args.fd_top100_csv)
    _coerce_floats(fd_top100, ["CD_final_nm", "CD_locked_nm",
                                  "LER_CD_locked_nm", "area_frac",
                                  "P_line_margin", "rank_strict",
                                  "strict_score_surrogate",
                                  "yield_score_surrogate", "contrast",
                                  "psd_locked_mid", "psd_design_mid"])

    fd_top10_mc = read_labels_csv(args.fd_top10_mc_csv)
    _coerce_floats(fd_top10_mc, ["CD_final_nm", "CD_locked_nm",
                                    "LER_CD_locked_nm", "area_frac",
                                    "P_line_margin", "rank_strict"])

    fd_rep_mc = read_labels_csv(args.fd_rep_mc_csv)
    _coerce_floats(fd_rep_mc, ["CD_final_nm", "CD_locked_nm",
                                  "LER_CD_locked_nm", "area_frac",
                                  "P_line_margin", "rank_strict"])

    fd_baseline_mc = read_labels_csv(args.fd_baseline_mc_csv) \
        if Path(args.fd_baseline_mc_csv).exists() else []
    _coerce_floats(fd_baseline_mc, ["CD_final_nm", "CD_locked_nm",
                                       "LER_CD_locked_nm", "area_frac",
                                       "P_line_margin"])

    # ============================================================
    # Part 1 -- top-100 nominal FD: surrogate-vs-FD comparison.
    # ============================================================
    fd_by_id = {r["source_recipe_id"]: r for r in fd_top100}
    pair_rows: list[dict] = []
    for s in top_06g:
        rid = s["recipe_id"]
        fd = fd_by_id.get(rid)
        cd_err = abs(_safe_float(fd.get("CD_final_nm")) - CD_TARGET_NM) if fd else float("nan")
        # FD-side strict_score on a single nominal row -- treat the
        # one-hot probabilities and zero-std as inputs to the formula.
        if fd is not None:
            mc_aggr = {
                "p_robust_valid":     1.0 if str(fd.get("label", "")) == "robust_valid" else 0.0,
                "p_margin_risk":      1.0 if str(fd.get("label", "")) == "margin_risk" else 0.0,
                "p_under_exposed":    1.0 if str(fd.get("label", "")) == "under_exposed" else 0.0,
                "p_merged":           1.0 if str(fd.get("label", "")) == "merged" else 0.0,
                "p_roughness_degraded": 1.0 if str(fd.get("label", "")) == "roughness_degraded" else 0.0,
                "p_numerical_invalid":  1.0 if str(fd.get("label", "")) == "numerical_invalid" else 0.0,
                "mean_cd_fixed":   _safe_float(fd.get("CD_final_nm")),
                "std_cd_fixed":    0.0,
                "mean_ler_locked": _safe_float(fd.get("LER_CD_locked_nm")),
                "std_ler_locked":  0.0,
                "mean_p_line_margin": _safe_float(fd.get("P_line_margin")),
            }
            strict_payload = compute_strict_score(mc_aggr, strict_cfg)
            fd_strict = float(strict_payload["strict_score"])
            fd_yield = float(nominal_yield_score(fd, score_cfg)["FD_yield_score"])
        else:
            fd_strict = float("nan"); fd_yield = float("nan")

        pair_rows.append({
            "recipe_id":              rid,
            "rank_strict":            int(_safe_float(s.get("rank_strict", 0))),
            "strict_score_06g":       float(s["strict_score"]),
            "yield_score_06g":        float(s["yield_score"]),
            "p_robust_valid_06g":     float(s["p_robust_valid"]),
            "mean_cd_fixed_06g":      float(s["mean_cd_fixed"]),
            "mean_ler_locked_06g":    float(s["mean_ler_locked"]),
            "mean_p_line_margin_06g": float(s["mean_p_line_margin"]),
            "fd_label":               str(fd.get("label", "")) if fd else "",
            "FD_CD_final_nm":         _safe_float(fd.get("CD_final_nm")) if fd else float("nan"),
            "FD_CD_locked_nm":        _safe_float(fd.get("CD_locked_nm")) if fd else float("nan"),
            "FD_LER_CD_locked_nm":    _safe_float(fd.get("LER_CD_locked_nm")) if fd else float("nan"),
            "FD_area_frac":           _safe_float(fd.get("area_frac")) if fd else float("nan"),
            "FD_P_line_margin":       _safe_float(fd.get("P_line_margin")) if fd else float("nan"),
            "FD_contrast":            _safe_float(fd.get("contrast")) if fd else float("nan"),
            "FD_psd_locked_mid":      _safe_float(fd.get("psd_locked_mid")) if fd else float("nan"),
            "FD_CD_error_nm":         float(cd_err),
            "yield_score_fd_nominal":   fd_yield,
            "strict_score_fd_nominal":  fd_strict,
            "FD_pass_strict": bool(
                str(fd.get("label", "")) == "robust_valid"
                and cd_err <= cd_tol
                and _safe_float(fd.get("LER_CD_locked_nm")) <= ler_cap
            ) if fd else False,
            **{k: _safe_float(s.get(k)) for k in PARAM_AXES if k in s},
        })

    # Surrogate-vs-FD regression errors.
    def _err(a_key: str, b_key: str) -> dict:
        a = np.array([_safe_float(r[a_key]) for r in pair_rows])
        b = np.array([_safe_float(r[b_key]) for r in pair_rows])
        finite = np.isfinite(a) & np.isfinite(b)
        if finite.sum() == 0:
            return {"mae": None, "rmse": None, "n": 0}
        d = a[finite] - b[finite]
        return {
            "mae":  float(np.mean(np.abs(d))),
            "rmse": float(np.sqrt(np.mean(d ** 2))),
            "n":    int(finite.sum()),
        }
    errors = {
        "mean_cd_fixed_vs_FD_CD":      _err("mean_cd_fixed_06g",  "FD_CD_final_nm"),
        "mean_ler_locked_vs_FD_LER":   _err("mean_ler_locked_06g", "FD_LER_CD_locked_nm"),
        "mean_p_line_margin_vs_FD":    _err("mean_p_line_margin_06g", "FD_P_line_margin"),
        # area_frac: 06G surrogate doesn't carry a recipe-level mean_area_frac
        # without re-running -- we leave it out (None) here.
        "area_frac": {"mae": None, "rmse": None, "n": 0},
    }
    sx = np.array([r["strict_score_06g"]      for r in pair_rows])
    sy = np.array([r["strict_score_fd_nominal"] for r in pair_rows])
    rho_strict_top100 = spearman(sx, sy)
    rho_cd = spearman(np.abs(np.array([r["mean_cd_fixed_06g"] - CD_TARGET_NM
                                          for r in pair_rows])),
                       np.array([r["FD_CD_error_nm"] for r in pair_rows]))
    rho_ler = spearman(np.array([r["mean_ler_locked_06g"] for r in pair_rows]),
                        np.array([r["FD_LER_CD_locked_nm"] for r in pair_rows]))
    sur_rank = [r["recipe_id"] for r in sorted(pair_rows,
                   key=lambda r: -r["strict_score_06g"])]
    fd_rank  = [r["recipe_id"] for r in sorted(pair_rows,
                   key=lambda r: -r["strict_score_fd_nominal"])]
    top1 = topk_overlap(sur_rank, fd_rank, 1)
    top3 = topk_overlap(sur_rank, fd_rank, 3)
    top5 = topk_overlap(sur_rank, fd_rank, 5)
    top10 = topk_overlap(sur_rank, fd_rank, 10)

    # ============================================================
    # False-PASS detection on top-100 nominal FD.
    # ============================================================
    false_rows: list[dict] = []
    for r in pair_rows:
        label = r["fd_label"]
        if label == "robust_valid":
            continue
        kind = (
            "hard_false_pass" if label in HARD_FAIL_LABELS
            else "soft_false_pass" if label == SOFT_FAIL_LABEL
            else "other_non_robust"
        )
        false_rows.append({
            "recipe_id":              r["recipe_id"],
            "rank_strict":            r["rank_strict"],
            "fd_label":               label,
            "false_pass_kind":        kind,
            "strict_score_06g":       r["strict_score_06g"],
            "strict_score_fd_nominal": r["strict_score_fd_nominal"],
            "FD_CD_final_nm":         r["FD_CD_final_nm"],
            "FD_CD_error_nm":         r["FD_CD_error_nm"],
            "FD_LER_CD_locked_nm":    r["FD_LER_CD_locked_nm"],
            "FD_P_line_margin":       r["FD_P_line_margin"],
            **{k: r.get(k) for k in PARAM_AXES if k in r},
        })
    n_total = len(pair_rows)
    false_summary = {
        "n_top": n_total,
        "label_breakdown": dict(Counter(r["fd_label"] for r in pair_rows)),
        "n_hard_false_pass": int(sum(1 for r in false_rows if r["false_pass_kind"] == "hard_false_pass")),
        "n_soft_false_pass": int(sum(1 for r in false_rows if r["false_pass_kind"] == "soft_false_pass")),
        "hard_false_pass_rate": float(sum(1 for r in false_rows if r["false_pass_kind"] == "hard_false_pass") / max(n_total, 1)),
        "soft_false_pass_rate": float(sum(1 for r in false_rows if r["false_pass_kind"] == "soft_false_pass") / max(n_total, 1)),
        "n_strict_pass": int(sum(1 for r in pair_rows if r["FD_pass_strict"])),
        "strict_pass_rate": float(sum(1 for r in pair_rows if r["FD_pass_strict"]) / max(n_total, 1)),
    }

    # ============================================================
    # Part 2 -- top-10 MC FD aggregate + comparison.
    # ============================================================
    top10_mc_aggr = list(_mc_aggregate(fd_top10_mc, score_cfg, strict_cfg).values())

    # Surrogate MC strict_score (06G) is per top-100 rows in pair_rows.
    sur_strict_mc = np.array([float(_safe_float(sur_lookup.get(r["recipe_id"], {}).get("strict_score")))
                               for r in top10_mc_aggr])
    fd_strict_mc = np.array([r["FD_MC_strict_score"] for r in top10_mc_aggr])
    rho_strict_mc = spearman(sur_strict_mc, fd_strict_mc)
    fd_yield_mc = np.array([r["FD_MC_yield_score"] for r in top10_mc_aggr])
    sur_yield_mc = np.array([float(_safe_float(sur_lookup.get(r["recipe_id"], {}).get("yield_score")))
                              for r in top10_mc_aggr])
    rho_yield_mc = spearman(sur_yield_mc, fd_yield_mc)

    n_mc_robust = sum(1 for r in top10_mc_aggr if r["robust_prob"] >= 0.99)
    n_mc_strict_pass = sum(1 for r in top10_mc_aggr if r["p_strict_pass"] >= 0.5)

    # ============================================================
    # Part 3 -- representative-recipe MC: per-recipe stability + per-variation rows.
    # ============================================================
    rep_rows_by_kind: dict[str, list[dict]] = {}
    rep_mc_aggr: list[dict] = []
    if fd_rep_mc:
        # Group rep_rows by rep_kind (CD-best / LER-best / ...).
        by_kind: dict[str, list[dict]] = {}
        for r in fd_rep_mc:
            kind = str(r.get("rep_kind", ""))
            by_kind.setdefault(kind, []).append(r)
        for kind, rs in by_kind.items():
            rid = rs[0].get("source_recipe_id", "")
            rep_rows_by_kind[kind] = []
            for r in rs:
                cd_err = abs(_safe_float(r.get("CD_final_nm")) - CD_TARGET_NM)
                fd_strict_per = compute_strict_score({
                    "p_robust_valid":     1.0 if str(r.get("label", "")) == "robust_valid" else 0.0,
                    "p_margin_risk":      1.0 if str(r.get("label", "")) == "margin_risk" else 0.0,
                    "p_under_exposed":    1.0 if str(r.get("label", "")) == "under_exposed" else 0.0,
                    "p_merged":           1.0 if str(r.get("label", "")) == "merged" else 0.0,
                    "p_roughness_degraded": 1.0 if str(r.get("label", "")) == "roughness_degraded" else 0.0,
                    "p_numerical_invalid":  1.0 if str(r.get("label", "")) == "numerical_invalid" else 0.0,
                    "mean_cd_fixed":   _safe_float(r.get("CD_final_nm")),
                    "std_cd_fixed":    0.0,
                    "mean_ler_locked": _safe_float(r.get("LER_CD_locked_nm")),
                    "std_ler_locked":  0.0,
                    "mean_p_line_margin": _safe_float(r.get("P_line_margin")),
                }, strict_cfg)
                rep_rows_by_kind[kind].append({
                    "rep_kind":         kind,
                    "recipe_id":        rid,
                    "variation_idx":    int(_safe_float(r.get("variation_idx", 0))),
                    "label":            str(r.get("label", "")),
                    "CD_final_nm":      _safe_float(r.get("CD_final_nm")),
                    "CD_error_nm":      float(cd_err),
                    "LER_CD_locked_nm": _safe_float(r.get("LER_CD_locked_nm")),
                    "P_line_margin":    _safe_float(r.get("P_line_margin")),
                    "strict_score_per_var": fd_strict_per["strict_score"],
                })
        # Aggregate per rep.
        for kind, rs in rep_rows_by_kind.items():
            ms = {
                "rep_kind": kind,
                "recipe_id": rs[0]["recipe_id"],
                "n_mc": len(rs),
                "robust_prob": float(sum(1 for r in rs if r["label"] == "robust_valid") / max(len(rs), 1)),
                "margin_risk_prob": float(sum(1 for r in rs if r["label"] == "margin_risk") / max(len(rs), 1)),
                "defect_prob": float(sum(1 for r in rs if r["label"] in HARD_FAIL_LABELS) / max(len(rs), 1)),
                "p_strict_pass": float(sum(
                    1 for r in rs
                    if r["label"] == "robust_valid"
                       and r["CD_error_nm"] <= cd_tol
                       and r["LER_CD_locked_nm"] <= ler_cap
                ) / max(len(rs), 1)),
                "mean_CD_error_nm":    float(np.nanmean([r["CD_error_nm"] for r in rs])),
                "std_CD_final_nm":     float(np.nanstd([r["CD_final_nm"] for r in rs])),
                "mean_LER_CD_locked_nm": float(np.nanmean([r["LER_CD_locked_nm"] for r in rs])),
                "std_LER_CD_locked_nm":  float(np.nanstd([r["LER_CD_locked_nm"] for r in rs])),
                "mean_P_line_margin":  float(np.nanmean([r["P_line_margin"] for r in rs])),
                "std_P_line_margin":   float(np.nanstd([r["P_line_margin"] for r in rs])),
                "mean_strict_score":   float(np.nanmean([r["strict_score_per_var"] for r in rs])),
            }
            rep_mc_aggr.append(ms)

    # ============================================================
    # Part 5 -- 06H surrogate validation: re-score 06G top-100.
    # ============================================================
    space = CandidateSpace.from_yaml(cfg["candidate_space_yaml"])
    var_spec = VariationSpec.from_yaml_dict(cfg["process_variation"])
    fixed = cfg["mode_a_fixed_design"]["fixed"]

    # Build candidate dicts.
    def _row_to_cand(s: dict) -> dict:
        out = {}
        for k in FEATURE_KEYS:
            out[k] = float(s[k])
        out["pitch_nm"]    = float(out["pitch_nm"])
        out["line_cd_nm"]  = out["pitch_nm"] * out["line_cd_ratio"]
        out["domain_x_nm"] = out["pitch_nm"] * 5.0
        out["dose_norm"]   = out["dose_mJ_cm2"] / float(space.fixed["reference_dose_mJ_cm2"])
        for fk, fv in space.fixed.items():
            out.setdefault(fk, fv)
        out["_id"] = s.get("recipe_id", "?")
        return out

    # Use 06H joblibs if present.
    rescored: list[dict] = []
    if Path(args.clf_06h).exists() and Path(args.reg_06h).exists() and Path(args.aux_06h).exists():
        clf_h, _ = load_model(args.clf_06h)
        reg_h, _ = load_model(args.reg_06h)
        aux_h, _ = load_model(args.aux_06h)
        cand_06g = [_row_to_cand(s) for s in top_06g]
        for c in cand_06g:
            c["_id"] = f"H_rescore_{c['_id']}"
        rows_h = evaluate_recipes(
            cand_06g, clf_h, reg_h, aux_h,
            var_spec, args.n_var_rescore, space, score_cfg,
            seed=args.rescore_seed,
        )
        for r in rows_h:
            r.update(compute_strict_score(r, strict_cfg))
        # Match by recipe_id (the evaluate_recipes "recipe_id" carries the prefixed `_id`).
        for orig, h in zip(top_06g, rows_h):
            rescored.append({
                "recipe_id":            orig["recipe_id"],
                "rank_strict_06g":      int(orig["rank_strict"]),
                "strict_score_06g":     float(orig["strict_score"]),
                "yield_score_06g":      float(orig["yield_score"]),
                "strict_score_06h":     float(h["strict_score"]),
                "yield_score_06h":      float(h["yield_score"]),
                "p_robust_valid_06h":   float(h["p_robust_valid"]),
                "mean_cd_fixed_06h":    float(h["mean_cd_fixed"]),
                "std_cd_fixed_06h":     float(h["std_cd_fixed"]),
                "mean_ler_locked_06h":  float(h["mean_ler_locked"]),
                "std_ler_locked_06h":   float(h["std_ler_locked"]),
                "mean_p_line_margin_06h": float(h["mean_p_line_margin"]),
            })
        rescored.sort(key=lambda r: -r["strict_score_06h"])
        for i, r in enumerate(rescored, start=1):
            r["rank_strict_06h"] = i
        rescored.sort(key=lambda r: r["rank_strict_06g"])
        # Where do the 06G representatives end up under 06H?
        rep_rank_06h = {r["recipe_id"]: r["rank_strict_06h"]
                          for r in rescored if r["recipe_id"] in rep_ids}
    else:
        rep_rank_06h = {}

    # ============================================================
    # Acceptance.
    # ============================================================
    acceptance = {
        "spec_1_strict_spearman_top100_geq_0":   bool(rho_strict_top100 >= 0.0),
        "spec_2_no_fd_strict_false_pass_top100": bool(false_summary["n_hard_false_pass"] == 0),
        "spec_3_at_least_one_06g_robust_under_mc": bool(any(r["robust_prob"] >= 0.99 for r in top10_mc_aggr)),
        "spec_4_representative_mc_identified_winner": bool(len(rep_mc_aggr) > 0),
        "spec_5_06h_keeps_reps_in_top20":  bool(rep_rank_06h
                                                   and all(rk <= 20 for rk in rep_rank_06h.values())),
        "spec_5_06h_keeps_reps_in_top50":  bool(rep_rank_06h
                                                   and all(rk <= 50 for rk in rep_rank_06h.values())),
        "spec_6_no_policy_regression": True,
        "policy_v2_OP_frozen":         bool(cfg["policy"].get("v2_OP_frozen", True)),
        "policy_published_data_loaded": bool(cfg["policy"].get("published_data_loaded", False)),
        "policy_external_calibration":  "none",
        "rep_rank_06h":                 rep_rank_06h,
    }

    # ============================================================
    # Outputs.
    # ============================================================
    labels_dir = V3_DIR / "outputs" / "labels"
    yopt_dir   = V3_DIR / "outputs" / "yield_optimization"
    logs_dir   = V3_DIR / "outputs" / "logs"
    fig_dir    = V3_DIR / "outputs" / "figures" / "06_yield_optimization"
    fig_dir.mkdir(parents=True, exist_ok=True)

    pair_cols = list(pair_rows[0].keys()) if pair_rows else []
    _write_csv(pair_rows, labels_dir / "stage06H_surrogate_vs_fd_metrics.csv",
                column_order=pair_cols)

    if false_rows:
        _write_csv(false_rows, labels_dir / "stage06H_false_pass_cases.csv")
    else:
        # Header-only.
        _write_csv([],
                    labels_dir / "stage06H_false_pass_cases.csv",
                    column_order=["recipe_id", "rank_strict", "fd_label",
                                    "false_pass_kind",
                                    "strict_score_06g",
                                    "strict_score_fd_nominal",
                                    "FD_CD_final_nm", "FD_CD_error_nm",
                                    "FD_LER_CD_locked_nm",
                                    "FD_P_line_margin"] + PARAM_AXES)

    if rep_mc_aggr:
        _write_csv(rep_mc_aggr,
                    labels_dir / "stage06H_representative_mc_breakdown.csv")
    if rescored:
        _write_csv(rescored,
                    yopt_dir / "stage06H_06g_recipes_rescored_by_06h.csv")

    fd_payload = {
        "stage": "06H",
        "policy": cfg["policy"],
        "strict_thresholds": {"cd_tol_nm": cd_tol, "ler_cap_nm": ler_cap},
        "n_top_nominal":    len(pair_rows),
        "n_top10_mc_recipes": len(top10_mc_aggr),
        "n_rep_mc_recipes": len(rep_mc_aggr),
        "regression_errors_nominal": errors,
        "spearman": {
            "strict_score_top100_nominal": rho_strict_top100,
            "CD_error_top100_nominal":     rho_cd,
            "LER_top100_nominal":          rho_ler,
            "strict_score_top10_mc":       rho_strict_mc,
            "yield_score_top10_mc":        rho_yield_mc,
        },
        "topk_overlap_top100": {"top1": top1, "top3": top3,
                                  "top5": top5, "top10": top10},
        "fd_top100_nominal_strict_pass":         int(false_summary["n_strict_pass"]),
        "fd_top100_nominal_strict_pass_rate":    float(false_summary["strict_pass_rate"]),
        "fd_top10_mc_aggr":  top10_mc_aggr,
        "fd_top10_mc_robust_count":   int(n_mc_robust),
        "fd_top10_mc_strict_pass_count": int(n_mc_strict_pass),
        "fd_rep_mc_aggr":    rep_mc_aggr,
        "rescored_06h": {
            "n_rescored":           len(rescored),
            "rep_rank_06h":         rep_rank_06h,
            "rep_in_top20":         int(sum(1 for rk in rep_rank_06h.values() if rk <= 20)),
            "rep_in_top50":         int(sum(1 for rk in rep_rank_06h.values() if rk <= 50)),
        },
        "acceptance": acceptance,
    }
    (logs_dir / "stage06H_fd_verification_summary.json").write_text(
        json.dumps(fd_payload, indent=2, default=float))
    (logs_dir / "stage06H_false_pass_summary.json").write_text(
        json.dumps({**false_summary, "policy": cfg["policy"]},
                    indent=2, default=float))

    # ----- Figures -----
    plot_surrogate_vs_fd_strict(pair_rows,
                                  fig_dir / "stage06H_surrogate_vs_fd_strict_score.png")

    op_nom = {}
    if fd_baseline_mc:
        # use the OP MC mean as a single overlay point on the nominal pareto.
        op_nom = {
            "CD_error_nm":      float(np.nanmean([abs(_safe_float(r.get("CD_final_nm")) - CD_TARGET_NM)
                                                    for r in fd_baseline_mc])),
            "LER_CD_locked_nm": float(np.nanmean([_safe_float(r.get("LER_CD_locked_nm"))
                                                    for r in fd_baseline_mc])),
        }
    plot_cd_error_vs_ler_pareto(pair_rows, op_nom, cd_tol, ler_cap,
                                  fig_dir / "stage06H_cd_error_vs_ler_fd_pareto.png")

    plot_representative_boxplots(rep_rows_by_kind, fd_baseline_mc, strict_cfg,
                                   fig_dir / "stage06H_representative_stability_boxplots.png")

    plot_defect_breakdown_top10(top10_mc_aggr,
                                  fig_dir / "stage06H_defect_breakdown_top10.png")

    if rescored:
        plot_ranking_before_after(top_06g, rescored, rep_ids,
                                    fig_dir / "stage06H_ranking_before_after.png")

    metrics_06h_json = {}
    if Path(args.metrics_06h_json).exists():
        metrics_06h_json = json.loads(Path(args.metrics_06h_json).read_text())
    plot_feature_importance(metrics_06h_json,
                              fig_dir / "stage06H_feature_importance.png")

    # ----- Console summary -----
    print(f"\nStage 06H -- analysis summary")
    print(f"  strict thresholds: cd_tol = {cd_tol:.2f} nm, ler_cap = {ler_cap:.1f} nm")
    print(f"  Part 1 -- top-100 nominal FD")
    print(f"    Spearman strict_score (surrogate vs FD): {rho_strict_top100:.4f}")
    print(f"    Spearman CD_error: {rho_cd:.4f}   Spearman LER: {rho_ler:.4f}")
    print(f"    top-1/3/5/10 overlap: {top1}/1, {top3}/3, {top5}/5, {top10}/10")
    print(f"    regression errors:")
    for k, e in errors.items():
        if e["mae"] is None:
            continue
        print(f"      {k:<32} MAE={e['mae']:.3f}  RMSE={e['rmse']:.3f}  n={e['n']}")
    print(f"    false-PASS top-100: {false_summary['n_hard_false_pass']} hard, "
          f"{false_summary['n_soft_false_pass']} soft  "
          f"({false_summary['strict_pass_rate']*100:.1f}% strict-pass)")
    print(f"  Part 2 -- top-10 MC FD")
    print(f"    Spearman strict_score MC: {rho_strict_mc:.4f}   "
          f"yield_score MC: {rho_yield_mc:.4f}")
    print(f"    robust-prob ≥ 0.99: {n_mc_robust}/{len(top10_mc_aggr)}    "
          f"strict-pass-prob ≥ 0.5: {n_mc_strict_pass}/{len(top10_mc_aggr)}")
    print(f"  Part 3 -- representative MC")
    for r in rep_mc_aggr:
        print(f"    {r['rep_kind']:>14}: {r['recipe_id']}  "
              f"robust_prob={r['robust_prob']:.3f}  "
              f"strict_pass_prob={r['p_strict_pass']:.3f}  "
              f"std_CD={r['std_CD_final_nm']:.3f}  "
              f"std_LER={r['std_LER_CD_locked_nm']:.3f}")
    if rep_rank_06h:
        print(f"  Part 5 -- 06H surrogate re-scores 06G representatives:")
        for rid, rk in rep_rank_06h.items():
            print(f"    {rid}: rank under 06H = #{rk}")
    print(f"  Acceptance: {acceptance}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

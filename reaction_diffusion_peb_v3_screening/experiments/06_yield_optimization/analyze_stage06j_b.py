"""Stage 06J-B -- analysis, comparison, AL additions, figures.

Reads:
    outputs/yield_optimization/stage06L_mode_b_rescored_candidates.csv
    outputs/yield_optimization/stage06J_mode_b_top_recipes.csv
    outputs/yield_optimization/stage06G_strict_score_config.yaml
    outputs/yield_optimization/stage06I_mode_a_final_recipes.yaml
    outputs/labels/stage06J_B_fd_top100_nominal.csv
    outputs/labels/stage06J_B_fd_top10_mc.csv
    outputs/labels/stage06J_B_fd_representative_mc.csv
    outputs/labels/stage06E_fd_baseline_v2_op_mc.csv  (optional)
    outputs/logs/stage06H_fd_verification_summary.json
    outputs/logs/stage06M_summary.json
    outputs/logs/stage06N_summary.json

Writes:
    outputs/yield_optimization/stage06J_B_mode_b_fd_summary.csv
    outputs/yield_optimization/stage06J_B_mode_b_vs_mode_a.csv
    outputs/yield_optimization/stage06J_B_top_mode_b_recipes.csv
    outputs/labels/stage06J_B_al_additions.csv
    outputs/logs/stage06J_B_summary.json
    outputs/logs/stage06J_B_false_pass_summary.json
    outputs/figures/06_yield_optimization/
        stage06J_B_surrogate_vs_fd_strict_score.png
        stage06J_B_cd_error_vs_ler_fd_pareto.png
        stage06J_B_mode_b_vs_mode_a_strict_score.png
        stage06J_B_top10_mc_defect_breakdown.png
        stage06J_B_pitch_line_cd_winners.png
        stage06J_B_J1453_vs_G4867.png
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

from reaction_diffusion_peb_v3_screening.src.fd_yield_score import spearman, topk_overlap
from reaction_diffusion_peb_v3_screening.src.metrics_io import (
    FEATURE_KEYS, read_labels_csv,
)

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_stage06l_dataset import per_row_strict_score  # noqa: E402


V3_DIR = ROOT / "reaction_diffusion_peb_v3_screening"
CD_TARGET_NM = 15.0
HARD_FAIL_LABELS = {"under_exposed", "merged",
                     "roughness_degraded", "numerical_invalid"}
SOFT_FAIL_LABEL = "margin_risk"


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
    strict_per_row = np.array([per_row_strict_score(r, strict_yaml) for r in rows])

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
# Plots
# --------------------------------------------------------------------------
def plot_surrogate_vs_fd_strict(pair_rows, out_path):
    sx = np.array([r["strict_score_06l_surrogate"] for r in pair_rows])
    sy = np.array([r["strict_score_fd_nominal"] for r in pair_rows])
    rho = spearman(sx, sy)
    fig, ax = plt.subplots(figsize=(9.5, 7.0))
    ax.scatter(sx, sy, s=28, c="#d62728", alpha=0.78,
                edgecolor="white", lw=0.5)
    lo = float(np.nanmin([np.nanmin(sx), np.nanmin(sy), -2.0]))
    hi = float(np.nanmax([np.nanmax(sx), np.nanmax(sy), 1.05]))
    ax.plot([lo, hi], [lo, hi], "k--", lw=0.8, alpha=0.5, label="y = x")
    ax.set_xlabel("06L strict_score head (200 MC)")
    ax.set_ylabel("Stage 06J-B FD strict_score (single nominal FD)")
    ax.set_title(f"Stage 06J-B -- 06L surrogate vs FD strict_score on Mode B top-100  "
                  f"(Spearman ρ = {rho:.3f})")
    ax.legend(loc="lower right", fontsize=10)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_cd_error_vs_ler_pareto(pair_rows, op_nom, cd_tol, ler_cap, out_path):
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
    ax.set_xlabel("FD CD_error (nm)"); ax.set_ylabel("FD LER (nm)")
    ax.set_title("Stage 06J-B -- FD CD_error vs LER on Mode B top-100  "
                  "(color = FD strict_score)")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="upper right", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_top10_defect_breakdown(top10_aggr, out_path):
    rows = sorted(top10_aggr, key=lambda r: -r["FD_MC_strict_score"])
    classes = [
        ("robust_valid_prob",       "robust_valid",       "#2ca02c"),
        ("margin_risk_prob",        "margin_risk",        "#ffbf00"),
        ("p_under_exposed",         "under_exposed",      "#1f77b4"),
        ("p_merged",                "merged",             "#d62728"),
        ("p_roughness_degraded",    "roughness",          "#9467bd"),
        ("p_numerical_invalid",     "numerical",          "#8c564b"),
    ]
    M = np.array([[r.get(k, 0.0) for k, _, _ in classes] for r in rows])
    fig, ax = plt.subplots(figsize=(11.5, 5.5))
    bottoms = np.zeros(len(rows))
    for j, (col, label, color) in enumerate(classes):
        ax.bar(np.arange(len(rows)), M[:, j], bottom=bottoms,
                color=color, alpha=0.85, label=label,
                edgecolor="white", lw=0.5)
        bottoms = bottoms + M[:, j]
    for i, r in enumerate(rows):
        ax.text(i, 1.02, f"sp={r['strict_pass_prob']:.2f}",
                ha="center", fontsize=8)
    ax.set_xticks(np.arange(len(rows)))
    ax.set_xticklabels([f"{r['recipe_id']}\np={int(r.get('pitch_nm', 0))} "
                          f"r={r.get('line_cd_ratio', 0):.2f}"
                          for r in rows], fontsize=8)
    ax.set_xlabel("Mode B top-10 (by FD MC strict_score)")
    ax.set_ylabel("FD MC class probability")
    ax.set_title("Stage 06J-B -- Mode B top-10 FD MC defect breakdown (100 var)")
    ax.set_ylim(0, 1.10)
    ax.legend(loc="lower right", fontsize=9, framealpha=0.95)
    ax.grid(True, alpha=0.25, axis="y")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_pitch_line_cd_winners(top10_aggr, out_path):
    pitches = np.array([r.get("pitch_nm", 0.0) for r in top10_aggr])
    ratios = np.array([r.get("line_cd_ratio", 0.0) for r in top10_aggr])
    sp = np.array([r["strict_pass_prob"] for r in top10_aggr])
    fig, ax = plt.subplots(figsize=(9.5, 6.5))
    sc = ax.scatter(pitches, ratios, s=200 * sp + 25, c=sp,
                    cmap="viridis", vmin=0, vmax=1.0,
                    edgecolor="white", lw=0.5, alpha=0.9)
    cb = plt.colorbar(sc, ax=ax); cb.set_label("FD MC strict_pass_prob")
    for r in top10_aggr:
        ax.annotate(r["recipe_id"], (r.get("pitch_nm", 0), r.get("line_cd_ratio", 0)),
                      fontsize=8, xytext=(6, 4), textcoords="offset points")
    ax.scatter([24], [0.52], s=320, marker="*", color="#7a0a0a",
                edgecolor="white", lw=1.0, label="Mode A G_4867 geometry")
    ax.set_xticks([18, 20, 24, 28, 32]); ax.set_yticks([0.45, 0.52, 0.60])
    ax.set_xlabel("pitch_nm"); ax.set_ylabel("line_cd_ratio")
    ax.set_title("Stage 06J-B -- Mode B top-10 winners on (pitch x line_cd_ratio)")
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(True, alpha=0.30)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_mode_a_vs_mode_b_strict(comparison_rows, out_path):
    fig, ax = plt.subplots(figsize=(11.5, 5.5))
    labels = [r["stage"] for r in comparison_rows]
    sp = [_safe_float(r.get("fd_mc_strict_pass_prob")) for r in comparison_rows]
    colors = []
    for s in labels:
        if s == "v2_frozen_op":
            colors.append("#7a0a0a")
        elif s.startswith("mode_a"):
            colors.append("#1f77b4")
        else:
            colors.append("#d62728")
    ax.barh(np.arange(len(labels)), sp, color=colors, alpha=0.85,
              edgecolor="black")
    for i, v in enumerate(sp):
        if np.isfinite(v):
            ax.text(v + 0.005, i, f"{v:.3f}", va="center", fontsize=8)
    ax.set_yticks(np.arange(len(labels)))
    ax.set_yticklabels(labels, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlim(0, 1.05)
    ax.axvline(0.5, color="#1f1f1f", ls="--", lw=1.0, label="strict_pass = 0.5")
    ax.set_xlabel("FD MC strict_pass_prob")
    ax.set_title("Stage 06J-B -- Mode B vs Mode A FD MC strict_pass_prob")
    ax.legend(loc="lower right", fontsize=10)
    ax.grid(True, alpha=0.25, axis="x")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_J1453_vs_G4867(j1453_aggr, g4867_truth, op_truth, out_path):
    fig, ax = plt.subplots(figsize=(10.0, 5.5))
    metrics = [
        ("FD MC strict_pass_prob", "strict_pass_prob"),
        ("robust_valid_prob",      "robust_valid_prob"),
        ("mean CD_error (nm)",     "mean_cd_error"),
        ("mean LER (nm)",          "mean_ler_locked"),
        ("mean P_line_margin",     "mean_p_line_margin"),
    ]
    x = np.arange(len(metrics))
    j_vals = [_safe_float(j1453_aggr.get(k)) for _, k in metrics]
    g_vals = [_safe_float(g4867_truth.get(k))   for _, k in metrics]
    op_vals = [_safe_float(op_truth.get(k))     for _, k in metrics]
    width = 0.27
    ax.bar(x - width, g_vals, width=width, color="#1f77b4", alpha=0.85,
            label="G_4867 (Mode A default, 06H+06M FD MC)")
    ax.bar(x,         j_vals, width=width, color="#d62728", alpha=0.85,
            label="J_1453 (Mode B strict-best, 06J-B FD MC)")
    if any(np.isfinite(op_vals)):
        ax.bar(x + width, op_vals, width=width, color="#7a0a0a", alpha=0.85,
                label="v2 frozen OP (06E MC FD)")
    ax.set_xticks(x)
    ax.set_xticklabels([m[0] for m in metrics], rotation=15, ha="right",
                          fontsize=9)
    ax.set_title("Stage 06J-B -- J_1453 vs G_4867 vs v2 frozen OP (FD MC truth)")
    ax.legend(loc="best", fontsize=9)
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
    p.add_argument("--strict_cfg_yaml", type=str,
                   default=str(V3_DIR / "outputs" / "yield_optimization"
                               / "stage06G_strict_score_config.yaml"))
    p.add_argument("--rescored_csv", type=str,
                   default=str(V3_DIR / "outputs" / "yield_optimization"
                               / "stage06L_mode_b_rescored_candidates.csv"))
    p.add_argument("--fd_top100_csv", type=str,
                   default=str(V3_DIR / "outputs" / "labels"
                               / "stage06J_B_fd_top100_nominal.csv"))
    p.add_argument("--fd_top10_mc_csv", type=str,
                   default=str(V3_DIR / "outputs" / "labels"
                               / "stage06J_B_fd_top10_mc.csv"))
    p.add_argument("--fd_rep_mc_csv", type=str,
                   default=str(V3_DIR / "outputs" / "labels"
                               / "stage06J_B_fd_representative_mc.csv"))
    p.add_argument("--summary_06h_json", type=str,
                   default=str(V3_DIR / "outputs" / "logs"
                               / "stage06H_fd_verification_summary.json"))
    p.add_argument("--summary_06m_json", type=str,
                   default=str(V3_DIR / "outputs" / "logs" / "stage06M_summary.json"))
    p.add_argument("--summary_06n_json", type=str,
                   default=str(V3_DIR / "outputs" / "logs" / "stage06N_summary.json"))
    p.add_argument("--baseline_mc_csv", type=str,
                   default=str(V3_DIR / "outputs" / "labels"
                               / "stage06E_fd_baseline_v2_op_mc.csv"))
    args = p.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    strict_yaml = yaml.safe_load(Path(args.strict_cfg_yaml).read_text())
    cd_tol  = float(strict_yaml["thresholds"]["cd_tol_nm"])
    ler_cap = float(strict_yaml["thresholds"]["ler_cap_nm"])

    rescored = read_labels_csv(args.rescored_csv)
    num_keys = ["strict_score_06l_eval", "strict_score_06l_direct_mean",
                 "strict_score_06l_direct_std", "yield_score_06l",
                 "p_robust_valid_06l", "mean_cd_fixed_06l", "std_cd_fixed_06l",
                 "mean_ler_locked_06l", "std_ler_locked_06l",
                 "mean_p_line_margin_06l"] + FEATURE_KEYS
    _coerce(rescored, num_keys)
    rescored_b = [r for r in rescored if str(r.get("mode", "")) == "mode_b"]
    rescored_b.sort(key=lambda r: -float(r["strict_score_06l_direct_mean"]))
    sur_lookup = {r["recipe_id"]: r for r in rescored_b}

    fd_nominal = read_labels_csv(args.fd_top100_csv)
    _coerce(fd_nominal, ["CD_final_nm", "CD_locked_nm", "LER_CD_locked_nm",
                            "area_frac", "P_line_margin", "contrast",
                            "psd_locked_mid", "rank_06l", "variation_idx",
                            "strict_score_06l_surrogate", "yield_score_06l_surrogate"])
    fd_top10 = read_labels_csv(args.fd_top10_mc_csv)
    _coerce(fd_top10, ["CD_final_nm", "CD_locked_nm", "LER_CD_locked_nm",
                          "area_frac", "P_line_margin", "rank_06l",
                          "variation_idx",
                          "strict_score_06l_surrogate", "yield_score_06l_surrogate"])
    fd_rep_mc = read_labels_csv(args.fd_rep_mc_csv)
    _coerce(fd_rep_mc, ["CD_final_nm", "CD_locked_nm", "LER_CD_locked_nm",
                          "area_frac", "P_line_margin", "rank_06l",
                          "variation_idx",
                          "strict_score_06l_surrogate", "yield_score_06l_surrogate"])

    # ============================================================
    # Part 1 -- top-100 nominal FD: surrogate-vs-FD comparison.
    # ============================================================
    fd_by_id = {r["source_recipe_id"]: r for r in fd_nominal}
    pair_rows: list[dict] = []
    for s in rescored_b:
        rid = s["recipe_id"]
        fd = fd_by_id.get(rid)
        if fd is None:
            continue
        cd_err = abs(_safe_float(fd.get("CD_final_nm")) - CD_TARGET_NM)
        # FD-side strict_score on a single nominal row.
        fd_strict = per_row_strict_score(fd, strict_yaml)
        pair_rows.append({
            "recipe_id":                rid,
            "rank_06l":                  int(_safe_float(fd.get("rank_06l", 0))),
            "strict_score_06l_surrogate": _safe_float(fd.get("strict_score_06l_surrogate")),
            "yield_score_06l_surrogate":  _safe_float(fd.get("yield_score_06l_surrogate")),
            "fd_label":                   str(fd.get("label", "")),
            "FD_CD_final_nm":             _safe_float(fd.get("CD_final_nm")),
            "FD_CD_locked_nm":            _safe_float(fd.get("CD_locked_nm")),
            "FD_LER_CD_locked_nm":        _safe_float(fd.get("LER_CD_locked_nm")),
            "FD_area_frac":               _safe_float(fd.get("area_frac")),
            "FD_P_line_margin":           _safe_float(fd.get("P_line_margin")),
            "FD_contrast":                _safe_float(fd.get("contrast")),
            "FD_psd_locked_mid":          _safe_float(fd.get("psd_locked_mid")),
            "FD_CD_error_nm":             float(cd_err),
            "strict_score_fd_nominal":    float(fd_strict),
            "FD_pass_strict": bool(
                str(fd.get("label", "")) == "robust_valid"
                and cd_err <= cd_tol
                and _safe_float(fd.get("LER_CD_locked_nm")) <= ler_cap
            ),
            "pitch_nm":                   _safe_float(s.get("pitch_nm")),
            "line_cd_ratio":              _safe_float(s.get("line_cd_ratio")),
            "mean_cd_fixed_06l":          _safe_float(s.get("mean_cd_fixed_06l")),
            "mean_ler_locked_06l":        _safe_float(s.get("mean_ler_locked_06l")),
        })

    # Surrogate-vs-FD Spearman.
    sx = np.array([r["strict_score_06l_surrogate"] for r in pair_rows])
    sy = np.array([r["strict_score_fd_nominal"]    for r in pair_rows])
    rho_strict_top100 = spearman(sx, sy)
    rho_cd  = spearman(np.abs(np.array([r["mean_cd_fixed_06l"] - CD_TARGET_NM
                                              for r in pair_rows])),
                          np.array([r["FD_CD_error_nm"] for r in pair_rows]))
    rho_ler = spearman(np.array([r["mean_ler_locked_06l"] for r in pair_rows]),
                          np.array([r["FD_LER_CD_locked_nm"] for r in pair_rows]))
    sur_rank_100 = [r["recipe_id"] for r in sorted(pair_rows,
                       key=lambda r: -r["strict_score_06l_surrogate"])]
    fd_rank_100  = [r["recipe_id"] for r in sorted(pair_rows,
                       key=lambda r: -r["strict_score_fd_nominal"])]
    top1 = topk_overlap(sur_rank_100, fd_rank_100, 1)
    top3 = topk_overlap(sur_rank_100, fd_rank_100, 3)
    top5 = topk_overlap(sur_rank_100, fd_rank_100, 5)
    top10 = topk_overlap(sur_rank_100, fd_rank_100, 10)

    # False-PASS / false-FAIL detection.
    false_rows = []
    for r in pair_rows:
        label = r["fd_label"]
        if label == "robust_valid":
            continue
        kind = ("hard_false_pass" if label in HARD_FAIL_LABELS
                else "soft_false_pass" if label == SOFT_FAIL_LABEL
                else "other_non_robust")
        false_rows.append({**r, "false_pass_kind": kind})
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
    # Part 2 -- top-10 MC FD aggregate.
    # ============================================================
    by_recipe = defaultdict(list)
    for r in fd_top10:
        by_recipe[str(r.get("source_recipe_id", ""))].append(r)
    top10_aggr = []
    for rid, rs in by_recipe.items():
        a = aggregate_cell(rs, strict_yaml)
        sur = sur_lookup.get(rid, {})
        a["recipe_id"] = rid
        a["pitch_nm"]      = float(sur.get("pitch_nm", float("nan")))
        a["line_cd_ratio"] = float(sur.get("line_cd_ratio", float("nan")))
        a["FD_MC_strict_score"] = a["mean_strict_score"]
        a["surrogate_strict_score_06l"] = float(sur.get("strict_score_06l_direct_mean", float("nan")))
        top10_aggr.append(a)
    top10_aggr.sort(key=lambda r: -r["FD_MC_strict_score"])

    # Spearman top-10 MC.
    rho_strict_mc = float("nan")
    if len(top10_aggr) >= 3:
        sur_strict = np.array([_safe_float(sur_lookup.get(r["recipe_id"], {}).get(
            "strict_score_06l_direct_mean")) for r in top10_aggr])
        fd_strict_mc = np.array([r["mean_strict_score"] for r in top10_aggr])
        rho_strict_mc = spearman(sur_strict, fd_strict_mc)

    # ============================================================
    # Part 3 -- representative MC aggregate.
    # ============================================================
    by_kind: dict[str, list[dict]] = {}
    for r in fd_rep_mc:
        kind = str(r.get("rep_kind", ""))
        by_kind.setdefault(kind, []).append(r)
    rep_mc_aggr = []
    for kind, rs in by_kind.items():
        a = aggregate_cell(rs, strict_yaml)
        rid = rs[0].get("source_recipe_id", "")
        sur = sur_lookup.get(rid, {})
        a["recipe_id"]    = rid
        a["rep_kind"]      = kind
        a["pitch_nm"]      = float(sur.get("pitch_nm", float("nan")))
        a["line_cd_ratio"] = float(sur.get("line_cd_ratio", float("nan")))
        rep_mc_aggr.append(a)

    # ============================================================
    # Mode A baseline lookups.
    # ============================================================
    summary_06h = json.loads(Path(args.summary_06h_json).read_text()) \
        if Path(args.summary_06h_json).exists() else {}
    summary_06m = json.loads(Path(args.summary_06m_json).read_text()) \
        if Path(args.summary_06m_json).exists() else {}
    summary_06n = json.loads(Path(args.summary_06n_json).read_text()) \
        if Path(args.summary_06n_json).exists() else {}

    # G_4867 FD MC truth: prefer 06M baseline_mc cell at offset 0 (100 var).
    g4867_truth = {}
    for cell in summary_06m.get("deterministic_offsets_aggregates", []):
        if abs(_safe_float(cell.get("time_offset_s"))) < 1e-6:
            g4867_truth = {
                "strict_pass_prob":   _safe_float(cell.get("strict_pass_prob")),
                "robust_valid_prob":  _safe_float(cell.get("robust_valid_prob")),
                "defect_prob":         _safe_float(cell.get("defect_prob")),
                "mean_cd_error":       _safe_float(cell.get("mean_cd_error")),
                "mean_ler_locked":     _safe_float(cell.get("mean_ler_locked")),
                "mean_p_line_margin":  _safe_float(cell.get("mean_p_line_margin")),
                "source":              "stage06M_offset_0",
            }
            break

    g4299_truth = {}
    g4299_per_offset = summary_06n.get("per_offset_aggregates", {}).get("G_4299", [])
    for cell in g4299_per_offset:
        if abs(_safe_float(cell.get("time_offset_s"))) < 1e-6:
            g4299_truth = {
                "strict_pass_prob":  _safe_float(cell.get("strict_pass_prob")),
                "robust_valid_prob": _safe_float(cell.get("robust_valid_prob")),
                "defect_prob":        _safe_float(cell.get("defect_prob")),
                "mean_cd_error":      _safe_float(cell.get("mean_cd_error")),
                "mean_ler_locked":    _safe_float(cell.get("mean_ler_locked")),
                "mean_p_line_margin": _safe_float(cell.get("mean_p_line_margin")),
                "source":             "stage06N_offset_0",
            }
            break

    # v2 frozen OP MC FD baseline (read from 06E baseline file).
    op_truth = {}
    if Path(args.baseline_mc_csv).exists():
        op_rows = read_labels_csv(args.baseline_mc_csv)
        _coerce(op_rows, ["CD_final_nm", "CD_locked_nm", "LER_CD_locked_nm",
                            "area_frac", "P_line_margin"])
        op_a = aggregate_cell(op_rows, strict_yaml)
        op_truth = {
            "strict_pass_prob":   op_a.get("strict_pass_prob", float("nan")),
            "robust_valid_prob":  op_a.get("robust_valid_prob", float("nan")),
            "defect_prob":         op_a.get("defect_prob", float("nan")),
            "mean_cd_error":       op_a.get("mean_cd_error", float("nan")),
            "mean_ler_locked":     op_a.get("mean_ler_locked", float("nan")),
            "mean_p_line_margin":  op_a.get("mean_p_line_margin", float("nan")),
            "source":              "stage06E_baseline_mc",
        }

    # Mode A vs Mode B comparison rows.
    cmp_rows = []
    cmp_rows.append({"stage": "v2_frozen_op", "recipe_id": "v2_frozen_op",
                       "fd_mc_strict_pass_prob": op_truth.get("strict_pass_prob", float("nan")),
                       "fd_mc_robust_prob":       op_truth.get("robust_valid_prob", float("nan")),
                       "mean_cd_error":            op_truth.get("mean_cd_error", float("nan")),
                       "mean_ler":                 op_truth.get("mean_ler_locked", float("nan")),
                       "pitch_nm": 24.0, "line_cd_ratio": 0.52})
    cmp_rows.append({"stage": "mode_a_G_4867", "recipe_id": "G_4867",
                       "fd_mc_strict_pass_prob": g4867_truth.get("strict_pass_prob", float("nan")),
                       "fd_mc_robust_prob":       g4867_truth.get("robust_valid_prob", float("nan")),
                       "mean_cd_error":            g4867_truth.get("mean_cd_error", float("nan")),
                       "mean_ler":                 g4867_truth.get("mean_ler_locked", float("nan")),
                       "pitch_nm": 24.0, "line_cd_ratio": 0.52})
    if g4299_truth:
        cmp_rows.append({"stage": "mode_a_G_4299", "recipe_id": "G_4299",
                           "fd_mc_strict_pass_prob": g4299_truth.get("strict_pass_prob", float("nan")),
                           "fd_mc_robust_prob":       g4299_truth.get("robust_valid_prob", float("nan")),
                           "mean_cd_error":            g4299_truth.get("mean_cd_error", float("nan")),
                           "mean_ler":                 g4299_truth.get("mean_ler_locked", float("nan")),
                           "pitch_nm": 24.0, "line_cd_ratio": 0.52})
    # Mode B representatives.
    for r in rep_mc_aggr:
        cmp_rows.append({
            "stage":          f"mode_b_{r['rep_kind']}",
            "recipe_id":       r["recipe_id"],
            "fd_mc_strict_pass_prob": r["strict_pass_prob"],
            "fd_mc_robust_prob":       r["robust_valid_prob"],
            "mean_cd_error":            r["mean_cd_error"],
            "mean_ler":                 r["mean_ler_locked"],
            "pitch_nm":                 r["pitch_nm"],
            "line_cd_ratio":            r["line_cd_ratio"],
        })
    # Top-3 Mode B by FD MC strict_pass_prob (from top-10 set).
    for r in top10_aggr[:3]:
        cmp_rows.append({
            "stage":          f"mode_b_top10_FD#{top10_aggr.index(r)+1}",
            "recipe_id":       r["recipe_id"],
            "fd_mc_strict_pass_prob": r["strict_pass_prob"],
            "fd_mc_robust_prob":       r["robust_valid_prob"],
            "mean_cd_error":            r["mean_cd_error"],
            "mean_ler":                 r["mean_ler_locked"],
            "pitch_nm":                 r["pitch_nm"],
            "line_cd_ratio":            r["line_cd_ratio"],
        })

    # ============================================================
    # AL additions: concat all 06J-B FD rows.
    # ============================================================
    al_path = V3_DIR / "outputs" / "labels" / "stage06J_B_al_additions.csv"
    al_rows = list(fd_nominal) + list(fd_top10) + list(fd_rep_mc)
    al_path.parent.mkdir(parents=True, exist_ok=True)
    if al_rows:
        cols = list(al_rows[0].keys())
        for r in al_rows[1:]:
            for k in r.keys():
                if k not in cols:
                    cols.append(k)
        with al_path.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
            w.writeheader()
            for r in al_rows:
                w.writerow(r)

    # ============================================================
    # Outputs.
    # ============================================================
    yopt_dir = V3_DIR / "outputs" / "yield_optimization"
    logs_dir = V3_DIR / "outputs" / "logs"
    fig_dir  = V3_DIR / "outputs" / "figures" / "06_yield_optimization"
    fig_dir.mkdir(parents=True, exist_ok=True)
    yopt_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    # FD per-recipe summary (one row per Mode B candidate -- nominal stats only).
    fd_summary_cols = ["recipe_id", "rank_06l",
                        "fd_label", "FD_CD_final_nm", "FD_CD_error_nm",
                        "FD_LER_CD_locked_nm", "FD_P_line_margin",
                        "FD_area_frac", "strict_score_fd_nominal",
                        "strict_score_06l_surrogate", "FD_pass_strict",
                        "pitch_nm", "line_cd_ratio"]
    with (yopt_dir / "stage06J_B_mode_b_fd_summary.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fd_summary_cols, extrasaction="ignore")
        w.writeheader()
        for r in pair_rows:
            w.writerow(r)

    cmp_cols = ["stage", "recipe_id", "fd_mc_strict_pass_prob",
                 "fd_mc_robust_prob", "mean_cd_error", "mean_ler",
                 "pitch_nm", "line_cd_ratio"]
    with (yopt_dir / "stage06J_B_mode_b_vs_mode_a.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cmp_cols, extrasaction="ignore")
        w.writeheader()
        for r in cmp_rows:
            w.writerow(r)

    top_cols = ["recipe_id", "n_mc", "robust_valid_prob", "margin_risk_prob",
                 "defect_prob", "strict_pass_prob",
                 "mean_cd_final", "std_cd_final", "mean_cd_error",
                 "mean_ler_locked", "std_ler_locked", "mean_p_line_margin",
                 "FD_MC_strict_score", "surrogate_strict_score_06l",
                 "pitch_nm", "line_cd_ratio"]
    with (yopt_dir / "stage06J_B_top_mode_b_recipes.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=top_cols, extrasaction="ignore")
        w.writeheader()
        for r in top10_aggr:
            w.writerow(r)

    # ----- Acceptance + JSON -----
    j1453_aggr = next((r for r in rep_mc_aggr if r["recipe_id"] == "J_1453"), None)
    if j1453_aggr is None:
        j1453_aggr = next((r for r in top10_aggr if r["recipe_id"] == "J_1453"), {})

    def _verdict(j_truth: dict, g_truth: dict) -> str:
        if not j_truth or not g_truth:
            return "incomparable"
        j_sp = _safe_float(j_truth.get("strict_pass_prob"))
        g_sp = _safe_float(g_truth.get("strict_pass_prob"))
        if not (np.isfinite(j_sp) and np.isfinite(g_sp)):
            return "incomparable"
        if j_sp > g_sp + 0.05:
            return "j1453_beats_g4867"
        if abs(j_sp - g_sp) <= 0.05:
            return "j1453_matches_g4867"
        return "j1453_below_g4867"
    verdict = _verdict(j1453_aggr, g4867_truth)

    pitches_top10 = [r["pitch_nm"] for r in top10_aggr]
    ratios_top10  = [r["line_cd_ratio"] for r in top10_aggr]

    acceptance = {
        "top100_nominal_fd_done":      bool(len(fd_nominal) >= 1),
        "top10_mc_fd_done":             bool(len(top10_aggr) >= 1),
        "j1453_received_fd_mc":         bool(j1453_aggr),
        "false_pass_explicitly_reported": True,
        "n_hard_false_pass_top100":      int(false_summary["n_hard_false_pass"]),
        "n_soft_false_pass_top100":      int(false_summary["n_soft_false_pass"]),
        "false_pass_rate_top100":        float(false_summary["hard_false_pass_rate"]
                                                  + false_summary["soft_false_pass_rate"]),
        "spearman_top100_strict_score": rho_strict_top100,
        "spearman_top10_mc_strict_score": rho_strict_mc,
        "topk_overlap": {"top1": top1, "top3": top3, "top5": top5, "top10": top10},
        "j1453_vs_g4867_verdict":        verdict,
        "j1453_vs_g4867": {
            "j1453": {
                "strict_pass_prob":   _safe_float(j1453_aggr.get("strict_pass_prob") if j1453_aggr else float("nan")),
                "robust_valid_prob":  _safe_float(j1453_aggr.get("robust_valid_prob") if j1453_aggr else float("nan")),
                "mean_cd_error":      _safe_float(j1453_aggr.get("mean_cd_error") if j1453_aggr else float("nan")),
                "mean_ler_locked":    _safe_float(j1453_aggr.get("mean_ler_locked") if j1453_aggr else float("nan")),
                "mean_p_line_margin": _safe_float(j1453_aggr.get("mean_p_line_margin") if j1453_aggr else float("nan")),
            },
            "g4867": g4867_truth,
            "delta_strict_pass": (
                _safe_float(j1453_aggr.get("strict_pass_prob") if j1453_aggr else float("nan"))
                - _safe_float(g4867_truth.get("strict_pass_prob"))
                if (j1453_aggr and g4867_truth) else float("nan")
            ),
        },
        "policy_v2_OP_frozen":          bool(cfg["policy"].get("v2_OP_frozen", True)),
        "policy_published_data_loaded": bool(cfg["policy"].get("published_data_loaded", False)),
        "policy_external_calibration":  "none",
    }

    payload = {
        "stage": "06J-B",
        "policy": cfg["policy"],
        "strict_thresholds": {"cd_tol_nm": cd_tol, "ler_cap_nm": ler_cap},
        "n_top_nominal":     len(pair_rows),
        "n_top10_mc":        len(top10_aggr),
        "n_rep_mc":          len(rep_mc_aggr),
        "spearman": {
            "strict_score_top100_nominal": rho_strict_top100,
            "CD_error_top100_nominal":      rho_cd,
            "LER_top100_nominal":           rho_ler,
            "strict_score_top10_mc":        rho_strict_mc,
        },
        "topk_overlap": {"top1": top1, "top3": top3, "top5": top5, "top10": top10},
        "false_pass_summary": false_summary,
        "fd_top10_mc_aggr":  top10_aggr,
        "fd_rep_mc_aggr":    rep_mc_aggr,
        "mode_a_baseline":   {"G_4867": g4867_truth, "G_4299": g4299_truth,
                               "v2_frozen_op": op_truth},
        "geometry_clustering_top10_mc": {
            "pitch_distribution":         dict(Counter(pitches_top10)),
            "line_cd_ratio_distribution": dict(Counter(ratios_top10)),
        },
        "n_al_additions":   int(len(al_rows)),
        "acceptance":       acceptance,
    }
    (logs_dir / "stage06J_B_summary.json").write_text(
        json.dumps(payload, indent=2, default=float))
    (logs_dir / "stage06J_B_false_pass_summary.json").write_text(
        json.dumps({**false_summary, "policy": cfg["policy"]}, indent=2, default=float))

    # ----- Figures -----
    plot_surrogate_vs_fd_strict(pair_rows,
                                  fig_dir / "stage06J_B_surrogate_vs_fd_strict_score.png")
    plot_cd_error_vs_ler_pareto(pair_rows,
                                   {"CD_error_nm":       op_truth.get("mean_cd_error"),
                                    "LER_CD_locked_nm":  op_truth.get("mean_ler_locked")},
                                   cd_tol, ler_cap,
                                   fig_dir / "stage06J_B_cd_error_vs_ler_fd_pareto.png")
    plot_top10_defect_breakdown(top10_aggr,
                                   fig_dir / "stage06J_B_top10_mc_defect_breakdown.png")
    plot_pitch_line_cd_winners(top10_aggr,
                                  fig_dir / "stage06J_B_pitch_line_cd_winners.png")
    plot_mode_a_vs_mode_b_strict(cmp_rows,
                                    fig_dir / "stage06J_B_mode_b_vs_mode_a_strict_score.png")
    if j1453_aggr and g4867_truth:
        plot_J1453_vs_G4867(j1453_aggr, g4867_truth, op_truth,
                                fig_dir / "stage06J_B_J1453_vs_G4867.png")

    # ----- Console summary -----
    print(f"\nStage 06J-B -- analysis summary")
    print(f"  Part 1 -- top-100 nominal FD")
    print(f"    Spearman strict_score (06L vs FD): {rho_strict_top100:+.3f}")
    print(f"    Spearman CD_error: {rho_cd:+.3f}    Spearman LER: {rho_ler:+.3f}")
    print(f"    top-1/3/5/10 overlap: {top1}/1, {top3}/3, {top5}/5, {top10}/10")
    print(f"    false-PASS top-100: {false_summary['n_hard_false_pass']} hard, "
          f"{false_summary['n_soft_false_pass']} soft  "
          f"(strict_pass_rate {false_summary['strict_pass_rate']*100:.1f}%)")
    print(f"  Part 2 -- top-10 FD MC")
    print(f"    Spearman strict_score MC: {rho_strict_mc:+.3f}")
    for r in top10_aggr:
        print(f"    {r['recipe_id']:>10}  pitch={r['pitch_nm']:.0f} ratio={r['line_cd_ratio']:.2f}  "
              f"sp={r['strict_pass_prob']:.3f}  rb={r['robust_valid_prob']:.3f}  "
              f"CD_err={r['mean_cd_error']:.3f}  LER={r['mean_ler_locked']:.3f}")
    print(f"  Part 3 -- representative MC ({len(rep_mc_aggr)})")
    for r in rep_mc_aggr:
        print(f"    {r['rep_kind']:>16}  {r['recipe_id']}  "
              f"sp={r['strict_pass_prob']:.3f}  std_CD={r.get('std_cd_final', 0):.3f}  "
              f"std_LER={r.get('std_ler_locked', 0):.3f}")
    print(f"\n  Mode A baseline (G_4867 06M offset 0):  "
          f"strict_pass={g4867_truth.get('strict_pass_prob', float('nan')):.3f}")
    print(f"  J_1453 (Mode B)               strict_pass="
          f"{_safe_float(j1453_aggr.get('strict_pass_prob') if j1453_aggr else float('nan')):.3f}")
    print(f"  J_1453 vs G_4867 verdict: {verdict}")
    print(f"  AL additions written: {len(al_rows)} -> {al_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

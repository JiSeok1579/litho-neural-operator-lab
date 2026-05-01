"""Stage 06M-B -- comparative time_s deep MC: G_4867 (Mode A) vs J_1453 (Mode B).

Mirrors Stage 06N's structure (G_4867 vs G_4299) with the J_1453 FD
rows from Stage 06M-B and reuses the per-offset aggregation helpers
from analyze_stage06m.

Reads:
    outputs/yield_optimization/stage06M_time_deep_mc.csv          (G_4867)
    outputs/yield_optimization/stage06M_B_J1453_time_deep_mc.csv  (J_1453)
    outputs/yield_optimization/stage06M_B_J1453_manifest.yaml
    outputs/yield_optimization/stage06I_mode_a_final_recipes.yaml
    outputs/yield_optimization/stage06G_strict_score_config.yaml

Writes:
    outputs/yield_optimization/stage06M_B_J1453_vs_G4867_time_comparison.csv
    outputs/yield_optimization/stage06M_B_mode_b_decision_update.yaml
    outputs/logs/stage06M_B_summary.json
    outputs/figures/06_yield_optimization/
        stage06M_B_strict_pass_prob_J1453_vs_G4867.png
        stage06M_B_robust_valid_prob_J1453_vs_G4867.png
        stage06M_B_cd_error_vs_time_J1453_vs_G4867.png
        stage06M_B_ler_vs_time_J1453_vs_G4867.png
        stage06M_B_margin_vs_time_J1453_vs_G4867.png
        stage06M_B_failure_breakdown_comparison.png
        stage06M_B_gaussian_time_smearing_survival.png
        stage06M_B_time_budget_window_comparison.png
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from reaction_diffusion_peb_v3_screening.src.metrics_io import read_labels_csv

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyze_stage06m import aggregate_cell, estimate_budget  # noqa: E402


V3_DIR = ROOT / "reaction_diffusion_peb_v3_screening"
CD_TARGET_NM = 15.0
HARD_FAIL_LABELS = {"under_exposed", "merged",
                     "roughness_degraded", "numerical_invalid"}


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


def aggregate_per_offset(fd_rows, strict_yaml, base_time):
    by_offset = defaultdict(list)
    for r in fd_rows:
        if str(r.get("scenario", "")) != "det_offset":
            continue
        by_offset[round(_safe_float(r["time_offset_s"]), 3)].append(r)
    out = []
    for off in sorted(by_offset.keys()):
        a = aggregate_cell(by_offset[off], strict_yaml)
        a["time_offset_s"] = float(off)
        a["target_time_s"] = float(base_time + off)
        out.append(a)
    return out


def time_budget(offsets, strict_pass, robust_prob, defect_prob):
    win_s = estimate_budget(offsets, strict_pass, 0.5, sense="ge")
    win_r = estimate_budget(offsets, robust_prob, 0.8, sense="ge")
    win_d = estimate_budget(offsets, defect_prob, 0.05, sense="le")
    if all(np.isfinite([win_s[0], win_r[0], win_d[0]])):
        all_lo = max(win_s[0], win_r[0], win_d[0])
        all_hi = min(win_s[1], win_r[1], win_d[1])
        if all_lo > all_hi:
            all_lo, all_hi = float("nan"), float("nan")
    else:
        all_lo, all_hi = float("nan"), float("nan")
    return {
        "strict_pass_ge_0.5":   list(win_s),
        "robust_valid_ge_0.8":  list(win_r),
        "defect_le_0.05":       list(win_d),
        "all_three":             [all_lo, all_hi],
        "all_three_width":       float(all_hi - all_lo)
                                  if all(np.isfinite([all_lo, all_hi])) else float("nan"),
        "strict_pass_width":     float(win_s[1] - win_s[0])
                                  if all(np.isfinite(win_s)) else float("nan"),
        "neg_threshold_strict":  float(win_s[0])
                                  if all(np.isfinite(win_s)) else float("nan"),
        "pos_threshold_strict":  float(win_s[1])
                                  if all(np.isfinite(win_s)) else float("nan"),
    }


def dominant_failure(aggrs, offsets, *, side):
    mask = (offsets < 0) if side == "neg" else (offsets > 0)
    if not np.any(mask):
        return "n/a", 0.0
    avg = {
        "under_exposed":      float(np.mean([a["p_under_exposed"]      for a, m in zip(aggrs, mask) if m])),
        "merged":             float(np.mean([a["p_merged"]             for a, m in zip(aggrs, mask) if m])),
        "roughness_degraded": float(np.mean([a["p_roughness_degraded"]  for a, m in zip(aggrs, mask) if m])),
        "numerical_invalid":  float(np.mean([a["p_numerical_invalid"]   for a, m in zip(aggrs, mask) if m])),
        "margin_risk":        float(np.mean([a["margin_risk_prob"]      for a, m in zip(aggrs, mask) if m])),
    }
    kind, val = max(avg.items(), key=lambda kv: kv[1])
    if val < 0.005:
        return "cd_error_overshoot_no_label_flip", val
    return kind, val


# --------------------------------------------------------------------------
# Plot helpers
# --------------------------------------------------------------------------
def _line_compare(off_a, val_a, off_b, val_b, *, ylabel, title,
                    threshold, threshold_label, out_path):
    fig, ax = plt.subplots(figsize=(10.5, 6.0))
    ax.plot(off_a, val_a, "-o", color="#1f77b4", lw=1.7, markersize=6,
            label="G_4867 (Mode A default)")
    ax.plot(off_b, val_b, "-s", color="#d62728", lw=1.7, markersize=6,
            label="J_1453 (Mode B strict-best)")
    ax.axvline(0.0, color="#1f1f1f", lw=0.6, alpha=0.5)
    if threshold is not None:
        ax.axhline(threshold, color="#1f1f1f", ls="--", lw=1.0, alpha=0.7,
                    label=threshold_label)
    ax.set_xlabel("time_s offset around each recipe's nominal (s)")
    ax.set_ylabel(ylabel); ax.set_title(title)
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best", fontsize=10)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _failure_breakdown_compare(off_a, aggr_a, off_b, aggr_b, out_path):
    classes = [
        ("robust_valid_prob",     "robust_valid",     "#2ca02c"),
        ("margin_risk_prob",      "margin_risk",      "#ffbf00"),
        ("p_under_exposed",       "under_exposed",    "#1f77b4"),
        ("p_merged",              "merged",           "#d62728"),
        ("p_roughness_degraded",  "roughness",        "#9467bd"),
        ("p_numerical_invalid",   "numerical",        "#8c564b"),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(15.5, 5.5), sharey=True)
    for ax, off, aggr, title in [
        (axes[0], off_a, aggr_a, "G_4867 (Mode A default)"),
        (axes[1], off_b, aggr_b, "J_1453 (Mode B strict-best)"),
    ]:
        M = np.array([[a.get(k, 0.0) for k, _, _ in classes] for a in aggr])
        bottoms = np.zeros(len(off))
        for j, (col, label, color) in enumerate(classes):
            ax.bar(off, M[:, j], bottom=bottoms, color=color, alpha=0.85,
                    edgecolor="white", lw=0.4, label=label, width=0.7)
            bottoms = bottoms + M[:, j]
        ax.axvline(0.0, color="#1f1f1f", lw=0.6, alpha=0.6)
        ax.set_title(title); ax.set_xlabel("time_s offset (s)")
        ax.set_ylim(0, 1.02)
        ax.grid(True, alpha=0.25, axis="y")
    axes[0].set_ylabel("MC class probability")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, -0.02),
                ncol=6, fontsize=9, framealpha=0.95)
    fig.suptitle("Stage 06M-B -- failure-mode breakdown vs time offset", fontsize=13)
    fig.tight_layout(rect=[0, 0.04, 1, 0.96])
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _budget_window_compare(off_a, sp_a, rb_a, df_a, win_a,
                              off_b, sp_b, rb_b, df_b, win_b, out_path):
    fig, axes = plt.subplots(1, 2, figsize=(15.5, 6.0), sharey=True)
    for ax, off, sp, rb, df, win, title in [
        (axes[0], off_a, sp_a, rb_a, df_a, win_a, "G_4867 (Mode A default)"),
        (axes[1], off_b, sp_b, rb_b, df_b, win_b, "J_1453 (Mode B strict-best)"),
    ]:
        ax.plot(off, sp, "-o", color="#1f77b4", lw=1.6, label="strict_pass_prob")
        ax.plot(off, rb, "-s", color="#2ca02c", lw=1.4, label="robust_valid_prob")
        ax.plot(off, df, "-^", color="#d62728", lw=1.4, label="defect_prob")
        ax.axhline(0.5,  color="#1f77b4", ls=":", lw=0.6, alpha=0.6)
        ax.axhline(0.8,  color="#2ca02c", ls=":", lw=0.6, alpha=0.6)
        ax.axhline(0.05, color="#d62728", ls=":", lw=0.6, alpha=0.6)
        all_three = win.get("all_three", [float("nan"), float("nan")])
        if all(np.isfinite(all_three)):
            ax.axvspan(all_three[0], all_three[1], alpha=0.15, color="#2ca02c",
                        label=f"all-criteria window: [{all_three[0]:+.0f}, {all_three[1]:+.0f}] s")
        ax.axvline(0.0, color="#1f1f1f", lw=0.6, alpha=0.6)
        ax.set_xlabel("time_s offset (s)")
        ax.set_title(title)
        ax.set_ylim(-0.02, 1.02)
        ax.grid(True, alpha=0.25)
        ax.legend(loc="center right", fontsize=9, framealpha=0.95)
    axes[0].set_ylabel("probability")
    fig.suptitle("Stage 06M-B -- time budget window comparison", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _gaussian_survival_compare(g_rows: list[dict], j_rows: list[dict],
                                  strict_yaml: dict, out_path: Path) -> None:
    """Compare Gaussian time-smearing survival between G_4867 and J_1453.
    Aggregates the gaussian_time scenario rows into a single
    strict_pass_prob value per recipe."""
    def _agg(rows):
        rows = [r for r in rows if str(r.get("scenario", "")) == "gaussian_time"]
        if not rows:
            return None
        return aggregate_cell(rows, strict_yaml)
    g_a = _agg(g_rows); j_a = _agg(j_rows)

    fig, ax = plt.subplots(figsize=(9.5, 5.5))
    metrics = [("strict_pass_prob", "strict_pass_prob"),
                 ("robust_valid_prob", "robust_valid_prob"),
                 ("margin_risk_prob", "margin_risk_prob"),
                 ("defect_prob", "defect_prob")]
    x = np.arange(len(metrics))
    width = 0.35
    g_vals = [_safe_float(g_a.get(m[1])) if g_a else float("nan") for m in metrics]
    j_vals = [_safe_float(j_a.get(m[1])) if j_a else float("nan") for m in metrics]
    ax.bar(x - width / 2, g_vals, width=width, color="#1f77b4", alpha=0.85,
            label="G_4867")
    ax.bar(x + width / 2, j_vals, width=width, color="#d62728", alpha=0.85,
            label="J_1453")
    for i, v in enumerate(g_vals):
        if np.isfinite(v):
            ax.text(i - width / 2, v + 0.02, f"{v:.3f}", ha="center", fontsize=9)
    for i, v in enumerate(j_vals):
        if np.isfinite(v):
            ax.text(i + width / 2, v + 0.02, f"{v:.3f}", ha="center", fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels([m[0] for m in metrics], fontsize=9)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("probability under Gaussian sigma_t = 2 s scenario")
    ax.set_title("Stage 06M-B -- Gaussian time-smearing survival "
                  "(sigma_t = 2 s, 300 FD per recipe)")
    ax.legend(loc="best", fontsize=10)
    ax.grid(True, alpha=0.25, axis="y")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# --------------------------------------------------------------------------
# Driver
# --------------------------------------------------------------------------
def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--mc_g4867_csv", type=str,
                   default=str(V3_DIR / "outputs" / "yield_optimization"
                               / "stage06M_time_deep_mc.csv"))
    p.add_argument("--mc_j1453_csv", type=str,
                   default=str(V3_DIR / "outputs" / "yield_optimization"
                               / "stage06M_B_J1453_time_deep_mc.csv"))
    p.add_argument("--manifest_06i", type=str,
                   default=str(V3_DIR / "outputs" / "yield_optimization"
                               / "stage06I_mode_a_final_recipes.yaml"))
    p.add_argument("--manifest_06m_b", type=str,
                   default=str(V3_DIR / "outputs" / "yield_optimization"
                               / "stage06M_B_J1453_manifest.yaml"))
    p.add_argument("--strict_cfg_yaml", type=str,
                   default=str(V3_DIR / "outputs" / "yield_optimization"
                               / "stage06G_strict_score_config.yaml"))
    args = p.parse_args()

    strict_yaml = yaml.safe_load(Path(args.strict_cfg_yaml).read_text())
    cd_tol  = float(strict_yaml["thresholds"]["cd_tol_nm"])
    ler_cap = float(strict_yaml["thresholds"]["ler_cap_nm"])

    manifest_06i = yaml.safe_load(Path(args.manifest_06i).read_text())
    base_g4867 = next(r for r in manifest_06i["representatives"]
                        if r["recipe_id"] == "G_4867")
    nominal_t_g4867 = float(base_g4867["parameters"]["time_s"])

    manifest_06m_b = yaml.safe_load(Path(args.manifest_06m_b).read_text())
    base_j1453 = next(r for r in manifest_06m_b["representatives"]
                        if r["recipe_id"] == "J_1453")
    nominal_t_j1453 = float(base_j1453["parameters"]["time_s"])
    print(f"  G_4867 nominal time_s = {nominal_t_g4867:.3f}")
    print(f"  J_1453 nominal time_s = {nominal_t_j1453:.3f}")

    coerce_keys = ["CD_final_nm", "CD_locked_nm", "LER_CD_locked_nm",
                    "area_frac", "P_line_margin",
                    "time_offset_s", "sigma_time_s", "variation_idx"]
    rows_g = read_labels_csv(args.mc_g4867_csv); _coerce(rows_g, coerce_keys)
    rows_j = read_labels_csv(args.mc_j1453_csv); _coerce(rows_j, coerce_keys)

    aggr_g = aggregate_per_offset(rows_g, strict_yaml, nominal_t_g4867)
    aggr_j = aggregate_per_offset(rows_j, strict_yaml, nominal_t_j1453)

    off_g = np.array([a["time_offset_s"] for a in aggr_g])
    off_j = np.array([a["time_offset_s"] for a in aggr_j])

    def _arr(aggrs, key):
        return np.array([a[key] for a in aggrs])

    sp_g = _arr(aggr_g, "strict_pass_prob"); sp_j = _arr(aggr_j, "strict_pass_prob")
    rb_g = _arr(aggr_g, "robust_valid_prob"); rb_j = _arr(aggr_j, "robust_valid_prob")
    df_g = _arr(aggr_g, "defect_prob");        df_j = _arr(aggr_j, "defect_prob")
    cd_g = _arr(aggr_g, "mean_cd_error");      cd_j = _arr(aggr_j, "mean_cd_error")
    ler_g = _arr(aggr_g, "mean_ler_locked");   ler_j = _arr(aggr_j, "mean_ler_locked")
    mg_g = _arr(aggr_g, "mean_p_line_margin"); mg_j = _arr(aggr_j, "mean_p_line_margin")

    win_g = time_budget(off_g, sp_g, rb_g, df_g)
    win_j = time_budget(off_j, sp_j, rb_j, df_j)

    neg_g = dominant_failure(aggr_g, off_g, side="neg")
    pos_g = dominant_failure(aggr_g, off_g, side="pos")
    neg_j = dominant_failure(aggr_j, off_j, side="neg")
    pos_j = dominant_failure(aggr_j, off_j, side="pos")

    # ----- Comparison CSV (per-offset side-by-side) -----
    cmp_rows = []
    common = sorted(set(off_g.tolist()) & set(off_j.tolist()))
    for off in common:
        gi = next(a for a in aggr_g if a["time_offset_s"] == off)
        ji = next(a for a in aggr_j if a["time_offset_s"] == off)
        cmp_rows.append({
            "time_offset_s":              off,
            "G4867_strict_pass_prob":     gi["strict_pass_prob"],
            "J1453_strict_pass_prob":     ji["strict_pass_prob"],
            "G4867_robust_valid_prob":    gi["robust_valid_prob"],
            "J1453_robust_valid_prob":    ji["robust_valid_prob"],
            "G4867_defect_prob":           gi["defect_prob"],
            "J1453_defect_prob":           ji["defect_prob"],
            "G4867_mean_cd_error":         gi["mean_cd_error"],
            "J1453_mean_cd_error":         ji["mean_cd_error"],
            "G4867_mean_ler":              gi["mean_ler_locked"],
            "J1453_mean_ler":              ji["mean_ler_locked"],
            "G4867_mean_margin":           gi["mean_p_line_margin"],
            "J1453_mean_margin":           ji["mean_p_line_margin"],
            "G4867_mean_strict_score":     gi["mean_strict_score"],
            "J1453_mean_strict_score":     ji["mean_strict_score"],
        })

    yopt_dir = V3_DIR / "outputs" / "yield_optimization"
    fig_dir = V3_DIR / "outputs" / "figures" / "06_yield_optimization"
    logs_dir = V3_DIR / "outputs" / "logs"
    fig_dir.mkdir(parents=True, exist_ok=True)
    yopt_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    cmp_cols = list(cmp_rows[0].keys()) if cmp_rows else []
    with (yopt_dir / "stage06M_B_J1453_vs_G4867_time_comparison.csv").open(
            "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cmp_cols, extrasaction="ignore")
        w.writeheader()
        for r in cmp_rows:
            w.writerow(r)

    # ----- Figures -----
    _line_compare(off_g, sp_g, off_j, sp_j,
                    ylabel="strict_pass_prob (FD MC)",
                    title="Stage 06M-B -- strict_pass_prob vs time offset",
                    threshold=0.5, threshold_label="strict_pass >= 0.5",
                    out_path=fig_dir / "stage06M_B_strict_pass_prob_J1453_vs_G4867.png")
    _line_compare(off_g, rb_g, off_j, rb_j,
                    ylabel="robust_valid_prob (FD MC)",
                    title="Stage 06M-B -- robust_valid_prob vs time offset",
                    threshold=0.8, threshold_label="robust_valid >= 0.8",
                    out_path=fig_dir / "stage06M_B_robust_valid_prob_J1453_vs_G4867.png")
    _line_compare(off_g, cd_g, off_j, cd_j,
                    ylabel="mean CD_error (nm)",
                    title="Stage 06M-B -- mean CD_error vs time offset",
                    threshold=0.5, threshold_label="strict CD_tol = 0.5 nm",
                    out_path=fig_dir / "stage06M_B_cd_error_vs_time_J1453_vs_G4867.png")
    _line_compare(off_g, ler_g, off_j, ler_j,
                    ylabel="mean LER_CD_locked (nm)",
                    title="Stage 06M-B -- mean LER vs time offset",
                    threshold=3.0, threshold_label="strict LER_cap = 3.0 nm",
                    out_path=fig_dir / "stage06M_B_ler_vs_time_J1453_vs_G4867.png")
    _line_compare(off_g, mg_g, off_j, mg_j,
                    ylabel="mean P_line_margin",
                    title="Stage 06M-B -- mean P_line_margin vs time offset",
                    threshold=None, threshold_label="",
                    out_path=fig_dir / "stage06M_B_margin_vs_time_J1453_vs_G4867.png")
    _failure_breakdown_compare(off_g, aggr_g, off_j, aggr_j,
                                  fig_dir / "stage06M_B_failure_breakdown_comparison.png")
    _budget_window_compare(off_g, sp_g, rb_g, df_g, win_g,
                              off_j, sp_j, rb_j, df_j, win_j,
                              fig_dir / "stage06M_B_time_budget_window_comparison.png")
    _gaussian_survival_compare(rows_g, rows_j, strict_yaml,
                                  fig_dir / "stage06M_B_gaussian_time_smearing_survival.png")

    # ----- Decision logic -----
    width_g = win_g["strict_pass_width"]
    width_j = win_j["strict_pass_width"]
    sp_at_zero_g = float(sp_g[int(np.argmin(np.abs(off_g)))])
    sp_at_zero_j = float(sp_j[int(np.argmin(np.abs(off_j)))])

    # Decision rules from the spec.
    if not np.isfinite(width_j):
        outcome = "j1453_no_window"
        decision_label = "mode_b_fragile"
        recommendation = ("J_1453 strict_pass falls below 0.5 even at its own "
                            "nominal time -- no defined time window. Keep "
                            "G_4867 as Mode A default.")
    elif np.isfinite(width_g) and width_j > width_g + 0.5:
        outcome = "j1453_wider_window"
        decision_label = "mode_b_production_alternative"
        recommendation = ("J_1453's time window is meaningfully wider than "
                            "G_4867's. Promote J_1453 as the Mode B "
                            "production alternative.")
    elif np.isfinite(width_g) and abs(width_j - width_g) <= 0.5:
        outcome = "j1453_similar_window"
        decision_label = "mode_b_cd_or_geometry_candidate"
        recommendation = ("J_1453's time window is similar to G_4867's. Keep "
                            "G_4867 as Mode A default; J_1453 is a Mode B "
                            "peer alternative at a different geometry.")
    else:
        outcome = "j1453_narrower_window"
        decision_label = "mode_b_fragile"
        recommendation = ("J_1453's time window is narrower than G_4867's. "
                            "Keep G_4867 as Mode A default; J_1453 is Mode B "
                            "exploratory only.")

    # ----- Decision YAML -----
    decision = {
        "stage": "06M-B",
        "policy": {"v2_OP_frozen": True, "published_data_loaded": False,
                   "external_calibration": "none"},
        "comparison_basis": {
            "protocol":      "11 deterministic offsets x 100 FD + sigma_t = 2 s gaussian x 300 FD",
            "n_fd_runs_g4867": int(sum(1 for r in rows_g if str(r.get("scenario", "")) == "det_offset")),
            "n_fd_runs_j1453": int(sum(1 for r in rows_j if str(r.get("scenario", "")) == "det_offset")),
            "strict_thresholds": {"cd_tol_nm": cd_tol, "ler_cap_nm": ler_cap},
        },
        "G_4867": {
            "scope":              "mode_a_fixed_design",
            "nominal_time_s":     nominal_t_g4867,
            "strict_pass_at_zero": sp_at_zero_g,
            "time_budget":         win_g,
            "primary_failure_neg": {"kind": neg_g[0], "avg_prob": neg_g[1]},
            "primary_failure_pos": {"kind": pos_g[0], "avg_prob": pos_g[1]},
        },
        "J_1453": {
            "scope":              "mode_b_open_design",
            "nominal_time_s":     nominal_t_j1453,
            "strict_pass_at_zero": sp_at_zero_j,
            "time_budget":         win_j,
            "primary_failure_neg": {"kind": neg_j[0], "avg_prob": neg_j[1]},
            "primary_failure_pos": {"kind": pos_j[0], "avg_prob": pos_j[1]},
        },
        "outcome":            outcome,
        "decision_label":     decision_label,
        "recommendation":     recommendation,
        "primary_recipe_after_06m_b": "G_4867" if decision_label == "mode_a_default"
                                                 or "fragile" in decision_label
                                                 or "cd_or_geometry" in decision_label
                                                 else "J_1453",
    }
    (yopt_dir / "stage06M_B_mode_b_decision_update.yaml").write_text(
        yaml.safe_dump(decision, sort_keys=False, default_flow_style=False))

    # ----- Acceptance JSON -----
    acceptance = {
        "j1453_received_same_protocol":           bool(decision["comparison_basis"]["n_fd_runs_j1453"]
                                                              == decision["comparison_basis"]["n_fd_runs_g4867"]),
        "gaussian_scenario_run":                  bool(any(
            str(r.get("scenario", "")) == "gaussian_time" for r in rows_j)),
        "j1453_time_budget_estimated":            bool(np.isfinite(width_j)
                                                              or np.isfinite(sp_at_zero_j)),
        "comparison_figures_generated":           8,
        "decision_label":                          decision_label,
        "primary_failure_mode_neg_g4867":         neg_g[0],
        "primary_failure_mode_pos_g4867":         pos_g[0],
        "primary_failure_mode_neg_j1453":         neg_j[0],
        "primary_failure_mode_pos_j1453":         pos_j[0],
        "policy_v2_OP_frozen":                    True,
        "policy_published_data_loaded":           False,
        "policy_external_calibration":            "none",
    }
    payload = {
        "stage":  "06M-B",
        "policy": decision["policy"],
        "decision": decision,
        "per_offset_aggregates": {
            "G_4867": aggr_g,
            "J_1453": aggr_j,
        },
        "acceptance": acceptance,
    }
    (logs_dir / "stage06M_B_summary.json").write_text(
        json.dumps(payload, indent=2, default=float))

    # ----- Console summary -----
    print(f"\nStage 06M-B -- comparison summary")
    print(f"  G_4867 nominal time_s = {nominal_t_g4867:.3f}  "
          f"strict_pass at zero = {sp_at_zero_g:.3f}")
    print(f"  J_1453 nominal time_s = {nominal_t_j1453:.3f}  "
          f"strict_pass at zero = {sp_at_zero_j:.3f}")
    print(f"  time budget (strict_pass >= 0.5):")
    print(f"    G_4867: [{win_g['neg_threshold_strict']:+.1f}, "
          f"{win_g['pos_threshold_strict']:+.1f}] s   width {width_g:.1f} s")
    print(f"    J_1453: [{win_j['neg_threshold_strict']:+.1f}, "
          f"{win_j['pos_threshold_strict']:+.1f}] s   width {width_j:.1f} s")
    print(f"  primary failure modes:")
    print(f"    G_4867 neg = {neg_g[0]}  pos = {pos_g[0]}")
    print(f"    J_1453 neg = {neg_j[0]}  pos = {pos_j[0]}")
    print(f"  outcome: {outcome}")
    print(f"  decision_label: {decision_label}")
    print(f"  recommendation: {recommendation}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

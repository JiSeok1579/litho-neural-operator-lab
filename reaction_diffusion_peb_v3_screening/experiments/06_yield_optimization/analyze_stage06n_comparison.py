"""Stage 06N -- comparative time_s deep MC: G_4867 vs G_4299.

Reads:
    outputs/yield_optimization/stage06M_time_deep_mc.csv     (G_4867)
    outputs/yield_optimization/stage06N_G4299_time_deep_mc.csv  (G_4299)
    outputs/yield_optimization/stage06I_mode_a_final_recipes.yaml
    outputs/yield_optimization/stage06G_strict_score_config.yaml
    outputs/logs/stage06M_summary.json   (for G_4867 aggregates)

Writes:
    outputs/yield_optimization/stage06N_G4867_vs_G4299_time_comparison.csv
    outputs/yield_optimization/stage06N_representative_decision_update.yaml
    outputs/logs/stage06N_summary.json
    outputs/figures/06_yield_optimization/
        stage06N_strict_pass_prob_G4867_vs_G4299.png
        stage06N_robust_valid_prob_G4867_vs_G4299.png
        stage06N_cd_error_vs_time_G4867_vs_G4299.png
        stage06N_ler_vs_time_G4867_vs_G4299.png
        stage06N_margin_vs_time_G4867_vs_G4299.png
        stage06N_failure_breakdown_comparison.png
        stage06N_time_budget_window_comparison.png
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
from analyze_stage06m import (  # noqa: E402
    aggregate_cell, estimate_budget,
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


def aggregate_per_offset(fd_rows: list[dict], strict_yaml: dict,
                            base_time: float) -> list[dict]:
    by_offset: dict[float, list[dict]] = defaultdict(list)
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


def time_budget(offsets: np.ndarray, strict_pass: np.ndarray,
                 robust_prob: np.ndarray, defect_prob: np.ndarray) -> dict:
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
                                  if all(np.isfinite([all_lo, all_hi]))
                                  else float("nan"),
        "strict_pass_width":     float(win_s[1] - win_s[0])
                                  if all(np.isfinite(win_s)) else float("nan"),
        "neg_threshold_strict":  float(win_s[0])
                                  if all(np.isfinite(win_s)) else float("nan"),
        "pos_threshold_strict":  float(win_s[1])
                                  if all(np.isfinite(win_s)) else float("nan"),
    }


def dominant_failure(aggrs: list[dict], offsets: np.ndarray, *, side: str) -> tuple[str, float]:
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
# Plots
# --------------------------------------------------------------------------
def _line_compare(off_a, val_a, off_b, val_b, *, ylabel, title,
                    threshold, threshold_label, out_path,
                    label_a="G_4867 (default / strict-best)",
                    label_b="G_4299 (margin-best)"):
    fig, ax = plt.subplots(figsize=(10.5, 6.0))
    ax.plot(off_a, val_a, "-o", color="#1f77b4", lw=1.7, markersize=6, label=label_a)
    ax.plot(off_b, val_b, "-s", color="#d62728", lw=1.7, markersize=6, label=label_b)
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
        ("robust_valid_prob", "robust_valid",       "#2ca02c"),
        ("margin_risk_prob",   "margin_risk",        "#ffbf00"),
        ("p_under_exposed",    "under_exposed",      "#1f77b4"),
        ("p_merged",           "merged",             "#d62728"),
        ("p_roughness_degraded","roughness",          "#9467bd"),
        ("p_numerical_invalid","numerical",          "#8c564b"),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(15.5, 5.5), sharey=True)
    for ax, off, aggr, title in [
        (axes[0], off_a, aggr_a, "G_4867 (default / strict-best)"),
        (axes[1], off_b, aggr_b, "G_4299 (margin-best)"),
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
    fig.suptitle("Stage 06N -- failure-mode breakdown vs time offset", fontsize=13)
    fig.tight_layout(rect=[0, 0.04, 1, 0.96])
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _budget_window_compare(off_a, sp_a, rb_a, df_a, win_a,
                              off_b, sp_b, rb_b, df_b, win_b,
                              out_path):
    fig, axes = plt.subplots(1, 2, figsize=(15.5, 6.0), sharey=True)
    for ax, off, sp, rb, df, win, title in [
        (axes[0], off_a, sp_a, rb_a, df_a, win_a, "G_4867 (default / strict-best)"),
        (axes[1], off_b, sp_b, rb_b, df_b, win_b, "G_4299 (margin-best)"),
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
    fig.suptitle("Stage 06N -- time budget window comparison", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# --------------------------------------------------------------------------
# Driver
# --------------------------------------------------------------------------
def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--mc_06m_csv", type=str,
                   default=str(V3_DIR / "outputs" / "yield_optimization"
                               / "stage06M_time_deep_mc.csv"))
    p.add_argument("--mc_06n_csv", type=str,
                   default=str(V3_DIR / "outputs" / "yield_optimization"
                               / "stage06N_G4299_time_deep_mc.csv"))
    p.add_argument("--manifest_yaml", type=str,
                   default=str(V3_DIR / "outputs" / "yield_optimization"
                               / "stage06I_mode_a_final_recipes.yaml"))
    p.add_argument("--strict_cfg_yaml", type=str,
                   default=str(V3_DIR / "outputs" / "yield_optimization"
                               / "stage06G_strict_score_config.yaml"))
    args = p.parse_args()

    strict_yaml = yaml.safe_load(Path(args.strict_cfg_yaml).read_text())
    cd_tol  = float(strict_yaml["thresholds"]["cd_tol_nm"])
    ler_cap = float(strict_yaml["thresholds"]["ler_cap_nm"])

    manifest = yaml.safe_load(Path(args.manifest_yaml).read_text())
    base_g4867 = next(r for r in manifest["representatives"]
                        if r["recipe_id"] == "G_4867")
    base_g4299 = next(r for r in manifest["representatives"]
                        if r["recipe_id"] == "G_4299")
    nominal_t_4867 = float(base_g4867["parameters"]["time_s"])
    nominal_t_4299 = float(base_g4299["parameters"]["time_s"])
    print(f"  G_4867 nominal time_s = {nominal_t_4867:.3f}")
    print(f"  G_4299 nominal time_s = {nominal_t_4299:.3f}")

    coerce_keys = ["CD_final_nm", "CD_locked_nm", "LER_CD_locked_nm",
                    "area_frac", "P_line_margin",
                    "time_offset_s", "sigma_time_s", "variation_idx"]
    rows_a = read_labels_csv(args.mc_06m_csv); _coerce(rows_a, coerce_keys)
    rows_b = read_labels_csv(args.mc_06n_csv); _coerce(rows_b, coerce_keys)

    aggr_a = aggregate_per_offset(rows_a, strict_yaml, nominal_t_4867)
    aggr_b = aggregate_per_offset(rows_b, strict_yaml, nominal_t_4299)

    off_a = np.array([a["time_offset_s"] for a in aggr_a])
    off_b = np.array([a["time_offset_s"] for a in aggr_b])
    sp_a = np.array([a["strict_pass_prob"]   for a in aggr_a])
    sp_b = np.array([a["strict_pass_prob"]   for a in aggr_b])
    rb_a = np.array([a["robust_valid_prob"]   for a in aggr_a])
    rb_b = np.array([a["robust_valid_prob"]   for a in aggr_b])
    df_a = np.array([a["defect_prob"]         for a in aggr_a])
    df_b = np.array([a["defect_prob"]         for a in aggr_b])
    cd_a = np.array([a["mean_cd_error"]       for a in aggr_a])
    cd_b = np.array([a["mean_cd_error"]       for a in aggr_b])
    ler_a = np.array([a["mean_ler_locked"]    for a in aggr_a])
    ler_b = np.array([a["mean_ler_locked"]    for a in aggr_b])
    mg_a = np.array([a["mean_p_line_margin"] for a in aggr_a])
    mg_b = np.array([a["mean_p_line_margin"] for a in aggr_b])

    # ----- Time budget windows -----
    win_a = time_budget(off_a, sp_a, rb_a, df_a)
    win_b = time_budget(off_b, sp_b, rb_b, df_b)

    # ----- Dominant failure modes -----
    neg_a = dominant_failure(aggr_a, off_a, side="neg")
    pos_a = dominant_failure(aggr_a, off_a, side="pos")
    neg_b = dominant_failure(aggr_b, off_b, side="neg")
    pos_b = dominant_failure(aggr_b, off_b, side="pos")

    # ----- Comparison CSV (per-offset side-by-side) -----
    cmp_rows = []
    common = sorted(set(off_a.tolist()) & set(off_b.tolist()))
    for off in common:
        ai = next(a for a in aggr_a if a["time_offset_s"] == off)
        bi = next(a for a in aggr_b if a["time_offset_s"] == off)
        cmp_rows.append({
            "time_offset_s":             off,
            "G4867_strict_pass_prob":    ai["strict_pass_prob"],
            "G4299_strict_pass_prob":    bi["strict_pass_prob"],
            "G4867_robust_valid_prob":   ai["robust_valid_prob"],
            "G4299_robust_valid_prob":   bi["robust_valid_prob"],
            "G4867_defect_prob":          ai["defect_prob"],
            "G4299_defect_prob":          bi["defect_prob"],
            "G4867_mean_cd_error":        ai["mean_cd_error"],
            "G4299_mean_cd_error":        bi["mean_cd_error"],
            "G4867_mean_ler":             ai["mean_ler_locked"],
            "G4299_mean_ler":             bi["mean_ler_locked"],
            "G4867_mean_margin":          ai["mean_p_line_margin"],
            "G4299_mean_margin":          bi["mean_p_line_margin"],
            "G4867_mean_strict_score":    ai["mean_strict_score"],
            "G4299_mean_strict_score":    bi["mean_strict_score"],
        })

    yopt_dir = V3_DIR / "outputs" / "yield_optimization"
    fig_dir = V3_DIR / "outputs" / "figures" / "06_yield_optimization"
    logs_dir = V3_DIR / "outputs" / "logs"
    fig_dir.mkdir(parents=True, exist_ok=True)
    yopt_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    cmp_cols = list(cmp_rows[0].keys()) if cmp_rows else []
    with (yopt_dir / "stage06N_G4867_vs_G4299_time_comparison.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cmp_cols, extrasaction="ignore")
        w.writeheader()
        for r in cmp_rows:
            w.writerow(r)

    # ----- Figures -----
    _line_compare(off_a, sp_a, off_b, sp_b,
                    ylabel="strict_pass_prob (FD MC)",
                    title="Stage 06N -- strict_pass_prob vs time offset",
                    threshold=0.5, threshold_label="strict_pass >= 0.5",
                    out_path=fig_dir / "stage06N_strict_pass_prob_G4867_vs_G4299.png")
    _line_compare(off_a, rb_a, off_b, rb_b,
                    ylabel="robust_valid_prob (FD MC)",
                    title="Stage 06N -- robust_valid_prob vs time offset",
                    threshold=0.8, threshold_label="robust_valid >= 0.8",
                    out_path=fig_dir / "stage06N_robust_valid_prob_G4867_vs_G4299.png")
    _line_compare(off_a, cd_a, off_b, cd_b,
                    ylabel="mean CD_error (nm)",
                    title="Stage 06N -- mean CD_error vs time offset",
                    threshold=0.5, threshold_label="strict CD_tol = 0.5 nm",
                    out_path=fig_dir / "stage06N_cd_error_vs_time_G4867_vs_G4299.png")
    _line_compare(off_a, ler_a, off_b, ler_b,
                    ylabel="mean LER_CD_locked (nm)",
                    title="Stage 06N -- mean LER vs time offset",
                    threshold=3.0, threshold_label="strict LER_cap = 3.0 nm",
                    out_path=fig_dir / "stage06N_ler_vs_time_G4867_vs_G4299.png")
    _line_compare(off_a, mg_a, off_b, mg_b,
                    ylabel="mean P_line_margin",
                    title="Stage 06N -- mean P_line_margin vs time offset",
                    threshold=None, threshold_label="",
                    out_path=fig_dir / "stage06N_margin_vs_time_G4867_vs_G4299.png")
    _failure_breakdown_compare(off_a, aggr_a, off_b, aggr_b,
                                fig_dir / "stage06N_failure_breakdown_comparison.png")
    _budget_window_compare(off_a, sp_a, rb_a, df_a, win_a,
                              off_b, sp_b, rb_b, df_b, win_b,
                              fig_dir / "stage06N_time_budget_window_comparison.png")

    # ----- Decision logic -----
    width_a = win_a["all_three_width"]
    width_b = win_b["all_three_width"]
    sp_strict_a = win_a["strict_pass_width"]
    sp_strict_b = win_b["strict_pass_width"]

    def _label_recipe(rid, win, baseline_strict_pass, nominal_cd_err,
                       cd_err_other, sp_w_other, sp_at_zero_other):
        """Relative role assignment -- compares the two recipes side-by-side
        rather than testing absolute thresholds."""
        sp_w = float(win.get("strict_pass_width", float("nan")))
        # No defined strict_pass window means baseline fails the 0.5 cut.
        if not np.isfinite(sp_w):
            return "fragile_strict_optimum"
        # Broad and high baseline -> stable.
        if sp_w >= 5.0 and baseline_strict_pass >= 0.7:
            return "stable_process_recipe"
        # Better strict_pass at zero AND not meaningfully worse on CD_err than
        # the other -> CD-accurate default.
        if (baseline_strict_pass >= sp_at_zero_other - 0.02
                and nominal_cd_err <= cd_err_other + 0.05):
            return "default_cd_accurate"
        # Wider window but worse on CD/strict -> margin alternative.
        if sp_w >= sp_w_other + 0.5 and sp_w >= 3.0:
            return "robust_margin_alternative"
        return "fragile_strict_optimum"

    zero_a_idx = int(np.argmin(np.abs(off_a)))
    zero_b_idx = int(np.argmin(np.abs(off_b)))
    sp_at_zero_a = float(sp_a[zero_a_idx])
    sp_at_zero_b = float(sp_b[zero_b_idx])
    cd_at_zero_a = float(cd_a[zero_a_idx])
    cd_at_zero_b = float(cd_b[zero_b_idx])

    label_a = _label_recipe("G_4867", win_a, sp_at_zero_a, cd_at_zero_a,
                              cd_at_zero_b, sp_strict_b, sp_at_zero_b)
    label_b = _label_recipe("G_4299", win_b, sp_at_zero_b, cd_at_zero_b,
                              cd_at_zero_a, sp_strict_a, sp_at_zero_a)

    # Decision summary: does G_4299 win on time-window width?
    g4299_wider = (np.isfinite(width_b) and (np.isfinite(width_a) and width_b > width_a + 0.5
                                                  or not np.isfinite(width_a) and width_b > 0))
    g4299_better_strict_pass = sp_at_zero_b > sp_at_zero_a + 0.05

    if g4299_wider and g4299_better_strict_pass:
        outcome = "G_4299_wins_promote_to_robust_default"
        recommendation = ("Promote G_4299 as the robust process default; keep G_4867 "
                            "as the CD-accurate strict-best alternative.")
    elif g4299_wider:
        outcome = "G_4299_has_wider_time_window"
        recommendation = ("Keep G_4867 as Mode A strict / CD-accurate default; "
                            "promote G_4299 as the margin/robustness alternative for "
                            "deployments where time control is loose.")
    else:
        outcome = "G_4867_remains_default"
        recommendation = ("G_4299 does not improve the time window. Keep G_4867 as "
                            "Mode A default; mark G_4299 as margin-interesting but "
                            "not more time-robust.")

    # ----- Update YAML (representative_decision_update) -----
    decision = {
        "stage": "06N",
        "policy": {
            "v2_OP_frozen":          True,
            "published_data_loaded": False,
            "external_calibration":  "none",
        },
        "comparison_basis": {
            "protocol":      "11 deterministic offsets x 100 FD + sigma_t = 2 s gaussian",
            "n_fd_runs_g4867": int(sum(1 for r in rows_a if str(r.get("scenario", "")) == "det_offset")),
            "n_fd_runs_g4299": int(sum(1 for r in rows_b if str(r.get("scenario", "")) == "det_offset")),
            "strict_thresholds": {"cd_tol_nm": cd_tol, "ler_cap_nm": ler_cap},
        },
        "G_4867": {
            "role_06i":              "fd_stability_best (default / strict-best)",
            "role_06n_assigned":     label_a,
            "nominal_time_s":        nominal_t_4867,
            "strict_pass_at_zero":   sp_at_zero_a,
            "mean_cd_error_at_zero": cd_at_zero_a,
            "time_budget":            win_a,
            "primary_failure_neg":   {"kind": neg_a[0], "avg_prob": neg_a[1]},
            "primary_failure_pos":   {"kind": pos_a[0], "avg_prob": pos_a[1]},
        },
        "G_4299": {
            "role_06i":              "margin_best",
            "role_06n_assigned":     label_b,
            "nominal_time_s":        nominal_t_4299,
            "strict_pass_at_zero":   sp_at_zero_b,
            "mean_cd_error_at_zero": cd_at_zero_b,
            "time_budget":            win_b,
            "primary_failure_neg":   {"kind": neg_b[0], "avg_prob": neg_b[1]},
            "primary_failure_pos":   {"kind": pos_b[0], "avg_prob": pos_b[1]},
        },
        "outcome":         outcome,
        "recommendation":  recommendation,
        "primary_recipe_after_06n": "G_4867" if outcome == "G_4867_remains_default" else "G_4299",
    }
    (yopt_dir / "stage06N_representative_decision_update.yaml").write_text(
        yaml.safe_dump(decision, sort_keys=False, default_flow_style=False))

    # ----- Acceptance JSON -----
    acceptance = {
        "g4299_received_same_protocol":          bool(decision["comparison_basis"]["n_fd_runs_g4299"]
                                                          == decision["comparison_basis"]["n_fd_runs_g4867"]),
        "g4299_time_budget_estimated":           bool(np.isfinite(width_b)
                                                          or np.isfinite(sp_strict_b)),
        "comparison_figures_generated":          7,
        "primary_failure_mode_neg_g4867":        neg_a[0],
        "primary_failure_mode_pos_g4867":        pos_a[0],
        "primary_failure_mode_neg_g4299":        neg_b[0],
        "primary_failure_mode_pos_g4299":        pos_b[0],
        "representative_role_assignment_updated": True,
        "policy_v2_OP_frozen":                    True,
        "policy_published_data_loaded":           False,
        "policy_external_calibration":            "none",
    }
    payload = {
        "stage": "06N",
        "policy": decision["policy"],
        "decision": decision,
        "per_offset_aggregates": {
            "G_4867": aggr_a,
            "G_4299": aggr_b,
        },
        "acceptance": acceptance,
    }
    (logs_dir / "stage06N_summary.json").write_text(
        json.dumps(payload, indent=2, default=float))

    # ----- Console summary -----
    print(f"\nStage 06N -- comparison summary")
    print(f"  G_4867 strict_pass at offset 0:    {sp_at_zero_a:.3f}")
    print(f"  G_4299 strict_pass at offset 0:    {sp_at_zero_b:.3f}")
    print(f"  G_4867 mean_cd_error at offset 0:  {cd_at_zero_a:.3f} nm")
    print(f"  G_4299 mean_cd_error at offset 0:  {cd_at_zero_b:.3f} nm")
    print(f"  time budget windows:")
    print(f"    G_4867 strict_pass >= 0.5:  [{win_a['neg_threshold_strict']:+.1f}, "
          f"{win_a['pos_threshold_strict']:+.1f}] s  (width {sp_strict_a:.1f} s)")
    print(f"    G_4299 strict_pass >= 0.5:  [{win_b['neg_threshold_strict']:+.1f}, "
          f"{win_b['pos_threshold_strict']:+.1f}] s  (width {sp_strict_b:.1f} s)")
    print(f"    G_4867 all-three:           {win_a['all_three']}  width {width_a:.1f} s")
    print(f"    G_4299 all-three:           {win_b['all_three']}  width {width_b:.1f} s")
    print(f"  primary failure mode:")
    print(f"    G_4867 neg = {neg_a[0]}  pos = {pos_a[0]}")
    print(f"    G_4299 neg = {neg_b[0]}  pos = {pos_b[0]}")
    print(f"  role assignment:")
    print(f"    G_4867 -> {label_a}")
    print(f"    G_4299 -> {label_b}")
    print(f"  outcome: {outcome}")
    print(f"  recommendation: {recommendation}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

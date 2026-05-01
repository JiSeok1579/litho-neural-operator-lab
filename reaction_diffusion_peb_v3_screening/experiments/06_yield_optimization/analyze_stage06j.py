"""Stage 06J -- analysis figures for the Mode B open-design exploration.

Reads:
    outputs/yield_optimization/stage06J_mode_b_recipe_summary.csv
    outputs/yield_optimization/stage06J_mode_b_top_recipes.csv
    outputs/yield_optimization/stage06J_mode_b_vs_mode_a_comparison.csv
    outputs/yield_optimization/stage06I_mode_a_final_recipes.yaml
    outputs/labels/stage06J_mode_b_fd_sanity.csv
    outputs/labels/stage06J_mode_b_fd_mc_optional.csv (optional)
    outputs/logs/stage06J_summary.json

Writes:
    outputs/figures/06_yield_optimization/
        stage06J_mode_b_pareto_cd_error_vs_ler.png
        stage06J_pitch_line_cd_top_candidates.png
        stage06J_mode_b_vs_mode_a_strict_score.png
        stage06J_mode_b_defect_breakdown.png
        stage06J_mode_b_fd_sanity_scatter.png
        stage06J_pitch_distribution_top100.png
"""
from __future__ import annotations

import argparse
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

from reaction_diffusion_peb_v3_screening.src.metrics_io import (
    FEATURE_KEYS, read_labels_csv,
)


V3_DIR = ROOT / "reaction_diffusion_peb_v3_screening"
CD_TARGET_NM = 15.0


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


# --------------------------------------------------------------------------
# Plots
# --------------------------------------------------------------------------
def plot_pareto_cd_ler(rows_all: list[dict], rows_top: list[dict],
                         baseline: dict, modeA_eval: list[dict],
                         out_path: Path) -> None:
    cd_all = np.array([abs(_safe_float(r.get("mean_cd_fixed")) - CD_TARGET_NM)
                         for r in rows_all])
    ler_all = np.array([_safe_float(r.get("mean_ler_locked")) for r in rows_all])
    score_all = np.array([_safe_float(r.get("strict_score")) for r in rows_all])

    cd_top = np.array([abs(_safe_float(r.get("mean_cd_fixed")) - CD_TARGET_NM)
                         for r in rows_top])
    ler_top = np.array([_safe_float(r.get("mean_ler_locked")) for r in rows_top])

    fig, ax = plt.subplots(figsize=(10.0, 7.0))
    sc = ax.scatter(cd_all, ler_all, s=8, c=score_all, cmap="viridis",
                    alpha=0.45, edgecolor="none")
    cb = plt.colorbar(sc, ax=ax)
    cb.set_label("strict_score (06J surrogate)")
    ax.scatter(cd_top, ler_top, s=22, c="#d62728", alpha=0.9, edgecolor="white",
                lw=0.5, label=f"strict top-{len(rows_top)}")

    if baseline:
        ax.scatter([abs(_safe_float(baseline.get("mean_cd_error")))],
                    [_safe_float(baseline.get("mean_ler_locked"))],
                    s=240, marker="*", color="#7a0a0a",
                    edgecolor="white", lw=1.0, label="v2 frozen OP")
    if modeA_eval:
        for r in modeA_eval:
            ax.scatter([_safe_float(r.get("mean_cd_error"))],
                        [_safe_float(r.get("mean_ler_locked"))],
                        s=70, marker="X", color="#1f77b4",
                        edgecolor="white", lw=0.6, alpha=0.95)
        # Single legend entry for the cluster.
        ax.scatter([], [], marker="X", color="#1f77b4", edgecolor="white",
                    lw=0.6, s=80, label="Mode A representatives (06I)")

    ax.axvline(0.5, color="#1f1f1f", ls="--", lw=1.0, label="strict CD_tol = 0.5 nm")
    ax.axhline(3.0, color="#1f1f1f", ls=":",  lw=1.0, label="strict LER_cap = 3.0 nm")
    ax.set_xlabel("mean CD_error (nm)")
    ax.set_ylabel("mean LER_CD_locked (nm)")
    ax.set_title("Stage 06J -- Mode B CD_error vs LER  (color = strict_score)")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="upper right", fontsize=9, framealpha=0.95)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_pitch_ratio_top(rows_top: list[dict], out_path: Path) -> None:
    pitches = np.array([_safe_float(r.get("pitch_nm")) for r in rows_top])
    ratios = np.array([_safe_float(r.get("line_cd_ratio")) for r in rows_top])
    score = np.array([_safe_float(r.get("strict_score")) for r in rows_top])

    fig, ax = plt.subplots(figsize=(9.5, 6.5))
    sc = ax.scatter(pitches, ratios, s=46, c=score, cmap="viridis",
                    edgecolor="white", lw=0.5, alpha=0.85)
    cb = plt.colorbar(sc, ax=ax); cb.set_label("strict_score (06J)")
    ax.scatter([24], [0.52], s=300, marker="*", color="#7a0a0a",
                edgecolor="white", lw=1.0, label="Mode A geometry (24, 0.52)")
    ax.set_xlabel("pitch_nm"); ax.set_ylabel("line_cd_ratio")
    ax.set_title(f"Stage 06J -- Mode B top-{len(rows_top)} geometry "
                  "(pitch x line_cd_ratio)")
    ax.set_xticks([18, 20, 24, 28, 32])
    ax.set_yticks([0.45, 0.52, 0.60])
    ax.grid(True, alpha=0.30)
    ax.legend(loc="lower right", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_pitch_distribution_top100(rows_top: list[dict], out_path: Path) -> None:
    pitches = [_safe_float(r.get("pitch_nm")) for r in rows_top]
    ratios = [_safe_float(r.get("line_cd_ratio")) for r in rows_top]
    pitch_options = [18.0, 20.0, 24.0, 28.0, 32.0]
    ratio_options = [0.45, 0.52, 0.60]

    fig, axes = plt.subplots(1, 2, figsize=(13.0, 5.5))

    counts_p = [pitches.count(p) for p in pitch_options]
    bar_colors = ["#2ca02c" if p == 24.0 else "#1f77b4" for p in pitch_options]
    axes[0].bar(range(len(pitch_options)), counts_p, color=bar_colors, alpha=0.85,
                  edgecolor="black")
    axes[0].set_xticks(range(len(pitch_options)))
    axes[0].set_xticklabels([f"{int(p)}" for p in pitch_options])
    axes[0].set_xlabel("pitch_nm")
    axes[0].set_ylabel(f"count out of top-{len(rows_top)}")
    axes[0].set_title("Pitch distribution (Mode A pitch = 24 highlighted)")
    for i, c in enumerate(counts_p):
        axes[0].text(i, c + 0.5, f"{c}", ha="center", fontsize=9)
    axes[0].grid(True, alpha=0.25, axis="y")

    counts_r = [ratios.count(r) for r in ratio_options]
    bar_colors_r = ["#2ca02c" if r == 0.52 else "#1f77b4" for r in ratio_options]
    axes[1].bar(range(len(ratio_options)), counts_r, color=bar_colors_r,
                  alpha=0.85, edgecolor="black")
    axes[1].set_xticks(range(len(ratio_options)))
    axes[1].set_xticklabels([f"{r:.2f}" for r in ratio_options])
    axes[1].set_xlabel("line_cd_ratio")
    axes[1].set_ylabel(f"count out of top-{len(rows_top)}")
    axes[1].set_title("line_cd_ratio distribution (Mode A ratio = 0.52 highlighted)")
    for i, c in enumerate(counts_r):
        axes[1].text(i, c + 0.5, f"{c}", ha="center", fontsize=9)
    axes[1].grid(True, alpha=0.25, axis="y")

    fig.suptitle("Stage 06J -- geometry clustering of Mode B top-100 by strict_score",
                  fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_strict_score_compare(cmp_rows: list[dict], out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(11.5, 5.5))
    labels = [r["stage"] for r in cmp_rows]
    scores = [_safe_float(r.get("strict_score")) for r in cmp_rows]
    colors = []
    for s in labels:
        if s == "v2_frozen_op":
            colors.append("#7a0a0a")
        elif s.startswith("mode_a"):
            colors.append("#1f77b4")
        else:
            colors.append("#d62728")
    ax.barh(np.arange(len(labels)), scores, color=colors, alpha=0.85,
              edgecolor="black")
    ax.set_yticks(np.arange(len(labels)))
    ax.set_yticklabels(labels, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("strict_score (06H surrogate, 200 MC)")
    ax.set_title("Stage 06J -- Mode B vs Mode A strict_score")
    ax.grid(True, alpha=0.25, axis="x")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_defect_breakdown(top_rows: list[dict], out_path: Path) -> None:
    rows = sorted(top_rows[:10], key=lambda r: -_safe_float(r.get("strict_score")))
    classes = [
        ("p_robust_valid",       "robust_valid",      "#2ca02c"),
        ("p_margin_risk",        "margin_risk",       "#ffbf00"),
        ("p_under_exposed",      "under_exposed",     "#1f77b4"),
        ("p_merged",             "merged",            "#d62728"),
        ("p_roughness_degraded", "roughness",         "#9467bd"),
        ("p_numerical_invalid",  "numerical",         "#8c564b"),
    ]
    M = np.array([[_safe_float(r.get(k)) for k, _, _ in classes] for r in rows])
    fig, ax = plt.subplots(figsize=(11.5, 5.5))
    bottoms = np.zeros(len(rows))
    for j, (_, label, color) in enumerate(classes):
        ax.bar(np.arange(len(rows)), M[:, j], bottom=bottoms,
                color=color, alpha=0.85, label=label,
                edgecolor="white", lw=0.5)
        bottoms = bottoms + M[:, j]
    for i, r in enumerate(rows):
        ax.text(i, 1.02, f"strict={_safe_float(r['strict_score']):.2f}",
                ha="center", fontsize=8)
    ax.set_xticks(np.arange(len(rows)))
    ax.set_xticklabels([f"{r['recipe_id']}\np={int(_safe_float(r.get('pitch_nm')))} "
                          f"r={_safe_float(r.get('line_cd_ratio')):.2f}"
                          for r in rows], fontsize=8)
    ax.set_xlabel("Mode B top-10 by strict_score")
    ax.set_ylabel("MC class probability (06H surrogate)")
    ax.set_title("Stage 06J -- Mode B top-10 defect breakdown")
    ax.set_ylim(0, 1.10)
    ax.legend(loc="lower right", fontsize=8, framealpha=0.95)
    ax.grid(True, alpha=0.25, axis="y")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_fd_sanity_scatter(fd_rows: list[dict], cd_tol: float, ler_cap: float,
                              out_path: Path) -> None:
    if not fd_rows:
        fig, ax = plt.subplots(figsize=(9.0, 5.5))
        ax.text(0.5, 0.5, "FD sanity check produced no rows.",
                ha="center", va="center", fontsize=12)
        ax.axis("off")
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        return
    cd = np.array([abs(_safe_float(r.get("CD_final_nm")) - CD_TARGET_NM) for r in fd_rows])
    ler = np.array([_safe_float(r.get("LER_CD_locked_nm")) for r in fd_rows])
    sur_strict = np.array([_safe_float(r.get("strict_score_surrogate")) for r in fd_rows])
    labels = np.array([str(r.get("label", "")) for r in fd_rows])
    roles = np.array([str(r.get("role", "")) for r in fd_rows])

    fig, ax = plt.subplots(figsize=(10.0, 6.5))
    label_color = {
        "robust_valid": "#2ca02c", "margin_risk": "#ffbf00",
        "under_exposed": "#1f77b4", "merged": "#d62728",
        "roughness_degraded": "#9467bd", "numerical_invalid": "#8c564b",
    }
    role_marker = {"strict_top": "o", "cd_top": "s", "balanced_top": "^"}
    for role, marker in role_marker.items():
        for lbl, color in label_color.items():
            mask = (roles == role) & (labels == lbl)
            if not np.any(mask):
                continue
            ax.scatter(cd[mask], ler[mask], marker=marker, color=color,
                        edgecolor="black", lw=0.5, s=70, alpha=0.92)

    ax.axvline(cd_tol, color="#1f1f1f", ls="--", lw=1.0,
                label=f"strict CD_tol = {cd_tol} nm")
    ax.axhline(ler_cap, color="#1f1f1f", ls=":", lw=1.0,
                label=f"strict LER_cap = {ler_cap} nm")

    handles = []
    for role, marker in role_marker.items():
        handles.append(plt.Line2D([], [], marker=marker, color="#444",
                                     lw=0, markersize=10, label=role))
    for lbl, c in label_color.items():
        handles.append(plt.scatter([], [], marker="o", c=c,
                                       edgecolor="black", lw=0.5, s=60, label=lbl))
    ax.legend(handles=handles, loc="upper right", fontsize=8, ncol=2,
                framealpha=0.95)
    ax.set_xlabel("FD CD_error_nm = |CD_final - 15|")
    ax.set_ylabel("FD LER_CD_locked_nm")
    ax.set_title("Stage 06J -- Mode B FD sanity (color = label, marker = role)")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# --------------------------------------------------------------------------
# Driver
# --------------------------------------------------------------------------
def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--summary_csv", type=str,
                   default=str(V3_DIR / "outputs" / "yield_optimization"
                               / "stage06J_mode_b_recipe_summary.csv"))
    p.add_argument("--top_csv", type=str,
                   default=str(V3_DIR / "outputs" / "yield_optimization"
                               / "stage06J_mode_b_top_recipes.csv"))
    p.add_argument("--cmp_csv", type=str,
                   default=str(V3_DIR / "outputs" / "yield_optimization"
                               / "stage06J_mode_b_vs_mode_a_comparison.csv"))
    p.add_argument("--fd_sanity_csv", type=str,
                   default=str(V3_DIR / "outputs" / "labels"
                               / "stage06J_mode_b_fd_sanity.csv"))
    p.add_argument("--strict_cfg_yaml", type=str,
                   default=str(V3_DIR / "outputs" / "yield_optimization"
                               / "stage06G_strict_score_config.yaml"))
    p.add_argument("--summary_json", type=str,
                   default=str(V3_DIR / "outputs" / "logs"
                               / "stage06J_summary.json"))
    args = p.parse_args()

    strict_yaml = yaml.safe_load(Path(args.strict_cfg_yaml).read_text())
    cd_tol  = float(strict_yaml["thresholds"]["cd_tol_nm"])
    ler_cap = float(strict_yaml["thresholds"]["ler_cap_nm"])

    rows_all = read_labels_csv(args.summary_csv)
    rows_top = read_labels_csv(args.top_csv)
    cmp_rows = read_labels_csv(args.cmp_csv)
    fd_rows  = read_labels_csv(args.fd_sanity_csv) if Path(args.fd_sanity_csv).exists() else []

    num_keys = ["rank_strict", "strict_score", "yield_score",
                 "strict_pass_prob_proxy",
                 "p_robust_valid", "p_margin_risk", "p_under_exposed",
                 "p_merged", "p_roughness_degraded", "p_numerical_invalid",
                 "mean_cd_fixed", "std_cd_fixed", "mean_cd_error",
                 "mean_cd_locked", "std_cd_locked",
                 "mean_ler_locked", "std_ler_locked",
                 "mean_p_line_margin", "std_p_line_margin",
                 "balanced_score_06j", "geometry_distance_to_modeA",
                 "strict_cd_pen", "strict_ler_pen",
                 "strict_cd_std_pen", "strict_ler_std_pen", "strict_margin_bonus",
                 "cd_error_penalty", "ler_penalty"] + FEATURE_KEYS
    _coerce(rows_all, num_keys); _coerce(rows_top, num_keys)
    _coerce(cmp_rows, ["strict_score", "yield_score", "mean_cd_error",
                         "mean_ler_locked", "p_robust_valid",
                         "mean_p_line_margin", "strict_pass_prob_proxy",
                         "pitch_nm", "line_cd_ratio"])
    _coerce(fd_rows, ["CD_final_nm", "CD_locked_nm", "LER_CD_locked_nm",
                       "area_frac", "P_line_margin",
                       "strict_score_surrogate", "yield_score_surrogate",
                       "strict_pass_prob_proxy_surrogate"])

    # v2 OP comparison row.
    baseline = next((r for r in cmp_rows if r["recipe_id"] == "v2_frozen_op"), None)
    modeA_eval = [r for r in cmp_rows if str(r.get("stage", "")).startswith("mode_a")]

    fig_dir = V3_DIR / "outputs" / "figures" / "06_yield_optimization"
    fig_dir.mkdir(parents=True, exist_ok=True)

    plot_pareto_cd_ler(rows_all, rows_top, baseline, modeA_eval,
                          fig_dir / "stage06J_mode_b_pareto_cd_error_vs_ler.png")
    plot_pitch_ratio_top(rows_top,
                            fig_dir / "stage06J_pitch_line_cd_top_candidates.png")
    plot_pitch_distribution_top100(rows_top,
                                       fig_dir / "stage06J_pitch_distribution_top100.png")
    plot_strict_score_compare(cmp_rows,
                                  fig_dir / "stage06J_mode_b_vs_mode_a_strict_score.png")
    plot_defect_breakdown(rows_top,
                              fig_dir / "stage06J_mode_b_defect_breakdown.png")
    plot_fd_sanity_scatter(fd_rows, cd_tol, ler_cap,
                              fig_dir / "stage06J_mode_b_fd_sanity_scatter.png")

    # ----- Console summary -----
    print(f"  Mode B total candidates: {len(rows_all)}")
    print(f"  Mode B top-100 by strict_score: {len(rows_top)}")
    print(f"  Comparison rows in cmp CSV:     {len(cmp_rows)}")
    print(f"  FD sanity FD rows:              {len(fd_rows)}")
    print(f"  figures -> {fig_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Stage 06D — figures.

Reads
    outputs/yield_optimization/stage06D_recipe_summary.csv
    outputs/yield_optimization/stage06D_top_recipes.csv
    outputs/yield_optimization/stage06D_top20_fd_check.csv
    outputs/labels/06_top_recipes_fixed_design_surrogate.csv
    outputs/logs/stage06D_summary.json
    outputs/logs/06_yield_optimization_summary.json

Writes
    outputs/figures/06_yield_optimization/
        stage06D_vs_06A_yield_distribution.png
        stage06D_parameter_shift.png
        stage06D_pareto_front.png
        stage06D_top_recipe_defect_breakdown.png
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

V3_DIR = ROOT / "reaction_diffusion_peb_v3_screening"


RECIPE_KNOBS = [
    "dose_mJ_cm2", "sigma_nm", "DH_nm2_s", "time_s",
    "Hmax_mol_dm3", "kdep_s_inv", "Q0_mol_dm3", "kq_s_inv",
]


def _safe_float(v) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return float("nan")


_NON_NUMERIC_COLS = {
    "recipe_id", "fd_label", "mode", "_id", "label",
    "source_recipe_id", "phase", "roughness_trigger", "source",
}


def _read_csv(path: Path) -> list[dict]:
    with path.open() as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        for k in list(r.keys()):
            if k in _NON_NUMERIC_COLS:
                continue
            r[k] = _safe_float(r[k])
    return rows


# --------------------------------------------------------------------------
# 1. Yield distribution histogram (06A vs 06D)
# --------------------------------------------------------------------------
def plot_yield_distribution(rows_06a_summary: list[dict],
                             rows_06d_summary: list[dict],
                             baseline_06a: float, baseline_06c: float,
                             best_06a: float, best_06d: float,
                             out_path: Path) -> None:
    score_06a = np.array([r["yield_score"] for r in rows_06a_summary])
    score_06d = np.array([r["yield_score_06c"] for r in rows_06d_summary])

    fig, ax = plt.subplots(figsize=(11.0, 6.0))
    bins = np.linspace(min(score_06a.min(), score_06d.min(), -2.0),
                       max(score_06a.max(), score_06d.max(), 1.05),
                       60)
    ax.hist(score_06a, bins=bins, alpha=0.55, color="#1f77b4",
            label=f"Stage 06A (n={len(score_06a)})")
    ax.hist(score_06d, bins=bins, alpha=0.55, color="#d62728",
            label=f"Stage 06D (n={len(score_06d)})")

    ax.axvline(baseline_06a, color="#1f77b4", lw=1.0, ls=":",
               label=f"v2 frozen OP — 06A surrogate ({baseline_06a:.3f})")
    ax.axvline(baseline_06c, color="#d62728", lw=1.0, ls=":",
               label=f"v2 frozen OP — 06C surrogate ({baseline_06c:.3f})")
    ax.axvline(best_06a, color="#1f77b4", lw=1.6, ls="-",
               label=f"06A best ({best_06a:.3f})")
    ax.axvline(best_06d, color="#d62728", lw=1.6, ls="-",
               label=f"06D best ({best_06d:.3f})")

    ax.set_xlabel("yield_score")
    ax.set_ylabel("count")
    ax.set_title("Stage 06A vs 06D yield_score distributions  "
                 "(5,000 fixed-design candidates each, 200 MC variations)")
    ax.legend(loc="upper left", fontsize=9, framealpha=0.95)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# --------------------------------------------------------------------------
# 2. Per-knob distribution shift (06A top-100 vs 06D top-100)
# --------------------------------------------------------------------------
def plot_parameter_shift(rows_06a_top: list[dict],
                          rows_06d_top: list[dict],
                          out_path: Path) -> None:
    n = len(RECIPE_KNOBS)
    cols = 4
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(15.5, 4.0 * rows),
                             squeeze=False)
    for idx, knob in enumerate(RECIPE_KNOBS):
        ax = axes[idx // cols, idx % cols]
        v_a = np.array([_safe_float(r.get(knob)) for r in rows_06a_top])
        v_d = np.array([_safe_float(r.get(knob)) for r in rows_06d_top])
        v_a = v_a[np.isfinite(v_a)]
        v_d = v_d[np.isfinite(v_d)]
        all_v = np.concatenate([v_a, v_d]) if v_a.size + v_d.size > 0 else np.array([0.0])
        bins = np.linspace(all_v.min(), all_v.max(), 22)
        ax.hist(v_a, bins=bins, alpha=0.55, color="#1f77b4",
                label=f"06A top-100 (μ={np.nanmean(v_a):.2f}, σ={np.nanstd(v_a):.2f})")
        ax.hist(v_d, bins=bins, alpha=0.55, color="#d62728",
                label=f"06D top-100 (μ={np.nanmean(v_d):.2f}, σ={np.nanstd(v_d):.2f})")
        ax.set_xlabel(knob)
        ax.set_ylabel("count")
        ax.legend(loc="upper right", fontsize=7, framealpha=0.9)
        ax.grid(True, alpha=0.2)
    # Hide any unused subplots.
    for idx in range(n, rows * cols):
        axes[idx // cols, idx % cols].axis("off")
    fig.suptitle("Stage 06D vs 06A top-100 recipe-knob distribution shift",
                 fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# --------------------------------------------------------------------------
# 3. Pareto front for Stage 06D
# --------------------------------------------------------------------------
def _pareto_mask(x: np.ndarray, y: np.ndarray, *, maximise_x: bool, maximise_y: bool) -> np.ndarray:
    sign_x = 1.0 if maximise_x else -1.0
    sign_y = 1.0 if maximise_y else -1.0
    xs = sign_x * x; ys = sign_y * y
    n = len(xs)
    keep = np.ones(n, dtype=bool)
    order = np.argsort(-xs)
    best_y = -np.inf
    for i in order:
        if not np.isfinite(xs[i]) or not np.isfinite(ys[i]):
            keep[i] = False
            continue
        if ys[i] >= best_y:
            best_y = ys[i]
        else:
            keep[i] = False
    return keep


def plot_pareto_front(rows_06d_summary: list[dict],
                       baseline_06c: float,
                       out_path: Path) -> None:
    score = np.array([r["yield_score_06c"]      for r in rows_06d_summary])
    p_rv  = np.array([r["p_robust_valid_06c"]   for r in rows_06d_summary])
    cd_pen = np.array([r.get("cd_error_penalty_06c", 0.0)
                       for r in rows_06d_summary])
    ler_pen = np.array([r.get("ler_penalty_06c", 0.0)
                        for r in rows_06d_summary])
    pen = cd_pen + ler_pen

    fig, ax = plt.subplots(figsize=(11.0, 7.0))
    sc = ax.scatter(p_rv, score, s=14, c=pen, cmap="viridis_r",
                    alpha=0.75, edgecolor="none")
    cb = plt.colorbar(sc, ax=ax)
    cb.set_label("CD_error + LER penalty (06C scoring)")

    mask = _pareto_mask(p_rv, score, maximise_x=True, maximise_y=True)
    pf_x = p_rv[mask]; pf_y = score[mask]
    order = np.argsort(pf_x)
    ax.plot(pf_x[order], pf_y[order], "o-", color="#d62728", lw=1.6,
            markersize=6, label=f"Pareto front (n={int(mask.sum())})")

    ax.axhline(baseline_06c, color="#1f1f1f", lw=1.0, ls="--",
               label=f"v2 frozen OP — 06C ({baseline_06c:.3f})")

    beats = score > baseline_06c
    if beats.any():
        ax.scatter(p_rv[beats], score[beats], s=18, marker="o",
                   facecolor="none", edgecolor="#d62728", lw=0.7,
                   alpha=0.9, label=f"beats v2 OP ({int(beats.sum())})")

    lo = max(float(np.quantile(score, 0.05)), baseline_06c - 1.5)
    hi = float(np.max(score)) + 0.05
    ax.set_ylim(lo, hi)
    ax.set_xlim(-0.02, 1.02)

    ax.set_xlabel("P(robust_valid) under process variation (06C)")
    ax.set_ylabel("yield_score (06C, clipped to 5th-pct …max)")
    ax.set_title("Stage 06D — Pareto front under refreshed 06C surrogate")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="lower right", fontsize=10)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# --------------------------------------------------------------------------
# 4. Top-10 defect breakdown
# --------------------------------------------------------------------------
DEFECT_CLASSES = [
    ("p_robust_valid", "robust_valid"),
    ("p_margin_risk", "margin_risk"),
    ("p_under_exposed", "under_exposed"),
    ("p_merged", "merged"),
    ("p_roughness_degraded", "roughness"),
    ("p_numerical_invalid", "numerical"),
]


def plot_top10_defect_breakdown(rows_06d_summary_full: list[dict],
                                  out_path: Path) -> None:
    """Use the full per-recipe summary so we can read the per-class
    probability columns from the original Stage 06A SUMMARY_COLUMNS
    layout. The simplified 06D top-recipes CSV only carries
    p_robust_valid, so we display a stacked bar based on what is
    available + the cd/ler penalty."""
    rows = rows_06d_summary_full[:10]
    fig, ax = plt.subplots(figsize=(11.0, 5.5))
    x = np.arange(len(rows))

    p_rv = np.array([r.get("p_robust_valid_06c", 0.0) for r in rows])
    cd_pen = np.array([r.get("cd_error_penalty_06c", 0.0) for r in rows])
    ler_pen = np.array([r.get("ler_penalty_06c", 0.0) for r in rows])
    score = np.array([r.get("yield_score_06c", 0.0) for r in rows])
    other = np.maximum(0.0, 1.0 - p_rv)        # 1 - P_robust = sum of all other class probs (in 06C scoring)

    ax.bar(x, p_rv, color="#2ca02c", alpha=0.85, edgecolor="white", lw=0.5,
           label="P(robust_valid)")
    ax.bar(x, other, bottom=p_rv,
           color="#d62728", alpha=0.55, edgecolor="white", lw=0.5,
           label="1 − P(robust_valid)  (any non-robust class)")

    # Annotate yield_score above each bar.
    for i, r in enumerate(rows):
        ax.text(i, 1.02, f"yield_score = {score[i]:.3f}",
                ha="center", fontsize=8, color="#1f1f1f")
        if cd_pen[i] > 0 or ler_pen[i] > 0:
            ax.text(i, p_rv[i] + 0.5 * other[i],
                    f"CDpen={cd_pen[i]:.2f}\nLERpen={ler_pen[i]:.2f}",
                    ha="center", fontsize=7, color="#1f1f1f")

    ax.set_xticks(x)
    ax.set_xticklabels([f"#{i+1}\n{r['recipe_id']}" for i, r in enumerate(rows)],
                       fontsize=8)
    ax.set_xlabel("Stage 06D top-10 recipe (06C surrogate ranking)")
    ax.set_ylabel("MC class probability mass")
    ax.set_title("Stage 06D top-10 — robust-valid mass + remaining "
                 "non-robust mass + CD/LER penalty")
    ax.set_ylim(0, 1.10)
    ax.legend(loc="upper right", fontsize=10, framealpha=0.95)
    ax.grid(True, alpha=0.2, axis="y")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--summary_06a_csv", type=str,
                   default=str(V3_DIR / "outputs" / "labels"
                               / "06_yield_optimization_summary.csv"))
    p.add_argument("--top_06a_csv", type=str,
                   default=str(V3_DIR / "outputs" / "labels"
                               / "06_top_recipes_fixed_design_surrogate.csv"))
    p.add_argument("--summary_06d_csv", type=str,
                   default=str(V3_DIR / "outputs" / "yield_optimization"
                               / "stage06D_recipe_summary.csv"))
    p.add_argument("--top_06d_csv", type=str,
                   default=str(V3_DIR / "outputs" / "yield_optimization"
                               / "stage06D_top_recipes.csv"))
    p.add_argument("--summary_06d_json", type=str,
                   default=str(V3_DIR / "outputs" / "logs" / "stage06D_summary.json"))
    p.add_argument("--summary_06a_json", type=str,
                   default=str(V3_DIR / "outputs" / "logs"
                               / "06_yield_optimization_summary.json"))
    args = p.parse_args()

    # 06A: filter to fixed_design rows of the merged summary CSV.
    rows_06a_full = _read_csv(Path(args.summary_06a_csv))
    rows_06a_summary = [r for r in rows_06a_full
                         if str(r.get("mode", "")) == "fixed_design"]
    rows_06a_top = _read_csv(Path(args.top_06a_csv))
    rows_06d_summary = _read_csv(Path(args.summary_06d_csv))
    rows_06d_top = _read_csv(Path(args.top_06d_csv))

    summary_06a = json.loads(Path(args.summary_06a_json).read_text())
    summary_06d = json.loads(Path(args.summary_06d_json).read_text())

    baseline_06a = float(summary_06a["v2_frozen_op_baseline"]["yield_score"])
    baseline_06c = float(summary_06d["v2_frozen_op_baseline"]["yield_score_06c"])
    best_06a = float(summary_06a["mode_summaries"]["fixed_design"]["best_yield_score"])
    best_06d = float(summary_06d["best_recipe_06d"]["yield_score_06c"])

    fig_dir = V3_DIR / "outputs" / "figures" / "06_yield_optimization"
    fig_dir.mkdir(parents=True, exist_ok=True)

    plot_yield_distribution(rows_06a_summary, rows_06d_summary,
                             baseline_06a, baseline_06c,
                             best_06a, best_06d,
                             fig_dir / "stage06D_vs_06A_yield_distribution.png")
    plot_parameter_shift(rows_06a_top, rows_06d_top,
                          fig_dir / "stage06D_parameter_shift.png")
    plot_pareto_front(rows_06d_summary, baseline_06c,
                       fig_dir / "stage06D_pareto_front.png")
    plot_top10_defect_breakdown(rows_06d_summary,
                                 fig_dir / "stage06D_top_recipe_defect_breakdown.png")

    print(f"figures → {fig_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

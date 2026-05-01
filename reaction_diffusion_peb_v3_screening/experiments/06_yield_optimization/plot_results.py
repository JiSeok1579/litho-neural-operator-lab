"""Stage 06A — figure generation for the yield-optimisation study.

Reads:
    outputs/labels/06_yield_optimization_summary.csv
    outputs/logs/06_yield_optimization_summary.json

Writes:
    outputs/figures/06_yield_optimization/pareto_front_fixed_design.png
    outputs/figures/06_yield_optimization/pareto_front_open_design.png
    outputs/figures/06_yield_optimization/defect_breakdown_heatmaps.png
    outputs/figures/06_yield_optimization/recipe_sensitivity.png
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


# Recipe knobs treated as the optimisation axes for the sensitivity plot.
RECIPE_KNOBS = [
    "dose_mJ_cm2", "sigma_nm", "DH_nm2_s", "time_s",
    "Hmax_mol_dm3", "kdep_s_inv", "Q0_mol_dm3", "kq_s_inv",
]

# Six-class colour palette aligned with Stage 04D figures.
DEFECT_CLASSES = [
    "p_robust_valid", "p_margin_risk",
    "p_under_exposed", "p_merged",
    "p_roughness_degraded", "p_numerical_invalid",
]
DEFECT_LABELS = [
    "robust_valid", "margin_risk",
    "under_exposed", "merged",
    "roughness", "numerical",
]


def _load_summary(path: Path) -> tuple[list[dict], dict]:
    with path.open() as f:
        rows = list(csv.DictReader(f))
    by_mode: dict[str, list[dict]] = {}
    for r in rows:
        for k, v in r.items():
            if k in {"mode", "recipe_id"}:
                continue
            try:
                r[k] = float(v)
            except (TypeError, ValueError):
                r[k] = float("nan")
        by_mode.setdefault(r["mode"], []).append(r)
    return rows, by_mode


def _baseline_score(by_mode: dict) -> float | None:
    rows = by_mode.get("v2_frozen_op_baseline", [])
    if not rows:
        return None
    return float(rows[0]["yield_score"])


# --------------------------------------------------------------------------
# Pareto-front helper
# --------------------------------------------------------------------------
def _pareto_mask(x: np.ndarray, y: np.ndarray, *, maximize_x: bool, maximize_y: bool) -> np.ndarray:
    """Return a boolean mask of non-dominated points."""
    sign_x = 1.0 if maximize_x else -1.0
    sign_y = 1.0 if maximize_y else -1.0
    xs = sign_x * x; ys = sign_y * y
    n = len(xs)
    keep = np.ones(n, dtype=bool)
    # Sort by xs descending; sweep keeping the running max of ys.
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


def plot_pareto(rows: list[dict], baseline_score: float | None,
                out_path: Path, mode_label: str) -> None:
    score   = np.array([r["yield_score"]      for r in rows])
    p_rv    = np.array([r["p_robust_valid"]   for r in rows])
    cd_pen  = np.array([r["cd_error_penalty"] for r in rows])
    ler_pen = np.array([r["ler_penalty"]      for r in rows])
    pen_total = cd_pen + ler_pen

    fig, ax = plt.subplots(figsize=(11.0, 7.0))
    sc = ax.scatter(p_rv, score, s=14, c=pen_total, cmap="viridis_r",
                    alpha=0.75, edgecolor="none")
    cb = plt.colorbar(sc, ax=ax)
    cb.set_label("CD_error + LER penalty")

    # Pareto front: maximise both axes.
    mask = _pareto_mask(p_rv, score, maximize_x=True, maximize_y=True)
    pf_x = p_rv[mask]; pf_y = score[mask]
    order = np.argsort(pf_x)
    ax.plot(pf_x[order], pf_y[order], "o-", color="#d62728", lw=1.6,
            markersize=6, label=f"Pareto front (n={int(mask.sum())})")

    if baseline_score is not None:
        ax.axhline(baseline_score, color="#1f1f1f", lw=1.2, ls="--",
                   alpha=0.85, label=f"v2 frozen OP yield_score = {baseline_score:.3f}")

    # Highlight recipes that beat the baseline.
    if baseline_score is not None:
        beats = score > baseline_score
        if beats.any():
            ax.scatter(p_rv[beats], score[beats], s=18, marker="o",
                       facecolor="none", edgecolor="#d62728", lw=0.7,
                       alpha=0.9, label=f"beats baseline ({int(beats.sum())})")

    # Y-axis clip — focus the view on the region of interest. Lower
    # bound = max(5th percentile, baseline − 1.5) so the eye sees the
    # band where most recipes live and where the baseline comparison is
    # readable. Heavy-penalty recipes are still scattered (just below
    # the visible region) but the score legend still spans them.
    if baseline_score is not None:
        lo = max(float(np.quantile(score, 0.05)), baseline_score - 1.5)
    else:
        lo = float(np.quantile(score, 0.05))
    hi = float(np.max(score)) + 0.05
    ax.set_ylim(lo, hi)
    ax.set_xlim(-0.02, 1.02)

    ax.set_xlabel("P(robust_valid) under process variation")
    ax.set_ylabel("yield_score (clipped to 5th-pct …max)")
    ax.set_title(f"Pareto front — yield_score vs P(robust_valid)  [{mode_label}]")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="lower right", fontsize=10)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# --------------------------------------------------------------------------
# Defect-breakdown heatmaps
# --------------------------------------------------------------------------
def plot_defect_heatmaps(by_mode: dict, top_n: int, out_path: Path) -> None:
    panels = []
    if "fixed_design" in by_mode and by_mode["fixed_design"]:
        panels.append(("fixed_design (Mode A)", by_mode["fixed_design"]))
    if "open_design" in by_mode and by_mode["open_design"]:
        panels.append(("open_design (Mode B)", by_mode["open_design"]))
    if not panels:
        return

    fig, axes = plt.subplots(1, len(panels),
                             figsize=(7.5 * len(panels), 7.0),
                             squeeze=False)
    for ax_idx, (mode, rows) in enumerate(panels):
        ax = axes[0, ax_idx]
        rows_top = sorted(rows, key=lambda r: -r["yield_score"])[:top_n]
        M = np.array([[r[k] for k in DEFECT_CLASSES] for r in rows_top])
        # Reverse rank so rank-1 sits on top.
        M = M[::-1]
        im = ax.imshow(M, aspect="auto", cmap="magma", vmin=0.0, vmax=1.0)
        ax.set_xticks(range(len(DEFECT_LABELS)))
        ax.set_xticklabels(DEFECT_LABELS, rotation=20, ha="right")
        # y-axis: rank ↑ visually = top of plot; show every 10th rank.
        n = M.shape[0]
        ranks_top_first = np.arange(1, n + 1)            # rank 1 at top
        rt = ranks_top_first
        tick_idx = list(range(0, n, max(1, n // 10)))
        # imshow row 0 = top → corresponds to rank 1.
        ax.set_yticks(tick_idx)
        ax.set_yticklabels([f"#{rt[i]}" for i in tick_idx], fontsize=8)
        ax.set_ylabel("yield_score rank")
        ax.set_title(f"top-{top_n} class-prob breakdown — {mode}")
        cb = plt.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
        cb.set_label("mean class probability")
    fig.suptitle("Top-N defect breakdown (per recipe; averaged over process variations)",
                 fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# --------------------------------------------------------------------------
# Recipe sensitivity — Spearman of yield_score vs each recipe knob (Mode A).
# --------------------------------------------------------------------------
def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    finite = np.isfinite(a) & np.isfinite(b)
    if finite.sum() < 3:
        return float("nan")
    a = a[finite]; b = b[finite]
    ra = np.argsort(np.argsort(a)).astype(float)
    rb = np.argsort(np.argsort(b)).astype(float)
    if np.std(ra) < 1e-12 or np.std(rb) < 1e-12:
        return float("nan")
    return float(np.corrcoef(ra, rb)[0, 1])


def plot_recipe_sensitivity(by_mode: dict, out_path: Path) -> None:
    rows = by_mode.get("fixed_design", [])
    if not rows:
        rows = by_mode.get("open_design", [])
    if not rows:
        return
    score = np.array([r["yield_score"] for r in rows])
    spearman = []
    for k in RECIPE_KNOBS:
        v = np.array([r.get(k, np.nan) for r in rows])
        spearman.append(_spearman(score, v))

    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    colors = ["#2ca02c" if s > 0 else "#d62728" for s in spearman]
    ax.bar(RECIPE_KNOBS, spearman, color=colors, alpha=0.85,
           edgecolor="#1f1f1f")
    for i, s in enumerate(spearman):
        ax.text(i, s + (0.02 if s >= 0 else -0.04), f"{s:+.2f}",
                ha="center", fontsize=9,
                color="#1f1f1f")
    ax.axhline(0.0, color="#777", lw=0.8)
    ax.set_ylim(-1.05, 1.05)
    ax.set_ylabel("Spearman ρ(yield_score, knob)")
    ax.set_title("Recipe sensitivity — fixed-design (Mode A) yield_score vs knob")
    ax.tick_params(axis="x", rotation=15)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--summary_csv", type=str,
                   default=str(V3_DIR / "outputs" / "labels"
                               / "06_yield_optimization_summary.csv"))
    p.add_argument("--summary_json", type=str,
                   default=str(V3_DIR / "outputs" / "logs"
                               / "06_yield_optimization_summary.json"))
    p.add_argument("--top_n", type=int, default=100)
    args = p.parse_args()

    rows, by_mode = _load_summary(Path(args.summary_csv))
    baseline = _baseline_score(by_mode)

    fig_dir = V3_DIR / "outputs" / "figures" / "06_yield_optimization"
    fig_dir.mkdir(parents=True, exist_ok=True)

    if "fixed_design" in by_mode:
        plot_pareto(by_mode["fixed_design"], baseline,
                    fig_dir / "pareto_front_fixed_design.png", "fixed_design (Mode A)")
    if "open_design" in by_mode:
        plot_pareto(by_mode["open_design"], baseline,
                    fig_dir / "pareto_front_open_design.png", "open_design (Mode B)")

    plot_defect_heatmaps(by_mode, args.top_n,
                         fig_dir / "defect_breakdown_heatmaps.png")
    plot_recipe_sensitivity(by_mode,
                            fig_dir / "recipe_sensitivity.png")

    print(f"figures → {fig_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

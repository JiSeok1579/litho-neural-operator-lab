"""Stage 06F -- strict Pareto ranking for saturated robust recipes.

Stage 06E showed that the v2 frozen-OP saturates the FD yield_score
at 1.0 under both nominal and 100x MC FD, and that 69/100 of the
06D top-100 nominal recipes also tie that ceiling without false-PASS.
At that point "beats frozen OP by yield_score" stops being a useful
discriminator. This stage replaces yield_score with a multi-objective
Pareto ranking on FD-derived metrics so we can still tell the
saturated, all-robust candidates apart.

Inputs (all from Stage 06E -- no new FD runs):
    outputs/labels/stage06E_fd_top100_nominal.csv
    outputs/labels/stage06E_fd_top10_mc.csv
    outputs/labels/stage06E_fd_baseline_v2_op.csv
    outputs/labels/stage06E_fd_baseline_v2_op_mc.csv
    outputs/yield_optimization/stage06D_top_recipes.csv

Outputs:
    outputs/yield_optimization/stage06F_pareto_nominal.csv
    outputs/yield_optimization/stage06F_pareto_mc.csv
    outputs/yield_optimization/stage06F_representative_recipes.csv
    outputs/yield_optimization/stage06F_threshold_sensitivity.csv
    outputs/logs/stage06F_summary.json
    outputs/figures/06_yield_optimization/
        stage06F_cd_error_vs_ler_pareto.png
        stage06F_mc_cd_std_vs_ler_std.png
        stage06F_pareto_front_colored_by_margin.png
        stage06F_representative_recipe_radar.png
        stage06F_threshold_survival_heatmap.png

Policy
    v2_OP_frozen           = true
    published_data_loaded  = false
    No external calibration. CD tolerance / LER cap thresholds in the
    sensitivity table are explicitly hypothetical and are NOT taken as
    spec truth. Stage 04C / 04D / 06B / 06E artefacts are not mutated.
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

from reaction_diffusion_peb_v3_screening.src.metrics_io import (
    FEATURE_KEYS,
    read_labels_csv,
)


V3_DIR = ROOT / "reaction_diffusion_peb_v3_screening"
CD_TARGET_NM = 15.0


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


# --------------------------------------------------------------------------
# Pareto rank + crowding distance (NSGA-II style, all objectives MIN).
# --------------------------------------------------------------------------
def pareto_rank(F: np.ndarray) -> np.ndarray:
    """For (N, K) matrix `F` (minimize all columns), return per-row
    Pareto rank starting at 1. O(N^2) -- fine for N<=200."""
    n = F.shape[0]
    rank = np.zeros(n, dtype=int)
    remaining = list(range(n))
    cur_rank = 1
    while remaining:
        front = []
        for i in remaining:
            dominated = False
            for j in remaining:
                if j == i:
                    continue
                if np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                    dominated = True
                    break
            if not dominated:
                front.append(i)
        for i in front:
            rank[i] = cur_rank
        remaining = [i for i in remaining if i not in front]
        cur_rank += 1
    return rank


def crowding_distance(F_front: np.ndarray) -> np.ndarray:
    """Crowding distance for points on a single Pareto front. NSGA-II."""
    n, k = F_front.shape
    if n == 0:
        return np.zeros(0)
    if n <= 2:
        return np.full(n, np.inf)
    cd = np.zeros(n)
    for k_idx in range(k):
        col = F_front[:, k_idx]
        order = np.argsort(col)
        cd[order[0]] = np.inf
        cd[order[-1]] = np.inf
        rng = float(col[order[-1]] - col[order[0]])
        if rng <= 1e-12:
            continue
        for i in range(1, n - 1):
            cd[order[i]] += (col[order[i + 1]] - col[order[i - 1]]) / rng
    return cd


def attach_pareto(rows: list[dict], F: np.ndarray, prefix: str = "") -> None:
    rank = pareto_rank(F)
    cd = np.zeros(len(rows))
    for r in np.unique(rank):
        idx = np.where(rank == r)[0]
        cd[idx] = crowding_distance(F[idx])
    for i, row in enumerate(rows):
        row[f"{prefix}pareto_rank"] = int(rank[i])
        row[f"{prefix}crowding_distance"] = float(cd[i])


# --------------------------------------------------------------------------
# Aggregation helpers.
# --------------------------------------------------------------------------
HARD_FAIL_LABELS = {"under_exposed", "merged",
                    "roughness_degraded", "numerical_invalid"}


def aggregate_mc(fd_mc_rows: list[dict]) -> dict[str, dict]:
    """Bucket MC rows by source_recipe_id and compute the aggregates
    required for Pareto ranking + balanced score."""
    buckets: dict[str, list[dict]] = {}
    for r in fd_mc_rows:
        rid = r.get("source_recipe_id", "")
        buckets.setdefault(rid, []).append(r)

    out: dict[str, dict] = {}
    for rid, rs in buckets.items():
        cd = np.array([_safe_float(r.get("CD_final_nm")) for r in rs])
        ler = np.array([_safe_float(r.get("LER_CD_locked_nm")) for r in rs])
        margin = np.array([_safe_float(r.get("P_line_margin")) for r in rs])
        labels = [str(r.get("label", "")) for r in rs]
        n = len(rs)
        n_robust = sum(1 for l in labels if l == "robust_valid")
        n_margin = sum(1 for l in labels if l == "margin_risk")
        n_hard = sum(1 for l in labels if l in HARD_FAIL_LABELS)

        cd_err = np.abs(cd - CD_TARGET_NM)
        out[rid] = {
            "recipe_id":         rid,
            "n_mc":              n,
            "mean_cd_final":     float(np.nanmean(cd)),
            "mean_cd_error":     float(np.nanmean(cd_err)),
            "std_cd_final":      float(np.nanstd(cd)),
            "mean_ler_locked":   float(np.nanmean(ler)),
            "std_ler_locked":    float(np.nanstd(ler)),
            "mean_p_line_margin": float(np.nanmean(margin)),
            "std_p_line_margin": float(np.nanstd(margin)),
            "robust_prob":       float(n_robust / max(n, 1)),
            "margin_risk_prob":  float(n_margin / max(n, 1)),
            "defect_prob":       float(n_hard / max(n, 1)),
            "label_counts":      dict(Counter(labels)),
        }
    return out


# --------------------------------------------------------------------------
# Z-score normalisation (population std).
# --------------------------------------------------------------------------
def _zscore(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    finite = np.isfinite(x)
    if finite.sum() == 0:
        return np.zeros_like(x)
    mu = float(np.nanmean(x[finite]))
    sigma = float(np.nanstd(x[finite]))
    if sigma <= 1e-12:
        return np.zeros_like(x)
    z = np.where(finite, (x - mu) / sigma, 0.0)
    return z


# --------------------------------------------------------------------------
# Spearman without scipy (rank correlation).
# --------------------------------------------------------------------------
def spearman(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64); b = np.asarray(b, dtype=np.float64)
    finite = np.isfinite(a) & np.isfinite(b)
    if finite.sum() < 3:
        return float("nan")
    a = a[finite]; b = b[finite]
    if np.unique(a).size < 2 or np.unique(b).size < 2:
        return float("nan")
    from scipy.stats import spearmanr
    rho, _ = spearmanr(a, b)
    return float(rho) if np.isfinite(rho) else float("nan")


# --------------------------------------------------------------------------
# CSV writer.
# --------------------------------------------------------------------------
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
# Plots
# --------------------------------------------------------------------------
def plot_cd_error_vs_ler_pareto(rows: list[dict], baseline: dict,
                                  out_path: Path) -> None:
    cd = np.array([r["CD_error_nm"] for r in rows])
    ler = np.array([r["LER_CD_locked_nm"] for r in rows])
    rank = np.array([r["nominal_pareto_rank"] for r in rows])

    fig, ax = plt.subplots(figsize=(10.0, 7.0))
    cmap = plt.get_cmap("viridis")
    max_rank = int(rank.max()) if rank.size else 1
    for r in range(1, max_rank + 1):
        m = rank == r
        if not np.any(m):
            continue
        c = cmap(0.05 + 0.9 * (r - 1) / max(max_rank - 1, 1))
        ax.scatter(cd[m], ler[m], s=46 if r == 1 else 22,
                    color=c, alpha=0.85 if r == 1 else 0.55,
                    edgecolor="white", lw=0.5,
                    label=f"rank {r}" if r <= 4 else None)
    front = np.where(rank == 1)[0]
    if front.size:
        order = np.argsort(cd[front])
        ax.plot(cd[front][order], ler[front][order],
                "-", color="#d62728", lw=1.4, alpha=0.7,
                label=f"Pareto rank-1 ({front.size})")

    if baseline:
        ax.axvline(baseline["CD_error_nm"], color="#1f1f1f", lw=1.0, ls="--",
                   label=f"v2 OP CD_error = {baseline['CD_error_nm']:.3f} nm")
        ax.axhline(baseline["LER_CD_locked_nm"], color="#1f1f1f", lw=1.0, ls=":",
                   label=f"v2 OP LER = {baseline['LER_CD_locked_nm']:.3f} nm")

    ax.set_xlabel("nominal CD_error_nm = |CD_final_nm - 15.0|")
    ax.set_ylabel("nominal LER_CD_locked_nm")
    ax.set_title("Stage 06F -- nominal-FD Pareto front "
                 "(CD_error vs LER, color = Pareto rank)")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="upper right", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_pareto_colored_by_margin(rows: list[dict], baseline: dict,
                                    out_path: Path) -> None:
    cd = np.array([r["CD_error_nm"] for r in rows])
    ler = np.array([r["LER_CD_locked_nm"] for r in rows])
    margin = np.array([r["P_line_margin_fd"] for r in rows])
    rank = np.array([r["nominal_pareto_rank"] for r in rows])

    fig, ax = plt.subplots(figsize=(10.0, 7.0))
    sc = ax.scatter(cd, ler, s=30, c=margin, cmap="viridis",
                    alpha=0.85, edgecolor="white", lw=0.4)
    cb = plt.colorbar(sc, ax=ax)
    cb.set_label("P_line_margin (higher = more margin)")

    front = np.where(rank == 1)[0]
    if front.size:
        order = np.argsort(cd[front])
        ax.plot(cd[front][order], ler[front][order],
                "-o", color="#d62728", lw=1.4, alpha=0.85,
                markersize=7, markerfacecolor="none", markeredgewidth=1.4,
                label=f"Pareto rank-1 ({front.size})")

    if baseline:
        ax.scatter([baseline["CD_error_nm"]], [baseline["LER_CD_locked_nm"]],
                    s=180, marker="*", color="#7a0a0a", edgecolor="white", lw=1.0,
                    label=f"v2 frozen OP")

    ax.set_xlabel("nominal CD_error_nm")
    ax.set_ylabel("nominal LER_CD_locked_nm")
    ax.set_title("Stage 06F -- nominal-FD Pareto front colored by P_line_margin")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="upper right", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_mc_std_scatter(mc_rows: list[dict], baseline_mc: dict,
                         out_path: Path) -> None:
    cdstd = np.array([r["std_cd_final"] for r in mc_rows])
    lerstd = np.array([r["std_ler_locked"] for r in mc_rows])
    defect = np.array([r["defect_prob"] for r in mc_rows])
    rank = np.array([r["mc_pareto_rank"] for r in mc_rows])

    fig, ax = plt.subplots(figsize=(10.0, 6.5))
    sc = ax.scatter(cdstd, lerstd, s=120, c=defect, cmap="Reds",
                    edgecolor="black", lw=0.6, vmin=0.0,
                    vmax=max(0.05, float(defect.max()) if defect.size else 0.05))
    cb = plt.colorbar(sc, ax=ax)
    cb.set_label("MC defect_prob")
    for i, r in enumerate(mc_rows):
        rid = r.get("recipe_id", "")
        suffix = f"#{r.get('rank_06c', '?')} {rid}"
        ax.annotate(suffix, (cdstd[i], lerstd[i]),
                    fontsize=7, xytext=(4, 4), textcoords="offset points")
    if baseline_mc:
        ax.scatter([baseline_mc["std_cd_final"]],
                    [baseline_mc["std_ler_locked"]],
                    s=240, marker="*", color="#1f77b4", edgecolor="white",
                    lw=1.0, label=f"v2 OP MC")

    front = np.where(rank == 1)[0]
    if front.size:
        order = np.argsort(cdstd[front])
        ax.plot(cdstd[front][order], lerstd[front][order],
                "-o", color="#d62728", lw=1.4, alpha=0.6,
                markersize=10, markerfacecolor="none", markeredgewidth=1.5,
                label=f"MC Pareto rank-1 ({front.size})")

    ax.set_xlabel("std(CD_final_nm) under MC (nm)")
    ax.set_ylabel("std(LER_CD_locked_nm) under MC (nm)")
    ax.set_title("Stage 06F -- top-10 MC FD: CD vs LER variability "
                 "(color = defect_prob)")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="lower right", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_representative_radar(reps: list[dict], op_norm: dict,
                                axis_keys: list[str],
                                out_path: Path) -> None:
    if not reps:
        return
    n_axes = len(axis_keys)
    angles = np.linspace(0, 2 * np.pi, n_axes, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(9.5, 8.0), subplot_kw={"polar": True})
    colors = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd"]

    for i, rep in enumerate(reps):
        vals = [rep["radar_norm"][k] for k in axis_keys]
        vals += vals[:1]
        ax.plot(angles, vals, "-o", color=colors[i % len(colors)],
                lw=1.6, markersize=6,
                label=f"{rep['kind']}: {rep['recipe_id']}")
        ax.fill(angles, vals, color=colors[i % len(colors)], alpha=0.12)

    if op_norm:
        op_vals = [op_norm.get(k, 0.0) for k in axis_keys]
        op_vals += op_vals[:1]
        ax.plot(angles, op_vals, "-", color="#1f1f1f", lw=1.4, alpha=0.7,
                label="v2 frozen OP")

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(axis_keys, fontsize=8)
    ax.set_ylim(0, 1.05)
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["0.25", "0.50", "0.75", "1.0"], fontsize=8)
    ax.set_title("Stage 06F -- representative recipes radar  "
                 "(higher = better, normalised over top-100)",
                 fontsize=11, pad=24)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.18),
              fontsize=9, ncol=2, framealpha=0.95)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_threshold_heatmap(survival: list[dict], cd_tols: list[float],
                            ler_caps: list[float], n_total: int,
                            out_path: Path) -> None:
    M = np.zeros((len(ler_caps), len(cd_tols)), dtype=int)
    for r in survival:
        i = ler_caps.index(r["ler_cap_nm"])
        j = cd_tols.index(r["cd_tol_nm"])
        M[i, j] = int(r["n_survivors"])

    fig, ax = plt.subplots(figsize=(8.0, 5.5))
    im = ax.imshow(M, cmap="YlGnBu", origin="lower",
                    vmin=0, vmax=n_total, aspect="auto")
    cb = plt.colorbar(im, ax=ax)
    cb.set_label(f"#recipes surviving (out of {n_total})")
    ax.set_xticks(range(len(cd_tols)))
    ax.set_xticklabels([f"+/-{t:.2f}" for t in cd_tols])
    ax.set_yticks(range(len(ler_caps)))
    ax.set_yticklabels([f"{c:.1f}" for c in ler_caps])
    ax.set_xlabel("CD tolerance (nm)")
    ax.set_ylabel("LER cap (nm)")
    ax.set_title("Stage 06F -- hypothetical threshold sensitivity "
                 "(NOT spec truth)")
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            color = "white" if M[i, j] > n_total / 2 else "black"
            ax.text(j, i, str(M[i, j]),
                    ha="center", va="center", color=color, fontsize=10)
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
    p.add_argument("--fd_nominal_csv", type=str,
                   default=str(V3_DIR / "outputs" / "labels"
                               / "stage06E_fd_top100_nominal.csv"))
    p.add_argument("--fd_mc_csv", type=str,
                   default=str(V3_DIR / "outputs" / "labels"
                               / "stage06E_fd_top10_mc.csv"))
    p.add_argument("--fd_baseline_csv", type=str,
                   default=str(V3_DIR / "outputs" / "labels"
                               / "stage06E_fd_baseline_v2_op.csv"))
    p.add_argument("--fd_baseline_mc_csv", type=str,
                   default=str(V3_DIR / "outputs" / "labels"
                               / "stage06E_fd_baseline_v2_op_mc.csv"))
    p.add_argument("--top_06d_csv", type=str,
                   default=str(V3_DIR / "outputs" / "yield_optimization"
                               / "stage06D_top_recipes.csv"))
    args = p.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())

    # ----- Load inputs -----
    fd_nominal = read_labels_csv(args.fd_nominal_csv)
    _coerce_floats(fd_nominal, ["CD_final_nm", "CD_locked_nm",
                                 "LER_CD_locked_nm", "area_frac",
                                 "P_line_margin", "rank_surrogate",
                                 "yield_score_surrogate",
                                 "yield_score_06a_rescore"])

    fd_mc = read_labels_csv(args.fd_mc_csv)
    _coerce_floats(fd_mc, ["CD_final_nm", "CD_locked_nm",
                            "LER_CD_locked_nm", "area_frac",
                            "P_line_margin", "rank_surrogate",
                            "yield_score_surrogate",
                            "yield_score_06a_rescore"])

    fd_baseline_nom = read_labels_csv(args.fd_baseline_csv)
    _coerce_floats(fd_baseline_nom, ["CD_final_nm", "CD_locked_nm",
                                      "LER_CD_locked_nm", "area_frac",
                                      "P_line_margin"])
    fd_baseline_mc = read_labels_csv(args.fd_baseline_mc_csv)
    _coerce_floats(fd_baseline_mc, ["CD_final_nm", "CD_locked_nm",
                                     "LER_CD_locked_nm", "area_frac",
                                     "P_line_margin"])

    top_06d = read_labels_csv(args.top_06d_csv)
    _coerce_floats(top_06d, ["rank_06c", "yield_score_06c",
                              "yield_score_06a"] + FEATURE_KEYS)
    sur_lookup = {r["recipe_id"]: r for r in top_06d}

    # ----- Part A — Pareto on top-100 nominal FD -----
    nom_rows: list[dict] = []
    for r in fd_nominal:
        rid = r.get("source_recipe_id", "")
        sur = sur_lookup.get(rid, {})
        cd_err = abs(_safe_float(r.get("CD_final_nm")) - CD_TARGET_NM)
        nom_rows.append({
            "recipe_id":         rid,
            "rank_surrogate":    int(_safe_float(r.get("rank_surrogate", 0))),
            "yield_score_06c":   _safe_float(sur.get("yield_score_06c")),
            "yield_score_06a":   _safe_float(sur.get("yield_score_06a")),
            "fd_label":          str(r.get("label", "")),
            "CD_final_nm":       _safe_float(r.get("CD_final_nm")),
            "CD_error_nm":       float(cd_err),
            "LER_CD_locked_nm":  _safe_float(r.get("LER_CD_locked_nm")),
            "P_line_margin_fd":  _safe_float(r.get("P_line_margin")),
            "area_frac_fd":      _safe_float(r.get("area_frac")),
            "defect_indicator":  0 if str(r.get("label", "")) == "robust_valid" else 1,
            **{k: _safe_float(sur.get(k)) for k in FEATURE_KEYS},
        })
    F_nom = np.array([
        [r["CD_error_nm"],
         r["LER_CD_locked_nm"],
         -r["P_line_margin_fd"],
         float(r["defect_indicator"])]
        for r in nom_rows
    ], dtype=np.float64)
    attach_pareto(nom_rows, F_nom, prefix="nominal_")

    # ----- Part B — MC aggregates + Pareto on top-10 MC FD -----
    mc_agg = aggregate_mc(fd_mc)
    mc_rows: list[dict] = []
    for rid, payload in mc_agg.items():
        sur = sur_lookup.get(rid, {})
        # Look up the matching nominal row's rank/CD/LER/margin for context.
        nom = next((r for r in nom_rows if r["recipe_id"] == rid), {})
        mc_rows.append({
            **payload,
            "rank_06c":          int(_safe_float(sur.get("rank_06c", 0))),
            "yield_score_06c":   _safe_float(sur.get("yield_score_06c")),
            "yield_score_06a":   _safe_float(sur.get("yield_score_06a")),
            "nominal_CD_error_nm":     nom.get("CD_error_nm", float("nan")),
            "nominal_LER_CD_locked_nm": nom.get("LER_CD_locked_nm", float("nan")),
            "nominal_P_line_margin":   nom.get("P_line_margin_fd", float("nan")),
            **{k: _safe_float(sur.get(k)) for k in FEATURE_KEYS},
        })
    mc_rows.sort(key=lambda r: r["rank_06c"])
    F_mc = np.array([
        [r["mean_cd_error"],
         r["mean_ler_locked"],
         r["std_cd_final"],
         r["std_ler_locked"],
         r["defect_prob"],
         -r["mean_p_line_margin"]]
        for r in mc_rows
    ], dtype=np.float64)
    attach_pareto(mc_rows, F_mc, prefix="mc_")

    # ----- v2 frozen OP reference points -----
    op_nom = {}
    if fd_baseline_nom:
        b = fd_baseline_nom[0]
        op_nom = {
            "CD_final_nm":      _safe_float(b.get("CD_final_nm")),
            "CD_error_nm":      abs(_safe_float(b.get("CD_final_nm")) - CD_TARGET_NM),
            "LER_CD_locked_nm": _safe_float(b.get("LER_CD_locked_nm")),
            "P_line_margin_fd": _safe_float(b.get("P_line_margin")),
            "label":            str(b.get("label", "")),
        }

    op_mc = {}
    if fd_baseline_mc:
        b_agg = aggregate_mc(fd_baseline_mc)
        # Single source_recipe_id == "v2_frozen_op".
        b = next(iter(b_agg.values()), {})
        op_mc = {
            "n_mc":              b.get("n_mc", 0),
            "mean_cd_final":     b.get("mean_cd_final", float("nan")),
            "mean_cd_error":     b.get("mean_cd_error", float("nan")),
            "std_cd_final":      b.get("std_cd_final", float("nan")),
            "mean_ler_locked":   b.get("mean_ler_locked", float("nan")),
            "std_ler_locked":    b.get("std_ler_locked", float("nan")),
            "mean_p_line_margin": b.get("mean_p_line_margin", float("nan")),
            "robust_prob":       b.get("robust_prob", float("nan")),
            "margin_risk_prob":  b.get("margin_risk_prob", float("nan")),
            "defect_prob":       b.get("defect_prob", float("nan")),
            "label_counts":      b.get("label_counts", {}),
        }

    # ----- Part C — ranking comparison (nominal) -----
    cd_err = np.array([r["CD_error_nm"] for r in nom_rows])
    ler = np.array([r["LER_CD_locked_nm"] for r in nom_rows])
    margin = np.array([r["P_line_margin_fd"] for r in nom_rows])
    cd_std_n = np.zeros_like(cd_err)   # zero for nominal (no per-recipe std)
    ler_std_n = np.zeros_like(ler)
    defect_n = np.array([float(r["defect_indicator"]) for r in nom_rows])

    balanced_n = (
        _zscore(cd_err) + _zscore(ler) + _zscore(cd_std_n) + _zscore(ler_std_n)
        + 2.0 * defect_n - _zscore(margin)
    )
    yield_06c = np.array([r["yield_score_06c"] for r in nom_rows])
    pareto_rank_n = np.array([r["nominal_pareto_rank"] for r in nom_rows])

    cd_only_rank = np.argsort(np.argsort(cd_err)) + 1
    ler_only_rank = np.argsort(np.argsort(ler)) + 1
    balanced_rank = np.argsort(np.argsort(balanced_n)) + 1
    yield_rank_06c = np.argsort(np.argsort(-yield_06c)) + 1
    for i, r in enumerate(nom_rows):
        r["balanced_score"]      = float(balanced_n[i])
        r["cd_only_rank"]        = int(cd_only_rank[i])
        r["ler_only_rank"]       = int(ler_only_rank[i])
        r["balanced_rank"]       = int(balanced_rank[i])
        r["yield_score_rank_06c"] = int(yield_rank_06c[i])

    # Spearman of yield rank vs each new rank.
    rho_yield_pareto   = spearman(yield_rank_06c, pareto_rank_n)
    rho_yield_cd       = spearman(yield_rank_06c, cd_only_rank)
    rho_yield_ler      = spearman(yield_rank_06c, ler_only_rank)
    rho_yield_balanced = spearman(yield_rank_06c, balanced_rank)
    rho_pareto_balanced = spearman(pareto_rank_n, balanced_rank)

    # Same for MC (rank by balanced_mc score).
    cd_err_mc  = np.array([r["mean_cd_error"]  for r in mc_rows])
    ler_mc     = np.array([r["mean_ler_locked"] for r in mc_rows])
    cd_std_mc  = np.array([r["std_cd_final"]    for r in mc_rows])
    ler_std_mc = np.array([r["std_ler_locked"]  for r in mc_rows])
    defect_mc  = np.array([r["defect_prob"]     for r in mc_rows])
    margin_mc  = np.array([r["mean_p_line_margin"] for r in mc_rows])

    balanced_m = (
        _zscore(cd_err_mc) + _zscore(ler_mc)
        + _zscore(cd_std_mc) + _zscore(ler_std_mc)
        + 2.0 * defect_mc - _zscore(margin_mc)
    )
    yield_mc_06c = np.array([r["yield_score_06c"] for r in mc_rows])
    yield_rank_06c_mc = np.argsort(np.argsort(-yield_mc_06c)) + 1
    balanced_rank_mc = np.argsort(np.argsort(balanced_m)) + 1
    pareto_rank_m = np.array([r["mc_pareto_rank"] for r in mc_rows])

    for i, r in enumerate(mc_rows):
        r["balanced_score_mc"]      = float(balanced_m[i])
        r["balanced_rank_mc"]       = int(balanced_rank_mc[i])
        r["yield_score_rank_06c_mc"] = int(yield_rank_06c_mc[i])

    rho_yield_pareto_mc = spearman(yield_rank_06c_mc, pareto_rank_m)
    rho_yield_balanced_mc = spearman(yield_rank_06c_mc, balanced_rank_mc)

    # ----- Part D — pick 4 representative recipes -----
    def pick_min(rows, key):
        finite = [r for r in rows if np.isfinite(_safe_float(r.get(key)))]
        return min(finite, key=lambda r: _safe_float(r.get(key))) if finite else None

    def pick_max(rows, key):
        finite = [r for r in rows if np.isfinite(_safe_float(r.get(key)))]
        return max(finite, key=lambda r: _safe_float(r.get(key))) if finite else None

    rep_cd = pick_min(nom_rows, "CD_error_nm")
    rep_ler = pick_min(nom_rows, "LER_CD_locked_nm")
    rep_balanced = pick_min(mc_rows, "balanced_score_mc")
    rep_margin = pick_max(mc_rows, "mean_p_line_margin")

    # Build a normalised radar payload over a fixed axis set so the four
    # representatives sit on the same scale. Axes must be "higher-is-better".
    radar_axes = [
        "CD_accuracy",     # 1 - CD_error_nm / max(CD_error_nm)
        "LER_quality",     # 1 - LER / max(LER)
        "MC_CD_stability", # 1 - std_cd / max(std_cd)
        "MC_LER_stability", # 1 - std_ler / max(std_ler)
        "robustness",      # robust_prob (or 1 if nominal-only)
        "margin",          # P_line_margin / max(P_line_margin)
    ]

    def _safe_div(a: float, b: float) -> float:
        if not np.isfinite(a) or not np.isfinite(b) or abs(b) < 1e-12:
            return 0.0
        return float(a) / float(b)

    cd_max  = float(np.nanmax([r["CD_error_nm"] for r in nom_rows])) or 1.0
    ler_max = float(np.nanmax([r["LER_CD_locked_nm"] for r in nom_rows])) or 1.0
    cd_std_max  = float(np.nanmax([r["std_cd_final"] for r in mc_rows])) or 1.0
    ler_std_max = float(np.nanmax([r["std_ler_locked"] for r in mc_rows])) or 1.0
    margin_max = float(np.nanmax([r["P_line_margin_fd"] for r in nom_rows])) or 1.0

    def _radar_for(rid: str, source_kind: str) -> dict[str, float]:
        nom = next((r for r in nom_rows if r["recipe_id"] == rid), {})
        mc = next((r for r in mc_rows if r["recipe_id"] == rid), {})
        return {
            "CD_accuracy":      max(0.0, 1.0 - _safe_div(nom.get("CD_error_nm", 0.0), cd_max)),
            "LER_quality":      max(0.0, 1.0 - _safe_div(nom.get("LER_CD_locked_nm", 0.0), ler_max)),
            "MC_CD_stability":  max(0.0, 1.0 - _safe_div(mc.get("std_cd_final", 0.0), cd_std_max)) if mc else 0.0,
            "MC_LER_stability": max(0.0, 1.0 - _safe_div(mc.get("std_ler_locked", 0.0), ler_std_max)) if mc else 0.0,
            "robustness":       float(mc.get("robust_prob", 1.0)) if mc else 1.0,
            "margin":           min(1.0, max(0.0, _safe_div(nom.get("P_line_margin_fd", 0.0), margin_max))),
        }

    rep_payloads = []
    for kind, src in [("CD-best", rep_cd),
                       ("LER-best", rep_ler),
                       ("balanced-best", rep_balanced),
                       ("margin-best", rep_margin)]:
        if src is None:
            continue
        rid = src["recipe_id"]
        nom = next((r for r in nom_rows if r["recipe_id"] == rid), {})
        mc  = next((r for r in mc_rows  if r["recipe_id"] == rid), {})
        rep_payloads.append({
            "kind":             kind,
            "recipe_id":        rid,
            "rank_06c":         int(_safe_float(sur_lookup.get(rid, {}).get("rank_06c", 0))),
            "yield_score_06c":  _safe_float(sur_lookup.get(rid, {}).get("yield_score_06c")),
            "fd_label":         nom.get("fd_label", ""),
            "CD_final_nm":      nom.get("CD_final_nm", float("nan")),
            "CD_error_nm":      nom.get("CD_error_nm", float("nan")),
            "LER_CD_locked_nm": nom.get("LER_CD_locked_nm", float("nan")),
            "P_line_margin_fd": nom.get("P_line_margin_fd", float("nan")),
            "mean_cd_error_mc": mc.get("mean_cd_error",   float("nan")) if mc else float("nan"),
            "std_cd_final_mc":  mc.get("std_cd_final",    float("nan")) if mc else float("nan"),
            "mean_ler_mc":      mc.get("mean_ler_locked", float("nan")) if mc else float("nan"),
            "std_ler_mc":       mc.get("std_ler_locked",  float("nan")) if mc else float("nan"),
            "robust_prob":      mc.get("robust_prob",     float("nan")) if mc else float("nan"),
            "margin_risk_prob": mc.get("margin_risk_prob", float("nan")) if mc else float("nan"),
            "defect_prob":      mc.get("defect_prob",     float("nan")) if mc else float("nan"),
            "mean_p_line_margin_mc": mc.get("mean_p_line_margin", float("nan")) if mc else float("nan"),
            "balanced_score_nom": nom.get("balanced_score", float("nan")),
            "balanced_score_mc":  mc.get("balanced_score_mc", float("nan")) if mc else float("nan"),
            "nominal_pareto_rank": nom.get("nominal_pareto_rank", -1),
            "mc_pareto_rank":     mc.get("mc_pareto_rank", -1) if mc else -1,
            **{k: _safe_float(sur_lookup.get(rid, {}).get(k)) for k in FEATURE_KEYS},
            "radar_norm":        _radar_for(rid, kind),
        })

    op_radar = {
        "CD_accuracy":      max(0.0, 1.0 - _safe_div(op_nom.get("CD_error_nm", 0.0), cd_max)) if op_nom else 0.0,
        "LER_quality":      max(0.0, 1.0 - _safe_div(op_nom.get("LER_CD_locked_nm", 0.0), ler_max)) if op_nom else 0.0,
        "MC_CD_stability":  max(0.0, 1.0 - _safe_div(op_mc.get("std_cd_final", 0.0), cd_std_max)) if op_mc else 0.0,
        "MC_LER_stability": max(0.0, 1.0 - _safe_div(op_mc.get("std_ler_locked", 0.0), ler_std_max)) if op_mc else 0.0,
        "robustness":       float(op_mc.get("robust_prob", 1.0)) if op_mc else 1.0,
        "margin":           min(1.0, max(0.0, _safe_div(op_nom.get("P_line_margin_fd", 0.0), margin_max))) if op_nom else 0.0,
    }

    # ----- Part E — threshold sensitivity (hypothetical, not spec) -----
    cd_tols = [1.0, 0.75, 0.5]
    ler_caps = [3.0, 2.7, 2.5]
    survival_rows = []
    for cd_tol in cd_tols:
        for ler_cap in ler_caps:
            n_surv = sum(
                1 for r in nom_rows
                if r["CD_error_nm"] <= cd_tol and r["LER_CD_locked_nm"] <= ler_cap
                   and r["fd_label"] == "robust_valid"
            )
            n_surv_mc = sum(
                1 for r in mc_rows
                if r["mean_cd_error"] <= cd_tol and r["mean_ler_locked"] <= ler_cap
                   and r["defect_prob"] == 0.0
            )
            survival_rows.append({
                "cd_tol_nm":        float(cd_tol),
                "ler_cap_nm":       float(ler_cap),
                "n_survivors":      int(n_surv),
                "n_survivors_top10_mc_strict": int(n_surv_mc),
                "n_total_top100":   len(nom_rows),
                "n_total_top10_mc": len(mc_rows),
            })

    # ----- Outputs (CSVs) -----
    yopt_dir = V3_DIR / "outputs" / "yield_optimization"
    logs_dir = V3_DIR / "outputs" / "logs"
    fig_dir  = V3_DIR / "outputs" / "figures" / "06_yield_optimization"
    fig_dir.mkdir(parents=True, exist_ok=True)

    nom_cols = [
        "recipe_id", "rank_surrogate", "yield_score_06c", "yield_score_06a",
        "fd_label", "CD_final_nm", "CD_error_nm",
        "LER_CD_locked_nm", "P_line_margin_fd", "area_frac_fd",
        "defect_indicator",
        "nominal_pareto_rank", "nominal_crowding_distance",
        "balanced_score", "balanced_rank",
        "cd_only_rank", "ler_only_rank", "yield_score_rank_06c",
    ] + FEATURE_KEYS
    _write_csv(nom_rows, yopt_dir / "stage06F_pareto_nominal.csv",
                column_order=nom_cols)

    mc_cols = [
        "recipe_id", "rank_06c", "yield_score_06c", "yield_score_06a",
        "n_mc", "mean_cd_final", "mean_cd_error", "std_cd_final",
        "mean_ler_locked", "std_ler_locked",
        "mean_p_line_margin", "std_p_line_margin",
        "robust_prob", "margin_risk_prob", "defect_prob",
        "nominal_CD_error_nm", "nominal_LER_CD_locked_nm",
        "nominal_P_line_margin",
        "mc_pareto_rank", "mc_crowding_distance",
        "balanced_score_mc", "balanced_rank_mc", "yield_score_rank_06c_mc",
    ] + FEATURE_KEYS
    _write_csv(mc_rows, yopt_dir / "stage06F_pareto_mc.csv",
                column_order=mc_cols)

    rep_cols = [
        "kind", "recipe_id", "rank_06c", "yield_score_06c", "fd_label",
        "CD_final_nm", "CD_error_nm", "LER_CD_locked_nm", "P_line_margin_fd",
        "mean_cd_error_mc", "std_cd_final_mc",
        "mean_ler_mc", "std_ler_mc",
        "robust_prob", "margin_risk_prob", "defect_prob",
        "mean_p_line_margin_mc",
        "balanced_score_nom", "balanced_score_mc",
        "nominal_pareto_rank", "mc_pareto_rank",
    ] + FEATURE_KEYS
    _write_csv(rep_payloads, yopt_dir / "stage06F_representative_recipes.csv",
                column_order=rep_cols)

    _write_csv(survival_rows, yopt_dir / "stage06F_threshold_sensitivity.csv",
                column_order=["cd_tol_nm", "ler_cap_nm", "n_survivors",
                                "n_survivors_top10_mc_strict",
                                "n_total_top100", "n_total_top10_mc"])

    # ----- Figures -----
    plot_cd_error_vs_ler_pareto(nom_rows, op_nom,
                                  fig_dir / "stage06F_cd_error_vs_ler_pareto.png")
    plot_pareto_colored_by_margin(nom_rows, op_nom,
                                    fig_dir / "stage06F_pareto_front_colored_by_margin.png")
    plot_mc_std_scatter(mc_rows, op_mc,
                         fig_dir / "stage06F_mc_cd_std_vs_ler_std.png")
    plot_representative_radar(rep_payloads, op_radar, radar_axes,
                                fig_dir / "stage06F_representative_recipe_radar.png")
    plot_threshold_heatmap(survival_rows, cd_tols, ler_caps, len(nom_rows),
                            fig_dir / "stage06F_threshold_survival_heatmap.png")

    # ----- Acceptance + JSON summary -----
    acceptance = {
        "pareto_nominal_computed":     bool(nom_rows),
        "pareto_mc_computed":          bool(mc_rows),
        "n_nominal":                   len(nom_rows),
        "n_mc":                        len(mc_rows),
        "n_pareto_rank1_nominal":      int(np.sum(pareto_rank_n == 1)),
        "n_pareto_rank1_mc":           int(np.sum(pareto_rank_m == 1)),
        "balanced_representative_recipe_id": rep_balanced["recipe_id"] if rep_balanced else None,
        "n_threshold_sensitivity_cells":     len(survival_rows),
        "policy_v2_OP_frozen":         bool(cfg["policy"].get("v2_OP_frozen", True)),
        "policy_published_data_loaded": bool(cfg["policy"].get("published_data_loaded", False)),
        "policy_external_calibration": "none",
        "thresholds_treated_as_spec":  False,
    }

    payload = {
        "stage": "06F",
        "policy": cfg["policy"],
        "input_files": {
            "fd_nominal":     str(args.fd_nominal_csv),
            "fd_mc":          str(args.fd_mc_csv),
            "fd_baseline":    str(args.fd_baseline_csv),
            "fd_baseline_mc": str(args.fd_baseline_mc_csv),
            "top_06d":        str(args.top_06d_csv),
        },
        "v2_frozen_op_nominal": op_nom,
        "v2_frozen_op_mc":      op_mc,
        "ranking_correlations": {
            "nominal": {
                "spearman_yield_vs_pareto":    rho_yield_pareto,
                "spearman_yield_vs_cd_only":   rho_yield_cd,
                "spearman_yield_vs_ler_only":  rho_yield_ler,
                "spearman_yield_vs_balanced":  rho_yield_balanced,
                "spearman_pareto_vs_balanced": rho_pareto_balanced,
            },
            "mc": {
                "spearman_yield_vs_pareto_mc":   rho_yield_pareto_mc,
                "spearman_yield_vs_balanced_mc": rho_yield_balanced_mc,
            },
        },
        "pareto_nominal": {
            "n_rank1": int(np.sum(pareto_rank_n == 1)),
            "n_rank2": int(np.sum(pareto_rank_n == 2)),
            "n_rank3": int(np.sum(pareto_rank_n == 3)),
        },
        "pareto_mc": {
            "n_rank1": int(np.sum(pareto_rank_m == 1)),
            "n_rank2": int(np.sum(pareto_rank_m == 2)),
            "n_rank3": int(np.sum(pareto_rank_m == 3)),
        },
        "representative_recipes": [
            {k: v for k, v in r.items() if k != "radar_norm"}
            for r in rep_payloads
        ],
        "threshold_sensitivity_caveat": (
            "These thresholds are HYPOTHETICAL exploration grids. "
            "They are NOT spec values, NOT externally calibrated, and "
            "must not be cited as pass/fail truth."
        ),
        "threshold_sensitivity": survival_rows,
        "acceptance": acceptance,
    }

    (logs_dir / "stage06F_summary.json").write_text(
        json.dumps(payload, indent=2, default=float))

    # ----- Console summary -----
    print(f"\nStage 06F -- Pareto ranking summary")
    print(f"  v2 frozen OP nominal: CD_error={op_nom.get('CD_error_nm', float('nan')):.3f} nm  "
          f"LER={op_nom.get('LER_CD_locked_nm', float('nan')):.3f} nm  "
          f"P_margin={op_nom.get('P_line_margin_fd', float('nan')):.3f}")
    print(f"  v2 frozen OP MC:      mean_CD_error={op_mc.get('mean_cd_error', float('nan')):.3f}  "
          f"std_CD={op_mc.get('std_cd_final', float('nan')):.3f}  "
          f"std_LER={op_mc.get('std_ler_locked', float('nan')):.3f}  "
          f"defect_prob={op_mc.get('defect_prob', float('nan')):.3f}")
    print(f"  Pareto nominal: rank-1 = {int(np.sum(pareto_rank_n == 1))} / {len(nom_rows)}")
    print(f"  Pareto MC:      rank-1 = {int(np.sum(pareto_rank_m == 1))} / {len(mc_rows)}")
    print(f"  Ranking correlations (Spearman of nominal ranks):")
    print(f"    yield_score vs Pareto:    rho = {rho_yield_pareto:.3f}")
    print(f"    yield_score vs CD-only:   rho = {rho_yield_cd:.3f}")
    print(f"    yield_score vs LER-only:  rho = {rho_yield_ler:.3f}")
    print(f"    yield_score vs balanced:  rho = {rho_yield_balanced:.3f}")
    print(f"    Pareto vs balanced:       rho = {rho_pareto_balanced:.3f}")
    print(f"  MC correlations:")
    print(f"    yield_score vs Pareto_MC:    rho = {rho_yield_pareto_mc:.3f}")
    print(f"    yield_score vs balanced_MC:  rho = {rho_yield_balanced_mc:.3f}")
    print(f"  Representative recipes:")
    for r in rep_payloads:
        print(f"    {r['kind']:>14}: {r['recipe_id']}  "
              f"(rank_06c=#{r['rank_06c']}, "
              f"CD_err={r['CD_error_nm']:.3f} nm, "
              f"LER={r['LER_CD_locked_nm']:.3f} nm, "
              f"defect_prob={r.get('defect_prob', 0.0):.3f})")
    print(f"  Threshold sensitivity cells: {len(survival_rows)} (hypothetical only)")
    print(f"  Acceptance: {acceptance}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

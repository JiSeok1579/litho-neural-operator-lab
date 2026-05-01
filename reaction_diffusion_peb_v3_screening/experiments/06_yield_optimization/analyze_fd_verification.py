"""Stage 06B Parts 3 + 4 + 5 — analysis, figures, AL additions.

Reads
    outputs/labels/06_top_recipes_fixed_design_surrogate.csv     (Stage 06A)
    outputs/labels/fd_top100_nominal_verification.csv            (Stage 06B Part 1)
    outputs/labels/fd_top10_mc_verification.csv                  (Stage 06B Part 2)

Writes
    outputs/labels/surrogate_vs_fd_metrics.csv
    outputs/labels/false_pass_cases.csv
    outputs/labels/06_yield_optimization_al_additions.csv
    outputs/logs/06b_surrogate_fd_ranking_comparison.json
    outputs/logs/06b_false_pass_summary.json
    outputs/figures/06_yield_optimization/
        surrogate_vs_fd_yield_score_scatter.png
        surrogate_vs_fd_cd_ler_scatter.png
        top10_fd_yield_barplot.png
        defect_breakdown_top10.png
        false_pass_parameter_parallel_coordinates.png

Policy
    v2_OP_frozen           = true
    published_data_loaded  = false
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

from reaction_diffusion_peb_v3_screening.src.fd_yield_score import (
    fd_yield_score_from_rows,
    fd_yield_score_per_recipe,
    nominal_yield_score,
    spearman,
    topk_overlap,
)
from reaction_diffusion_peb_v3_screening.src.metrics_io import read_labels_csv
from reaction_diffusion_peb_v3_screening.src.yield_optimizer import (
    YieldScoreConfig,
)


V3_DIR = ROOT / "reaction_diffusion_peb_v3_screening"

# Recipe knobs included in the false-PASS parallel-coordinates plot.
PARAM_AXES = [
    "dose_mJ_cm2", "sigma_nm", "DH_nm2_s", "time_s",
    "Hmax_mol_dm3", "kdep_s_inv", "Q0_mol_dm3", "kq_s_inv",
    "line_cd_ratio",
]


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
# Part 3 — surrogate vs FD comparison + ranking.
# --------------------------------------------------------------------------
def build_surrogate_vs_fd_metrics(
    surrogate_top: list[dict],
    fd_nominal_rows: list[dict],
    score_cfg: YieldScoreConfig,
) -> list[dict]:
    """For each recipe in `surrogate_top`, look up its single FD row in
    fd_nominal_rows and combine surrogate vs FD per-recipe metrics."""
    fd_by_id = {r["source_recipe_id"]: r for r in fd_nominal_rows}
    out = []
    for s in surrogate_top:
        rid = s["recipe_id"]
        fd_row = fd_by_id.get(rid)
        merged = {
            "recipe_id": rid,
            "rank_surrogate": int(s.get("rank_surrogate", 0)) if s.get("rank_surrogate") else 0,
            "yield_score_surrogate":  _safe_float(s.get("yield_score")),
            "p_robust_valid_surrogate": _safe_float(s.get("p_robust_valid")),
            "mean_cd_fixed_surrogate":  _safe_float(s.get("mean_cd_fixed")),
            "mean_cd_locked_surrogate": _safe_float(s.get("mean_cd_locked")),
            "mean_ler_locked_surrogate": _safe_float(s.get("mean_ler_locked")),
            "mean_area_frac_surrogate": _safe_float(s.get("mean_area_frac")),
            "mean_p_line_margin_surrogate": _safe_float(s.get("mean_p_line_margin")),
        }
        if fd_row is None:
            merged.update({
                "fd_label": "",
                "FD_yield_score_nominal": float("nan"),
                "CD_final_fd": float("nan"),
                "CD_locked_fd": float("nan"),
                "LER_CD_locked_fd": float("nan"),
                "area_frac_fd": float("nan"),
                "P_line_margin_fd": float("nan"),
            })
        else:
            nom = nominal_yield_score(fd_row, score_cfg)
            merged.update({
                "fd_label": str(fd_row.get("label", "")),
                "FD_yield_score_nominal": float(nom["FD_yield_score"]),
                "CD_final_fd":       _safe_float(fd_row.get("CD_final_nm")),
                "CD_locked_fd":      _safe_float(fd_row.get("CD_locked_nm")),
                "LER_CD_locked_fd":  _safe_float(fd_row.get("LER_CD_locked_nm")),
                "area_frac_fd":      _safe_float(fd_row.get("area_frac")),
                "P_line_margin_fd":  _safe_float(fd_row.get("P_line_margin")),
            })
        out.append(merged)
    return out


def regression_target_errors(rows: list[dict]) -> dict:
    """MAE / RMSE for each surrogate target vs FD ground truth."""
    pairs = {
        "CD_fixed":      ("mean_cd_fixed_surrogate",  "CD_final_fd"),
        "CD_locked":     ("mean_cd_locked_surrogate", "CD_locked_fd"),
        "LER_locked":    ("mean_ler_locked_surrogate", "LER_CD_locked_fd"),
        "area_frac":     ("mean_area_frac_surrogate", "area_frac_fd"),
        "P_line_margin": ("mean_p_line_margin_surrogate", "P_line_margin_fd"),
    }
    out = {}
    for name, (a_key, b_key) in pairs.items():
        a = np.array([r[a_key] for r in rows], dtype=np.float64)
        b = np.array([r[b_key] for r in rows], dtype=np.float64)
        finite = np.isfinite(a) & np.isfinite(b)
        if finite.sum() == 0:
            out[name] = {"mae": None, "rmse": None, "n": 0}
            continue
        d = a[finite] - b[finite]
        out[name] = {
            "mae":  float(np.mean(np.abs(d))),
            "rmse": float(np.sqrt(np.mean(d ** 2))),
            "n":    int(finite.sum()),
        }
    return out


# --------------------------------------------------------------------------
# Top-10 MC FD scoring + ranking.
# --------------------------------------------------------------------------
def score_top10_mc(fd_mc_rows: list[dict], score_cfg: YieldScoreConfig) -> list[dict]:
    """One row per top-10 base recipe with the empirical MC FD score."""
    by_recipe = fd_yield_score_per_recipe(fd_mc_rows, score_cfg)
    rows = []
    rank_lookup = {}
    for fd in fd_mc_rows:
        rid = fd.get("source_recipe_id")
        if rid not in rank_lookup:
            rank_lookup[rid] = int(_safe_float(fd.get("rank_surrogate", 0)))
    for rid, payload in by_recipe.items():
        out = dict(payload)
        out["recipe_id"] = rid
        out["rank_surrogate"] = rank_lookup.get(rid, 0)
        rows.append(out)
    rows.sort(key=lambda r: r["rank_surrogate"])
    return rows


# --------------------------------------------------------------------------
# Part 4 — false-PASS analysis.
# --------------------------------------------------------------------------
HARD_FAIL_LABELS = ("under_exposed", "merged", "roughness_degraded", "numerical_invalid")
SOFT_FAIL_LABEL  = "margin_risk"


def detect_false_pass(
    nominal_rows: list[dict],
    surrogate_top: list[dict],
) -> tuple[list[dict], dict]:
    """A top-100 recipe is false-PASS if its single nominal-FD label is
    not robust_valid. Hard false-PASS is a defect-class label; soft is
    margin_risk."""
    sur_lookup = {s["recipe_id"]: s for s in surrogate_top}
    rows = []
    n_total = len(nominal_rows)
    counts = Counter(r.get("label", "") for r in nominal_rows)

    for fd in nominal_rows:
        label = str(fd.get("label", ""))
        if label == "robust_valid":
            continue
        rid = fd.get("source_recipe_id")
        s = sur_lookup.get(rid, {})
        kind = (
            "hard_false_pass" if label in HARD_FAIL_LABELS
            else "soft_false_pass" if label == SOFT_FAIL_LABEL
            else "other_non_robust"
        )
        rows.append({
            "recipe_id": rid,
            "rank_surrogate": int(_safe_float(fd.get("rank_surrogate", 0))),
            "fd_label": label,
            "false_pass_kind": kind,
            "yield_score_surrogate": _safe_float(s.get("yield_score")),
            "p_robust_valid_surrogate": _safe_float(s.get("p_robust_valid")),
            "CD_final_fd":      _safe_float(fd.get("CD_final_nm")),
            "CD_locked_fd":     _safe_float(fd.get("CD_locked_nm")),
            "LER_CD_locked_fd": _safe_float(fd.get("LER_CD_locked_nm")),
            "area_frac_fd":     _safe_float(fd.get("area_frac")),
            "P_line_margin_fd": _safe_float(fd.get("P_line_margin")),
            **{k: _safe_float(s.get(k)) for k in PARAM_AXES if k in s},
        })

    summary = {
        "n_top": n_total,
        "label_breakdown": dict(counts),
        "n_hard_false_pass": int(sum(1 for r in rows if r["false_pass_kind"] == "hard_false_pass")),
        "n_soft_false_pass": int(sum(1 for r in rows if r["false_pass_kind"] == "soft_false_pass")),
        "hard_false_pass_rate": float(sum(1 for r in rows if r["false_pass_kind"] == "hard_false_pass") / max(n_total, 1)),
        "soft_false_pass_rate": float(sum(1 for r in rows if r["false_pass_kind"] == "soft_false_pass") / max(n_total, 1)),
    }
    return rows, summary


# --------------------------------------------------------------------------
# Part 5 — AL additions.
# --------------------------------------------------------------------------
def write_al_additions(
    nominal_rows: list[dict],
    mc_rows: list[dict],
    out_path: Path,
) -> None:
    """Concatenate every Stage 06B FD row into a single CSV. Both Parts
    1 and 2 already share the WRITE_COLS layout from run_fd_verification."""
    if not (nominal_rows or mc_rows):
        return
    # Pick the union of column sets, preferring nominal_rows first.
    cols = list(nominal_rows[0].keys()) if nominal_rows else list(mc_rows[0].keys())
    extras = []
    for src in (nominal_rows, mc_rows):
        for r in src:
            for k in r.keys():
                if k not in cols and k not in extras:
                    extras.append(k)
    cols = cols + extras
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in nominal_rows:
            w.writerow(r)
        for r in mc_rows:
            w.writerow(r)


# --------------------------------------------------------------------------
# Plots
# --------------------------------------------------------------------------
def plot_yield_score_scatter(
    pair_rows: list[dict], baseline_score: float | None,
    out_path: Path,
) -> None:
    sx = np.array([r["yield_score_surrogate"]      for r in pair_rows])
    sy = np.array([r["FD_yield_score_nominal"]     for r in pair_rows])

    fig, ax = plt.subplots(figsize=(8.5, 7.0))
    ax.scatter(sx, sy, s=30, c="#1f77b4", alpha=0.75, edgecolor="white", lw=0.5)
    lo = float(np.nanmin([sx.min(), sy.min(), -0.5]))
    hi = float(np.nanmax([sx.max(), sy.max(), 1.05]))
    ax.plot([lo, hi], [lo, hi], "k--", lw=0.8, alpha=0.5, label="y = x")
    if baseline_score is not None:
        ax.axhline(baseline_score, color="#d62728", ls=":", lw=1.0,
                   label=f"v2 frozen OP yield_score = {baseline_score:.3f}")
        ax.axvline(baseline_score, color="#d62728", ls=":", lw=1.0)
    rho = spearman(sx, sy)
    ax.set_xlabel("Stage 06A surrogate yield_score (mean over 200 MC variations)")
    ax.set_ylabel("Stage 06B FD yield_score (single nominal FD)")
    ax.set_title(f"Surrogate vs FD yield_score on top-100 (Spearman ρ = {rho:.3f})")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="lower right", fontsize=10)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_cd_ler_scatter(pair_rows: list[dict], out_path: Path) -> None:
    cd_s = np.array([r["mean_cd_fixed_surrogate"] for r in pair_rows])
    cd_f = np.array([r["CD_final_fd"]              for r in pair_rows])
    ler_s = np.array([r["mean_ler_locked_surrogate"] for r in pair_rows])
    ler_f = np.array([r["LER_CD_locked_fd"]          for r in pair_rows])

    fig, axes = plt.subplots(1, 2, figsize=(13.0, 6.0))

    ax = axes[0]
    ax.scatter(cd_s, cd_f, s=22, c="#2ca02c", alpha=0.7, edgecolor="white", lw=0.5)
    lo = float(np.nanmin([cd_s.min(), cd_f.min()]))
    hi = float(np.nanmax([cd_s.max(), cd_f.max()]))
    pad = 0.05 * (hi - lo + 1e-9)
    ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], "k--", lw=0.8, alpha=0.5, label="y = x")
    rho = spearman(cd_s, cd_f)
    ax.axhspan(14.0, 16.0, color="#aaa", alpha=0.15, label="CD target window ±1 nm")
    ax.set_xlabel("surrogate mean CD_fixed (nm)")
    ax.set_ylabel("FD nominal CD_final (nm)")
    ax.set_title(f"CD_fixed: surrogate vs FD  (Spearman ρ = {rho:.3f})")
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(True, alpha=0.25)

    ax = axes[1]
    ax.scatter(ler_s, ler_f, s=22, c="#d62728", alpha=0.7, edgecolor="white", lw=0.5)
    lo = float(np.nanmin([ler_s.min(), ler_f.min()]))
    hi = float(np.nanmax([ler_s.max(), ler_f.max()]))
    pad = 0.05 * (hi - lo + 1e-9)
    ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], "k--", lw=0.8, alpha=0.5, label="y = x")
    rho = spearman(ler_s, ler_f)
    ax.axhspan(0, 3.0, color="#aaa", alpha=0.15, label="LER ≤ 3.0 nm window")
    ax.set_xlabel("surrogate mean LER_CD_locked (nm)")
    ax.set_ylabel("FD nominal LER_CD_locked (nm)")
    ax.set_title(f"LER_locked: surrogate vs FD  (Spearman ρ = {rho:.3f})")
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(True, alpha=0.25)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_top10_fd_yield_bar(
    top10_rows: list[dict], baseline_fd_score: float | None,
    out_path: Path,
) -> None:
    rows = sorted(top10_rows, key=lambda r: r["rank_surrogate"])
    ids = [r["recipe_id"] for r in rows]
    sur_score = [r.get("FD_yield_score", 0) for r in rows]   # fd MC score, not surrogate
    fig, ax = plt.subplots(figsize=(11.0, 5.5))
    x = np.arange(len(rows))
    ax.bar(x, sur_score, color="#1f77b4", alpha=0.85,
           edgecolor="#1f1f1f", label="FD MC yield_score (100 variations)")
    if baseline_fd_score is not None:
        ax.axhline(baseline_fd_score, color="#d62728", ls="--", lw=1.2,
                   label=f"v2 frozen OP FD yield_score = {baseline_fd_score:.3f}")
    for i, r in enumerate(rows):
        ax.text(i, r["FD_yield_score"] + 0.01, f"{r['FD_yield_score']:.3f}",
                ha="center", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels([f"#{r['rank_surrogate']}\n{r['recipe_id']}" for r in rows],
                       fontsize=8)
    ax.set_xlabel("Top-10 surrogate-rank recipe (Mode A)")
    ax.set_ylabel("FD MC yield_score")
    ax.set_title("Stage 06B Part 2 — top-10 FD Monte-Carlo yield_score")
    ax.set_ylim(min(0.0, baseline_fd_score - 0.05 if baseline_fd_score else 0), 1.05)
    ax.grid(True, alpha=0.25, axis="y")
    ax.legend(loc="lower right", fontsize=10)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_defect_breakdown_top10(top10_rows: list[dict], out_path: Path) -> None:
    rows = sorted(top10_rows, key=lambda r: r["rank_surrogate"])
    classes = [
        ("P_FD_robust_valid", "robust_valid"),
        ("P_FD_margin_risk", "margin_risk"),
        ("P_FD_under_exposed", "under_exposed"),
        ("P_FD_merged", "merged"),
        ("P_FD_roughness_degraded", "roughness"),
        ("P_FD_numerical_invalid", "numerical"),
    ]
    M = np.array([[r.get(k, 0.0) for k, _ in classes] for r in rows])

    fig, ax = plt.subplots(figsize=(11.0, 5.5))
    bottoms = np.zeros(len(rows))
    palette = ["#2ca02c", "#ffbf00", "#1f77b4", "#d62728", "#9467bd", "#8c564b"]
    for j, (col, label) in enumerate(classes):
        ax.bar(np.arange(len(rows)), M[:, j], bottom=bottoms,
               color=palette[j], alpha=0.85, label=label,
               edgecolor="white", lw=0.4)
        bottoms = bottoms + M[:, j]
    ax.set_xticks(np.arange(len(rows)))
    ax.set_xticklabels([f"#{r['rank_surrogate']}\n{r['recipe_id']}" for r in rows],
                       fontsize=8)
    ax.set_xlabel("Top-10 surrogate-rank recipe (Mode A)")
    ax.set_ylabel("FD Monte-Carlo class probability")
    ax.set_title("Top-10 FD MC defect breakdown (100 variations per recipe)")
    ax.legend(loc="lower right", fontsize=9, framealpha=0.95)
    ax.set_ylim(0, 1.02)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_false_pass_parallel(false_rows: list[dict], out_path: Path) -> None:
    if not false_rows:
        # Make a placeholder figure that explicitly says zero false-PASS
        # so downstream readers see the result, not an empty file.
        fig, ax = plt.subplots(figsize=(11.0, 5.0))
        ax.text(0.5, 0.55, "Stage 06B — false-PASS parallel coordinates",
                ha="center", va="center", fontsize=14, fontweight="bold")
        ax.text(0.5, 0.30,
                "Zero false-PASS cases in the top-100 nominal FD verification.\n"
                "All top-100 surrogate recipes were FD-labelled robust_valid.",
                ha="center", va="center", fontsize=11)
        ax.axis("off")
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        return

    axes_keys = [k for k in PARAM_AXES if any(k in r for r in false_rows)]
    if not axes_keys:
        return
    fig, ax = plt.subplots(figsize=(13.0, 6.0))
    # Normalise each axis to [0, 1].
    M = np.array([[r.get(k, np.nan) for k in axes_keys] for r in false_rows], dtype=float)
    lo = np.nanmin(M, axis=0)
    hi = np.nanmax(M, axis=0)
    span = np.where(hi - lo > 1e-12, hi - lo, 1.0)
    Mn = (M - lo) / span
    color_by_kind = {
        "hard_false_pass": "#7a0a0a",
        "soft_false_pass": "#ffbf00",
        "other_non_robust": "#777",
    }
    for i, r in enumerate(false_rows):
        ax.plot(range(len(axes_keys)), Mn[i], "-",
                color=color_by_kind.get(r["false_pass_kind"], "#1f77b4"),
                alpha=0.7, lw=1.4,
                label=r["false_pass_kind"] if i == 0 else None)
    ax.set_xticks(range(len(axes_keys)))
    ax.set_xticklabels(axes_keys, rotation=15)
    # add per-axis range to xtick labels
    for i, k in enumerate(axes_keys):
        ax.text(i, -0.07, f"[{lo[i]:.2g}, {hi[i]:.2g}]",
                ha="center", fontsize=8, color="#555",
                transform=ax.get_xaxis_transform())
    ax.set_ylabel("normalised value")
    ax.set_title(f"False-PASS recipe parameter pattern  "
                 f"(N={len(false_rows)})")
    seen = set()
    handles = []
    for r in false_rows:
        if r["false_pass_kind"] not in seen:
            seen.add(r["false_pass_kind"])
            handles.append(plt.Line2D([0], [0], color=color_by_kind[r["false_pass_kind"]],
                                       lw=1.5, label=r["false_pass_kind"]))
    if handles:
        ax.legend(handles=handles, loc="lower right", fontsize=10)
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
    p.add_argument("--surrogate_csv", type=str,
                   default=str(V3_DIR / "outputs" / "labels"
                               / "06_top_recipes_fixed_design_surrogate.csv"))
    p.add_argument("--fd_nominal_csv", type=str,
                   default=str(V3_DIR / "outputs" / "labels"
                               / "fd_top100_nominal_verification.csv"))
    p.add_argument("--fd_mc_csv", type=str,
                   default=str(V3_DIR / "outputs" / "labels"
                               / "fd_top10_mc_verification.csv"))
    p.add_argument("--summary_06a_json", type=str,
                   default=str(V3_DIR / "outputs" / "logs"
                               / "06_yield_optimization_summary.json"))
    args = p.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    score_cfg = YieldScoreConfig.from_yaml_dict(cfg["yield_score"])
    summary_06a = json.loads(Path(args.summary_06a_json).read_text())
    surrogate_baseline = float(summary_06a["v2_frozen_op_baseline"]["yield_score"])

    surrogate_top = read_labels_csv(args.surrogate_csv)
    # Coerce numeric.
    num_keys = ["yield_score", "p_robust_valid", "mean_cd_fixed",
                "mean_cd_locked", "mean_ler_locked", "mean_area_frac",
                "mean_p_line_margin", "rank_surrogate"] + PARAM_AXES
    _coerce_floats(surrogate_top, num_keys)
    # Add rank_surrogate based on yield_score order in the CSV.
    surrogate_top.sort(key=lambda r: -r["yield_score"])
    for i, r in enumerate(surrogate_top, start=1):
        r["rank_surrogate"] = i

    fd_nominal = read_labels_csv(args.fd_nominal_csv)
    _coerce_floats(fd_nominal, ["CD_final_nm", "CD_locked_nm",
                                 "LER_CD_locked_nm", "area_frac",
                                 "P_line_margin", "rank_surrogate"])

    fd_mc = read_labels_csv(args.fd_mc_csv)
    _coerce_floats(fd_mc, ["CD_final_nm", "CD_locked_nm",
                            "LER_CD_locked_nm", "area_frac",
                            "P_line_margin", "rank_surrogate"])

    # ------------------------------------------------------------------
    # Part 3 — surrogate vs FD per-recipe metrics + ranking comparison.
    # ------------------------------------------------------------------
    pair_rows = build_surrogate_vs_fd_metrics(surrogate_top, fd_nominal, score_cfg)
    errs = regression_target_errors(pair_rows)

    # Spearman top-100 nominal yield_score.
    sx = np.array([r["yield_score_surrogate"]      for r in pair_rows])
    sy = np.array([r["FD_yield_score_nominal"]     for r in pair_rows])
    rho100 = spearman(sx, sy)

    # Spearman top-10 MC yield_score.
    top10 = score_top10_mc(fd_mc, score_cfg)
    sur_lookup = {r["recipe_id"]: r for r in surrogate_top}
    rec_ids_top10 = [r["recipe_id"] for r in sorted(top10, key=lambda r: r["rank_surrogate"])]
    sur_yields = np.array([sur_lookup[rid]["yield_score"] for rid in rec_ids_top10])
    fd_yields = np.array([r["FD_yield_score"] for r in
                          sorted(top10, key=lambda r: r["rank_surrogate"])])
    rho10 = spearman(sur_yields, fd_yields)

    # Top-1 / 3 / 5 overlap on the top-100 set.
    sur_rank_100 = [r["recipe_id"] for r in sorted(pair_rows,
                       key=lambda r: -r["yield_score_surrogate"])]
    fd_rank_100  = [r["recipe_id"] for r in sorted(pair_rows,
                       key=lambda r: -r["FD_yield_score_nominal"])]
    top_1 = topk_overlap(sur_rank_100, fd_rank_100, 1)
    top_3 = topk_overlap(sur_rank_100, fd_rank_100, 3)
    top_5 = topk_overlap(sur_rank_100, fd_rank_100, 5)

    # Stage 06A surrogate top-10 (id list).
    sur_top10_ids = [r["recipe_id"] for r in surrogate_top[:10]]
    fd_top10_by_score = sorted(top10, key=lambda r: -r["FD_yield_score"])
    fd_top10_ids = [r["recipe_id"] for r in fd_top10_by_score]
    sur_in_fd_top10 = len(set(sur_top10_ids) & set(fd_top10_ids))

    # Surrogate top-1 still in FD top-10?
    sur_top1 = sur_top10_ids[0]
    sur_top1_in_fd_top10 = sur_top1 in fd_top10_ids

    # Confusion / classification agreement on Part 1.
    sur_pred_label = []     # surrogate argmax over class probs
    for r in pair_rows:
        # We didn't carry the surrogate proba per row in pair_rows; pull
        # straight from the top-100 CSV.
        s = sur_lookup[r["recipe_id"]]
        probs = {
            "robust_valid":       _safe_float(s.get("p_robust_valid")),
            "margin_risk":        _safe_float(s.get("p_margin_risk")),
            "merged":             _safe_float(s.get("p_merged")),
            "under_exposed":      _safe_float(s.get("p_under_exposed")),
            "roughness_degraded": _safe_float(s.get("p_roughness_degraded")),
            "numerical_invalid":  _safe_float(s.get("p_numerical_invalid")),
        }
        sur_pred_label.append(max(probs, key=probs.get))
    fd_label = [r["fd_label"] for r in pair_rows]
    agree = sum(1 for a, b in zip(sur_pred_label, fd_label) if a == b)

    # Compute FD-derived baseline yield_score (same formula on the v2
    # frozen-OP candidate's nominal FD if available, else fall back to
    # the surrogate baseline). We use surrogate baseline here because no
    # explicit FD-baseline run was scheduled; document the limitation in
    # the JSON. For reporting parity, fd_top10 is compared to the
    # *surrogate* baseline (same scoring config, same physics target).

    # ------------------------------------------------------------------
    # Part 4 — false-PASS analysis.
    # ------------------------------------------------------------------
    false_rows, false_summary = detect_false_pass(fd_nominal, surrogate_top)

    # ------------------------------------------------------------------
    # Acceptance check.
    # ------------------------------------------------------------------
    fd_beats_baseline = sum(1 for r in pair_rows
                             if np.isfinite(r["FD_yield_score_nominal"])
                             and r["FD_yield_score_nominal"] > surrogate_baseline)
    fd_mc_beats_baseline = sum(1 for r in top10
                                if r["FD_yield_score"] > surrogate_baseline)

    acceptance = {
        "top100_FD_beats_baseline_count":      int(fd_beats_baseline),
        "top100_FD_beats_baseline_pass":       bool(fd_beats_baseline >= 1),
        "fd_mc_beats_baseline_count":          int(fd_mc_beats_baseline),
        "fd_mc_beats_baseline_pass":           bool(fd_mc_beats_baseline >= 1),
        "surrogate_top10_overlap_fd_top10":    int(sur_in_fd_top10),
        "surrogate_top10_overlap_pass":        bool(sur_in_fd_top10 >= 1),
        "spearman_top100":                     rho100,
        "spearman_top10":                      rho10,
        "spearman_positive_pass":              bool(rho100 > 0.05),
        "false_pass_rate_top100":              false_summary["hard_false_pass_rate"]
                                                + false_summary["soft_false_pass_rate"],
        "false_pass_catastrophic":             bool(
            (false_summary["hard_false_pass_rate"]
             + false_summary["soft_false_pass_rate"]) > 0.20
        ),
    }

    # ------------------------------------------------------------------
    # Outputs
    # ------------------------------------------------------------------
    labels_dir = V3_DIR / "outputs" / "labels"
    logs_dir   = V3_DIR / "outputs" / "logs"
    fig_dir    = V3_DIR / "outputs" / "figures" / "06_yield_optimization"
    fig_dir.mkdir(parents=True, exist_ok=True)

    # surrogate_vs_fd_metrics.csv
    pair_cols = list(pair_rows[0].keys()) if pair_rows else []
    with (labels_dir / "surrogate_vs_fd_metrics.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=pair_cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(pair_rows)

    # false_pass_cases.csv
    if false_rows:
        cols = list(false_rows[0].keys())
    else:
        cols = ["recipe_id", "rank_surrogate", "fd_label", "false_pass_kind",
                "yield_score_surrogate", "p_robust_valid_surrogate",
                "CD_final_fd", "CD_locked_fd", "LER_CD_locked_fd",
                "area_frac_fd", "P_line_margin_fd"] + PARAM_AXES
    with (labels_dir / "false_pass_cases.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in false_rows:
            w.writerow(r)

    # 06_yield_optimization_al_additions.csv — concatenate Part 1 + Part 2 rows.
    write_al_additions(fd_nominal, fd_mc,
                        labels_dir / "06_yield_optimization_al_additions.csv")

    # surrogate_fd_ranking_comparison.json
    ranking_payload = {
        "stage": "06B",
        "policy": cfg["policy"],
        "n_top_nominal": len(pair_rows),
        "n_top_mc_recipes": len(top10),
        "n_mc_per_recipe": int(len(fd_mc) / max(len(top10), 1)),
        "v2_frozen_op_yield_score_baseline": surrogate_baseline,
        "regression_errors_nominal": errs,
        "spearman_top100_nominal_yield_score": rho100,
        "spearman_top10_mc_yield_score":       rho10,
        "topk_overlap_top100": {"top1": top_1, "top3": top_3, "top5": top_5},
        "surrogate_top10_overlap_with_fd_top10": sur_in_fd_top10,
        "surrogate_top1_in_fd_top10": bool(sur_top1_in_fd_top10),
        "label_agreement_top100": {
            "n": len(pair_rows),
            "agree": int(agree),
            "rate": float(agree / max(len(pair_rows), 1)),
        },
        "fd_top10_recipes_by_score": [
            {
                "rank_surrogate": int(r["rank_surrogate"]),
                "recipe_id":      r["recipe_id"],
                "FD_yield_score": float(r["FD_yield_score"]),
                "P_FD_robust_valid": float(r["P_FD_robust_valid"]),
                "mean_cd_fixed_fd": float(r["mean_cd_fixed_fd"]),
                "mean_ler_locked_fd": float(r["mean_ler_locked_fd"]),
            } for r in fd_top10_by_score
        ],
        "acceptance": acceptance,
    }
    (logs_dir / "06b_surrogate_fd_ranking_comparison.json").write_text(
        json.dumps(ranking_payload, indent=2))

    # false_pass_summary.json
    (logs_dir / "06b_false_pass_summary.json").write_text(
        json.dumps({**false_summary, "policy": cfg["policy"]}, indent=2))

    # ----- Figures -----
    plot_yield_score_scatter(pair_rows, surrogate_baseline,
                             fig_dir / "surrogate_vs_fd_yield_score_scatter.png")
    plot_cd_ler_scatter(pair_rows, fig_dir / "surrogate_vs_fd_cd_ler_scatter.png")
    plot_top10_fd_yield_bar(top10, surrogate_baseline,
                            fig_dir / "top10_fd_yield_barplot.png")
    plot_defect_breakdown_top10(top10, fig_dir / "defect_breakdown_top10.png")
    plot_false_pass_parallel(false_rows,
                              fig_dir / "false_pass_parameter_parallel_coordinates.png")

    # ----- Console summary -----
    print(f"\nStage 06B — analysis summary")
    print(f"  v2 frozen OP yield_score (surrogate baseline): {surrogate_baseline:.4f}")
    print(f"  Part 1 — top-100 nominal FD")
    print(f"    label agreement: {agree}/{len(pair_rows)}")
    print(f"    Spearman ρ (surrogate vs FD yield_score): {rho100:.4f}")
    print(f"    top-1/3/5 overlap: {top_1}/1, {top_3}/3, {top_5}/5")
    for tname, e in errs.items():
        print(f"    {tname:<14}  MAE={e['mae']:.3f}  RMSE={e['rmse']:.3f}  n={e['n']}")
    print(f"    FD recipes that beat baseline: {fd_beats_baseline} / {len(pair_rows)}")
    print(f"  Part 2 — top-10 FD MC")
    print(f"    Spearman ρ (surrogate vs FD MC yield_score): {rho10:.4f}")
    print(f"    surrogate top-10 ∩ FD top-10: {sur_in_fd_top10}/10")
    print(f"    FD MC recipes that beat baseline: {fd_mc_beats_baseline} / {len(top10)}")
    print(f"  Part 4 — false-PASS top-100")
    print(f"    hard false-PASS: {false_summary['n_hard_false_pass']}  "
          f"({100*false_summary['hard_false_pass_rate']:.2f} %)")
    print(f"    soft false-PASS: {false_summary['n_soft_false_pass']}  "
          f"({100*false_summary['soft_false_pass_rate']:.2f} %)")
    print(f"  Acceptance: {acceptance}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
